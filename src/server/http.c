#include "src/server/internal.h"
#include <ctype.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <strings.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

static int json_response(struct mg_connection *c,int status,sj_value *v){
    char *body=sj_dump(v);sj_free(v);if(!body){status=500;body=strdup("{\"error\":{\"code\":\"out_of_memory\"}}");}
    size_t n=body?strlen(body):0;
    mg_printf(c,"HTTP/1.1 %d %s\r\nContent-Type: application/json\r\nContent-Length: %zu\r\nCache-Control: no-store\r\nX-Content-Type-Options: nosniff\r\n%sConnection: close\r\n\r\n",status,mg_get_response_code_text(c,status),n,status==429?"Retry-After: 2\r\n":"");
    if (strcmp(mg_get_request_info(c)->request_method, "HEAD") && body)
        mg_write(c, body, n);
    free(body);
    return status;
}
static int error_response(struct mg_connection *c,const srv_error *e){
    sj_value *v=sj_object(),*err=sj_object();sj_add(err,"message",sj_string(e->message));sj_add(err,"type",sj_string(e->status>=500?"server_error":"invalid_request_error"));
    sj_add(err,"code",sj_string(e->code));sj_add(err,"param",sj_string(*e->field?e->field:NULL));sj_add(v,"error",err);return json_response(c,e->status?e->status:500,v);
}
static int problem(struct mg_connection *c,int status,const char *code,const char *message){srv_error e={0};srv_fail(&e,status,code,NULL,"%s",message);return error_response(c,&e);}
static int valid_id(const char *id){if(!id||!*id||strlen(id)>=SRV_ID)return 0;for(;*id;id++)if(!isalnum((unsigned char)*id)&&*id!='_')return 0;return 1;}
static int authorized(srv *s,struct mg_connection *c){
    if (!s->config.api_key)
        return 1;
    const char *a = mg_get_header(c, "Authorization");
    if (!a || strncmp(a, "Bearer ", 7))
        return 0;
    a += 7;
    size_t n=strlen(s->config.api_key),m=strlen(a);unsigned difference=(unsigned)(n^m);
    for (size_t i = 0; i < n; i++)
        difference |= (unsigned char)s->config.api_key[i] ^ (i < m ? (unsigned char)a[i] : 0);
    return difference == 0;
}
static int query(struct mg_connection *c,const char *name,char *out,size_t size){const char *q=mg_get_request_info(c)->query_string;return q?mg_get_var(q,strlen(q),name,out,size):-1;}
static int query_int(struct mg_connection *c,const char *name,int fallback,int min,int max,int *out){
    char buf[64];int n=query(c,name,buf,sizeof(buf));if(n==-1){*out=fallback;return 1;}if(n<=0)return 0;
    for (char *p = buf; *p; p++)
        if (!isdigit((unsigned char)*p))
            return 0;
    char *end;
    errno = 0;
    long v = strtol(buf, &end, 10);
    if (errno || *end || v < min || v > max)
        return 0;
    *out = (int)v;
    return 1;
}
static sj_value *capabilities(srv *s){
    sj_value *v=sj_object();sj_add(v,"schema_version",sj_int(1));sj_add(v,"request_extension",sj_string("h3cli: string"));
    sj_add(v,"sglang_commit",sj_string("7ee7bef79decaf78f7669994f74e6f75c7784c14"));sj_add(v,"build_sha256",sj_string(s->executable_sha));
#ifdef __APPLE__
    sj_add(v,"backend",sj_string("metal"));
    sj_add(v,"reference_video_min_decoded_frames",sj_int(48));
#else
    sj_add(v,"backend",sj_string("cuda"));
    sj_add(v,"reference_video_min_decoded_frames",sj_int(5));
#endif
    sj_add(v,"request",h3_submission_schema());sj_add(v,"options",h3_option_schema());sj_add(v,"quality",sj_parse("[\"lossless\",\"extra-high\",\"high\"]",strlen("[\"lossless\",\"extra-high\",\"high\"]"),NULL,0));
    sj_add(v,"precedence",sj_string("native defaults < SGLang quality < SGLang fields < native quality < explicit native flags"));
    sj_add(v,"queue_limit",sj_int(s->config.queue_limit));sj_add(v,"job_timeout_seconds",sj_int(s->config.timeout));
    sj_add(v,"max_upload_bytes",sj_uint(s->config.upload_limit));sj_add(v,"max_storage_bytes",sj_uint(s->config.storage_limit));sj_add(v,"url_inputs",sj_bool(s->config.allow_urls&&!s->config.offline));
    sj_add(v,"models_offline",sj_bool(s->config.offline));sj_add(v,"model_download_timeout_seconds",sj_int(s->config.model_timeout));
    sj_add(v,"single_worker",sj_bool(1));sj_add(v,"cold_start_per_variant",sj_bool(1));sj_add(v,"numerical_parity_claim",sj_bool(0));return v;
}

typedef struct {srv *s;sj_value *body,*asset;srv_error error;char key[128],name[256],*text,*temp;size_t used,total_text;uint64_t file_bytes;int fd,assets_only,fields,have_file,current_file;} form;
static int form_finish(form *f){
    if(!*f->key)return 1;
    if(f->fd>=0){int ok=fsync(f->fd)==0;if(close(f->fd))ok=0;f->fd=-1;if(!ok)return srv_fail(&f->error,500,"storage_error","file","cannot finish upload");}
    else if(!f->current_file){
        if (!f->text)
            f->text = strdup("");
        if (!f->text)
            return 0;
        sj_value *value=NULL;
        const char *strings[]={"h3cli","model","prompt","task","quality","size","reference_url","input_reference","negative_prompt","output_mode","generator_device","sampling_method",NULL};
        int literal=0;for(size_t i=0;strings[i];i++)if(!strcmp(f->key,strings[i]))literal=1;
        if(literal)value=sj_string(f->text);else value=sj_parse(f->text,f->used,f->error.message,sizeof(f->error.message));
        if(!value||!sj_add(f->body,f->key,value))return srv_fail(&f->error,400,"invalid_multipart",f->key,"duplicate or invalid multipart field");
    }
    free(f->text);f->text=NULL;f->used=0;f->key[0]=0;f->current_file=0;return 1;
}
static int field_found(const char *key,const char *filename,char *path,size_t pathlen,void *opaque){
    (void)path;(void)pathlen;form *f=opaque;if(!form_finish(f)||++f->fields>128||strlen(key)>=sizeof(f->key))goto bad;
    snprintf(f->key,sizeof(f->key),"%s",key);
    if(filename&&*filename){
        if (f->have_file || strcmp(key, f->assets_only ? "file" : "input_reference"))
            goto bad;
        f->have_file = 1;
        f->current_file = 1;
        snprintf(f->name,sizeof(f->name),"%s",filename);char id[SRV_ID];if(!srv_id(id,"upload_"))goto bad;
        char *base=srv_path(f->s->config.state,"incoming");f->temp=base?srv_path(base,id):NULL;free(base);
        f->fd=f->temp?open(f->temp,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600):-1;if(f->fd<0)goto bad;
    }else if(f->assets_only)goto bad;
    return MG_FORM_FIELD_STORAGE_GET;
bad:srv_fail(&f->error,400,"invalid_multipart",key,"invalid, repeated or excessive multipart field");return MG_FORM_FIELD_STORAGE_ABORT;
}
static int field_get(const char *key,const char *value,size_t count,void *opaque){
    (void)key;form *f=opaque;
    if(f->fd>=0){
        if((uint64_t)count>f->s->config.upload_limit-f->file_bytes){srv_fail(&f->error,413,"upload_limit","file","asset exceeds upload limit");return MG_FORM_FIELD_HANDLE_ABORT;}
        size_t at=0;while(at<count){ssize_t n=write(f->fd,value+at,count-at);if(n<0&&errno==EINTR)continue;if(n<=0){srv_fail(&f->error,500,"storage_error","file","upload write failed");return MG_FORM_FIELD_HANDLE_ABORT;}at+=(size_t)n;}f->file_bytes+=count;
    }else{
        if(count>SRV_JSON_LIMIT-f->total_text||memchr(value,0,count)){srv_fail(&f->error,413,"body_limit",f->key,"text fields too large or contain NUL");return MG_FORM_FIELD_HANDLE_ABORT;}
        char *next=realloc(f->text,f->used+count+1);if(!next)return MG_FORM_FIELD_HANDLE_ABORT;f->text=next;memcpy(next+f->used,value,count);f->used+=count;f->total_text+=count;next[f->used]=0;
    }return MG_FORM_FIELD_HANDLE_GET;
}
static int submit(struct mg_connection *c,srv *s,int video,int assets_only){
    const struct mg_request_info *ri=mg_get_request_info(c);const char *type=mg_get_header(c,"Content-Type");srv_error error={0};sj_value *body=NULL,*result=NULL,*asset=NULL;int ok=0;uint64_t reserved=0;
    if(ri->content_length<0)return problem(c,411,"length_required","Content-Length is required");
    uint64_t limit=type&&!strncasecmp(type,"multipart/form-data",19)?s->config.upload_limit+SRV_JSON_LIMIT:SRV_JSON_LIMIT;
    if((uint64_t)ri->content_length>limit)return problem(c,413,"body_limit","request exceeds body limit");
    pthread_mutex_lock(&s->mutex);uint64_t used=srv_tree_bytes(s->config.state);
    if(used>s->config.storage_limit||(uint64_t)ri->content_length>s->config.storage_limit-used||s->storage_reserved>s->config.storage_limit-used-(uint64_t)ri->content_length){pthread_mutex_unlock(&s->mutex);return problem(c,413,"storage_quota","managed storage quota exhausted");}
    reserved=(uint64_t)ri->content_length;s->storage_reserved+=reserved;pthread_mutex_unlock(&s->mutex);
    if(type&&!strncasecmp(type,"multipart/form-data",19)){
        form f={.s=s,.body=sj_object(),.fd=-1,.assets_only=assets_only};struct mg_form_data_handler handlers={.field_found=field_found,.field_get=field_get,.user_data=&f};
        int count=mg_handle_form_request(c,&handlers);ok=count>0&&!f.error.status&&form_finish(&f);
        if(ok&&f.temp){ok=srv_asset_upload(s,f.temp,f.name,&asset,&error);if(ok&&!assets_only)ok=sj_add(f.body,"input_reference",sj_string(sj_field(asset,"uri")));}
        if(ok&&assets_only&&!asset){srv_fail(&error,400,"missing_file","file","one file field is required");ok=0;}
        if (f.fd >= 0)
            close(f.fd);
        if (f.temp)
            unlink(f.temp);
        free(f.temp);
        free(f.text);
        body = f.body;
        if (!ok && f.error.status)
            error = f.error;
    }else if(!assets_only&&type&&!strncasecmp(type,"application/json",16)){
        size_t n=(size_t)ri->content_length;char *text=malloc(n+1);size_t got=0;
        if(text){while(got<n){int r=mg_read(c,text+got,n-got);if(r<=0)break;got+=(size_t)r;}if(got==n)body=sj_parse(text,n,error.message,sizeof(error.message));free(text);}ok=body!=NULL;
    }else srv_fail(&error,415,"unsupported_media_type","Content-Type","use application/json or multipart/form-data");
    if(ok&&assets_only){result=asset;asset=NULL;}
    else if(ok)ok=srv_submit(s,body,video,mg_get_header(c,"Idempotency-Key"),&result,&error);
    if(!ok&&asset){srv_error ignored={0};srv_asset_delete(s,sj_field(asset,"id"),&ignored);}
    sj_free(asset);sj_free(body);pthread_mutex_lock(&s->mutex);s->storage_reserved-=reserved;pthread_mutex_unlock(&s->mutex);
    if(!ok){sj_free(result);if(!error.status){char detail[SRV_ERROR];snprintf(detail,sizeof(detail),"%s",*error.message?error.message:"invalid or incomplete request body");srv_fail(&error,400,"invalid_request","body","%s",detail);}return error_response(c,&error);}
    return json_response(c,assets_only?201:video?200:202,result);
}
static void release_download(srv *s,const char *owner){pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;
    if(sqlite3_prepare_v2(s->db,"DELETE FROM leases WHERE owner=? AND kind='download'",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,owner,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);}
static int download(struct mg_connection *c,srv *s,const char *job,const char *artifact,int video){
    int variant;if(!query_int(c,"variant",0,0,9,&variant))return problem(c,400,"invalid_variant","variant must be 0..9");
    char lease[SRV_ID],*path=NULL,*mime=NULL,*hash=NULL;int fd=-1;uint64_t bytes=0;sqlite3_stmt *q=NULL;
    if (!srv_id(lease, "dl_"))
        return problem(c, 500, "storage_error", "cannot create download lease");
    pthread_mutex_lock(&s->mutex);
    const char *sql=video?
      "SELECT a.path,t.mime,a.sha,a.id FROM artifacts t JOIN assets a ON a.id=t.id JOIN jobs j ON j.id=t.job JOIN variants v ON v.job=t.job AND v.idx=t.variant WHERE t.job=? AND j.deleted=0 AND j.video=1 AND v.status='completed' AND t.variant=? AND t.mime='video/mp4' ORDER BY t.name LIMIT 1":
      "SELECT a.path,t.mime,a.sha,a.id FROM artifacts t JOIN assets a ON a.id=t.id JOIN jobs j ON j.id=t.job JOIN variants v ON v.job=t.job AND v.idx=t.variant WHERE t.job=? AND j.deleted=0 AND (v.status='completed' OR t.name='worker.log') AND t.id=?";
    if(sqlite3_prepare_v2(s->db,sql,-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,job,-1,SQLITE_TRANSIENT);if(video)sqlite3_bind_int(q,2,variant);else sqlite3_bind_text(q,2,artifact,-1,SQLITE_TRANSIENT);
        if(sqlite3_step(q)==SQLITE_ROW){path=strdup((const char *)sqlite3_column_text(q,0));mime=strdup((const char *)sqlite3_column_text(q,1));hash=strdup((const char *)sqlite3_column_text(q,2));char *id=strdup((const char *)sqlite3_column_text(q,3));sqlite3_finalize(q);q=NULL;
            if(id&&sqlite3_prepare_v2(s->db,"INSERT INTO leases(owner,asset,kind) VALUES(?,?,'download')",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,lease,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,id,-1,SQLITE_TRANSIENT);if(sqlite3_step(q)==SQLITE_DONE&&path)fd=open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);}free(id);}}
    sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);free(path);
    struct stat st;if(fd<0||fstat(fd,&st)||!S_ISREG(st.st_mode)){if(fd>=0)close(fd);free(mime);free(hash);release_download(s,lease);return problem(c,404,"not_found","published artifact not found");}bytes=(uint64_t)st.st_size;
    uint64_t first=0,last=bytes?bytes-1:0;int status=200;const char *range=mg_get_header(c,"Range");
    if(range){char *end;errno=0;if(strncmp(range,"bytes=",6)||strchr(range,','))status=416;
        else if(range[6]=='-'){uint64_t suffix=strtoull(range+7,&end,10);if(errno||*end||!suffix||!bytes)status=416;else first=suffix>=bytes?0:bytes-suffix;}
        else {first=strtoull(range+6,&end,10);if(errno||end==range+6||*end!='-'||first>=bytes)status=416;else if(end[1]){char *tail;last=strtoull(end+1,&tail,10);if(errno||*tail||last<first)status=416;else if(last>=bytes)last=bytes-1;}}
        if(status!=416)status=206;
    }
    if(status==416){mg_printf(c,"HTTP/1.1 416 Range Not Satisfiable\r\nContent-Range: bytes */%"PRIu64"\r\nContent-Length: 0\r\nConnection: close\r\n\r\n",bytes);}
    else {uint64_t length=bytes?last-first+1:0;mg_printf(c,"HTTP/1.1 %d %s\r\nContent-Type: %s\r\nContent-Length: %"PRIu64"\r\nAccept-Ranges: bytes\r\nETag: \"%s\"\r\nX-Content-Type-Options: nosniff\r\n",status,mg_get_response_code_text(c,status),mime?mime:"application/octet-stream",length,hash?hash:"");
        if (status == 206)
            mg_printf(c, "Content-Range: bytes %" PRIu64 "-%" PRIu64 "/%" PRIu64 "\r\n", first, last, bytes);
        mg_printf(c, "Connection: close\r\n\r\n");
        if(strcmp(mg_get_request_info(c)->request_method,"HEAD")&&lseek(fd,(off_t)first,SEEK_SET)>=0){char buf[65536];while(length){size_t want=length<sizeof(buf)?(size_t)length:sizeof(buf);ssize_t n=read(fd,buf,want);if(n<0&&errno==EINTR)continue;if(n<=0||mg_write(c,buf,(size_t)n)!=n)break;length-=(uint64_t)n;}}
    }close(fd);free(mime);free(hash);release_download(s,lease);return status;
}
static int events(struct mg_connection *c,srv *s,const char *job){
    sj_value *v=srv_job(s,job,0,0);if(!v)return problem(c,404,"not_found","job not found");sj_free(v);
    int64_t after=0;const char *last=mg_get_header(c,"Last-Event-ID");if(last){char *end;errno=0;after=strtoll(last,&end,10);if(errno||*end||after<0)return problem(c,400,"invalid_event_id","invalid Last-Event-ID");}
    if(atomic_fetch_add(&s->sse_clients,1)>=4){atomic_fetch_sub(&s->sse_clients,1);return problem(c,429,"stream_limit","at most four concurrent event streams");}
    mg_printf(c,"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\nCache-Control: no-store\r\nConnection: close\r\n\r\nretry: 1000\n\n");
    double deadline=srv_now()+10;int connected=1;
    while(connected&&!atomic_load(&s->stopping)&&srv_now()<deadline){
        /* Read status before draining events. Completion and its event are
         * published under the same queue mutex; reading status afterwards
         * could observe a new completion and close before sending its event. */
        v=srv_job(s,job,0,0);
        const char *state=sj_field(v,"status");
        int terminal=!state||(strcmp(state,"queued")&&strcmp(state,"running"));
        sj_free(v);
        sj_value *batch=srv_events(s,job,after);
        for(size_t i=0;batch&&i<batch->count;i++){sj_value *event=batch->items[i];sj_i64(sj_get(event,"event_id"),&after);char *body=sj_dump(event);
            if (!body || mg_printf(c, "id: %" PRId64 "\ndata: %s\n\n", after, body) <= 0)
                connected = 0;
            free(body);
            if (!connected)
                break;
        }
        sj_free(batch);
        if(terminal)break;
        struct timespec delay={0,100000000};nanosleep(&delay,NULL);
    }atomic_fetch_sub(&s->sse_clients,1);return 200;
}
static int preview(struct mg_connection *c,srv *s,const char *job){
    int variant;if(!query_int(c,"variant",0,0,9,&variant))return problem(c,400,"invalid_variant","invalid preview variant");sj_value *v=srv_job(s,job,0,0);if(!v)return problem(c,404,"not_found","job not found");sj_free(v);
    char relative[160];snprintf(relative,sizeof(relative),"jobs/%s/v%d/work/preview.ppm",job,variant);char *path=srv_path(s->config.state,relative);int fd=path?open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC):-1;free(path);struct stat st;
    if(fd<0||fstat(fd,&st)||!S_ISREG(st.st_mode)){if(fd>=0)close(fd);return problem(c,404,"not_found","preview not available");}
    mg_printf(c,"HTTP/1.1 200 OK\r\nContent-Type: image/x-portable-pixmap\r\nContent-Length: %lld\r\nCache-Control: no-store\r\nConnection: close\r\n\r\n",(long long)st.st_size);
    if(strcmp(mg_get_request_info(c)->request_method,"HEAD")){char buf[65536];ssize_t n;while((n=read(fd,buf,sizeof(buf)))>0)if(mg_write(c,buf,(size_t)n)!=n)break;}close(fd);return 200;
}
static int handler(struct mg_connection *c,void *opaque){
    srv *s=opaque;const struct mg_request_info *ri=mg_get_request_info(c);const char *method=ri->request_method,*url=ri->local_uri;int get=!strcmp(method,"GET"),head=!strcmp(method,"HEAD"),post=!strcmp(method,"POST"),del=!strcmp(method,"DELETE");
    mg_disable_connection_keep_alive(c);
    if(!strcmp(url,"/health")||!strcmp(url,"/liveness")||!strcmp(url,"/readiness")){
        if (!get && !head)
            return problem(c, 405, "method_not_allowed", "use GET");
        sj_value *v = sj_object();
        int ready = !atomic_load(&s->stopping) && (!strcmp(url, "/liveness") || !atomic_load(&s->storage_failed));
        sj_add(v, "status", sj_string(ready ? "ok" : "stopping"));
        return json_response(c, ready ? 200 : 503, v);
    }
    if(!authorized(s,c))return problem(c,401,"unauthorized","valid Bearer authorization required");
    if(!strcmp(url,"/v1/models")){if(!get&&!head)return problem(c,405,"method_not_allowed","use GET");sj_value *v=sj_object(),*a=sj_array(),*m=sj_object();sj_add(v,"object",sj_string("list"));sj_add(m,"id",sj_string(s->config.model_id));sj_add(m,"object",sj_string("model"));sj_add(m,"owned_by",sj_string("h3cli"));sj_add(a,NULL,m);sj_add(v,"data",a);return json_response(c,200,v);}
    if(!strcmp(url,"/v1/h3/capabilities")||!strcmp(url,"/v1/h3/schema")){if(!get&&!head)return problem(c,405,"method_not_allowed","use GET");return json_response(c,200,capabilities(s));}
    if(!strcmp(url,"/v1/h3/assets")){if(!post)return problem(c,405,"method_not_allowed","use POST");return submit(c,s,0,1);}
    if(!strncmp(url,"/v1/h3/assets/",14)){const char *id=url+14;if(!valid_id(id))return problem(c,404,"not_found","asset not found");
        if(get||head){sj_value *v=srv_asset_get(s,id);return v?json_response(c,200,v):problem(c,404,"not_found","asset not found");}
        if(del){srv_error e={0};if(!srv_asset_delete(s,id,&e))return error_response(c,&e);sj_value *v=sj_object();sj_add(v,"id",sj_string(id));sj_add(v,"deleted",sj_bool(1));return json_response(c,200,v);}return problem(c,405,"method_not_allowed","use GET or DELETE");}
    int video=!strncmp(url,"/v1/videos",10)&&(url[10]==0||url[10]=='/');int native=!strncmp(url,"/v1/h3/jobs",11)&&(url[11]==0||url[11]=='/');
    if (!video && !native)
        return problem(c, 404, "not_found", "route not found");
    const char *tail = url + (video ? 10 : 11);
    if(!*tail){
        if (post)
            return submit(c, s, video, 0);
        if (get || head) {
            int limit;
            char after[SRV_ID], order[16];
            if (!query_int(c, "limit", 20, 1, 100, &limit))
                return problem(c, 400, "invalid_limit", "limit must be 1..100");
            int a=query(c,"after",after,sizeof(after)),o=query(c,"order",order,sizeof(order));if(a<-1||(a>=0&&!valid_id(after))||o<-1||(o>=0&&strcmp(order,"asc")&&strcmp(order,"desc")))return problem(c,400,"invalid_query","invalid cursor or order");return json_response(c,200,srv_jobs(s,video,a>=0?after:NULL,limit,o>=0&&!strcmp(order,"asc")));
        }
        return problem(c, 405, "method_not_allowed", "use GET or POST");
    }
    char id[SRV_ID];tail++;const char *slash=strchr(tail,'/');size_t n=slash?(size_t)(slash-tail):strlen(tail);if(n>=sizeof(id))return problem(c,404,"not_found","job not found");memcpy(id,tail,n);id[n]=0;if(!valid_id(id))return problem(c,404,"not_found","job not found");tail=slash?slash:"";
    if(!*tail){if(get||head){sj_value *v=srv_job(s,id,video,0);return v?json_response(c,200,v):problem(c,404,"not_found","job not found");}
        if(del){sj_value *existing=srv_job(s,id,video,0);if(!existing)return problem(c,404,"not_found","job not found");sj_free(existing);srv_error e={0};if(!srv_cancel(s,id,1,&e))return error_response(c,&e);sj_value *v=sj_object();sj_add(v,"id",sj_string(id));sj_add(v,"object",sj_string(video?"video.deleted":"h3.job.deleted"));sj_add(v,"deleted",sj_bool(1));return json_response(c,200,v);}return problem(c,405,"method_not_allowed","use GET or DELETE");}
    if(video&&!strcmp(tail,"/content")){if(!get&&!head)return problem(c,405,"method_not_allowed","use GET or HEAD");return download(c,s,id,NULL,1);}
    if(native&&!strcmp(tail,"/cancel")){if(!post)return problem(c,405,"method_not_allowed","use POST");srv_error e={0};if(!srv_cancel(s,id,0,&e))return error_response(c,&e);return json_response(c,200,srv_job(s,id,0,0));}
    if(native&&!strcmp(tail,"/events")){if(!get)return problem(c,405,"method_not_allowed","use GET");return events(c,s,id);}
    if(native&&!strcmp(tail,"/preview")){if(!get&&!head)return problem(c,405,"method_not_allowed","use GET or HEAD");return preview(c,s,id);}
    if(native&&!strncmp(tail,"/artifacts/",11)&&valid_id(tail+11)){if(!get&&!head)return problem(c,405,"method_not_allowed","use GET or HEAD");return download(c,s,id,tail+11,0);}
    return problem(c,404,"not_found","route not found");
}
static int log_message(const struct mg_connection *c,const char *message){(void)c;fprintf(stderr,"h3cli: HTTP: %s\n",message);return 1;}
int srv_http_start(srv *s,char *error,size_t size){
    char listen[256];if(strchr(s->config.host,':'))snprintf(listen,sizeof(listen),"[%s]:%d",s->config.host,s->config.port);else snprintf(listen,sizeof(listen),"%s:%d",s->config.host,s->config.port);
    const char *options[]={"listening_ports",listen,"num_threads","16","request_timeout_ms","10000","keep_alive_timeout_ms","1000","enable_keep_alive","no","max_request_size","32768",NULL};
    struct mg_callbacks callbacks={.log_message=log_message};s->http=mg_start(&callbacks,s,options);if(!s->http){snprintf(error,size,"cannot bind HTTP listener or initialize CivetWeb");return 0;}mg_set_request_handler(s->http,"/",handler,s);return 1;
}
void srv_http_stop(srv *s){if(s->http){mg_stop(s->http);s->http=NULL;}}
