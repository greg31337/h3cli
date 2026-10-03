#include "src/server/internal.h"
#include "src/weights/safetensors.h"
#include "src/platform.h"
#include <arpa/inet.h>
#include <curl/curl.h>
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <netinet/in.h>
#include <stdlib.h>
#include <string.h>
#include <sys/socket.h>
#include <sys/stat.h>
#include <unistd.h>
static int inside(const char *root,const char *path){size_t n=strlen(root);return !strncmp(root,path,n)&&(!path[n]||path[n]=='/'||!strcmp(root,"/"));}
static int safe_name(const char *name){
    if(!name||!*name||*name=='/'||strlen(name)>512)return 0;
    const char *p=name;while(*p){const char *end=strchr(p,'/');size_t n=end?(size_t)(end-p):strlen(p);
        if(!n||(n==1&&*p=='.')||(n==2&&!memcmp(p,"..",2)))return 0;
        for (size_t i = 0; i < n; i++)
            if ((unsigned char)p[i] < 32 || p[i] == '\\')
                return 0;
        p = end ? end + 1 : p + n;
    }
    return name[strlen(name)-1]!='/';
}
/* Walk canonical absolute paths with O_NOFOLLOW at every component. */
static int checked_open(const char *path){
    if (!path || *path != '/')
        return -1;
    char *copy = strdup(path + 1);
    if (!copy)
        return -1;
    int current=open("/",O_RDONLY|O_DIRECTORY|O_CLOEXEC);char *save=NULL,*part=strtok_r(copy,"/",&save);
    while(part&&current>=0){char *next=strtok_r(NULL,"/",&save);int fd=openat(current,part,O_RDONLY|O_NOFOLLOW|O_CLOEXEC|(next?O_DIRECTORY:0));close(current);current=fd;part=next;}
    free(copy);return current;
}
static int copy_fd(int input,const char *dest,uint64_t limit,char *error,size_t size){
    struct stat st;if(fstat(input,&st)||!S_ISREG(st.st_mode)||st.st_size<0||(uint64_t)st.st_size>limit){snprintf(error,size,"input is not a bounded regular file");return 0;}
    int output=open(dest,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC,0600);if(output<0){snprintf(error,size,"cannot create input snapshot: %s",strerror(errno));return 0;}
    char buf[65536];uint64_t total=0;int ok=1;
    while(ok){ssize_t got=read(input,buf,sizeof(buf));if(got<0){if(errno==EINTR)continue;ok=0;break;}if(!got)break;
        if((uint64_t)got>limit-total){ok=0;break;}total+=(uint64_t)got;size_t at=0;
        while(at<(size_t)got){ssize_t n=write(output,buf+at,(size_t)got-at);if(n<0&&errno==EINTR)continue;if(n<=0){ok=0;break;}at+=(size_t)n;}
    }
    struct stat after;if(fstat(input,&after)||after.st_size!=st.st_size||total!=(uint64_t)st.st_size)ok=0;
    else {struct timespec before_time=h3_stat_mtime(&st),after_time=h3_stat_mtime(&after),before_change=h3_stat_ctime(&st),after_change=h3_stat_ctime(&after);
        if(before_time.tv_sec!=after_time.tv_sec||before_time.tv_nsec!=after_time.tv_nsec||before_change.tv_sec!=after_change.tv_sec||before_change.tv_nsec!=after_change.tv_nsec)ok=0;}
    if (fsync(output))
        ok = 0;
    if (close(output))
        ok = 0;
    if (!ok) {
        unlink(dest);
        snprintf(error, size, "input copy failed, changed or exceeded limit");
    }
    return ok;
}
static int copy_path(const char *input,const char *dest,uint64_t limit,char *error,size_t size){
    int fd=checked_open(input);if(fd<0){snprintf(error,size,"input does not exist or traverses a symlink");return 0;}
    int ok=copy_fd(fd,dest,limit,error,size);close(fd);return ok;
}
static int allowed(srv *s,const char *path,int model){
    for(size_t i=0;i<s->config.read_root_count;i++)if(inside(s->config.read_roots[i],path))return 1;
    if(model)return inside(s->config.model,path)||inside(s->config.models_root,path);
    return 0;
}
static int db_asset_locked(srv *s,const char *id,char **path,char **owner){
    sqlite3_stmt *q=NULL;int ok=0;
    if(sqlite3_prepare_v2(s->db,"SELECT a.path,a.owner FROM assets a LEFT JOIN jobs j ON a.owner=j.id WHERE a.id=? AND (a.owner IS NULL OR j.deleted=0)",-1,&q,NULL)!=SQLITE_OK)return 0;
    sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);
    if(sqlite3_step(q)==SQLITE_ROW){*path=strdup((const char *)sqlite3_column_text(q,0));const char *p=(const char *)sqlite3_column_text(q,1);if(owner)*owner=p?strdup(p):NULL;ok=*path!=NULL;}
    sqlite3_finalize(q);return ok;
}
static int lease_locked(srv *s,const char *job,const char *asset){
    sqlite3_stmt *q=NULL;int ok=sqlite3_prepare_v2(s->db,"INSERT OR IGNORE INTO leases(owner,asset,kind) VALUES(?,?,'input')",-1,&q,NULL)==SQLITE_OK;
    if(ok){sqlite3_bind_text(q,1,job,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,asset,-1,SQLITE_TRANSIENT);ok=sqlite3_step(q)==SQLITE_DONE;}sqlite3_finalize(q);return ok;
}
static int resolve(srv *s,const char *uri,const char *job,int model,char **path,char *error,size_t size){
    *path=NULL;const char *id=NULL;char asset[SRV_ID],job_id[SRV_ID];int artifact=0;
    if(!strncmp(uri,"asset://",8))id=uri+8;
    else if(!strncmp(uri,"artifact://",11)){
        const char *slash=strchr(uri+11,'/');if(!slash||(size_t)(slash-uri-11)>=sizeof(job_id)||strlen(slash+1)>=sizeof(asset))goto denied;
        memcpy(job_id,uri+11,(size_t)(slash-uri-11));job_id[slash-uri-11]=0;strcpy(asset,slash+1);id=asset;artifact=1;
    }
    if(id){
        if(strlen(id)>=SRV_ID||!safe_name(id)||strchr(id,'/'))goto denied;
        pthread_mutex_lock(&s->mutex);char *owner=NULL;int ok=db_asset_locked(s,id,path,&owner);
        if(ok&&artifact&&(!owner||strcmp(owner,job_id)))ok=0;
        if (ok)
            ok = lease_locked(s, job, id);
        free(owner);
        pthread_mutex_unlock(&s->mutex);
        if (ok)
            return 1;
        free(*path);
        *path = NULL;
        goto denied;
    }
    if(strstr(uri,"://"))goto denied;
    char *absolute=uri[0]=='/'?strdup(uri):srv_path(s->config.source_root,uri);char *real=absolute?(model?h3_models_absolute(absolute,error,size):realpath(absolute,NULL)):NULL;free(absolute);
    if(!real||!allowed(s,real,model)){free(real);goto denied;}
    struct stat missing;
    if(model&&lstat(real,&missing)&&errno==ENOENT&&
       (inside(s->config.model,real)||inside(s->config.models_root,real))){*path=real;return 1;}
    int fd=checked_open(real);struct stat st;int ok=fd>=0&&!fstat(fd,&st)&&(S_ISREG(st.st_mode)||(model&&S_ISDIR(st.st_mode)));if(fd>=0)close(fd);
    if(!ok){free(real);goto denied;}*path=real;return 1;
denied:snprintf(error,size,"input is unavailable or outside configured read roots; upload it as an asset");return 0;
}
typedef struct {int fd;uint64_t bytes,limit;int failed;} url_sink;
static size_t url_write(char *ptr,size_t unit,size_t count,void *opaque){
    url_sink *sink=opaque;if(unit&&count>SIZE_MAX/unit)return 0;size_t n=unit*count;if((uint64_t)n>sink->limit-sink->bytes){sink->failed=1;return 0;}
    size_t at=0;while(at<n){ssize_t wrote=write(sink->fd,ptr+at,n-at);if(wrote<0&&errno==EINTR)continue;if(wrote<=0){sink->failed=1;return 0;}at+=(size_t)wrote;}sink->bytes+=(uint64_t)n;return n;
}
static int public_address(const struct sockaddr *address){
    if(address->sa_family==AF_INET){uint32_t a=ntohl(((const struct sockaddr_in *)address)->sin_addr.s_addr);
        if((a>>24)==0||(a>>24)==10||(a>>24)==127||(a>>24)>=224||(a>>16)==0xa9fe||(a>>20)==0xac1||(a>>16)==0xc0a8||(a>>22)==0x191)return 0;
        if ((a >> 24) == 192 || ((a >> 15) == 0x18c6) || (a >> 24) == 198 || (a >> 24) == 203)
            return 0;
        return 1;
    }
    if(address->sa_family==AF_INET6){const struct in6_addr *a=&((const struct sockaddr_in6 *)address)->sin6_addr;
        if(IN6_IS_ADDR_V4MAPPED(a)){struct sockaddr_in mapped={.sin_family=AF_INET};memcpy(&mapped.sin_addr,a->s6_addr+12,4);return public_address((const struct sockaddr *)&mapped);}
        // Global-unicast only; exclude documentation/transition ranges.
        if((a->s6_addr[0]&0xe0)!=0x20)return 0;
        if (a->s6_addr[0] == 0x20 &&
            (a->s6_addr[1] == 2 || (a->s6_addr[1] == 1 && (a->s6_addr[2] == 0 || a->s6_addr[2] == 0x0d))))
            return 0;
        return 1;
    }return 0;
}
static curl_socket_t url_socket(void *opaque,curlsocktype purpose,struct curl_sockaddr *address){
    (void)opaque;(void)purpose;if(!public_address(&address->addr))return CURL_SOCKET_BAD;
    int fd=socket(address->family,address->socktype,address->protocol);if(fd>=0)fcntl(fd,F_SETFD,FD_CLOEXEC);return fd;
}
static int import_url(srv *s,const char *url,const char *dest,char *error,size_t size){
    if(s->config.offline||!s->config.allow_urls){snprintf(error,size,"URL inputs are disabled; upload media instead");return 0;}
    CURL *curl=curl_easy_init();if(!curl)return 0;url_sink sink={.fd=open(dest,O_WRONLY|O_CREAT|O_EXCL|O_CLOEXEC,0600),.limit=s->config.upload_limit};
    if(sink.fd<0){curl_easy_cleanup(curl);return 0;}
    curl_easy_setopt(curl,CURLOPT_URL,url);
#if LIBCURL_VERSION_NUM >= 0x075500
    curl_easy_setopt(curl,CURLOPT_PROTOCOLS_STR,"http,https");
    curl_easy_setopt(curl,CURLOPT_REDIR_PROTOCOLS_STR,"http,https");
#else
    curl_easy_setopt(curl,CURLOPT_PROTOCOLS,(long)(CURLPROTO_HTTP|CURLPROTO_HTTPS));
    curl_easy_setopt(curl,CURLOPT_REDIR_PROTOCOLS,(long)(CURLPROTO_HTTP|CURLPROTO_HTTPS));
#endif
    curl_easy_setopt(curl,CURLOPT_FOLLOWLOCATION,1L);curl_easy_setopt(curl,CURLOPT_MAXREDIRS,3L);
    curl_easy_setopt(curl,CURLOPT_TIMEOUT,30L);curl_easy_setopt(curl,CURLOPT_CONNECTTIMEOUT,10L);curl_easy_setopt(curl,CURLOPT_NOSIGNAL,1L);curl_easy_setopt(curl,CURLOPT_PROXY,"");
    curl_easy_setopt(curl,CURLOPT_OPENSOCKETFUNCTION,url_socket);curl_easy_setopt(curl,CURLOPT_WRITEFUNCTION,url_write);curl_easy_setopt(curl,CURLOPT_WRITEDATA,&sink);
    curl_easy_setopt(curl,CURLOPT_MAXFILESIZE_LARGE,(curl_off_t)sink.limit);curl_easy_setopt(curl,CURLOPT_FAILONERROR,1L);
    const char *ca=getenv("CURL_CA_BUNDLE");if(!ca||!*ca)ca=getenv("SSL_CERT_FILE");
    if(ca&&*ca)curl_easy_setopt(curl,CURLOPT_CAINFO,ca);
    CURLcode result=curl_easy_perform(curl);int ok=result==CURLE_OK&&!sink.failed&&sink.bytes>0&&!fsync(sink.fd);close(sink.fd);curl_easy_cleanup(curl);
    if(!ok){
        unlink(dest);
        if(result==CURLE_SSL_CACERT_BADFILE||result==CURLE_PEER_FAILED_VERIFICATION)
            snprintf(error,size,"HTTPS certificate verification failed; check the system CA store or CURL_CA_BUNDLE/SSL_CERT_FILE");
        else snprintf(error,size,"URL import rejected, failed or exceeded its limits");
    }return ok;
}
static int copy_sidecars(srv *s,const char *from,const char *to,char *error,size_t size){
    const char *suffixes[]={".presentation",".lora.json"};
    for(size_t i=0;i<2;i++){char *src=NULL,*dst=NULL;int made=asprintf(&src,"%s%s",from,suffixes[i])>=0&&asprintf(&dst,"%s%s",to,suffixes[i])>=0;
        struct stat st;int ok=made;if(made&&!lstat(src,&st))ok=copy_path(src,dst,s->config.upload_limit,error,size);free(src);free(dst);if(!ok)return 0;}
    return 1;
}
static int import_input(srv *s,const char *job,const char *uri,const char *dest,int model,int state,char **resolved,char *error,size_t size){
    *resolved=NULL;
    if(!strncmp(uri,"http://",7)||!strncmp(uri,"https://",8)){
        if(model){snprintf(error,size,"model URL downloads are not supported");return 0;}
        if (!import_url(s, uri, dest, error, size))
            return 0;
        *resolved = strdup(dest);
        return *resolved != NULL;
    }
    char *path;if(!resolve(s,uri,job,model,&path,error,size))return 0;
    if(model){*resolved=path;return 1;}
    int unpack=state?srv_bundle_unpack(path,dest,s->config.upload_limit,error,size):2;
    int ok=unpack==1||(unpack==2&&copy_path(path,dest,s->config.upload_limit,error,size)&&copy_sidecars(s,path,dest,error,size));free(path);
    if (ok)
        *resolved = strdup(dest);
    return ok && *resolved;
}
int srv_assets_prepare(srv *s,h3_submission *submission,const char *job,char *error,size_t size){
    char *jobs=srv_path(s->config.state,"jobs"),*root=jobs?srv_path(jobs,job):NULL,*inputs=root?srv_path(root,"inputs"):NULL;free(jobs);free(root);
    if(!inputs||!srv_mkdirs(inputs,error,size)){free(inputs);return 0;}
    h3_request *r=&submission->request;
    if(s->config.offline&&!h3_request_get(r,"offline")&&!h3_request_add(r,"offline",NULL,NULL,H3_SOURCE_MANAGED,error,size)){free(inputs);return 0;}
    int offline=s->config.offline||h3_request_get(r,"offline")!=NULL;
    for(size_t i=0;offline&&i<r->count;i++){
        const h3_option_value *v=&r->values[i];
        if(!v->option->repeatable&&h3_request_get(r,v->option->name)!=v)continue;
        if((v->option->path==H3_PATH_MEDIA||v->option->path==H3_PATH_STATE)&&
           ((v->value&&(!strncmp(v->value,"http://",7)||!strncmp(v->value,"https://",8)))||
            (v->second&&(!strncmp(v->second,"http://",7)||!strncmp(v->second,"https://",8))))){
            free(inputs);snprintf(error,size,"offline: URL inputs are disabled");return 0;
        }
    }
    if(!h3_request_get(r,"model-dir")){
        if(strcmp(submission->model,s->config.model_id)&&strcmp(submission->model,"h3cli")&&strcmp(submission->model,"MiniMaxAI/MiniMax-H3")){free(inputs);snprintf(error,size,"unknown served model ID");return 0;}
        if(!h3_request_add(r,"model-dir",s->config.model,NULL,H3_SOURCE_MANAGED,error,size)){free(inputs);return 0;}
    }
    for(size_t i=0;i<r->count;i++){
        h3_option_value *v=&r->values[i];h3_path_role role=v->option->path;
        if(role!=H3_PATH_MEDIA&&role!=H3_PATH_STATE&&role!=H3_PATH_MODEL)continue;
        // Ignore superseded scalar paths before they can trigger an import.
        if(!v->option->repeatable&&h3_request_get(r,v->option->name)!=v)continue;
        char *original=v->value,*scale=NULL,*lora_path=NULL;
        if(!strcmp(v->option->name,"lora")){
            lora_path=strdup(original);if(!lora_path){free(inputs);return 0;}char *colon=strrchr(lora_path,':');double number;
            char *literal=original[0]=='/'?strdup(original):srv_path(s->config.source_root,original);struct stat literal_stat;
            int literal_exists=literal&&stat(literal,&literal_stat)==0;free(literal);
            if(!literal_exists&&colon&&lj_number(colon+1,&number)){scale=strdup(colon);*colon=0;}original=lora_path;
        }
        char leaf[64];snprintf(leaf,sizeof(leaf),"input-%03zu",i);char *dest=srv_path(inputs,leaf),*path=NULL;
        int ok=dest&&import_input(s,job,original,dest,role==H3_PATH_MODEL,role==H3_PATH_STATE,&path,error,size);free(dest);free(lora_path);
        if(!ok){free(scale);free(inputs);return 0;}
        if(scale){char *joined=NULL;if(asprintf(&joined,"%s%s",path,scale)<0){free(scale);free(path);free(inputs);return 0;}free(scale);free(path);path=joined;}
        free(v->value);v->value=path;
        if(v->second){snprintf(leaf,sizeof(leaf),"audio-%03zu",i);dest=srv_path(inputs,leaf);path=NULL;
            ok=dest&&import_input(s,job,v->second,dest,0,0,&path,error,size);free(dest);if(!ok){free(inputs);return 0;}free(v->second);v->second=path;}
    }
    int input_dir=open(inputs,O_RDONLY|O_CLOEXEC);int synced=input_dir>=0&&fsync(input_dir)==0;if(input_dir>=0)close(input_dir);free(inputs);if(!synced){snprintf(error,size,"cannot sync imported inputs");return 0;}
    int iw=0,ih=0;double audio=0;
    if(submission->width_needed||submission->height_needed){
        const h3_option_value *first=h3_request_get(r,"first-frame");if(!first)first=h3_request_get(r,"last-frame");
        if(first){sj_value *probe=NULL;if(!srv_probe(first->value,&probe,error,size))return 0;
            const sj_value *streams=sj_get(probe,"streams");if(streams)for(size_t i=0;i<streams->count;i++){
                const sj_value *stream=streams->items[i];const char *type=sj_field(stream,"codec_type");if(type&&!strcmp(type,"video")){
                    int64_t w,h;if(sj_i64(sj_get(stream,"width"),&w)&&sj_i64(sj_get(stream,"height"),&h)&&w<=INT_MAX&&h<=INT_MAX){iw=(int)w;ih=(int)h;break;}}}
            sj_free(probe);}
    }
    const sj_value *duration=sj_get(sj_get(submission->transport,"target"),"duration_seconds");
    int derive=submission->duration_needed&&(!duration||duration->type==LJ_NULL),sources=0;
    if(derive||submission->needs_embedded_audio){
        for(size_t i=0;i<r->count;i++){h3_option_value *v=&r->values[i];int soundtrack=!strcmp(v->option->name,"ref-video"),standalone=!strcmp(v->option->name,"ref-audio");
            if (!soundtrack && !standalone && !v->second)
                continue;
            if (derive)
                sources++;
            sj_value *probe = NULL;
            if (!srv_probe(v->second ? v->second : v->value, &probe, error, size))
                return 0;
            int found=0;const sj_value *streams=sj_get(probe,"streams");if(streams)for(size_t j=0;j<streams->count;j++){
                const sj_value *stream=streams->items[j];const char *type=sj_field(stream,"codec_type");if(type&&!strcmp(type,"audio")){found=1;const char *d=sj_field(stream,"duration");if(d)lj_number(d,&audio);}}
            if(found&&audio<=0){const char *d=sj_field(sj_get(probe,"format"),"duration");if(d)lj_number(d,&audio);}sj_free(probe);
            if(soundtrack&&v->source==H3_SOURCE_SGLANG&&submission->required_audio[i]&&!found){snprintf(error,size,"video_audio condition requires an embedded audio stream");return 0;}
        }
    }
    if(derive&&(strcmp(submission->task,"ref2va")||sources!=1)){snprintf(error,size,"target.duration_seconds requires exactly one effective audio-bearing reference");return 0;}
    if(!h3_submission_geometry(submission,iw,ih,audio,error,size))return 0;
    h3_model_plan plan={0};
    int ok=h3_request_add(r,"models-path",s->config.models_root,NULL,H3_SOURCE_MANAGED,error,size)&&
        h3_models_resolve(r,&plan,error,size)&&h3_models_apply_paths(r,&plan,error,size);
    for(size_t i=0;ok&&i<plan.count;i++){
        struct stat st;
        const char *path=plan.items[i].destination;
        if(stat(path,&st)&&(!inside(s->config.model,path)&&!inside(s->config.models_root,path))){
            snprintf(error,size,"missing model outside administrator-managed destinations; provision it locally first");ok=0;
        }
    }
    h3_models_plan_free(&plan);return ok;
}
static int output_names_valid(const h3_request *r,const char *work,char *error,size_t size){
    /* A native state also owns its transfer bundle and optional sidecars. */
    const char *suffixes[]={"",".presentation",".lora.json",".h3bundle"};
    for(size_t i=0;i<r->count;i++){
        const h3_option_value *a=&r->values[i];if(a->option->path!=H3_PATH_OUTPUT)continue;
        const char *relative=a->value+strlen(work)+1;
        if(!strcmp(relative,"preview.ppm")&&h3_request_get(r,"show")){snprintf(error,size,"output name conflicts with live preview");return 0;}
        for(size_t j=i+1;j<r->count;j++){
            const h3_option_value *b=&r->values[j];if(b->option->path!=H3_PATH_OUTPUT)continue;
            size_t ac=!strncmp(a->option->name,"save-",5)?4:1,bc=!strncmp(b->option->name,"save-",5)?4:1;
            for(size_t x=0;x<ac;x++)for(size_t y=0;y<bc;y++){
                char *ap=NULL,*bp=NULL;int allocated=asprintf(&ap,"%s%s",a->value,suffixes[x])>=0&&asprintf(&bp,"%s%s",b->value,suffixes[y])>=0;
                int conflict=!allocated||!strcmp(ap,bp)||(!strncmp(ap,bp,strlen(bp))&&ap[strlen(bp)]=='/')||(!strncmp(bp,ap,strlen(ap))&&bp[strlen(ap)]=='/');free(ap);free(bp);
                if(conflict){snprintf(error,size,"--%s and --%s output names overlap",a->option->name,b->option->name);return 0;}
            }
        }
    }return 1;
}
int srv_variant_request(srv *s,const char *job,int variant,const h3_submission *submission,h3_request *r,char *error,size_t size){
    char relative[128];snprintf(relative,sizeof(relative),"jobs/%s/v%d/work",job,variant);char *work=srv_path(s->config.state,relative);
    if(!work||!srv_mkdirs(work,error,size)){free(work);return 0;}
    for(size_t i=0;i<submission->request.count;i++){
        const h3_option_value *v=&submission->request.values[i];if(!v->option->repeatable&&h3_request_get(&submission->request,v->option->name)!=v)continue;
        const char *value=v->value;char *managed=NULL;
        if(v->option->path==H3_PATH_OUTPUT||v->option->path==H3_PATH_CACHE){
            if(!safe_name(value)||!strcmp(value,"completion.json")||!strcmp(value,"worker.log")||strstr(value,".tmp")||strstr(value,".part")){snprintf(error,size,"--%s requires a safe relative artifact/cache name",v->option->name);goto fail;}
            char *base=v->option->path==H3_PATH_OUTPUT?strdup(work):srv_path(s->config.state,"cache");managed=base?srv_path(base,value):NULL;free(base);if(!managed)goto fail;
            char *parent=strdup(managed),*slash=parent?strrchr(parent,'/'):NULL;if(slash)*slash=0;
            int ok=parent&&srv_mkdirs(parent,error,size);free(parent);if(!ok){free(managed);goto fail;}value=managed;
        }
        int ok=h3_request_add(r,v->option->name,value,v->second,v->source,error,size);free(managed);if(!ok)goto fail;
    }
    const char *operation=h3_request_operation(r);
    if(!strcmp(operation,"generate")||!strcmp(operation,"still")){
        char seed[32];snprintf(seed,sizeof(seed),"%"PRIu64,submission->seeds[variant]);h3_request_remove(r,"seed");
        if(!h3_request_add(r,"seed",seed,NULL,H3_SOURCE_MANAGED,error,size))goto fail;
    }
    if(!h3_request_get(r,"output")&&strcmp(operation,"info")&&strcmp(operation,"help")&&strcmp(operation,"inspect_upscale")){
        char *path=srv_path(work,!strcmp(operation,"still")||!strcmp(operation,"decode_still")?"image.png":"video.mp4");
        int ok=path&&h3_request_add(r,"output",path,NULL,H3_SOURCE_MANAGED,error,size);free(path);if(!ok)goto fail;
    }
    if(h3_request_get(r,"lora")&&!h3_request_get(r,"lora-cache")){
        char *path=srv_path(s->config.state,"cache/lora");int ok=path&&h3_request_add(r,"lora-cache",path,NULL,H3_SOURCE_MANAGED,error,size);free(path);if(!ok)goto fail;
    }
    if (!output_names_valid(r, work, error, size))
        goto fail;
    free(work);
    return 1;
fail:free(work);h3_request_free(r);return 0;
}
static const char *asset_type(const char *path,char *error,size_t size){
    unsigned char header[16];int fd=open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);if(fd<0)return NULL;ssize_t n=read(fd,header,sizeof(header));close(fd);
    if(n==16){unsigned version=(unsigned)header[8]|((unsigned)header[9]<<8)|((unsigned)header[10]<<16)|((unsigned)header[11]<<24);
        int current=(!memcmp(header,"H3AV\r\n\032\n",8)&&version==3)||(!memcmp(header,"H3COND\0\0",8)&&version==2)||(!memcmp(header,"H3SAMPLE",8)&&version==2)||(!memcmp(header,"H3UPSRC\1",8)&&version==2);
        if(current)return "application/vnd.h3cli.state";
        uint64_t length=0;for(int i=0;i<8;i++)length|=(uint64_t)header[i]<<(i*8);
        if(length>0&&length<=16*1024*1024){h3_st_header st={0};int ok=h3_st_read_header(path,&st,error,size);h3_st_free_header(&st);if(ok)return "application/vnd.safetensors";}
    }
    sj_value *probe=NULL;if(!srv_probe(path,&probe,error,size))return NULL;
    const char *type=NULL;const sj_value *streams=sj_get(probe,"streams");
    for(size_t i=0;streams&&i<streams->count;i++){const char *kind=sj_field(streams->items[i],"codec_type"),*codec=sj_field(streams->items[i],"codec_name");
        if(kind&&!strcmp(kind,"video")){type=codec&&!strcmp(codec,"png")?"image/png":codec&&!strcmp(codec,"mjpeg")?"image/jpeg":"video/mp4";break;}
        if(kind&&!strcmp(kind,"audio"))type="audio/octet-stream";
    }sj_free(probe);if(!type)snprintf(error,size,"unrecognized media or current saved-state container");return type;
}
int srv_asset_upload(srv *s,const char *temp,const char *name,sj_value **result,srv_error *error){
    (void)name;struct stat st;if(lstat(temp,&st)||!S_ISREG(st.st_mode)||st.st_size<=0||(uint64_t)st.st_size>s->config.upload_limit)return srv_fail(error,413,"upload_limit","file","upload is empty or exceeds the asset limit");
    char id[SRV_ID],hash[65],detail[SRV_ERROR]={0};char *extracted=NULL;
    if(!srv_id(id,"asset_")||asprintf(&extracted,"%s.state",temp)<0)return srv_fail(error,500,"storage_error","file","cannot identify upload");
    int bundle=srv_bundle_unpack(temp,extracted,s->config.upload_limit,detail,sizeof(detail));
    if(!bundle){free(extracted);return srv_fail(error,400,"invalid_bundle","file","%s",detail);}
    const char *source=bundle==1?extracted:temp;
    const char *type=asset_type(source,detail,sizeof(detail));
    if(!type){const char *suffixes[]={"",".presentation",".lora.json"};for(int i=0;i<3;i++){char *p=NULL;if(asprintf(&p,"%s%s",extracted,suffixes[i])>=0){unlink(p);free(p);}}free(extracted);return srv_fail(error,400,"invalid_asset","file","unrecognized media, safetensors or current state container");}
    if(!srv_file_hash(source,hash)){free(extracted);return srv_fail(error,500,"storage_error","file","cannot hash upload");}
    char *base=srv_path(s->config.state,"assets"),*path=base?srv_path(base,id):NULL;if(!path){free(base);free(extracted);return 0;}
    pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;int ok=sqlite3_prepare_v2(s->db,"INSERT INTO assets(id,path,bytes,sha,created,content_type,bundle) VALUES(?,?,?,?,?,?,?)",-1,&q,NULL)==SQLITE_OK;
    if(ok){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,path,-1,SQLITE_TRANSIENT);struct stat primary={0};if(stat(source,&primary))ok=0;sqlite3_bind_int64(q,3,primary.st_size);sqlite3_bind_text(q,4,hash,-1,SQLITE_TRANSIENT);sqlite3_bind_double(q,5,srv_now());sqlite3_bind_text(q,6,type,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,7,bundle==1);}
    int fd=open(source,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);if(fd<0||fsync(fd))ok=0;if(fd>=0)close(fd);
    if(ok&&rename(source,path))ok=0;
    if(ok&&bundle==1){const char *suffixes[]={".presentation",".lora.json"};for(int i=0;i<2;i++){char *a=NULL,*b=NULL;if(asprintf(&a,"%s%s",source,suffixes[i])<0||asprintf(&b,"%s%s",path,suffixes[i])<0)ok=0;else if(rename(a,b)&&errno!=ENOENT)ok=0;free(a);free(b);}}
    int directory=open(base,O_RDONLY|O_CLOEXEC);if(directory<0||fsync(directory))ok=0;if(directory>=0)close(directory);
    if (ok)
        ok = sqlite3_step(q) == SQLITE_DONE;
    sqlite3_finalize(q);
    pthread_mutex_unlock(&s->mutex);
    const char *suffixes[]={"",".presentation",".lora.json"};for(int i=0;i<3;i++){char *p=NULL;if(asprintf(&p,"%s%s",extracted,suffixes[i])>=0){unlink(p);free(p);}if(!ok&&asprintf(&p,"%s%s",path,suffixes[i])>=0){unlink(p);free(p);}}
    free(base);free(extracted);free(path);if(!ok)return srv_fail(error,500,"storage_error","file","cannot commit uploaded asset");
    *result=srv_asset_get(s,id);return *result!=NULL;
}
sj_value *srv_asset_get(srv *s,const char *id){
    pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;sj_value *v=NULL;
    if(sqlite3_prepare_v2(s->db,"SELECT a.bytes,a.sha,a.created,a.owner,a.content_type,a.bundle FROM assets a LEFT JOIN jobs j ON a.owner=j.id WHERE a.id=? AND (a.owner IS NULL OR j.deleted=0)",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);
        if(sqlite3_step(q)==SQLITE_ROW){v=sj_object();sj_add(v,"id",sj_string(id));sj_add(v,"object",sj_string("asset"));sj_add(v,"content_type",sj_string((const char *)sqlite3_column_text(q,4)));sj_add(v,"bundle",sj_bool(sqlite3_column_int(q,5)));
            sj_add(v,"bytes",sj_int(sqlite3_column_int64(q,0)));sj_add(v,"sha256",sj_string((const char *)sqlite3_column_text(q,1)));sj_add(v,"created_at",sj_int((int64_t)sqlite3_column_double(q,2)));
            char uri[SRV_ID+16];snprintf(uri,sizeof(uri),"asset://%s",id);sj_add(v,"uri",sj_string(uri));}}
    sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);return v;
}
int srv_asset_delete(srv *s,const char *id,srv_error *error){
    pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;char *path=NULL;int status=404;
    if(sqlite3_prepare_v2(s->db,"SELECT path,owner,(SELECT count(*) FROM leases WHERE asset=assets.id) FROM assets WHERE id=?",-1,&q,NULL)==SQLITE_OK){
        sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);if(sqlite3_step(q)==SQLITE_ROW){status=sqlite3_column_type(q,1)!=SQLITE_NULL||sqlite3_column_int(q,2)?409:200;if(status==200)path=strdup((const char *)sqlite3_column_text(q,0));}}
    sqlite3_finalize(q);q=NULL;
    if(status==200&&path){if(sqlite3_prepare_v2(s->db,"DELETE FROM assets WHERE id=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);if(sqlite3_step(q)!=SQLITE_DONE)status=500;}else status=500;sqlite3_finalize(q);if(status==200){unlink(path);const char *suffixes[]={".presentation",".lora.json"};for(int i=0;i<2;i++){char *side=NULL;if(asprintf(&side,"%s%s",path,suffixes[i])>=0){unlink(side);free(side);}}}}
    free(path);pthread_mutex_unlock(&s->mutex);if(status==200)return 1;
    return srv_fail(error,status,status==409?"asset_in_use":"asset_unavailable","asset",status==409?"asset is leased or belongs to a job":"asset unavailable");
}
static const char *mime(const char *name){const char *ext=strrchr(name,'.');if(!ext)return "application/octet-stream";
    if(!strcmp(ext,".h3bundle"))return "application/vnd.h3cli.bundle";
    if (!strcmp(ext, ".mp4"))
        return "video/mp4";
    if (!strcmp(ext, ".png"))
        return "image/png";
    if (!strcmp(ext, ".ppm"))
        return "image/x-portable-pixmap";
    if (!strcmp(ext, ".json"))
        return "application/json";
    if (!strcmp(ext, ".log") || !strcmp(ext, ".txt"))
        return "text/plain";
    return "application/octet-stream";
}
static int collect_files(const char *root,const char *relative,sj_value *files,char *error,size_t size){
    char *dir=*relative?srv_path(root,relative):strdup(root);DIR *d=dir?opendir(dir):NULL;if(!d){free(dir);return 0;}struct dirent *entry;int ok=1;
    while(ok&&(entry=readdir(d))){if(!strcmp(entry->d_name,".")||!strcmp(entry->d_name,".."))continue;
        char *name=*relative?srv_path(relative,entry->d_name):strdup(entry->d_name),*path=name?srv_path(root,name):NULL;struct stat st;
        if(!path||lstat(path,&st)){free(name);free(path);ok=0;break;}
        if(S_ISDIR(st.st_mode)){ok=collect_files(root,name,files,error,size);free(name);free(path);continue;}
        if(!strcmp(entry->d_name,"completion.json")||strstr(entry->d_name,".tmp")||strstr(entry->d_name,".part")){free(name);free(path);continue;}
        char hash[65];if(!S_ISREG(st.st_mode)||!safe_name(name)||!srv_file_hash(path,hash)){free(name);free(path);ok=0;break;}
        int fd=open(path,O_RDONLY|O_NOFOLLOW|O_CLOEXEC);if(fd<0||fsync(fd))ok=0;if(fd>=0)close(fd);
        if(ok){sj_value *v=sj_object();sj_add(v,"name",sj_string(name));sj_add(v,"mime",sj_string(mime(name)));sj_add(v,"sha256",sj_string(hash));sj_add(v,"bytes",sj_uint((uint64_t)st.st_size));ok=sj_add(files,NULL,v);}
        free(name);free(path);
    }closedir(d);int directory=open(dir,O_RDONLY|O_CLOEXEC);if(directory<0||fsync(directory))ok=0;if(directory>=0)close(directory);free(dir);if(!ok)snprintf(error,size,"artifact validation failed");return ok;
}
int srv_prepare_publish(const char *work,sj_value *result,char *error,size_t size){
    const sj_value *outputs=sj_get(result,"outputs"),*fake=sj_get(result,"test_only");int64_t version;
    if(!sj_i64(sj_get(result,"schema"),&version)||version!=1||!outputs||outputs->type!=LJ_ARRAY){snprintf(error,size,"invalid completion manifest");return 0;}
    for(size_t i=0;i<outputs->count;i++){
        const char *name=sj_field(outputs->items[i],"name"),*option=sj_field(outputs->items[i],"option");
        if(!safe_name(name)||!option){snprintf(error,size,"invalid output manifest name");return 0;}char *path=srv_path(work,name);struct stat st;
        if(!path||lstat(path,&st)||(!S_ISREG(st.st_mode)&&!S_ISDIR(st.st_mode))){free(path);snprintf(error,size,"required output was not written: %.200s",name);return 0;}
        int ok=1;
        if(!strcmp(option,"output")&&!fake){
            sj_value *probe=NULL;ok=srv_probe(path,&probe,error,size);const sj_value *streams=sj_get(probe,"streams");int visual=0;
            for(size_t j=0;streams&&j<streams->count;j++){const sj_value *stream=streams->items[j];const char *type=sj_field(stream,"codec_type");
                if(type&&!strcmp(type,"video")){int64_t w,h;if(sj_i64(sj_get(stream,"width"),&w)&&sj_i64(sj_get(stream,"height"),&h)){sj_set(result,"width",sj_int(w));sj_set(result,"height",sj_int(h));visual=1;}
                    const char *frames=sj_field(stream,"nb_frames");if(frames){char *end;long n=strtol(frames,&end,10);if(!*end&&n>0){const sj_value *old=sj_get(result,"frames");if(old)sj_set(result,"generated_frames",sj_clone(old));sj_set(result,"frames",sj_int(n));}}
                    sj_set(result,"frame_rate",sj_clone(sj_get(stream,"r_frame_rate")));}}
            sj_free(probe);ok=ok&&visual;if(!ok&&!*error)snprintf(error,size,"output is not readable visual media");
        }
        if(ok&&!fake&&(!strncmp(option,"save-",5))){char *bundle=NULL;if(asprintf(&bundle,"%s.h3bundle",path)<0)ok=0;
            if(bundle){struct stat bs;if(lstat(bundle,&bs)==0)unlink(bundle);ok=srv_bundle_write(path,bundle,error,size);free(bundle);}}
        free(path);if(!ok)return 0;
    }
    sj_value *files=sj_array();if(!collect_files(work,"",files,error,size)){sj_free(files);return 0;}if(!sj_set(result,"artifacts",files))return 0;
    char *manifest=srv_path(work,"completion.json");int ok=manifest&&srv_write_json(manifest,result,error,size);free(manifest);return ok;
}
int srv_publish(srv *s,const char *job,int variant,const char *work,const sj_value *result,char *error,size_t size){
    const sj_value *files=sj_get(result,"artifacts");if(!files||files->type!=LJ_ARRAY){snprintf(error,size,"completion lacks validated artifacts");return 0;}
    char *parent=strdup(work);if(!parent)return 0;char *slash=strrchr(parent,'/');if(!slash){free(parent);return 0;}*slash=0;
    char *published=srv_path(parent,"result");if(!published){free(parent);return 0;}
    struct stat st;int exists=!lstat(published,&st),ok=1;
    if(!exists&&rename(work,published)){free(parent);free(published);snprintf(error,size,"cannot atomically publish artifacts");return 0;}
    #ifdef H3_SERVER_TESTING
    if(!exists&&getenv("H3_SERVER_TEST_CRASH_AFTER_RENAME"))_exit(86);
#endif
    int directory=open(parent,O_RDONLY|O_CLOEXEC);if(directory<0||fsync(directory))ok=0;if(directory>=0)close(directory);free(parent);
    for(size_t i=0;ok&&i<files->count;i++){
        const sj_value *file=files->items[i];const char *name=sj_field(file,"name"),*hash=sj_field(file,"sha256"),*type=sj_field(file,"mime");uint64_t bytes;
        if(!safe_name(name)||!hash||strlen(hash)!=64||!type||!sj_u64(sj_get(file,"bytes"),&bytes)||bytes>INT64_MAX){ok=0;break;}
        char *path=srv_path(published,name),id[SRV_ID];ok=path&&srv_id(id,"art_");sqlite3_stmt *q=NULL;
        if(ok&&sqlite3_prepare_v2(s->db,"INSERT INTO assets(id,path,bytes,sha,created,owner,content_type) VALUES(?,?,?,?,?,?,?)",-1,&q,NULL)==SQLITE_OK){
            sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,path,-1,SQLITE_TRANSIENT);sqlite3_bind_int64(q,3,(sqlite3_int64)bytes);sqlite3_bind_text(q,4,hash,-1,SQLITE_TRANSIENT);sqlite3_bind_double(q,5,srv_now());sqlite3_bind_text(q,6,job,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,7,type,-1,SQLITE_TRANSIENT);ok=sqlite3_step(q)==SQLITE_DONE;
        }else ok=0;sqlite3_finalize(q);q=NULL;
        if(ok&&sqlite3_prepare_v2(s->db,"INSERT INTO artifacts(id,job,variant,name,mime) VALUES(?,?,?,?,?)",-1,&q,NULL)==SQLITE_OK){
            sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,3,variant);sqlite3_bind_text(q,4,name,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,5,type,-1,SQLITE_TRANSIENT);ok=sqlite3_step(q)==SQLITE_DONE;
        }else ok=0;sqlite3_finalize(q);free(path);
    }free(published);if(!ok)snprintf(error,size,"cannot commit artifact metadata");return ok;
}

/* Tombstones retain idempotency/history; storage is collected only after leases end. */
void srv_gc(srv *s,int startup){
    pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;sj_value *jobs=sj_array();
    if(sqlite3_prepare_v2(s->db,"SELECT id FROM jobs WHERE deleted=1 AND status NOT IN ('queued','running') AND NOT EXISTS (SELECT 1 FROM assets a JOIN leases l ON l.asset=a.id WHERE a.owner=jobs.id)",-1,&q,NULL)==SQLITE_OK)
        while(sqlite3_step(q)==SQLITE_ROW)sj_add(jobs,NULL,sj_string((const char *)sqlite3_column_text(q,0)));
    sqlite3_finalize(q);
    for(size_t i=0;i<jobs->count;i++){const char *id=sj_text(jobs->items[i]);q=NULL;
        if(sqlite3_prepare_v2(s->db,"DELETE FROM assets WHERE owner=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);
        char relative[96];snprintf(relative,sizeof(relative),"jobs/%s",id);char *p=srv_path(s->config.state,relative);if(p){srv_remove_tree(p);free(p);}
    }sj_free(jobs);
    if(startup){
        char *incoming=srv_path(s->config.state,"incoming");DIR *d=incoming?opendir(incoming):NULL;struct dirent *entry;
        while (d && (entry = readdir(d)))
            if (strcmp(entry->d_name, ".") && strcmp(entry->d_name, "..")) {
                char *p = srv_path(incoming, entry->d_name);
                if (p) {
                    srv_remove_tree(p);
                    free(p);
                }
            }
        if (d)
            closedir(d);
        free(incoming);
        const char *dirs[]={"jobs","assets"};for(int i=0;i<2;i++){char *base=srv_path(s->config.state,dirs[i]);d=base?opendir(base):NULL;
            while(d&&(entry=readdir(d))){
                if(!strcmp(entry->d_name,".")||!strcmp(entry->d_name,".."))continue;
                /* Compare complete IDs; truncation could retain an orphan that
                 * happens to share a prefix with a live job or asset. */
                char id[sizeof(entry->d_name)];
                memcpy(id,entry->d_name,strlen(entry->d_name)+1);
                char *dot=strchr(id,'.');if(dot)*dot=0;
                q=NULL;int keep=1;const char *sql=i?"SELECT 1 FROM assets WHERE id=?":"SELECT 1 FROM jobs WHERE id=?";
                if(sqlite3_prepare_v2(s->db,sql,-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);keep=sqlite3_step(q)==SQLITE_ROW;}sqlite3_finalize(q);
                if(!keep){char *p=srv_path(base,entry->d_name);if(p){srv_remove_tree(p);free(p);}}}
            if (d)
                closedir(d);
            free(base);
        }
    }pthread_mutex_unlock(&s->mutex);
}
