#include "src/server/internal.h"
#include "src/h3.h"
#include <math.h>
#include <errno.h>
#include <fcntl.h>
#include <inttypes.h>
#include <poll.h>
#include <signal.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/resource.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
int srv_db_exec(srv *s,const char *sql,char *error,size_t size){char *message=NULL;int rc=sqlite3_exec(s->db,sql,NULL,NULL,&message);
    if(rc!=SQLITE_OK)atomic_store(&s->storage_failed,1);
    if (rc != SQLITE_OK)
        snprintf(error, size, "queue database: %s", message ? message : sqlite3_errmsg(s->db));
    sqlite3_free(message);
    return rc == SQLITE_OK;
}
static int sql(srv *s,const char *text){int ok=sqlite3_exec(s->db,text,NULL,NULL,NULL)==SQLITE_OK;if(!ok)atomic_store(&s->storage_failed,1);else if(!strcmp(text,"COMMIT"))atomic_store(&s->storage_failed,0);return ok;}
static void release_inputs(srv *s,const char *id){sqlite3_stmt *q=NULL;if(sqlite3_prepare_v2(s->db,"DELETE FROM leases WHERE owner=? AND kind='input'",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);}
static void aggregate(srv *s,const char *id){
    sqlite3_stmt *q=NULL;int total=0,complete=0,failed=0,cancelled=0,running=0,interrupted=0,preparing=0;
    if(sqlite3_prepare_v2(s->db,"SELECT status,count(*) FROM variants WHERE job=? GROUP BY status",-1,&q,NULL)==SQLITE_OK){
        sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);while(sqlite3_step(q)==SQLITE_ROW){const char *status=(const char *)sqlite3_column_text(q,0);int n=sqlite3_column_int(q,1);total+=n;
            if(!strcmp(status,"completed"))complete+=n;else if(!strcmp(status,"failed"))failed+=n;else if(!strcmp(status,"cancelled"))cancelled+=n;else if(!strcmp(status,"interrupted"))interrupted+=n;else if(!strcmp(status,"running"))running+=n;else if(!strcmp(status,"preparing"))preparing+=n;}}
    sqlite3_finalize(q);q=NULL;
    const char *status=failed?"failed":interrupted?"interrupted":cancelled&&!running&&!preparing?"cancelled":complete==total?"completed":running||complete?"running":"queued";
    int terminal=!strcmp(status,"failed")||!strcmp(status,"interrupted")||!strcmp(status,"cancelled")||!strcmp(status,"completed");
    if(sqlite3_prepare_v2(s->db,"UPDATE jobs SET status=?,progress=CASE WHEN ?='completed' THEN 100 ELSE progress END,finished=CASE WHEN ? THEN ? ELSE finished END WHERE id=?",-1,&q,NULL)==SQLITE_OK){
        sqlite3_bind_text(q,1,status,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,status,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,3,terminal);sqlite3_bind_double(q,4,srv_now());sqlite3_bind_text(q,5,id,-1,SQLITE_TRANSIENT);sqlite3_step(q);}
    sqlite3_finalize(q);if(terminal)release_inputs(s,id);
}
static int schema(srv *s,char *error,size_t size){
    sqlite3_stmt *q=NULL;int version=-1;if(sqlite3_prepare_v2(s->db,"PRAGMA user_version",-1,&q,NULL)==SQLITE_OK&&sqlite3_step(q)==SQLITE_ROW)version=sqlite3_column_int(q,0);sqlite3_finalize(q);
    if(version!=0&&version!=1&&version!=2&&version!=3){snprintf(error,size,"unsupported queue schema %d",version);return 0;}
    if(version==0){
        q=NULL;int tables=0;if(sqlite3_prepare_v2(s->db,"SELECT count(*) FROM sqlite_master WHERE type='table' AND name NOT LIKE 'sqlite_%'",-1,&q,NULL)==SQLITE_OK&&sqlite3_step(q)==SQLITE_ROW)tables=sqlite3_column_int(q,0);sqlite3_finalize(q);
        if(tables){snprintf(error,size,"unversioned nonempty queue database");return 0;}
        const char *create=
          "BEGIN IMMEDIATE;"
          "CREATE TABLE jobs(id TEXT PRIMARY KEY,created REAL NOT NULL,started REAL,finished REAL,status TEXT NOT NULL,video INTEGER NOT NULL,n INTEGER NOT NULL,model TEXT NOT NULL,request TEXT NOT NULL,effective TEXT NOT NULL,progress INTEGER NOT NULL DEFAULT 0,deleted INTEGER NOT NULL DEFAULT 0,cancel INTEGER NOT NULL DEFAULT 0,error TEXT,error_code TEXT,idempotency TEXT UNIQUE,fingerprint TEXT NOT NULL,build TEXT NOT NULL);"
          "CREATE TABLE variants(job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,idx INTEGER NOT NULL,status TEXT NOT NULL,request TEXT NOT NULL,seed TEXT NOT NULL,started REAL,finished REAL,result TEXT,pid INTEGER,PRIMARY KEY(job,idx));"
          "CREATE TABLE assets(id TEXT PRIMARY KEY,path TEXT NOT NULL,bytes INTEGER NOT NULL,sha TEXT NOT NULL,created REAL NOT NULL,owner TEXT REFERENCES jobs(id) ON DELETE CASCADE,content_type TEXT NOT NULL DEFAULT 'application/octet-stream',bundle INTEGER NOT NULL DEFAULT 0);"
          "CREATE TABLE leases(owner TEXT NOT NULL,asset TEXT NOT NULL REFERENCES assets(id),kind TEXT NOT NULL,PRIMARY KEY(owner,asset,kind));"
          "CREATE TABLE artifacts(id TEXT PRIMARY KEY REFERENCES assets(id) ON DELETE CASCADE,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,variant INTEGER NOT NULL,name TEXT NOT NULL,mime TEXT NOT NULL,UNIQUE(job,variant,name));"
          "CREATE TABLE events(seq INTEGER PRIMARY KEY AUTOINCREMENT,job TEXT NOT NULL REFERENCES jobs(id) ON DELETE CASCADE,data TEXT NOT NULL);"
          "CREATE INDEX queue_fifo ON jobs(status,created);CREATE INDEX artifact_jobs ON artifacts(job,variant);CREATE INDEX event_jobs ON events(job,seq);"
          "PRAGMA user_version=3;COMMIT;";
        if(!srv_db_exec(s,create,error,size)){sql(s,"ROLLBACK");return 0;}
    }
    if(version==1&&!srv_db_exec(s,"BEGIN IMMEDIATE;ALTER TABLE jobs ADD COLUMN build TEXT NOT NULL DEFAULT '';PRAGMA user_version=2;COMMIT;",error,size)){sql(s,"ROLLBACK");return 0;}
    if((version==1||version==2)&&!srv_db_exec(s,"BEGIN IMMEDIATE;ALTER TABLE assets ADD COLUMN content_type TEXT NOT NULL DEFAULT 'application/octet-stream';ALTER TABLE assets ADD COLUMN bundle INTEGER NOT NULL DEFAULT 0;PRAGMA user_version=3;COMMIT;",error,size)){sql(s,"ROLLBACK");return 0;}
    q=NULL;int ok=sqlite3_prepare_v2(s->db,"PRAGMA quick_check",-1,&q,NULL)==SQLITE_OK&&sqlite3_step(q)==SQLITE_ROW&&!strcmp((const char *)sqlite3_column_text(q,0),"ok");sqlite3_finalize(q);
    if (!ok)
        snprintf(error, size, "queue database integrity check failed");
    return ok;
}
int srv_db_open(srv *s,char *error,size_t size){
    char *path=srv_path(s->config.state,"queue.sqlite3");if(!path)return 0;
    int rc=sqlite3_open_v2(path,&s->db,SQLITE_OPEN_READWRITE|SQLITE_OPEN_CREATE|SQLITE_OPEN_FULLMUTEX,NULL);free(path);
    if(rc!=SQLITE_OK){snprintf(error,size,"cannot open queue database");return 0;}sqlite3_busy_timeout(s->db,5000);
    if(!srv_db_exec(s,"PRAGMA journal_mode=WAL;PRAGMA synchronous=FULL;PRAGMA foreign_keys=ON;",error,size)||!schema(s,error,size))return 0;
    // Reconcile workers that finished publication immediately before a parent crash.
    sqlite3_stmt *q=NULL;
    if(sqlite3_prepare_v2(s->db,"SELECT job,idx FROM variants WHERE status IN ('running','preparing')",-1,&q,NULL)!=SQLITE_OK)return 0;
    sj_value *pending=sj_array();while(sqlite3_step(q)==SQLITE_ROW){sj_value *v=sj_object();sj_add(v,"job",sj_string((const char *)sqlite3_column_text(q,0)));sj_add(v,"idx",sj_int(sqlite3_column_int(q,1)));sj_add(pending,NULL,v);}sqlite3_finalize(q);
    if(pending->count){struct timespec grace={3,200000000};while(nanosleep(&grace,&grace)&&errno==EINTR){}}
    for(size_t i=0;i<pending->count;i++){
        const char *job=sj_field(pending->items[i],"job");int64_t index=0;sj_i64(sj_get(pending->items[i],"idx"),&index);
        char rel[160];snprintf(rel,sizeof(rel),"jobs/%s/v%lld/result/completion.json",job,(long long)index);char *manifest=srv_path(s->config.state,rel);
        sj_value *result=manifest?srv_read_json(manifest,error,size):NULL;free(manifest);
        if(!result){snprintf(rel,sizeof(rel),"jobs/%s/v%lld/work/completion.json",job,(long long)index);manifest=srv_path(s->config.state,rel);result=manifest?srv_read_json(manifest,error,size):NULL;free(manifest);}
        snprintf(rel,sizeof(rel),"jobs/%s/v%lld/work",job,(long long)index);char *work=srv_path(s->config.state,rel);
        int recovered=0;char *text=result?sj_dump(result):NULL;
        char *ready=work?strdup(work):NULL;struct stat check;
        if(ready&&stat(ready,&check)){free(ready);snprintf(rel,sizeof(rel),"jobs/%s/v%lld/result",job,(long long)index);ready=srv_path(s->config.state,rel);}
        int prepared=result&&ready&&srv_prepare_publish(ready,result,error,size);free(ready);free(text);text=prepared?sj_dump(result):NULL;
        if(prepared&&work&&sql(s,"BEGIN IMMEDIATE")){
            recovered=srv_publish(s,job,(int)index,work,result,error,size);if(recovered){
                q=NULL;recovered=sqlite3_prepare_v2(s->db,"UPDATE variants SET status='completed',finished=?,result=?,pid=NULL WHERE job=? AND idx=?",-1,&q,NULL)==SQLITE_OK;
                if(recovered){sqlite3_bind_double(q,1,srv_now());sqlite3_bind_text(q,2,text,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,3,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,4,(int)index);recovered=sqlite3_step(q)==SQLITE_DONE;}sqlite3_finalize(q);}
            if(recovered)recovered=sql(s,"COMMIT");else sql(s,"ROLLBACK");}
        if(!recovered){q=NULL;if(sqlite3_prepare_v2(s->db,"UPDATE variants SET status='interrupted',finished=?,pid=NULL WHERE job=? AND idx=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_double(q,1,srv_now());sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,3,(int)index);sqlite3_step(q);}sqlite3_finalize(q);}
        if(!recovered){q=NULL;if(sqlite3_prepare_v2(s->db,"UPDATE variants SET status='cancelled',finished=? WHERE job=? AND status='queued'",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_double(q,1,srv_now());sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);}
        free(work);free(text);sj_free(result);aggregate(s,job);
    }sj_free(pending);error[0]=0;
    return srv_db_exec(s,"DELETE FROM leases WHERE kind='download' OR (kind='input' AND owner NOT IN (SELECT id FROM jobs WHERE status IN ('queued','running')));",error,size);
}
static sj_value *parse_stored(const unsigned char *text){char error[128];return text?sj_parse((const char *)text,strlen((const char *)text),error,sizeof(error)):NULL;}
static sj_value *job_locked(srv *s,const char *id,int video,int include_deleted){
    sqlite3_stmt *q=NULL;sj_value *v=NULL,*h=NULL;const char *query="SELECT created,started,finished,status,video,n,model,effective,progress,deleted,error,error_code,build FROM jobs WHERE id=?";
    if (sqlite3_prepare_v2(s->db, query, -1, &q, NULL) != SQLITE_OK)
        return NULL;
    sqlite3_bind_text(q, 1, id, -1, SQLITE_TRANSIENT);
    if(sqlite3_step(q)!=SQLITE_ROW||(!include_deleted&&sqlite3_column_int(q,9))||(video&&!sqlite3_column_int(q,4))){sqlite3_finalize(q);return NULL;}
    char internal_copy[32];snprintf(internal_copy,sizeof(internal_copy),"%s",sqlite3_column_text(q,3));const char *internal=internal_copy,*status=internal;int deleted=sqlite3_column_int(q,9),count=sqlite3_column_int(q,5);
    if(video){if(deleted)status="deleted";else if(!strcmp(internal,"running"))status="in_progress";else if(!strcmp(internal,"cancelled")||!strcmp(internal,"interrupted"))status="failed";}
    v=sj_object();h=sj_object();sj_add(v,"id",sj_string(id));sj_add(v,"object",sj_string(video?"video":"h3.job"));sj_add(v,"model",sj_string((const char *)sqlite3_column_text(q,6)));
    sj_add(v,"status",sj_string(status));sj_add(v,"progress",sj_int(sqlite3_column_int(q,8)));sj_add(v,"created_at",sj_int((int64_t)sqlite3_column_double(q,0)));
    sj_add(v,"completed_at",sqlite3_column_type(q,2)==SQLITE_NULL?sj_null():sj_int((int64_t)sqlite3_column_double(q,2)));
    sj_add(v,"num_outputs",sj_int(count));sj_add(v,"file_path",sj_null());sj_add(v,"file_paths",sj_null());
    sj_value *effective=parse_stored(sqlite3_column_text(q,7));if(!effective)effective=sj_object();sj_add(h,"effective",effective);sj_add(h,"state",sj_string(internal));sj_add(h,"deleted",sj_bool(deleted));
    double created=sqlite3_column_double(q,0),started=sqlite3_column_type(q,1)==SQLITE_NULL?0:sqlite3_column_double(q,1),finished=sqlite3_column_type(q,2)==SQLITE_NULL?0:sqlite3_column_double(q,2);
    sj_add(h,"queue_seconds",sj_real((started!=0?started:srv_now())-created));sj_add(h,"run_seconds",sj_real(started!=0?((finished!=0?finished:srv_now())-started):0));
    sj_add(h,"total_seconds",sj_real((finished!=0?finished:srv_now())-created));sj_add(h,"admitted_build_sha256",sj_string((const char *)sqlite3_column_text(q,12)));
    const char *message=(const char *)sqlite3_column_text(q,10),*code=(const char *)sqlite3_column_text(q,11);
    if(message||!strcmp(internal,"cancelled")||!strcmp(internal,"interrupted")){sj_value *error=sj_object();sj_add(error,"code",sj_string(code?code:internal));sj_add(error,"message",sj_string(message?message:internal));sj_add(v,"error",error);}else sj_add(v,"error",sj_null());
    sqlite3_finalize(q);q=NULL;
    sj_value *variants=sj_array();
    if(sqlite3_prepare_v2(s->db,"SELECT idx,status,result,seed FROM variants WHERE job=? ORDER BY idx",-1,&q,NULL)==SQLITE_OK){
        sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);while(sqlite3_step(q)==SQLITE_ROW){sj_value *row=sj_object();sj_add(row,"index",sj_int(sqlite3_column_int(q,0)));sj_add(row,"status",sj_string((const char *)sqlite3_column_text(q,1)));
            sj_value *result=parse_stored(sqlite3_column_text(q,2));sj_add(row,"result",result?result:sj_null());const sj_value *actual_seed=sj_get(result,"seed");const char *stored=(const char *)sqlite3_column_text(q,3);
            sj_add(row,"seed",actual_seed&&actual_seed->text?sj_string(actual_seed->text):sj_string(stored&&strcmp(stored,"inherited")?stored:NULL));sj_add(variants,NULL,row);}}
    sqlite3_finalize(q);q=NULL;sj_add(h,"variants",variants);
    if(sqlite3_prepare_v2(s->db,"SELECT data FROM events WHERE job=? ORDER BY seq DESC",-1,&q,NULL)==SQLITE_OK){
        sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);
        while(sqlite3_step(q)==SQLITE_ROW){
            sj_value *e=parse_stored(sqlite3_column_text(q,0));const char *type=sj_field(e,"type");
            if(type&&!strcmp(type,"model_preparation")){
                if(!strcmp(internal,"queued"))sj_add(h,"phase",sj_string(sj_field(e,"phase")));
                sj_add(h,"model_preparation",e);break;}sj_free(e);
        }
    }
    sqlite3_finalize(q);q=NULL;
    const char *worker_build=NULL;for(size_t i=0;i<variants->count;i++){worker_build=sj_field(sj_get(variants->items[i],"result"),"worker_build_sha256");if(worker_build)break;}
    sj_add(h,"worker_build_sha256",sj_string(worker_build));
    sj_value *artifacts=sj_array();
    if(sqlite3_prepare_v2(s->db,"SELECT a.id,a.variant,a.name,a.mime,b.bytes,b.sha FROM artifacts a JOIN assets b ON a.id=b.id WHERE a.job=? ORDER BY a.variant,a.name",-1,&q,NULL)==SQLITE_OK){
        sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);while(sqlite3_step(q)==SQLITE_ROW){sj_value *row=sj_object();const char *art=(const char *)sqlite3_column_text(q,0);char url[180],uri[180];
            snprintf(url,sizeof(url),"/v1/h3/jobs/%s/artifacts/%s",id,art);snprintf(uri,sizeof(uri),"artifact://%s/%s",id,art);
            sj_add(row,"id",sj_string(art));sj_add(row,"variant",sj_int(sqlite3_column_int(q,1)));sj_add(row,"name",sj_string((const char *)sqlite3_column_text(q,2)));sj_add(row,"content_type",sj_string((const char *)sqlite3_column_text(q,3)));
            sj_add(row,"bytes",sj_int(sqlite3_column_int64(q,4)));sj_add(row,"sha256",sj_string((const char *)sqlite3_column_text(q,5)));sj_add(row,"url",sj_string(url));sj_add(row,"uri",sj_string(uri));sj_add(artifacts,NULL,row);}}
    sqlite3_finalize(q);sj_add(h,"artifacts",artifacts);
    const sj_value *geometry=effective;for(size_t i=0;i<variants->count;i++){const sj_value *r=sj_get(variants->items[i],"result");if(r&&r->type==LJ_OBJECT&&sj_get(r,"width")){geometry=r;break;}}
    int64_t w=0,ht=0,frames=0;sj_i64(sj_get(geometry,"width"),&w);sj_i64(sj_get(geometry,"height"),&ht);sj_i64(sj_get(geometry,"frames"),&frames);
    char dimensions[64],seconds[64];snprintf(dimensions,sizeof(dimensions),"%lldx%lld",(long long)w,(long long)ht);snprintf(seconds,sizeof(seconds),"%.6g",(double)frames/24);
    sj_add(v,"size",sj_string(w&&ht?dimensions:""));sj_add(v,"seconds",sj_string(frames?seconds:""));const char *quality=sj_field(effective,"quality");sj_add(v,"quality",sj_string(quality?quality:"standard"));
    sj_add(v,"url",sj_null());if(video&&!strcmp(internal,"completed")){char url[128];snprintf(url,sizeof(url),"/v1/videos/%s/content",id);sj_set(v,"url",sj_string(url));}
    sj_add(v,"expires_at",sj_null());sj_add(v,"remixed_from_video_id",sj_null());sj_add(v,"revised_prompt",sj_null());sj_add(v,"action",sj_null());
    const sj_value *first=variants->count?sj_get(variants->items[0],"result"):NULL;const sj_value *wall=sj_get(first,"wall_seconds"),*rss=sj_get(first,"peak_rss_bytes");
    sj_add(v,"inference_time_s",wall?sj_clone(wall):sj_null());sj_add(v,"peak_memory_mb",rss?sj_real(rss->number/(1024*1024)):sj_null());
    sj_add(v,"h3",h);return v;
}
sj_value *srv_job(srv *s,const char *id,int video,int include_deleted){pthread_mutex_lock(&s->mutex);sj_value *v=job_locked(s,id,video,include_deleted);pthread_mutex_unlock(&s->mutex);return v;}
sj_value *srv_jobs(srv *s,int video,const char *after,int limit,int ascending){
    pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;sj_value *v=sj_object(),*a=sj_array();sj_add(v,"object",sj_string("list"));sj_add(v,"data",a);
    const char *query=ascending?
      "SELECT id FROM jobs WHERE deleted=0 AND (?=0 OR video=1) AND (? IS NULL OR (created,id)>(SELECT created,id FROM jobs WHERE id=?)) ORDER BY created,id LIMIT ?":
      "SELECT id FROM jobs WHERE deleted=0 AND (?=0 OR video=1) AND (? IS NULL OR (created,id)<(SELECT created,id FROM jobs WHERE id=?)) ORDER BY created DESC,id DESC LIMIT ?";
    if(sqlite3_prepare_v2(s->db,query,-1,&q,NULL)==SQLITE_OK){sqlite3_bind_int(q,1,video);if(after){sqlite3_bind_text(q,2,after,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,3,after,-1,SQLITE_TRANSIENT);}else {sqlite3_bind_null(q,2);sqlite3_bind_null(q,3);}sqlite3_bind_int(q,4,limit);
        while(sqlite3_step(q)==SQLITE_ROW){sj_value *job=job_locked(s,(const char *)sqlite3_column_text(q,0),video,0);if(job)sj_add(a,NULL,job);}}
    sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);return v;
}
void srv_event_locked(srv *s,const char *job,const sj_value *event){
    char *text=sj_dump(event);sqlite3_stmt *q=NULL;if(text&&sqlite3_prepare_v2(s->db,"INSERT INTO events(job,data) VALUES(?,?)",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,job,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,text,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);free(text);q=NULL;
    if(sqlite3_prepare_v2(s->db,"DELETE FROM events WHERE job=? AND seq < coalesce((SELECT seq FROM events WHERE job=? ORDER BY seq DESC LIMIT 1 OFFSET 511),0)",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,job,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);
}
sj_value *srv_events(srv *s,const char *job,int64_t after){
    pthread_mutex_lock(&s->mutex);sj_value *a=sj_array();sqlite3_stmt *q=NULL;
    if(sqlite3_prepare_v2(s->db,"SELECT seq,data FROM events WHERE job=? AND seq>? ORDER BY seq LIMIT 512",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int64(q,2,after);
        while(sqlite3_step(q)==SQLITE_ROW){sj_value *v=parse_stored(sqlite3_column_text(q,1));if(v){sj_add(v,"event_id",sj_int(sqlite3_column_int64(q,0)));sj_add(a,NULL,v);}}}
    sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);return a;
}
static int option_int(const h3_request *r,const char *name,int fallback){const h3_option_value *v=h3_request_get(r,name);return v?atoi(v->value):fallback;}
static sj_value *effective(const h3_submission *sub){
    const h3_request *r=&sub->request;const char *op=h3_request_operation(r);sj_value *v=sj_object(),*overrides=sj_array();
    sj_add(v,"operation",sj_string(op));sj_add(v,"task",sj_string(sub->task));sj_add(v,"width",sj_int(option_int(r,"width",H3_DEFAULT_WIDTH)));sj_add(v,"height",sj_int(option_int(r,"height",H3_DEFAULT_HEIGHT)));
    int frames=option_int(r,"frames",H3_DEFAULT_FRAMES);const h3_option_value *seconds=h3_request_get(r,"seconds");if(seconds)frames=(int)llround(strtod(seconds->value,NULL)*24);
    if(!strcmp(op,"still"))frames=1;else frames+=(5-frames%17+17)%17;sj_add(v,"frames",sj_int(frames));
    const h3_option_value *quality=h3_request_get(r,"quality");const h3_quality_preset *preset=NULL;
    if(quality)for(size_t i=0;i<5;i++)if(!strcmp(quality->value,h3_quality_presets[i].name))preset=&h3_quality_presets[i];
    sj_add(v,"quality",sj_string(quality?quality->value:NULL));sj_add(v,"steps",sj_int(option_int(r,"steps",preset?preset->steps:H3_DEFAULT_STEPS)));
    int reuse=preset?preset->reuse:1,adaptive=preset?preset->adaptive:0;
#ifdef __APPLE__
    if(adaptive){reuse=2;adaptive=0;}
#endif
    sj_add(v,"reuse",sj_int(option_int(r,"reuse",reuse)));const h3_option_value *cache=h3_request_get(r,"adaptive-cache");sj_add(v,"adaptive_cache",sj_string(cache?cache->value:adaptive?"conservative":"off"));
    int preview=preset?preset->preview:0;for(size_t i=0;i<r->count;i++){const h3_option_value *x=&r->values[i];if(!strcmp(x->option->group,"preview"))preview=!strcmp(x->option->name,"preview-vae");if(x->source==H3_SOURCE_NATIVE)sj_add(overrides,NULL,sj_string(x->option->name));}
    sj_add(v,"zoom",sj_int(option_int(r,"zoom",2)));sj_add(v,"preview_vae",sj_bool(preview));sj_add(v,"native_overrides",overrides);
    if(strcmp(op,"generate")&&strcmp(op,"still")){
        const char *fields[]={"width","height","frames","steps","reuse"};
        for(size_t i=0;i<5;i++)if(!h3_request_get(r,fields[i]))sj_set(v,fields[i],sj_null());
        sj_add(v,"restored_from_state",sj_bool(strcmp(op,"help")&&strcmp(op,"info")));
    }
    return v;
}
static int submit_locked(srv *s,const sj_value *body,int video,const char *key,sj_value **result,srv_error *error){
    *result=NULL;h3_submission submission={0};char detail[SRV_ERROR]={0},id[SRV_ID]={0},fingerprint[65];int reserved=0,ok=0,durable=0;char *folder=NULL,*request_text=NULL,*effective_text=NULL;
    h3_request requests[10]={{0}};
    if(!h3_submission_parse(body,&submission,detail,sizeof(detail)))return srv_fail(error,400,"invalid_request","request","%s",detail);
    const char *op=h3_request_operation(&submission.request);
    if((!strcmp(op,"generate")||!strcmp(op,"still"))&&!h3_request_get(&submission.request,"prompt")){srv_fail(error,400,"missing_prompt","prompt","fresh generation requires a prompt or -p in h3cli");goto done;}
    if(video&&(!strcmp(op,"still")||!strcmp(op,"decode_still")||!strcmp(op,"info")||!strcmp(op,"help")||!strcmp(op,"inspect_upscale")||h3_request_get(&submission.request,"state-only")||h3_request_get(&submission.request,"stop-after-step"))){srv_fail(error,400,"requires_native_endpoint","h3cli","use /v1/h3/jobs for still, checkpoint, state-only and inspection operations");goto done;}
    sj_value *canonical=sj_object(),*scalar=sj_object(),*repeat=sj_array();
    for(size_t i=0;i<submission.request.count;i++){
        const h3_option_value *v=&submission.request.values[i];
        if(v->option->repeatable){sj_value *item=sj_object();sj_add(item,"name",sj_string(v->option->name));sj_add(item,"value",sj_string(v->value));if(v->second)sj_add(item,"second",sj_string(v->second));sj_add(repeat,NULL,item);}
        else {sj_value *item=sj_object();sj_add(item,"name",sj_string(v->option->name));sj_add(item,"value",v->value?sj_string(v->value):sj_bool(1));sj_set(scalar,v->option->group,item);}
    }
    sj_add(canonical,"scalar",scalar);sj_add(canonical,"repeat",repeat);sj_add(canonical,"video_endpoint",sj_bool(video));sj_add(canonical,"model",sj_string(h3_request_get(&submission.request,"model-dir")?"native":submission.model));sj_add(canonical,"variants",sj_int(submission.variants));
    const sj_value *target=sj_get(submission.transport,"target");if(target&&(submission.width_needed||submission.height_needed||submission.duration_needed))sj_add(canonical,"target",sj_clone(target));
    sj_value *seeds=sj_array();for(int i=0;i<submission.variants;i++)sj_add(seeds,NULL,sj_uint(submission.seeds[i]));sj_add(canonical,"seeds",seeds);
    sj_sort(canonical);request_text=sj_dump(canonical);sj_free(canonical);if(!request_text||!srv_text_hash(request_text,fingerprint)||!srv_id(id,"job_"))goto storage;
    pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;
    if(key&&*key){
        if(strlen(key)>256){pthread_mutex_unlock(&s->mutex);srv_fail(error,400,"invalid_idempotency_key","Idempotency-Key","key exceeds 256 bytes");goto done;}
        if(sqlite3_prepare_v2(s->db,"SELECT id,fingerprint FROM jobs WHERE idempotency=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,key,-1,SQLITE_TRANSIENT);
            if(sqlite3_step(q)==SQLITE_ROW){int match=!strcmp((const char *)sqlite3_column_text(q,1),fingerprint);if(match)*result=job_locked(s,(const char *)sqlite3_column_text(q,0),video,1);sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);if(!match)srv_fail(error,409,"idempotency_conflict","Idempotency-Key","key was used for a different effective request");ok=match&&*result;goto done;}}
        sqlite3_finalize(q);q=NULL;
    }
    int waiting=0;if(sqlite3_prepare_v2(s->db,"SELECT count(*) FROM variants WHERE status IN ('queued','preparing')",-1,&q,NULL)==SQLITE_OK&&sqlite3_step(q)==SQLITE_ROW)waiting=sqlite3_column_int(q,0);sqlite3_finalize(q);
    if(atomic_load(&s->stopping)||waiting+s->reservations+submission.variants>s->config.queue_limit){pthread_mutex_unlock(&s->mutex);srv_fail(error,429,"queue_full","n","queue has no capacity for all variants");goto done;}
    s->reservations+=submission.variants;reserved=1;pthread_mutex_unlock(&s->mutex);
    char relative[96];snprintf(relative,sizeof(relative),"jobs/%s",id);folder=srv_path(s->config.state,relative);
    if(!folder||!srv_mkdirs(folder,detail,sizeof(detail)))goto storage;
    if(!srv_assets_prepare(s,&submission,id,detail,sizeof(detail))){srv_fail(error,400,"invalid_input","request","%s",detail);goto done;}
    if(!srv_resources_save(s,&submission.request,id,detail,sizeof(detail))){srv_fail(error,400,"model_resource","request","%s",detail);goto done;}
    if(srv_tree_bytes(s->config.state)>s->config.storage_limit){srv_fail(error,413,"storage_quota","request","managed storage quota exhausted");goto done;}
    sj_value *metadata=effective(&submission);effective_text=sj_dump(metadata);sj_free(metadata);if(!effective_text)goto storage;
    for(int i=0;i<submission.variants;i++){
        if(!srv_variant_request(s,id,i,&submission,&requests[i],detail,sizeof(detail))){srv_fail(error,400,"invalid_path","h3cli","%s",detail);goto done;}
        snprintf(relative,sizeof(relative),"jobs/%s/v%d/work",id,i);char *work=srv_path(s->config.state,relative);
        int valid=work&&srv_worker_validate(s,&requests[i],work,detail,sizeof(detail));free(work);
        if(!valid){srv_fail(error,400,"native_preflight","h3cli","%s",detail);goto done;}
    }
    pthread_mutex_lock(&s->mutex);int committed=sql(s,"BEGIN IMMEDIATE");q=NULL;
    if(committed&&sqlite3_prepare_v2(s->db,"INSERT INTO jobs(id,created,status,video,n,model,request,effective,idempotency,fingerprint,build) VALUES(?,?,'queued',?,?,?,?,?,?,?,?)",-1,&q,NULL)==SQLITE_OK){
        sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);sqlite3_bind_double(q,2,srv_now());sqlite3_bind_int(q,3,video);sqlite3_bind_int(q,4,submission.variants);
        sqlite3_bind_text(q,5,h3_request_get(&submission.request,"model-dir")?s->config.model_id:submission.model,-1,SQLITE_TRANSIENT);
        sqlite3_bind_text(q,6,request_text,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,7,effective_text,-1,SQLITE_TRANSIENT);if(key&&*key)sqlite3_bind_text(q,8,key,-1,SQLITE_TRANSIENT);else sqlite3_bind_null(q,8);sqlite3_bind_text(q,9,fingerprint,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,10,s->executable_sha,-1,SQLITE_TRANSIENT);committed=sqlite3_step(q)==SQLITE_DONE;
    }else committed=0;sqlite3_finalize(q);q=NULL;
    for(int i=0;committed&&i<submission.variants;i++){
        sj_value *encoded=h3_request_json(&requests[i]);char *text=sj_dump(encoded);sj_free(encoded);char seed[32];if(strcmp(op,"generate")&&strcmp(op,"still")&&!h3_request_get(&submission.request,"seed"))snprintf(seed,sizeof(seed),"inherited");
        else snprintf(seed,sizeof(seed),"%"PRIu64,submission.seeds[i]);
        committed=text&&sqlite3_prepare_v2(s->db,"INSERT INTO variants(job,idx,status,request,seed) VALUES(?,?,'queued',?,?)",-1,&q,NULL)==SQLITE_OK;
        if(committed){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,2,i);sqlite3_bind_text(q,3,text,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,4,seed,-1,SQLITE_TRANSIENT);committed=sqlite3_step(q)==SQLITE_DONE;}sqlite3_finalize(q);q=NULL;free(text);
    }
    if(committed)committed=sql(s,"COMMIT");else sql(s,"ROLLBACK");
    if(committed){durable=1;sj_value *event=sj_object();sj_add(event,"type",sj_string("queued"));srv_event_locked(s,id,event);sj_free(event);*result=job_locked(s,id,video,0);pthread_cond_broadcast(&s->changed);ok=*result!=NULL;}
    pthread_mutex_unlock(&s->mutex);if(!committed)goto storage;if(!ok)srv_fail(error,500,"response_error","request","job committed but response allocation failed; retry with the same idempotency key");goto done;
storage:atomic_store(&s->storage_failed,1);srv_fail(error,500,"storage_error","request","%s",*detail?detail:"unable to commit job");
done:
    if(reserved){pthread_mutex_lock(&s->mutex);s->reservations-=submission.variants;if(!durable)release_inputs(s,id);pthread_mutex_unlock(&s->mutex);}
    if (!durable && folder)
        srv_remove_tree(folder);
    for (int i = 0; i < 10; i++)
        h3_request_free(&requests[i]);
    free(folder);
    free(request_text);
    free(effective_text);
    h3_submission_free(&submission);
    return ok;
}
int srv_submit(srv *s,const sj_value *body,int video,const char *key,sj_value **result,srv_error *error){
    pthread_mutex_lock(&s->admission);int ok=submit_locked(s,body,video,key,result,error);pthread_mutex_unlock(&s->admission);
    if(!ok){char *safe=srv_redact(s,error->message);if(safe){snprintf(error->message,sizeof(error->message),"%s",safe);free(safe);}}
    return ok;
}
int srv_cancel(srv *s,const char *id,int deletion,srv_error *error){
    pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;int exists=0;
    if(sqlite3_prepare_v2(s->db,"SELECT deleted FROM jobs WHERE id=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,id,-1,SQLITE_TRANSIENT);exists=sqlite3_step(q)==SQLITE_ROW&&!sqlite3_column_int(q,0);}sqlite3_finalize(q);q=NULL;
    if(!exists){pthread_mutex_unlock(&s->mutex);return srv_fail(error,404,"not_found","job","job not found");}
    int ok=sql(s,"BEGIN IMMEDIATE");if(ok&&sqlite3_prepare_v2(s->db,"UPDATE jobs SET cancel=1,deleted=CASE WHEN ? THEN 1 ELSE deleted END WHERE id=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_int(q,1,deletion);sqlite3_bind_text(q,2,id,-1,SQLITE_TRANSIENT);ok=sqlite3_step(q)==SQLITE_DONE;}else ok=0;sqlite3_finalize(q);q=NULL;
    if(ok&&sqlite3_prepare_v2(s->db,"UPDATE variants SET status='cancelled',finished=? WHERE job=? AND status='queued'",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_double(q,1,srv_now());sqlite3_bind_text(q,2,id,-1,SQLITE_TRANSIENT);ok=sqlite3_step(q)==SQLITE_DONE;}else ok=0;sqlite3_finalize(q);
    if(ok){aggregate(s,id);ok=sql(s,"COMMIT");}else sql(s,"ROLLBACK");
    if(ok){sj_value *event=sj_object();sj_add(event,"type",sj_string(deletion?"deleted":"cancelling"));srv_event_locked(s,id,event);sj_free(event);pthread_cond_broadcast(&s->changed);}pthread_mutex_unlock(&s->mutex);
    return ok?1:srv_fail(error,500,"storage_error","job","could not commit cancellation");
}
static int cancellation(srv *s,const char *job){sqlite3_stmt *q=NULL;int cancel=0;
    if(sqlite3_prepare_v2(s->db,"SELECT cancel FROM jobs WHERE id=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,job,-1,SQLITE_TRANSIENT);if(sqlite3_step(q)==SQLITE_ROW)cancel=sqlite3_column_int(q,0);}sqlite3_finalize(q);return cancel;}
static void events_read(srv *s,const char *job,int variant,char *buffer,size_t *used,int fd){
    for(;;){ssize_t n=read(fd,buffer+*used,8191-*used);if(n<=0)break;*used+=(size_t)n;buffer[*used]=0;
        char *start=buffer,*end;while((end=strchr(start,'\n'))){char error[128];sj_value *event=sj_parse(start,(size_t)(end-start),error,sizeof(error));
            if(event&&event->type==LJ_OBJECT){sj_add(event,"variant",sj_int(variant));if(sj_field(event,"type")&&!strcmp(sj_field(event,"type"),"preview")){char url[180];snprintf(url,sizeof(url),"/v1/h3/jobs/%s/preview?variant=%d",job,variant);sj_add(event,"url",sj_string(url));}pthread_mutex_lock(&s->mutex);srv_event_locked(s,job,event);
                const char *phase=sj_field(event,"phase");int64_t done=0,total=0;sj_i64(sj_get(event,"completed"),&done);sj_i64(sj_get(event,"total"),&total);
                int progress=phase&&strstr(phase,"denoise")?10+(total>0?(int)(70.0*(double)done/(double)total):0):phase&&strstr(phase,"decode")?85:phase&&strstr(phase,"FFmpeg")?95:1;
                if (progress > 99)
                    progress = 99;
                sqlite3_stmt *q = NULL;
                if(sqlite3_prepare_v2(s->db,"UPDATE jobs SET progress=max(progress,(?*99+?)/n) WHERE id=? AND status='running'",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_int(q,1,variant);sqlite3_bind_int(q,2,progress);sqlite3_bind_text(q,3,job,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);}
            sj_free(event);start=end+1;}
        size_t remaining=*used-(size_t)(start-buffer);memmove(buffer,start,remaining);*used=remaining;if(*used==8191)*used=0;
    }
}
static void sanitize_log(srv *s,const char *work){
    char *path=srv_path(work,"worker.log");FILE *f=path?fopen(path,"rb"):NULL;char *text=malloc(SRV_LOG_LIMIT+1);size_t n=f&&text?fread(text,1,SRV_LOG_LIMIT,f):0;if(f)fclose(f);
    if(text){text[n]=0;char *safe=srv_redact(s,text);if(safe&&path){f=fopen(path,"wb");if(f){fwrite(safe,1,strlen(safe),f);fflush(f);fsync(fileno(f));fclose(f);}}free(safe);}free(text);free(path);
}
static sj_value *failure_manifest(const char *work,const char *status){
    char *log=srv_path(work,"worker.log");struct stat st;char hash[65];if(!log||stat(log,&st)||!srv_file_hash(log,hash)){free(log);return NULL;}free(log);
    sj_value *v=sj_object(),*files=sj_array(),*file=sj_object();sj_add(v,"schema",sj_int(1));sj_add(v,"kind",sj_string("diagnostic"));sj_add(v,"status",sj_string(status));sj_add(v,"outputs",sj_array());
    sj_add(file,"name",sj_string("worker.log"));sj_add(file,"mime",sj_string("text/plain"));sj_add(file,"sha256",sj_string(hash));sj_add(file,"bytes",sj_int(st.st_size));sj_add(files,NULL,file);sj_add(v,"artifacts",files);return v;
}
typedef struct {
    int launched,exited,status,cancelled,timed_out,quota;
    double seconds;
    struct rusage usage;
} execution;
static execution supervise(srv *s,const h3_request *request,const char *work,const char *job,int variant,
                            int mode,FILE *f,char *error,size_t size){
    sqlite3_stmt *q=NULL;
    pid_t pid=0;int event_fd=-1,control_fd=-1,log_fd=-1,status=0;struct rusage usage={0};
    int launched=srv_worker_launch(s,request,mode,work,&pid,&event_fd,&control_fd,&log_fd,error,size);
        pthread_mutex_lock(&s->mutex);s->active_pid=pid;
        q=NULL;if(pid&&sqlite3_prepare_v2(s->db,"UPDATE variants SET pid=? WHERE job=? AND idx=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_int(q,1,pid);sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,3,variant);sqlite3_step(q);}sqlite3_finalize(q);pthread_mutex_unlock(&s->mutex);
        double begin=srv_now(),stop_at=0,last_quota=0;int cancelled=0,timed_out=0,quota=0,exited=0;size_t log_bytes=0,event_bytes=0;char event_buffer[8192];
        while(launched&&!exited){
            char buffer[8192];ssize_t n;while((n=read(log_fd,buffer,sizeof(buffer)))>0){size_t keep=(size_t)n;if(keep>SRV_LOG_LIMIT-log_bytes)keep=SRV_LOG_LIMIT-log_bytes;if(keep&&fwrite(buffer,1,keep,f)!=keep){quota=1;}log_bytes+=keep;}
            events_read(s,job,variant,event_buffer,&event_bytes,event_fd);
            pid_t result=wait4(pid,&status,WNOHANG,&usage);if(result==pid){exited=1;break;}if(result<0&&errno!=EINTR)break;
            pthread_mutex_lock(&s->mutex);cancelled=cancellation(s,job);pthread_mutex_unlock(&s->mutex);
            timed_out=srv_now()-begin>(mode==2?s->config.model_timeout:s->config.timeout);
            if(srv_now()-last_quota>=2){last_quota=srv_now();if(srv_tree_bytes(s->config.state)>s->config.storage_limit)quota=1;}
            if(stop_at==0&&(cancelled||timed_out||quota||atomic_load(&s->stopping))){char c='c';ssize_t sent=write(control_fd,&c,1);(void)sent;stop_at=srv_now();}
            if(stop_at!=0&&srv_now()-stop_at>4)kill(-pid,SIGKILL);
            struct pollfd fds[2]={{log_fd,POLLIN,0},{event_fd,POLLIN,0}};poll(fds,2,50);
        }
        if(launched){if(!exited){kill(-pid,SIGKILL);while(wait4(pid,&status,0,&usage)<0&&errno==EINTR){}}else kill(-pid,SIGTERM);
            char buffer[8192];ssize_t n;while((n=read(log_fd,buffer,sizeof(buffer)))>0){size_t keep=(size_t)n;if(keep>SRV_LOG_LIMIT-log_bytes)keep=SRV_LOG_LIMIT-log_bytes;if(keep)fwrite(buffer,1,keep,f);log_bytes+=keep;}
            events_read(s,job,variant,event_buffer,&event_bytes,event_fd);close(event_fd);close(control_fd);close(log_fd);}

    return (execution){launched,exited,status,cancelled,timed_out,quota,srv_now()-begin,usage};
}
void *srv_scheduler(void *opaque){
    srv *s=opaque;
    while(!atomic_load(&s->stopping)){
        pthread_mutex_lock(&s->mutex);sqlite3_stmt *q=NULL;char job[SRV_ID]={0};int variant=0;char *encoded=NULL;
        if(sqlite3_prepare_v2(s->db,"SELECT v.job,v.idx,v.request FROM variants v JOIN jobs j ON v.job=j.id WHERE v.status='queued' AND j.status IN ('queued','running') AND j.cancel=0 AND j.deleted=0 ORDER BY j.created,j.id,v.idx LIMIT 1",-1,&q,NULL)==SQLITE_OK&&sqlite3_step(q)==SQLITE_ROW){snprintf(job,sizeof(job),"%s",sqlite3_column_text(q,0));variant=sqlite3_column_int(q,1);encoded=strdup((const char *)sqlite3_column_text(q,2));}sqlite3_finalize(q);q=NULL;
        if(!*job){struct timespec until;clock_gettime(CLOCK_REALTIME,&until);until.tv_sec++;pthread_cond_timedwait(&s->changed,&s->mutex,&until);pthread_mutex_unlock(&s->mutex);continue;}
        int claimed=sql(s,"BEGIN IMMEDIATE");
        if(claimed&&sqlite3_prepare_v2(s->db,"UPDATE variants SET status='preparing' WHERE job=? AND idx=?",-1,&q,NULL)==SQLITE_OK){
            sqlite3_bind_text(q,1,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,2,variant);claimed=sqlite3_step(q)==SQLITE_DONE;
        }else claimed=0;
        sqlite3_finalize(q);q=NULL;
        if(claimed)claimed=sql(s,"COMMIT");else sql(s,"ROLLBACK");
        snprintf(s->active_job,sizeof(s->active_job),"%s",job);s->active_variant=variant;pthread_mutex_unlock(&s->mutex);
        char error[SRV_ERROR]={0};h3_request request={0};sj_value *json=encoded?sj_parse(encoded,strlen(encoded),error,sizeof(error)):NULL;free(encoded);
        char rel[128];snprintf(rel,sizeof(rel),"jobs/%s/v%d/work",job,variant);char *work=srv_path(s->config.state,rel),*log=work?srv_path(work,"worker.log"):NULL;
        FILE *f=log?fopen(log,"wb"):NULL;free(log);
        execution run={0};int prepared=0;double preparation_seconds=0;
        int valid=claimed&&json&&work&&f&&h3_request_from_json(json,&request,error,sizeof(error));sj_free(json);
        if(valid){
            run=supervise(s,&request,work,job,variant,2,f,error,sizeof(error));preparation_seconds=run.seconds;
            prepared=run.launched&&run.exited&&WIFEXITED(run.status)&&!WEXITSTATUS(run.status)&&
                !run.cancelled&&!run.timed_out&&!run.quota&&!atomic_load(&s->stopping);
        }
        if(prepared){
            pthread_mutex_lock(&s->mutex);prepared=!cancellation(s,job)&&sql(s,"BEGIN IMMEDIATE");q=NULL;
            if(prepared&&sqlite3_prepare_v2(s->db,"UPDATE variants SET status='running',started=? WHERE job=? AND idx=?",-1,&q,NULL)==SQLITE_OK){
                sqlite3_bind_double(q,1,srv_now());sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,3,variant);prepared=sqlite3_step(q)==SQLITE_DONE;
            }else prepared=0;sqlite3_finalize(q);q=NULL;
            if(prepared&&sqlite3_prepare_v2(s->db,"UPDATE jobs SET status='running',started=coalesce(started,?) WHERE id=?",-1,&q,NULL)==SQLITE_OK){
                sqlite3_bind_double(q,1,srv_now());sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);prepared=sqlite3_step(q)==SQLITE_DONE;
            }else prepared=0;sqlite3_finalize(q);q=NULL;
            if(prepared)prepared=sql(s,"COMMIT");else sql(s,"ROLLBACK");pthread_mutex_unlock(&s->mutex);
            if(prepared)run=supervise(s,&request,work,job,variant,0,f,error,sizeof(error));
        }
        int launched=run.launched,exited=run.exited,status=run.status,cancelled=run.cancelled,timed_out=run.timed_out,quota=run.quota;
        struct rusage usage=run.usage;
        if(f){fflush(f);fsync(fileno(f));fclose(f);}if(work)sanitize_log(s,work);h3_request_free(&request);
        char *manifest=work?srv_path(work,"completion.json"):NULL;sj_value *result=manifest?srv_read_json(manifest,error,sizeof(error)):NULL;free(manifest);
        if(!result&&launched&&exited&&WIFEXITED(status)&&WEXITSTATUS(status))snprintf(error,sizeof(error),"worker exited with status %d; see worker.log artifact",WEXITSTATUS(status));
        else if(!result&&launched&&exited&&WIFSIGNALED(status))snprintf(error,sizeof(error),"worker terminated by signal %d; see worker.log artifact",WTERMSIG(status));
        const char *final=cancelled?"cancelled":atomic_load(&s->stopping)?"interrupted":launched&&exited&&WIFEXITED(status)&&WEXITSTATUS(status)==0&&result?"completed":"failed";
        if(timed_out){final="failed";snprintf(error,sizeof(error),prepared?"job exceeded its execution deadline":"model preparation exceeded its deadline");}if(quota){final="failed";snprintf(error,sizeof(error),"job exceeded the managed storage quota");}
        #ifdef H3_SERVER_TESTING
        if(result&&getenv("H3_SERVER_TEST_PUBLICATION_DELAY")){struct timespec delay={1,0};nanosleep(&delay,NULL);}
#endif
        if(result){
            sj_add(result,"model_preparation_seconds",sj_real(preparation_seconds));
            sj_add(result,"worker_build_sha256",sj_string(s->executable_sha));
#ifdef __APPLE__
            sj_add(result,"peak_rss_bytes",sj_uint((uint64_t)usage.ru_maxrss));
#else
            sj_add(result,"peak_rss_bytes",sj_uint((uint64_t)usage.ru_maxrss*1024));
#endif
            sj_add(result,"cpu_user_seconds",sj_real((double)usage.ru_utime.tv_sec+(double)usage.ru_utime.tv_usec/1e6));
            sj_add(result,"cpu_system_seconds",sj_real((double)usage.ru_stime.tv_sec+(double)usage.ru_stime.tv_usec/1e6));
        }
        if(!strcmp(final,"completed")&&!srv_prepare_publish(work,result,error,sizeof(error)))final="failed";
        if(!strcmp(final,"completed")&&srv_tree_bytes(s->config.state)>s->config.storage_limit){final="failed";snprintf(error,sizeof(error),"publication exceeds managed storage quota");}
        char *safe_error=srv_redact(s,error);if(safe_error){snprintf(error,sizeof(error),"%s",safe_error);free(safe_error);}
        sj_value *diagnostic=strcmp(final,"completed")&&work?failure_manifest(work,final):NULL;
        pthread_mutex_lock(&s->mutex);if(cancellation(s,job))final="cancelled";int transaction=sql(s,"BEGIN IMMEDIATE");
        if(transaction&&!strcmp(final,"completed")&&!srv_publish(s,job,variant,work,result,error,sizeof(error))){sql(s,"ROLLBACK");transaction=sql(s,"BEGIN IMMEDIATE");final="failed";}
        if(transaction&&strcmp(final,"completed")&&diagnostic){char ignored[SRV_ERROR];if(!srv_publish(s,job,variant,work,diagnostic,ignored,sizeof(ignored))){sql(s,"ROLLBACK");transaction=sql(s,"BEGIN IMMEDIATE");}}
        sj_free(diagnostic);
        char *metadata=result?sj_dump(result):NULL;q=NULL;
        if(transaction&&sqlite3_prepare_v2(s->db,"UPDATE variants SET status=?,finished=?,result=?,pid=NULL WHERE job=? AND idx=?",-1,&q,NULL)==SQLITE_OK){
            sqlite3_bind_text(q,1,final,-1,SQLITE_TRANSIENT);sqlite3_bind_double(q,2,srv_now());if(metadata&&!strcmp(final,"completed"))sqlite3_bind_text(q,3,metadata,-1,SQLITE_TRANSIENT);else sqlite3_bind_null(q,3);sqlite3_bind_text(q,4,job,-1,SQLITE_TRANSIENT);sqlite3_bind_int(q,5,variant);transaction=sqlite3_step(q)==SQLITE_DONE;}else transaction=0;sqlite3_finalize(q);q=NULL;free(metadata);
        if(strcmp(final,"completed")){
            if(sqlite3_prepare_v2(s->db,"UPDATE variants SET status='cancelled',finished=? WHERE job=? AND status='queued'",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_double(q,1,srv_now());sqlite3_bind_text(q,2,job,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);q=NULL;
            if(sqlite3_prepare_v2(s->db,"UPDATE jobs SET error=?,error_code=? WHERE id=?",-1,&q,NULL)==SQLITE_OK){sqlite3_bind_text(q,1,*error?error:final,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,2,timed_out?(prepared?"deadline_exceeded":"model_download_timeout"):quota?"storage_quota":final,-1,SQLITE_TRANSIENT);sqlite3_bind_text(q,3,job,-1,SQLITE_TRANSIENT);sqlite3_step(q);}sqlite3_finalize(q);
        }
        aggregate(s,job);if(transaction)sql(s,"COMMIT");else sql(s,"ROLLBACK");
        sj_value *event=sj_object();sj_add(event,"type",sj_string(final));sj_add(event,"variant",sj_int(variant));srv_event_locked(s,job,event);sj_free(event);
        s->active_pid=0;s->active_job[0]=0;pthread_mutex_unlock(&s->mutex);sj_free(result);free(work);
    }return NULL;
}
