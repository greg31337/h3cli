#ifndef H3_SERVER_INTERNAL_H
#define H3_SERVER_INTERNAL_H
#include "src/server/server.h"
#include "src/server/sglang.h"
#include "src/models/models.h"
#include "third_party/civetweb/include/civetweb.h"
#include <pthread.h>
#include <sqlite3.h>
#include <stdatomic.h>
#include <sys/types.h>
#define SRV_ID 48
#define SRV_ERROR 1024
#define SRV_JSON_LIMIT (1024u*1024u)
#define SRV_EVENT_LIMIT 512u
#define SRV_LOG_LIMIT (2u*1024u*1024u)
typedef struct {
    char *host,*state,*model,*model_id,*executable,*source_root,*api_key,*models_root;
    char **read_roots;size_t read_root_count;
    int port,queue_limit,timeout,allow_urls,offline,model_timeout;
    uint64_t upload_limit,storage_limit;
} srv_config;
typedef struct {
    srv_config config;
    sqlite3 *db;
    pthread_mutex_t mutex,admission;
    pthread_cond_t changed;
    pthread_t scheduler;
    struct mg_context *http;
    atomic_int stopping,sse_clients,storage_failed;
    int lock_fd, reservations;
    pid_t active_pid;
    char active_job[SRV_ID];
    int active_variant;
    char executable_sha[65];
    uint64_t storage_reserved;
} srv;
typedef struct {int status;char code[64],field[128],message[SRV_ERROR];} srv_error;
char *srv_redact(srv *s,const char *text);
int srv_fail(srv_error *e,int status,const char *code,const char *field,const char *format,...);
double srv_now(void);
int srv_id(char out[SRV_ID],const char *prefix);
char *srv_path(const char *a,const char *b);
int srv_mkdirs(const char *path,char *error,size_t size);
int srv_write_json(const char *path,const sj_value *value,char *error,size_t size);
sj_value *srv_read_json(const char *path,char *error,size_t size);
int srv_file_hash(const char *path,char out[65]);
int srv_text_hash(const char *text,char out[65]);
int srv_remove_tree(const char *path);
uint64_t srv_tree_bytes(const char *path);
int srv_probe(const char *path,sj_value **out,char *error,size_t size);
int srv_capture(char *const argv[],char **text,size_t limit,int timeout,char *error,size_t size);
int srv_db_open(srv *s,char *error,size_t size);
int srv_db_exec(srv *s,const char *sql,char *error,size_t size);
sj_value *srv_job(srv *s,const char *id,int video,int include_deleted);
sj_value *srv_jobs(srv *s,int video,const char *after,int limit,int ascending);
int srv_submit(srv *s,const sj_value *body,int video,const char *key,sj_value **result,srv_error *error);
int srv_cancel(srv *s,const char *id,int deletion,srv_error *error);
void *srv_scheduler(void *opaque);
void srv_event_locked(srv *s,const char *job,const sj_value *event);
sj_value *srv_events(srv *s,const char *job,int64_t after);
int srv_assets_prepare(srv *s,h3_submission *submission,const char *job,char *error,size_t size);
int srv_asset_upload(srv *s,const char *temp,const char *name,sj_value **result,srv_error *error);
sj_value *srv_asset_get(srv *s,const char *id);
int srv_asset_delete(srv *s,const char *id,srv_error *error);
int srv_resources_save(srv *s,const h3_request *r,const char *job,char *error,size_t size);
int srv_resources_refresh(const h3_request *r,const char *work,char *error,size_t size);
int srv_resources_check(const char *work,char *error,size_t size);
int srv_bundle_write(const char *state,const char *output,char *error,size_t size);
int srv_bundle_unpack(const char *input,const char *output,uint64_t limit,char *error,size_t size);
int srv_prepare_publish(const char *work,sj_value *result,char *error,size_t size);
void srv_gc(srv *s,int startup);
int srv_publish(srv *s,const char *job,int variant,const char *work,const sj_value *result,char *error,size_t size);
int srv_variant_request(srv *s,const char *job,int variant,const h3_submission *submission,h3_request *request,char *error,size_t size);
int srv_worker_launch(srv *s,const h3_request *request,int validate,const char *work,pid_t *pid,int *events,int *control,int *logs,char *error,size_t size);
int srv_worker_validate(srv *s,const h3_request *request,const char *work,char *error,size_t size);
int srv_http_start(srv *s,char *error,size_t size);
void srv_http_stop(srv *s);
#endif
