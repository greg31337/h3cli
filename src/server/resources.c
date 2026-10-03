#include "src/server/internal.h"
#include "src/digest.h"
#include "src/platform.h"
#include <dirent.h>
#include <errno.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

/* File identities, not directory mtimes: installing an unrelated model group
 * must not invalidate a queued job. Native state fingerprints remain separate. */
static int identity(const char *path,char digest[65]) {
    struct stat st;
    if(stat(path,&st)||!S_ISREG(st.st_mode))return 0;
    struct timespec mt=h3_stat_mtime(&st),ct=h3_stat_ctime(&st);
    char stamp[256];
    int n=snprintf(stamp,sizeof(stamp),"%llu:%llu:%llu:%lld:%ld:%lld:%ld:%u\n",
        (unsigned long long)st.st_dev,(unsigned long long)st.st_ino,(unsigned long long)st.st_size,
        (long long)mt.tv_sec,mt.tv_nsec,(long long)ct.tv_sec,ct.tv_nsec,(unsigned)st.st_mode);
    h3_sha256_ctx hash;h3_sha256_init_fast(&hash);h3_sha256_update(&hash,stamp,(h3_sha256_size)n);
    uint8_t raw[32];h3_sha256_final(raw,&hash);
    for(int i=0;i<32;i++)snprintf(digest+2*i,3,"%02x",raw[i]);
    return 1;
}
static int resource(sj_value *a,const char *path,char *error,size_t size) {
    for(size_t i=0;i<a->count;i++)if(!strcmp(sj_field(a->items[i],"path"),path))return 1;
    struct stat st;
    if(stat(path,&st)){if(errno==ENOENT)return 1;snprintf(error,size,"model resource inaccessible");return 0;}
    if(!S_ISREG(st.st_mode)){snprintf(error,size,"model resource is not a regular file");return 0;}
    char hash[65];if(!identity(path,hash))return 0;
    sj_value *v=sj_object();sj_add(v,"path",sj_string(path));sj_add(v,"metadata_sha256",sj_string(hash));return sj_add(a,NULL,v);
}
static int tree(sj_value *a,const char *path,int depth,char *error,size_t size) {
    struct stat st;
    if(stat(path,&st))return errno==ENOENT;
    if(S_ISREG(st.st_mode))return resource(a,path,error,size);
    if(!S_ISDIR(st.st_mode)||depth>32||a->count>100000)return 0;
    struct dirent **entries=NULL;int n=scandir(path,&entries,NULL,alphasort),ok=n>=0;
    for(int i=0;i<n;i++){
        if(ok&&strcmp(entries[i]->d_name,".")&&strcmp(entries[i]->d_name,"..")){
            char *child=srv_path(path,entries[i]->d_name);ok=child&&tree(a,child,depth+1,error,size);free(child);
        }
        free(entries[i]);
    }
    free(entries);return ok;
}
static int snapshot(const h3_request *r,const char *destination,int persist_plan,char *error,size_t size) {
    h3_model_plan p={0};
    if(!h3_models_resolve(r,&p,error,size))return 0;
    sj_value *a=sj_array(),*dependencies=sj_array();int ok=a&&dependencies;
    unsigned groups=0;
    for(size_t i=0;ok&&i<p.count;i++){
        groups|=p.items[i].artifact->groups;
        ok=resource(a,p.items[i].destination,error,size);
        sj_value *v=sj_object();sj_add(v,"path",sj_string(p.items[i].destination));
        sj_add(v,"sha256",sj_string(p.items[i].artifact->sha256));sj_add(v,"bytes",sj_uint(p.items[i].artifact->bytes));sj_add(dependencies,NULL,v);
    }
    /* Preserve identities of custom selected checkpoints and their metadata,
     * including additional files used by native saved-state validation. */
    if(strcmp(h3_request_operation(r),"decode_av"))for(unsigned bit=1;ok&&bit<=2;bit<<=1)if(groups&bit){
        char *dir=srv_path(p.paths.main,bit==1?"FL2VA":"Ref2VA");ok=dir&&tree(a,dir,0,error,size);free(dir);
    }
    for(size_t i=0;ok&&i<r->count;i++)if(!strcmp(r->values[i].option->name,"lora")){
        char *path=strdup(r->values[i].value);if(!path){ok=0;break;}
        struct stat st;if(stat(path,&st)){char *colon=strrchr(path,':');if(colon)*colon=0;}
        ok=tree(a,path,0,error,size);free(path);
    }
    if(ok)ok=srv_write_json(destination,a,error,size);
    if(ok&&persist_plan){
        sj_value *plan=sj_object();sj_add(plan,"catalog_sha256",sj_string(h3_model_catalog_sha256));
        sj_add(plan,"files",dependencies);dependencies=NULL;
        char *name=NULL;if(asprintf(&name,"%s.plan",destination)<0)name=NULL;
        ok=name&&srv_write_json(name,plan,error,size);free(name);sj_free(plan);
    }
    h3_models_plan_free(&p);sj_free(a);sj_free(dependencies);return ok;
}
int srv_resources_save(srv *s,const h3_request *r,const char *job,char *error,size_t size) {
    char relative[128];snprintf(relative,sizeof(relative),"jobs/%s/resources.json",job);
    char *path=srv_path(s->config.state,relative);int ok=path&&snapshot(r,path,1,error,size);free(path);return ok;
}
int srv_resources_check(const char *work,char *error,size_t size) {
    char *path=srv_path(work,"../../resources.json");sj_value *a=path?srv_read_json(path,error,size):NULL;free(path);int ok=a&&a->type==LJ_ARRAY;
    for(size_t i=0;ok&&i<a->count;i++){
        const char *file=sj_field(a->items[i],"path"),*expected=sj_field(a->items[i],"metadata_sha256");char hash[65];
        ok=file&&expected&&identity(file,hash)&&!strcmp(expected,hash);
    }
    sj_free(a);if(!ok)snprintf(error,size,"configured model/component/LoRA metadata changed after admission; submit a fresh job");return ok;
}
int srv_resources_refresh(const h3_request *r,const char *work,char *error,size_t size) {
    if(!srv_resources_check(work,error,size))return 0;
    char *path=srv_path(work,"../../resources.json");int ok=path&&snapshot(r,path,0,error,size);free(path);
    if(ok){
        sj_value *ready=sj_object();
        sj_add(ready,"catalog_sha256",sj_string(h3_model_catalog_sha256));
        sj_add(ready,"ready",sj_bool(1));
        path=srv_path(work,"models-ready.json");
        ok=path&&srv_write_json(path,ready,error,size);free(path);sj_free(ready);
    }
    return ok;
}
