/* Native runtime LoRA folding. Recipe 1: ascending rank FP32 products/adds,
 * ordered adapters, one BF16 ties-even rounding. Never compile with fast-math. */
#include "src/weights/lora.h"
#include "src/weights/lora_json.h"
#include "src/weights/safetensors.h"
#include "src/digest.h"
#include "src/memory.h"
#include "src/platform.h"
#include <dirent.h>
#include <errno.h>
#include <fcntl.h>
#include <float.h>
#include <fenv.h>
#include <limits.h>
#include <locale.h>
#ifdef __APPLE__
#include <xlocale.h>
#endif
#include <math.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <sys/statvfs.h>
#include <time.h>
#include <unistd.h>
#ifdef __APPLE__
#include <sys/clonefile.h>
#else
#include <sys/ioctl.h>
#include <linux/fs.h>
#endif
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#if defined(__clang__)
/* Entry explicitly establishes round-to-nearest and gradual underflow.
 * The kernel may vectorize columns; each lane keeps the prescribed K order. */
#pragma STDC FENV_ACCESS OFF
#pragma STDC FP_CONTRACT OFF
#endif
#define IDENTITY "h3_lora_manifest.json"
#define RECIPE "h3-runtime-lora/schema=1/mapping=1/recipe=1"
#define IO_SIZE 65536u
#define MIN(a,b) ((a)<(b)?(a):(b))

typedef struct { char *path; double scale; } adapter_arg;
struct h3_lora_selection { adapter_arg *args; size_t count, budget; char *cache; };
typedef struct {
    char *name, *path;
    uint64_t size;
    unsigned char digest[32];
    struct stat stamp;
    h3_st_header st;
    int cloned;
} source_file;
typedef struct {
    char *module;
    size_t adapter, file;
    const h3_st_tensor *base, *a, *b, *alpha_tensor;
    uint64_t start, stop, rank;
    float scale;
    double alpha;
    int explicit_alpha;
} target;
typedef struct {
    const h3_lora_selection *selection;
    source_file *files, *adapters;
    size_t file_count, target_count;
    target *targets;
    lj_value *config;
    h3_progress_callback progress;
    void *opaque;
    char *error; size_t error_size;
    size_t scratch_peak, plan_bytes;
    int cancelled;
    double last_check;
    const char *last_phase;
    double hash_seconds, copy_seconds, fold_seconds, validation_seconds;
    size_t cloned_files, copied_files;
} work;
static int fail(work *w,const char *fmt,...) {
    va_list ap;va_start(ap,fmt);
    if(w->error&&w->error_size)vsnprintf(w->error,w->error_size,fmt,ap);
    va_end(ap);return 0;
}
static double now(void) {struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}
static int checkpoint(work *w,const char *phase,int done,int total) {
    double stamp=now();
    if(!total&&w->last_phase&&!strcmp(w->last_phase,phase)&&stamp-w->last_check<0.1)return !w->cancelled;
    w->last_phase=phase;w->last_check=stamp;
    int cancelled=w->progress?w->progress(phase,done,total,w->opaque):0;
    int ok=h3_memory_checkpoint(cancelled,phase,w->error,w->error_size);
    if(!ok)w->cancelled=1;
    return ok;
}
static char *join(const char *a,const char *b) {
    size_t n=strlen(a),m=strlen(b);if(n>SIZE_MAX-m-2)return NULL;
    char *p=malloc(n+m+2);if(p)snprintf(p,n+m+2,"%s/%s",a,b);return p;
}
static int suffix(const char *s,const char *end) {size_t n=strlen(s),m=strlen(end);return n>=m&&!strcmp(s+n-m,end);}
static void hex(const unsigned char *d,char out[65]) {for(int i=0;i<32;i++)snprintf(out+2*i,3,"%02x",d[i]);}
static void hash_u64(h3_sha256_ctx *h,uint64_t n) {unsigned char b[8];for(int i=0;i<8;i++)b[i]=(unsigned char)(n>>(8*i));h3_sha256_update(h,b,8);}
static void hash_string(h3_sha256_ctx *h,const char *s) {hash_u64(h,strlen(s));h3_sha256_update(h,s,(h3_sha256_size)strlen(s));}
static int same_stamp(const struct stat *a,const struct stat *b) {
    struct timespec am=h3_stat_mtime(a),bm=h3_stat_mtime(b),ac=h3_stat_ctime(a),bc=h3_stat_ctime(b);
    return a->st_dev==b->st_dev&&a->st_ino==b->st_ino&&a->st_size==b->st_size&&
        am.tv_sec==bm.tv_sec&&am.tv_nsec==bm.tv_nsec&&ac.tv_sec==bc.tv_sec&&ac.tv_nsec==bc.tv_nsec;
}
static int io_read(work *w,int fd,void *buf,size_t n,uint64_t off) {
    unsigned char *p=buf;
    while(n){ssize_t r=pread(fd,p,n,(off_t)off);if(r<0&&errno==EINTR)continue;
        if(r<=0)return fail(w,"LoRA read failed: %s",r?strerror(errno):"truncated file");
        p+=r;n-=(size_t)r;off+=(uint64_t)r;}return 1;
}
static int injected(work *w,const char *name) {
#ifdef H3_LORA_TESTING
    const char *fault=getenv("H3_TEST_LORA_FAULT");
    if(fault&&!strcmp(fault,name)){errno=EIO;return fail(w,"injected LoRA %s failure",name);}
#else
    (void)w;(void)name;
#endif
    return 1;
}
static int io_write(work *w,int fd,const void *buf,size_t n,uint64_t off) {
    if(!injected(w,"write")||!injected(w,"enospc"))return 0;
    const unsigned char *p=buf;
    while(n){size_t amount=n;
#ifdef H3_LORA_TESTING
        if(getenv("H3_TEST_LORA_SHORT_WRITE")&&amount>7)amount=7;
#endif
        ssize_t r=pwrite(fd,p,amount,(off_t)off);if(r<0&&errno==EINTR)continue;
        if(r<=0)return fail(w,"LoRA write failed: %s",strerror(errno));
        p+=r;n-=(size_t)r;off+=(uint64_t)r;}return 1;
}
static int hash_file(work *w,source_file *f,int recheck) {
    if(!checkpoint(w,"LoRA hashing",0,0))return 0;
    int fd=open(f->path,O_RDONLY);struct stat a={0},b={0},pathstat={0};
    if(fd<0)return fail(w,"%s: %s",f->path,strerror(errno));
    int ok=fstat(fd,&a)==0&&S_ISREG(a.st_mode)&&a.st_size>=0;
    if(recheck&&!same_stamp(&a,&f->stamp))ok=0;
    h3_sha256_ctx h;h3_sha256_init_fast(&h);unsigned char buf[IO_SIZE],digest[32];
    for(uint64_t off=0;ok&&off<(uint64_t)a.st_size;){
        size_t n=(size_t)MIN(IO_SIZE,(uint64_t)a.st_size-off);
        ok=io_read(w,fd,buf,n,off);if(ok)h3_sha256_update(&h,buf,(h3_sha256_size)n);off+=n;
        if(!(off%(16*1024*1024)))ok=ok&&checkpoint(w,"LoRA hashing",0,0);
    }
    h3_sha256_final(digest,&h);
    ok=ok&&fstat(fd,&b)==0&&stat(f->path,&pathstat)==0&&same_stamp(&a,&b)&&same_stamp(&a,&pathstat);
    close(fd);
    if(!ok){if(w->cancelled)return 0;return fail(w,"LoRA source unreadable or changed: %s",f->path);}
    if(recheck&&memcmp(digest,f->digest,32))return fail(w,"LoRA source content changed: %s",f->path);
    if(!recheck){f->stamp=a;f->size=(uint64_t)a.st_size;memcpy(f->digest,digest,32);}return 1;
}
static void file_free(source_file *f) {free(f->name);free(f->path);h3_st_free_header(&f->st);}
static void work_free(work *w) {
    for(size_t i=0;i<w->file_count;i++)file_free(&w->files[i]);
    free(w->files);
    if(w->adapters)for(size_t i=0;i<w->selection->count;i++)file_free(&w->adapters[i]);
    free(w->adapters);
    for(size_t i=0;i<w->target_count;i++)free(w->targets[i].module);
    free(w->targets);lj_free(w->config);
}
void h3_lora_selection_free(h3_lora_selection *s) {
    if(!s)return;
    if(s->args)for(size_t i=0;i<s->count;i++)free(s->args[i].path);
    free(s->args);free(s->cache);free(s);
}
int h3_lora_selection_create(const h3_lora_options *o,h3_lora_selection **out,char *error,size_t cap) {
    work w={.error=error,.error_size=cap};if(error&&cap)error[0]=0;
    if(!out)return fail(&w,"missing LoRA selection output");
    *out=NULL;
    if(!o||!o->count){if(o&&(o->cache_dir||o->memory_mib))return fail(&w,"LoRA cache/memory requires --lora");return 1;}
    if(!o->adapters||o->count>SIZE_MAX/sizeof(adapter_arg))return fail(&w,"invalid LoRA selection");
    h3_lora_selection *s=calloc(1,sizeof(*s));if(!s)return fail(&w,"out of memory");
    s->count=o->count;
    if(!injected(&w,"selection-allocation")){h3_lora_selection_free(s);return 0;}
    s->args=calloc(s->count,sizeof(*s->args));
    size_t mib=o->memory_mib?o->memory_mib:512;
    if(mib>SIZE_MAX/(1024*1024)||!s->args)goto bad;
    s->budget=mib*1024*1024;
    for(size_t i=0;i<s->count;i++){
        const char *arg=o->adapters[i];if(!arg||!*arg)goto bad;
        char *path=strdup(arg);if(!path)goto bad;
        struct stat st;double scale=1;
        if(stat(path,&st)||!S_ISREG(st.st_mode)){
            char *sep=strrchr(path,':');
            if(sep){*sep++=0;if(!lj_number(sep,&scale)){free(path);goto bad;}}
        }
        if(!*path||!isfinite(scale)||fabs(scale)>FLT_MAX){free(path);goto bad;}
        s->args[i].path=realpath(path,NULL);free(path);
        if(!s->args[i].path||stat(s->args[i].path,&st)||!S_ISREG(st.st_mode)||access(s->args[i].path,R_OK))goto bad;
        /* A caller's flush mode must not turn a binary64 subnormal into the
         * canonical zero selection. Normalize signed zero by its bits. */
        uint64_t scale_bits;memcpy(&scale_bits,&scale,sizeof(scale_bits));
        s->args[i].scale=(scale_bits<<1)==0?0:scale;
    }
    if(o->cache_dir){if(!*o->cache_dir)goto bad;s->cache=strdup(o->cache_dir);}
    else {
        const char *xdg=getenv("XDG_CACHE_HOME"),*home=getenv("HOME");
        if(xdg&&*xdg=='/')s->cache=join(xdg,"h3/lora");
        else if(home&&*home=='/')s->cache=join(home,".cache/h3/lora");
        else {fail(&w,"LoRA requires an absolute HOME/XDG_CACHE_HOME or --lora-cache");goto cleanup;}
    }
    if(!s->cache)goto bad;
    if(*s->cache!='/'){char *cwd=getcwd(NULL,0);char *p=cwd?join(cwd,s->cache):NULL;free(cwd);free(s->cache);s->cache=p;if(!p)goto bad;}
    *out=s;return 1;
bad:fail(&w,"invalid LoRA arguments: readable PATH[:finite SCALE], nonempty cache, positive memory required");
cleanup:h3_lora_selection_free(s);return 0;
}
void h3_lora_selection_print(const h3_lora_selection *s,FILE *out) {
    if(!s)return;
    fprintf(out,"Runtime LoRA: %zu adapter(s), %zu MiB scratch, cache %s\n",s->count,s->budget/(1024*1024),s->cache);
    for(size_t i=0;i<s->count;i++)fprintf(out,"  [%zu] %s : %.17g\n",i+1,s->args[i].path,s->args[i].scale);
}
void h3_lora_variant_free(h3_lora_variant *v) {if(!v)return;if(v->transformer&&v->lease>=0)close(v->lease);free(v->transformer);memset(v,0,sizeof(*v));v->lease=-1;}

static int reserve_plan(work *w,size_t bytes) {
    if(bytes>w->selection->budget || w->plan_bytes>w->selection->budget-bytes ||
       w->selection->budget-w->plan_bytes-bytes<524288)
        return fail(w,"LoRA memory budget too small for plan and bounded arithmetic; increase --lora-memory-mib");
    w->plan_bytes+=bytes;return 1;
}
static int read_header(work *w,source_file *f) {
    unsigned char prefix[8];int fd=open(f->path,O_RDONLY);
    if(fd<0)return fail(w,"cannot open safetensors header");
    int ok=io_read(w,fd,prefix,8,0);close(fd);if(!ok)return 0;
    uint64_t n=0;for(int i=0;i<8;i++)n|=(uint64_t)prefix[i]<<(8*i);
    if(n>SIZE_MAX/16||!reserve_plan(w,(size_t)n*16))return fail(w,"LoRA header exceeds memory budget");
    return h3_st_read_header(f->path,&f->st,w->error,w->error_size);
}
static int entry_order(const struct dirent **a,const struct dirent **b) {return strcmp((*a)->d_name,(*b)->d_name);}
static int inventory(work *w,const char *dir) {
    /* Bound scandir's temporary storage before it allocates a whole listing. */
    DIR *pre=opendir(dir);struct dirent *pre_entry;
    if(!pre)return fail(w,"cannot open transformer directory: %s",strerror(errno));
    while((pre_entry=readdir(pre)))if(strcmp(pre_entry->d_name,".")&&strcmp(pre_entry->d_name,"..")) {
        if(!reserve_plan(w,512)){closedir(pre);return 0;}
    }
    closedir(pre);
    struct dirent **entries=NULL;int n=scandir(dir,&entries,NULL,entry_order);if(n<0)return fail(w,"%s: %s",dir,strerror(errno));
    int ok=1;size_t shards=0;
    for(int i=0;i<n;i++){
        const char *name=entries[i]->d_name;
        if(!strcmp(name,".")||!strcmp(name,"..")){free(entries[i]);continue;}
        if(ok){
            if(!reserve_plan(w,1024+2*strlen(name)+2*strlen(dir))){ok=0;free(entries[i]);continue;}
            source_file *a=realloc(w->files,(w->file_count+1)*sizeof(*a));
            if(!a)ok=fail(w,"out of memory indexing transformer");
            else {
                w->files=a;source_file *f=&a[w->file_count++];memset(f,0,sizeof(*f));
                f->name=strdup(name);f->path=join(dir,name);
                ok=f->name&&f->path&&hash_file(w,f,0);
                if(ok&&suffix(name,".safetensors")){
                    ok=read_header(w,f);shards++;
                }
                if(ok&&!strcmp(name,"config.json")){
                    if(f->size>SIZE_MAX/16||!reserve_plan(w,(size_t)f->size*16)){ok=0;free(entries[i]);continue;}
                    w->config=lj_file(f->path,w->error,w->error_size);
                    ok=w->config&&w->config->type==LJ_OBJECT;
                }
            }
        }
        free(entries[i]);
    }free(entries);
    if(!ok)return 0;
    if(!shards||!w->config)return fail(w,"transformer requires config.json and safetensors shards");
    for(size_t i=0;i<w->file_count;i++)if(!strcmp(w->files[i].name,IDENTITY))
        fprintf(stderr,"LoRA: input is already folded; parent identity is included in the source inventory and cache key\n");
    size_t tensors=0;
    for(size_t i=0;i<w->file_count;i++)for(size_t j=0;j<w->files[i].st.tensor_count;j++){
        const char *name=w->files[i].st.tensors[j].name;tensors++;
        for(size_t k=0;k<i;k++)if(h3_st_find(&w->files[k].st,name))return fail(w,"duplicate base tensor: %s",name);
    }
    for(size_t i=0;i<w->file_count;i++)if(suffix(w->files[i].name,".safetensors.index.json")){
        if(w->files[i].size>SIZE_MAX/16||!reserve_plan(w,(size_t)w->files[i].size*16))return 0;
        lj_value *index=lj_file(w->files[i].path,w->error,w->error_size);const lj_value *map=lj_get(index,"weight_map");
        ok=map&&map->type==LJ_OBJECT&&map->count==tensors;
        for(size_t j=0;ok&&j<map->count;j++){
            const lj_value *item=map->items[j];int found=0;
            if(item->type==LJ_STRING)for(size_t k=0;k<w->file_count;k++)
                if(!strcmp(item->text,w->files[k].name)&&h3_st_find(&w->files[k].st,item->key))found++;
            ok=found==1;
        }
        lj_free(index);if(!ok)return fail(w,"transformer index does not exactly match shard tensors");
    }
    return 1;
}
static const h3_st_tensor *find_base(work *w,const char *name,size_t *file) {
    for(size_t i=0;i<w->file_count;i++){
        const h3_st_tensor *t=h3_st_find(&w->files[i].st,name);if(t){*file=i;return t;}
    }return NULL;
}
static int architecture(const char *s) {
    const char *words[]={"pdd","vdn","controlnet","dora","lora_magnitude_vector","output_heads","diffusion_head"};
    for(size_t i=0;i<sizeof(words)/sizeof(*words);i++){
        size_t n=strlen(words[i]);for(const char *p=s;*p;p++)
            if((p==s||strchr("._ /-",p[-1]))&&!strncasecmp(p,words[i],n)&&(!p[n]||strchr("._ /-",p[n])))return 1;
    }return 0;
}
static float decode(const unsigned char *p,h3_dtype dtype) {
    uint32_t bits=(uint32_t)p[0]|((uint32_t)p[1]<<8);float v;
    if(dtype==H3_DTYPE_F32)bits|=((uint32_t)p[2]<<16)|((uint32_t)p[3]<<24);
    else if(dtype==H3_DTYPE_BF16)bits<<=16;
    else {
        unsigned sign=bits>>15,exp=(bits>>10)&31,mant=bits&1023;
        if(!exp)return (sign?-1.0f:1.0f)*ldexpf((float)mant,-24);
        bits=(sign<<31)|(exp==31?0x7f800000u:((exp+112)<<23))|(mant<<13);
    }
    memcpy(&v,&bits,4);return v;
}
static int floats(h3_dtype d) {return d==H3_DTYPE_F32||d==H3_DTYPE_F16||d==H3_DTYPE_BF16;}
static int read_floats(work *w,int fd,const h3_st_tensor *t,uint64_t index,size_t n,float *out,unsigned char *buf) {
    size_t size=h3_dtype_size(t->dtype);
    if(!io_read(w,fd,buf,n*size,t->file_offset+index*size))return 0;
    for(size_t i=0;i<n;i++){
        out[i]=decode(buf+i*size,t->dtype);if(!isfinite(out[i]))return fail(w,"non-finite LoRA/base tensor: %s",t->name);
    }return 1;
}
static int numeric(const lj_value *v,double *n) {
    if(!v)return 0;
    if(v->type==LJ_NUMBER){*n=v->number;return 1;}
    return v->type==LJ_STRING&&lj_number(v->text,n);
}
static int meta_equal(const lj_value *a,const lj_value *b) {
    double x,y;if(numeric(a,&x)&&numeric(b,&y))return x==y;
    if(a->type==LJ_STRING&&b->type==LJ_STRING)return !strcmp(a->text,b->text);
    if(a->type==b->type&&(a->type==LJ_NULL||a->type==LJ_BOOL))return a->number==b->number;
    if(a->type==LJ_STRING&&b->type==LJ_BOOL)return !strcmp(a->text,b->number!=0?"true":"false")||!strcmp(a->text,b->number!=0?"True":"False");
    return 0;
}
static lj_value *metadata(work *w,const h3_st_header *st) {
    lj_value *root=calloc(1,sizeof(*root));if(!root)return NULL;root->type=LJ_OBJECT;
    for(size_t i=0;i<st->metadata_count;i++){
        lj_value *v=calloc(1,sizeof(*v));if(!v)goto bad;
        v->type=LJ_STRING;v->key=strdup(st->metadata_keys[i]);v->text=strdup(st->metadata_values[i]);
        lj_value **a=realloc(root->items,(root->count+1)*sizeof(*a));
        if(!a||!v->key||!v->text){if(a)root->items=a;lj_free(v);goto bad;}root->items=a;a[root->count++]=v;
    }
    const char *nested[]={"adapter_config","ss_network_args"};
    for(size_t i=0;i<2;i++){
        const lj_value *v=lj_get(root,nested[i]);if(!v)continue;
        lj_value *m=v->type==LJ_STRING?lj_parse(v->text,strlen(v->text),w->error,w->error_size):NULL;
        if(!m||m->type!=LJ_OBJECT){lj_free(m);goto bad;}
        for(size_t j=0;j<m->count;j++){
            const lj_value *old=lj_get(root,m->items[j]->key);
            if(old){if(!meta_equal(old,m->items[j])){lj_free(m);goto bad;}continue;}
            lj_value **a=realloc(root->items,(root->count+1)*sizeof(*a));if(!a){lj_free(m);goto bad;}
            root->items=a;a[root->count++]=m->items[j];m->items[j]=NULL;
        }lj_free(m);
    }
    const char *fields[]={"format","architecture","adapter_type","peft_type","ss_network_module","method","algorithm"};
    for(size_t i=0;i<root->count;i++)if(strcmp(root->items[i]->key,"use_dora")&&architecture(root->items[i]->key))goto bad;
    for(size_t i=0;i<sizeof(fields)/sizeof(*fields);i++){
        const lj_value *v=lj_get(root,fields[i]);if(v&&v->type==LJ_STRING&&architecture(v->text))goto bad;
    }
    const char *flags[]={"use_rslora","use_dora","fan_in_fan_out"};
    for(size_t i=0;i<3;i++){
        const lj_value *v=lj_get(root,flags[i]);if(!v||v->type==LJ_NULL||(v->type==LJ_BOOL&&v->number==0))continue;
        if(v->type==LJ_STRING&&(!strcmp(v->text,"false")||!strcmp(v->text,"False")||!strcmp(v->text,"0")))continue;
        goto bad;
    }
    const lj_value *peft=lj_get(root,"peft_type");
    if((peft&&(peft->type!=LJ_STRING||strcmp(peft->text,"LORA")))||lj_get(root,"rank_pattern")||lj_get(root,"alpha_pattern"))goto bad;
    return root;
bad:lj_free(root);fail(w,"unsupported architecture or inconsistent adapter metadata");return NULL;
}
static int normalize(work *w,target *t) {
    const char *m=t->module;
    const char *prefix[]={"base_model.model.","model.diffusion_model.","diffusion_model.","transformer."};
    for(size_t i=0;i<4;i++)if(!strncmp(m,prefix[i],strlen(prefix[i])))m+=strlen(prefix[i]);
    char *name=NULL;
    if(!strncmp(m,"lora_unet_",10)){
        int matches=0;
        for(size_t i=0;i<w->file_count;i++)for(size_t j=0;j<w->files[i].st.tensor_count;j++){
            const char *n=w->files[i].st.tensors[j].name;if(!suffix(n,".weight"))continue;
            char *flat=strndup(n,strlen(n)-7);if(!flat)return fail(w,"out of memory");
            for(char *p=flat;*p;p++)if(*p=='.')*p='_';
            if(!strcmp(flat,m+10)){matches++;free(name);name=strdup(n);}free(flat);
        }
        if(matches!=1){free(name);return fail(w,"unknown/ambiguous ComfyUI target: %s",m);}
    } else {
        const char *tail=m;char *module=NULL;
        if(!strncmp(m,"transformer_blocks.",19)){
            module=malloc(strlen(m)+1);if(module)snprintf(module,strlen(m)+1,"blocks.%s",m+19);tail=module;
        }
        if(!tail)return fail(w,"out of memory");
        name=malloc(strlen(tail)+32);if(!name){free(module);return fail(w,"out of memory");}
        strcpy(name,tail);free(module);
        int slice=-1;
        if(suffix(name,".attn.to_q"))slice=0;
        if(suffix(name,".attn.to_k"))slice=1;
        if(suffix(name,".attn.to_v"))slice=2;
        if(slice>=0){
            const char *b=!strncmp(name,"token_refiner.",14)?name+14:name;
            if(strncmp(b,"blocks.",7)){free(name);return fail(w,"invalid split QKV target");}
            b+=7;if(*b<'0'||*b>'9'){free(name);return fail(w,"invalid QKV block number");}
            while(*b>='0'&&*b<='9')b++;
            if(strncmp(b,".attn.to_",9)){free(name);return fail(w,"invalid QKV target");}
            strcpy(name+strlen(name)-4,"qkv_proj.weight");
            t->base=find_base(w,name,&t->file);
            if(!t->base||t->base->ndim!=2||!t->base->shape[0]||t->base->shape[0]%3){free(name);return fail(w,"invalid fused QKV dimensions");}
            uint64_t width=t->base->shape[0]/3;const lj_value *h=lj_get(w->config,"num_attention_heads"),*d=lj_get(w->config,"attention_head_dim");
            if(h&&d&&(h->type!=LJ_NUMBER||d->type!=LJ_NUMBER||strpbrk(h->text,".eE")||strpbrk(d->text,".eE")||h->number<=0||d->number<=0||floor(h->number)!=h->number||floor(d->number)!=d->number||h->number*d->number!=(double)width)){
                free(name);return fail(w,"QKV dimensions disagree with config");
            }
            t->start=(uint64_t)slice*width;t->stop=t->start+width;
        }else{
            if(suffix(name,".attn.to_out.0"))strcpy(name+strlen(name)-8,"out_proj");
            strcat(name,".weight");
        }
    }
    if(!t->base)t->base=find_base(w,name,&t->file);
    free(name);
    if(!t->base)return fail(w,"unmatched adapter target: %s",t->module);
    if(t->base->dtype!=H3_DTYPE_BF16||t->base->ndim!=2||!t->base->shape[0]||!t->base->shape[1])return fail(w,"base target must be nonempty BF16 matrix: %s",t->module);
    if(!t->stop)t->stop=t->base->shape[0];
    return 1;
}
static int target_compare(const void *a,const void *b) {return strcmp(((const target *)a)->module,((const target *)b)->module);}
static int plan_adapter(work *w,size_t ai) {
    source_file *f=&w->adapters[ai];f->path=strdup(w->selection->args[ai].path);
    if(!f->path||!hash_file(w,f,0)||!read_header(w,f))return 0;
    lj_value *meta=metadata(w,&f->st);if(!meta)return 0;int ok=0;size_t first=w->target_count;
    for(size_t i=0;i<f->st.tensor_count;i++){
        const h3_st_tensor *st=&f->st.tensors[i];if(architecture(st->name)){fail(w,"unsupported architectural adapter: %s",st->name);goto end;}
        const char *parts[]={".lora_A.weight",".lora_down.weight",".lora_A.default.weight",".lora_down.default.weight",".lora_B.weight",".lora_up.weight",".lora_B.default.weight",".lora_up.default.weight",".alpha"};
        int part=-1;size_t len=0;
        for(int p=0;p<9;p++)if(suffix(st->name,parts[p])){part=p;len=strlen(st->name)-strlen(parts[p]);break;}
        if(part<0||!len){fail(w,"unsupported adapter tensor: %s",st->name);goto end;}
        char *module=strndup(st->name,len);if(!module)goto end;
        size_t ti;for(ti=first;ti<w->target_count;ti++)if(!strcmp(w->targets[ti].module,module))break;
        if(ti==w->target_count){target *a=realloc(w->targets,(ti+1)*sizeof(*a));if(!a){free(module);goto end;}
            w->targets=a;memset(&a[ti],0,sizeof(*a));a[ti].module=module;a[ti].adapter=ai;w->target_count++;
        }else free(module);
        target *t=&w->targets[ti];const h3_st_tensor **slot=part<4?&t->a:part<8?&t->b:&t->alpha_tensor;
        if(*slot){fail(w,"duplicate LoRA pair: %s",t->module);goto end;}*slot=st;
    }
    if(first==w->target_count){fail(w,"adapter contains no LoRA pairs");goto end;}
    qsort(w->targets+first,w->target_count-first,sizeof(target),target_compare);
    for(size_t i=first;i<w->target_count;i++){
        target *t=&w->targets[i];
        if(!t->a||!t->b){fail(w,"missing A/B pair or orphan alpha: %s",t->module);goto end;}
        if(!floats(t->a->dtype)||!floats(t->b->dtype)||t->a->ndim!=2||t->b->ndim!=2){fail(w,"A/B must be BF16/F16/F32 matrices");goto end;}
        if(!normalize(w,t))goto end;
        t->rank=t->a->shape[0];
        if(!t->rank||t->b->shape[1]!=t->rank||t->a->shape[1]!=t->base->shape[1]||t->b->shape[0]!=t->stop-t->start){fail(w,"LoRA rank/orientation/shape mismatch: %s",t->module);goto end;}
        for(size_t j=first;j<i;j++)if(t->base==w->targets[j].base&&t->start<w->targets[j].stop&&t->stop>w->targets[j].start){fail(w,"overlapping targets within adapter: %s",t->module);goto end;}
        const char *ranks[]={"rank","r","lora_rank","ss_network_dim","network_dim"};double num;
        for(size_t j=0;j<5;j++){const lj_value *v=lj_get(meta,ranks[j]);if(v&&(!numeric(v,&num)||num!=(double)t->rank)){fail(w,"rank metadata disagrees: %s",t->module);goto end;}}
        const char *alphas[]={"alpha","lora_alpha","ss_network_alpha","network_alpha"};double alpha=(double)t->rank;int has=0;
        for(size_t j=0;j<4;j++){const lj_value *v=lj_get(meta,alphas[j]);if(!v)continue;
            if(!numeric(v,&num)||num<0||(has&&num!=alpha)){fail(w,"invalid/conflicting alpha metadata");goto end;}alpha=num;has=1;}
        if(t->alpha_tensor){
            const h3_st_tensor *a=t->alpha_tensor;unsigned char buf[4];float v;int fd=open(f->path,O_RDONLY);
            int good=fd>=0&&floats(a->dtype)&&h3_st_tensor_elements(a)==1&&read_floats(w,fd,a,0,1,&v,buf);if(fd>=0)close(fd);
            if(!good||v<0||(has&&(double)v!=alpha)){fail(w,"invalid/conflicting alpha tensor");goto end;}alpha=v;
        }
        t->alpha=alpha;t->explicit_alpha=has||t->alpha_tensor!=NULL;
        double scale=w->selection->args[ai].scale*alpha/(double)t->rank;
        if(!isfinite(scale)||fabs(scale)>FLT_MAX){fail(w,"effective LoRA scale overflows FP32");goto end;}t->scale=(float)scale;
    }
    /* Scan every adapter value, including zero-strength selections. */
    {int fd=open(f->path,O_RDONLY);if(fd<0)goto end;unsigned char buf[IO_SIZE];
    for(size_t i=0;i<f->st.tensor_count;i++){
        const h3_st_tensor *t=&f->st.tensors[i];size_t sz=h3_dtype_size(t->dtype);
        for(uint64_t off=0;off<t->data_end-t->data_begin;){size_t n=(size_t)MIN(IO_SIZE,t->data_end-t->data_begin-off);
            if(!io_read(w,fd,buf,n,t->file_offset+off)||!checkpoint(w,"LoRA planning",0,0)){close(fd);goto end;}
            for(size_t j=0;j<n;j+=sz)if(!isfinite(decode(buf+j,t->dtype))){close(fd);fail(w,"non-finite adapter tensor: %s",t->name);goto end;}
            off+=n;
        }
    }close(fd);}
    {char digest[65];hex(f->digest,digest);size_t fallbacks=0;
    for(size_t i=first;i<w->target_count;i++)if(!w->targets[i].explicit_alpha)fallbacks++;
    fprintf(stderr,"LoRA adapter %zu: %s scale %.17g, %zu targets, alpha=rank fallback on %zu target(s)%s\n",
        ai+1,f->path,w->selection->args[ai].scale,w->target_count-first,fallbacks,
        !strcmp(digest,"5f3a626cd72c93a8b9318d6760c510bc5092d2ab13aaba1f932c5bab07a416d3")?", pinned Turbo v4 profile (steps remain explicit)":"");}
    ok=1;
end:lj_free(meta);return ok;
}

static int validate_zero_targets(work *w) {
    unsigned char raw[IO_SIZE];
    for(size_t i=0;i<w->target_count;i++) {
        target *t=&w->targets[i];int seen=0,active=0;
        for(size_t j=0;j<w->target_count;j++)if(w->targets[j].base==t->base) {
            if(j<i)seen=1;
            if(w->targets[j].scale!=0)active=1;
        }
        if(seen||active)continue;
        int fd=open(w->files[t->file].path,O_RDONLY);if(fd<0)return fail(w,"cannot validate zero-strength base target");
        uint64_t bytes=t->base->data_end-t->base->data_begin;int ok=1;
        for(uint64_t off=0;ok&&off<bytes;) {
            size_t n=(size_t)MIN(IO_SIZE,bytes-off);
            ok=io_read(w,fd,raw,n,t->base->file_offset+off)&&checkpoint(w,"LoRA validating zero-strength targets",0,0);
            for(size_t k=0;ok&&k<n;k+=2) {
                unsigned value=(unsigned)raw[k]|((unsigned)raw[k+1]<<8);
                if((value&0x7f80)==0x7f80)ok=fail(w,"non-finite zero-strength base target: %s",t->base->name);
            }
            off+=n;
        }
        close(fd);if(!ok)return 0;
    }
    return 1;
}

/* Rank tiling never restarts a dot product; column SIMD never reassociates K.
 * Larger budgets amortize I/O with wider/taller tiles without changing bytes. */
static int fold_tensor(work *w,const h3_st_tensor *base,size_t file,const char *outdir) {
    int active=0;for(size_t i=0;i<w->target_count;i++)if(w->targets[i].base==base&&w->targets[i].scale!=0)active=1;
    if(!active)return 1;
    size_t available=w->selection->budget-w->plan_bytes;
    int large=available>=8*1024*1024;
    size_t cols=(size_t)MIN(base->shape[1],large?4096u:512u),rows=(size_t)MIN(base->shape[0],large?128u:32u),ranks=128;
    size_t floats_n=2*rows*cols+ranks*cols+rows*ranks+cols;
    size_t bytes=floats_n*sizeof(float)+(cols>ranks?cols:ranks)*4+cols*2+rows*3*8*4+cols*3*4+1024;
    if(bytes+IO_SIZE>w->selection->budget-w->plan_bytes)return fail(w,"LoRA scratch budget cannot hold one tile");
    if(!h3_memory_check(bytes,"LoRA folding",w->error,w->error_size))return 0;
    if(!injected(w,"allocation"))return 0;
    float *storage=malloc(floats_n*sizeof(float));unsigned char *raw=malloc((cols>ranks?cols:ranks)*4),*packed=malloc(cols*2);
    char *path=join(outdir,w->files[file].name);int src=-1,dst=-1,ok=0;
    if(!storage||!raw||!packed||!path) {fail(w,"out of memory folding");goto end;}
    if(bytes+IO_SIZE+w->plan_bytes>w->scratch_peak)w->scratch_peak=bytes+IO_SIZE+w->plan_bytes;
    src=open(w->files[file].path,O_RDONLY);dst=open(path,O_RDWR|O_NOFOLLOW);
    if(src<0||dst<0){fail(w,"cannot open fold source/output: %s",strerror(errno));goto end;}
    float *weight=storage,*delta=weight+rows*cols,*a=delta+rows*cols,*b=a+ranks*cols,*check=b+rows*ranks;
    for(uint64_t r0=0;r0<base->shape[0];r0+=rows)for(uint64_t c0=0;c0<base->shape[1];c0+=cols){
        size_t nr=(size_t)MIN(rows,base->shape[0]-r0),nc=(size_t)MIN(cols,base->shape[1]-c0);
        if(!checkpoint(w,"LoRA folding",0,0))goto end;
        for(size_t r=0;r<nr;r++)if(!read_floats(w,src,base,(r0+r)*base->shape[1]+c0,nc,weight+r*cols,raw))goto end;
        /* Three independent dense matrix-vector probes. Compute B(Ax) in
         * FP64, separately from the FP32 BA kernel, for every row/column tile. */
        double expected[rows][3],bound[rows][3];memset(expected,0,sizeof(expected));memset(bound,0,sizeof(bound));
        float x[cols][3];
        for(size_t c=0;c<nc;c++)for(int p=0;p<3;p++) {
            uint32_t v=(uint32_t)(c0+c+1)*UINT32_C(2654435761)+(uint32_t)p*UINT32_C(2246822519);
            v^=v>>13;v*=UINT32_C(3266489917);v^=v>>16;
            x[c][p]=(float)((int)(v&65535)-32768)/32768.0f;
        }
        for(size_t r=0;r<nr;r++)for(size_t c=0;c<nc;c++)for(int p=0;p<3;p++) {
            double v=(double)weight[r*cols+c]*x[c][p];expected[r][p]+=v;bound[r][p]+=fabs(v)*FLT_EPSILON;
        }
        for(size_t ti=0;ti<w->target_count;ti++){
            target *t=&w->targets[ti];if(t->base!=base||t->scale==0||r0>=t->stop||r0+nr<=t->start)continue;
            uint64_t lo=r0>t->start?r0:t->start,hi=MIN(r0+nr,t->stop);
            size_t skip=(size_t)(lo-r0),nrows=(size_t)(hi-lo);memset(delta,0,rows*cols*sizeof(float));
            int fd=open(w->adapters[t->adapter].path,O_RDONLY);if(fd<0){fail(w,"cannot reopen adapter");goto end;}
            double probe[rows][3],magnitude[rows][3];memset(probe,0,sizeof(probe));memset(magnitude,0,sizeof(magnitude));int good=1;
            for(uint64_t k0=0;k0<t->rank&&good;k0+=ranks){
                size_t nk=(size_t)MIN(ranks,t->rank-k0);
                for(size_t k=0;k<nk&&good;k++)good=read_floats(w,fd,t->a,(k0+k)*base->shape[1]+c0,nc,a+k*cols,raw);
                for(size_t r=0;r<nrows&&good;r++)good=read_floats(w,fd,t->b,(lo-t->start+r)*t->rank+k0,nk,b+r*ranks,raw);
                if(!good)break;
                for(size_t r=0;r<nrows;r++)for(size_t k=0;k<nk;k++){
                    float bv=b[r*ranks+k];float *dr=delta+(r+skip)*cols;const float *ak=a+k*cols;
                    for(size_t c=0;c<nc;c++)dr[c]=dr[c]+bv*ak[c];
                }
                for(size_t k=0;k<nk;k++) {
                    double ax[3]={0},absolute[3]={0};
                    for(size_t c=0;c<nc;c++)for(int p=0;p<3;p++) {
                        double v=(double)a[k*cols+c]*x[c][p];ax[p]+=v;absolute[p]+=fabs(v);
                    }
                    for(size_t r=0;r<nrows;r++)for(int p=0;p<3;p++) {
                        probe[r+skip][p]+=(double)b[r*ranks+k]*ax[p];
                        magnitude[r+skip][p]+=fabs((double)b[r*ranks+k])*absolute[p];
                    }
                }
                good=checkpoint(w,"LoRA folding",0,0);
            }close(fd);if(!good)goto end;
            for(size_t r=skip;r<skip+nrows;r++)for(size_t c=0;c<nc;c++){
                float d=delta[r*cols+c]*t->scale;float v=weight[r*cols+c]+d;
                if(!isfinite(d)||!isfinite(v)){fail(w,"non-finite FP32 folding result: %s",base->name);goto end;}weight[r*cols+c]=v;
            }
            for(size_t r=0;r<nr;r++)for(int p=0;p<3;p++) {
                expected[r][p]+=probe[r][p]*t->scale;
                bound[r][p]+=(magnitude[r][p]*fabs((double)t->scale)+fabs(expected[r][p]))*((double)t->rank+4)*FLT_EPSILON+1e-30;
            }
        }
        for(size_t r=0;r<nr;r++)for(int p=0;p<3;p++) {
            double actual=0;for(size_t c=0;c<nc;c++)actual+=(double)weight[r*cols+c]*x[c][p];
            if(!isfinite(expected[r][p])||fabs(actual-expected[r][p])>bound[r][p]){fail(w,"independent FP32 matrix-vector probe failed");goto end;}
        }
        for(size_t r=0;r<nr;r++){
            for(size_t c=0;c<nc;c++){
                float value=weight[r*cols+c];uint32_t bits;memcpy(&bits,&value,4);
                uint16_t v=(uint16_t)((bits+0x7fffu+((bits>>16)&1))>>16);
                if((v&0x7fff)==0x7f80){fail(w,"BF16 overflow: %s",base->name);goto end;}
                packed[2*c]=(unsigned char)v;packed[2*c+1]=(unsigned char)(v>>8);
            }
            uint64_t offset=base->file_offset+((r0+r)*base->shape[1]+c0)*2;
            if(!io_write(w,dst,packed,nc*2,offset)||!read_floats(w,dst,base,(r0+r)*base->shape[1]+c0,nc,check,raw))goto end;
            if(memcmp(raw,packed,nc*2)){fail(w,"stored BF16 verification failed");goto end;}
            for(int p=0;p<3;p++) {
                double actual=0,round_bound=0;
                for(size_t c=0;c<nc;c++) {
                    actual+=(double)check[c]*x[c][p];
                    round_bound+=fabs((double)weight[r*cols+c]*x[c][p])/256+ldexp(1.0,-133);
                }
                if(fabs(actual-expected[r][p])>bound[r][p]+round_bound) {
                    fail(w,"independent stored BF16 matrix-vector probe failed");goto end;
                }
            }
        }
    }
    ok=1;
end:if(src>=0)close(src);if(dst>=0)close(dst);free(path);free(storage);free(raw);free(packed);return ok;
}
static int safe_name(const char *s) {return s&&*s&&strcmp(s,".")&&strcmp(s,"..")&&!strchr(s,'/');}
/* Resolve existing ancestors before creating anything. This also normalizes
 * dot components, so overlap checks cannot be bypassed with /../ spellings. */
static char *future_path(const char *path) {
    char *copy=strdup(path),*normal=calloc(strlen(path)+2,1);
    if(!copy||!normal){free(copy);free(normal);return NULL;}
    strcpy(normal,"/");char *save=NULL;
    for(char *part=strtok_r(copy,"/",&save);part;part=strtok_r(NULL,"/",&save)) {
        if(!strcmp(part,"."))continue;
        if(!strcmp(part,"..")) {char *slash=strrchr(normal,'/');if(slash!=normal)*slash=0;else normal[1]=0;continue;}
        if(strlen(normal)>1)strcat(normal,"/");
        strcat(normal,part);
    }
    free(copy);copy=strdup(normal);if(!copy){free(normal);return NULL;}
    char *resolved=NULL;size_t end=strlen(copy);
    for(;;) {
        resolved=realpath(copy,NULL);if(resolved)break;
        if(errno!=ENOENT){free(copy);free(normal);return NULL;}
        char *slash=strrchr(copy,'/');if(!slash||slash==copy){copy[1]=0;end=1;resolved=realpath(copy,NULL);break;}
        end=(size_t)(slash-copy);*slash=0;
    }
    char *result=NULL;
    if(resolved){size_t n=strlen(resolved)+strlen(normal+end)+2;result=malloc(n);
        if(result)snprintf(result,n,"%s%s%s",resolved,end==1&&normal[end]?"/":"",normal+end);}
    free(copy);free(normal);free(resolved);return result;
}
static int private_dir(work *w,const char *path,int create) {
    struct stat st;
    if(lstat(path,&st)){
        if(errno!=ENOENT||!create)return fail(w,"cache directory %s: %s",path,strerror(errno));
        char *parent=strdup(path);if(!parent)return fail(w,"out of memory");char *slash=strrchr(parent,'/');
        if(slash&&slash!=parent){*slash=0;struct stat ps;if(stat(parent,&ps)&&!private_dir(w,parent,1)){free(parent);return 0;}}
        free(parent);if(mkdir(path,0700)&&errno!=EEXIST)return fail(w,"cannot create cache %s: %s",path,strerror(errno));
        if(lstat(path,&st))return fail(w,"cannot stat cache %s",path);
    }
    if(!S_ISDIR(st.st_mode)||st.st_uid!=geteuid()||(st.st_mode&0077))return fail(w,"cache must be an owned directory with private permissions and no symlinks: %s",path);
    return 1;
}
static int under(const char *path,const char *dir) {size_t n=strlen(dir);return !strncmp(path,dir,n)&&(path[n]==0||path[n]=='/');}
static int sync_dir(work *w,const char *path) {if(!injected(w,"fsync"))return 0;int fd=open(path,O_RDONLY|O_DIRECTORY|O_NOFOLLOW);int ok=fd>=0&&fsync(fd)==0;if(fd>=0)close(fd);return ok?1:fail(w,"cache directory fsync failed: %s",strerror(errno));}
/* Only called for our private staging trees, under their stable key lock. */
static int remove_tree(const char *path) {
    struct stat st;if(lstat(path,&st))return errno==ENOENT;
    if(st.st_uid!=geteuid())return 0;
    if(!S_ISDIR(st.st_mode))return unlink(path)==0;
    DIR *d=opendir(path);if(!d)return 0;struct dirent *e;int ok=1;
    while((e=readdir(d)))if(strcmp(e->d_name,".")&&strcmp(e->d_name,"..")){
        char *p=join(path,e->d_name);if(!p||!remove_tree(p))ok=0;free(p);
    }
    closedir(d);return ok&&rmdir(path)==0;
}
static int lock_wait(work *w,int fd,int mode) {
    if(!injected(w,"lock"))return 0;
    for(;;){if(!flock(fd,mode|LOCK_NB))return 1;
        if(errno!=EWOULDBLOCK&&errno!=EAGAIN&&errno!=EINTR)return fail(w,"LoRA cache lock failed: %s",strerror(errno));
        if(!checkpoint(w,"LoRA waiting for cache lock",0,0))return 0;
        struct timespec t={0,100000000};nanosleep(&t,NULL);
    }
}
static int copy_file(work *w,source_file *src,const char *dest,int replace_clone) {
    if(!injected(w,"copy")||!injected(w,"permission"))return 0;
#ifdef __APPLE__
    if(!replace_clone&&!clonefile(src->path,dest,0)){w->cloned_files++;src->cloned=1;return chmod(dest,0600)==0;}
#endif
    int in=open(src->path,O_RDONLY),out=open(dest,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600),ok=0;
    if(in<0||out<0){fail(w,"cannot create independent cache copy: %s",strerror(errno));goto end;}
#ifndef __APPLE__
    if(!replace_clone&&!ioctl(out,FICLONE,in)){w->cloned_files++;src->cloned=1;ok=1;goto end;}
#endif
    struct statvfs space;uint64_t remaining=UINT64_C(64)*1024*1024;
    size_t first=(size_t)(src-w->files),limit=replace_clone?first+1:w->file_count;
    for(size_t i=first;i<limit;i++) {
        if(w->files[i].size>UINT64_MAX-remaining){fail(w,"LoRA disk plan overflows");goto end;}
        remaining+=w->files[i].size;
    }
    for(int poll=0;;poll++) {
        int fits=fstatvfs(out,&space)||(remaining<=(uint64_t)space.f_bavail*space.f_frsize);
#ifdef H3_LORA_TESTING
        if(replace_clone&&getenv("H3_TEST_LORA_DELAYED_QUOTA")&&poll<3)fits=0;
#endif
        if(fits)break;
        /* Reflink quota release can lag unlink. Allow a bounded, cancellable
         * grace period only when replacing our just-removed staged clone. */
        if(!replace_clone||poll>=50){fail(w,"insufficient cache disk space: full-copy fallback needs %llu more bytes; choose --lora-cache on a larger volume",(unsigned long long)remaining);goto end;}
        if(!checkpoint(w,"LoRA waiting for released disk quota",0,0))goto end;
        struct timespec delay={0,100000000};nanosleep(&delay,NULL);
    }
    w->copied_files++;
    unsigned char buf[IO_SIZE];ok=1;
    for(uint64_t off=0;ok&&off<src->size;){size_t n=(size_t)MIN(IO_SIZE,src->size-off);
        ok=io_read(w,in,buf,n,off)&&io_write(w,out,buf,n,off);off+=n;
        if(!(off%(16*1024*1024)))ok=ok&&checkpoint(w,"LoRA copying",0,0);
    }
end:if(in>=0)close(in);if(out>=0)close(out);return ok;
}
static int source_stamps(work *w) {
    for(size_t i=0;i<w->file_count+w->selection->count;i++){
        source_file *f=i<w->file_count?&w->files[i]:&w->adapters[i-w->file_count];struct stat current;
        if(stat(f->path,&current)||!same_stamp(&current,&f->stamp)){fail(w,"source changed before cache reuse: %s",f->path);return 0;}
    }
    return 1;
}
static int unchanged(work *w,const char *source) {
    DIR *directory=opendir(source);struct dirent *entry;if(!directory)return fail(w,"transformer disappeared");size_t count=0;
    while((entry=readdir(directory)))if(strcmp(entry->d_name,".")&&strcmp(entry->d_name,"..")) {count++;if(count>w->file_count)break;}
    closedir(directory);
    if(count!=w->file_count)return fail(w,"source transformer inventory changed");
    for(size_t i=0;i<w->file_count;i++)if(!hash_file(w,&w->files[i],1))return 0;
    for(size_t i=0;i<w->selection->count;i++)if(!hash_file(w,&w->adapters[i],1))return 0;
    return 1;
}
static int untouched(work *w,size_t fi,const char *dir) {
    if(!checkpoint(w,"LoRA validating untouched bytes",0,0))return 0;
    source_file *f=&w->files[fi];if(!strcmp(f->name,IDENTITY))return 1;
    char *p=join(dir,f->name);if(!p)return fail(w,"out of memory");int a=open(f->path,O_RDONLY),b=open(p,O_RDONLY|O_NOFOLLOW);free(p);
    int ok=a>=0&&b>=0;unsigned char x[IO_SIZE/2],y[IO_SIZE/2];
    for(uint64_t off=0;ok&&off<f->size;){size_t n=(size_t)MIN(sizeof(x),f->size-off);ok=io_read(w,a,x,n,off)&&io_read(w,b,y,n,off);
        /* Discard only the planned target row ranges from this comparison. */
        for(size_t i=0;i<w->target_count;i++){
            target *t=&w->targets[i];if(t->file!=fi||t->scale==0)continue;
            uint64_t lo=t->base->file_offset+t->start*t->base->shape[1]*2,hi=t->base->file_offset+t->stop*t->base->shape[1]*2;
            uint64_t begin=lo>off?lo:off,end=MIN(hi,off+n);
            if(begin<end){memset(x+(size_t)(begin-off),0,(size_t)(end-begin));memset(y+(size_t)(begin-off),0,(size_t)(end-begin));}
        }
        if(ok&&memcmp(x,y,n))ok=fail(w,"untouched source bytes changed: %s",f->name);
        off+=n;
        if(!(off%(16*1024*1024)))ok=ok&&checkpoint(w,"LoRA validating untouched bytes",0,0);
    }
    if(a>=0)close(a);
    if(b>=0)close(b);
    return ok;
}
static int finish_file(work *w,FILE *f) {int ok=fflush(f)==0&&fsync(fileno(f))==0;if(fclose(f))ok=0;return ok?1:fail(w,"LoRA manifest flush failed: %s",strerror(errno));}
static int identity_emit(work *w,FILE *f,const char *key,int mode) {
    fprintf(f,"{\"schema\":1,\"mapping\":1,\"recipe\":1,\"mode\":\"%s\",\"key\":\"%s\",\"adapters\":[",mode?"Ref2VA":"FL2VA",key);
    for(size_t i=0;i<w->selection->count;i++){
        char digest[65];hex(w->adapters[i].digest,digest);uint64_t bits;memcpy(&bits,&w->selection->args[i].scale,8);
        fprintf(f,"%s{\"sha256\":\"%s\",\"scale_bits\":\"%016llx\"}",i?",":"",digest,(unsigned long long)bits);
    }
    fputs("]}\n",f);return !ferror(f);
}
static int identity_write(work *w,const char *dir,const char *key,int mode) {
    char *p=join(dir,IDENTITY);if(!p)return fail(w,"out of memory");
    int fd=open(p,O_WRONLY|O_CREAT|O_TRUNC|O_NOFOLLOW,0600);free(p);
    if(fd<0)return fail(w,"cannot write LoRA identity");
    FILE *f=fdopen(fd,"w");if(!f){close(fd);return 0;}
    int written=identity_emit(w,f,key,mode);return finish_file(w,f)&&written;
}
static int identity_digest(work *w,const char *key,int mode,unsigned char digest[32]) {
    char *text=NULL;size_t bytes=0;FILE *f=open_memstream(&text,&bytes);if(!f)return 0;
    int ok=identity_emit(w,f,key,mode);if(fclose(f))ok=0;
    if(ok){h3_sha256_ctx hash;h3_sha256_init_fast(&hash);h3_sha256_update(&hash,text,(h3_sha256_size)bytes);h3_sha256_final(digest,&hash);}
    free(text);return ok;
}
static int manifest_write(work *w,const char *stage,const char *key,int mode) {
    char *transformer=join(stage,"transformer"),*path=join(stage,"manifest.json");if(!path||!transformer){free(path);free(transformer);return 0;}
    int fd=open(path,O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);free(path);
    FILE *f=fd>=0?fdopen(fd,"w"):NULL;if(!f){if(fd>=0)close(fd);free(transformer);return fail(w,"cannot create cache manifest");}
    fprintf(f,"{\"schema\":1,\"key\":\"%s\",\"mode\":%d,\"scratch_peak\":%zu,\"sources\":[",key,mode,w->scratch_peak);
    for(size_t i=0;i<w->selection->count;i++){if(i)fputc(',',f);lj_quote(f,w->selection->args[i].path);}fputs("],\"source_files\":[",f);
    for(size_t i=0;i<w->file_count;i++){
        source_file *src=&w->files[i];char digest[65];hex(src->digest,digest);
        if(i)fputc(',',f);
        fputs("{\"name\":",f);lj_quote(f,src->name);fputs(",\"path\":",f);lj_quote(f,src->path);
        fprintf(f,",\"size\":%llu,\"sha256\":\"%s\"}",(unsigned long long)src->size,digest);
    }
    fprintf(f,"],\"timing_seconds\":{\"hash\":%.6f,\"copy\":%.6f,\"fold\":%.6f,\"validation\":%.6f},\"clone_files\":%zu,\"copy_files\":%zu,\"targets\":[",
        w->hash_seconds,w->copy_seconds,w->fold_seconds,w->validation_seconds,w->cloned_files,w->copied_files);
    for(size_t i=0;i<w->target_count;i++){
        target *t=&w->targets[i];if(i)fputc(',',f);fputs("{\"module\":",f);lj_quote(f,t->module);fputs(",\"target\":",f);lj_quote(f,t->base->name);
        uint64_t alpha;memcpy(&alpha,&t->alpha,8);uint32_t scale;memcpy(&scale,&t->scale,4);
        fprintf(f,",\"adapter\":%zu,\"rows\":[%llu,%llu],\"rank\":%llu,\"alpha_binary64\":\"%016llx\",\"effective_scale_binary32\":\"%08x\",\"alpha_fallback\":%s}",
            t->adapter,(unsigned long long)t->start,(unsigned long long)t->stop,(unsigned long long)t->rank,(unsigned long long)alpha,scale,t->explicit_alpha?"false":"true");
    }
    fputs("],\"files\":[",f);
    /* Output file records follow; source paths/timing remain outside the
     * transformer and therefore cannot disturb exact resume fingerprints. */
    int ok=1,first=1;int has_identity=0;
    for(size_t i=0;i<w->file_count;i++)if(!strcmp(w->files[i].name,IDENTITY))has_identity=1;
    for(size_t i=0;ok&&i<w->file_count+(size_t)!has_identity;i++){
        const char *name=i<w->file_count?w->files[i].name:IDENTITY;
        source_file out={.path=join(transformer,name)};
        ok=out.path&&hash_file(w,&out,0);
        if(ok){int file=open(out.path,O_RDONLY|O_NOFOLLOW);ok=file>=0&&fsync(file)==0;if(file>=0)close(file);}
        if(ok){char digest[65];hex(out.digest,digest);if(!first)fputc(',',f);first=0;
            fputs("{\"name\":",f);lj_quote(f,name);fprintf(f,",\"size\":%llu,\"sha256\":\"%s\"}",(unsigned long long)out.size,digest);
        }file_free(&out);
    }
    fputs("]}\n",f);ok=finish_file(w,f)&&ok;free(transformer);return ok;
}
static int verify_entry(work *w,const char *entry,const char *key,int mode) {
    char *path=join(entry,"manifest.json"),*dir=join(entry,"transformer");struct stat st;
    if(!path||!dir){free(path);free(dir);return 0;}
    int ok=lstat(entry,&st)==0&&S_ISDIR(st.st_mode)&&st.st_uid==geteuid()&&lstat(path,&st)==0&&S_ISREG(st.st_mode)&&
        lstat(dir,&st)==0&&S_ISDIR(st.st_mode);
    if(ok){
        DIR *d=opendir(entry);struct dirent *e;size_t count=0;
        if(!d)ok=0;
        else {
            while((e=readdir(d)))if(strcmp(e->d_name,".")&&strcmp(e->d_name,"..")) {
                if(strcmp(e->d_name,"manifest.json")&&strcmp(e->d_name,"transformer"))ok=0;
                count++;
            }
            closedir(d);if(count!=2)ok=0;
        }
    }
    unsigned char stored_identity[32]={0};
    if(ok&&(lstat(path,&st)||st.st_size<0||(uint64_t)st.st_size>(w->selection->budget-w->plan_bytes-IO_SIZE)/16))ok=0;
    lj_value *manifest=ok?lj_file(path,w->error,w->error_size):NULL;
    const lj_value *schema=lj_get(manifest,"schema"),*k=lj_get(manifest,"key"),*m=lj_get(manifest,"mode"),*files=lj_get(manifest,"files");
    ok=schema&&schema->type==LJ_NUMBER&&schema->number==1&&k&&k->type==LJ_STRING&&!strcmp(k->text,key)&&m&&m->type==LJ_NUMBER&&m->number==mode&&files&&files->type==LJ_ARRAY;
    size_t expected=w->file_count+1;for(size_t i=0;i<w->file_count;i++)if(!strcmp(w->files[i].name,IDENTITY))expected--;
    if(ok&&files->count!=expected)ok=0;
    DIR *directory=ok?opendir(dir):NULL;struct dirent *item;size_t count=0;
    if(!directory)ok=0;
    else {
        while((item=readdir(directory)))if(strcmp(item->d_name,".")&&strcmp(item->d_name,"..")) {
            count++;if(count>expected){ok=0;break;}
        }
        closedir(directory);
    }
    if(count!=expected)ok=0;
    for(size_t i=0;ok&&i<files->count;i++){
        const lj_value *v=files->items[i],*name=lj_get(v,"name"),*size=lj_get(v,"size"),*digest=lj_get(v,"sha256");
        ok=name&&name->type==LJ_STRING&&safe_name(name->text)&&size&&size->type==LJ_NUMBER&&digest&&digest->type==LJ_STRING&&strlen(digest->text)==64;
        if(!ok)break;
        int known=!strcmp(name->text,IDENTITY);for(size_t j=0;j<w->file_count;j++)if(!strcmp(name->text,w->files[j].name))known=1;
        for(size_t j=0;j<i;j++){const lj_value *other=lj_get(files->items[j],"name");if(other&&!strcmp(other->text,name->text))known=0;}
        if(!known){ok=0;break;}
        source_file f={.path=join(dir,name->text)};
        ok=f.path&&lstat(f.path,&st)==0&&S_ISREG(st.st_mode)&&st.st_nlink==1&&st.st_uid==geteuid()&&hash_file(w,&f,0);
        if(ok){char h[65];hex(f.digest,h);ok=(double)f.size==size->number&&!strcmp(h,digest->text);if(!strcmp(name->text,IDENTITY))memcpy(stored_identity,f.digest,32);}file_free(&f);
    }
    /* Enforce canonical deterministic identity bytes, not merely equivalent
     * JSON. Rebuilding the same key must retain the exact model fingerprint. */
    if(ok){unsigned char identity_hash[32];ok=identity_digest(w,key,mode,identity_hash)&&!memcmp(identity_hash,stored_identity,32);}
    lj_free(manifest);free(path);free(dir);return ok;
}

int h3_lora_prepare(const h3_lora_selection *selection,const char *source,int mode,
    h3_progress_callback progress,void *opaque,h3_lora_variant *variant,char *error,size_t cap) {
    if(error&&cap)error[0]=0;
    if(!variant){if(error&&cap)snprintf(error,cap,"missing LoRA variant output");return 0;}
    if(!selection){memset(variant,0,sizeof(*variant));variant->lease=-1;return 1;}
    work w={.selection=selection,.progress=progress,.opaque=opaque,.error=error,.error_size=cap};
    double start=now(),hashed=0,verified=0;int ok=0,lock=-1,published=0,hit=0;
    char *root=NULL,*locks=NULL,*variants=NULL,*staging=NULL,*invalid=NULL,*entry=NULL,*lockpath=NULL,*stage=NULL,*outdir=NULL;
    char key[65]={0};locale_t previous_locale=uselocale((locale_t)0),numeric_locale=newlocale(LC_NUMERIC_MASK,"C",(locale_t)0);fenv_t env;int env_saved=fegetenv(&env)==0;
#if defined(__SSE__)
    unsigned mxcsr=_mm_getcsr();_mm_setcsr(mxcsr&~0x8040u);
#endif
#if defined(__aarch64__)
    uint64_t fpcr;__asm__ volatile("mrs %0, fpcr":"=r"(fpcr));uint64_t gradual=fpcr&~((UINT64_C(1)<<24)|(UINT64_C(1)<<19)|UINT64_C(3));__asm__ volatile("msr fpcr, %0"::"r"(gradual));
#endif
    if(!numeric_locale||!uselocale(numeric_locale)){fail(&w,"cannot establish locale-independent LoRA metadata");goto end;}
    if(!env_saved||fesetround(FE_TONEAREST)){fail(&w,"cannot establish deterministic LoRA floating-point environment");goto end;}
    if(!source||mode<0||mode>1||selection->budget<1024*1024){fail(&w,"invalid LoRA source/mode or scratch budget");goto end;}
    if(!checkpoint(&w,"LoRA planning and hashing",0,0)||!inventory(&w,source))goto end;
    w.adapters=calloc(selection->count,sizeof(*w.adapters));if(!w.adapters){fail(&w,"out of memory");goto end;}
    for(size_t i=0;i<selection->count;i++)if(!plan_adapter(&w,i))goto end;
    if(!validate_zero_targets(&w))goto end;
    w.scratch_peak=w.plan_bytes+IO_SIZE;
    h3_sha256_ctx hash;h3_sha256_init_fast(&hash);hash_string(&hash,RECIPE);hash_u64(&hash,(uint64_t)mode);hash_u64(&hash,w.file_count);
    for(size_t i=0;i<w.file_count;i++){hash_string(&hash,w.files[i].name);hash_u64(&hash,w.files[i].size);h3_sha256_update(&hash,w.files[i].digest,32);}
    hash_u64(&hash,selection->count);
    for(size_t i=0;i<selection->count;i++){h3_sha256_update(&hash,w.adapters[i].digest,32);uint64_t bits;memcpy(&bits,&selection->args[i].scale,8);hash_u64(&hash,bits);}
    unsigned char digest[32];h3_sha256_final(digest,&hash);hex(digest,key);hashed=now()-start;w.hash_seconds=hashed;
    fprintf(stderr,"LoRA %s: %zu adapter(s), %zu targets, key %s (source hashing/planning %.2fs)\n",mode?"Ref2VA":"FL2VA",selection->count,w.target_count,key,hashed);
    root=future_path(selection->cache);char *resolved_source=realpath(source,NULL);
    if(!root||!resolved_source){free(resolved_source);fail(&w,"cannot resolve cache/source directories");goto end;}
    int overlap=under(root,resolved_source)||under(resolved_source,root);
    for(size_t i=0;i<selection->count;i++)if(under(selection->args[i].path,root))overlap=1;
    free(resolved_source);if(overlap){fail(&w,"LoRA cache must not overlap transformer or adapter sources");goto end;}
    /* Check the spelling too: an existing cache-root symlink is not accepted. */
    struct stat cache_stat;
    if(!lstat(selection->cache,&cache_stat)&&S_ISLNK(cache_stat.st_mode)){fail(&w,"LoRA cache root cannot be a symlink");goto end;}
    if(!private_dir(&w,root,1))goto end;
    locks=join(root,"locks");variants=join(root,"variants");staging=join(root,"staging");invalid=join(root,"invalid");
    if(!locks||!variants||!staging||!invalid)goto oom;
    if(!private_dir(&w,locks,1)||!private_dir(&w,variants,1)||!private_dir(&w,staging,1)||!private_dir(&w,invalid,1))goto end;
    char lockname[80];snprintf(lockname,sizeof(lockname),"%s.lock",key);lockpath=join(locks,lockname);entry=join(variants,key);
    if(!lockpath||!entry)goto oom;
    lock=open(lockpath,O_RDONLY|O_NOFOLLOW);
    if(lock<0&&errno==ENOENT)lock=open(lockpath,O_RDWR|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(lock<0&&errno==EEXIST)lock=open(lockpath,O_RDONLY|O_NOFOLLOW);
    struct stat lockstat;if(lock<0||fstat(lock,&lockstat)||!S_ISREG(lockstat.st_mode)||lockstat.st_uid!=geteuid()){
        fail(&w,"cannot open stable LoRA cache lock: %s",strerror(errno));goto end;}
    if(!lock_wait(&w,lock,LOCK_SH))goto end;
    double verify_start=now();
    if(verify_entry(&w,entry,key,mode)){hit=1;verified=now()-verify_start;goto ready;}
    if(w.cancelled||!checkpoint(&w,"LoRA cache miss",0,0))goto end;
    if(error&&cap)error[0]=0;
    /* flock upgrades need not be atomic. Explicitly drop/reacquire and recheck. */
    if(flock(lock,LOCK_UN)||!lock_wait(&w,lock,LOCK_EX))goto end;
    if(verify_entry(&w,entry,key,mode)){hit=1;goto downgrade;}
    if(w.cancelled||!checkpoint(&w,"LoRA preparing cache",0,0))goto end;
    if(error&&cap)error[0]=0;
    struct stat st;
    if(!lstat(entry,&st)){
        if(st.st_uid!=geteuid()){fail(&w,"refusing to repair cache entry owned by another user");goto end;}
        char name[112];snprintf(name,sizeof(name),"%s.%ld.%.0f",key,(long)getpid(),now()*1e9);char *quarantine=join(invalid,name);
        if(!injected(&w,"repair")||!quarantine||rename(entry,quarantine)){free(quarantine);fail(&w,"cannot quarantine invalid LoRA cache: %s",strerror(errno));goto end;}
        fprintf(stderr,"LoRA: quarantined invalid entry at %s; rebuilding once\n",quarantine);free(quarantine);
        if(!sync_dir(&w,variants)||!sync_dir(&w,invalid))goto end;
    }else if(errno!=ENOENT){fail(&w,"cannot inspect LoRA cache: %s",strerror(errno));goto end;}
    DIR *d=opendir(staging);if(!d){fail(&w,"cannot inspect staging directory");goto end;}struct dirent *e;
    while((e=readdir(d)))if(!strncmp(e->d_name,key,64)&&e->d_name[64]=='.'){
        char *p=join(staging,e->d_name);if(!p||!remove_tree(p)){free(p);closedir(d);fail(&w,"cannot clean interrupted LoRA staging entry");goto end;}free(p);
    }
    closedir(d);
    char pattern[80];snprintf(pattern,sizeof(pattern),"%s.XXXXXX",key);stage=join(staging,pattern);
    if(!stage||!mkdtemp(stage)){fail(&w,"cannot create LoRA staging: %s",strerror(errno));goto end;}
    outdir=join(stage,"transformer");if(!outdir||!private_dir(&w,outdir,1))goto end;
    /* Check enough logical capacity for a full fallback before copying. CoW
     * may still succeed when logical space is tight, so each fallback checks
     * its actual requirement immediately before any write. */
    double phase_start=now();
    for(size_t i=0;i<w.file_count;i++){
        if(!checkpoint(&w,"LoRA copying transformer",(int)i,(int)w.file_count))goto end;
        char *dest=join(outdir,w.files[i].name);int copied=dest&&copy_file(&w,&w.files[i],dest,0);free(dest);if(!copied)goto end;
    }
    /* A successful CoW clone still needs space when its target pages change. */
    uint64_t cow_reserve=UINT64_C(64)*1024*1024;
    for(size_t i=0;i<w.target_count;i++) {
        target *t=&w.targets[i];if(t->scale==0||!w.files[t->file].cloned)continue;
        int seen=0;for(size_t j=0;j<i;j++)if(w.targets[j].base==t->base&&w.targets[j].scale!=0)seen=1;
        if(seen)continue;
        uint64_t bytes=t->base->data_end-t->base->data_begin;
        if(bytes>UINT64_MAX-cow_reserve){fail(&w,"LoRA CoW disk plan overflows");goto end;}cow_reserve+=bytes;
    }
    struct statvfs free_space;
    int cow_fits=statvfs(outdir,&free_space)||cow_reserve<=(uint64_t)free_space.f_bavail*free_space.f_frsize;
#ifdef H3_LORA_TESTING
    if(getenv("H3_TEST_LORA_LOW_COW_SPACE"))cow_fits=0;
#endif
    if(!cow_fits) {
        /* Some quota-backed filesystems charge a clone's whole logical size.
         * Replace modified clones one shard at a time with checked full copies
         * instead of requiring a second model's worth of reported free space.
         * Only our unpublished staging files are removed; sources stay intact. */
        fprintf(stderr,"LoRA: insufficient reported CoW headroom; trying independent full copies\n");
        for(size_t i=0;i<w.file_count;i++) {
            int modified=0;
            for(size_t j=0;j<w.target_count;j++)if(w.targets[j].file==i&&w.targets[j].scale!=0)modified=1;
            if(!w.files[i].cloned||!modified)continue;
            if(!checkpoint(&w,"LoRA full-copy fallback",(int)i,(int)w.file_count))goto end;
            char *dest=join(outdir,w.files[i].name);
            if(!dest)goto oom;
            if(unlink(dest)){free(dest);fail(&w,"cannot replace staged CoW clone: %s",strerror(errno));goto end;}
            w.files[i].cloned=0;w.cloned_files--;
            int copied=copy_file(&w,&w.files[i],dest,1);free(dest);
            if(!copied)goto end;
        }
        if(!statvfs(outdir,&free_space)&&UINT64_C(64)*1024*1024>(uint64_t)free_space.f_bavail*free_space.f_frsize) {
            fail(&w,"insufficient cache space after full-copy fallback; choose another --lora-cache");goto end;
        }
    }
    w.copy_seconds=now()-phase_start;phase_start=now();
    for(size_t i=0;i<w.file_count;i++)for(size_t j=0;j<w.files[i].st.tensor_count;j++)
        if(!fold_tensor(&w,&w.files[i].st.tensors[j],i,outdir))goto end;
    w.fold_seconds=now()-phase_start;phase_start=now();
    for(size_t i=0;i<w.file_count;i++)if(!untouched(&w,i,outdir))goto end;
    if(!unchanged(&w,source)||!identity_write(&w,outdir,key,mode))goto end;
    w.validation_seconds=now()-phase_start;
    if(!manifest_write(&w,stage,key,mode)||!sync_dir(&w,outdir)||!sync_dir(&w,stage))goto end;
    if(!checkpoint(&w,"LoRA publishing",0,0))goto end;
    if(!source_stamps(&w)||!injected(&w,"rename"))goto end;
    if(rename(stage,entry)){fail(&w,"atomic LoRA publication failed: %s",strerror(errno));goto end;}published=1;
    if(!sync_dir(&w,variants)||!sync_dir(&w,staging))goto end;
downgrade:
    if(flock(lock,LOCK_UN)||!lock_wait(&w,lock,LOCK_SH))goto end;
    verify_start=now();if(!verify_entry(&w,entry,key,mode)){fail(&w,"LoRA cache verification failed after lease conversion");goto end;}verified=now()-verify_start;
ready:
    if(!source_stamps(&w))goto end;
    if(!checkpoint(&w,hit?"LoRA verified cache hit":"LoRA fold complete",1,1))goto end;
    {char *effective=join(entry,"transformer");if(!effective)goto oom;
    memset(variant,0,sizeof(*variant));variant->transformer=effective;variant->lease=lock;lock=-1;strcpy(variant->key,key);
    variant->cache_hit=hit;variant->hash_seconds=hashed;variant->verify_seconds=verified;variant->prepare_seconds=now()-start;variant->scratch_peak=w.scratch_peak;}
    if(!hit)fprintf(stderr,"LoRA phases: copy %.2fs (%zu cloned, %zu full copies), fold %.2fs, input/untouched validation %.2fs\n",
        w.copy_seconds,w.cloned_files,w.copied_files,w.fold_seconds,w.validation_seconds);
    fprintf(stderr,"LoRA %s: %s; verification %.2fs, preparation %.2fs, owned arithmetic/I/O scratch %zu bytes\n",mode?"Ref2VA":"FL2VA",hit?"verified cache hit":"fold published",verified,variant->prepare_seconds,w.scratch_peak);
    if(error&&cap)error[0]=0;
    ok=1;goto end;
oom:fail(&w,"out of memory preparing LoRA cache");
end:
    if(!ok&&error&&cap&&!error[0])snprintf(error,cap,"runtime LoRA preparation failed");
    if(stage&&!published)remove_tree(stage);
    if(lock>=0)close(lock);
    free(root);free(locks);free(variants);free(staging);free(invalid);free(entry);free(lockpath);free(stage);free(outdir);work_free(&w);
    if(numeric_locale){uselocale(previous_locale);freelocale(numeric_locale);}
    if(env_saved)fesetenv(&env);
#if defined(__SSE__)
    _mm_setcsr(mxcsr);
#endif
#if defined(__aarch64__)
    __asm__ volatile("msr fpcr, %0"::"r"(fpcr));
#endif
    return ok;
}

char *h3_lora_provenance(const h3_lora_selection *s,const h3_lora_variant *v,const char *base,int mode) {
    if(!s||!v||!v->transformer)return NULL;
    char *text=NULL;size_t size=0;FILE *f=open_memstream(&text,&size);if(!f)return NULL;
    fprintf(f,"{\"schema\":1,\"informational_only\":true,\"runtime_fold_key\":\"%s\",\"mode\":\"%s\",\"base\":",v->key,mode?"Ref2VA":"FL2VA");
    lj_quote(f,base);fputs(",\"adapters\":[",f);
    for(size_t i=0;i<s->count;i++){
        if(i)fputc(',',f);
        fputs("{\"path\":",f);lj_quote(f,s->args[i].path);
        uint64_t bits;memcpy(&bits,&s->args[i].scale,8);
        fprintf(f,",\"scale_binary64\":\"%016llx\"}",(unsigned long long)bits);
    }
    fputs("]}\n",f);int ok=!ferror(f);if(fclose(f))ok=0;
    if(!ok){free(text);return NULL;}return text;
}
int h3_lora_save_provenance(const char *state_path,const char *json,char *error,size_t cap) {
    if(!json)return 1;
    work w={.error=error,.error_size=cap};if(!state_path||!*state_path)return fail(&w,"missing LoRA provenance state path");size_t n=strlen(state_path)+48;
    char *path=malloc(n),*tmp=malloc(n);if(!path||!tmp){free(path);free(tmp);return fail(&w,"out of memory saving LoRA provenance");}
    snprintf(path,n,"%s.lora.json",state_path);snprintf(tmp,n,"%s.XXXXXX",path);
    int fd=mkstemp(tmp),ok=0;
    if(fd<0){fail(&w,"cannot stage LoRA provenance: %s",strerror(errno));goto end;}
    FILE *f=fdopen(fd,"w");if(!f){close(fd);fail(&w,"cannot open LoRA provenance");goto end;}
    int written=fputs(json,f)!=EOF;ok=finish_file(&w,f)&&written;
    if(ok&&rename(tmp,path)){ok=fail(&w,"cannot commit LoRA provenance: %s",strerror(errno));}
end:unlink(tmp);free(path);free(tmp);return ok;
}
