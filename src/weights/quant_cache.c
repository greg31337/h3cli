#include "src/weights/quant_cache.h"
#include "src/weights/quant.h"
#include "src/digest.h"
#include "src/platform.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/file.h>
#include <unistd.h>

static int fail(char *error,size_t size,const char *message,const char *path) {
    if(error&&size)snprintf(error,size,"%s: %s",message,path?path:strerror(errno));
    return 0;
}
static void put(unsigned char *p,uint64_t v,unsigned n) {
    for(unsigned i=0;i<n;i++)p[i]=(unsigned char)(v>>(8*i));
}
static int same(const struct stat *a,const struct stat *b) {
    struct timespec am=h3_stat_mtime(a),bm=h3_stat_mtime(b),ac=h3_stat_ctime(a),bc=h3_stat_ctime(b);
    return S_ISREG(b->st_mode)&&a->st_dev==b->st_dev&&a->st_ino==b->st_ino&&a->st_size==b->st_size&&
        am.tv_sec==bm.tv_sec&&am.tv_nsec==bm.tv_nsec&&ac.tv_sec==bc.tv_sec&&ac.tv_nsec==bc.tv_nsec;
}
static int io(int fd,void *data,size_t bytes,int writing) {
    unsigned char *p=data;
    while(bytes) {
        ssize_t n=writing?write(fd,p,bytes):read(fd,p,bytes);
        if(n<0&&errno==EINTR)continue;
        if(n<=0)return 0;
        p+=(size_t)n;bytes-=(size_t)n;
    }
    return 1;
}
static void header(const h3_quant_cache *c,unsigned char h[H3_QUANT_CACHE_HEADER]) {
    memset(h,0,H3_QUANT_CACHE_HEADER);memcpy(h,"H3QMT002",8);
    put(h+8,H3_QUANT_CACHE_VERSION,4);put(h+12,H3_QUANT_VERSION,4);
    put(h+16,c->mode,4);put(h+20,c->rows,4);put(h+24,c->columns,4);
    put(h+32,c->bytes,8);put(h+40,c->scale,8);put(h+48,c->global,8);
    memcpy(h+56,c->key,32);put(h+88,c->offset,8);
}
int h3_quant_cache_unchanged(const h3_quant_cache *c,char *error,size_t size) {
    struct stat st;
    if(stat(c->source,&st)||!same(&c->stamp,&st))
        return fail(error,size,"quantization source changed during loading/preparation; retry",c->source);
    return 1;
}
void h3_quant_cache_close(h3_quant_cache *c) {
    if(!c)return;
    if(c->lock>=0){flock(c->lock,LOCK_UN);close(c->lock);}
    free(c->source);free(c->artifact);free(c);
}
h3_quant_cache *h3_quant_cache_open(const char *directory,const char *source,
    uint64_t offset,unsigned rows,unsigned columns,int mode,
    uint64_t bytes,uint64_t scale,uint64_t global,char *error,size_t size) {
    h3_quant_cache *c=calloc(1,sizeof(*c));
    if(!c){fail(error,size,"cannot allocate quantization metadata",NULL);return NULL;}
    c->lock=-1;
    if(!directory||!*directory||!source||!rows||!columns||rows%128||columns%128||
       (uint64_t)rows*columns>UINT32_MAX||(mode!=H3_QUANT_FP8&&mode!=H3_QUANT_NVFP4)||
       !bytes||bytes>SIZE_MAX-H3_QUANT_CACHE_HEADER||scale>=global||global>bytes||bytes-global<16) {
        fail(error,size,"invalid quantization metadata arguments",source);goto failed;
    }
    c->source=realpath(source,NULL);
    if(!c->source||stat(c->source,&c->stamp)||!S_ISREG(c->stamp.st_mode)||c->stamp.st_size<0||
       offset>(uint64_t)c->stamp.st_size||(uint64_t)rows*columns*2>(uint64_t)c->stamp.st_size-offset) {
        fail(error,size,"invalid quantized source range",source);goto failed;
    }
    c->offset=offset;c->rows=rows;c->columns=columns;c->mode=(unsigned)mode;
    c->bytes=bytes;c->scale=scale;c->global=global;
    /* Hash only a small identity record and canonical pathname. In particular,
     * do not open, mmap, or hash the original weight payload on a cache hit. */
    unsigned char identity[112]={0};struct timespec mt=h3_stat_mtime(&c->stamp),ct=h3_stat_ctime(&c->stamp);
    const uint64_t values[]={H3_QUANT_CACHE_VERSION,H3_QUANT_VERSION,(uint64_t)mode,
        rows,columns,offset,(uint64_t)c->stamp.st_dev,(uint64_t)c->stamp.st_ino,
        (uint64_t)c->stamp.st_size,(uint64_t)mt.tv_sec,(uint64_t)mt.tv_nsec,(uint64_t)ct.tv_sec,(uint64_t)ct.tv_nsec,bytes};
    for(size_t i=0;i<sizeof(values)/sizeof(*values);i++)put(identity+8*i,values[i],8);
    h3_sha256_ctx hash;h3_sha256_init_fast(&hash);
    h3_sha256_update(&hash,identity,sizeof(identity));
    h3_sha256_update(&hash,c->source,(h3_sha256_size)strlen(c->source)+1);
    h3_sha256_final(c->key,&hash);
    char key[65];for(size_t i=0;i<32;i++)snprintf(key+2*i,3,"%02x",c->key[i]);
    if(asprintf(&c->artifact,"%s/m%d-%s-%ux%u-%s.h3q",directory,H3_QUANT_CACHE_VERSION,
                h3_quant_name(mode),rows,columns,key)<0)c->artifact=NULL;
    char *lock=NULL;
    if(!c->artifact||asprintf(&lock,"%s.lock",c->artifact)<0) {
        fail(error,size,"cannot allocate quantization cache path",source);goto failed;
    }
    c->lock=open(lock,O_CREAT|O_RDWR,0600);free(lock);
    if(c->lock<0||flock(c->lock,LOCK_EX)) {
        fail(error,size,"cannot lock quantization artifact",c->artifact);goto failed;
    }
    if(!h3_quant_cache_unchanged(c,error,size))goto failed;
    return c;
failed:
    h3_quant_cache_close(c);return NULL;
}
int h3_quant_cache_read(h3_quant_cache *c,void *payload,char *error,size_t size) {
    int fd=open(c->artifact,O_RDONLY);
    if(fd<0) {
        if(errno==ENOENT)return h3_quant_cache_unchanged(c,error,size)?0:-1;
        fail(error,size,"cannot read quantization artifact",c->artifact);return -1;
    }
    struct stat before,after;unsigned char wanted[H3_QUANT_CACHE_HEADER],actual[H3_QUANT_CACHE_HEADER];header(c,wanted);
    int ok=!fstat(fd,&before)&&S_ISREG(before.st_mode)&&before.st_size>=0&&
        (uint64_t)before.st_size==H3_QUANT_CACHE_HEADER+c->bytes&&
        io(fd,actual,sizeof(actual),0)&&!memcmp(actual,wanted,sizeof(actual))&&
        io(fd,payload,(size_t)c->bytes,0)&&!fstat(fd,&after)&&same(&before,&after);
    close(fd);
    if(!ok){fail(error,size,"corrupt/incompatible metadata quantization artifact; remove it and prepare again",c->artifact);return -1;}
    return h3_quant_cache_unchanged(c,error,size)?1:-1;
}
int h3_quant_cache_write(h3_quant_cache *c,const void *payload,char *error,size_t size) {
    if(!h3_quant_cache_unchanged(c,error,size))return 0;
    char *temporary=NULL;
    if(asprintf(&temporary,"%s.tmp.XXXXXX",c->artifact)<0)
        return fail(error,size,"cannot allocate quantization staging path",c->artifact);
    int fd=mkstemp(temporary);unsigned char h[H3_QUANT_CACHE_HEADER];header(c,h);
    int ok=fd>=0;
    if(ok) {
        ok=io(fd,h,sizeof(h),1)&&io(fd,(void*)payload,(size_t)c->bytes,1)&&!fsync(fd);
        if(close(fd))ok=0;
    }
    if(ok)ok=h3_quant_cache_unchanged(c,error,size);
    if(ok)ok=!rename(temporary,c->artifact);
    if(!ok){unlink(temporary);fail(error,size,"cannot publish quantization artifact (source changed or I/O failure)",c->artifact);}
    free(temporary);return ok;
}
