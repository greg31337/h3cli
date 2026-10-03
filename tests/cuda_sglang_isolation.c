/* Source-independent fast fixture: also build against the pre-change library.
 * The optional reference path stresses context/TLS isolation in one process. */
#include "src/gpu.h"
#include "src/execution.h"
#ifdef H3_CUDA_USE_SGLANG_FLASH
#include "src/sglang/sglang.h"
#endif
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static char error[1024];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error);exit(1);}}while(0)
enum {R=257,K=128,N=256,E=R*K};
static uint16_t *evaluate(int reference) {
#ifdef H3_CUDA_USE_SGLANG_FLASH
    int old=h3_sglang_exchange(reference);
#endif
    h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
#ifdef H3_CUDA_USE_SGLANG_FLASH
    h3_sglang_exchange(old);
#endif
    uint16_t *input=malloc(E*2),*weights=malloc(K*N*2),bias[N],scale[K],*gold=malloc(E*2),*got=malloc(E*2);
    CHECK(input&&weights&&gold&&got);
    for(int j=0;j<E;j++)input[j]=(uint16_t)(0x3d00+j%256);
    for(int j=0;j<K*N;j++)weights[j]=(uint16_t)(0x3c00+(j*13)%128);
    for(int j=0;j<N;j++)bias[j]=(uint16_t)(0x3b80+j%64);
    for(int j=0;j<K;j++)scale[j]=0x3f80;
    h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,input,E),*w=h3_gpu_tensor_from_bf16(g,weights,K*N),
        *b=h3_gpu_tensor_from_bf16(g,bias,N),*s=h3_gpu_tensor_from_bf16(g,scale,K),
        *y=h3_gpu_tensor_new_bf16(g,R*N),*a=h3_gpu_tensor_new_bf16(g,E),
        *n=h3_gpu_tensor_new_bf16(g,E),*o=h3_gpu_tensor_new_bf16(g,E);
    CHECK(x&&w&&b&&s&&y&&a&&n&&o);uint64_t plateau=0;
    for(int repeat=0;repeat<4;repeat++) {
        CHECK(h3_gpu_begin(g));
        int ok=h3_gpu_linear_bf16(g,y,x,w,b,R,K,N)&&h3_gpu_swiglu_bf16(g,a,y,R,K)&&
            h3_gpu_rms_norm_bf16(g,n,a,s,R,K,1e-5f)&&h3_gpu_sdpa_bf16(g,o,n,n,n,R,1,K,1.f/sqrtf(K))&&h3_gpu_submit(g);
        if (!ok)
            snprintf(error, sizeof(error), "%s", h3_gpu_error(g));
        CHECK(ok);
        CHECK(h3_gpu_tensor_read_bf16(o,got,E));
        for(int j=0;j<E;j++)CHECK((got[j]&0x7f80)!=0x7f80);
        if(!repeat)memcpy(gold,got,E*2);else CHECK(!memcmp(gold,got,E*2));
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
        if(!repeat)plateau=stats.live_bytes;else CHECK(stats.live_bytes==plateau);
        /* Invalid arguments and cancellation cannot poison the next submit. */
        CHECK(h3_gpu_begin(g));CHECK(!h3_gpu_linear_bf16(g,y,x,w,b,R,K,N+1));h3_gpu_cancel(g);
    }
    h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);h3_gpu_tensor_free(b);h3_gpu_tensor_free(s);
    h3_gpu_tensor_free(y);h3_gpu_tensor_free(a);h3_gpu_tensor_free(n);h3_gpu_tensor_free(o);h3_gpu_free(g);
    free(input);free(weights);free(got);return gold;
}
#ifdef H3_CUDA_USE_SGLANG_FLASH
static void host_weight_cache(void) {
    char path[]="/tmp/h3-reference-weight-XXXXXX";
    int fd=mkstemp(path);CHECK(fd>=0);
    uint16_t original[4096],changed[4096],got[4096];
    for(int i=0;i<4096;i++){original[i]=(uint16_t)(0x3c80+i%128);changed[i]=(uint16_t)(0x3d80+i%128);}
    CHECK(write(fd,original,sizeof(original))==sizeof(original));CHECK(!fsync(fd));
    int old=h3_sglang_exchange(1);h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));
    h3_sglang_exchange(old);CHECK(g);
    h3_gpu_tensor *t=h3_gpu_tensor_new_bf16(g,4096);CHECK(t);
    h3_gpu_stats initial,loaded;CHECK(h3_gpu_get_stats(g,&initial));
    for(int repeat=0;repeat<4;repeat++) {
        CHECK(h3_gpu_sglang_preload_weight(g,path,0,4096));
        CHECK(h3_gpu_tensor_stream_file_bf16(t,path,0,4096,error,sizeof(error)));
        CHECK(h3_gpu_tensor_read_bf16(t,got,4096));CHECK(!memcmp(got,original,sizeof(got)));
        CHECK(h3_gpu_get_stats(g,&loaded));CHECK(loaded.pinned_bytes==initial.pinned_bytes+sizeof(got));
    }
    CHECK(pwrite(fd,changed,sizeof(changed),0)==sizeof(changed));CHECK(!fsync(fd));
    CHECK(h3_gpu_tensor_stream_file_bf16(t,path,0,4096,error,sizeof(error)));
    CHECK(h3_gpu_tensor_read_bf16(t,got,4096));CHECK(!memcmp(got,changed,sizeof(got)));
    CHECK(h3_gpu_release_weight_cache(g));CHECK(h3_gpu_get_stats(g,&loaded));CHECK(loaded.pinned_bytes==initial.pinned_bytes);
    /* Optional cache admission failure must leave normal streaming usable. */
    char *previous=getenv("H3_MEMORY_LIMIT_BYTES");previous=previous?strdup(previous):NULL;
    CHECK(!setenv("H3_MEMORY_LIMIT_BYTES","1",1));
    CHECK(h3_gpu_sglang_preload_weight(g,path,0,4096));
    CHECK(h3_gpu_tensor_stream_file_bf16(t,path,0,4096,error,sizeof(error)));
    CHECK(h3_gpu_tensor_read_bf16(t,got,4096));CHECK(!memcmp(got,changed,sizeof(got)));
    CHECK(h3_gpu_get_stats(g,&loaded));CHECK(loaded.pinned_bytes==initial.pinned_bytes);
    if(previous){CHECK(!setenv("H3_MEMORY_LIMIT_BYTES",previous,1));free(previous);}else CHECK(!unsetenv("H3_MEMORY_LIMIT_BYTES"));
    h3_gpu_tensor_free(t);h3_gpu_free(g);CHECK(!close(fd));CHECK(!unlink(path));
    puts("reference host weight hits, stat invalidation, release and pressure fallback passed");
}
static void decoder_failure_recovery(void) {
    int old=h3_sglang_exchange(1);h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));
    h3_sglang_exchange(old);CHECK(g);CHECK(h3_gpu_video_vae_configure(g,0)>=0);
    float input[K],weights[K*N];for(int i=0;i<K;i++)input[i]=100000.f;
    for(int i=0;i<K*N;i++)weights[i]=.01f;
    h3_gpu_tensor *x=h3_gpu_tensor_from_f32(g,input,K),*w=h3_gpu_tensor_from_f32(g,weights,K*N),
        *y=h3_gpu_tensor_new_f32(g,N),*half=h3_gpu_sglang_vae_weight(g,w);
    CHECK(x&&w&&y&&half);h3_gpu_tensor_free(w);
    CHECK(h3_gpu_begin(g));CHECK(h3_gpu_video_linear(g,y,x,half,NULL,1,K,N));
    CHECK(!h3_gpu_submit(g));CHECK(strstr(h3_gpu_error(g),"nonfinite"));h3_gpu_cancel(g);
    uint64_t plateau=0;
    for(int repeat=0;repeat<4;repeat++) {
        for(int i=0;i<K;i++)input[i]=.1f;
        CHECK(h3_gpu_tensor_write_f32(x,input,K));CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_video_linear(g,y,x,half,NULL,1,K,N));CHECK(h3_gpu_submit(g));
        float out[N];CHECK(h3_gpu_tensor_read_f32(y,out,N));
        for(int i=0;i<N;i++)CHECK(isfinite(out[i])&&out[i]>.1f&&out[i]<.2f);
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
        if(!repeat)plateau=stats.live_bytes;else CHECK(stats.live_bytes==plateau);
    }
    h3_gpu_tensor_free(x);h3_gpu_tensor_free(half);h3_gpu_tensor_free(y);h3_gpu_free(g);
    puts("reference FP16 overflow detection, cancellation and recovery passed");
}
#endif
int main(int argc,char **argv) {
    CHECK(argc==2);uint16_t *fast=NULL,*ref=NULL;
    /* Both reference→fast→reference and fast→reference→fast are exercised. */
    for(int i=0;i<5;i++) {
        int reference=i%2;
#ifndef H3_CUDA_USE_SGLANG_FLASH
        if(reference)continue;
#endif
        uint16_t *out=evaluate(reference),**gold=reference?&ref:&fast;
        if(*gold){CHECK(!memcmp(*gold,out,E*2));free(out);}else *gold=out;
    }
    FILE *f=fopen(argv[1],"wbx");CHECK(f&&fast);CHECK(fwrite(fast,2,E,f)==E);CHECK(!fclose(f));
#ifdef H3_CUDA_USE_SGLANG_FLASH
    decoder_failure_recovery();
    host_weight_cache();
#endif
    free(fast);free(ref);puts("fast fixture and mixed-context isolation passed");return 0;
}
