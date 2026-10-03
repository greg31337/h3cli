#import <Foundation/Foundation.h>
#import <CommonCrypto/CommonDigest.h>
#import <os/signpost.h>
#include "src/metal/ane.h"
#include "src/metal/ane_split.h"
#include "src/profile.h"
#include <math.h>
#include <time.h>

@interface H3ANESplit : NSObject {
@public
    h3_metal_attention_options options;
    uint64_t decided;
    uint32_t rowsByBlock[50];
    char weightHash[50][65];
}
@property(nonatomic) h3_ane *engine;
@property(nonatomic) int k,n,nextRows;
@property(nonatomic) double gpuRowSeconds;
@property(nonatomic,strong) dispatch_queue_t worker;
@end
@implementation H3ANESplit
- (void)dealloc {h3_ane_free(_engine);}
@end
static double stamp(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}
static os_log_t trace_log(void) {
    static os_log_t log;static dispatch_once_t once;
    dispatch_once(&once,^{log=os_log_create("org.h3.ane","qkv");});return log;
}
h3_ane_split *h3_ane_split_create(h3_metal_attention_options options,int k,int n) {
    if(!h3_metal_options_valid(options,NULL,0)||!options.ane_mode||k<1||n<1)return NULL;
    H3ANESplit *s=[H3ANESplit new];s->options=options;s.k=k;s.n=n;s.nextRows=options.ane_rows;
    s.worker=dispatch_queue_create("org.h3.ane.worker",DISPATCH_QUEUE_SERIAL);
    h3_ane_stats stats={0};char error[4096]={0};
    s.engine=h3_ane_create(options.ane_chunk,k,n,1024,getenv("H3_ANE_CACHE_DIR"),&stats,error,sizeof(error));
    if(h3_profile_steps_enabled())fprintf(stderr,"h3_ane_setup {\"recipe\":1,\"mode\":%d,\"chunk\":%d,\"k\":%d,\"n\":%d,\"ready\":%s,\"cache_hit\":%s,\"load_seconds\":%.6f,\"ane_matmuls\":%d,\"total_matmuls\":%d,\"array_bytes\":%zu,\"system_wired_bytes\":%llu,\"process_footprint_bytes\":%llu}\n",
        options.ane_mode,options.ane_chunk,k,n,s.engine?"true":"false",stats.cache_hit?"true":"false",stats.load_seconds,stats.ane_matmuls,stats.total_matmuls,stats.memory_bytes,(unsigned long long)stats.system_wired_bytes,(unsigned long long)stats.process_footprint_bytes);
    if(!s.engine)fprintf(stderr,"h3cli: ANE unavailable; retaining GPU QKV: %s\n",error);
    return (__bridge_retained h3_ane_split *)s;
}
void h3_ane_split_free(h3_ane_split *split){if(split){H3ANESplit *s=CFBridgingRelease(split);(void)s;}}
void h3_ane_split_export(h3_ane_split *split,uint64_t *decided,uint32_t rows[50]) {
    H3ANESplit *s=(__bridge H3ANESplit *)split;if(!s){*decided=0;memset(rows,0,50*sizeof(*rows));return;}
    *decided=s->decided;memcpy(rows,s->rowsByBlock,sizeof(s->rowsByBlock));
}
int h3_ane_split_restore(h3_ane_split *split,uint64_t decided,const uint32_t rows[50]) {
    H3ANESplit *s=(__bridge H3ANESplit *)split;
    if(!s)return decided==0;
    if(!rows||(decided>>50))return 0;
    /* Resume cannot silently substitute GPU arithmetic for recorded ANE rows. */
    for(unsigned b=0;b<50;b++)if(rows[b]&&(!s.engine||!(decided&(1ull<<b))||
        rows[b]>(unsigned)s->options.ane_rows||rows[b]%(unsigned)s->options.ane_chunk))return 0;
    s->decided=decided;memcpy(s->rowsByBlock,rows,sizeof(s->rowsByBlock));return 1;
}
int h3_ane_split_linear(h3_ane_split *opaque,h3_gpu *gpu,h3_gpu_tensor *out,
    h3_gpu_tensor *input,h3_gpu_tensor *weight,unsigned rows,unsigned block,int step) {
    H3ANESplit *s=(__bridge H3ANESplit *)opaque;
    if(!s||!gpu||block>=50||!rows||!out||!input||!weight||
       h3_gpu_tensor_dtype(out)!=H3_GPU_BF16||h3_gpu_tensor_dtype(input)!=H3_GPU_BF16||h3_gpu_tensor_dtype(weight)!=H3_GPU_BF16||
       h3_gpu_tensor_elements(out)<(size_t)rows*s.n||h3_gpu_tensor_elements(input)<(size_t)rows*s.k||
       h3_gpu_tensor_elements(weight)<(size_t)s.k*s.n)return 0;
    BOOL known=(s->decided&(1ull<<block))!=0;
    unsigned chunk=(unsigned)s->options.ane_chunk;
    unsigned budget=MIN((unsigned)s->options.ane_rows,(rows/2/chunk)*chunk);
    BOOL probe=!known&&s.engine&&s->options.ane_mode==3&&block%8==0;
    unsigned ar=known?s->rowsByBlock[block]:!s.engine||probe?0:
        MIN(budget,(unsigned)(s->options.ane_mode==3?s.nextRows:s->options.ane_rows));
    if(ar>budget)return 0;
    if(!known){s->decided|=1ull<<block;s->rowsByBlock[block]=ar;}
    double begin=stamp();
    if(!ar&&!probe) {
        if(h3_profile_steps_enabled())fprintf(stderr,"h3_ane {\"step\":%d,\"block\":%u,\"mode\":%d,\"rows\":%u,\"ane_rows\":0,\"frozen\":%s,\"fallback\":%s}\n",step,block,s->options.ane_mode,rows,known?"true":"false",s.engine?"false":"true");
        return h3_gpu_linear_bf16(gpu,out,input,weight,NULL,rows,(unsigned)s.k,(unsigned)s.n);
    }
    /* Materialize the ready AdaLN input. Only independent row shards overlap;
     * norm/RoPE/attention is encoded after both shards have completed. */
    if(!h3_gpu_submit(gpu))return 0;
    double ready=stamp();
    if(probe) {
        if(!h3_gpu_begin(gpu)||!h3_gpu_linear_bf16(gpu,out,input,weight,NULL,rows,(unsigned)s.k,(unsigned)s.n)||!h3_gpu_submit(gpu))return 0;
        double elapsed=stamp()-ready;s.gpuRowSeconds=elapsed/rows;
        if(h3_profile_steps_enabled())fprintf(stderr,"h3_ane {\"step\":%d,\"block\":%u,\"mode\":3,\"rows\":%u,\"ane_rows\":0,\"probe\":true,\"gpu_wait_seconds\":%.6f,\"drain_seconds\":%.6f,\"region_seconds\":%.6f}\n",step,block,rows,elapsed,ready-begin,elapsed);
        return h3_gpu_begin(gpu);
    }
    const uint16_t *x=h3_gpu_tensor_contents(input),*w=h3_gpu_tensor_contents(weight);
    uint16_t *destination=h3_gpu_tensor_contents(out);
    if(!x||!w||!destination)return 0;
    if(!s->weightHash[block][0]) {
        unsigned char hash[32];CC_SHA256(w,(CC_LONG)((size_t)s.k*s.n*2),hash);
        for(unsigned i=0;i<32;i++)snprintf(s->weightHash[block]+i*2,3,"%02x",hash[i]);
    }
    uint16_t *pending=malloc((size_t)ar*s.n*2);
    if(!pending) {
        if(h3_profile_steps_enabled())fprintf(stderr,"h3_ane {\"step\":%d,\"block\":%u,\"mode\":%d,\"rows\":%u,\"ane_rows\":%u,\"fallback\":true,\"reason\":\"allocation\"}\n",step,block,s->options.ane_mode,rows,ar);
        else fprintf(stderr,"h3cli: ANE staging allocation failed; retaining GPU QKV\n");
        return h3_gpu_begin(gpu)&&h3_gpu_linear_bf16(gpu,out,input,weight,NULL,rows,(unsigned)s.k,(unsigned)s.n);
    }
    __block int aneOK=0;__block h3_ane_stats stats={0};char errorStorage[4096]={0};char *error=errorStorage;
    dispatch_group_t group=dispatch_group_create();
    dispatch_group_async(group,s.worker,^{@autoreleasepool {
        os_signpost_id_t id=os_signpost_id_generate(trace_log());
        os_signpost_interval_begin(trace_log(),id,"ANE QKV", "step=%d block=%u rows=%u",step,block,ar);
        aneOK=h3_ane_predict(s.engine,x+(size_t)(rows-ar)*s.k,w,pending,(int)ar,&stats,error,sizeof(errorStorage));
        os_signpost_interval_end(trace_log(),id,"ANE QKV", "%s","complete");
    }});
    if(s->options.ane_mode==1)dispatch_group_wait(group,DISPATCH_TIME_FOREVER);
    double gb=stamp();h3_gpu_stats before={0},after={0};h3_gpu_get_stats(gpu,&before);
    int ok=h3_gpu_begin(gpu)&&h3_gpu_linear_bf16(gpu,out,input,weight,NULL,rows-ar,(unsigned)s.k,(unsigned)s.n)&&h3_gpu_submit(gpu);
    double ge=stamp();h3_gpu_get_stats(gpu,&after);
    dispatch_group_wait(group,DISPATCH_TIME_FOREVER);
    double joined=stamp();
    if(aneOK&&ok)memcpy(destination+(size_t)(rows-ar)*s.n,pending,(size_t)ar*s.n*2);
    free(pending);
    if(!ok)return 0;
    double region=stamp()-ready;
    if(s->options.ane_mode==3&&!known) {
        double gpuRate=s.gpuRowSeconds>0?s.gpuRowSeconds:(ge-gb)/(rows-ar);
        double aneRate=(stats.pack_seconds+stats.predict_seconds+stats.unpack_seconds)/ar;
        unsigned ideal=aneRate>0?(unsigned)((double)rows*gpuRate/(gpuRate+aneRate)):0;
        ideal=MIN(budget,(ideal/chunk)*chunk);
        /* Adapt at most one chunk between neighboring first-encounter blocks.
         * Decisions are frozen per block and serialized for exact resumption. */
        unsigned next=ideal>ar?MIN(ideal,ar+chunk):MAX(ideal,ar>chunk?ar-chunk:0);
        if(!aneOK||region>gpuRate*rows*1.03)next=0;
        s.nextRows=(int)next;
    }
    if(h3_profile_steps_enabled())fprintf(stderr,"h3_ane {\"step\":%d,\"block\":%u,\"mode\":%d,\"rows\":%u,\"ane_rows\":%u,\"frozen\":%s,\"weight_sha256\":\"%s\",\"weight_scale_exponent\":%d,\"input_scale_exponent_min\":%d,\"input_scale_exponent_max\":%d,\"system_wired_bytes\":%llu,\"process_footprint_bytes\":%llu,\"pack_seconds\":%.6f,\"predict_seconds\":%.6f,\"unpack_seconds\":%.6f,\"gpu_seconds\":%.6f,\"gpu_wait_seconds\":%.6f,\"drain_seconds\":%.6f,\"region_seconds\":%.6f,\"join_seconds\":%.6f,\"host_predict_overlap_seconds\":%.6f,\"next_rows\":%d,\"fallback\":%s}\n",
        step,block,s->options.ane_mode,rows,ar,known?"true":"false",s->weightHash[block],stats.weight_scale_exponent,stats.input_scale_exponent_min,stats.input_scale_exponent_max,(unsigned long long)stats.system_wired_bytes,(unsigned long long)stats.process_footprint_bytes,stats.pack_seconds,stats.predict_seconds,stats.unpack_seconds,
        after.gpu_seconds-before.gpu_seconds,ge-gb,ready-begin,region,joined-ge,
        fmax(0,fmin(ge,stats.predict_end)-fmax(gb,stats.predict_begin)),s.nextRows,aneOK?"false":"true");
    if(!h3_gpu_begin(gpu))return 0;
    if(!aneOK) {
        fprintf(stderr,"h3cli: ANE QKV recovery step %d block %u: %s\n",step,block,error);
        return h3_gpu_linear_bf16(gpu,out,input,weight,NULL,rows,(unsigned)s.k,(unsigned)s.n);
    }
    return 1;
}
