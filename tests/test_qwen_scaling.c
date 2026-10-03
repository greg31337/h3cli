/* Independent scalar FP32 oracle; long cases sample query rows, GPU runs all. */
#include "src/gpu.h"
#include "src/device.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while (0)
static float fp(uint16_t b) { uint32_t u=(uint32_t)b<<16; float f; memcpy(&f,&u,4); return f; }
static uint16_t bf(float f) { uint32_t u; memcpy(&u,&f,4); return (uint16_t)((u+0x7fff+((u>>16)&1))>>16); }
static uint32_t rng=72;
static float random_value(void) { rng^=rng<<13; rng^=rng>>17; rng^=rng<<5; return (float)(rng&65535)/32768.0f-1; }
static double now(void) { struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t); return (double)t.tv_sec+(double)t.tv_nsec*1e-9; }
static const char *modes[]={"legacy","scaled-q","reference"};
int main(int argc,char **argv) {
    char error[512]; unsetenv("H3_MPS_GQA");
#ifndef SCALING_BASELINE
    h3_qwen_gqa_scale_mode mode;
    CHECK(h3_qwen_gqa_scale_parse(NULL,&mode,error,sizeof(error)));
    CHECK(mode==H3_QWEN_GQA_REFERENCE);
    CHECK(h3_qwen_gqa_scale_parse("",&mode,error,sizeof(error)));
    CHECK(mode==H3_QWEN_GQA_REFERENCE);
    for(int i=0;i<3;i++) { CHECK(h3_qwen_gqa_scale_parse(modes[i],&mode,error,sizeof(error))); CHECK(!strcmp(modes[i],h3_qwen_gqa_scale_name(mode))); }
    CHECK(!h3_qwen_gqa_scale_parse("REFERENCE",&mode,error,sizeof(error)) && strstr(error,"H3_QWEN_GQA_SCALE_MODE"));
#endif
    h3_device_info device; CHECK(h3_device_query(&device,error,sizeof(error)));
    h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error)); CHECK(gpu);
    const struct { const char *name; unsigned n,kind; } cases[]={
        {"small",7,0},{"medium",128,0},{"race-662",662,0},{"long",2048,0},{"near-tied",65,1},
        {"dominant",65,2},{"large",65,3},{"bf16-boundary",65,4}};
    for(size_t c=0;c<sizeof(cases)/sizeof(*cases);c++) {
        unsigned n=cases[c].n, H=64,K=8,D=128; float scale=1/sqrtf((float)D);
        size_t nq=(size_t)n*H*D,nk=(size_t)n*K*D;
        uint16_t *q=malloc(nq*2),*k=malloc(nk*2),*v=malloc(nk*2),*out=malloc(nq*2),*previous=malloc(nq*2);
        float *want=calloc(nq,4),*scores=malloc(n*4); CHECK(q&&k&&v&&out&&previous&&want&&scores);rng=72;
        for(size_t i=0;i<nq;i++) q[i]=bf(random_value()*2);
        for(size_t i=0;i<nk;i++) { k[i]=bf(random_value()*2); v[i]=bf(random_value()); }
        if(cases[c].kind==1) { for(size_t i=0;i<nk;i++) k[i]=bf(0.5f+random_value()*.002f); }
        if(cases[c].kind==2) { for(size_t i=0;i<nq;i++) q[i]=bf(8+random_value()); for(size_t i=0;i<nk;i++) k[i]=bf((i/(K*D)%7==0?8:-8)+random_value()); }
        if(cases[c].kind==3) { for(size_t i=0;i<nq;i++) q[i]=bf(fp(q[i])*64);for(size_t i=0;i<nk;i++) k[i]=bf(fp(k[i])*64); }
        if(cases[c].kind==4) { for(size_t i=0;i<nq;i++) q[i]=bf((i%2?1:-1)*(1.00390625f+random_value()*.008f)); }
        unsigned stride=n>128?n/16:1;
        for(unsigned row=0;row<n;row++) {
            if(row%stride && row!=n-1) continue;
            for(unsigned h=0;h<H;h++) {
                unsigned kh=h/(H/K); float maximum=-INFINITY,sum=0;
                for(unsigned j=0;j<=row;j++) {
                    float dot=0;
                    for(unsigned d=0;d<D;d++) dot=fmaf(fp(q[((size_t)row*H+h)*D+d]),fp(k[((size_t)j*K+kh)*D+d]),dot);
                    scores[j]=dot*scale; maximum=fmaxf(maximum,scores[j]);
                }
                for(unsigned j=0;j<=row;j++) { scores[j]=expf(scores[j]-maximum);sum+=scores[j]; }
                for(unsigned d=0;d<D;d++) {
                    float result=0;
                    for(unsigned j=0;j<=row;j++) result=fmaf(scores[j]/sum,fp(v[((size_t)j*K+kh)*D+d]),result);
                    want[((size_t)row*H+h)*D+d]=result;
                }
            }
        }
        h3_gpu_tensor *qt=h3_gpu_tensor_from_bf16(gpu,q,nq),*kt=h3_gpu_tensor_from_bf16(gpu,k,nk),*vt=h3_gpu_tensor_from_bf16(gpu,v,nk),*ot=h3_gpu_tensor_new_bf16(gpu,nq);CHECK(qt&&kt&&vt&&ot);
        double errs[3]={0};
        for(int m=0;m<3;m++) {
#ifdef SCALING_BASELINE
            if(m) break;
#endif
            CHECK(!setenv("H3_QWEN_GQA_SCALE_MODE",modes[m],1));
            h3_gqa_limits limit;CHECK(h3_gpu_gqa_causal_limits(gpu,&limit));CHECK(n<=limit.max_sequence);
            CHECK(!strcmp(device.backend,"metal") ? limit.static_bytes==1024 : limit.static_bytes>0 && limit.static_bytes<=limit.device_bytes);
            enum { REPEATS=32, TIMINGS=REPEATS-1 };
            double timings[TIMINGS];h3_gpu_stats before,after;size_t repeat_mismatch=0;
            for(int rep=0;rep<REPEATS;rep++) {
                CHECK(h3_gpu_get_stats(gpu,&before));double start=now();
                CHECK(h3_gpu_begin(gpu));CHECK(h3_gpu_gqa_causal_bf16(gpu,ot,qt,kt,vt,n,H,K,D,scale));CHECK(h3_gpu_submit(gpu));
                CHECK(h3_gpu_get_stats(gpu,&after));if(rep) timings[rep-1]=now()-start;
                CHECK(after.allocated_bytes==before.allocated_bytes);
                CHECK(h3_gpu_tensor_read_bf16(ot,out,nq));
                if(rep)for(size_t i=0;i<nq;i++)repeat_mismatch+=out[i]!=previous[i];
                memcpy(previous,out,nq*2);
            }
            for(int a=0;a<TIMINGS;a++)for(int b=a+1;b<TIMINGS;b++)if(timings[b]<timings[a]){double t=timings[a];timings[a]=timings[b];timings[b]=t;}
            CHECK(h3_gpu_tensor_read_bf16(ot,out,nq));
            double max=0,abs=0,sq=0,norm=0,rounded_sq=0;size_t count=0,mismatch=0;
            for(size_t i=0;i<nq;i++) {
                CHECK(isfinite(fp(out[i])));
                unsigned row=(unsigned)(i/(H*D));if(row%stride && row!=n-1)continue;
                double delta=(double)fp(out[i])-want[i],rd=(double)fp(out[i])-fp(bf(want[i]));
                max=fmax(max,fabs(delta));abs+=fabs(delta);sq+=delta*delta;norm+=(double)want[i]*want[i];rounded_sq+=rd*rd;count++;mismatch+=out[i]!=bf(want[i]);
            }
            errs[m]=sqrt(rounded_sq/(double)count);
#ifndef SCALING_BASELINE
            CHECK(repeat_mismatch==0);
            if(m==2) CHECK(errs[m]<3e-5);
#endif
            printf("{\"case\":\"%s\",\"mode\":\"%s\",\"sequence\":%u,\"query_heads\":%u,\"kv_heads\":%u,\"head_dim\":%u,\"samples\":%zu,\"max_abs\":%.9g,\"mean_abs\":%.9g,\"rmse\":%.9g,\"relative_l2\":%.9g,\"bf16_rmse\":%.9g,\"bf16_mismatch\":%.9g,\"repeat_mismatch\":%zu,\"median_seconds\":%.9g,\"gpu_seconds\":%.9g,\"allocated_bytes\":%llu,\"static_bytes\":%zu,\"max_sequence\":%zu}\n",
                cases[c].name,modes[m],n,H,K,D,count,max,abs/(double)count,sqrt(sq/(double)count),sqrt(sq/fmax(norm,1e-30)),errs[m],(double)mismatch/(double)count,repeat_mismatch,timings[TIMINGS/2],after.gpu_seconds-before.gpu_seconds,(unsigned long long)after.live_bytes,limit.static_bytes,limit.max_sequence);fflush(stdout);
            if(argc==2) {char path[1024];snprintf(path,sizeof(path),"%s/%s-%s.bf16",argv[1],cases[c].name,modes[m]);FILE *f=fopen(path,"wb");CHECK(f);CHECK(fwrite(out,2,nq,f)==nq);CHECK(!fclose(f));}
            CHECK(sqrt(sq/fmax(norm,1e-30))<.2);
#ifndef SCALING_BASELINE
            if(m==2) {
                /* Check actual dispatch, not just the parser's enum value. */
                for(int empty=0;empty<2;empty++) {
                    if(empty) CHECK(!setenv("H3_QWEN_GQA_SCALE_MODE","",1));
                    else CHECK(!unsetenv("H3_QWEN_GQA_SCALE_MODE"));
                    CHECK(h3_gpu_begin(gpu));
                    CHECK(h3_gpu_gqa_causal_bf16(gpu,ot,qt,kt,vt,n,H,K,D,scale));
                    CHECK(h3_gpu_submit(gpu));
                    CHECK(h3_gpu_tensor_read_bf16(ot,previous,nq));
                    CHECK(!memcmp(previous,out,nq*2));
                }
            }
#endif
        }
#ifndef SCALING_BASELINE
        CHECK(errs[2] <= errs[0]+1e-5);CHECK(errs[2] <= errs[1]+2e-5);
#endif
        h3_gpu_tensor_free(qt);h3_gpu_tensor_free(kt);h3_gpu_tensor_free(vt);h3_gpu_tensor_free(ot);free(q);free(k);free(v);free(out);free(previous);free(want);free(scores);
    }
#ifndef SCALING_BASELINE
    CHECK(!setenv("H3_QWEN_GQA_SCALE_MODE","invalid",1));CHECK(!h3_gpu_gqa_causal_preflight(gpu,1,error,sizeof(error)));CHECK(strstr(error,"H3_QWEN_GQA_SCALE_MODE"));
    CHECK(!setenv("H3_QWEN_GQA_SCALE_MODE","reference",1));CHECK(!setenv("H3_MPS_GQA","1",1));CHECK(!h3_gpu_gqa_causal_preflight(gpu,1,error,sizeof(error)));CHECK(strstr(error,"H3_MPS_GQA"));unsetenv("H3_MPS_GQA");
#endif
    h3_gpu_free(gpu);return 0;
}
