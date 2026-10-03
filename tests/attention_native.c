/* Independent inputs and a file interface for the pinned Python SDPA/Sage oracle.
 * Args: MODE SEQUENCE OUTPUT.bin HEAD_MAJOR ITERATIONS [QKV.bin|zero|constant|nan]
 * QKV.bin contains three contiguous sequence-major BF16 arrays. */
#include "src/gpu.h"
#include "src/denoise/attention.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include "src/cuda/cuda_sol_policy.h"
#include "src/denoise/sol.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char error[1024];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);goto failed;}}while(0)
static uint16_t bf(float f){uint32_t x;memcpy(&x,&f,4);x+=0x7fff+((x>>16)&1);return (uint16_t)(x>>16);}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}
static float random_value(uint32_t *state){*state^=*state<<13;*state^=*state>>17;*state^=*state<<5;return (float)((int)(*state%65537)-32768)/16384.f;}
static uint64_t hash(const uint16_t *p,size_t n){uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ull;}return h;}
int main(int argc,char **argv) {
    h3_gpu *g=NULL;h3_gpu_tensor *tensor[4]={0};uint16_t *host=NULL;FILE *input=NULL;int result=1,mode=0;
    if(argc<6||!h3_attention_parse(argv[1],&mode))return 2;
    char *end=NULL;unsigned long length=strtoul(argv[2],&end,10);if(!*argv[2]||*end||!length||length>200000)return 2;
    unsigned seq=(unsigned)length;int head_major=atoi(argv[4]),iterations=atoi(argv[5]);
    if((head_major!=0&&head_major!=1)||iterations<1||iterations>20)return 2;
    size_t n=(size_t)seq*56*128,guard=256;uint64_t input_hash[3];
    if(getenv("H3_TEST_SHARED_POLICY")) {
        h3_params p=H3_PARAMS_DEFAULT;p.cuda_attention=mode;h3_cuda_policy policy;
        CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);h3_sglang_exchange(4);
    }
    h3_cuda_sol_options sol=H3_CUDA_SOL_DEFAULT;
    if(getenv("H3_TEST_SOL_MIN_EXACT"))sol.min_exact=strtof(getenv("H3_TEST_SOL_MIN_EXACT"),NULL);
    CHECK(h3_cuda_sol_options_valid(sol,error,sizeof(error)));h3_cuda_sol_exchange(sol);
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    CHECK(h3_gpu_dit_attention_configure(g,mode));
    if(mode==H3_ATTENTION_SOL) {
        unsigned qb=(seq+31)/32,kb=(seq+63)/64;
        h3_sol_layout layout={seq,qb,kb,0,calloc(qb,sizeof(h3_sol_block)),calloc(kb,sizeof(h3_sol_block))};CHECK(layout.query&&layout.key);
        if(argc>7){
            FILE *f=fopen(argv[7],"rb");CHECK(f);unsigned header[3];
            int ok=fread(header,sizeof(header),1,f)==1&&header[0]==seq&&header[1]==qb&&header[2]==kb&&
                fread(layout.query,sizeof(h3_sol_block),qb,f)==qb&&fread(layout.key,sizeof(h3_sol_block),kb,f)==kb&&fgetc(f)==EOF;
            fclose(f);CHECK(ok);
        } else for(int which=0;which<2;which++) {
            unsigned block=which?64:32,count=which?kb:qb;h3_sol_block *m=which?layout.key:layout.query;
            for(unsigned b=0;b<count;b++){m[b].first_frame=m[b].last_frame=(int)(b*block/64);m[b].rows=seq-b*block<block?seq-b*block:block;m[b].protect=b==0;}
        }
        CHECK(h3_gpu_dit_sol_layout(g,&layout));h3_sol_layout_free(&layout);
    }
    host=malloc((n+guard)*2);CHECK(host);
    const char *pattern=argc>6?argv[6]:"random";
    int generated=!strcmp(pattern,"random")||!strcmp(pattern,"zero")||!strcmp(pattern,"constant")||!strcmp(pattern,"nan");
    if(!generated){input=fopen(pattern,"rb");CHECK(input);}
    uint32_t rng=0x5a317d6b;
    for(unsigned j=0;j<3;j++){
        if(input){CHECK(fread(host,2,n,input)==n);}
        else for(size_t i=0;i<n;i++)host[i]=bf(!strcmp(pattern,"zero")?0:!strcmp(pattern,"constant")?0.5f:random_value(&rng));
        if(j==0&&!strcmp(pattern,"nan"))host[0]=0x7fc0;
        for(size_t i=n;i<n+guard;i++)host[i]=0x7f7f;
        input_hash[j]=hash(host,n+guard);
        tensor[j]=h3_gpu_tensor_from_bf16(g,host,n+guard);CHECK(tensor[j]);
    }
    if(input){CHECK(fgetc(input)==EOF);fclose(input);input=NULL;}
    memset(host,0xff,n*2);tensor[3]=h3_gpu_tensor_from_bf16(g,host,n+guard);CHECK(tensor[3]);
    double elapsed=0,cold=0,samples[20]={0};
    for(int i=0;i<=iterations;i++){
        double start=now();CHECK(h3_gpu_begin(g));CHECK(h3_gpu_dit_attention_noise(g,3,.5f,.5f));
        CHECK(h3_gpu_dit_sdpa_bf16(g,tensor[3],tensor[0],tensor[1],tensor[2],seq,56,128,1.f/sqrtf(128.f),head_major,24,3));
        if(!strcmp(pattern,"nan")){CHECK(!h3_gpu_submit(g));CHECK(strstr(h3_gpu_error(g),"nonfinite"));result=0;goto failed;}
        CHECK(h3_gpu_submit(g));double t=now()-start;if(i){elapsed+=t;samples[i-1]=t;}else cold=t;
    }
    for(unsigned j=0;j<3;j++){CHECK(h3_gpu_tensor_read_bf16(tensor[j],host,n+guard));CHECK(hash(host,n+guard)==input_hash[j]);}
    CHECK(h3_gpu_tensor_read_bf16(tensor[3],host,n+guard));
    for(size_t i=n;i<n+guard;i++)CHECK(host[i]==0x7f7f);
    for(size_t i=0;i<n;i++){
        CHECK((host[i]&0x7f80)!=0x7f80);
        if(!strcmp(pattern,"zero"))CHECK((host[i]&0x7fff)==0);
        if(!strcmp(pattern,"constant")){uint32_t b=(uint32_t)host[i]<<16;float f;memcpy(&f,&b,4);CHECK(fabsf(f-0.5f)<0.02f);}
    }
    if(strcmp(argv[3],"-")){FILE *f=fopen(argv[3],"wb");CHECK(f);int ok=fwrite(host,2,n,f)==n;if(fclose(f))ok=0;CHECK(ok);}
    h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
    CHECK(mode!=1||stats.sage2_attention_dispatches==(uint64_t)(iterations+1));
    CHECK(mode!=2||stats.sage3_attention_dispatches==(uint64_t)(iterations+1));
    CHECK(mode||(!stats.sage2_attention_dispatches&&!stats.sage3_attention_dispatches));
    printf("{\"mode\":\"%s\",\"sequence\":%u,\"heads\":56,\"dimension\":128,\"head_major\":%d,\"iterations\":%d,\"cold_seconds\":%.9f,\"mean_seconds\":%.9f,\"peak_device_bytes\":%llu,\"dispatches\":%llu,\"samples_seconds\":[",
        argv[1],seq,head_major,iterations,cold,elapsed/iterations,(unsigned long long)stats.peak_live_bytes,(unsigned long long)stats.attention_dispatches);
    for(int i=0;i<iterations;i++)printf("%s%.9f",i?",":"",samples[i]);
    puts("]}");
    result=0;
failed:
    if(input)fclose(input);
    for(int i=0;i<4;i++)h3_gpu_tensor_free(tensor[i]);
    h3_gpu_free(g);free(host);return result;
}
