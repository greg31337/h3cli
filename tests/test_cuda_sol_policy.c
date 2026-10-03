#include "src/h3.h"
#include "src/cuda/cuda_sol_policy.h"
#include "src/denoise/attention.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#x,error);exit(1);}checks++;}while(0)
int main(void){char error[512]={0};unsigned checks=0;
    h3_cuda_sol_options o=H3_CUDA_SOL_DEFAULT,p=o;h3_params params=H3_PARAMS_DEFAULT;
    CHECK(o.q_block==32&&o.kv_block==64&&o.min_exact==0.f&&o.dense_steps==1);
    CHECK(h3_cuda_sol_options_valid(o,error,sizeof(error)));
    CHECK(!strcmp(h3_cuda_sol_dense_reason(o,49,0,.2f,.2f),"early-evaluation"));
    CHECK(!strcmp(h3_cuda_sol_dense_reason(o,0,1,.2f,.2f),"early-layer"));
    CHECK(!h3_cuda_sol_dense_reason(o,1,1,.2f,.2f));
    p.dense_sigma=.8f;CHECK(!strcmp(h3_cuda_sol_dense_reason(p,1,1,.2f,.9f),"high-noise"));
    CHECK(!strcmp(h3_cuda_sol_dense_reason(p,1,1,NAN,.1f),"invalid-noise"));
    p.min_exact=1;CHECK(!strcmp(h3_cuda_sol_dense_reason(p,49,1,.2f,.2f),"all-exact"));
    p=o;p.q_block=16;CHECK(!h3_cuda_sol_options_valid(p,error,sizeof(error)));
    p=o;p.kv_block=128;CHECK(!h3_cuda_sol_options_valid(p,error,sizeof(error)));
    p=o;p.tau=NAN;CHECK(!h3_cuda_sol_options_valid(p,error,sizeof(error)));
    p=o;p.min_exact=-1;CHECK(!h3_cuda_sol_options_valid(p,error,sizeof(error)));
    p=o;p.dense_sigma=-.1f;CHECK(!h3_cuda_sol_options_valid(p,error,sizeof(error)));
    params.cuda_attention=H3_ATTENTION_SOL;CHECK(h3_cuda_sol_params_valid(&params,error,sizeof(error)));
    params.core_reuse=2;CHECK(!h3_cuda_sol_params_valid(&params,error,sizeof(error)));params.core_reuse=1;
    params.backend_set=1;CHECK(!h3_cuda_sol_params_valid(&params,error,sizeof(error)));params.backend_set=0;
    params.cuda_sol_set=256;CHECK(!h3_cuda_sol_params_valid(&params,error,sizeof(error)));params.cuda_sol_set=0;
    params.cuda_attention=0;params.cuda_sol_set=H3_SOL_MIN;CHECK(!h3_cuda_sol_params_valid(&params,error,sizeof(error)));
    p=o;p.min_exact=.5f;CHECK(!h3_cuda_sol_options_match(p,o,H3_SOL_MIN));CHECK(h3_cuda_sol_options_match(p,o,H3_SOL_Q));
    h3_cuda_sol_options saved=h3_cuda_sol_exchange(p);CHECK(h3_cuda_sol_current().min_exact==.5f);
    h3_cuda_sol_options nested=h3_cuda_sol_exchange(o);CHECK(nested.min_exact==.5f);h3_cuda_sol_exchange(nested);CHECK(h3_cuda_sol_current().min_exact==.5f);h3_cuda_sol_exchange(saved);
    const unsigned seqs[]={1,31,32,33,63,64,65,243,22000,110000,10000000};
    for(unsigned i=0;i<sizeof(seqs)/sizeof(*seqs);i++)for(unsigned q=32;q<=64;q*=2){
        o.q_block=(int)q;h3_cuda_sol_plan a,b;
        CHECK(h3_cuda_sol_plan_make(seqs[i],56,o,H3_CUDA_SOL_WORKSPACE_BYTES,&a,error,sizeof(error)));
        CHECK(a.bytes<=H3_CUDA_SOL_WORKSPACE_BYTES&&a.query_blocks==(seqs[i]+q-1)/q&&a.key_blocks==(seqs[i]+63)/64);
        CHECK(h3_cuda_sol_plan_make(seqs[i],56,o,H3_CUDA_SOL_WORKSPACE_BYTES,&b,error,sizeof(error))&&!memcmp(&a,&b,sizeof(a)));
        CHECK(a.fault<a.counts&&a.counts+56<=a.bytes);
    }
    h3_cuda_sol_plan a;CHECK(!h3_cuda_sol_plan_make(UINT32_MAX,56,o,H3_CUDA_SOL_WORKSPACE_BYTES,&a,error,sizeof(error)));
    CHECK(!h3_cuda_sol_plan_make(243,57,o,H3_CUDA_SOL_WORKSPACE_BYTES,&a,error,sizeof(error)));
    CHECK(!h3_cuda_sol_plan_make(243,56,o,1024,&a,error,sizeof(error)));
    printf("PASS %u CUDA SOL policy/planner assertions\n",checks);return 0;
}
