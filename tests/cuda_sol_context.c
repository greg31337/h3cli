/* GPU context lifetime, policy isolation and bit-exact public dense bypass. */
#include "src/gpu.h"
#include "src/denoise/attention.h"
#include "src/cuda/cuda_sol_policy.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include "src/device.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);return 1;}checks++;}while(0)
static uint16_t bf(float x){uint32_t b;memcpy(&b,&x,4);b+=0x7fff+((b>>16)&1);return (uint16_t)(b>>16);}
static float fp(uint16_t x){uint32_t b=(uint32_t)x<<16;float f;memcpy(&f,&b,4);return f;}
int main(void){char error[512]={0};unsigned checks=0;h3_gpu*g=NULL;uint64_t free_baseline=0;
    h3_params p=H3_PARAMS_DEFAULT;p.cuda_attention=H3_ATTENTION_SOL;h3_cuda_policy policy;
    CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);h3_sglang_exchange(4);
    /* Fail at two ownership boundaries, after stream/handle creation and
     * after successful SOL workspace admission. No real OOM is required. */
    for(int stage=0;stage<2;stage++) {
        CHECK(!setenv("H3_CUDA_TEST_MEMORY_BUDGET",stage?"536870912":"536870911",1));
        CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);
        g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
        if(!stage)CHECK(!h3_gpu_dit_attention_configure(g,H3_ATTENTION_SOL));
        else {CHECK(h3_gpu_dit_attention_configure(g,H3_ATTENTION_SOL));CHECK(!h3_gpu_tensor_new_bf16(g,1));}
        CHECK(strstr(h3_gpu_error(g),"injected test budget"));
        h3_gpu_free(g);g=NULL;CHECK(!unsetenv("H3_CUDA_TEST_MEMORY_BUDGET"));
    }
    CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);
    for(int trial=0;trial<7;trial++){
        unsigned seq=trial==6?315:trial%2?129:257,qb=trial%2?64:32,heads=56;
        size_t n=(size_t)seq*heads*128;uint16_t*x=malloc(n*2),*reference=malloc(n*2),*actual=malloc(n*2);
        CHECK(x&&reference&&actual);for(size_t i=0;i<n;i++)x[i]=bf((float)((int)(i%29)-14)*.02f);h3_cuda_sol_options options=H3_CUDA_SOL_DEFAULT;options.q_block=(int)qb;options.min_exact=1;
        if(trial==6){options.min_exact=0;options.dense_steps=options.dense_layers=0;}
        h3_cuda_sol_options previous=h3_cuda_sol_exchange(options);
        g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);CHECK(h3_gpu_dit_attention_configure(g,H3_ATTENTION_SOL));
        h3_sol_layout layout={seq,(seq+qb-1)/qb,(seq+63)/64,seq,NULL,NULL};
        layout.query=calloc(layout.query_blocks,sizeof(*layout.query));layout.key=calloc(layout.key_blocks,sizeof(*layout.key));CHECK(layout.query&&layout.key);
        for(int which=0;which<2;which++){
            unsigned block=which?64:qb,count=which?layout.key_blocks:layout.query_blocks;h3_sol_block*m=which?layout.key:layout.query;
            for(unsigned b=0;b<count;b++){m[b].protect=1;m[b].first_frame=m[b].last_frame=-1;m[b].rows=seq-b*block<block?seq-b*block:block;}
        }
        if(trial==6){
            h3_sol_layout_free(&layout);
            h3_layout_spec spec={.text_len=11,.latent_t=1,.latent_h=30,.latent_w=40,.audio_t=2,.frame_count=1};
            h3_layout still={0};CHECK(h3_still_layout_build(&spec,&still,error,sizeof(error)));
            CHECK(h3_sol_layout_build(&still,qb,64,&layout,error,sizeof(error)));
            CHECK(layout.sequence==seq&&layout.protected_rows==seq);
            for(unsigned i=0;i<layout.query_blocks;i++)CHECK(layout.query[i].protect);
            for(unsigned i=0;i<layout.key_blocks;i++)CHECK(layout.key[i].protect);
            h3_layout_free(&still);
        }
        CHECK(h3_gpu_dit_sol_layout(g,&layout));CHECK(!h3_gpu_dit_sol_layout(g,&layout));h3_sol_layout_free(&layout);
        /* Changing request TLS after configuration must not alter this context. */
        options.min_exact=0;h3_cuda_sol_exchange(options);
        h3_gpu_tensor *q=h3_gpu_tensor_from_bf16(g,x,n),*out=h3_gpu_tensor_new_bf16(g,n);CHECK(q&&out);
        CHECK(h3_gpu_begin(g));CHECK(h3_gpu_dit_attention_noise(g,1,.5f,.5f));
        CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,q,q,seq,heads,128,1.f/sqrtf(128.f),0,1,1));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_bf16(out,actual,n));
        CHECK(h3_gpu_begin(g));CHECK(h3_gpu_sdpa_bf16(g,out,q,q,q,seq,heads,128,1.f/sqrtf(128.f)));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_bf16(out,reference,n));
        if(trial==6){
            double norm=0,diff=0,peak=0,maxdiff=0;
            for(size_t i=0;i<n;i++){double a=fp(actual[i]),r=fp(reference[i]),d=a-r;CHECK(isfinite(a));norm+=r*r;diff+=d*d;peak=fmax(peak,fabs(r));maxdiff=fmax(maxdiff,fabs(d));}
            CHECK(sqrt(diff/norm)<=.01&&maxdiff/peak<=.02);
            printf("Still layout: %u protected rows; dense relative L2 %.9g, max-relative %.9g; check emitted approximate=0 counters\n",seq,sqrt(diff/norm),maxdiff/peak);
            uint16_t saved=x[0];x[0]=0x7fc0;
            CHECK(h3_gpu_tensor_write_bf16(q,x,n));CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,q,q,seq,heads,128,1.f/sqrtf(128.f),0,1,1));
            CHECK(!h3_gpu_submit(g));CHECK(strstr(h3_gpu_error(g),"nonfinite"));h3_gpu_cancel(g);
            x[0]=saved;CHECK(h3_gpu_tensor_write_bf16(q,x,n));CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,q,q,seq,heads,128,1.f/sqrtf(128.f),0,1,1));CHECK(h3_gpu_submit(g));
            CHECK(h3_gpu_tensor_read_bf16(out,actual,n));for(size_t i=0;i<n;i++)CHECK(isfinite(fp(actual[i])));
            puts("PASS SOL nonfinite submission, cancellation and finite same-context recovery");
        } else CHECK(!memcmp(actual,reference,n*2));
        CHECK(!h3_gpu_dit_attention_noise(g,1,NAN,.5f));
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));CHECK(stats.attention_workspace_bytes==H3_CUDA_SOL_WORKSPACE_BYTES);CHECK(stats.sage2_attention_dispatches==0&&stats.sage3_attention_dispatches==0);
        h3_gpu_tensor_free(out);h3_gpu_tensor_free(q);h3_gpu_free(g);g=NULL;
        h3_device_info memory;CHECK(h3_device_query(&memory,error,sizeof(error)));
        /* The first normal/fast contexts warm CUDA module caches. Subsequent
         * teardown must return the 512 MiB reservation; allow 32 MiB of
         * driver/module metadata when the final still kernel loads. */
        if(trial==1)free_baseline=memory.free_device_memory;
        if(trial>1)CHECK(memory.free_device_memory+(UINT64_C(32)<<20)>=free_baseline);
        printf("Context %d after teardown: free_device_bytes=%llu\n",trial,(unsigned long long)memory.free_device_memory);
        h3_cuda_sol_exchange(previous);free(x);free(actual);free(reference);
    }
    printf("PASS %u CUDA SOL context checks; allocation failure cleanup, seven shared-policy context lifetimes, TLS isolation, Q32/Q64, policy isolation, exact public bypass, protected still layout\n",checks);return 0;
}
