/* Recoverable failures, cancellation and immutable inputs on the native API. */
#include "src/gpu.h"
#include "src/execution.h"
#include "src/denoise/attention.h"
#include "src/sglang/sglang.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,g?h3_gpu_error(g):error);return 1;}}while(0)
int main(void) {
    char error[1024];h3_gpu *g=NULL;const size_t n=17*56*128;
    uint16_t *host=calloc(n,2),*reference=calloc(n,2);CHECK(host&&reference);
    for(size_t i=0;i<n;i++)host[i]=0x3f00;
    for(int mode=1;mode<=2;mode++) {
        h3_params p=H3_PARAMS_DEFAULT;p.cuda_attention=mode;h3_cuda_policy policy;
        CHECK(!setenv("H3_CUDA_TEST_MEMORY_BUDGET","134217728",1));
        CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);h3_sglang_exchange(4);
        g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
        CHECK(!h3_gpu_dit_attention_configure(g,mode));CHECK(strstr(h3_gpu_error(g),"budget"));
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats)&&stats.live_bytes==0);
        h3_gpu_free(g);g=NULL;CHECK(!unsetenv("H3_CUDA_TEST_MEMORY_BUDGET"));
        CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);
        for(int repeat=0;repeat<3;repeat++) {
            g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
            CHECK(h3_gpu_dit_attention_configure(g,mode));
            CHECK(!h3_gpu_dit_attention_configure(g,mode));
            h3_gpu_tensor *q=h3_gpu_tensor_from_bf16(g,host,n),*k=h3_gpu_tensor_from_bf16(g,host,n),
                *v=h3_gpu_tensor_from_bf16(g,host,n),*out=h3_gpu_tensor_new_bf16(g,n),*bad=h3_gpu_tensor_new_f32(g,n);
            CHECK(q&&k&&v&&out&&bad);CHECK(h3_gpu_begin(g));
            CHECK(!h3_gpu_dit_sdpa_bf16(g,out,q,k,v,0,56,128,.08837890625f,0,0,0));
            CHECK(!h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,55,128,.08837890625f,0,0,0));
            CHECK(!h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,56,64,.08837890625f,0,0,0));
            CHECK(!h3_gpu_dit_sdpa_bf16(g,q,q,k,v,17,56,128,.08837890625f,0,0,0));
            CHECK(!h3_gpu_dit_sdpa_bf16(g,out,NULL,k,v,17,56,128,.08837890625f,0,0,0));
            CHECK(!h3_gpu_dit_sdpa_bf16(g,bad,q,k,v,17,56,128,.08837890625f,0,0,0));
            CHECK(!h3_gpu_dit_sdpa_bf16(g,out,q,k,v,300000,56,128,.08837890625f,0,0,0));
            h3_gpu_cancel(g);CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,56,128,.08837890625f,0,0,0));
            h3_gpu_cancel(g);CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,56,128,.08837890625f,0,0,1));CHECK(h3_gpu_submit(g));
            CHECK(h3_gpu_tensor_read_bf16(out,reference,n));
            CHECK(h3_gpu_begin(g));CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,56,128,.08837890625f,0,1,2));CHECK(h3_gpu_submit(g));
            uint16_t *actual=calloc(n,2);CHECK(actual);CHECK(h3_gpu_tensor_read_bf16(out,actual,n));CHECK(!memcmp(actual,reference,n*2));
            CHECK(h3_gpu_tensor_read_bf16(q,actual,n));CHECK(!memcmp(actual,host,n*2));
            for(size_t i=0;i<n;i++)actual[i]=0x3f80;
            CHECK(h3_gpu_tensor_write_bf16(v,actual,n));CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,56,128,.08837890625f,0,1,3));CHECK(h3_gpu_submit(g));
            CHECK(h3_gpu_tensor_read_bf16(out,actual,n));CHECK(memcmp(actual,reference,n*2));
            host[0]=0x7fc0;CHECK(h3_gpu_tensor_write_bf16(q,host,n));CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,56,128,1.f/sqrtf(128.f),0,1,3));
            CHECK(!h3_gpu_submit(g));CHECK(strstr(h3_gpu_error(g),"nonfinite"));h3_gpu_cancel(g);
            host[0]=0x3f00;CHECK(h3_gpu_tensor_write_bf16(q,host,n));CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,17,56,128,1.f/sqrtf(128.f),0,1,3));CHECK(h3_gpu_submit(g));
            CHECK(h3_gpu_tensor_read_bf16(out,actual,n));for(size_t i=0;i<n;i++)CHECK((actual[i]&0x7f80)!=0x7f80);
            free(actual);h3_gpu_tensor_free(q);h3_gpu_tensor_free(k);h3_gpu_tensor_free(v);h3_gpu_tensor_free(out);h3_gpu_tensor_free(bad);
            CHECK(h3_gpu_get_stats(g,&stats)&&stats.live_bytes==H3_ATTENTION_WORKSPACE_BYTES);
            h3_gpu_free(g);g=NULL;
        }
    }
    free(host);free(reference);puts("PASS attention allocation failure, invalid input, cancellation, reuse and cleanup");return 0;
}
