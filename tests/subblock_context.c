#include "src/gpu.h"
#include "src/denoise/attention.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);return 1;}}while(0)
static uint16_t bf(float f){uint32_t b;memcpy(&b,&f,4);return (uint16_t)((b+0x7fff+((b>>16)&1))>>16);}
int main(int argc,char **argv) {
    char error[512];h3_gpu *g=NULL;
    unsigned seq=0;const unsigned heads=56;
    h3_layout_ref refs[]={{H3_LAYOUT_REF_IMAGE,1,8,10,0},{H3_LAYOUT_REF_IMAGE,1,6,14,0},
        {H3_LAYOUT_REF_VIDEO,6,8,10,10},{H3_LAYOUT_REF_AUDIO,0,0,0,12}};
    h3_temporal_shape temporal=h3_temporal(56);
    h3_layout_spec spec={.text_len=19,.latent_t=temporal.video_t,.latent_h=32,.latent_w=32,
        .audio_t=temporal.audio_t,.frame_count=temporal.frame_count,.references=refs,.reference_count=4};
    h3_layout reference_layout={0};CHECK(h3_layout_build(&spec,&reference_layout,error,sizeof(error)));
    seq=(unsigned)reference_layout.seq_len;CHECK(seq>=4096&&seq<=8192);
    CHECK(argc==1||argc==3);
    if(argc==3){FILE *f=fopen(argv[1],"rb");CHECK(f&&fread(&seq,4,1,f)==1);fclose(f);CHECK(seq>=4096&&seq<=8192);}
    const size_t n=(size_t)seq*heads*128;
    uint16_t *x=malloc(n*2),*dense=malloc(n*2),*actual=malloc(n*2);CHECK(x&&dense&&actual);
    for(size_t i=0;i<n;i++)x[i]=bf((float)((int)((i*7)%67)-33)/32);
    for(int trial=0;trial<3;trial++) {
        h3_params p=H3_PARAMS_DEFAULT;p.cuda_attention=trial==2?0:H3_ATTENTION_SUBBLOCK;p.subblock_sparsity=trial==1?0:.75f;
        h3_cuda_policy policy;CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));
        h3_cuda_policy_exchange(policy);h3_sglang_exchange(4);
        g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
        CHECK(h3_gpu_dit_attention_configure(g,p.cuda_attention));
        h3_gpu_stats initial={0};CHECK(h3_gpu_get_stats(g,&initial));
        CHECK(initial.attention_workspace_bytes==(trial==2?0:H3_ATTENTION_WORKSPACE_BYTES));
        h3_sol_layout layout={seq,(seq+63)/64,(seq+63)/64,65,NULL,NULL};
        layout.query=calloc(layout.query_blocks,sizeof(*layout.query));layout.key=calloc(layout.key_blocks,sizeof(*layout.key));CHECK(layout.query&&layout.key);
        if(argc==1) {h3_sol_layout_free(&layout);CHECK(h3_sol_layout_build(&reference_layout,64,64,&layout,error,sizeof(error)));}
        if(argc==3) {
            FILE *f=fopen(argv[1],"rb");uint32_t header[3];CHECK(f&&fread(header,sizeof(header),1,f)==1);
            CHECK(header[0]==seq&&header[1]==layout.query_blocks&&header[2]==layout.key_blocks);
            CHECK(fread(layout.query,sizeof(*layout.query),layout.query_blocks,f)==layout.query_blocks&&fread(layout.key,sizeof(*layout.key),layout.key_blocks,f)==layout.key_blocks);
            CHECK(fgetc(f)==EOF);fclose(f);
        }
        unsigned char *protected=calloc(layout.query_blocks,1);CHECK(protected);
        for(unsigned b=0;b<layout.query_blocks;b++)protected[b]=layout.query[b].protect!=0;
        CHECK(h3_gpu_dit_sol_layout(g,&layout));h3_sol_layout_free(&layout);
        h3_gpu_tensor *q=argc==3?h3_gpu_tensor_load_bf16(g,argv[2],0,n):h3_gpu_tensor_from_bf16(g,x,n),*out=h3_gpu_tensor_new_bf16(g,n);
        h3_gpu_tensor *k=argc==3?h3_gpu_tensor_load_bf16(g,argv[2],n*2,n):q,*v=argc==3?h3_gpu_tensor_load_bf16(g,argv[2],n*4,n):q;CHECK(q&&k&&v&&out);
        for(int hm=0;hm<2;hm++) {
            CHECK(h3_gpu_begin(g));CHECK(h3_gpu_dit_attention_noise(g,9,.5f,.4f));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,seq,heads,128,1/sqrtf(128.f),hm,1,9));CHECK(h3_gpu_submit(g));
            CHECK(h3_gpu_tensor_read_bf16(out,dense,n));
            h3_gpu_stats before={0};CHECK(h3_gpu_get_stats(g,&before));
            CHECK(h3_gpu_begin(g));CHECK(h3_gpu_dit_attention_noise(g,10,.45f,.35f));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,seq,heads,128,1/sqrtf(128.f),hm,0,10));CHECK(h3_gpu_submit(g));
            CHECK(h3_gpu_tensor_read_bf16(out,actual,n));CHECK(!memcmp(dense,actual,n*2));
            h3_gpu_stats probe={0};CHECK(h3_gpu_get_stats(g,&probe));
            h3_cuda_policy_exchange((h3_cuda_policy){0}); /* existing context owns its policy */
            CHECK(h3_gpu_begin(g));CHECK(h3_gpu_dit_attention_noise(g,10,.45f,.35f));
            CHECK(h3_gpu_dit_sdpa_bf16(g,out,q,k,v,seq,heads,128,1/sqrtf(128.f),hm,1,10));CHECK(h3_gpu_submit(g));
            CHECK(h3_gpu_tensor_read_bf16(out,actual,n));
            if(trial)CHECK(!memcmp(dense,actual,n*2));
            for(unsigned row=0;row<seq;row++)for(unsigned h=0;h<heads;h++) {
                size_t offset=(hm?(size_t)h*seq+row:(size_t)row*heads+h)*128;
                if(protected[row/64])CHECK(!memcmp(dense+offset,actual+offset,256));
                for(unsigned d=0;d<128;d++)CHECK((actual[offset+d]&0x7f80)!=0x7f80);
            }
            h3_gpu_stats after={0};CHECK(h3_gpu_get_stats(g,&after));
            if(!trial)CHECK(after.subblock_calls==probe.subblock_calls+1&&after.subblock_protected_calls>probe.subblock_protected_calls);
            else CHECK(after.subblock_calls==probe.subblock_calls);
            printf("{\"trial\":%d,\"head_major\":%d,\"sequence\":%u,\"dense_seconds\":%.9g,\"candidate_seconds\":%.9g,\"router_seconds\":%.9g,\"sparse_kernel_seconds\":%.9g,\"sparse_calls\":%llu,\"selected\":%llu,\"possible\":%llu}\n",
                trial,hm,seq,probe.main_attention_seconds-before.main_attention_seconds,after.main_attention_seconds-probe.main_attention_seconds,
                after.subblock_router_seconds-before.subblock_router_seconds,after.subblock_kernel_seconds-before.subblock_kernel_seconds,
                (unsigned long long)(after.subblock_calls-before.subblock_calls),(unsigned long long)(after.subblock_selected-before.subblock_selected),(unsigned long long)(after.subblock_possible-before.subblock_possible));
        }
        free(protected);if(argc==3){h3_gpu_tensor_free(k);h3_gpu_tensor_free(v);}
        h3_gpu_tensor_free(q);h3_gpu_tensor_free(out);h3_gpu_free(g);g=NULL;
    }
    /* Admission failure must leave no workspace behind in a later request. */
    CHECK(!setenv("H3_CUDA_TEST_MEMORY_BUDGET","536870911",1));
    h3_params p=H3_PARAMS_DEFAULT;p.cuda_attention=H3_ATTENTION_SUBBLOCK;
    h3_cuda_policy policy;CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    CHECK(!h3_gpu_dit_attention_configure(g,p.cuda_attention));h3_gpu_free(g);g=NULL;
    CHECK(!unsetenv("H3_CUDA_TEST_MEMORY_BUDGET"));h3_cuda_policy_exchange((h3_cuda_policy){0});
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    h3_gpu_stats final={0};CHECK(h3_gpu_get_stats(g,&final));CHECK(final.attention_workspace_bytes==0);h3_gpu_free(g);g=NULL;
    h3_layout_free(&reference_layout);free(x);free(actual);free(dense);puts("PASS SubBlock warmup/probe/full-budget bypass, dense protected queries, layouts, memory admission and context isolation");return 0;
}
