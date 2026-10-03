#include "src/cuda/cuda_sol_policy.h"
#include "src/denoise/sol.h"
#include "src/h3.h"
#include "src/denoise/attention.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
static _Thread_local h3_cuda_sol_options current=H3_CUDA_SOL_DEFAULT;
h3_cuda_sol_options h3_cuda_sol_current(void){return current;}
h3_cuda_sol_options h3_cuda_sol_exchange(h3_cuda_sol_options o){h3_cuda_sol_options old=current;current=o;return old;}
int h3_cuda_sol_options_valid(h3_cuda_sol_options o,char *e,size_t n) {
    if((o.q_block!=32&&o.q_block!=64)||o.kv_block!=64||o.dense_layers<0||o.dense_layers>50||
       o.dense_steps<0||o.dense_steps>1000||o.local_radius<0||o.local_radius>10000||
       !isfinite(o.tau)||o.tau<0||o.tau>16||!isfinite(o.min_exact)||o.min_exact<0||o.min_exact>1||
       !isfinite(o.dense_sigma)||(o.dense_sigma!=-1&&(o.dense_sigma<0||o.dense_sigma>1))) {
        if (e && n)
            snprintf(
                e, n,
                "invalid CUDA SOL policy: Q=32|64, KV=64, layers=0..50, steps=0..1000, radius=0..10000, tau=0..16, min-exact=0..1, sigma=-1|0..1");
        return 0;
    }return 1;
}
int h3_cuda_sol_options_match(h3_cuda_sol_options a,h3_cuda_sol_options b,unsigned m) {
    return (!(m&H3_SOL_Q)||a.q_block==b.q_block)&&(!(m&H3_SOL_KV)||a.kv_block==b.kv_block)&&
        (!(m&H3_SOL_LAYERS)||a.dense_layers==b.dense_layers)&&(!(m&H3_SOL_STEPS)||a.dense_steps==b.dense_steps)&&
        (!(m&H3_SOL_RADIUS)||a.local_radius==b.local_radius)&&(!(m&H3_SOL_TAU)||a.tau==b.tau)&&
        (!(m&H3_SOL_MIN)||a.min_exact==b.min_exact)&&(!(m&H3_SOL_SIGMA)||a.dense_sigma==b.dense_sigma);
}
const char *h3_cuda_sol_dense_reason(h3_cuda_sol_options o,unsigned block,int step,float video,float audio) {
    if(step<0||!isfinite(video)||!isfinite(audio)||video<0||video>1||audio<0||audio>1)return "invalid-noise";
    if(o.min_exact>=1)return "all-exact";
    if(step<o.dense_steps)return "early-evaluation";
    if(o.dense_sigma>=0&&fmaxf(video,audio)>=o.dense_sigma)return "high-noise";
    if(block<(unsigned)o.dense_layers)return "early-layer";
    return NULL;
}
static int reserve(size_t *cursor,size_t *offset,size_t count,size_t width) {
    if(*cursor>SIZE_MAX-255||count>SIZE_MAX/width)return 0;
    size_t start=(*cursor+255)&~(size_t)255,bytes=count*width;
    if (start > SIZE_MAX - bytes)
        return 0;
    *offset = start;
    *cursor = start + bytes;
    return 1;
}
int h3_cuda_sol_plan_make(uint32_t seq,uint32_t heads,h3_cuda_sol_options o,size_t cap,
                         h3_cuda_sol_plan *p,char *e,size_t n) {
    if(p)memset(p,0,sizeof(*p));
    if(!p||!seq||seq>10000000||!heads||heads>56||!h3_cuda_sol_options_valid(o,e,n))goto failed;
    if(cap>H3_CUDA_SOL_WORKSPACE_BYTES)cap=H3_CUDA_SOL_WORKSPACE_BYTES;
    for(unsigned group=heads<8?heads:8;group;group/=2) {
        for(unsigned slab=128;slab;slab/=2) {
            h3_cuda_sol_plan a={0};size_t c=0;
            a.sequence=seq;a.heads=heads;a.query_blocks=(seq+(unsigned)o.q_block-1)/(unsigned)o.q_block;
            a.key_blocks=(seq+63)/64;a.head_group=group;a.query_slab=slab<a.query_blocks?slab:a.query_blocks;
#define R(field,count,width) if(!reserve(&c,&a.field,(size_t)(count),width))goto failed
            R(query_meta,a.query_blocks,sizeof(h3_sol_block));R(key_meta,a.key_blocks,sizeof(h3_sol_block));
            R(q_centroid,(size_t)group*a.query_blocks*128,2);
            R(k_centroid,(size_t)group*a.key_blocks*128,2);R(v_centroid,(size_t)group*a.key_blocks*128,2);
            R(moments,(size_t)group*256,4);R(routes,(size_t)group*a.query_slab*a.key_blocks,1);
            R(output,(size_t)group*a.query_slab*(unsigned)o.q_block*128,2);
            R(fault,1,4);R(counts,7,8);
#undef R
            a.bytes=c;if(c<=cap){*p=a;return 1;}
        }
    }
failed:if(e&&n)snprintf(e,n,"CUDA SOL shape/overflow/workspace plan is invalid or exceeds its fixed budget");return 0;
}

int h3_cuda_sol_params_valid(const h3_params *p,char *e,size_t n) {
    const char *why=NULL;
    if(!p)why="missing CUDA SOL parameters";
    else if(p->cuda_sol_set&~255u)why="invalid CUDA SOL explicit-option mask";
    else if(p->cuda_attention!=H3_ATTENTION_SOL) {
        if(p->cuda_sol_set&&!p->resume_sampler_state)why="CUDA SOL controls require --cuda-attention sol";
    } else {
        const char *reduction=getenv("H3_TOKEN_REDUCTION");
        if(!h3_cuda_sol_options_valid(p->cuda_sol,e,n))return 0;
        if(p->backend||p->backend_set||p->token_reduction||p->denoise_reuse!=1||p->core_reuse!=1||p->dit_layers!=50||
           (reduction&&*reduction&&strcmp(reduction,"0")))
            why="CUDA SOL requires CUDA, all 50 layers, reuse/core-reuse 1 and token reduction off";
    }
    if(why){if(e&&n)snprintf(e,n,"%s",why);return 0;}return 1;
}
