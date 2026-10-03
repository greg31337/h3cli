#include "src/denoise/subblock.h"
#include "src/denoise/attention.h"
#include "src/denoise/sol.h"
#include <math.h>
#include <stdio.h>
#include <string.h>
unsigned h3_subblock_budget(unsigned blocks,float sparsity) {
    if(!blocks||!isfinite(sparsity)||sparsity<0||sparsity>=1)return 0;
    double keep=8*ceil((1-(double)sparsity)*(double)blocks/8);
    return keep>=blocks?blocks:(unsigned)keep;
}
static int reserve(size_t *c,size_t *offset,size_t count,size_t width) {
    if(*c>SIZE_MAX-255||!width||count>SIZE_MAX/width)return 0;
    size_t start=(*c+255)&~(size_t)255;
    if(start>SIZE_MAX-count*width)return 0;
    *offset=start;*c=start+count*width;return 1;
}
int h3_subblock_plan_make(unsigned seq,unsigned heads,float sparsity,size_t cap,
    h3_subblock_plan *p,char *e,size_t n) {
    if(p)memset(p,0,sizeof(*p));
    if(!p||!seq||seq>10000000||!heads||heads>56)goto fail;
    unsigned blocks=(seq+63)/64,keep=h3_subblock_budget(blocks,sparsity);
    if(!keep)goto fail;
    if(cap>H3_ATTENTION_WORKSPACE_BYTES)cap=H3_ATTENTION_WORKSPACE_BYTES;
    for(unsigned group=heads;group;group/=2) {
        for(unsigned slab=128;slab;slab/=2) {
            h3_subblock_plan a={0};size_t c=0;
            a.sequence=seq;a.heads=heads;a.blocks=blocks;a.cells=blocks*4;
            a.head_group=group;a.query_slab=slab<blocks?slab:blocks;a.keep=keep;
#define R(f,count,width) if(!reserve(&c,&a.f,(size_t)(count),width))goto fail
            R(q_meta,blocks,sizeof(h3_sol_block));R(k_meta,blocks,sizeof(h3_sol_block));
            R(q_pool,(size_t)group*a.cells*128,2);R(k_pool,(size_t)group*a.cells*128,2);
            R(scores,(size_t)group*a.query_slab*blocks,4);
            R(routes,(size_t)group*a.query_slab*blocks,1);
            R(output,(size_t)group*a.query_slab*64*128,2);
            R(fault,1,4);R(counts,4,8);
#undef R
            a.bytes=c;if(c<=cap){*p=a;return 1;}
        }
    }
fail:if(e&&n)snprintf(e,n,"SubBlock shape/overflow/workspace plan is invalid or exceeds its fixed budget");return 0;
}
int h3_subblock_warmup(int configured) { return configured?configured:10; }
const char *h3_subblock_dense_reason(unsigned sequence,unsigned block,int step,float sparsity,int warmup) {
    if(step<h3_subblock_warmup(warmup))return "warmup";
    if(!block)return "probe";
    if(sequence<4096)return "short";
    unsigned blocks=(sequence+63)/64;
    if(h3_subblock_budget(blocks,sparsity)==blocks)return "full-budget";
    return NULL;
}
