#ifndef H3_SUBBLOCK_H
#define H3_SUBBLOCK_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define H3_SUBBLOCK_VERSION 1
#define H3_SUBBLOCK_PLAN_VERSION 1
typedef struct {
    unsigned sequence, heads, blocks, cells, head_group, query_slab, keep;
    size_t q_meta,k_meta,q_pool,k_pool,scores,routes,output,fault,counts,bytes;
} h3_subblock_plan;
unsigned h3_subblock_budget(unsigned blocks,float sparsity);
int h3_subblock_warmup(int configured);
int h3_subblock_plan_make(unsigned sequence,unsigned heads,float sparsity,
    size_t cap,h3_subblock_plan *plan,char *error,size_t size);
const char *h3_subblock_dense_reason(unsigned sequence,unsigned block,int step,
    float sparsity,int warmup);
#ifdef __cplusplus
}
#endif
#endif
