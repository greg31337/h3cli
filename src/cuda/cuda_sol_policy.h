#ifndef H3_CUDA_SOL_POLICY_H
#define H3_CUDA_SOL_POLICY_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
#define H3_CUDA_SOL_VERSION 1
#define H3_CUDA_SOL_PLAN_VERSION 1
#define H3_CUDA_SOL_WORKSPACE_BYTES ((size_t)512*1024*1024)
typedef struct {
    int q_block, kv_block, dense_layers, dense_steps, local_radius;
    float tau, min_exact, dense_sigma;
} h3_cuda_sol_options;
#define H3_CUDA_SOL_DEFAULT {32,64,1,1,1,1.0f,0.0f,-1.0f}
enum { H3_SOL_Q=1, H3_SOL_KV=2, H3_SOL_LAYERS=4, H3_SOL_STEPS=8,
       H3_SOL_RADIUS=16, H3_SOL_TAU=32, H3_SOL_MIN=64, H3_SOL_SIGMA=128 };
int h3_cuda_sol_options_valid(h3_cuda_sol_options o,char *error,size_t size);
int h3_cuda_sol_options_match(h3_cuda_sol_options a,h3_cuda_sol_options b,unsigned mask);
const char *h3_cuda_sol_dense_reason(h3_cuda_sol_options o,unsigned block,int step,float video,float audio);
h3_cuda_sol_options h3_cuda_sol_current(void);
h3_cuda_sol_options h3_cuda_sol_exchange(h3_cuda_sol_options o);
/* All offsets are within an admitted workspace; no free-memory-dependent plan. */
typedef struct {
    uint32_t sequence,heads,query_blocks,key_blocks,head_group,query_slab;
    size_t query_meta,key_meta,q_centroid,k_centroid,v_centroid,moments;
    size_t routes,output,fault,counts,bytes;
} h3_cuda_sol_plan;
int h3_cuda_sol_plan_make(uint32_t sequence,uint32_t heads,h3_cuda_sol_options o,
                         size_t cap,h3_cuda_sol_plan *plan,char *error,size_t size);
#ifdef __cplusplus
}
#endif
#endif
