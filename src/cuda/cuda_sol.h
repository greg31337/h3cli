#ifndef H3_CUDA_SOL_H
#define H3_CUDA_SOL_H
#include "src/cuda/cuda_sol_policy.h"
#include "src/denoise/sol.h"
#include <cuda_runtime.h>
struct h3_cuda_sol_stats {
    uint64_t calls,head_groups,slabs,workspace_bytes,profiled_regions,unprofiled_regions;
    uint64_t pairs[7]; /* exact, approximate, protected, local, floor, recovery, tail */
    double finite_seconds,summary_seconds,route_seconds,fused_seconds,output_seconds;
};
#ifdef H3_CUDA_USE_SOL
void *h3_cuda_sol_create(void *workspace,size_t bytes,cudaStream_t stream,bool profile,char *error,size_t size);
void h3_cuda_sol_free(void *context);
int h3_cuda_sol_reset(void *context,char *error,size_t size);
int h3_cuda_sol_configure(void *context,h3_cuda_sol_options options,const h3_sol_layout *layout,
                          unsigned heads,char *error,size_t size);
int h3_cuda_sol_run(void *context,void *out,const void *q,const void *k,const void *v,
                    unsigned seq,unsigned heads,float scale,bool head_major,bool force_exact,char *error,size_t size);
/* Caller has synchronized its stream. No implicit per-layer waits in run(). */
int h3_cuda_sol_collect(void *context,h3_cuda_sol_stats *stats,char *error,size_t size);
/* Test-only bounded route export; call after run, before another dispatch. */
int h3_cuda_sol_routes(void *context,unsigned char *out,size_t bytes,char *error,size_t size);
#else
static inline void *h3_cuda_sol_create(void *,size_t,cudaStream_t,bool,char *,size_t){return nullptr;}
static inline void h3_cuda_sol_free(void *){}
static inline int h3_cuda_sol_reset(void *,char *,size_t){return 1;}
static inline int h3_cuda_sol_configure(void *,h3_cuda_sol_options,const h3_sol_layout *,unsigned,char *,size_t){return 0;}
static inline int h3_cuda_sol_run(void *,void *,const void *,const void *,const void *,unsigned,unsigned,float,bool,bool,char *,size_t){return 0;}
static inline int h3_cuda_sol_collect(void *,h3_cuda_sol_stats *,char *,size_t){return 1;}
#endif
#endif
