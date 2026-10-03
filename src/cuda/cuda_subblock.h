#ifndef H3_CUDA_SUBBLOCK_H
#define H3_CUDA_SUBBLOCK_H
#include "src/denoise/sol.h"
#include "src/denoise/subblock.h"
#include <cuda_runtime.h>
struct h3_cuda_subblock_stats {
    uint64_t calls, workspace_bytes, profiled_regions, unprofiled_regions,router_calls;
    uint64_t pairs[4]; /* selected, possible, protected queries, added protected keys */
    double finite_seconds, pool_seconds, route_seconds, kernel_seconds, output_seconds;
};
#ifdef H3_CUDA_USE_SUBBLOCK
void *h3_cuda_subblock_create(void *,size_t,cudaStream_t,bool,char *,size_t);
void h3_cuda_subblock_free(void *);
int h3_cuda_subblock_reset(void *,char *,size_t);
int h3_cuda_subblock_configure(void *,const h3_sol_layout *,unsigned,float,char *,size_t);
int h3_cuda_subblock_run(void *,void *,const void *,const void *,const void *,
    unsigned,unsigned,float,bool,bool,bool,char *,size_t);
int h3_cuda_subblock_collect(void *,h3_cuda_subblock_stats *,char *,size_t);
/* Bounded single-group/slab diagnostic exports; never used by generation. */
int h3_cuda_subblock_routes(void *,unsigned char *,float *,size_t,char *,size_t);
#else
static inline void *h3_cuda_subblock_create(void *,size_t,cudaStream_t,bool,char *,size_t){return nullptr;}
static inline void h3_cuda_subblock_free(void *){}
static inline int h3_cuda_subblock_reset(void *,char *,size_t){return 1;}
static inline int h3_cuda_subblock_configure(void *,const h3_sol_layout *,unsigned,float,char *,size_t){return 0;}
static inline int h3_cuda_subblock_run(void *,void *,const void *,const void *,const void *,unsigned,unsigned,float,bool,bool,bool,char *,size_t){return 0;}
static inline int h3_cuda_subblock_collect(void *,h3_cuda_subblock_stats *,char *,size_t){return 1;}
#endif
#endif
