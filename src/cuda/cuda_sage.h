#ifndef H3_CUDA_SAGE_H
#define H3_CUDA_SAGE_H
#include <stddef.h>
#include <stdint.h>
#include <cuda_runtime.h>
struct h3_sage_stats {
    uint64_t calls[3], head_groups, workspace_bytes;
    double reduction_seconds, pack_seconds, correction_seconds, kernel_seconds, output_seconds;
};
#ifdef H3_CUDA_USE_SAGE
void *h3_sage_create(void *workspace,size_t bytes,cudaStream_t stream,bool profile,char *error,size_t size);
void h3_sage_free(void *context);
int h3_sage_reset(void *context,char *error,size_t size);
int h3_sage_run(void *context,int mode,void *out,const void *q,const void *k,const void *v,
                unsigned seq,unsigned heads,float scale,bool head_major,char *error,size_t size);
int h3_sage_collect(void *context,h3_sage_stats *stats,char *error,size_t size);
#else
static inline void *h3_sage_create(void *,size_t,cudaStream_t,bool,char *,size_t){return nullptr;}
static inline void h3_sage_free(void *){}
static inline int h3_sage_reset(void *,char *,size_t){return 1;}
static inline int h3_sage_run(void *,int,void *,const void *,const void *,const void *,unsigned,unsigned,float,bool,char *,size_t){return 0;}
static inline int h3_sage_collect(void *,h3_sage_stats *,char *,size_t){return 1;}
#endif
#endif
