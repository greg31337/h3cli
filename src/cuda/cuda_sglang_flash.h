#ifndef H3_CUDA_SGLANG_FLASH_H
#define H3_CUDA_SGLANG_FLASH_H
#include <cuda_runtime_api.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
int h3_sglang_flash(void *out,const void *q,const void *k,const void *v,float *lse,
    unsigned batch,unsigned sequence,unsigned heads,unsigned kv_heads,unsigned dim,
    float scale,int causal,int head_major_output,cudaStream_t stream,char *error,size_t size);
int h3_sglang_flash_vae(void *out,const void *q,const void *k,const void *v,float *lse,
    unsigned sequence,unsigned heads,float scale,cudaStream_t stream,char *error,size_t size);
/* Noncausal main-DiT query slice, preserving original QKV/output strides. */
int h3_sglang_flash_query_range(void *out,const void *q,const void *k,const void *v,float *lse,
    unsigned sequence,unsigned heads,unsigned first,unsigned count,float scale,
    int head_major_output,cudaStream_t stream,char *error,size_t size);
#ifdef __cplusplus
}
#endif
#endif
