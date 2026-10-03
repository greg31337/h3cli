#ifndef H3_CUDA_CUDNN_H
#define H3_CUDA_CUDNN_H
#include <cuda_runtime_api.h>
#include <stddef.h>
#include <stdint.h>
#ifdef H3_CUDA_USE_CUDNN
/* Independent reference-audio context. Caller owns the bounded workspace and
 * channels-first buffers; an output of nullptr only prepares/queries a plan. */
void *h3_cudnn_reference_create(cudaStream_t stream);
void h3_cudnn_reference_free(void *state);
int h3_cudnn_reference_conv(void *state,float *out,const float *in,const float *weight,
    unsigned batch,unsigned length,unsigned ci,unsigned co,unsigned kernel,
    unsigned stride,unsigned padding,unsigned dilation,bool transpose,
    void *workspace,size_t capacity,size_t *required,char *error,size_t error_size);
/* N,D,H,W,Cin,Cout,Kd,Kh,Kw,Sd,Sh,Sw; input is already padded. */
int h3_cudnn_reference_conv3d(void *state,float *out,const float *in,const float *weight,
    const unsigned shape[12],void *workspace,size_t capacity,size_t *required,char *error,size_t error_size);
#else
static inline void *h3_cudnn_reference_create(cudaStream_t) {return nullptr;}
static inline void h3_cudnn_reference_free(void *) {}
static inline int h3_cudnn_reference_conv(void *,float *,const float *,const float *,
    unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,bool,
    void *,size_t,size_t *,char *,size_t) {return 0;}
static inline int h3_cudnn_reference_conv3d(void *,float *,const float *,const float *,
    const unsigned *,void *,size_t,size_t *,char *,size_t) {return 0;}
#endif
#endif
