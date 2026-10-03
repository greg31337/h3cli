#ifndef H3_UPSCALE_GPU_H
#define H3_UPSCALE_GPU_H
#include "src/gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
typedef enum {
    H3_UP_NORM, H3_UP_COLUMNS, H3_UP_DEPTHWISE, H3_UP_MODULATE,
    H3_UP_RESIZE, H3_UP_BIAS
} h3_upscale_gpu_op;
typedef struct {
    uint32_t time,height,width,channels,out_height,out_width,first,rows;
    uint32_t kernel_time,kernel_space;
} h3_upscale_gpu_shape;
/* NDHWC BF16, batch=1; GN groups=32, epsilon=1e-5, full T/H/W.
 * Counts validates all integer products and bounds before backend dispatch. */
int h3_upscale_gpu_counts(h3_upscale_gpu_op op,h3_upscale_gpu_shape p,size_t counts[4]);
int h3_gpu_upscale(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *a,const h3_gpu_tensor *b,h3_upscale_gpu_op op,
    h3_upscale_gpu_shape p);
/* BF16 operands/output, full F32 accumulation; convolution bias is separate. */
int h3_gpu_upscale_linear(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *weight,uint32_t rows,uint32_t inputs,uint32_t outputs);
#ifdef __cplusplus
}
#endif
#endif
