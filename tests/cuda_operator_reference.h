/* Direct kernels used only as numerical oracles for CUDA operator tests. */
#ifndef CUDA_OPERATOR_REFERENCE_H
#define CUDA_OPERATOR_REFERENCE_H
#include "src/gpu.h"
#ifdef __cplusplus
extern "C" {
#endif
int test_cuda_attention_reference(h3_gpu *g, h3_gpu_tensor *out,
    const h3_gpu_tensor *q, const h3_gpu_tensor *k, const h3_gpu_tensor *v,
    uint32_t sequence, uint32_t heads, uint32_t dim, float scale, int head_major);
int test_cuda_conv_reference(h3_gpu *g, h3_gpu_tensor *out,
    const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
    uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
    uint32_t channels_in, uint32_t channels_out, uint32_t kernel, uint32_t stride);
#ifdef __cplusplus
}
#endif
#endif
