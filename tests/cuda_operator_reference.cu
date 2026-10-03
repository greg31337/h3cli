/* Include the backend to access its direct kernels without restoring a
 * production dispatch switch. This replaces gpu_cuda.o in these tests only. */
#include "../src/cuda/gpu_cuda.cu"
#include "cuda_operator_reference.h"

int test_cuda_attention_reference(h3_gpu *g, h3_gpu_tensor *out,
    const h3_gpu_tensor *q, const h3_gpu_tensor *k, const h3_gpu_tensor *v,
    uint32_t sequence, uint32_t heads, uint32_t dim, float scale, int head_major) {
    if (!launch_ready(g)) return 0;
    /* The BF16 operator contract rounds the scale before F32 accumulation. */
    uint32_t bits;
    memcpy(&bits, &scale, sizeof(bits));
    bits = (bits + 0x7fff + ((bits >> 16) & 1)) & 0xffff0000u;
    memcpy(&scale, &bits, sizeof(scale));
    attention_online<<<dim3(sequence, heads, 1), 128, 0, g->compute>>>(
        (ushort *)tensor_pointer(out), (ushort *)tensor_pointer(q),
        (ushort *)tensor_pointer(k), (ushort *)tensor_pointer(v),
        sequence, heads, heads, dim, scale, false, head_major != 0, 0);
    return launch_status(g, "test direct attention");
}

int test_cuda_conv_reference(h3_gpu *g, h3_gpu_tensor *out,
    const h3_gpu_tensor *input, const h3_gpu_tensor *weight, const h3_gpu_tensor *bias,
    uint32_t batch, uint32_t depth, uint32_t height, uint32_t width,
    uint32_t channels_in, uint32_t channels_out, uint32_t kernel, uint32_t stride) {
    if (!launch_ready(g)) return 0;
    uint32_t od = (depth - kernel) / stride + 1;
    uint32_t oh = (height - kernel) / stride + 1;
    uint32_t ow = (width - kernel) / stride + 1;
    size_t count = (size_t)batch * od * oh * ow * channels_out;
    conv3d_kernel<<<(unsigned)((count + 255) / 256), 256, 0, g->compute>>>(
        (float *)tensor_pointer(out), (float *)tensor_pointer(input),
        (float *)tensor_pointer(weight), (float *)tensor_pointer(bias),
        batch, depth, height, width, channels_in, channels_out,
        kernel, kernel, kernel, stride, stride, stride, od, oh, ow);
    return launch_status(g, "test direct convolution");
}
