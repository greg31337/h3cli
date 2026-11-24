/* Descriptor-only AudioVAE coverage through 736 frames. No movie, PCM,
 * activation, weight or workspace tensors are allocated or executed. */
#include "src/cuda/cuda_cudnn.h"
#include "src/geometry.h"
#include <algorithm>
#include <cstdio>
#include <cstdlib>

static size_t maximum_workspace, maximum_layout;
static unsigned checks;
static unsigned frames;

static void check(void *context, unsigned length, unsigned ci, unsigned co,
                  unsigned kernel, unsigned stride, unsigned padding,
                  unsigned dilation, bool transpose) {
    char error[1024] = {};
    size_t bytes = 0;
    unsigned output = transpose ? (length - 1) * stride - 2 * padding + kernel :
        (length + 2 * padding - dilation * (kernel - 1) - 1) / stride + 1;
    size_t layout = 2 * std::max(size_t(length) * ci, size_t(output) * co) * sizeof(float);
    if (!h3_cudnn_reference_conv(context, nullptr, nullptr, nullptr, 2, length,
            ci, co, kernel, stride, padding, dilation, transpose, nullptr, 0,
            &bytes, error, sizeof(error)) || layout > (64u << 20)) {
        std::fprintf(stderr, "FAIL frames=%u length=%u channels=%u->%u kernel=%u stride=%u dilation=%u transpose=%d layout=%zu: %s\n",
            frames, length, ci, co, kernel, stride, dilation, transpose, layout, error);
        std::exit(1);
    }
    maximum_workspace = std::max(maximum_workspace, bytes);
    maximum_layout = std::max(maximum_layout, layout);
    ++checks;
    if (ci == 2048 && co == 1024)
        std::printf("frames=%u audio_T=%u pre_workspace_bytes=%zu\n", frames, length, bytes);
}

int main() {
    static const unsigned rates[] = {5, 5, 2, 2, 2, 2, 2};
    static const unsigned kernels[] = {9, 9, 4, 4, 4, 4, 4};
    for (frames = 5; frames <= H3_CUDA_MAX_FRAMES; frames += 17) {
        void *context = h3_cudnn_reference_create(nullptr);
        if (!context) { std::fputs("FAIL cuDNN 9.20 context unavailable\n", stderr); return 1; }
        unsigned length = (frames * 40 + 12) / 24;
        check(context, length, 32, 2048, 1, 1, 0, 1, false);
        check(context, length, 2048, 1024, 7, 1, 3, 1, false);
        for (unsigned stage = 0; stage < 7; ++stage) {
            unsigned ci = 1024 >> stage, co = ci / 2;
            unsigned kernel = kernels[stage], stride = rates[stage], padding = (kernel - stride) / 2;
            check(context, length, ci, co, kernel, stride, padding, 1, true);
            length *= stride;
            for (unsigned k : {3u, 7u, 11u}) {
                for (unsigned dilation : {1u, 3u, 5u})
                    check(context, length, co, co, k, 1, dilation * (k - 1) / 2, dilation, false);
                check(context, length, co, co, k, 1, (k - 1) / 2, 1, false);
            }
        }
        check(context, length, 8, 1, 7, 1, 3, 1, false);
        h3_cudnn_reference_free(context);
    }
    std::printf("PASS %u descriptor-only convolution checks; max_workspace_bytes=%zu max_layout_bytes=%zu\n",
        checks, maximum_workspace, maximum_layout);
    return 0;
}
