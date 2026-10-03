#include "src/gpu.h"
#include "src/conditioning/text_encoder.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
#define CHECK(c) do { checks++; if (!(c)) { \
    fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #c, error); exit(1); \
} } while (0)

int main(void) {
    char error[512] = {0};
    unsetenv("H3_MPS_GQA");
    h3_gpu *gpu = h3_gpu_create("src/metal/shaders.metal", error, sizeof(error));
    CHECK(gpu);
    h3_gqa_limits limits;
    CHECK(h3_gpu_gqa_causal_limits(gpu, &limits));
    size_t n = limits.direct_max_sequence;
    CHECK(n > 1 && n <= UINT32_MAX && limits.alignment_bytes == 16);
    CHECK(n < h3_gpu_gqa_causal_max_sequence(gpu));
    CHECK(n == ((limits.device_bytes - limits.static_bytes) & ~(size_t)15) / 4);
    printf("GQA: device=%zu static=%zu alignment=%zu maximum=%zu\n",
        limits.device_bytes, limits.static_bytes, limits.alignment_bytes, n);
    uint16_t *zeros = calloc(n + 1, sizeof(*zeros));
    uint16_t *half = malloc((n + 1) * sizeof(*half));
    CHECK(zeros && half);
    for (size_t i = 0; i <= n; i++) half[i] = 0x3f00;
    h3_gpu_tensor *q = h3_gpu_tensor_from_bf16(gpu, zeros, n + 1);
    h3_gpu_tensor *v = h3_gpu_tensor_from_bf16(gpu, half, n + 1);
    h3_gpu_tensor *out = h3_gpu_tensor_new_bf16(gpu, n + 1);
    CHECK(q && v && out);
    for (size_t sequence = n - 1; sequence <= n + 1; sequence++) {
        CHECK(h3_gpu_gqa_causal_preflight(gpu, sequence, error, sizeof(error)));
        h3_gpu_stats before, after;
        CHECK(h3_gpu_get_stats(gpu, &before));
        CHECK(h3_gpu_begin(gpu));
        CHECK(h3_gpu_gqa_causal_bf16(gpu, out, q, q, v, (uint32_t)sequence, 1, 1, 1, 1.0f));
        CHECK(h3_gpu_get_stats(gpu, &after));
        CHECK(after.direct_dispatches == before.direct_dispatches + 1);
        CHECK(h3_gpu_submit(gpu));
        CHECK(h3_gpu_tensor_read_bf16(out, half, sequence));
        for (size_t i = 0; i < sequence; i++) CHECK((half[i] & 0x7f80) != 0x7f80);
    }
    CHECK(!h3_gpu_gqa_causal_preflight(gpu, SIZE_MAX, error, sizeof(error)));
    CHECK(!h3_gpu_gqa_causal_preflight(gpu, 0, error, sizeof(error)));
    free(zeros); free(half);
    h3_gpu_tensor_free(q); h3_gpu_tensor_free(v); h3_gpu_tensor_free(out);
    h3_gpu_free(gpu);
    printf("ok: %d active-device GQA preflight and actual dispatch boundary checks\n", checks);
    return 0;
}
