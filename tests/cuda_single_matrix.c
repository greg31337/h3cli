/* Mutable twelve-option operator/lifetime regression; no model or frozen edits. */
#include "src/gpu.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include "src/weights/quant.h"
#include "src/denoise/attention.h"
#include "src/cuda/cuda_sol_policy.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d %s: %s / %s\n",__LINE__,#x,error,g?h3_gpu_error(g):"");exit(1);}}while(0)
static uint16_t bf(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fff+((u>>16)&1);return (uint16_t)(u>>16);}
int main(int argc,char **argv){
    if (argc != 2)
        return 2;
    char error[1024] = {0};
    h3_gpu *g = NULL;
    enum { ROWS = 129, K = 128, N = 56 * 128 };
    size_t count = (size_t)ROWS * N;
    uint16_t *w = malloc((size_t)K * N * 2), *x = malloc(ROWS * K * 2), *out = malloc(count * 2),
             *dense = malloc(count * 2);
    CHECK(w && x && out && dense);
    for (size_t i = 0; i < (size_t)K * N; i++)
        w[i] = bf((float)((int)(i % 7) - 3) * .003f);
    for (int i = 0; i < ROWS * K; i++)
        x[i] = bf((float)((i % 11) - 5) * .02f);
    char source[] = "/tmp/h3-single-matrix-XXXXXX";
    int fd = mkstemp(source);
    CHECK(fd >= 0);
    CHECK(write(fd, w, (size_t)K * N * 2) == (ssize_t)((size_t)K * N * 2));
    CHECK(!close(fd));
    for (int pass = 0; pass < 2; pass++)
        for (int cell = 0; cell < 12; cell++) {
            int index = pass ? 11 - cell : cell, attention = index / 3, precision = index % 3;
            h3_params p = H3_PARAMS_DEFAULT;
            p.cuda_attention = attention;
            p.cuda_denoise_quant = precision;
            p.cuda_denoise_quant_cache = precision ? argv[1] : NULL;
            h3_cuda_policy policy;
            CHECK(h3_cuda_policy_resolve(&p, "cuda", 0, &policy, error, sizeof(error)));
            CHECK(h3_cuda_policy_preflight(&policy, error, sizeof(error)));
            h3_cuda_policy old = h3_cuda_policy_exchange(policy);
            int sg = h3_sglang_exchange(4);
            g = h3_gpu_create(NULL, error, sizeof(error));
            CHECK(g);
            h3_cuda_policy_exchange(old);
            h3_sglang_exchange(sg);
            CHECK(h3_gpu_dit_attention_configure(g, attention));
            if (attention == H3_ATTENTION_SOL) {
                h3_sol_layout layout = {ROWS, (ROWS + 31) / 32, (ROWS + 63) / 64, ROWS, NULL, NULL};
                layout.query = calloc(layout.query_blocks, sizeof(*layout.query));
                layout.key = calloc(layout.key_blocks, sizeof(*layout.key));
                CHECK(layout.query && layout.key);
                for (unsigned i = 0; i < layout.query_blocks; i++) {
                    layout.query[i].protect = 1;
                    layout.query[i].rows = ROWS - i * 32 < 32 ? ROWS - i * 32 : 32;
                    layout.query[i].first_frame = layout.query[i].last_frame = -1;
                }
                for (unsigned i = 0; i < layout.key_blocks; i++) {
                    layout.key[i].protect = 1;
                    layout.key[i].rows = ROWS - i * 64 < 64 ? ROWS - i * 64 : 64;
                    layout.key[i].first_frame = layout.key[i].last_frame = -1;
                }
                CHECK(h3_gpu_dit_sol_layout(g, &layout));
                h3_sol_layout_free(&layout);
            }
            /* Admission takes complete H3 block geometry; actual probe tensors stay small. */
            const uint64_t core_weights = 2ull * (4ull * 7168 * 5376 + 3ull * 14336 * 5376);
            if (precision)
                CHECK(h3_gpu_quant_configure(g, precision, argv[1], core_weights, 1 << 20, 1));
            h3_gpu_tensor *weight =
                precision ? h3_gpu_quant_load(g, source, 0, N, K) : h3_gpu_tensor_from_bf16(g, w, (size_t)K * N);
            h3_gpu_tensor *input = h3_gpu_tensor_from_bf16(g, x, ROWS * K),
                          *projected = h3_gpu_tensor_new_bf16(g, count), *output = h3_gpu_tensor_new_bf16(g, count);
            CHECK(weight && input && projected && output);
            uint64_t plateau = 0;
            for (int repeat = 0; repeat < 3; repeat++) {
                CHECK(h3_gpu_begin(g));
                CHECK(h3_gpu_linear_bf16(g, projected, input, weight, NULL, ROWS, K, N));
                CHECK(h3_gpu_dit_attention_noise(g, 2, .5f, .5f));
                CHECK(h3_gpu_dit_sdpa_bf16(g, output, projected, projected, projected, ROWS, 56, 128,
                                           1.f / sqrtf(128.f), 0, 2, 2));
                CHECK(h3_gpu_submit(g));
                CHECK(h3_gpu_tensor_read_bf16(output, out, count));
                for (size_t i = 0; i < count; i++)
                    CHECK((out[i] & 0x7f80) != 0x7f80);
                if (!attention && !precision) {
                    if (!pass && !repeat)
                        memcpy(dense, out, count * 2);
                    else
                        CHECK(!memcmp(dense, out, count * 2));
                }
                h3_gpu_stats st;
                CHECK(h3_gpu_get_stats(g, &st));
                if (precision)
                    CHECK(st.quant_projection_dispatches == (uint64_t)repeat + 1);
                else
                    CHECK(!st.quant_projection_dispatches && !st.quant_cache_hits && !st.quant_cache_misses);
                if (attention == 1)
                    CHECK(st.sage2_attention_dispatches > 0);
                if (attention == 2)
                    CHECK(st.sage3_attention_dispatches > 0);
                if (!repeat)
                    plateau = st.live_bytes;
                else
                    CHECK(plateau == st.live_bytes);
                CHECK(h3_gpu_begin(g));
                CHECK(!h3_gpu_linear_bf16(g, projected, input, weight, NULL, ROWS, K, N + 1));
                h3_gpu_cancel(g);
            }
            h3_gpu_tensor_free(output);
            h3_gpu_tensor_free(projected);
            h3_gpu_tensor_free(input);
            h3_gpu_tensor_free(weight);
            h3_gpu_free(g);
            g = NULL;
            printf("PASS matrix pass=%d attention=%s precision=%s finite/tails/actual-dispatch/plateau/cancellation\n",
                   pass, h3_attention_name(attention), h3_quant_name(precision));
        }
 CHECK(!unlink(source));free(w);free(x);free(out);free(dense);puts("PASS all twelve single-pipeline options in both context orders");return 0;
}
