#ifndef H3_Q8_H
#define H3_Q8_H
#include <stddef.h>
#include <stdint.h>

/* Recipe 1: signed symmetric weight-only groups along K, round-to-even,
 * float scales, BF16 reconstruction and activations/output. Direct simdgroup
 * uses FP32 sums; the alternate bounded-dequant path retains MPSGraph GEMM.
 * Storage: row-major int8[N,K], zero alignment padding, float[N,ceil(K/64)].
 * No activation quantization, persistent file cache, or checkpoint mutation. */
#define H3_Q8_VERSION 1
#define H3_Q8_GROUP 64
int h3_q8_layout(uint32_t rows, uint32_t columns, size_t *scale_offset,
                 size_t *bytes);
int h3_q8_pack_rows(const uint16_t *source, int8_t *weights, float *scales,
                    uint32_t rows, uint32_t columns);
#endif
