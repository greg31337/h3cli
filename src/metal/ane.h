#ifndef H3_ANE_H
#define H3_ANE_H
#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif
/* The optional CoreML research backend always consumes BF16 source tensors.
 * No source weights are modified. A failed call must not publish its output. */
typedef struct h3_ane h3_ane;
typedef struct {
    double load_seconds, pack_seconds, predict_seconds, unpack_seconds;
    /* Bounds of the prediction sequence, including gaps between chunks. */
    double predict_begin, predict_end;
    size_t memory_bytes;
    uint64_t system_wired_bytes, process_footprint_bytes;
    int weight_scale_exponent, input_scale_exponent_min, input_scale_exponent_max;
    int cache_hit, ane_matmuls, total_matmuls;
} h3_ane_stats;
int h3_ane_emit(const char *directory, int rows, int k, int n, int tile,
                char *error, size_t error_size);
h3_ane *h3_ane_create(int chunk, int k, int n, int tile,
                     const char *cache_directory, h3_ane_stats *stats,
                     char *error, size_t error_size);
void h3_ane_free(h3_ane *ane);
/* Synchronous; the caller can dispatch it on a persistent worker queue.
 * x: [rows,k], w: [n,k], y: [rows,n]; y is committed only on success. */
int h3_ane_predict(h3_ane *ane, const uint16_t *x, const uint16_t *w,
                   uint16_t *y, int rows, h3_ane_stats *stats,
                   char *error, size_t error_size);
#ifdef __cplusplus
}
#endif
#endif
