#ifndef H3_RESIDENCY_H
#define H3_RESIDENCY_H
#include <stdint.h>
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { H3_WEIGHTS_AUTO, H3_WEIGHTS_RESIDENT, H3_WEIGHTS_STREAM };
typedef struct {
    int mode;
    int max_resident; /* -1 means uncapped; test-only, never increases capacity. */
    uint64_t capacity_limit; /* test-only total tensor capacity; 0 means uncapped. */
    uint64_t allocation_limit; /* allocation fault injection, not planner input. */
} h3_weight_options;
typedef struct {
    h3_weight_options options;
    uint64_t active_mask, resident_mask;
    uint64_t free_bytes, live_bytes, reserve_bytes, future_bytes;
    uint64_t block_bytes, resident_bytes, slot_bytes;
    unsigned active_count, resident_count;
    int effective_mode; /* auto here means partial; otherwise resident/stream. */
} h3_weight_plan;
const char *h3_weight_mode_name(int mode);
int h3_weight_options_read(h3_weight_options *out,int force_stream,char *error,size_t size);
int h3_weight_options_equal(const h3_weight_options *a,const h3_weight_options *b);
int h3_weight_plan_build(h3_weight_plan *out,h3_weight_options options,
    uint64_t active_mask,uint64_t block_bytes,uint64_t available,uint64_t live,
    uint64_t future,uint64_t reserve,char *error,size_t size);
/* Strictly decreases a positive ceiling, reaching zero within seven retries. */
int h3_weight_retry_cap(unsigned count);
/* Existing packed descriptor ABI: payload, aligned scale plane and global slot. */
int h3_weight_packed_size(uint32_t rows,uint32_t columns,int mode,
    uint64_t *scale,uint64_t *global,uint64_t *bytes);
#ifdef __cplusplus
}
#endif
#endif
