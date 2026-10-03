#ifndef H3_MEMORY_H
#define H3_MEMORY_H

#include <stddef.h>
#include <stdint.h>
#ifdef __cplusplus
extern "C" {
#endif

#define H3_MIN_AVAILABLE_MEMORY (UINT64_C(10) * 1024 * 1024 * 1024)
#define H3_MAX_PROCESS_MEMORY UINT64_C(110000000000) /* 110 GB, decimal. */

/* Query success and check success are 1. Checks fail closed if monitoring is
 * unavailable. Headroom alone is insufficient when macOS compresses/swaps. */
int h3_memory_available_bytes(uint64_t *bytes);
uint64_t h3_memory_minimum_bytes(void);
int h3_memory_check(uint64_t reserve_bytes, const char *phase,
                    char *error, size_t error_size);
/* Call at a safe boundary, passing the progress callback's cancellation result.
 * Returns 1 to continue, 0 to leave through the caller's normal cleanup path. */
int h3_memory_checkpoint(int cancelled, const char *phase,
                         char *error, size_t error_size);

/* Deterministic, thread-local test injection; NULL restores the OS query.
 * H3_TEST_MIN_AVAILABLE_MEMORY_BYTES overrides the safety floor for tests. */
typedef int (*h3_memory_query_fn)(uint64_t *bytes, void *opaque);
void h3_memory_set_test_query(h3_memory_query_fn query, void *opaque);

typedef struct {
    uint64_t physical_total, available, resident, physical_footprint;
    uint64_t process_compressed, system_compressed, swap_used, system_wired;
    int process_valid, system_valid, swap_valid, available_valid;
} h3_memory_snapshot;
void h3_memory_sample(h3_memory_snapshot *snapshot);
uint64_t h3_memory_limit_bytes(const h3_memory_snapshot *snapshot);
/* GPU bytes overlap the process footprint; compare the maximum, never their
 * sum. This also catches reserved Metal resources before first GPU use. */
int h3_memory_check_gpu(uint64_t reserve_bytes, uint64_t allocated_bytes,
                        const char *phase, char *error, size_t error_size);
int h3_memory_error(const char *error);
typedef void (*h3_memory_snapshot_fn)(h3_memory_snapshot *, void *);
void h3_memory_set_test_snapshot(h3_memory_snapshot_fn query, void *opaque);

#ifdef __cplusplus
}
#endif

#endif
