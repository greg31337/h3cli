#ifndef H3_CLI_STILL_BATCH_H
#define H3_CLI_STILL_BATCH_H
#include "src/h3.h"

typedef struct {
    void (*item)(int index, int count, uint64_t seed, int decoding, void *opaque);
    void (*result)(const h3_result *result, const char *output, void *opaque);
    void *opaque;
} h3_still_batch_hooks;

/* Owned filename. Count 1 preserves the original; larger counts insert
 * -seed-N before the final extension (or append it when there is none). */
char *h3_still_batch_path(const char *path, int count, uint64_t seed);
int h3_still_batch_validate(const h3_params *params, int count, char *error, size_t size);
/* One context, one conditioning/DiT preparation, then one decoder lifetime.
 * An anonymous temporary file stages latents; live host arrays stay bounded. */
int h3_still_batch_run(h3_ctx *ctx, const char *prompt, const h3_params *params,
    int count, const h3_still_batch_hooks *hooks, char *error, size_t size);
#endif
