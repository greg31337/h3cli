#ifndef H3_WEIGHTS_H
#define H3_WEIGHTS_H

#include "src/gpu.h"
#include "src/weights/safetensors.h"

#include <stddef.h>
#include <stdint.h>

typedef struct h3_weight_store h3_weight_store;

/* Open every safetensors header in a component directory without reading
 * tensor payloads. */
h3_weight_store *h3_weight_store_open(const char *directory,
                                      char *error, size_t error_size);
/* Explicit component file; never searches adjacent video shards. */
h3_weight_store *h3_weight_store_open_file(const char *path, char *error, size_t size);
/* Image component only: finite IEEE F16 -> F32, <= 8 MiB staging. */
h3_gpu_tensor *h3_weight_load_f16_f32(const h3_weight_store *store, h3_gpu *gpu,
    const char *name, int ndim, const uint64_t *shape, char *error, size_t size);
float h3_weight_f16_to_f32(uint16_t bits);
void h3_weight_store_free(h3_weight_store *store);
size_t h3_weight_store_shards(const h3_weight_store *store);

const h3_st_tensor *h3_weight_find(const h3_weight_store *store,
                                   const char *name,
                                   const h3_st_header **header);

/* Validate an exact BF16 shape, allocate a shared Metal buffer, and read the
 * payload directly into that buffer with no intermediate host allocation. */
h3_gpu_tensor *h3_weight_load_bf16(const h3_weight_store *store, h3_gpu *gpu,
                                   const char *name, int ndim,
                                   const uint64_t *shape,
                                   char *error, size_t error_size);
h3_gpu_tensor *h3_weight_load_f32(const h3_weight_store *store, h3_gpu *gpu,
                                  const char *name, int ndim,
                                  const uint64_t *shape,
                                  char *error, size_t error_size);

#endif
