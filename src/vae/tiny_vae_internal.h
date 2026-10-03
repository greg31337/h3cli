#ifndef H3_TINY_VAE_INTERNAL_H
#define H3_TINY_VAE_INTERNAL_H
#include "src/vae/tiny_vae.h"
#include "src/weights/safetensors.h"
typedef struct { char name[80]; int ndim; uint64_t shape[4]; } h3_tiny_weight;
int h3_tiny_weights(h3_tiny_weight weights[64]);
void *h3_tiny_backend_load(const char *path, char *error, size_t size);
void h3_tiny_backend_free(void *backend);
void h3_tiny_backend_reset(void *backend);
void h3_tiny_backend_profile(void *backend);
/* Input/output NHWC; temporal batch is the leading dimension. Output has 4*T
 * frames and 16*H,16*W pixels. The caller frees output. History stays on device. */
int h3_tiny_backend_chunk(void *backend, const float *input, int t, int h, int w,
                          float **output, char *error, size_t size);
#endif
