#ifndef H3_TINY_VAE_H
#define H3_TINY_VAE_H
#include "src/vae/video_vae.h"

typedef struct h3_tiny_vae h3_tiny_vae;
/* Validation does not create a device or load unrelated model components. */
int h3_tiny_vae_validate(const char *path, char digest[65], char *error, size_t size);
h3_tiny_vae *h3_tiny_vae_load(const char *path, char *error, size_t size);
void h3_tiny_vae_free(h3_tiny_vae *decoder);
const char *h3_tiny_vae_digest(const h3_tiny_vae *decoder);
/* Cache identity includes the device and CUDA convolution/workspace policy. */
int h3_tiny_vae_cache_matches(const h3_tiny_vae *decoder, const char *digest);
int h3_tiny_vae_stream(h3_tiny_vae *decoder, const float *latent, int time,
    int height, int width, int batch, int stop_frame,
    h3_video_batch_callback callback, void *opaque,
    h3_video_vae_progress progress, void *progress_opaque, char *error, size_t size);
int h3_tiny_vae_decode(h3_tiny_vae *decoder, const float *latent, int time,
    int height, int width, int selected_frame, h3_video_frames *frames,
    h3_video_vae_progress progress, void *opaque, char *error, size_t size);
int h3_tiny_vae_frames(int time);
/* Conservative activation/scratch allowance, not a measured allocation. */
uint64_t h3_tiny_vae_memory_reserve(int height, int width, int batch);
#endif
