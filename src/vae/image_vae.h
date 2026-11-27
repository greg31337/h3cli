#ifndef H3_IMAGE_VAE_H
#define H3_IMAGE_VAE_H
#include "src/weights/safetensors.h"
#include "src/vae/video_vae.h"
#include "src/vae/video_encoder.h"

#define H3_STILL_MAX_PIXELS (4096 * 4096)
#define H3_STILL_MAX_DIMENSION 8192
#define H3_STILL_DECODER_RECIPE "stock-video-repeat5-frame3-f32-v1"
/* The schema-2 compatibility digest binds the normalized latent format and
 * stock VAE whitening, not the retired F16 image encoder's tensor bytes. */
int h3_still_vae_compatibility(const char *weight_directory, char digest[65],
    char *error, size_t error_size);
int h3_still_geometry(int latent_h, int latent_w, char *, size_t);
int h3_still_protect_vae(const char *directory, const char *output,
    const char *latent_output, char *error, size_t error_size);

/* Owned channel-major F32 [24,1,h,w]. No video/audio state semantics. */
struct h3_still_latent {
    int height, width;
    float *values;
    char compatibility_sha256[65];
    int reference_model; /* 0 FL2VA, 1 Ref2VA: selects the stock VAE on re-decode. */
};
int h3_still_latent_load(const char *, h3_still_latent *, char *, size_t);
int h3_still_latent_save(const char *, const h3_still_latent *, char *, size_t);
void h3_still_latent_free(h3_still_latent *);
/* Shared preflight and borrowed-latent delivery for generation. */
int h3_still_options(const h3_params *, char *, size_t);
h3_result *h3_decode_still_values(const h3_still_latent *, const char *,
    const h3_decode_options *, char *, size_t);
/* Batch-local still decoder, never shared with a video decoder or live DiT.
 * All items must use the same VAE identity and latent canvas. Zero initialize. */
typedef struct {
    h3_video_vae_decoder *decoder;
    char *directory;
    int height, width;
    char compatibility[65];
} h3_still_decode_cache;
void h3_still_decode_cache_clear(h3_still_decode_cache *cache);
h3_result *h3_decode_still_values_cached(const h3_still_latent *, const char *,
    const h3_decode_options *, h3_still_decode_cache *, char *, size_t);
#endif
