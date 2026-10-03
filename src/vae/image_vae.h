#ifndef H3_IMAGE_VAE_H
#define H3_IMAGE_VAE_H
#include "src/weights/safetensors.h"
#include "src/vae/video_vae.h"
#include "src/vae/video_encoder.h"

typedef struct {
    int output_slice;
    float mean[24], deviation[24];
    char artifact_sha256[65], compatibility_sha256[65], identity[65];
} h3_image_vae_info;
/* No GPU allocation. Strict full_decoder_v1 metadata and tensor inventory. */
int h3_image_vae_validate(const h3_st_header *, h3_image_vae_info *, char *, size_t);
/* Also hashes the actual file and encoder/normalization compatibility. */
int h3_image_vae_inspect(const char *, h3_image_vae_info *, char *, size_t);
int h3_still_geometry(int latent_h, int latent_w, char *, size_t);

typedef struct h3_image_vae_decoder h3_image_vae_decoder;
h3_image_vae_decoder *h3_image_vae_load(const char *path, const char *shader,
    int latent_h, int latent_w, h3_video_vae_progress, void *, char *, size_t);
const h3_image_vae_info *h3_image_vae_info_get(const h3_image_vae_decoder *);
int h3_image_vae_decode(h3_image_vae_decoder *, const float *normalized,
    h3_video_vae_progress, void *, h3_video_frames *, char *, size_t);
void h3_image_vae_free(h3_image_vae_decoder *);
/* Deterministic posterior mean, checkpoint's own encoder, then full teardown. */
int h3_image_vae_encode(const char *path, const char *shader, const float *pixels,
    int height, int width, h3_video_encoder_progress, void *,
    h3_video_latent *, char *, size_t);

/* Owned channel-major F32 [24,1,h,w]. No video/audio state semantics. */
struct h3_still_latent {
    int height, width;
    float *values;
    char compatibility_sha256[65];
};
int h3_still_latent_load(const char *, h3_still_latent *, char *, size_t);
int h3_still_latent_save(const char *, const h3_still_latent *, char *, size_t);
void h3_still_latent_free(h3_still_latent *);
/* Shared preflight and borrowed-latent delivery for generation. */
int h3_still_options(const h3_params *, char *, size_t);
h3_result *h3_decode_still_values(const h3_still_latent *, const char *,
    const h3_decode_options *, char *, size_t);
#endif
