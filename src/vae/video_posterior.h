#ifndef H3_VIDEO_POSTERIOR_H
#define H3_VIDEO_POSTERIOR_H

#include "src/vae/video_encoder.h"
#include <stdint.h>

typedef struct {
    int vae_frames;
    int padding_frames;
    int padded_frames;
    int chunks;
    int raw_time;
    int time;
} h3_video_chunk_plan;

int h3_video_chunk_plan_build(int vae_frames, h3_video_chunk_plan *plan);
/* Return an owned channel-major 17-frame chunk. source_frames is the original
 * RGB channel stride; padding repeats the last VAE-selected frame. */
float *h3_video_chunk_extract(const float *pixels, int source_frames,
    int height, int width, const h3_video_chunk_plan *plan, int chunk);
/* Concatenate raw chunk moments, then remove exactly three trailing slices.
 * The borrowed chunks are never modified. GPU stats belong to the caller. */
int h3_video_moments_join(const h3_video_moments *chunks, int count,
    int vae_frames, h3_video_moments *output, char *error, size_t error_size);

/* A fresh seed-42 CPU PyTorch-compatible contiguous F32 draw on every call.
 * Dedicated to conditioning; does not read or modify the request's h3_rng. */
int h3_video_posterior_epsilon(float *values, size_t count);
/* Explicit epsilon isolates posterior mathematics from RNG differences.
 * moments=[mean,logvar] with count values per half. Output may alias epsilon. */
int h3_video_posterior_sample(const float *moments, const float *epsilon,
    size_t count, float *sample);
uint16_t h3_video_f32_to_f16(float value);
float h3_video_f16_to_f32(uint16_t value);
/* FP16 round-trip, then F32 per-channel normalization. No sampling here. */
int h3_video_posterior_normalize(const float *sample, size_t channel_elements,
    const float mean[24], const float deviation[24], float *normalized);

#endif
