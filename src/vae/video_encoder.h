#ifndef H3_VIDEO_ENCODER_H
#define H3_VIDEO_ENCODER_H

#include "src/gpu.h"

#include <stddef.h>

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major normalized F32: [24,time,height,width]. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_video_latent;

typedef struct {
    int time;
    int height;
    int width;
    /* Channel-major F32 [48,T,H,W]: mean then log-variance, before sampling. */
    float *values;
    h3_gpu_stats gpu_stats;
} h3_video_moments;

/* Internal spatial plan, in pixels. Owned arrays; release with the paired free. */
typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} h3_video_tile_axis;
int h3_video_tile_axis_build(int extent, h3_video_tile_axis *axis,
    char *error, size_t error_size);
void h3_video_tile_axis_free(h3_video_tile_axis *axis);

/* Progress returns 0 to continue, nonzero to cancel at this boundary.
 * (0, 0) is an indeterminate setup/loading checkpoint, not completion. */
typedef int (*h3_video_encoder_progress)(int completed_tiles,
                                          int total_tiles, void *opaque);

/* Encode channel-major RGB [3,T,H,W] pixels in [0,1]. Spatial axes must be
 * multiples of 16. The released 256px/64px overlap tiling is preserved. */
int h3_video_vae_encode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *pixels, int frames, int height, int width,
                        h3_video_encoder_progress progress, void *progress_opaque,
                        h3_video_latent *output,
                        char *error, size_t error_size);
void h3_video_latent_free(h3_video_latent *latent);

/* Raw continuous causal CNN with spatial stitching of all 48 moments.
 * No temporal wrapper, posterior selection, rounding or latent normalization. */
int h3_video_vae_encode_moments(const char *weight_directory,
    const char *shader_source_path, const float *pixels, int frames,
    int height, int width, h3_video_encoder_progress progress, void *progress_opaque,
    h3_video_moments *output, char *error, size_t error_size);
/* Borrow row-major tile pointers, each channel-major [48,T,tileH/16,tileW/16]. */
int h3_video_moments_stitch(float **tiles, int time, int height, int width,
    h3_video_moments *output, char *error, size_t error_size);
void h3_video_moments_free(h3_video_moments *moments);

/* Released temporal moments: independent 17-frame causal chunks, then drop 3.
 * source_frames is the normalized RGB channel stride; only the legal
 * vae_frames prefix is encoded. Input remains borrowed and unchanged. */
int h3_ref2va_video_vae_moments(const char *weight_directory,
    const char *shader_source_path, const float *pixels, int source_frames,
    int vae_frames, int height, int width, h3_video_encoder_progress progress,
    void *progress_opaque, h3_video_moments *output, char *error, size_t error_size);
/* Complete released video conditioning: moments, fresh seed-42 posterior,
 * FP16 round-trip, F32 latent normalization. Returns owned [24,T,H,W]. */
int h3_ref2va_video_vae_encode(const char *weight_directory,
    const char *shader_source_path, const float *pixels, int source_frames,
    int vae_frames, int height, int width, h3_video_encoder_progress progress,
    void *progress_opaque, h3_video_latent *output, char *error, size_t error_size);

#endif
