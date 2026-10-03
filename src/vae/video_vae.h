#ifndef H3_VIDEO_VAE_H
#define H3_VIDEO_VAE_H

#include "src/gpu.h"

#include <stddef.h>

typedef struct {
    int frames;
    int height;
    int width;
    /* Frame-major, row-major interleaved RGB F32 in [0,1]. */
    float *rgb;
    h3_gpu_stats gpu_stats;
} h3_video_frames;

/* Progress returns 0 to continue, nonzero to cancel at this boundary.
 * Decode/preview counters accumulate block work across all spatial tiles and
 * temporal chunks in that invocation; completion follows output assembly. */
typedef int (*h3_video_vae_progress)(int completed_blocks, int total_blocks,
                                      void *opaque);
/* Borrowed frame-major RGB [0,1]; zero continues, nonzero cancels. */
typedef int (*h3_video_batch_callback)(const float *rgb, int first, int count,
                                     int width, int height, void *opaque);

typedef struct h3_video_vae_decoder h3_video_vae_decoder;

/* Resident tiled decoder used by live denoising previews. It decodes one
 * representative middle frame per preview call and can then produce the final
 * complete video without loading the 9.7 GiB weight set again. */
/* Returns 0 with an actionable error for unsupported explicit tile values. */
int h3_video_vae_tile_pixels(int pixel_height, int pixel_width,
                             char *error, size_t error_size);

h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size);
int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size);
int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_frames *output,
                        char *error, size_t error_size);
/* Callbacks belong to this invocation, never to the cached decoder. */
int h3_video_vae_decoder_preview_progress(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size);
int h3_video_vae_decoder_decode_progress(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size);
void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder);
int h3_video_vae_decoder_stream_progress(h3_video_vae_decoder *decoder,
    const float *normalized_latent, int latent_time,
    h3_video_batch_callback callback, void *opaque,
    h3_video_vae_progress progress, void *progress_opaque,
    char *error, size_t error_size);

/* Decode aligned H3 temporal chunks, using the released overlap/blend rules in
 * time and, by default, 256-pixel overlapping tiles in space.
 * H3_VAE_TILE_PIXELS accepts numeric overrides or "auto" for the
 * geometry-based tile heuristic. Policy is fixed when a resident decoder loads.
 * The two-token mode remains
 * available for bounded component diagnostics. */
int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size);
/* Separate resident weight-loading and decoding progress. The original entry
 * point above sends both sequences to its single callback.
 * The single-tile streaming path interleaves loading/evaluation and reports
 * only decode progress. */
int h3_video_vae_decode_phased(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress load_progress,
                        h3_video_vae_progress decode_progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size);
void h3_video_frames_free(h3_video_frames *frames);

#endif
