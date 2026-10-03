#ifndef H3_HOST_H
#define H3_HOST_H

#include <stddef.h>
#include <stdint.h>

#define H3_CANVAS_MULTIPLE 32
#define H3_MAX_PIXELS (768 * 1344)
#define H3_FPS 24
#define H3_AUDIO_LATENT_FPS 40
#define H3_VAE_SPATIAL_RATIO 16
#define H3_VIDEO_SIGMA_SHIFT 12.0
#define H3_AUDIO_SIGMA_SHIFT 3.0
#define H3_MAX_STEPS 1000

typedef struct {
    int frame_count;
    int video_t;
    int audio_t;
} h3_temporal_shape;

typedef struct {
    int video_prefix_t;
    int audio_prefix_t;
} h3_denoise_prefix;

typedef enum {
    H3_ROW_GENERATED_VIDEO, H3_ROW_GENERATED_AUDIO,
    H3_ROW_PRESERVED_VIDEO, H3_ROW_PRESERVED_AUDIO,
    /* Ten quantized nonzero strengths per stream. The original four class
     * values, and all text/reference modality tags, retain their meaning. */
    H3_ROW_BRIDGE_VIDEO_FIRST = 4,
    H3_ROW_BRIDGE_AUDIO_FIRST = 14,
    H3_TARGET_ROW_CLASSES = 24
} h3_target_row_class;

int h3_continuation_context(int frames, h3_denoise_prefix *prefix);
/* Start of a video temporal row in frame units, using the packed RoPE grid's
 * repeating 1,4,4,4,4 frame spans. Returns -1 on invalid/overflowing input. */
int h3_video_time_boundary_frames(int temporal_index);
float h3_prefix_timestep(h3_target_row_class kind, float sigma_v, float sigma_a);
h3_target_row_class h3_prefix_row_class(h3_denoise_prefix prefix,
    int audio, size_t row, int latent_h, int latent_w, int audio_t);
void h3_prefix_mask_velocity(h3_denoise_prefix prefix, int video_t,
    int latent_h, int latent_w, int audio_t, float *video, float *audio);
int h3_prefix_euler_step(h3_denoise_prefix prefix, int video_t,
    int latent_h, int latent_w, int audio_t, float *video, float *audio,
    const float *video_velocity, const float *audio_velocity,
    float sigma_v, float next_v, float sigma_a, float next_a);

typedef struct {
    double t;
    double h;
    double w;
} h3_position;

typedef enum {
    H3_SEG_TEXT,
    H3_SEG_COND,
    H3_SEG_REF_IMAGE,
    H3_SEG_REF_AUDIO,
    H3_SEG_AUDIO,
    H3_SEG_VIDEO
} h3_segment_kind;

typedef struct {
    size_t start;
    size_t stop;
    h3_segment_kind kind;
} h3_segment;

typedef enum {
    H3_LAYOUT_REF_IMAGE,
    H3_LAYOUT_REF_AUDIO,
    H3_LAYOUT_REF_VIDEO
} h3_layout_ref_kind;

typedef struct {
    h3_layout_ref_kind kind;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
} h3_layout_ref;

typedef struct {
    int text_len;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    int frame_count;
    const int *keyframes;
    size_t keyframe_count;
    const h3_layout_ref *references;
    size_t reference_count;
} h3_layout_spec;

typedef struct {
    size_t seq_len;
    h3_segment *segments;
    size_t segment_count;
    h3_position *positions;
    size_t img_cond_rows;
    size_t img_target_rows;
    size_t audio_cond_rows;
    size_t audio_target_rows;
    int signature[5];
    h3_denoise_prefix prefix;
    int frozen_audio; /* Explicit upscale stage; full clean audio, no video prefix. */
    /* Optional borrowed temporal plan; DiT loading takes its own copy. */
    const struct h3_bridge_profile *bridge;
} h3_layout;

typedef struct {
    int steps;
    float video[H3_MAX_STEPS + 1];
    float audio[H3_MAX_STEPS + 1];
} h3_sigma_schedule;

typedef struct {
    uint64_t state;
    uint64_t increment;
    float spare;
    int has_spare;
} h3_rng;

int h3_align_frame_count(int requested);
int h3_video_latent_t(int frame_count);
/* Raw continuous causal encoder geometry; not released Ref2VA video geometry. */
int h3_video_encoder_latent_t(int frame_count);
h3_temporal_shape h3_temporal(int requested_frames);
void h3_latent_canvas(int width, int height, int *latent_w, int *latent_h);
int h3_adapt_canvas(int width, int height, int *adapted_w, int *adapted_h);
/* Compatibility wrapper: max_short_edge=0 selects match; 2048 selects max.
 * Other values are rejected. New callers should use the checked resolver. */
int h3_reference_image_canvas(int width, int height,
                              int target_width, int target_height,
                              int max_short_edge,
                              int *adapted_w, int *adapted_h);
/* Shared image geometry: mode is h3_reference_image_size. max/high fix the
 * short/long edge to 2048, allow upscaling and require a 1:4..4:1 source ratio. */
#define H3_REFERENCE_IMAGE_MAX_PATCHES 65536u
typedef struct {
    int width, height;
    size_t patches, tokens;
} h3_reference_image_shape;
int h3_reference_image_resolve(int width, int height, int target_width,
    int target_height, int mode, h3_reference_image_shape *shape,
    char *error, size_t error_size);
const char *h3_reference_image_size_name(int mode);
/* Ref2VA video references use the normal target-style canvas, except that a
 * smaller source is never enlarged. */
int h3_reference_video_canvas(int width, int height,
                              int *adapted_w, int *adapted_h);

double h3_time_shift_sigma(double sigma, double from_shift, double to_shift);
double h3_time_shift_slope(double sigma, double from_shift, double to_shift);
int h3_schedule_build(int steps, h3_sigma_schedule *schedule);
/* Released linear base grid: evaluations model forwards plus terminal zero. */
int h3_serving_schedule_build(int evaluations, h3_sigma_schedule *schedule);

int h3_still_layout_build(const h3_layout_spec *, h3_layout *, char *, size_t);
int h3_layout_build(const h3_layout_spec *spec, h3_layout *layout,
                    char *error, size_t error_size);
void h3_layout_free(h3_layout *layout);
const char *h3_segment_name(h3_segment_kind kind);

void h3_rng_seed(h3_rng *rng, uint64_t seed);
uint32_t h3_rng_u32(h3_rng *rng);
float h3_rng_normal(h3_rng *rng);
void h3_rng_fill_normal(h3_rng *rng, float *values, size_t count);

/* Resize interleaved RGB24 frames with Accelerate/vImage high-quality
 * resampling. The caller owns *output. Identity geometry still returns an
 * independent copy. */
int h3_resize_rgb24_high_quality(const uint8_t *input, int frames,
                                 int input_width, int input_height,
                                 int output_width, int output_height,
                                 uint8_t **output);

int h3_res_step(float *output, const float *sample, const float *denoised,
                const float *old_denoised, size_t count,
                const float *sigmas, int step, int total_steps);
int h3_euler_velocity_step(float *sample, const float *velocity, size_t count,
                           float sigma, float sigma_next);

#endif
