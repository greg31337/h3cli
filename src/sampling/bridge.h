#ifndef H3_BRIDGE_H
#define H3_BRIDGE_H

#include "src/h3.h"
#include "src/host.h"

enum {
    H3_BRIDGE_LEVELS = 10,
    H3_BRIDGE_MAX_VIDEO_T = 107,
    H3_BRIDGE_MAX_AUDIO_T = 604
};

/* Owned, dimension-independent temporal plan. No token-sized float masks.
 * Class ids are h3_target_row_class values. Only active classes need timestep
 * vectors; quantization uses ten equal subdivisions of max_strength. */
typedef struct h3_bridge_profile {
    h3_denoise_prefix prefix;
    int context_frames;
    int video_bridge_t, video_exact_t;
    int bridge_frames;
    int audio_bridge_t, audio_exact_t;
    h3_bridge_profile_type type;
    float max_strength;
    int class_count;
    uint8_t active[H3_TARGET_ROW_CLASSES];
    float class_mask[H3_TARGET_ROW_CLASSES];
    uint8_t video_classes[H3_BRIDGE_MAX_VIDEO_T];
    uint8_t audio_classes[H3_BRIDGE_MAX_AUDIO_T];
} h3_bridge_profile;

const char *h3_bridge_profile_name(h3_bridge_profile_type type);
/* Normalized time [0,1); time >= 1 is exact. Invalid input returns NaN. */
float h3_bridge_strength(h3_bridge_profile_type type, float maximum, double time);
/* Warning is optional and separate from an error: one exact video row is
 * valid. On failure the destination is unchanged. Context is 39+51*k <= 362. */
int h3_bridge_profile_build(int context_frames, int video_steps, float maximum,
    h3_bridge_profile_type type, h3_bridge_profile *profile,
    char *warning, size_t warning_size, char *error, size_t error_size);
int h3_bridge_profile_valid(const h3_bridge_profile *profile);
/* Local row index within the authoritative target VIDEO or AUDIO segment.
 * Stereo audio is [left ticks][right ticks], not interleaved ticks. */
h3_target_row_class h3_bridge_row_class(const h3_bridge_profile *profile,
    int audio, size_t row, int latent_h, int latent_w, int audio_t);
float h3_bridge_class_sigma(const h3_bridge_profile *profile,
    h3_target_row_class kind, float sigma_v, float sigma_a);
float h3_bridge_class_timestep(const h3_bridge_profile *profile,
    h3_target_row_class kind, float sigma_v, float sigma_a);
/* Call once on raw predicted/reused velocity, before Euler. Fractional
 * scaling is NOT idempotent (unlike the existing hard mask). */
int h3_bridge_mask_velocity(const h3_bridge_profile *profile, int video_t,
    int latent_h, int latent_w, int audio_t, float *video, float *audio);
typedef struct {
    size_t elements[H3_TARGET_ROW_CLASSES];
    size_t changed[H3_TARGET_ROW_CLASSES];
    double raw_square[H3_TARGET_ROW_CLASSES];
    double scaled_square[H3_TARGET_ROW_CLASSES];
    double update_square[H3_TARGET_ROW_CLASSES];
} h3_bridge_step_stats;
/* Raw velocities are scaled exactly once in place. Exact latent positions are
 * skipped entirely, preserving every bit. Optional statistics describe this
 * transition, including rounding in the actual F32 Euler updates. */
int h3_bridge_euler_step(const h3_bridge_profile *profile, int video_t,
    int latent_h, int latent_w, int audio_t, float *video, float *audio,
    float *video_velocity, float *audio_velocity,
    float sigma_v, float next_v, float sigma_a, float next_a,
    h3_bridge_step_stats *stats);
int h3_bridge_check_exact(const h3_bridge_profile *profile, int video_t,
    int latent_h, int latent_w, int audio_t, const float *video, const float *audio,
    const float *initial_video, const float *initial_audio);
/* H3 rectified-flow interpolation, with separate F32 products. No RNG draws.
 * Endpoints preserve clean/noise bits; invalid sigma returns NaN. Exact video
 * uses its original 0.999/0.001 expression instead of this helper. */
float h3_flow_mix(float clean, float noise, float sigma);

#endif
