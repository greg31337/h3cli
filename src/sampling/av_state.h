#ifndef H3_AV_STATE_H
#define H3_AV_STATE_H
#include "src/h3.h"
#include "src/host.h"
#include "src/sampling/bridge.h"

struct h3_av_state {
    h3_av_state_info info;
    float *video;
    float *audio;
};

/* Internal allocation accepts only released target geometry. */
int h3_av_state_shape(int width, int height, int frames, h3_av_state_info *info);
int h3_geometry_valid(int width,int height,int profile);
int h3_av_state_shape_profile(int width,int height,int frames,int profile,h3_av_state_info *info);
h3_av_state *h3_av_state_new_profile(int width,int height,int frames,int profile,uint64_t seed,
    const uint8_t signature[32]);
h3_av_state *h3_av_state_new(int width, int height, int frames, uint64_t seed,
                             const uint8_t signature[32]);
/* Reference-only local identity: version + file stat metadata, no weight scan.
 * Current AV files use this metadata identity on both backends. */
int h3_av_state_metadata_signature(const char *model_dir,int ref2va,
                           uint8_t signature[32],char *error,size_t size);
int h3_av_state_validate_continuation(const h3_av_state *state,
    int width, int height, int frames, int context,
    const uint8_t signature[32], h3_denoise_prefix *prefix,
    char *error, size_t size);
/* Inputs must already contain the COMPLETE normal target noise. */
int h3_av_state_insert_prefix(const h3_av_state *source,
    const h3_av_state_info *target, h3_denoise_prefix prefix,
    float *video, float *audio, int augment_video);
/* T019/T020 preparation only: input is the complete ordinary video noise.
 * Does not initialize audio or run the sampler. Source and suffix are intact. */
int h3_av_state_insert_bridge_video(const h3_av_state *source,
    const h3_av_state_info *target, const h3_bridge_profile *profile,
    float initial_sigma, float *video);
/* Joint bridge initialization from complete ordinary AV target noise. */
int h3_av_state_insert_bridge(const h3_av_state *source,
    const h3_av_state_info *target, const h3_bridge_profile *profile,
    float initial_sigma_v, float initial_sigma_a, float *video, float *audio);
void h3_av_state_fingerprint(const h3_av_state *state, uint8_t digest[32]);
double h3_av_now(void);
#endif
