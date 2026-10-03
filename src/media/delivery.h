#ifndef H3_DELIVERY_H
#define H3_DELIVERY_H
#include "src/h3.h"
#include "src/vae/tiny_vae.h"
#include "src/vae/audio_vae.h"

/* Mutates only the decoded waveform (prefix compaction), never AV latents.
 * Exactly one decoder is required. All callbacks borrow their buffers. */
int h3_deliver_video(h3_tiny_vae *tiny, h3_video_vae_decoder *full,
    const float *latent, int time, int height, int width,
    const h3_presentation *presentation, h3_audio_waveform *audio,
    const h3_decode_options *options, char *error, size_t size);
/* Consumes an owned complete AV state on success or failure. */
h3_result *h3_decode_av_owned(const char *model_dir,h3_av_state *state,
    const h3_presentation *presentation,const h3_decode_options *options,char *error,size_t size);
int h3_presentation_validate(const h3_presentation *p,
    const h3_av_state_info *info, char *error, size_t size);
/* 1 found, 0 absent, -1 invalid. Invalid metadata is never treated as absent. */
int h3_presentation_load(const char *state_path, const h3_av_state *state,
    h3_presentation *p, char *error, size_t size);
#define H3_PRESENTATION_VERSION 9
const char *h3_preview_vae_path(const char *path);
int h3_preview_vae_options(int enabled, const char *path, char *error, size_t size);
#endif
