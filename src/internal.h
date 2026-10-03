#ifndef H3_INTERNAL_H
#define H3_INTERNAL_H

#include "src/h3.h"
#include "src/weights/lora.h"
#include "src/host.h"
#include "src/conditioning/text_encoder.h"

#include <stdarg.h>

int h3_request_params_valid(const h3_params *params,const h3_device_info *device,
                            char *error,size_t error_size);

struct h3_dit;
struct h3_video_vae_decoder;
struct h3_tiny_vae;

struct h3_ctx {
    char *model_dir;
    h3_lora_selection *lora;
    h3_lora_variant lora_variants[2];
    char error[512];
    h3_device_info device;
    h3_model_info model;
    uint8_t av_signature[2][32];
    int av_signature_ready[2];
    int cache_enabled;
    char *conditioning_key;
    size_t conditioning_tokens;
    size_t conditioning_width;
    uint16_t *conditioning_values;
    uint8_t *conditioning_tags;
    h3_text_diagnostics *conditioning_diagnostics;
    float *conditioning_video_rows;
    size_t conditioning_video_elements;
    float *conditioning_audio_rows;
    size_t conditioning_audio_elements;
    h3_layout_ref *conditioning_references;
    size_t conditioning_reference_count;
    int conditioning_present;
    char *dit_key;
    struct h3_dit *dit;
    uint8_t dit_sampler_key[32];
    int dit_sampler_key_ready;
    char *video_decoder_key;
    struct h3_video_vae_decoder *video_decoder;
    struct h3_tiny_vae *tiny_decoder;
};

h3_result *h3_generate_upscale_state(h3_ctx *ctx,h3_sampler_state *state,const h3_upscale_options *options);

void h3_set_error(h3_ctx *ctx, const char *format, ...)
    __attribute__((format(printf, 2, 3)));

/* Deterministic cache keys, also exercised by host regression tests. */
/* Internal packing boundary, shared with host regression tests. */
int h3_augment_conditions(const h3_params *params, int ref2va,
    int render_width, int render_height, const h3_layout_ref *references,
    float *video, size_t video_elements, float *audio, size_t audio_elements);
char *h3_conditioning_key(const char *prompt, const h3_params *params,
    int render_width, int render_height, int ref2va);
char *h3_prepared_key(const char *conditioning, const h3_params *params,
    int render_width, int render_height);

#endif
