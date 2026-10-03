#ifndef H3_SAMPLER_STATE_H
#define H3_SAMPLER_STATE_H

#include "src/h3.h"
#include "src/denoise/adaptive_cache.h"
#include "src/host.h"
#include "src/conditioning/text_encoder.h"
#include "src/sampling/bridge.h"
#include "src/media/refvideo.h"
#include "src/upscale/upscale.h"

#define H3_SAMPLE_VERSION 2
#define H3_SAMPLE_BACKEND_VERSION 1

/* Portable derived BF16 tensors. IDs: 1 refined text, 2 final AdaLN,
 * 100..149 block AdaLN. Runtime objects and immutable weights are excluded. */
#define H3_PREPARED_MAX 52
typedef struct {
    uint32_t id;
    size_t elements;
    uint16_t *values;
} h3_prepared_tensor;
typedef struct {
    uint32_t version;
    uint8_t key[32];
    size_t count;
    h3_prepared_tensor tensors[H3_PREPARED_MAX];
} h3_prepared_cache;

/* Canonical CPU state. All pointers are owned. No Metal objects or model
 * weights are part of a checkpoint. next_step is a completed-transition count. */
struct h3_sampler_state {
    uint32_t version;
    uint32_t quant_version;
    uint32_t attention_version, attention_plan;
    uint32_t metal_attention_version;
    uint64_t ane_decided; /* Bit per block; timing decisions freeze after first encounter. */
    uint32_t ane_rows[50];
    uint32_t sampler_mode; /* 0 = CPU-state Metal, 1 = GPU-state Metal */
    int total_steps, next_step, reuse_interval;
    int last_evaluated, previous_evaluated;
    h3_sigma_schedule sigmas;
    uint8_t selected[H3_MAX_STEPS];
    size_t video_elements, audio_elements;
    float *video, *audio;
    float *last_video_velocity, *previous_video_velocity;
    float *last_audio_velocity, *previous_audio_velocity;
    float *original_video_noise, *original_audio_noise;
    h3_rng video_rng, audio_rng;
    uint64_t video_random_count, audio_random_count;
    uint32_t rng_version;

    h3_params params; /* scalar generation options; no borrowed pointers */
    int ref2va, conditioned, continuation, context_frames;
    int render_width, render_height, aligned_frames;
    int latent_t, latent_h, latent_w, audio_t;
    float spatial_rope_scale;
    h3_text_embedding text;
    float *condition_video, *condition_audio;
    size_t condition_video_elements, condition_audio_elements;
    h3_layout_ref *references;
    size_t reference_count;
    h3_layout layout;
    h3_bridge_profile bridge;
    char *prompt;
    /* Length-delimited provenance records, documented in sampler-state.md. */
    uint8_t *provenance;
    size_t provenance_bytes;
    uint32_t *token_ids;
    size_t token_count;
    uint32_t *presentation_positions;
    uint64_t *presentation_spans;
    size_t presentation_span_count;
    uint8_t model_fingerprint[32], av_signature[32];
    uint8_t source_fingerprint[32], video_tail_hash[32], audio_tail_hash[32];
    float clean_coefficient, noise_coefficient;
    int audio_preservation;
    char *build_id, *environment;
    uint32_t backend_version;
    uint32_t refvideo_pipeline;
    h3_device_info device;
    uint16_t *gpu_last_video, *gpu_previous_video;
    uint16_t *gpu_last_audio, *gpu_previous_audio;
    uint32_t execution_version, core_forward_count, core_residual_ready;
    size_t core_rows, core_columns, core_elements;
    uint16_t *core_residual;
    uint32_t adaptive_version;
    h3_adaptive_history adaptive_history;
    size_t adaptive_elements;
    uint16_t *adaptive_anchor, *adaptive_delta;
    uint32_t reduction_enabled, reduction_active, reduction_begin, reduction_end;
    uint32_t reduction_early_steps, reduction_early_end;
    float reduction_scale;
    size_t full_sequence, reduced_sequence;
    h3_prepared_cache prepared;
    uint32_t resume_count, resume_step, resume_format;
    uint8_t resume_hash[32], loaded_hash[32];
    h3_upscale_record upscale;
};

void h3_prepared_cache_free(h3_prepared_cache *cache);
const h3_prepared_tensor *h3_prepared_find(const h3_prepared_cache *cache,
    uint32_t id, size_t elements);
int h3_sampler_prepared_key(const h3_sampler_state *state, uint8_t key[32]);
size_t h3_sampler_prepared_bytes(const h3_sampler_state *state);
void h3_sampler_log(const h3_sampler_state *state, const char *action,
    uint64_t bytes, double seconds);
/* The CLI restores this selector from the checkpoint. API callers must supply
 * the matching selector for video checkpoints; other generation fields remain
 * checkpoint-authoritative. No original reference file is opened here. */

/* Also used by the ordinary Euler wrapper, without generation metadata. */
h3_sampler_state *h3_sampler_state_create(const h3_sigma_schedule *sigmas,
    size_t video_elements, size_t audio_elements, int reuse_interval);
int h3_sampler_layout_validate(const h3_sampler_state *state,char *error,size_t size);
/* Validate adaptive counts and committed history before loading large tensors. */
int h3_sampler_adaptive_metadata_validate(const h3_sampler_state *state,char *error,size_t size);
int h3_sampler_state_validate(const h3_sampler_state *state,
    char *error, size_t size);
int h3_sampler_state_capture_effective(h3_sampler_state *state, const char *model_dir, const char *transformer,
    const h3_device_info *device, const h3_params *params, const char *prompt,
    const h3_text_embedding *text, const h3_layout *layout,
    const h3_layout_ref *references, size_t reference_count,
    const float *condition_video, size_t video_elements,
    const float *condition_audio, size_t audio_elements, int conditioned,
    const uint8_t av_signature[32], char *error, size_t size);
int h3_sampler_capture_upscale(h3_sampler_state *state,const h3_sampler_state *parent,
    const char *model_dir,const h3_device_info *device,const h3_params *params,const h3_layout *layout,
    const h3_layout_ref *references,const float *video,size_t video_elements,
    const float *audio,size_t audio_elements,const uint8_t av_signature[32],char *error,size_t size);
int h3_sampler_state_compatible_effective(const h3_sampler_state *state,
    const char *model_dir, const char *transformer, const h3_device_info *device,
    char *error, size_t size);
int h3_sampler_checkpoint_options(const h3_params *params,
    const h3_device_info *device, char *error, size_t size);
/* Content fingerprints cache individual file hashes using stat identity,
 * size, nanosecond mtime AND ctime; cache lives under outputs/, not weights. */
int h3_sampler_model_fingerprint_effective(const char *model_dir, const char *transformer,
    int ref2va, uint8_t digest[32], char *error, size_t size);
/* Current local metadata identity; strict weight verification is separate. */
int h3_sampler_model_metadata_effective(const char *model_dir,const char *transformer,
    int ref2va,uint8_t digest[32],char *error,size_t size);
void h3_sampler_hash(const void *data, size_t bytes, uint8_t digest[32]);
/* Current conditioning component metadata, using canonical model traversal. */
int h3_sampler_component_metadata(const char *model_dir,const char *transformer,
    int ref2va,const char *component,uint8_t digest[32],char *error,size_t size);
int h3_sampler_source_fingerprint(const char *path, uint8_t digest[32], uint64_t *bytes);
char *h3_sampler_environment(void);
/* Bounded planning read. Full state validation still precedes inference. */
int h3_sampler_file_validate(const char *path,int source,char *error,size_t size);
int h3_sampler_model_requirements(const char *path, int source, int *ref2va,
    int *upscale, char *error, size_t size);

#endif
