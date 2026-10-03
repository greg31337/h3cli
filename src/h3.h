/* Public API for the h3cli MiniMax-H3 inference engine. */
#ifndef H3_H
#define H3_H

#include <stddef.h>
#include <stdint.h>
#include "src/backend.h"
#include "src/cuda/cuda_sol_policy.h"
#include "src/media/output.h"

#ifdef __cplusplus
extern "C" {
#endif

#define H3_VERSION "0.1.0-dev"
#define H3_DEFAULT_WIDTH 864
#define H3_DEFAULT_HEIGHT 480
#define H3_DEFAULT_FRAMES 56
#define H3_DEFAULT_STEPS 20
#define H3_DEFAULT_DIT_LAYERS 50
#define H3_MIN_DIT_LAYERS 35

typedef struct h3_ctx h3_ctx;
typedef struct h3_result h3_result;
typedef struct h3_still_latent h3_still_latent;
typedef struct h3_av_state h3_av_state;
typedef struct h3_sampler_state h3_sampler_state;
typedef struct h3_upscale_source h3_upscale_source;
typedef struct h3_upscale_plan h3_upscale_plan;
typedef struct {
    int source_width,source_height,width,height,frames,video_t,audio_t;
    int geometry_profile; /* 0 ordinary, 1 explicit upscale larger canvas */
    size_t video_elements,audio_elements,sequence_rows;
    uint64_t network_reserve_bytes;
} h3_upscale_plan_info;
typedef enum { H3_RESULT_COMPLETE = 0, H3_RESULT_PAUSED = 1 } h3_result_status;

typedef enum {
    H3_CONTINUE_HARD = 0,
    H3_CONTINUE_BRIDGE = 1
} h3_continuation_mode;

typedef enum {
    H3_BRIDGE_STEPPED = 0,
    H3_BRIDGE_LINEAR = 1,
    H3_BRIDGE_EASE_OUT = 2
} h3_bridge_profile_type;

typedef struct {
    uint32_t version;
    int geometry_profile;
    int render_width, render_height, frames;
    int video_t, latent_h, latent_w, audio_t;
    uint64_t seed;
    size_t video_elements, audio_elements;
    uint8_t compatibility[32];
} h3_av_state_info;

typedef struct {
    size_t embedding_entries;
    size_t embedding_bytes;
    int prepared_dit;
    int video_decoder;
} h3_cache_info;

typedef enum {
    H3_REFERENCE_IMAGE = 1,
    H3_REFERENCE_VIDEO = 2,
    H3_REFERENCE_AUDIO = 3,
    H3_REFERENCE_VIDEO_AUDIO = 4
} h3_reference_kind;

typedef struct {
    h3_reference_kind kind;
    const char *path;
    const char *audio_path;
    int include_embedded_audio;
} h3_reference;

typedef enum {
    H3_REFERENCE_IMAGE_MATCH = 0,
    H3_REFERENCE_IMAGE_MAX = 1,
    H3_REFERENCE_IMAGE_HIGH = 2
} h3_reference_image_size;

typedef struct {
    int width;
    int height;
    int stride;
    const uint8_t *rgb;
    int frame_index;
    int frame_count;
    /* Non-negative only for an intermediate denoising preview. */
    int denoise_step;
    int denoise_steps;
} h3_frame;

typedef int (*h3_frame_callback)(const h3_frame *frame, void *opaque);
/* total == 0 denotes indeterminate setup/loading, not a completed phase. */
typedef int (*h3_progress_callback)(const char *phase, int completed, int total,
                                    void *opaque);
/* Model selection, copied by h3_load_dir_with_lora. Folding is lazy per mode.
 * Arguments are ordered PATH[:SCALE] strings; a literal existing file wins.
 * Zero memory_mib selects 512 MiB. NULL cache_dir selects XDG/HOME cache. */
typedef struct {
    const char *const *adapters;
    size_t count;
    const char *cache_dir;
    size_t memory_mib;
} h3_lora_options;
h3_ctx *h3_load_dir_with_lora(const char *model_dir, const h3_lora_options *options);
/* Exact raw sampler state before sampling and after each Euler transition,
 * before display previews. This is the authoritative latent-state interface;
 * preview images can instead show an estimated clean sample. Arrays are
 * borrowed for the duration of the call. Nonzero return cancels sampling. */
typedef int (*h3_latent_callback)(int completed, int total,
    const float *video, size_t video_elements,
    const float *audio, size_t audio_elements, void *opaque);

typedef struct {
    int width;
    int height;
    int frames;
    int steps;
    uint64_t seed;
    const char *output_path;
    const char *first_frame;
    const char *last_frame;
    const h3_reference *references;
    size_t reference_count;
    /* match: down-only render area (default); high/max: long/short edge 2048,
     * upscaling allowed, 32px nearest-even grid and source ratio 1:4..4:1. */
    h3_reference_image_size reference_image_size;
    /* Evaluate one of every N denoiser steps. 1 is the close-reference path,
     * 2 is the validated fast path, and 3 is the aggressive fast path. */
    int denoise_reuse;
    /* Number of gate-ranked DiT residual blocks to retain. 50 is exact,
     * 45 is the validated fast setting, and 40 is more aggressive. */
    int dit_layers;
    /* Recompute the transformer core every N denoiser steps while refreshing
     * the timestep head each step. 1 is exact, 4 fast, and 6 aggressive. */
    int core_reuse;
    /* Pair adjacent horizontal video tokens through middle DiT blocks while
     * preserving their full-resolution residual. Early noisy evaluations use
     * a deeper reduced interval. This is a validated aggressive speed mode. */
    int token_reduction;
    /* Use one int8 activation scale per FC2 row and the M5 full-K kernel.
     * Faster, but more numerically aggressive than grouped int8. */
    int use_int8_row_fc2;
    /* Restore the released spatial RoPE grid at 256x256. The default applies
     * a visually validated half-scale grid only at that native canvas. */
    int use_reference_rope;
    /* Keep only two original BF16 DiT blocks in memory and overlap reading the
     * next block from the checkpoint with execution of the current block. */
    int ssd_streaming;
    /* Optional lower internal model canvas. Both must be zero (exact output
     * canvas) or valid same-aspect dimensions no larger than width/height. */
    int render_width;
    int render_height;
    /* Force the portable close-reference BF16/MPS MLP implementation instead
     * of the fastest validated native MLP supported by the current GPU. */
    int use_slower_bf16_mlp;
    /* Force the portable close-reference BF16 QKV projection. */
    int use_slower_bf16_qkv;
    /* Force the portable BF16 attention-output projection. */
    int use_slower_bf16_attention_output;
    /* Materialize row-major BF16 after SDPA before int8 quantization. */
    int use_slower_row_major_attention_output;
    /* Keep int8 projection-input quantization as standalone kernels. */
    int use_slower_unfused_int8_inputs;
    /* Keep Q/K norm and RoPE as a separate kernel after int8 QKV. */
    int use_slower_unfused_qkv_rope;
    /* Force scalar BF16 loads in the fused Q/K RMS reducer. */
    int use_slower_scalar_qkv_rms;
    /* Reread int8 dequantization scales from device memory per output. */
    int use_slower_uncached_int8_scales;
    /* Use the generic runtime-bound FC1 TensorOps K loop. */
    int use_slower_dynamic_fc1_k;
    /* Force the original 256-thread FC2 grouped activation quantizer. */
    int use_slower_grouped_quantizer;
    /* Decode one representative display frame after every Euler step.
     * The default estimated clean latent (H3_PREVIEW_MODE=denoised) may change
     * substantially early in sampling. H3_PREVIEW_MODE=noisy displays raw
     * sampler state. Display estimates are never resume states. */
    int preview_denoise;
    h3_frame_callback on_frame;
    h3_progress_callback on_progress;
    void *callback_opaque;
    /* Borrowed for this call. Zero context selects 39 frames. The complete
     * target is decoded; by default delivery omits the protected prefix. */
    const h3_av_state *continuation;
    int continuation_context_frames;
    int keep_continuation_prefix;
    h3_latent_callback on_latent_step;
    /* Continuation defaults to CPU Euler; H3_GPU_SAMPLER=1 explicitly selects GPU
     * state updates (H3_CPU_SAMPLER=1 overrides). Use reuse=1/2/3 or core_reuse
     * 1/4/6, never both >1. All layers and token reduction off are required.
     * Hard mode ignores all three bridge tuning fields. */
    h3_continuation_mode continuation_mode;
    int bridge_video_steps;
    float bridge_max_strength;
    h3_bridge_profile_type bridge_profile;
    /* -1 means finish the original schedule; otherwise an absolute completed
     * transition count, including zero. Checkpoints support CPU/GPU state. */
    int stop_after_step;
    const char *save_sampler_state;
    const char *resume_sampler_state;
    int preview_on_stop;
    /* Approximate TAEH3 reconstruction; never changes sampling or encoders.
     * NULL resolves to models/preview-vae/taeh3.safetensors relative to the working directory. */
    int preview_vae;
    const char *preview_vae_model;
    /* 0 off (default), 1 native FP8, 2 native NVFP4; repeated CUDA DiT only.
     * Conservative adaptive cache keeps block 0 projections in BF16 (recipe 3).
     * SubBlock without adaptive cache quantizes all 50 blocks (recipe 4).
     * Cache NULL uses ~/.cache/h3/denoise-quant. On resume, mode is restored;
     * cuda_denoise_quant_set distinguishes an explicit off override. */
    int cuda_denoise_quant;
    const char *cuda_denoise_quant_cache;
    int cuda_denoise_quant_set;
    /* Main CUDA DiT attention: 0 default, 1 SageAttention 2++, 2 SageAttention 3.
     * Resume restores the saved choice; _set distinguishes explicit default. */
    int cuda_attention;
    int cuda_attention_set;
    /* Native attention is an explicit hybrid on M4. */
    h3_backend backend;
    h3_attention_mode attention_mode;
    /* Explicit-selection bits: 1 backend, 2 attention, 4 reserved, 8 options. */
    int backend_set;
    const char *save_conditioning;
    const char *load_conditioning;
    /* Opt-in schedule-specific AdaLN records in addition to invariants. */
    int conditioning_schedule;
    h3_metal_attention_options metal_attention;
    /* Direct T=1 image target; auxiliary audio T=2 is sampled
     * and discarded. No video rounding or audio decode. Explicit image VAE. */
    int still;
    const char *image_vae;
    const char *save_still_latent;
    h3_cuda_sol_options cuda_sol;
    unsigned cuda_sol_set; /* Explicit shared SOL option bits; resume checks only these. */
    /* Internal resolved arithmetic; h3_generate derives this from the device.
     * Callers must leave zero. This is not an execution-policy selector. */
    int _arithmetic_recipe;
    /* Native cache, opt-in CUDA SM120; zero preserves ordinary execution. */
    int adaptive_cache, adaptive_cache_set;
    float subblock_sparsity; /* zero retains every block; CLI default is 0.75 */
    int subblock_sparsity_set;
    /* Zero preserves fixed defaults (4 adaptive / 10 SubBlock), including
     * short schedules. Explicit values: 2..16, at most steps-2. Resume restores
     * saved values and rejects conflicting explicit selections. */
    int adaptive_cache_warmup, subblock_warmup;
    int adaptive_cache_warmup_set, subblock_warmup_set;
    /* Clean latent source export; opt-in and zero-compatible. state_only
     * requires this export and skips all preview/VAEs/media delivery. */
    const char *save_upscale_state;
    int state_only;
    /* Explicit large-canvas profile for upscale stages/bounded validation.
     * Ordinary CLI generation never selects this field. */
    int geometry_profile;
    /* Resource ceiling only. Zero selects 4096 MiB for new requests; resume
     * omission restores the saved ceiling. */
    uint64_t adaptive_cache_max_bytes;
    int adaptive_cache_max_bytes_set;
    /* Omission selects preset defaults; _set preserves explicit threshold zero.
     * Resume restores saved effective values and requires matching overrides. */
    float adaptive_cache_threshold;
    int adaptive_cache_max_hits;
    int adaptive_cache_threshold_set, adaptive_cache_max_hits_set;
    h3_output_encoding output_encoding;
} h3_params;
int h3_cuda_sol_params_valid(const h3_params *p,char *error,size_t size);

#define H3_PARAMS_DEFAULT { \
    H3_DEFAULT_WIDTH, H3_DEFAULT_HEIGHT, H3_DEFAULT_FRAMES, H3_DEFAULT_STEPS, \
    UINT64_C(42), NULL, NULL, NULL, NULL, 0, H3_REFERENCE_IMAGE_MATCH, \
    1, H3_DEFAULT_DIT_LAYERS, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, NULL, NULL, NULL, NULL, 39, 0, NULL, \
    H3_CONTINUE_HARD, 8, 0.50f, H3_BRIDGE_STEPPED, -1, NULL, NULL, 0, 0, NULL, 0, NULL, 0, 0, 0, \
    H3_BACKEND_MPSGRAPH_REFERENCE, H3_ATTN_DENSE, 0, NULL, NULL, 0, H3_METAL_ATTENTION_DEFAULT, 0, NULL, NULL, H3_CUDA_SOL_DEFAULT, 0, 0, 0, 0, 0.75f, 0, 0, 0, 0, 0, NULL, 0, 0, 0, 0, 0, 0, 0, 0, {H3_OUTPUT_PRODUCTION,0,0,0} \
}

typedef struct {
    char name[128];
    char architecture[128];
    uint64_t physical_memory;
    uint64_t recommended_working_set;
    uint64_t max_buffer_length;
    int apple_gpu_family;
    int metal4;
    int unified_memory;
    char backend[16];
    int device_index;
    int cuda_compute_major, cuda_compute_minor;
    int cuda_driver_version, cuda_runtime_version;
    uint64_t device_memory, free_device_memory;
    int multiprocessor_count;
    uint8_t cuda_uuid[16];
} h3_device_info;

typedef struct {
    uint64_t bytes;
    uint64_t tensor_bytes;
    size_t files;
    size_t tensors;
} h3_component_info;

typedef struct {
    h3_component_info text_encoder;
    h3_component_info fl2va_transformer;
    h3_component_info ref2va_transformer;
    h3_component_info video_vae;
    h3_component_info audio_vae;
} h3_model_info;

/* Presentation is separate from the complete, canonical AV latent state. */
typedef struct {
    int version, ref2va;
    int render_width, render_height, width, height;
    int trim_frames, trim_samples, fps, sample_rate, codec_version;
    char tiny_sha256[65]; /* empty for original decoding */
    int cuda_denoise_quant;
    unsigned quant_version;
    char quant_model_sha256[65]; /* SHA-256 of content or metadata identity, as tagged below */
    int quant_model_metadata; /* 1: local file-metadata identity; 0: explicit strict content verification */
    int cuda_attention;
    unsigned attention_version, attention_plan; /* Current attention provenance. */
    h3_cuda_sol_options cuda_sol; /* SOL only. */
    int adaptive_cache;
    unsigned adaptive_version;
    float subblock_sparsity; /* SubBlock only. */
    int adaptive_cache_warmup, subblock_warmup; /* Effective counts, zero if disabled. */
    float adaptive_cache_threshold;
    int adaptive_cache_max_hits;
    /* Explicit geometry/identity plus optional completed upscale recipe. */
    int geometry_profile, av_metadata_identity, upscale_recipe, upscale_steps;
    float upscale_sigma;
    char upscale_parent_sha256[65], upscale_artifact_sha256[65];
} h3_presentation;

typedef struct {
    const char *output_path;
    int preview_vae;
    const char *preview_vae_model;
    h3_frame_callback on_frame;
    h3_progress_callback on_progress;
    void *callback_opaque;
    h3_output_encoding output_encoding;
} h3_decode_options;

typedef struct {
    const char *model_path;
    int refine_steps; /* 0,2,3,4; K=0 performs no noise draws or DiT calls. */
    float sigma;int sigma_set;
    uint64_t seed;int seed_set;
    int stop_after_step; /* -1 completes; otherwise absolute within K. */
    const char *save_sampler_state;
    int state_only;
    h3_decode_options delivery;
    h3_latent_callback on_latent_step;
} h3_upscale_options;
#define H3_UPSCALE_OPTIONS_DEFAULT {NULL,4,.25f,0,0,0,-1,NULL,0,{0},NULL}
/* Source remains caller-owned; result owns clean AV or the paused checkpoint. */
h3_result *h3_upscale(h3_ctx *ctx,const h3_upscale_source *source,const h3_upscale_options *options);

typedef enum { H3_RESULT_VIDEO = 0, H3_RESULT_STILL = 1 } h3_result_kind;

struct h3_result {
    h3_result_kind kind;
    h3_still_latent *still_latent; /* owned; NULL on a video result */
    int width;
    int height;
    int frames;
    int fps;
    int sample_rate;
    uint64_t seed;
    /* Complete untrimmed final target, owned by this result. */
    h3_av_state *av_state;
    int audio_samples;
    h3_result_status status;
    int completed_steps, total_steps;
    h3_sampler_state *sampler_state; /* owned paused state; no clean .h3av */
    /* Optional diagnostic provenance; clean .h3av bytes remain canonical. */
    uint32_t resume_count, resume_step, resume_format;
    uint8_t resume_hash[32];
    h3_presentation presentation;
    /* Owned informational runtime selection; saved as .lora.json, never used
     * to reject AV continuation. Sampler compatibility uses the model digest. */
    char *lora_provenance;
};

/* Load model metadata and initialize the Metal device. Weights remain unmapped. */
h3_ctx *h3_load_dir(const char *model_dir);
void h3_free(h3_ctx *ctx);

const char *h3_last_error(const h3_ctx *ctx);
const h3_device_info *h3_device(const h3_ctx *ctx);
const h3_model_info *h3_model(const h3_ctx *ctx);

/* Repeated library-request reuse. Disabled by default so one-shot callers retain
 * the original phase-by-phase memory lifetime. */
void h3_cache_set_enabled(h3_ctx *ctx, int enabled);
void h3_cache_clear(h3_ctx *ctx);
void h3_cache_get_info(const h3_ctx *ctx, h3_cache_info *info);

/* Generate media, delivering decoded frames incrementally through on_frame. */
h3_result *h3_generate(h3_ctx *ctx, const char *prompt,
                       const h3_params *params);
void h3_result_free(h3_result *result);
/* Decoder-only entry point: no prompt, tokenizer, encoders or denoiser. */
/* Direct T=1 image decode; no model directory, text encoder, DiT or audio.
 * options only uses output_path/on_frame/on_progress/callback_opaque. One final
 * borrowed RGB24 callback; nonzero cancels before PNG publication. */
h3_result *h3_decode_still_latent(const char *latent_path, const char *image_vae,
    const h3_decode_options *options, char *error, size_t error_size);

h3_result *h3_decode_av_state(const char *model_dir, const char *state_path,
    const h3_decode_options *options, char *error, size_t error_size);
/* Saves canonical .h3av plus fingerprint-associated .presentation sidecar. */
int h3_result_save_av_state(const h3_result *result, const char *path,
    char *error, size_t error_size);

const h3_av_state *h3_result_av_state(const h3_result *result);
const h3_av_state_info *h3_av_state_get_info(const h3_av_state *state);
const float *h3_av_state_video(const h3_av_state *state);
const float *h3_av_state_audio(const h3_av_state *state);
h3_av_state *h3_av_state_clone(const h3_av_state *state);
void h3_av_state_free(h3_av_state *state);
int h3_av_state_save(const h3_av_state *state, const char *path,
                     char *error, size_t error_size);
h3_av_state *h3_av_state_load(const char *path, char *error, size_t error_size);
void h3_sampler_state_free(h3_sampler_state *state);
/* Owning portable clean source; loading does not open models or original media. */
h3_upscale_source *h3_upscale_source_load(const char *path, char *error, size_t size);
h3_upscale_source *h3_upscale_source_import_sampler(const char *path, char *error, size_t size);
int h3_upscale_source_save(const h3_upscale_source *source, const char *path, char *error, size_t size);
void h3_upscale_source_free(h3_upscale_source *source);
h3_upscale_plan *h3_upscale_plan_create(const h3_upscale_source *source,char *error,size_t size);
const h3_upscale_plan_info *h3_upscale_plan_get_info(const h3_upscale_plan *plan);
void h3_upscale_plan_free(h3_upscale_plan *plan);
int h3_sampler_state_save(const h3_sampler_state *state, const char *path,
    char *error, size_t error_size);
h3_sampler_state *h3_sampler_state_load_with_budget(const char *path,
    uint64_t bytes, int explicit_budget, char *error, size_t error_size);
h3_sampler_state *h3_sampler_state_load(const char *path,
    char *error, size_t error_size);

#ifdef __cplusplus
}
#endif
#endif
