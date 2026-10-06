#include "src/log.h"
#include "src/sglang/sglang.h"
#include "src/denoise/attention.h"
#include "src/denoise/adaptive_cache.h"
#include "src/denoise/subblock.h"
#include "src/execution.h"
#include "src/memory.h"
#include "src/denoise/dit.h"
#include "src/profile.h"
#ifdef __APPLE__
#include "src/metal/metal_fp16.h"
#include "src/metal/ane_split.h"
#endif
#include "src/sampling/sampler_state.h"
#include "src/testing/test_teacher.h"

#include "src/denoise/dit_schedule.h"
#include "src/weights/weights.h"
#include "src/weights/quant.h"
#include "src/weights/q8.h"

#include <math.h>
#include <pthread.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

enum {
    TEXT_DIM = 5120,
    HIDDEN = 5376,
    HEADS = 56,
    HEAD_DIM = 128,
    INNER = HEADS * HEAD_DIM,
    FFN = 14336,
    VIDEO_CHANNELS = 24,
    VIDEO_PATCH = 96,
    AUDIO_CHANNELS = 32,
    AUDIO_STREAMS = 2,
    ROPE_FREQS = 16,
    ROPE_HALF = 48,
    SLOTS = 6,
    FINAL_SLOTS = 2
};

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *qkv_int8;
    h3_gpu_tensor *qkv_scales;
    h3_gpu_tensor *q_norm;
    h3_gpu_tensor *k_norm;
    h3_gpu_tensor *out;
    h3_gpu_tensor *out_int8;
    h3_gpu_tensor *out_scales;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *fc2;
    h3_gpu_tensor *fc1_int8;
    h3_gpu_tensor *fc1_scales;
    h3_gpu_tensor *fc2_int8;
    h3_gpu_tensor *fc2_scales;
} h3_dit_block;

enum {
    STREAM_QKV,
    STREAM_OUT,
    STREAM_FC1,
    STREAM_FC2,
    STREAM_MATRICES
};

typedef struct {
    const char *path;
    uint64_t file_offset;
    size_t elements;
    unsigned field;
} h3_dit_stream_source;

typedef struct {
    h3_dit_stream_source sources[STREAM_MATRICES];
} h3_dit_stream_layer;

struct h3_dit {
#ifdef __APPLE__
    h3_ane_split *ane;
#endif
    h3_latent_callback on_latent_step;
    void *latent_opaque;
    h3_gpu *gpu;
    h3_weight_store *weights;
    h3_dit_schedule *schedule;
    int fused_mlp;
    int nax_mlp;
    int int8_mlp;
    int int8_qkv;
    int int8_attention_out;
    int keep_bf16_qkv;
    int keep_bf16_attention_out;
    int use_slower_row_major_attention_output;
    int use_slower_unfused_int8_inputs;
    int use_slower_unfused_qkv_rope;
    int use_slower_scalar_qkv_rms;
    int use_slower_uncached_int8_scales;
    int use_slower_dynamic_fc1_k;
    int use_slower_grouped_quantizer;
    int use_int8_row_fc2;
    int ssd_streaming;
    h3_weight_plan weight_plan;
    int explicit_ssd_streaming;
    int sglang_reference;
    int cuda_quant;
    int metal_q8;
    int keep_bf16_mlp;
    int activation_aliases;
    int fused_patch_projection;
    int fused_patch_pack;
    int token_reduction;
    int token_reduction_active;
    unsigned token_reduction_begin;
    unsigned token_reduction_end;
    unsigned token_reduction_early_steps;
    unsigned token_reduction_early_end;
    float token_reduction_scale;
    float spatial_rope_scale;
    int bf16_final;
    unsigned core_reuse_interval;
    unsigned core_forward_count;
    int adaptive_mode, adaptive_pending, adaptive_refresh, captured_attention;
    int adaptive_warmup, subblock_warmup, adaptive_max_hits;
    float adaptive_threshold, adaptive_score;
    unsigned adaptive_recipe;
    uint64_t adaptive_budget;
    unsigned last_executed_blocks;
    unsigned adaptive_phase;
    h3_adaptive_history adaptive_history;
    h3_adaptive_plan adaptive_plan;
    h3_gpu_tensor *adaptive_anchor, *adaptive_delta, *adaptive_scratch, *adaptive_reduction;
    float adaptive_scores[H3_ADAPTIVE_REGIONS];
    h3_adaptive_regions adaptive_regions;
    const char *adaptive_reason;
    int core_residual_ready;
    int resume_sampler_mode;
    unsigned active_block_count;
    uint8_t block_active[H3_DIT_BLOCKS];
    h3_layout layout;
    h3_bridge_profile bridge;
    h3_sigma_schedule sigmas;
    int latent_t;
    int latent_h;
    int latent_w;
    int audio_t;
    uint32_t text_rows;
    uint32_t video_condition_rows;
    uint32_t audio_condition_rows;
    uint32_t audio_rows;
    uint32_t video_rows;
    uint32_t video_total_rows;
    uint32_t audio_total_rows;
    uint32_t audio_target_start;
    uint32_t video_target_start;
    uint32_t sequence;
    uint32_t reduced_sequence;
    uint32_t reduced_video_rows;
    uint32_t token_baseline_rows;
    h3_gpu_tensor *refined_text;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *reduced_rope_cos;
    h3_gpu_tensor *reduced_rope_sin;
    h3_gpu_tensor **row_maps;
    h3_gpu_tensor **reduced_row_maps;
    h3_gpu_tensor **final_audio_maps;
    h3_gpu_tensor **final_video_maps;
    h3_gpu_tensor *video_patch_w;
    h3_gpu_tensor *video_patch_b;
    h3_gpu_tensor *audio_patch_w;
    h3_gpu_tensor *audio_patch_b;
    h3_dit_block blocks[H3_DIT_BLOCKS];
    h3_dit_block stream_slots[2];
    h3_dit_stream_layer stream_layers[H3_DIT_BLOCKS];
    unsigned stream_ready_layer;
    unsigned stream_ready_slot;
    uint64_t stream_bytes, stream_read_layers;
    double stream_read_seconds;
    double stream_wait_seconds;
    h3_gpu_tensor *final_norm;
    h3_gpu_tensor *final_video_w;
    h3_gpu_tensor *final_video_b;
    h3_gpu_tensor *final_audio_w;
    h3_gpu_tensor *final_audio_b;
    h3_gpu_tensor *video_input;
    h3_gpu_tensor *audio_input;
    h3_gpu_tensor *video_projected_f32;
    h3_gpu_tensor *audio_projected_f32;
    h3_gpu_tensor *video_projected;
    h3_gpu_tensor *audio_projected;
    h3_gpu_tensor *video_projection_map;
    h3_gpu_tensor *audio_projection_map;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *core_input;
    h3_gpu_tensor *core_residual;
    h3_gpu_tensor *mod_attention;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *attention_heads;
    h3_gpu_tensor *attention_output;
    h3_gpu_tensor *token_pool_pairs;
    h3_gpu_tensor *token_baseline_indices;
    h3_gpu_tensor *token_expand_parents;
    h3_gpu_tensor *token_original;
    int token_original_in_qkv;
    size_t token_original_offset;
    size_t token_baseline_offset;
    h3_gpu_tensor *mod_mlp;
    h3_gpu_tensor *fc1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *mlp_output;
    h3_gpu_tensor *int8_activation;
    h3_gpu_tensor *int8_activation_scales;
    h3_gpu_tensor *final_audio_input;
    h3_gpu_tensor *final_video_input;
    h3_gpu_tensor *final_audio_inverse;
    h3_gpu_tensor *final_video_inverse;
    h3_gpu_tensor *final_audio_norm;
    h3_gpu_tensor *final_video_norm;
    h3_gpu_tensor *final_audio_f32;
    h3_gpu_tensor *final_video_f32;
    h3_gpu_tensor *audio_output;
    h3_gpu_tensor *video_output;
    h3_gpu_tensor *audio_output_bf16;
    h3_gpu_tensor *video_output_bf16;
    h3_gpu_tensor *previous_audio_velocity;
    h3_gpu_tensor *previous_video_velocity;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

void h3_dit_set_latent_callback(h3_dit *dit, h3_latent_callback callback, void *opaque) {
    if (dit) { dit->on_latent_step=callback; dit->latent_opaque=opaque; }
}

static unsigned command_block_interval(const h3_dit *dit) {
    const char *value = getenv("H3_DIT_COMMAND_BLOCKS");
    if (value && *value) {
        char *end = NULL;
        long parsed = strtol(value, &end, 10);
        return end != value && !*end && parsed >= 0 &&
               parsed <= H3_DIT_BLOCKS ? (unsigned)parsed : 0;
    }
#ifdef __APPLE__
    /* Long-sequence MPSGraph workspaces dominate the command lifetime. Larger
     * batches were appropriate for short native matrix paths, not this case. */
    if(dit->sequence>=65536)return 1;
    if(dit->sequence>=8192)return 5;
#endif
    if (h3_gpu_is_m5(dit->gpu))
        return dit->active_block_count * 3 / 5;
    return dit->active_block_count == H3_DIT_BLOCKS ? 30u : 0u;
}

static int gpu_op(h3_dit *dit, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(dit->gpu));
    return 0;
}

static int report(h3_dit_progress progress, void *opaque, const char *phase,
                   int completed, int total, char *error, size_t error_size) {
    return h3_memory_checkpoint(progress && progress(phase, completed, total, opaque),
                                phase, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *bf1(h3_dit *dit, const char *name, uint64_t width,
                          char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 1, shape,
                               error, error_size);
}

static h3_gpu_tensor *bf2(h3_dit *dit, const char *name, uint64_t rows,
                          uint64_t columns, char *error, size_t error_size) {
#ifdef __APPLE__
    if(dit->metal_q8&&h3_quant_projection(name,1)) {
        const h3_st_header *header=NULL;
        const h3_st_tensor *tensor=h3_weight_find(dit->weights,name,&header);
        if(!tensor||tensor->dtype!=H3_DTYPE_BF16||tensor->ndim!=2||
           tensor->shape[0]!=rows||tensor->shape[1]!=columns||rows>UINT32_MAX||columns>UINT32_MAX) {
            fail(error,error_size,"invalid Q8 BF16 source %s",name);return NULL;
        }
        h3_gpu_tensor *result=h3_gpu_q8_load(dit->gpu,header->path,tensor->file_offset,(uint32_t)rows,(uint32_t)columns);
        if(!result)fail(error,error_size,"cannot prepare Q8 %s: %s",name,h3_gpu_error(dit->gpu));
        return result;
    }
#endif
    if (h3_quant_projection_policy(name,dit->cuda_quant,dit->adaptive_mode)) {
        const h3_st_header *header=NULL;
        const h3_st_tensor *tensor=h3_weight_find(dit->weights,name,&header);
        if (!tensor || tensor->dtype!=H3_DTYPE_BF16 || tensor->ndim!=2 ||
            tensor->shape[0]!=rows || tensor->shape[1]!=columns) {
            fail(error,error_size,"invalid quantized DiT source %s",name);return NULL;
        }
        h3_gpu_tensor *result=h3_gpu_quant_load(dit->gpu,header->path,tensor->file_offset,(uint32_t)rows,(uint32_t)columns);
        if(!result)fail(error,error_size,"cannot prepare %s: %s",name,h3_gpu_error(dit->gpu));
        return result;
    }
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(dit->weights, dit->gpu, name, 2, shape,
                               error, error_size);
}

static h3_gpu_tensor *f1(h3_dit *dit, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 1, shape,
                              error, error_size);
}

static h3_gpu_tensor *f2(h3_dit *dit, const char *name, uint64_t rows,
                         uint64_t columns, char *error, size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(dit->weights, dit->gpu, name, 2, shape,
                              error, error_size);
}

static int copy_layout(h3_dit *dit, const h3_layout *layout,
                       char *error, size_t error_size) {
    dit->layout = *layout;
    dit->layout.segments = NULL;
    dit->layout.positions = NULL;
    if (layout->bridge) {
        if (!h3_bridge_profile_valid(layout->bridge) ||
            layout->bridge->prefix.video_prefix_t != layout->prefix.video_prefix_t ||
            layout->bridge->prefix.audio_prefix_t != layout->prefix.audio_prefix_t) {
            fail(error, error_size, "bridge profile does not match the target prefix"); return 0;
        }
        dit->bridge = *layout->bridge;
        dit->layout.bridge = &dit->bridge;
    }
    if (layout->segment_count) {
        dit->layout.segments = malloc(layout->segment_count *
                                      sizeof(*layout->segments));
        if (!dit->layout.segments) goto oom;
        memcpy(dit->layout.segments, layout->segments,
               layout->segment_count * sizeof(*layout->segments));
    }
    if (layout->seq_len) {
        dit->layout.positions = malloc(layout->seq_len *
                                       sizeof(*layout->positions));
        if (!dit->layout.positions) goto oom;
        memcpy(dit->layout.positions, layout->positions,
               layout->seq_len * sizeof(*layout->positions));
    }
    return 1;
oom:
    fail(error, error_size, "out of memory copying packed H3 layout");
    h3_layout_free(&dit->layout);
    return 0;
}

static int validate_layout(h3_dit *dit, const h3_text_embedding *text,
                           char *error, size_t error_size) {
    const h3_layout *layout = &dit->layout;
    if (!text || !text->values || text->width != TEXT_DIM || !text->tokens ||
        layout->signature[0] != (int)text->tokens ||
        !layout->segments || layout->segment_count < 3 ||
        layout->segments[0].kind != H3_SEG_TEXT ||
        layout->segments[layout->segment_count - 1].kind != H3_SEG_VIDEO ||
        layout->signature[1] < 1 || layout->signature[2] < 2 ||
        layout->signature[3] < 2 || layout->signature[4] < 1 ||
        layout->signature[2] % 2 || layout->signature[3] % 2 ||
        layout->seq_len > UINT32_MAX || text->tokens > UINT32_MAX ||
        layout->img_cond_rows > UINT32_MAX ||
        layout->prefix.video_prefix_t < 0 ||
        layout->prefix.video_prefix_t >= layout->signature[1] ||
        layout->prefix.audio_prefix_t < 0 ||
        (layout->frozen_audio?(layout->frozen_audio!=1||layout->prefix.audio_prefix_t!=layout->signature[4]||
            layout->prefix.video_prefix_t||layout->bridge):layout->prefix.audio_prefix_t>=layout->signature[4]) ||
        layout->audio_cond_rows > UINT32_MAX ||
        layout->audio_target_rows > UINT32_MAX ||
        layout->img_target_rows > UINT32_MAX) {
        fail(error, error_size,
             "DiT requires a valid contiguous H3 packed layout");
        return 0;
    }
    size_t cursor = 0, text_rows = 0, video_condition = 0;
    size_t audio_condition = 0, video_target = 0, audio_target = 0;
    unsigned target_video_segments = 0, target_audio_segments = 0;
    for (size_t index = 0; index < layout->segment_count; index++) {
        const h3_segment *segment = &layout->segments[index];
        if (segment->start != cursor || segment->stop < segment->start ||
            segment->stop > layout->seq_len) {
            fail(error, error_size, "DiT layout segments are not contiguous");
            return 0;
        }
        size_t rows = segment->stop - segment->start;
        switch (segment->kind) {
        case H3_SEG_TEXT: text_rows += rows; break;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE: video_condition += rows; break;
        case H3_SEG_REF_AUDIO: audio_condition += rows; break;
        case H3_SEG_AUDIO:
            audio_target += rows;
            target_audio_segments++;
            dit->audio_target_start = (uint32_t)segment->start;
            break;
        case H3_SEG_VIDEO:
            video_target += rows;
            target_video_segments++;
            dit->video_target_start = (uint32_t)segment->start;
            break;
        default:
            fail(error, error_size, "DiT layout contains an unknown segment");
            return 0;
        }
        cursor = segment->stop;
    }
    if (cursor != layout->seq_len || text_rows != text->tokens ||
        video_condition != layout->img_cond_rows ||
        audio_condition != layout->audio_cond_rows ||
        video_target != layout->img_target_rows ||
        audio_target != layout->audio_target_rows ||
        target_video_segments != 1 || target_audio_segments != 1 ||
        video_condition > UINT32_MAX - video_target ||
        audio_condition > UINT32_MAX - audio_target) {
        fail(error, error_size, "DiT layout row-source counts are inconsistent");
        return 0;
    }
    if (text->tags) {
        for (size_t index = 0; index < text->tokens; index++) {
            if (text->tags[index] >= H3_DIT_MODALITIES) {
                fail(error, error_size, "DiT text presentation has an invalid tag");
                return 0;
            }
        }
    }
    dit->latent_t = layout->signature[1];
    dit->latent_h = layout->signature[2];
    dit->latent_w = layout->signature[3];
    dit->audio_t = layout->signature[4];
    dit->text_rows = (uint32_t)text->tokens;
    dit->video_condition_rows = (uint32_t)video_condition;
    dit->audio_condition_rows = (uint32_t)audio_condition;
    dit->audio_rows = (uint32_t)layout->audio_target_rows;
    dit->video_rows = (uint32_t)layout->img_target_rows;
    dit->video_total_rows = (uint32_t)(video_condition + video_target);
    dit->audio_total_rows = (uint32_t)(audio_condition + audio_target);
    dit->sequence = (uint32_t)layout->seq_len;
    return 1;
}

static int configure_token_reduction(h3_dit *dit, int requested,
                                     char *error, size_t error_size) {
    const char *enabled = getenv("H3_TOKEN_REDUCTION");
    if (!requested &&
        (!enabled || !*enabled || !strcmp(enabled, "0"))) return 1;
    unsigned begin = 4, end = 30;
    const char *range = getenv("H3_TOKEN_REDUCTION_BLOCKS");
    if (range && *range) {
        char *middle = NULL;
        unsigned long parsed_begin = strtoul(range, &middle, 10);
        if (middle == range || *middle != ':') {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_BLOCKS must be BEGIN:END");
            return 0;
        }
        char *tail = NULL;
        unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
        if (tail == middle + 1 || *tail || parsed_begin >= parsed_end ||
            parsed_end > H3_DIT_BLOCKS) {
            fail(error, error_size,
                 "token-reduction block range must satisfy 0 <= BEGIN < END <= 50");
            return 0;
        }
        begin = (unsigned)parsed_begin;
        end = (unsigned)parsed_end;
    }
    /* Coarse structure is tolerant of a deeper reduced stack while the first
     * noisy samples form. Restore earlier once fine detail starts resolving. */
    unsigned early_steps = end < 40 ? 10 : 0;
    unsigned early_end = end < 40 ? 40 : end;
    const char *early = getenv("H3_TOKEN_REDUCTION_EARLY");
    if (early && *early) {
        if (!strcmp(early, "0")) {
            early_steps = 0;
            early_end = end;
        } else {
            char *middle = NULL;
            unsigned long parsed_steps = strtoul(early, &middle, 10);
            if (middle == early || *middle != ':') {
                fail(error, error_size,
                     "H3_TOKEN_REDUCTION_EARLY must be STEPS:END");
                return 0;
            }
            char *tail = NULL;
            unsigned long parsed_end = strtoul(middle + 1, &tail, 10);
            if (tail == middle + 1 || *tail || !parsed_steps ||
                parsed_steps > 1000 || parsed_end <= end ||
                parsed_end > H3_DIT_BLOCKS) {
                fail(error, error_size,
                     "early token reduction requires STEPS > 0 and "
                     "base END < END <= 50");
                return 0;
            }
            early_steps = (unsigned)parsed_steps;
            early_end = (unsigned)parsed_end;
        }
    }
    float scale = 1.0f;
    const char *scale_text = getenv("H3_TOKEN_REDUCTION_SCALE");
    if (scale_text && *scale_text) {
        char *tail = NULL;
        scale = strtof(scale_text, &tail);
        if (tail == scale_text || *tail || !isfinite(scale) ||
            scale < 0.0f || scale > 2.0f) {
            fail(error, error_size,
                 "H3_TOKEN_REDUCTION_SCALE must be in [0, 2]");
            return 0;
        }
    }
    uint32_t spatial_height = (uint32_t)dit->latent_h / 2;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint64_t reduced_video =
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * reduced_width;
    if (!spatial_height || !spatial_width ||
        (uint64_t)(uint32_t)dit->latent_t * spatial_height * spatial_width !=
            dit->video_rows ||
        dit->video_target_start + dit->video_rows != dit->sequence ||
        reduced_video > UINT32_MAX ||
        reduced_video > UINT32_MAX - dit->video_target_start) {
        fail(error, error_size,
             "token reduction requires the target video to end the packed layout");
        return 0;
    }
    dit->token_reduction = 1;
    dit->token_reduction_begin = begin;
    dit->token_reduction_end = end;
    dit->token_reduction_early_steps = early_steps;
    dit->token_reduction_early_end = early_end;
    dit->token_reduction_scale = scale;
    dit->reduced_video_rows = (uint32_t)reduced_video;
    dit->token_baseline_rows = dit->video_rows - dit->reduced_video_rows;
    dit->reduced_sequence = dit->video_target_start +
                            dit->reduced_video_rows;
    return 1;
}

static void token_pool_sources(const h3_dit *dit, uint32_t reduced_row,
                               uint32_t *first, uint32_t *second) {
    if (reduced_row < dit->video_target_start) {
        *first = reduced_row;
        *second = reduced_row;
        return;
    }
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = reduced_row - dit->video_target_start;
    uint32_t source = dit->video_target_start +
        (local / reduced_width) * spatial_width +
        (local % reduced_width) * 2;
    *first = source;
    *second = source + ((source - dit->video_target_start) % spatial_width + 1 <
                        spatial_width ? 1u : 0u);
}

static uint32_t token_reduced_parent(const h3_dit *dit, uint32_t full_row) {
    if (full_row < dit->video_target_start) return full_row;
    uint32_t spatial_width = (uint32_t)dit->latent_w / 2;
    uint32_t reduced_width = (spatial_width + 1) / 2;
    uint32_t local = full_row - dit->video_target_start;
    return dit->video_target_start + (local / spatial_width) * reduced_width +
           (local % spatial_width) / 2;
}

static int load_block(h3_dit *dit, h3_dit_block *block, const char *prefix,
                      char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
#define LOAD2(field, suffix, rows, columns) do {                               \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf2(dit, name, rows, columns, error, error_size);            \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD2(qkv, "attn.qkv_proj.weight", INNER * 3, HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
    LOAD2(out, "attn.out_proj.weight", HIDDEN, INNER);
    LOAD2(fc1, "mlp.fc1.weight", FFN * 2, HIDDEN);
    LOAD2(fc2, "mlp.fc2.weight", HIDDEN, FFN);
#undef LOAD1
#undef LOAD2
    return 1;
}

static int load_block_norms(h3_dit *dit, h3_dit_block *block,
                            const char *prefix,
                            char *error, size_t error_size) {
    char name[160];
#define LOAD1(field, suffix, width) do {                                       \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = bf1(dit, name, width, error, error_size);                    \
    if (!block->field) return 0;                                                \
} while (0)
    LOAD1(norm1, "norm1.weight", HIDDEN);
    LOAD1(norm2, "norm2.weight", HIDDEN);
    LOAD1(q_norm, "attn.q_norm.weight", HEAD_DIM);
    LOAD1(k_norm, "attn.k_norm.weight", HEAD_DIM);
#undef LOAD1
    return 1;
}

static void free_block(h3_dit_block *block) {
    free_tensor(&block->norm1);
    free_tensor(&block->norm2);
    free_tensor(&block->qkv);
    free_tensor(&block->qkv_int8);
    free_tensor(&block->qkv_scales);
    free_tensor(&block->q_norm);
    free_tensor(&block->k_norm);
    free_tensor(&block->out);
    free_tensor(&block->out_int8);
    free_tensor(&block->out_scales);
    free_tensor(&block->fc1);
    free_tensor(&block->fc2);
    free_tensor(&block->fc1_int8);
    free_tensor(&block->fc1_scales);
    free_tensor(&block->fc2_int8);
    free_tensor(&block->fc2_scales);
}

static double stream_now(void) {
    struct timespec value;
    if (clock_gettime(CLOCK_MONOTONIC, &value) != 0) return 0.0;
    return (double)value.tv_sec + (double)value.tv_nsec * 1e-9;
}

/* Opt-in cross-backend diagnostics. The normal execution path does no I/O or
 * extra submission; these dumps make accumulated BF16 drift inspectable. */
static int debug_dit_tensor(h3_dit *dit, const char *label,
                            const h3_gpu_tensor *tensor) {
    const char *directory=getenv("H3_DEBUG_DIT_DIR");
    if(!directory || !*directory)return 1;
    size_t n=h3_gpu_tensor_elements(tensor);uint16_t *values=malloc(n*2);
    int ok=values && h3_gpu_submit(dit->gpu) && h3_gpu_tensor_read_bf16(tensor,values,n);
    char path[4096];
    if(ok)ok=snprintf(path,sizeof(path),"%s/%s.bf16",directory,label)<(int)sizeof(path);
    if(ok){FILE *file=fopen(path,"wb");ok=file && fwrite(values,2,n,file)==n;if(file && fclose(file))ok=0;}
    free(values);return h3_gpu_begin(dit->gpu) && ok;
}
static int debug_dit_step(int step) {
    const char *p=getenv("H3_DEBUG_DIT_STEPS");if(!p||!*p)return 1;
    while(*p){char *end=NULL;long n=strtol(p,&end,10);if(end==p)return 0;if(n==step)return 1;if(*end!=',')return 0;p=end+1;}
    return 0;
}
static int debug_native_step(int step,const char *name,const float *data,size_t count) {
    const char *directory=getenv("H3_TEST_NATIVE_STEP_DIR");
    if(!directory||!*directory)return 1;
    char path[4096];
    if(snprintf(path,sizeof(path),"%s/step-%03d-%s.f32",directory,step+1,name)>=(int)sizeof(path))return 0;
    FILE *file=fopen(path,"wbx");if(!file)return 0;
    int ok=fwrite(data,sizeof(float),count,file)==count;if(fclose(file))ok=0;return ok;
}

static int sglang_boundary(h3_dit *dit,const char *name,const h3_gpu_tensor *tensor,size_t count,int step,unsigned block) {
    const char *directory=getenv("H3_TEST_SGLANG_DIR");
    if(!directory || !*directory || block!=0)return 1;
    if(getenv("H3_TEST_SGLANG_BLOCK_ONLY") && (!strncmp(name,"refiner-",8)||!strcmp(name,"condition-proj")))return 1;
    int selected=h3_sglang_capture_step(step,dit->sigmas.steps);
    if (selected < 0)
        return 0;
    if (!selected)
        return 1;
    if(count>h3_gpu_tensor_elements(tensor))return 0;
    uint16_t *data=malloc(count*sizeof(*data));
    int ok=data && h3_gpu_submit(dit->gpu) && h3_gpu_tensor_read_bf16(tensor,data,count);
    char file[128];snprintf(file,sizeof(file),"step-%03d.block-0.%s.bf16",step,name);
    if(ok)ok=h3_sglang_dump(file,data,count*sizeof(*data));
    free(data);return h3_gpu_begin(dit->gpu) && ok;
}

static int final_f32_head(h3_dit *dit,h3_gpu_tensor *out,const h3_gpu_tensor *input,
    const h3_gpu_tensor *weight,const h3_gpu_tensor *bias,uint32_t rows,uint32_t columns) {
#ifndef __APPLE__
    /* The oracle chooses its GEMM for all padded rows before selecting the
     * media. Retain that reduction algorithm on our compact media buffers. */
    if(dit->sglang_reference)return h3_gpu_sglang_head_f32(dit->gpu,out,input,weight,bias,
        rows,HIDDEN,columns,(dit->sequence+63u)/64u*64u);
#endif
    return h3_gpu_linear_f32(dit->gpu,out,input,weight,bias,rows,HIDDEN,columns);
}

static int compare_stream_sources(const void *left, const void *right) {
    const h3_dit_stream_source *a = left;
    const h3_dit_stream_source *b = right;
    int path = strcmp(a->path, b->path);
    if (path) return path;
    if (a->file_offset < b->file_offset) return -1;
    return a->file_offset > b->file_offset;
}

static int prepare_stream_source(h3_dit *dit,
                                 h3_dit_stream_source *source,
                                 const char *name, uint64_t rows,
                                 uint64_t columns, unsigned field,
                                 char *error, size_t error_size) {
    const h3_st_header *header = NULL;
    const h3_st_tensor *tensor = h3_weight_find(dit->weights, name, &header);
    if (!tensor) {
        fail(error, error_size, "required streaming weight is absent: %s",
             name);
        return 0;
    }
    if (!header || tensor->dtype != H3_DTYPE_BF16 || tensor->ndim != 2 ||
        tensor->shape[0] != rows || tensor->shape[1] != columns ||
        rows > SIZE_MAX / columns) {
        fail(error, error_size, "streaming weight has the wrong schema: %s",
             name);
        return 0;
    }
    source->path = header->path;
    source->file_offset = tensor->file_offset;
    source->elements = (size_t)(rows * columns);
    source->field = field;
    return 1;
}

static int prepare_stream_layer(h3_dit *dit, unsigned layer,
                                char *error, size_t error_size) {
    char name[160];
    h3_dit_stream_layer *stream = &dit->stream_layers[layer];
#define SOURCE(index, suffix, rows, columns, field) do {                        \
    snprintf(name, sizeof(name), "blocks.%u.%s", layer, suffix);              \
    if (!prepare_stream_source(dit, &stream->sources[index], name,             \
                               rows, columns, field, error, error_size))        \
        return 0;                                                               \
} while (0)
    SOURCE(0, "attn.qkv_proj.weight", INNER * 3, HIDDEN, STREAM_QKV);
    SOURCE(1, "attn.out_proj.weight", HIDDEN, INNER, STREAM_OUT);
    SOURCE(2, "mlp.fc1.weight", FFN * 2, HIDDEN, STREAM_FC1);
    SOURCE(3, "mlp.fc2.weight", HIDDEN, FFN, STREAM_FC2);
#undef SOURCE
    qsort(stream->sources, STREAM_MATRICES, sizeof(stream->sources[0]),
          compare_stream_sources);
#ifndef __APPLE__
    if(dit->sglang_reference)for(unsigned i=0;i<STREAM_MATRICES;i++) {
        h3_dit_stream_source *source=&stream->sources[i];
        if(!gpu_op(dit,h3_gpu_sglang_preload_weight(dit->gpu,source->path,
                source->file_offset,source->elements),error,error_size,
                "preload bounded reference host weight"))return 0;
    }
#endif
    return 1;
}

static int allocate_stream_slot(h3_dit *dit, h3_dit_block *slot,
                                char *error, size_t error_size) {
    slot->qkv = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    slot->out = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)HIDDEN * INNER);
    slot->fc1 = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)FFN * 2 * HIDDEN);
    slot->fc2 = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)HIDDEN * FFN);
    if (!slot->qkv || !slot->out || !slot->fc1 || !slot->fc2) {
        fail(error, error_size, "cannot allocate BF16 SSD layer slot: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static h3_gpu_tensor *stream_slot_target(h3_dit_block *slot,
                                         unsigned field) {
    if (field == STREAM_QKV) return slot->qkv;
    if (field == STREAM_OUT) return slot->out;
    if (field == STREAM_FC1) return slot->fc1;
    if (field == STREAM_FC2) return slot->fc2;
    return NULL;
}

typedef struct {
    h3_dit *dit;
    unsigned layer;
    unsigned slot;
    int ok;
    uint64_t bytes;
    double seconds;
    char error[512];
} h3_dit_stream_job;

static int read_stream_layer(h3_dit_stream_job *job) {
    h3_dit_stream_layer *layer = &job->dit->stream_layers[job->layer];
    h3_dit_block *slot = &job->dit->stream_slots[job->slot];
    double started = stream_now();
    job->ok = 1;
    job->bytes = 0;
    job->error[0] = '\0';
    for (unsigned index = 0; index < STREAM_MATRICES; index++) {
        const h3_dit_stream_source *source = &layer->sources[index];
        h3_gpu_tensor *target = stream_slot_target(slot, source->field);
        if (!target || !h3_gpu_tensor_stream_file_bf16(
                target, source->path, source->file_offset, source->elements,
                job->error, sizeof(job->error))) {
            if (!job->error[0])
                snprintf(job->error, sizeof(job->error),
                         "invalid BF16 streaming destination");
            job->ok = 0;
            break;
        }
        job->bytes += (uint64_t)source->elements * sizeof(uint16_t);
    }
    job->seconds = stream_now() - started;
    return job->ok;
}

static void *read_stream_layer_thread(void *opaque) {
    read_stream_layer(opaque);
    return NULL;
}

static int quantize_block_mlp(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->fc1_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)FFN * 2 * HIDDEN);
    block->fc1_scales = h3_gpu_tensor_new_f32(dit->gpu, FFN * 2);
    block->fc2_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * FFN);
    block->fc2_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->fc1_int8 && block->fc1_scales &&
             block->fc2_int8 && block->fc2_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc1_int8, block->fc1_scales, block->fc1,
                 FFN * 2, HIDDEN) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->fc2_int8, block->fc2_scales, block->fc2,
                 HIDDEN, FFN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size, "cannot quantize DiT MLP weights: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_mlp) {
        free_tensor(&block->fc1);
        free_tensor(&block->fc2);
    }
    return 1;
}

static int quantize_block_qkv(h3_dit *dit, h3_dit_block *block,
                              char *error, size_t error_size) {
    block->qkv_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)INNER * 3 * HIDDEN);
    block->qkv_scales = h3_gpu_tensor_new_f32(dit->gpu, INNER * 3);
    int ok = block->qkv_int8 && block->qkv_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->qkv_int8, block->qkv_scales, block->qkv,
                 INNER * 3, HIDDEN) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size, "cannot quantize DiT QKV weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_qkv) free_tensor(&block->qkv);
    return 1;
}

static int quantize_block_attention_out(h3_dit *dit, h3_dit_block *block,
                                        char *error, size_t error_size) {
    block->out_int8 = h3_gpu_tensor_new_i8(
        dit->gpu, (size_t)HIDDEN * INNER);
    block->out_scales = h3_gpu_tensor_new_f32(dit->gpu, HIDDEN);
    int ok = block->out_int8 && block->out_scales &&
             h3_gpu_begin(dit->gpu) &&
             h3_gpu_quantize_weight_int8(
                 dit->gpu, block->out_int8, block->out_scales, block->out,
                 HIDDEN, INNER) &&
             h3_gpu_submit(dit->gpu);
    if (!ok) {
        fail(error, error_size,
             "cannot quantize DiT attention-output weight: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (!dit->keep_bf16_attention_out) free_tensor(&block->out);
    return 1;
}

static int run_refiner_block(h3_dit *dit, const h3_dit_block *weight,
                             h3_gpu_tensor *hidden, h3_gpu_tensor *norm,
                             h3_gpu_tensor *qkv, h3_gpu_tensor *query,
                             h3_gpu_tensor *key, h3_gpu_tensor *value,
                             h3_gpu_tensor *heads, h3_gpu_tensor *branch,
                             h3_gpu_tensor *fc1, h3_gpu_tensor *activated,
                             unsigned index,char *error, size_t error_size) {
    uint32_t rows = dit->text_rows;
#define OP(call, label) do {                                                    \
    if (!gpu_op(dit, (call), error, error_size, label)) return 0;               \
} while (0)
#define CAP(t,n,c) do { char name[96];snprintf(name,sizeof(name),"refiner-%u.%s",index,n); \
    OP(sglang_boundary(dit,name,t,(size_t)rows*(c),0,0),"refiner capture"); } while(0)
    CAP(hidden,"input",HIDDEN);
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm1, rows,
                             HIDDEN, 1e-5f), "refiner attention norm");
    CAP(norm,"norm1",HIDDEN);
    OP(h3_gpu_linear_bf16(dit->gpu, qkv, norm, weight->qkv, NULL, rows,
                           HIDDEN, INNER * 3), "refiner QKV");
    CAP(qkv,"qkv",INNER*3);
    OP(h3_gpu_grouped_qkv_rope_bf16(
                             dit->gpu, query, key, value, qkv, weight->q_norm,
                             weight->k_norm, weight->q_norm, weight->q_norm,
                             rows, HEADS, HEAD_DIM, 0, 1e-5f),
       "refiner QK norm");
    CAP(query,"query",INNER);CAP(key,"key",INNER);CAP(value,"value",INNER);
    OP(h3_gpu_sdpa_bf16(dit->gpu, heads, query, key, value, rows, HEADS,
                         HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "refiner attention");
    CAP(heads,"attention",INNER);
    OP(h3_gpu_linear_bf16(dit->gpu, branch, heads, weight->out, NULL, rows,
                           INNER, HIDDEN), "refiner attention output");
    CAP(branch,"projection",HIDDEN);
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner attention residual");
    CAP(hidden,"residual",HIDDEN);
    OP(h3_gpu_rms_norm_bf16(dit->gpu, norm, hidden, weight->norm2, rows,
                             HIDDEN, 1e-5f), "refiner MLP norm");
    CAP(norm,"norm2",HIDDEN);
    OP(h3_gpu_linear_bf16(dit->gpu, fc1, norm, weight->fc1, NULL, rows,
                           HIDDEN, FFN * 2), "refiner MLP input");
    CAP(fc1,"fc1",FFN*2);
    OP(h3_gpu_swiglu_bf16(dit->gpu, activated, fc1, rows, FFN),
       "refiner SwiGLU");
    OP(h3_gpu_linear_bf16(dit->gpu, branch, activated, weight->fc2, NULL,
                           rows, FFN, HIDDEN), "refiner MLP output");
    OP(h3_gpu_add_bf16(dit->gpu, hidden, hidden, branch, rows * HIDDEN),
       "refiner MLP residual");
    CAP(hidden,"output",HIDDEN);
#undef CAP
#undef OP
    return 1;
}

static int refine_text(h3_dit *dit, const h3_text_embedding *text,
                       char *error, size_t error_size) {
    h3_gpu_tensor *source = h3_gpu_tensor_from_bf16(
        dit->gpu, text->values, text->tokens * TEXT_DIM);
    h3_gpu_tensor *condition_w = bf2(dit, "condition_proj.weight", HIDDEN,
                                     TEXT_DIM, error, error_size);
    h3_gpu_tensor *condition_b = bf1(dit, "condition_proj.bias", HIDDEN,
                                     error, error_size);
    h3_dit_block refiner[2];
    memset(refiner, 0, sizeof(refiner));
    h3_gpu_tensor *final_norm = NULL;
    h3_gpu_tensor *norm = NULL, *qkv = NULL, *query = NULL, *key = NULL;
    h3_gpu_tensor *value = NULL, *heads = NULL, *branch = NULL, *fc1 = NULL;
    h3_gpu_tensor *activated = NULL;
    int ok = source && condition_w && condition_b &&
        load_block(dit, &refiner[0], "token_refiner.blocks.0.",
                   error, error_size) &&
        load_block(dit, &refiner[1], "token_refiner.blocks.1.",
                   error, error_size);
    if (ok) final_norm = bf1(dit, "token_refiner.final_norm.weight", HIDDEN,
                             error, error_size);
    size_t rows = dit->text_rows;
    if (ok && final_norm) {
        dit->refined_text = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        norm = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        qkv = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER * 3);
        query = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        key = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        value = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        heads = h3_gpu_tensor_new_bf16(dit->gpu, rows * INNER);
        branch = h3_gpu_tensor_new_bf16(dit->gpu, rows * HIDDEN);
        fc1 = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN * 2);
        activated = h3_gpu_tensor_new_bf16(dit->gpu, rows * FFN);
        ok = dit->refined_text && norm && qkv && query && key && value &&
             heads && branch && fc1 && activated;
    }
    if (!ok) {
        if (!error || !*error)
            fail(error, error_size, "cannot allocate token-refiner tensors: %s",
                 h3_gpu_error(dit->gpu));
        goto cleanup;
    }
    ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                "begin token refinement") &&
         gpu_op(dit, h3_gpu_linear_bf16(
             dit->gpu, dit->refined_text, source, condition_w, condition_b,
             dit->text_rows, TEXT_DIM, HIDDEN), error, error_size,
             "condition projection") &&
         sglang_boundary(dit,"condition-proj",dit->refined_text,(size_t)dit->text_rows*HIDDEN,0,0) &&
         run_refiner_block(dit, &refiner[0], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             0,error, error_size) &&
         run_refiner_block(dit, &refiner[1], dit->refined_text, norm, qkv,
             query, key, value, heads, branch, fc1, activated,
             1,error, error_size) &&
         gpu_op(dit, h3_gpu_rms_norm_bf16(
             dit->gpu, dit->refined_text, dit->refined_text, final_norm,
             dit->text_rows, HIDDEN, 1e-5f), error, error_size,
             "refiner final norm") &&
         sglang_boundary(dit,"refined-text",dit->refined_text,(size_t)dit->text_rows*HIDDEN,0,0) &&
         gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                "submit token refinement");
cleanup:
    free_tensor(&source);
    free_tensor(&condition_w);
    free_tensor(&condition_b);
    free_block(&refiner[0]);
    free_block(&refiner[1]);
    free_tensor(&final_norm);
    free_tensor(&norm);
    free_tensor(&qkv);
    free_tensor(&query);
    free_tensor(&key);
    free_tensor(&value);
    free_tensor(&heads);
    free_tensor(&branch);
    free_tensor(&fc1);
    free_tensor(&activated);
    return ok;
}

static int prepare_rope(h3_dit *dit, char *error, size_t error_size) {
    h3_gpu_tensor *inverse_tensor = f1(dit, "rope.inv_freq", ROPE_FREQS,
                                       error, error_size);
    float inverse[ROPE_FREQS];
    if (!inverse_tensor ||
        !h3_gpu_tensor_read_f32(inverse_tensor, inverse, ROPE_FREQS)) {
        free_tensor(&inverse_tensor);
        if (!error || !*error) fail(error, error_size, "cannot read RoPE frequencies");
        return 0;
    }
    free_tensor(&inverse_tensor);
    float spatial_scale = dit->spatial_rope_scale;
    size_t count = (size_t)dit->sequence * ROPE_HALF;
    size_t reduced_count = dit->token_reduction ?
        (size_t)dit->reduced_sequence * ROPE_HALF : 0;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    float *reduced_cosines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_cosines)) : NULL;
    float *reduced_sines = reduced_count ?
        malloc(reduced_count * sizeof(*reduced_sines)) : NULL;
    if (!cosines || !sines ||
        (reduced_count && (!reduced_cosines || !reduced_sines))) {
        free(cosines);
        free(sines);
        free(reduced_cosines);
        free(reduced_sines);
        fail(error, error_size, "out of memory allocating DiT RoPE tables");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++) {
        float axes[] = {(float)dit->layout.positions[row].t,
                        (float)dit->layout.positions[row].h * spatial_scale,
                        (float)dit->layout.positions[row].w * spatial_scale};
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                /* CUDA and host libm differ at rare BF16 ties. The pinned
                 * oracle computes trigonometry on CUDA after FP32 angles. */
                cosines[index] = dit->sglang_reference ? angle : cosf(angle);
                sines[index] = sinf(angle);
            }
        }
    }
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        uint32_t first, second;
        token_pool_sources(dit, row, &first, &second);
        float axes[] = {
            (float)((dit->layout.positions[first].t +
                     dit->layout.positions[second].t) * 0.5),
            (float)((dit->layout.positions[first].h +
                     dit->layout.positions[second].h) * 0.5) * spatial_scale,
            (float)((dit->layout.positions[first].w +
                     dit->layout.positions[second].w) * 0.5) * spatial_scale
        };
        for (uint32_t axis = 0; axis < 3; axis++) {
            for (uint32_t frequency = 0; frequency < ROPE_FREQS; frequency++) {
                size_t index = (size_t)row * ROPE_HALF +
                               axis * ROPE_FREQS + frequency;
                float angle = axes[axis] * inverse[frequency];
                reduced_cosines[index] = cosf(angle);
                reduced_sines[index] = sinf(angle);
            }
        }
    }
    h3_gpu_tensor *cos_f32 = h3_gpu_tensor_from_f32(dit->gpu, cosines, count);
    h3_gpu_tensor *sin_f32 = h3_gpu_tensor_from_f32(dit->gpu, sines, count);
    h3_gpu_tensor *reduced_cos_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_cosines, reduced_count) : NULL;
    h3_gpu_tensor *reduced_sin_f32 = reduced_count ?
        h3_gpu_tensor_from_f32(dit->gpu, reduced_sines, reduced_count) : NULL;
    free(cosines);
    free(sines);
    free(reduced_cosines);
    free(reduced_sines);
    dit->rope_cos = h3_gpu_tensor_new_bf16(dit->gpu, count);
    dit->rope_sin = h3_gpu_tensor_new_bf16(dit->gpu, count);
    if (reduced_count) {
        dit->reduced_rope_cos = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
        dit->reduced_rope_sin = h3_gpu_tensor_new_bf16(
            dit->gpu, reduced_count);
    }
    int ok = cos_f32 && sin_f32 && dit->rope_cos && dit->rope_sin &&
        (!reduced_count || (reduced_cos_f32 && reduced_sin_f32 &&
                            dit->reduced_rope_cos &&
                            dit->reduced_rope_sin));
    if (ok) {
        ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                    "begin RoPE setup");
#ifndef __APPLE__
        if (ok && dit->sglang_reference)
            ok = gpu_op(dit, h3_gpu_sglang_rope_trig(dit->gpu,dit->rope_cos,
                dit->rope_sin,cos_f32,(uint32_t)count),error,error_size,"CUDA RoPE trigonometry");
        else
#endif
        if (ok) ok = gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_cos, cos_f32, (uint32_t)count),
                 error, error_size, "RoPE cosine cast") &&
             gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                 dit->gpu, dit->rope_sin, sin_f32, (uint32_t)count),
                 error, error_size, "RoPE sine cast");
        if (ok && reduced_count) {
            ok = gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_cos, reduced_cos_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE cosine cast") &&
                 gpu_op(dit, h3_gpu_cast_f32_to_bf16(
                     dit->gpu, dit->reduced_rope_sin, reduced_sin_f32,
                     (uint32_t)reduced_count), error, error_size,
                     "reduced RoPE sine cast");
        }
        if (ok) ok =
             gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit RoPE setup");
    } else if (!error || !*error) {
        fail(error, error_size, "cannot allocate DiT RoPE buffers: %s",
             h3_gpu_error(dit->gpu));
    }
    free_tensor(&cos_f32);
    free_tensor(&sin_f32);
    free_tensor(&reduced_cos_f32);
    free_tensor(&reduced_sin_f32);
    return ok;
}

static int prepare_maps(h3_dit *dit, const h3_text_embedding *text,
                        char *error, size_t error_size) {
    int steps = h3_dit_schedule_steps(dit->schedule);
    dit->row_maps = calloc((size_t)steps, sizeof(*dit->row_maps));
    if (dit->token_reduction)
        dit->reduced_row_maps = calloc((size_t)steps,
                                       sizeof(*dit->reduced_row_maps));
    dit->final_audio_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_audio_maps));
    dit->final_video_maps = calloc((size_t)steps,
                                   sizeof(*dit->final_video_maps));
    uint32_t *rows = malloc((size_t)dit->sequence * sizeof(*rows));
    uint32_t *reduced = dit->token_reduction ?
        malloc((size_t)dit->reduced_sequence * sizeof(*reduced)) : NULL;
    uint32_t *audio = malloc((size_t)dit->audio_rows * sizeof(*audio));
    uint32_t *video = malloc((size_t)dit->video_rows * sizeof(*video));
    if (!dit->row_maps || !dit->final_audio_maps || !dit->final_video_maps ||
        (dit->token_reduction && (!dit->reduced_row_maps || !reduced)) ||
        !rows || !audio || !video) {
        fail(error, error_size, "out of memory allocating modulation row maps");
        free(rows); free(reduced); free(audio); free(video);
        return 0;
    }
    for (int step = 0; step < steps; step++) {
        if (!h3_dit_schedule_row_map(dit->schedule, step, &dit->layout,
                                     text->tags, text->tokens, rows,
                                     dit->sequence)) {
            fail(error, error_size, "cannot construct modulation row map");
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
        if (dit->token_reduction) {
            for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
                uint32_t first, second;
                token_pool_sources(dit, row, &first, &second);
                (void)second;
                reduced[row] = rows[first];
            }
            dit->reduced_row_maps[step] = h3_gpu_tensor_from_u32(
                dit->gpu, reduced, dit->reduced_sequence);
        }
        if(dit->sglang_reference && step==0 && getenv("H3_TEST_SGLANG_BLOCK_ONLY")) {
            uint32_t count=h3_dit_schedule_time_rows(dit->schedule);
            float *times=malloc((size_t)count*sizeof(float));
            if(!times){free(rows);free(reduced);free(audio);free(video);return 0;}
            for(uint32_t i=0;i<count;i++)times[i]=h3_dit_schedule_timestep(dit->schedule,i);
            int ok=h3_sglang_dump("block-times.f32",times,(size_t)count*sizeof(float));
            free(times);
            if(!ok){free(rows);free(reduced);free(audio);free(video);return 0;}
        }
        for (uint32_t index = 0; index < dit->audio_rows; index++)
            audio[index] = rows[dit->audio_target_start+index]/H3_DIT_MODALITIES;
        for (uint32_t index = 0; index < dit->video_rows; index++)
            video[index] = rows[dit->video_target_start+index]/H3_DIT_MODALITIES;
        dit->row_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, rows, dit->sequence);
        dit->final_audio_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_rows);
        dit->final_video_maps[step] = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_rows);
        if (!dit->row_maps[step] ||
            (dit->token_reduction && !dit->reduced_row_maps[step]) ||
            !dit->final_audio_maps[step] ||
            !dit->final_video_maps[step]) {
            fail(error, error_size, "cannot allocate modulation row maps: %s",
                 h3_gpu_error(dit->gpu));
            free(rows); free(reduced); free(audio); free(video);
            return 0;
        }
    }
    free(rows); free(reduced); free(audio); free(video);
    return 1;
}

static int prepare_projection_maps(h3_dit *dit, char *error,
                                   size_t error_size) {
    unsigned video_segments = 0, audio_segments = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        h3_segment_kind kind = dit->layout.segments[index].kind;
        if (kind == H3_SEG_COND || kind == H3_SEG_REF_IMAGE ||
            kind == H3_SEG_VIDEO)
            video_segments++;
        else if (kind != H3_SEG_TEXT)
            audio_segments++;
    }
    uint32_t *video = video_segments > 1 ?
        malloc((size_t)dit->video_total_rows * sizeof(*video)) : NULL;
    uint32_t *audio = audio_segments > 1 ?
        malloc((size_t)dit->audio_total_rows * sizeof(*audio)) : NULL;
    if ((video_segments > 1 && !video) || (audio_segments > 1 && !audio)) {
        free(video); free(audio);
        fail(error, error_size, "out of memory allocating projection maps");
        return 0;
    }
    size_t video_offset = 0, audio_offset = 0;
    for (size_t index = 0; index < dit->layout.segment_count; index++) {
        const h3_segment *segment = &dit->layout.segments[index];
        size_t rows = segment->stop - segment->start;
        if (segment->kind == H3_SEG_COND ||
            segment->kind == H3_SEG_REF_IMAGE ||
            segment->kind == H3_SEG_VIDEO) {
            for (size_t row = 0; video && row < rows; row++)
                video[video_offset + row] = (uint32_t)(segment->start + row);
            video_offset += rows;
        } else if (segment->kind != H3_SEG_TEXT) {
            for (size_t row = 0; audio && row < rows; row++)
                audio[audio_offset + row] = (uint32_t)(segment->start + row);
            audio_offset += rows;
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        free(video); free(audio);
        fail(error, error_size, "projection map rows are inconsistent");
        return 0;
    }
    if (video)
        dit->video_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, video, dit->video_total_rows);
    if (audio)
        dit->audio_projection_map = h3_gpu_tensor_from_u32(
            dit->gpu, audio, dit->audio_total_rows);
    free(video); free(audio);
    if ((video_segments > 1 && !dit->video_projection_map) ||
        (audio_segments > 1 && !dit->audio_projection_map)) {
        fail(error, error_size, "cannot allocate projection map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static int prepare_token_reduction_maps(h3_dit *dit, char *error,
                                        size_t error_size) {
    if (!dit->token_reduction) return 1;
    size_t pair_count = (size_t)dit->reduced_sequence * 2;
    uint32_t *pairs = malloc(pair_count * sizeof(*pairs));
    uint32_t *baseline_indices = malloc(
        (size_t)dit->reduced_sequence * sizeof(*baseline_indices));
    uint32_t *parents = malloc((size_t)dit->sequence * sizeof(*parents));
    if (!pairs || !baseline_indices || !parents) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size,
             "out of memory allocating token-reduction maps");
        return 0;
    }
    uint32_t baseline_row = 0;
    for (uint32_t row = 0; row < dit->reduced_sequence; row++) {
        token_pool_sources(dit, row, &pairs[(size_t)row * 2],
                           &pairs[(size_t)row * 2 + 1]);
        baseline_indices[row] =
            row >= dit->video_target_start &&
            pairs[(size_t)row * 2] != pairs[(size_t)row * 2 + 1] ?
                baseline_row++ : UINT32_MAX;
    }
    if (baseline_row != dit->token_baseline_rows) {
        free(pairs);
        free(baseline_indices);
        free(parents);
        fail(error, error_size, "token-reduction baseline map is inconsistent");
        return 0;
    }
    for (uint32_t row = 0; row < dit->sequence; row++)
        parents[row] = token_reduced_parent(dit, row);
    dit->token_pool_pairs = h3_gpu_tensor_from_u32(
        dit->gpu, pairs, pair_count);
    dit->token_baseline_indices = h3_gpu_tensor_from_u32(
        dit->gpu, baseline_indices, dit->reduced_sequence);
    dit->token_expand_parents = h3_gpu_tensor_from_u32(
        dit->gpu, parents, dit->sequence);
    free(pairs);
    free(baseline_indices);
    free(parents);
    if (!dit->token_pool_pairs || !dit->token_baseline_indices ||
        !dit->token_expand_parents) {
        fail(error, error_size,
             "cannot allocate token-reduction map tensors: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    return 1;
}

static void configure_active_blocks(h3_dit *dit, unsigned active) {
    memset(dit->block_active, 1, sizeof(dit->block_active));
    dit->active_block_count = active;
    unsigned skipped = H3_DIT_BLOCKS - active;
    for (unsigned index = 0; index < skipped; index++) {
        unsigned block = ((2 * index + 1) * H3_DIT_BLOCKS) / (2 * skipped);
        if (block == 0) block = 1;
        if (block >= H3_DIT_BLOCKS - 1) block = H3_DIT_BLOCKS - 2;
        dit->block_active[block] = 0;
    }
}

static unsigned first_active_block(const h3_dit *dit) {
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block] && !(dit->weight_plan.resident_mask & (UINT64_C(1)<<block))) return block;
    return H3_DIT_BLOCKS;
}

static unsigned next_active_block(const h3_dit *dit, unsigned current) {
    for (unsigned block = current + 1; block < H3_DIT_BLOCKS; block++)
        if (dit->block_active[block] && !(dit->weight_plan.resident_mask & (UINT64_C(1)<<block))) return block;
    return H3_DIT_BLOCKS;
}

static void configure_gate_ranked_blocks(h3_dit *dit) {
    if(getenv("H3_TEST_DIT_BLOCKS"))return;
    const char *policy = getenv("H3_DIT_LAYER_POLICY");
    if ((policy && !strcmp(policy, "uniform")) ||
        dit->active_block_count == H3_DIT_BLOCKS) return;
    typedef struct { unsigned block; double score; } block_score;
    /* The first two and final blocks establish/close the residual stream.
     * Block 1 has a small gate but proved structurally essential in decoded
     * A/B renders, so magnitude ranking must not treat it as disposable. */
    block_score scores[H3_DIT_BLOCKS - 3];
    for (unsigned block = 2; block + 1 < H3_DIT_BLOCKS; block++) {
        double score = h3_dit_schedule_gate_score(dit->schedule, block);
        if (score < 0.0) return;
        scores[block - 2] = (block_score){block, score};
    }
    unsigned count = H3_DIT_BLOCKS - 3;
    for (unsigned left = 0; left < count; left++) {
        unsigned least = left;
        for (unsigned right = left + 1; right < count; right++)
            if (scores[right].score < scores[least].score) least = right;
        block_score temporary = scores[left];
        scores[left] = scores[least];
        scores[least] = temporary;
    }
    memset(dit->block_active, 1, sizeof(dit->block_active));
    unsigned skipped = H3_DIT_BLOCKS - dit->active_block_count;
    for (unsigned index = 0; index < skipped; index++)
        dit->block_active[scores[index].block] = 0;
    if (getenv("H3_PROFILE")) {
        fprintf(stderr, "h3cli: gate-ranked DiT skips");
        for (unsigned index = 0; index < skipped; index++)
            fprintf(stderr, " %u(%.4g)", scores[index].block,
                    scores[index].score);
        fputc('\n', stderr);
    }
}

static int load_core(h3_dit *dit, h3_dit_progress progress, void *opaque,
                     char *error, size_t error_size) {
    for (unsigned index = 0; index < H3_DIT_BLOCKS; index++) {
        if (!h3_memory_check(0, "load transformer core", error, error_size)) return 0;
        if (!dit->block_active[index]) {
            if (!report(progress, opaque, "load transformer core", (int)index + 1,
                   H3_DIT_BLOCKS, error, error_size)) return 0;
            continue;
        }
        char prefix[64];
        snprintf(prefix, sizeof(prefix), "blocks.%u.", index);
        if (dit->ssd_streaming && !(dit->weight_plan.resident_mask & (UINT64_C(1)<<index))) {
            if (!load_block_norms(dit, &dit->blocks[index], prefix,
                                  error, error_size) ||
                !prepare_stream_layer(dit, index, error, error_size))
                return 0;
        } else {
            if (!load_block(dit, &dit->blocks[index], prefix,
                            error, error_size)) return 0;
            if (dit->int8_mlp &&
                !quantize_block_mlp(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_qkv &&
                !quantize_block_qkv(dit, &dit->blocks[index],
                                    error, error_size)) return 0;
            if (dit->int8_attention_out &&
                !quantize_block_attention_out(
                    dit, &dit->blocks[index], error, error_size)) return 0;
        }
        if (!report(progress, opaque, "load transformer core", (int)index + 1,
                   H3_DIT_BLOCKS, error, error_size)) return 0;
    }
    if (dit->ssd_streaming) {
        if (!allocate_stream_slot(dit, &dit->stream_slots[0],
                                  error, error_size) ||
            !allocate_stream_slot(dit, &dit->stream_slots[1],
                                  error, error_size)) return 0;
        unsigned first = first_active_block(dit);
        if (first == H3_DIT_BLOCKS) {
            fail(error, error_size, "SSD stream has no active DiT block");
            return 0;
        }
        h3_dit_stream_job job = {
            .dit = dit, .layer = first, .slot = 0
        };
        if (!read_stream_layer(&job)) {
            fail(error, error_size, "cannot prime DiT SSD stream: %s",
                 job.error);
            return 0;
        }
        dit->stream_ready_layer = first;
        dit->stream_ready_slot = 0;
        dit->stream_bytes += job.bytes;dit->stream_read_layers++;
        dit->stream_read_seconds += job.seconds;
    }
    dit->video_patch_w = f2(dit, "video_patch_proj.weight", HIDDEN,
                            VIDEO_PATCH, error, error_size);
    dit->video_patch_b = f1(dit, "video_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->audio_patch_w = f2(dit, "audio_patch_proj.weight", HIDDEN,
                            AUDIO_CHANNELS, error, error_size);
    dit->audio_patch_b = f1(dit, "audio_patch_proj.bias", HIDDEN,
                            error, error_size);
    dit->final_norm = bf1(dit, "final_layer.norm.weight", HIDDEN,
                          error, error_size);
    dit->final_video_w = f2(dit, "final_layer.video_out.weight", VIDEO_PATCH,
                            HIDDEN, error, error_size);
    dit->final_video_b = f1(dit, "final_layer.video_out.bias", VIDEO_PATCH,
                            error, error_size);
    dit->final_audio_w = f2(dit, "final_layer.audio_out.weight", AUDIO_CHANNELS,
                            HIDDEN, error, error_size);
    dit->final_audio_b = f1(dit, "final_layer.audio_out.bias", AUDIO_CHANNELS,
                            error, error_size);
    if (dit->bf16_final && dit->final_video_w && dit->final_video_b &&
        dit->final_audio_w && dit->final_audio_b) {
        h3_gpu_tensor *source[4] = {
            dit->final_video_w, dit->final_video_b,
            dit->final_audio_w, dit->final_audio_b
        };
        size_t elements[4] = {
            (size_t)VIDEO_PATCH * HIDDEN, VIDEO_PATCH,
            (size_t)AUDIO_CHANNELS * HIDDEN, AUDIO_CHANNELS
        };
        h3_gpu_tensor *target[4] = {0};
        int ok = 1;
        for (unsigned index = 0; index < 4; index++) {
            target[index] = h3_gpu_tensor_new_bf16(dit->gpu,
                                                    elements[index]);
            if (!target[index]) ok = 0;
        }
        if (ok) ok = h3_gpu_begin(dit->gpu);
        for (unsigned index = 0; ok && index < 4; index++)
            ok = h3_gpu_cast_f32_to_bf16(dit->gpu, target[index],
                                         source[index],
                                         (uint32_t)elements[index]);
        if (ok) ok = h3_gpu_submit(dit->gpu);
        if (ok) {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(source[index]);
            dit->final_video_w = target[0];
            dit->final_video_b = target[1];
            dit->final_audio_w = target[2];
            dit->final_audio_b = target[3];
        } else {
            for (unsigned index = 0; index < 4; index++)
                h3_gpu_tensor_free(target[index]);
            fail(error, error_size, "cannot convert DiT final weights: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return dit->video_patch_w && dit->video_patch_b && dit->audio_patch_w &&
           dit->audio_patch_b && dit->final_norm && dit->final_video_w &&
           dit->final_video_b && dit->final_audio_w && dit->final_audio_b;
}

static int allocate_activations(h3_dit *dit, char *error, size_t error_size) {
    size_t sequence = dit->sequence;
    size_t audio = dit->audio_rows;
    size_t video = dit->video_rows;
    size_t audio_total = dit->audio_total_rows;
    size_t video_total = dit->video_total_rows;
    dit->activation_aliases = !getenv("H3_DISABLE_DIT_ACTIVATION_ALIAS");
    dit->fused_patch_projection =
        !getenv("H3_DISABLE_FUSED_PATCH_CAST") && !getenv("H3_SCALAR_PATCH");
    dit->fused_patch_pack = dit->fused_patch_projection &&
        !getenv("H3_DISABLE_FUSED_PATCH_PACK");
#define BF(field, elements) (dit->field = h3_gpu_tensor_new_bf16(dit->gpu, (elements)))
#define F32(field, elements) (dit->field = h3_gpu_tensor_new_f32(dit->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(video_input, video_total * VIDEO_PATCH),
        F32(audio_input, audio_total * AUDIO_CHANNELS),
        BF(hidden, sequence * HIDDEN),
        BF(mod_attention, sequence * HIDDEN),
        BF(qkv, sequence * INNER * 3),
        BF(query, sequence * INNER),
        BF(key, sequence * INNER),
        BF(value, sequence * INNER),
        BF(attention_output, sequence * HIDDEN),
        F32(final_audio_inverse, audio),
        F32(final_video_inverse, video),
        BF(audio_output_bf16, audio * AUDIO_CHANNELS),
        BF(video_output_bf16, video * VIDEO_PATCH)
    };
#undef BF
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            fail(error, error_size, "cannot allocate DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_pack) {
        dit->video_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected = h3_gpu_tensor_new_bf16(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected || !dit->audio_projected) {
            fail(error, error_size,
                 "cannot allocate packed patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_patch_projection) {
        dit->video_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video_total * HIDDEN);
        dit->audio_projected_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio_total * HIDDEN);
        if (!dit->video_projected_f32 || !dit->audio_projected_f32) {
            fail(error, error_size,
                 "cannot allocate separate patch projections: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->activation_aliases) {
        dit->attention_heads = dit->qkv;
        dit->mod_mlp = dit->qkv;
        dit->mlp_output = NULL;
    } else {
        dit->attention_heads = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * INNER);
        dit->mod_mlp = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->mlp_output = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
    }
    if (!dit->attention_heads || !dit->mod_mlp ||
        (!dit->activation_aliases && !dit->mlp_output)) {
        fail(error, error_size,
             "cannot allocate DiT activation buffers: %s",
             h3_gpu_error(dit->gpu));
        return 0;
    }
    if (getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_input = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_input = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_input || !dit->final_video_input) {
            fail(error, error_size,
                 "cannot allocate separate final DiT slices: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final || getenv("H3_DISABLE_FUSED_FINAL_HEAD") ||
        getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        dit->final_audio_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, audio * HIDDEN);
        dit->final_video_norm = h3_gpu_tensor_new_bf16(
            dit->gpu, video * HIDDEN);
        if (!dit->final_audio_norm || !dit->final_video_norm) {
            fail(error, error_size,
                 "cannot allocate separate final DiT normalization: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->bf16_final) {
        dit->final_audio_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, audio * HIDDEN);
        dit->final_video_f32 = h3_gpu_tensor_new_f32(
            dit->gpu, video * HIDDEN);
        dit->audio_output = h3_gpu_tensor_new_f32(
            dit->gpu, audio * AUDIO_CHANNELS);
        dit->video_output = h3_gpu_tensor_new_f32(
            dit->gpu, video * VIDEO_PATCH);
        if (!dit->final_audio_f32 || !dit->final_video_f32 ||
            !dit->audio_output || !dit->video_output) {
            fail(error, error_size,
                 "cannot allocate F32 DiT final activations: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (!dit->fused_mlp) {
        dit->fc1 = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN * 2);
    }
    if (!dit->fused_mlp || dit->nax_mlp || dit->int8_mlp) {
        dit->activated = h3_gpu_tensor_new_bf16(dit->gpu, sequence * FFN);
        if ((!dit->fused_mlp && !dit->fc1) || !dit->activated) {
            fail(error, error_size,
                 "cannot allocate diagnostic DiT MLP tensors: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->int8_mlp || dit->int8_qkv || dit->int8_attention_out) {
        size_t padded_sequence = (sequence + 127) & ~(size_t)127;
        dit->int8_activation = h3_gpu_tensor_new_i8(
            dit->gpu, padded_sequence * FFN);
        dit->int8_activation_scales = h3_gpu_tensor_new_f32(
            dit->gpu, padded_sequence * (FFN / 1024));
        if (!dit->int8_activation || !dit->int8_activation_scales) {
            fail(error, error_size,
                 "cannot allocate int8 DiT activation arena: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->token_reduction) {
        size_t full_elements = sequence * HIDDEN;
        size_t qkv_capacity = sequence * INNER * 3;
        size_t qkv_used = (size_t)dit->reduced_sequence * INNER * 3;
        size_t baseline_elements =
            (size_t)dit->token_baseline_rows * HIDDEN;
        size_t attention_capacity = sequence * HIDDEN;
        size_t attention_used =
            (size_t)dit->reduced_sequence * HIDDEN;
        dit->token_original_in_qkv =
            qkv_used <= qkv_capacity &&
            full_elements <= qkv_capacity - qkv_used &&
            qkv_used <= UINT32_MAX &&
            full_elements <= UINT32_MAX - qkv_used;
        if (dit->token_original_in_qkv)
            dit->token_original_offset = qkv_used;
        else
            dit->token_original = h3_gpu_tensor_new_bf16(
                dit->gpu, full_elements);
        dit->token_baseline_offset = attention_used;
        if (attention_used > attention_capacity ||
            baseline_elements > attention_capacity - attention_used ||
            attention_used > UINT32_MAX ||
            baseline_elements > UINT32_MAX - attention_used ||
            (!dit->token_original_in_qkv && !dit->token_original)) {
            fail(error, error_size,
                 "cannot allocate token-reduction residual state: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    if (dit->core_reuse_interval > 1) {
        dit->core_input = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        dit->core_residual = h3_gpu_tensor_new_bf16(
            dit->gpu, sequence * HIDDEN);
        if (!dit->core_input || !dit->core_residual) {
            fail(error, error_size,
                 "cannot allocate DiT core residual cache: %s",
                 h3_gpu_error(dit->gpu));
            return 0;
        }
    }
    return 1;
}

typedef struct {
    h3_dit_progress callback;
    void *opaque;
} schedule_progress;

static int schedule_report(int completed, int total, void *opaque) {
    schedule_progress *state = opaque;
    return state->callback && state->callback("precompute AdaLN", completed, total, state->opaque);
}

static h3_dit *load_dit(const char *weight_directory,
                        const char *shader_source_path,
                        const h3_text_embedding *text,
                        const h3_layout *layout,
                        const h3_sigma_schedule *sigmas,
                        unsigned active_blocks,
                        unsigned core_reuse_interval,
                        int token_reduction,
                        int ssd_streaming,
                        float spatial_rope_scale,
                        int use_slower_bf16_mlp,
                        int use_slower_bf16_qkv,
                        int use_slower_bf16_attention_output,
                        int use_slower_row_major_attention_output,
                        int use_slower_unfused_int8_inputs,
                        int use_slower_unfused_qkv_rope,
                        int use_slower_scalar_qkv_rms,
                        int use_slower_uncached_int8_scales,
                        int use_slower_dynamic_fc1_k,
                        int use_slower_grouped_quantizer,
                        int use_int8_row_fc2,
                        const float *condition_video_rows,
                        size_t condition_video_elements,
                        const float *condition_audio_rows,
                        size_t condition_audio_elements,
                        int released_refvideo,
                        h3_dit_progress progress, void *progress_opaque,
                        const h3_sampler_state *resume,
                        const h3_weight_options *retry_options,int retry_cap,
                        char *error, size_t error_size) {
#ifdef __APPLE__
    (void)retry_options;(void)retry_cap;
#endif
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !layout || !sigmas ||
        (ssd_streaming != 0 && ssd_streaming != 1) ||
        (released_refvideo != 0 && released_refvideo != 1) ||
        !isfinite(spatial_rope_scale) || spatial_rope_scale <= 0.0f ||
        active_blocks < H3_DIT_BLOCKS / 2 ||
        active_blocks > H3_DIT_BLOCKS || core_reuse_interval < 1 ||
        core_reuse_interval > 6) {
        fail(error, error_size, "invalid DiT load arguments");
        return NULL;
    }
    if (!report(progress, progress_opaque, "DiT initialization", 0, 1,
                 error, error_size)) return NULL;
    h3_dit *dit = calloc(1, sizeof(*dit));
    if (!dit) {
        fail(error, error_size, "out of memory creating DiT model");
        return NULL;
    }
    dit->fused_mlp = getenv("H3_DISABLE_FUSED_MLP") == NULL;
    dit->metal_q8=h3_backend_current().metal.weight_format==H3_WEIGHT_Q8;
    if(dit->metal_q8) {
        if(h3_backend_current().backend!=H3_BACKEND_METAL||ssd_streaming||h3_backend_current().metal.ane_mode) {
            fail(error,error_size,"Q8 requires Metal, resident weights, and --metal-ane off");goto failed;
        }
        dit->fused_mlp=0;
    }
    /* The released final heads are F32, but their inputs are already BF16.
     * Converting these small weights once selects the Iris-derived tiled
     * linear and eliminates two full-width casts plus the scalar F32 kernel.
     * Keep the old path available for close-reference diagnosis. */
    dit->sglang_reference=h3_sglang_requested();
    dit->bf16_final = !dit->sglang_reference && getenv("H3_DIT_F32_FINAL") == NULL;
    dit->core_reuse_interval = core_reuse_interval;
    dit->adaptive_mode=h3_cuda_policy_current().adaptive_cache;
    dit->adaptive_threshold=h3_cuda_policy_current().adaptive_cache_threshold;
    dit->adaptive_max_hits=h3_cuda_policy_current().adaptive_cache_max_hits;
    dit->adaptive_budget=h3_adaptive_budget(h3_cuda_policy_current().adaptive_cache_max_bytes);
    dit->captured_attention=h3_cuda_policy_current().attention;
    dit->adaptive_warmup=h3_adaptive_warmup(h3_cuda_policy_current().adaptive_cache_warmup);
    dit->subblock_warmup=h3_subblock_warmup(h3_cuda_policy_current().subblock_warmup);
    dit->resume_sampler_mode = resume ? (int)resume->sampler_mode : -1;
    dit->ssd_streaming = ssd_streaming;
    dit->explicit_ssd_streaming = ssd_streaming;
    dit->spatial_rope_scale = spatial_rope_scale;
    configure_active_blocks(dit, active_blocks);
    const char *bench_blocks=getenv("H3_TEST_DIT_BLOCKS");
    if(bench_blocks) {
        if(!getenv("H3_TEST_MAX_EVALUATIONS") || strcmp(getenv("H3_TEST_MAX_EVALUATIONS"),"6") ||
            (strcmp(bench_blocks,"1")&&strcmp(bench_blocks,"5")&&strcmp(bench_blocks,"50")) ||
            active_blocks!=50 || core_reuse_interval!=1 || token_reduction || resume) {
            fail(error,error_size,"benchmark block counts require 1/5/50, the six-evaluation test budget, full-depth settings, no reuse/reduction and no sampler resume");goto failed;
        }
        unsigned count=(unsigned)strtoul(bench_blocks,NULL,10);
        memset(dit->block_active,0,sizeof(dit->block_active));memset(dit->block_active,1,count);
        dit->active_block_count=count;
        fprintf(stderr,"h3cli: TEST workload executes first %u blocks; not a full-depth quality result\n",count);
    }
    if (!copy_layout(dit, layout, error, error_size) ||
        !validate_layout(dit, text, error, error_size) ||
        !configure_token_reduction(dit, token_reduction,
                                   error, error_size)) goto failed;
    dit->adaptive_recipe=h3_adaptive_execution_recipe(dit->adaptive_mode,h3_cuda_policy_current().projection_precision,
        dit->layout.img_cond_rows||dit->layout.audio_cond_rows,
        dit->layout.prefix.video_prefix_t!=0 && !dit->layout.frozen_audio);
    if(dit->adaptive_mode) {
        if(!dit->video_rows||!dit->audio_rows||dit->video_target_start>dit->sequence-dit->video_rows||
           dit->audio_target_start>dit->sequence-dit->audio_rows){fail(error,error_size,"invalid adaptive target ranges");goto failed;}
        H3_VERBOSE("h3cli: adaptive preset=%s threshold=%.9g max_hits=%d recipe=%u warmup=%d\n",
            h3_adaptive_name(dit->adaptive_mode),dit->adaptive_threshold,dit->adaptive_max_hits,dit->adaptive_recipe,dit->adaptive_warmup);
    }
    if(dit->adaptive_recipe==H3_ADAPTIVE_CONTINUATION_VERSION &&
       (h3_cuda_policy_current().projection_precision ||
        !h3_adaptive_regions_build(&dit->layout,&dit->adaptive_regions,error,error_size)))goto failed;
    if(!h3_adaptive_plan_recipe(dit->adaptive_mode,dit->adaptive_recipe,dit->sequence,HIDDEN,dit->adaptive_budget,&dit->adaptive_plan,error,error_size))goto failed;
    if ((layout->prefix.video_prefix_t || layout->prefix.audio_prefix_t) &&
        (dit->token_reduction ||
         (core_reuse_interval != 1 && (!layout->bridge ||
          (core_reuse_interval != 4 && core_reuse_interval != 6))) ||
         dit->active_block_count != H3_DIT_BLOCKS)) {
        fail(error,error_size,"continuation requires all 50 blocks and token reduction off; core reuse is 1 for hard or 1/4/6 for bridge");
        goto failed;
    }
    size_t wanted_video_condition =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio_condition =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video_condition ||
        condition_audio_elements != wanted_audio_condition ||
        (wanted_video_condition && !condition_video_rows) ||
        (wanted_audio_condition && !condition_audio_rows)) {
        fail(error, error_size,
             "condition row elements do not match the packed DiT layout");
        goto failed;
    }
    dit->sigmas = *sigmas;
    dit->weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!dit->weights) goto failed;
#ifndef __APPLE__
    h3_cuda_policy captured=h3_cuda_policy_current();
    if(retry_options)captured.weights=*retry_options;
    else if(!captured.weights_captured && !h3_weight_options_read(&captured.weights,ssd_streaming,error,error_size))goto failed;
    captured.weights_captured=1;
    h3_cuda_policy previous=h3_cuda_policy_exchange(captured);
#endif
    dit->gpu = h3_gpu_create(shader_source_path, error, error_size);
#ifndef __APPLE__
    h3_cuda_policy_exchange(previous);
#endif
    if (!dit->gpu) goto failed;
    if(dit->adaptive_mode) {
        dit->adaptive_anchor=h3_gpu_tensor_alloc(dit->gpu,dit->adaptive_plan.elements,H3_GPU_BF16,H3_GPU_DEVICE_ONLY);
        dit->adaptive_delta=h3_gpu_tensor_alloc(dit->gpu,dit->adaptive_plan.elements,H3_GPU_BF16,H3_GPU_DEVICE_ONLY);
        dit->adaptive_scratch=h3_gpu_tensor_alloc(dit->gpu,dit->adaptive_plan.elements,H3_GPU_BF16,H3_GPU_DEVICE_ONLY);
        dit->adaptive_reduction=h3_gpu_tensor_alloc(dit->gpu,dit->adaptive_plan.scratch_bytes/4,H3_GPU_F32,H3_GPU_DEVICE_ONLY);
        if(!dit->adaptive_anchor||!dit->adaptive_delta||!dit->adaptive_scratch||!dit->adaptive_reduction){fail(error,error_size,"adaptive cache device allocation failed before weight admission: %s",h3_gpu_error(dit->gpu));goto failed;}
    }
    if(!h3_gpu_dit_attention_configure(dit->gpu,h3_attention_current())) {
        fail(error,error_size,"%s",h3_gpu_error(dit->gpu));goto failed;
    }
#ifndef __APPLE__
    if(h3_attention_current()==H3_ATTENTION_SOL||h3_attention_current()==H3_ATTENTION_SUBBLOCK||getenv("H3_TEST_ATTENTION_CAPTURE_DIR")) {
        h3_cuda_sol_options sol=h3_cuda_sol_current();h3_sol_layout protection={0};
        if(h3_attention_current()==H3_ATTENTION_SUBBLOCK)sol.q_block=sol.kv_block=64;
        if(!h3_sol_layout_build(layout,(unsigned)sol.q_block,(unsigned)sol.kv_block,&protection,error,error_size))goto failed;
        int protected_ok=h3_gpu_dit_sol_layout(dit->gpu,&protection);h3_sol_layout_free(&protection);
        if(!protected_ok){fail(error,error_size,"%s",h3_gpu_error(dit->gpu));goto failed;}
    }
#endif
#ifdef __APPLE__
    h3_backend_scope native=h3_backend_current();
    if(native.backend==H3_BACKEND_METAL) {
        if(native.metal.ane_mode) {
            dit->ane=h3_ane_split_create(native.metal,HIDDEN,INNER*3);
            if(!dit->ane || (resume&&!h3_ane_split_restore(dit->ane,resume->ane_decided,resume->ane_rows))) {
                fail(error,error_size,"cannot restore the recorded ANE execution plan");goto failed;
            }
        }
        H3_VERBOSE("h3cli: hybrid: %s + native %s attention, dense kernel=%s; production unqualified\n",
            dit->metal_q8?(native.metal.q8_kernel?"native Q8/BF16 simdgroup core linears; BF16 sensitive weights":
                "Q8 core weights + bounded Metal dequantization + MPSGraph BF16 linears"):
            native.metal.ane_mode?"MPSGraph BF16 linears with scaled FP16 ANE QKV shards":"MPSGraph BF16 linears",
            h3_attention_mode_name(native.attention),h3_metal_candidate_name(native.metal.candidate));
        if(native.attention==H3_ATTN_SOL) {
            h3_sol_layout protection={0};
            if(!h3_sol_layout_build(layout,(unsigned)native.metal.q_block,(unsigned)native.metal.kv_block,
                &protection,error,error_size))goto failed;
            int protected_ok=h3_gpu_native_attention_layout(dit->gpu,&protection);
            h3_sol_layout_free(&protection);
            if(!protected_ok){fail(error,error_size,"%s",h3_gpu_error(dit->gpu));goto failed;}
        }
    }
#endif
    dit->cuda_quant=h3_quant_current().mode;
    if(dit->cuda_quant) {
        uint64_t weights=(uint64_t)dit->active_block_count*2*(4*(uint64_t)INNER*HIDDEN+3*(uint64_t)FFN*HIDDEN);
        uint64_t activations=(uint64_t)dit->sequence*2*(12*(uint64_t)HIDDEN+8*(uint64_t)INNER+4*(uint64_t)FFN);
        if(!h3_gpu_quant_configure(dit->gpu,dit->cuda_quant,h3_quant_current().cache,weights,activations,ssd_streaming)) {
            fail(error,error_size,"%s",h3_gpu_error(dit->gpu));goto failed;
        }
        /* Packed streaming belongs to the backend descriptor, not BF16 SSD slots. */
        dit->ssd_streaming=0;
        dit->fused_mlp=0;
    }
    #ifdef __APPLE__
    if (!dit->ssd_streaming && !dit->cuda_quant) {
        uint64_t core_weights = (uint64_t)dit->active_block_count * 2 *
            (4 * (uint64_t)INNER * HIDDEN + 3 * (uint64_t)FFN * HIDDEN);
        if(dit->metal_q8)core_weights=core_weights/2+core_weights/(H3_Q8_GROUP/2);
        uint64_t activation_estimate = (uint64_t)dit->sequence * 2 *
            (12 * (uint64_t)HIDDEN + 8 * (uint64_t)INNER + 4 * (uint64_t)FFN);
        int streaming = h3_gpu_plan_weights(dit->gpu,
            core_weights + (UINT64_C(1) << 30), activation_estimate);
        if (streaming < 0) {
            fail(error, error_size, "%s", h3_gpu_error(dit->gpu));
            goto failed;
        }
        dit->ssd_streaming = streaming;
    }
    #endif
    dit->nax_mlp = dit->fused_mlp && h3_gpu_has_nax_mlp(dit->gpu);
    dit->int8_mlp = !dit->ssd_streaming && dit->fused_mlp &&
                    !use_slower_bf16_mlp &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_qkv = !dit->ssd_streaming && !use_slower_bf16_qkv &&
                    dit->sequence >= 128 &&
                    h3_gpu_has_int8_mlp(dit->gpu);
    dit->int8_attention_out = !dit->ssd_streaming &&
                              !use_slower_bf16_attention_output &&
                              dit->sequence >= 128 &&
                              h3_gpu_has_int8_mlp(dit->gpu);
    if(h3_backend_current().backend==H3_BACKEND_METAL)
        dit->nax_mlp=dit->int8_mlp=dit->int8_qkv=dit->int8_attention_out=0;
    dit->use_slower_row_major_attention_output =
        use_slower_row_major_attention_output;
    dit->use_slower_unfused_int8_inputs =
        use_slower_unfused_int8_inputs;
    dit->use_slower_unfused_qkv_rope =
        use_slower_unfused_qkv_rope;
    dit->use_slower_scalar_qkv_rms = use_slower_scalar_qkv_rms;
    dit->use_slower_uncached_int8_scales =
        use_slower_uncached_int8_scales;
    dit->use_slower_dynamic_fc1_k = use_slower_dynamic_fc1_k;
    dit->keep_bf16_attention_out = dit->int8_attention_out &&
        (getenv("H3_INT8_KEEP_BF16_ATTENTION_OUT") ||
         getenv("H3_BENCH_INT8_ATTENTION_OUT_AB"));
    dit->keep_bf16_qkv = dit->int8_qkv &&
        (getenv("H3_INT8_KEEP_BF16_QKV") ||
         getenv("H3_BENCH_INT8_QKV_AB"));
    dit->use_slower_grouped_quantizer = use_slower_grouped_quantizer;
    dit->use_int8_row_fc2 = dit->int8_mlp && use_int8_row_fc2;
    dit->keep_bf16_mlp = dit->int8_mlp &&
        (getenv("H3_INT8_KEEP_BF16_MLP") ||
         getenv("H3_BENCH_INT8_MLP_AB") ||
         getenv("H3_INT8_MLP_STAGE"));
    h3_gpu_profile_set_label(dit->gpu, "H3 DiT");
    uint8_t prepared_key[32];
    int cache_ok=resume && resume->prepared.version==1 &&
        h3_sampler_prepared_key(resume,prepared_key) && !memcmp(prepared_key,resume->prepared.key,32);
    const h3_prepared_tensor *refined=cache_ok?h3_prepared_find(&resume->prepared,1,text->tokens*HIDDEN):NULL;
    const h3_conditioning *conditioning=h3_conditioning_current();
    if(!refined&&conditioning)refined=h3_prepared_find(&conditioning->prepared,1,text->tokens*HIDDEN);
    if(refined) dit->refined_text=h3_gpu_tensor_from_bf16(dit->gpu,refined->values,refined->elements);
    if(!dit->refined_text) {
        if (!report(progress,progress_opaque,"refine text",0,1,error,error_size)) goto failed;
        if(!refine_text(dit,text,error,error_size)) goto failed;
        if (!report(progress,progress_opaque,"refine text",1,1,error,error_size)) goto failed;
    } else H3_VERBOSE("h3cli: prepared refined text restored (raw BF16)\n");
    schedule_progress schedule_state = {progress, progress_opaque};
    double bridge_prepare_start = layout->bridge ? stream_now() : 0;
    if(cache_ok) dit->schedule=h3_dit_schedule_import(dit->gpu,resume);
    if(!dit->schedule&&conditioning&&conditioning->has_schedule) {
        h3_sampler_state cached={0};cached.sigmas=*sigmas;cached.layout=*layout;cached.prepared=conditioning->prepared;
        dit->schedule=h3_dit_schedule_import(dit->gpu,&cached);
        if(!dit->schedule){fail(error,error_size,"h3cond: invalid schedule tensor shapes");goto failed;}
    }
    if(dit->schedule) H3_VERBOSE("h3cli: prepared timestep/AdaLN tensors restored (raw BF16)\n");
    if(!dit->schedule) dit->schedule = layout->bridge ? h3_dit_schedule_precompute_bridge(
        dit->weights, dit->gpu, sigmas, dit->layout.bridge,
        schedule_report, &schedule_state, error, error_size) :
        (released_refvideo ? h3_dit_schedule_precompute_refvideo : h3_dit_schedule_precompute)(
        dit->weights, dit->gpu, sigmas,
        dit->video_condition_rows != 0 || layout->prefix.video_prefix_t != 0,
        dit->audio_condition_rows != 0 || layout->prefix.audio_prefix_t != 0,
        schedule_report, &schedule_state,
        error, error_size);
    if (dit->schedule) {
        if (layout->bridge && getenv("H3_PROFILE")) {
            h3_dit_schedule *hard = h3_dit_schedule_plan(sigmas, 1, 1, error, error_size);
            if (!hard) goto failed;
            uint32_t rows = h3_dit_schedule_time_rows(dit->schedule);
            uint32_t extra = rows - h3_dit_schedule_time_rows(hard);
            size_t row_bytes = (H3_DIT_BLOCKS*H3_DIT_MODALITIES*H3_DIT_ADALN_SLOTS*H3_DIT_HIDDEN+2*H3_DIT_HIDDEN)*2;
            H3_VERBOSE("h3cli: bridge modulation preparation %.6f s; %u time rows (%u extra), %zu extra AdaLN bytes, %zu class-plan bytes\n",
                stream_now()-bridge_prepare_start, rows, extra, (size_t)extra*row_bytes,
                2*sizeof(h3_bridge_profile)+(size_t)sigmas->steps*H3_TARGET_ROW_CLASSES*sizeof(uint32_t));
            h3_dit_schedule_free(hard);
        } else if ((layout->prefix.video_prefix_t || layout->prefix.audio_prefix_t) && getenv("H3_PROFILE")) {
            size_t extra = (size_t)(layout->prefix.video_prefix_t && !dit->video_condition_rows) +
                           (size_t)(layout->prefix.audio_prefix_t && !dit->audio_condition_rows);
            H3_VERBOSE("h3cli: continuation modulation: %zu extra time rows, %zu bytes AdaLN; prefix=%d/%d\n",extra,
                extra*(H3_DIT_BLOCKS*H3_DIT_MODALITIES*H3_DIT_ADALN_SLOTS*H3_DIT_HIDDEN+2*H3_DIT_HIDDEN)*2,
                layout->prefix.video_prefix_t,layout->prefix.audio_prefix_t);
        }
        configure_gate_ranked_blocks(dit);
        h3_dit_schedule_prune(dit->schedule, dit->block_active,
                              H3_DIT_BLOCKS);
    }
    if (!dit->schedule || !prepare_rope(dit, error, error_size) ||
        !prepare_maps(dit, text, error, error_size) ||
        !prepare_projection_maps(dit, error, error_size) ||
        !prepare_token_reduction_maps(dit, error, error_size)) goto failed;
#ifndef __APPLE__
    if(!dit->cuda_quant) {
        uint64_t mask=0;for(unsigned i=0;i<H3_DIT_BLOCKS;i++)if(dit->block_active[i])mask|=UINT64_C(1)<<i;
        uint64_t block_bytes=2*(4*(uint64_t)INNER*HIDDEN+3*(uint64_t)FFN*HIDDEN);
        uint64_t future=(uint64_t)dit->sequence*2*(12*(uint64_t)HIDDEN+8*(uint64_t)INNER+4*(uint64_t)FFN)+(UINT64_C(1)<<30);
        if(!h3_gpu_plan_bf16_weights(dit->gpu,mask,block_bytes,future,ssd_streaming,retry_cap,&dit->weight_plan)) {
            fail(error,error_size,"%s",h3_gpu_error(dit->gpu));goto failed;
        }
        dit->ssd_streaming=dit->weight_plan.resident_count<dit->weight_plan.active_count;
    }
#endif
    if (!load_core(dit, progress, progress_opaque, error, error_size) ||
        !allocate_activations(dit, error, error_size)) goto failed;
    if ((wanted_video_condition && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows,
             wanted_video_condition)) ||
        (wanted_audio_condition && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows,
             wanted_audio_condition))) {
        fail(error, error_size, "cannot write persistent DiT condition rows");
        goto failed;
    }
    h3_gpu_profile_mark(dit->gpu, "load");
    if (!report(progress, progress_opaque, "DiT initialization", 1, 1,
                 error, error_size)) goto failed;
    return dit;
failed:
    if ((dit->cuda_quant ? !ssd_streaming : dit->weight_plan.resident_count>0) && h3_gpu_should_retry_streaming(dit->gpu)) {
        h3_weight_options options=dit->weight_plan.options;
        if(dit->cuda_quant)options=h3_cuda_policy_current().weights;
        int next_cap=h3_weight_retry_cap(dit->weight_plan.resident_count);
        int force_stream=dit->cuda_quant?1:ssd_streaming;
        fprintf(stderr,"h3cli: automatic weight allocation failed: %s; release attempt, retry resident ceiling=%d%s\n",error,next_cap,dit->cuda_quant?" compressed-stream":"");
        h3_dit_free(dit);
        return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
            active_blocks, core_reuse_interval, token_reduction, force_stream, spatial_rope_scale,
            use_slower_bf16_mlp, use_slower_bf16_qkv, use_slower_bf16_attention_output,
            use_slower_row_major_attention_output, use_slower_unfused_int8_inputs,
            use_slower_unfused_qkv_rope, use_slower_scalar_qkv_rms,
            use_slower_uncached_int8_scales, use_slower_dynamic_fc1_k,
            use_slower_grouped_quantizer, use_int8_row_fc2,
            condition_video_rows, condition_video_elements, condition_audio_rows,
            condition_audio_elements, released_refvideo, progress, progress_opaque,
            resume, &options, next_cap, error, error_size);
    }
    h3_dit_free(dit);
    return NULL;
}

h3_dit *h3_dit_load_t2va(const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    NULL, 0, NULL, 0, 0, progress, progress_opaque,
                    NULL, NULL, -1, error, error_size);
}

h3_dit *h3_dit_load_conditioned(
                         const char *weight_directory,
                         const char *shader_source_path,
                         const h3_text_embedding *text,
                         const h3_layout *layout,
                         const h3_sigma_schedule *sigmas,
                         unsigned active_blocks,
                         unsigned core_reuse_interval,
                         int token_reduction,
                         int ssd_streaming,
                         float spatial_rope_scale,
                         int use_slower_bf16_mlp,
                         int use_slower_bf16_qkv,
                         int use_slower_bf16_attention_output,
                         int use_slower_row_major_attention_output,
                         int use_slower_unfused_int8_inputs,
                         int use_slower_unfused_qkv_rope,
                         int use_slower_scalar_qkv_rms,
                         int use_slower_uncached_int8_scales,
                         int use_slower_dynamic_fc1_k,
                         int use_slower_grouped_quantizer,
                         int use_int8_row_fc2,
                         const float *condition_video_rows,
                         size_t condition_video_elements,
                         const float *condition_audio_rows,
                         size_t condition_audio_elements,
                         int released_refvideo,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return load_dit(weight_directory, shader_source_path, text, layout, sigmas,
                    active_blocks, core_reuse_interval, token_reduction,
                    ssd_streaming,
                    spatial_rope_scale,
                    use_slower_bf16_mlp, use_slower_bf16_qkv,
                    use_slower_bf16_attention_output,
                    use_slower_row_major_attention_output,
                    use_slower_unfused_int8_inputs,
                    use_slower_unfused_qkv_rope,
                    use_slower_scalar_qkv_rms,
                    use_slower_uncached_int8_scales,
                    use_slower_dynamic_fc1_k,
                    use_slower_grouped_quantizer,
                    use_int8_row_fc2,
                    condition_video_rows, condition_video_elements,
                    condition_audio_rows, condition_audio_elements,
                    released_refvideo, progress, progress_opaque, NULL, NULL, -1, error, error_size);
}

h3_dit *h3_dit_load_resume(const char *weights,const char *shader,
    const h3_sampler_state *s,h3_dit_progress progress,void *opaque,char *error,size_t size) {
    const h3_params *p=&s->params;
    return load_dit(weights,shader,&s->text,&s->layout,&s->sigmas,
        (unsigned)p->dit_layers,(unsigned)p->core_reuse,p->token_reduction,p->ssd_streaming,s->spatial_rope_scale,
        p->use_slower_bf16_mlp,p->use_slower_bf16_qkv,p->use_slower_bf16_attention_output,
        p->use_slower_row_major_attention_output,p->use_slower_unfused_int8_inputs,p->use_slower_unfused_qkv_rope,
        p->use_slower_scalar_qkv_rms,p->use_slower_uncached_int8_scales,p->use_slower_dynamic_fc1_k,
        p->use_slower_grouped_quantizer,p->use_int8_row_fc2,
        s->condition_video,s->condition_video_elements,s->condition_audio,s->condition_audio_elements,
        s->refvideo_pipeline==H3_REFVIDEO_RELEASED_V1,progress,opaque,s,NULL,-1,error,size);
}

static int enter_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_pool_bf16(
            dit->gpu, dit->attention_output, dit->hidden, 0,
            original, dit->token_original_offset, dit->attention_output,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_pool_pairs, dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN),
            error, error_size, "snapshot and pool video tokens")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int enter_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_pool_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            dit->hidden, 0, original, dit->token_original_offset,
            dit->attention_output, dit->token_baseline_offset,
            dit->token_baseline_indices, dit->token_pool_pairs,
            weight->norm1, modulation, dit->reduced_row_maps[step],
            dit->sequence, dit->reduced_sequence, dit->token_baseline_rows,
            HIDDEN, SLOTS, 0, 1, 1e-5f), error, error_size,
            "snapshot, pool, and apply attention AdaLN")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = swap;
    dit->token_reduction_active = 1;
    return 1;
}

static int leave_token_reduction(h3_dit *dit, char *error,
                                 size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    if (!gpu_op(dit, h3_gpu_token_expand_delta_bf16(
            dit->gpu, dit->mod_attention, original,
            dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, dit->sequence,
            dit->reduced_sequence, dit->token_baseline_rows, HIDDEN,
            dit->video_target_start,
            dit->token_reduction_scale), error, error_size,
            "restore full video-token grid")) return 0;
    h3_gpu_tensor *swap = dit->hidden;
    dit->hidden = dit->mod_attention;
    dit->mod_attention = swap;
    dit->token_reduction_active = 0;
    return 1;
}

static int leave_token_reduction_adaln(h3_dit *dit, unsigned block,
                                       int step, char *error,
                                       size_t error_size) {
    h3_gpu_tensor *original = dit->token_original_in_qkv ?
        dit->qkv : dit->token_original;
    h3_dit_block *weight = &dit->blocks[block];
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(
        dit->schedule, block);
    if (!gpu_op(dit, h3_gpu_token_expand_adaln_bf16(
            dit->gpu, dit->attention_output, dit->mod_attention,
            original, dit->token_original_offset, dit->hidden, dit->hidden,
            dit->token_baseline_offset, dit->token_baseline_indices,
            dit->token_expand_parents, weight->norm1, modulation,
            dit->row_maps[step], dit->sequence, dit->reduced_sequence,
            dit->token_baseline_rows, HIDDEN, dit->video_target_start,
            dit->token_reduction_scale, SLOTS, 0, 1, 1e-5f),
            error, error_size, "restore tokens and apply attention AdaLN"))
        return 0;
    h3_gpu_tensor *reduced = dit->hidden;
    dit->hidden = dit->attention_output;
    dit->attention_output = reduced;
    dit->token_reduction_active = 0;
    return 1;
}

static int run_block(h3_dit *dit, unsigned index, int step,
                     h3_dit_block *weight,
                     int attention_adaln_ready,
                     int attention_input_quantized,
                     int fuse_next_attention, unsigned next_index,
                     int *next_attention_adaln_ready,
                     int *next_attention_input_quantized,
                     char *error, size_t error_size) {
    if(!h3_memory_check(0,"DiT block",error,error_size))return 0;
    h3_profile_memory(dit->gpu,"DiT block",step,index,0);
    h3_profile_span whole=h3_profile_component_begin(dit->gpu,"DiT complete block",index,step);
    h3_profile_span attention_region=h3_profile_component_begin(dit->gpu,"DiT complete attention region",index,step);
    if(whole.started<0||attention_region.started<0)return 0;
    const h3_gpu_tensor *modulation = h3_dit_schedule_block(dit->schedule,
                                                            index);
    h3_gpu_tensor *row_map = dit->token_reduction_active ?
        dit->reduced_row_maps[step] : dit->row_maps[step];
    h3_gpu_tensor *rope_cos = dit->token_reduction_active ?
        dit->reduced_rope_cos : dit->rope_cos;
    h3_gpu_tensor *rope_sin = dit->token_reduction_active ?
        dit->reduced_rope_sin : dit->rope_sin;
    uint32_t rows = dit->token_reduction_active ?
        dit->reduced_sequence : dit->sequence;
    if(index==0&&step==0&&getenv("H3_TEST_ATTENTION_CAPTURE_DIR")) {
        for(size_t i=0;i<dit->layout.segment_count;i++) {
            const h3_segment *segment=&dit->layout.segments[i];
            fprintf(stderr,"h3cli: attention validation segment kind=%d start=%zu stop=%zu\n",
                (int)segment->kind,segment->start,segment->stop);
        }
    }
#define OP(call, label) do {                                                    \
    h3_profile_span span=h3_profile_component_begin(dit->gpu,label,index,step); \
    int operation_ok=span.started>=0 && (call);                                \
    int profile_ok=h3_profile_component_end(dit->gpu,span,label,index,step);    \
    if (!gpu_op(dit, operation_ok && profile_ok, error, error_size, label)) return 0; \
} while (0)
#ifdef __APPLE__
    int diagnostic_ranges=getenv("H3_TEST_NATIVE_RANGES")!=NULL;
    if(diagnostic_ranges && (dit->int8_qkv || rows<=2048)) {
        fail(error,error_size,"range diagnostics require large-sequence BF16 QKV with materialized raw projection");return 0;
    }
#define RANGE(t,n,w,s,o,stage) do { if(diagnostic_ranges && !gpu_op(dit, \
    h3_gpu_diagnostic_range(dit->gpu,t,n,w,s,o,index,stage),error,error_size,"GPU range diagnostic"))return 0; } while(0)
#else
#define RANGE(t,n,w,s,o,stage) ((void)0)
#endif
    const char *capture_directory=getenv("H3_TEST_SGLANG_DIR");
    int capture=capture_directory && *capture_directory;
    if(capture) OP(sglang_boundary(dit,"input",dit->hidden,(size_t)rows*HIDDEN,step,index),"capture block input");
    if(capture) {
        OP(sglang_boundary(dit,"modulation",modulation,h3_gpu_tensor_elements(modulation),step,index),"capture modulation");
        OP(sglang_boundary(dit,"norm-weight",weight->norm1,HIDDEN,step,index),"capture norm weight");
        if(index==0 && step==0) {
            uint32_t *host=malloc((size_t)rows*sizeof(*host));
            int ok=host && h3_gpu_submit(dit->gpu) && h3_gpu_tensor_read_u32(row_map,host,rows) &&
                h3_sglang_dump("block-row-map.u32",host,(size_t)rows*sizeof(*host));
            free(host);OP(h3_gpu_begin(dit->gpu) && ok,"capture block row map");
        }
    }
    RANGE(dit->hidden,rows*HIDDEN,rows*HIDDEN,rows*HIDDEN,0,0);
    if (!attention_adaln_ready)
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->mod_attention, dit->hidden,
            weight->norm1, modulation, row_map, rows, HIDDEN, SLOTS,
            0, 1, 1e-5f), "DiT attention AdaLN");
    if(capture) OP(sglang_boundary(dit,"mod-attention",dit->mod_attention,(size_t)rows*HIDDEN,step,index),"capture attention AdaLN");
    if (dit->int8_qkv && !getenv("H3_DISABLE_INT8_QKV")) {
        OP(h3_gpu_grouped_qkv_linear_rope_int8(
            dit->gpu, dit->query, dit->key, dit->value,
            dit->int8_activation, dit->int8_activation_scales,
            dit->mod_attention, weight->qkv_int8, weight->qkv_scales,
            weight->q_norm, weight->k_norm, rope_cos, rope_sin,
            rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF, 1e-5f,
            attention_input_quantized,
            dit->use_slower_unfused_qkv_rope,
            dit->use_slower_scalar_qkv_rms,
            dit->use_slower_uncached_int8_scales),
           "DiT int8 QKV projection/norm/RoPE");
    }
#ifdef __APPLE__
    else if(dit->ane) {
        OP(h3_ane_split_linear(dit->ane,dit->gpu,dit->qkv,dit->mod_attention,weight->qkv,rows,index,step),"DiT GPU/ANE QKV");
        if(h3_backend_current().metal.layout_fusion)
            OP(h3_gpu_mixed_prepare(dit->gpu,dit->query,dit->key,dit->value,dit->qkv,weight->q_norm,weight->k_norm,
                rope_cos,rope_sin,rows,HEADS,HEAD_DIM,ROPE_HALF,1e-5f,1.0f/sqrtf((float)HEAD_DIM)),"DiT ANE QKV preparation");
        else
            OP(h3_gpu_grouped_qkv_rope_bf16(dit->gpu,dit->query,dit->key,dit->value,dit->qkv,weight->q_norm,weight->k_norm,
                rope_cos,rope_sin,rows,HEADS,HEAD_DIM,ROPE_HALF,1e-5f),"DiT ANE QKV norm/RoPE");
    }
#endif
    else {
        OP(h3_gpu_grouped_qkv_linear_rope_bf16(
            dit->gpu, dit->query, dit->key, dit->value, dit->qkv,
            dit->mod_attention, weight->qkv, weight->q_norm, weight->k_norm,
            rope_cos, rope_sin, rows, HIDDEN, HEADS, HEAD_DIM, ROPE_HALF,
            1e-5f), "DiT QKV projection/norm/RoPE");
    }
    if(capture) OP(sglang_boundary(dit,"qkv",dit->qkv,(size_t)rows*(3*INNER),step,index),"capture QKV");
    if(capture) OP(sglang_boundary(dit,"query",dit->query,(size_t)rows*INNER,step,index),"capture Q");
    if(capture) OP(sglang_boundary(dit,"key",dit->key,(size_t)rows*INNER,step,index),"capture K");
    if(capture) OP(sglang_boundary(dit,"value",dit->value,(size_t)rows*INNER,step,index),"capture V");
    RANGE(dit->qkv,rows*INNER,HEAD_DIM,HEAD_DIM*3,0,1);
    RANGE(dit->qkv,rows*INNER,HEAD_DIM,HEAD_DIM*3,HEAD_DIM,2);
    RANGE(dit->qkv,rows*INNER,HEAD_DIM,HEAD_DIM*3,HEAD_DIM*2,3);
    RANGE(dit->query,rows*INNER,rows*INNER,rows*INNER,0,4);
    RANGE(dit->key,rows*INNER,rows*INNER,rows*INNER,0,5);
    RANGE(dit->value,rows*INNER,rows*INNER,rows*INNER,0,6);
    int int8_attention_output = dit->int8_attention_out &&
        !getenv("H3_DISABLE_INT8_ATTENTION_OUT");
    int head_major_attention_output = int8_attention_output &&
        !dit->use_slower_row_major_attention_output &&
        !dit->use_slower_uncached_int8_scales &&
        !getenv("H3_DISABLE_HEAD_MAJOR_ATTENTION_OUTPUT");
    #ifdef __APPLE__
    OP(h3_gpu_native_noise(dit->gpu,step,dit->sigmas.video[step],dit->sigmas.audio[step]),"DiT attention noise policy");
    #else
    OP(h3_gpu_dit_attention_noise(dit->gpu,step,dit->sigmas.video[step],dit->sigmas.audio[step]),"DiT attention noise policy");
    #endif
    OP(h3_gpu_dit_sdpa_bf16(dit->gpu, dit->attention_heads,
        dit->query, dit->key, dit->value, rows, HEADS, HEAD_DIM,
        1.0f / sqrtf((float)HEAD_DIM), head_major_attention_output, index, step),
       "DiT full attention");
    if(capture) OP(sglang_boundary(dit,"attention",dit->attention_heads,(size_t)rows*INNER,step,index),"capture SDPA");
    RANGE(dit->attention_heads,rows*INNER,rows*INNER,rows*INNER,0,7);
    if (int8_attention_output) {
        if (head_major_attention_output)
            OP(h3_gpu_linear_int8_head_major_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, HEADS, HEAD_DIM,
                HIDDEN), "DiT head-major int8 attention output");
        else
            OP(h3_gpu_linear_int8_bf16(
                dit->gpu, dit->attention_output, dit->int8_activation,
                dit->int8_activation_scales, dit->attention_heads,
                weight->out_int8, weight->out_scales, rows, INNER, HIDDEN,
                dit->use_slower_uncached_int8_scales),
               "DiT int8 attention output");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->attention_output,
            dit->attention_heads, weight->out, NULL, rows, INNER, HIDDEN),
           "DiT attention output");
    }
    if(capture) OP(sglang_boundary(dit,"projection",dit->attention_output,(size_t)rows*HIDDEN,step,index),"capture projection");
    RANGE(dit->attention_output,rows*HIDDEN,rows*HIDDEN,rows*HIDDEN,0,8);
    if(!h3_profile_component_end(dit->gpu,attention_region,"DiT complete attention region",index,step))return 0;
    if(index==0 && step==0 && getenv("H3_TEST_NATIVE_BOUNDARIES") &&
       !debug_dit_tensor(dit,"attention-projection",dit->attention_output))return 0;
    int fused_int8_mlp_input = dit->int8_mlp &&
        !dit->use_slower_unfused_int8_inputs &&
        !getenv("H3_DISABLE_FUSED_INT8_MLP_INPUT") &&
        !getenv("H3_INT8_MLP_STAGE");
    if (fused_int8_mlp_input) {
        uint32_t padded_rows = (rows + 127u) & ~127u;
        OP(h3_gpu_gate_adaln_quantize_int8(
            dit->gpu, dit->hidden, dit->int8_activation,
            dit->int8_activation_scales, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map, rows, padded_rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate, MLP AdaLN and int8 quantization");
    } else if (!getenv("H3_DISABLE_FUSED_GATE_ADALN")) {
        OP(h3_gpu_gate_adaln_bf16(
            dit->gpu, dit->hidden, dit->mod_mlp, dit->hidden,
            dit->attention_output, weight->norm2, modulation, modulation,
            row_map,
            rows, HIDDEN, SLOTS, 2, 3, 4, 1e-5f),
           "DiT fused attention gate and MLP AdaLN");
    } else {
        OP(h3_gpu_gate_bf16(dit->gpu, dit->hidden, dit->hidden,
            dit->attention_output, modulation, row_map, rows, HIDDEN,
            SLOTS, 2), "DiT attention gate");
        OP(h3_gpu_adaln_bf16(
            dit->gpu, dit->mod_mlp, dit->hidden, weight->norm2,
            modulation, row_map, rows, HIDDEN, SLOTS, 3, 4, 1e-5f),
           "DiT MLP AdaLN");
    }
    h3_gpu_tensor *mlp_output = dit->activation_aliases ?
        dit->attention_output : dit->mlp_output;
    if(capture) OP(sglang_boundary(dit,"attention-residual",dit->hidden,(size_t)rows*HIDDEN,step,index),"capture gated attention");
    if(capture) OP(sglang_boundary(dit,"mod-mlp",dit->mod_mlp,(size_t)rows*HIDDEN,step,index),"capture MLP AdaLN");
    RANGE(dit->hidden,rows*HIDDEN,rows*HIDDEN,rows*HIDDEN,0,9);
    if (dit->int8_mlp &&
        (!getenv("H3_DISABLE_INT8_MLP") ||
         !weight->fc1 || !weight->fc2)) {
        OP(h3_gpu_mlp_int8_bf16(
            dit->gpu, mlp_output, dit->activated, dit->int8_activation,
            dit->int8_activation_scales, dit->mod_mlp,
            weight->fc1_int8, weight->fc1_scales,
            weight->fc2_int8, weight->fc2_scales,
            weight->fc1, weight->fc2,
            rows, HIDDEN, FFN, HIDDEN,
            dit->use_slower_grouped_quantizer,
            dit->use_slower_dynamic_fc1_k, dit->use_int8_row_fc2,
            fused_int8_mlp_input),
           "DiT int8 fused MLP");
    } else if (dit->nax_mlp && !getenv("H3_DISABLE_NAX_MLP")) {
        OP(h3_gpu_mlp_nax_bf16(dit->gpu, mlp_output, dit->activated,
            dit->mod_mlp, weight->fc1, weight->fc2, rows, HIDDEN, FFN,
            HIDDEN), "DiT NAX fused MLP");
    } else if (dit->fused_mlp) {
        OP(h3_gpu_mlp_bf16(dit->gpu, mlp_output, dit->mod_mlp,
            weight->fc1, weight->fc2, rows, HIDDEN, FFN, HIDDEN),
           "DiT fused MLP");
    } else {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->fc1, dit->mod_mlp, weight->fc1,
            NULL, rows, HIDDEN, FFN * 2), "DiT MLP input");
        OP(h3_gpu_swiglu_bf16(dit->gpu, dit->activated, dit->fc1, rows, FFN),
           "DiT SwiGLU");
        OP(h3_gpu_linear_bf16(dit->gpu, mlp_output, dit->activated,
            weight->fc2, NULL, rows, FFN, HIDDEN), "DiT MLP output");
    }
    RANGE(mlp_output,rows*HIDDEN,rows*HIDDEN,rows*HIDDEN,0,10);
    if (fuse_next_attention) {
        h3_dit_block *next_weight = &dit->blocks[next_index];
        const h3_gpu_tensor *next_modulation = h3_dit_schedule_block(
            dit->schedule, next_index);
        int fuse_int8_qkv_input = dit->int8_qkv &&
            !dit->use_slower_unfused_int8_inputs &&
            !getenv("H3_DISABLE_INT8_QKV") &&
            !getenv("H3_DISABLE_FUSED_INT8_QKV_INPUT");
        if (fuse_int8_qkv_input) {
            uint32_t padded_rows = (rows + 127u) & ~127u;
            OP(h3_gpu_gate_adaln_quantize_int8(
                dit->gpu, dit->hidden, dit->int8_activation,
                dit->int8_activation_scales, dit->hidden, mlp_output,
                next_weight->norm1, modulation, next_modulation, row_map,
                rows, padded_rows, HIDDEN, SLOTS, 5, 0, 1, 1e-5f),
               "DiT fused MLP gate, next attention AdaLN and int8 quantization");
            *next_attention_input_quantized = 1;
        } else {
            OP(h3_gpu_gate_adaln_bf16(
                dit->gpu, dit->hidden, dit->mod_attention, dit->hidden,
                mlp_output, next_weight->norm1, modulation,
                next_modulation, row_map, rows, HIDDEN, SLOTS, 5, 0, 1,
                1e-5f), "DiT fused MLP gate and next attention AdaLN");
            *next_attention_input_quantized = 0;
        }
        *next_attention_adaln_ready = 1;
    } else {
        OP(h3_gpu_gate_bf16(
            dit->gpu, dit->hidden, dit->hidden, mlp_output,
            modulation, row_map, rows, HIDDEN, SLOTS, 5), "DiT MLP gate");
    }
    if(capture) OP(sglang_boundary(dit,"output",dit->hidden,(size_t)rows*HIDDEN,step,index),"capture block output");
    if(capture && dit->sglang_reference && getenv("H3_TEST_SGLANG_BLOCK_ONLY")) {
        fail(error,error_size,"test-only first block capture complete; no sampler update executed");
        return 0;
    }
    RANGE(dit->hidden,rows*HIDDEN,rows*HIDDEN,rows*HIDDEN,0,11);
#undef RANGE
#undef OP
    return h3_profile_component_end(dit->gpu,whole,"DiT complete block",index,step);
}

/* Core residual reuse must not freeze the timestep-dependent final heads.
 * Audit the actual uploaded maps, including steps that skip all core blocks. */
static int audit_bridge_heads(h3_dit *dit, int step, int evaluated_core,
                              const h3_gpu_tensor *audio_map,
                              const h3_gpu_tensor *video_map,
                              char *error, size_t error_size) {
    if (!dit->layout.bridge || dit->core_reuse_interval == 1 ||
        !(getenv("H3_PROFILE") || getenv("H3_BRIDGE_DIAGNOSTICS"))) return 1;
    const uint32_t counts[] = {dit->video_rows, dit->audio_rows};
    const uint32_t starts[] = {dit->video_target_start, dit->audio_target_start};
    const h3_gpu_tensor *heads[] = {video_map, audio_map};
    uint32_t *blocks = malloc((size_t)dit->sequence * sizeof(*blocks));
    int ok = blocks && h3_gpu_tensor_read_u32(dit->row_maps[step], blocks, dit->sequence);
    for (int s = 0; s < 2 && ok; s++) {
        uint32_t *head = malloc((size_t)counts[s] * sizeof(*head));
        ok = head && h3_gpu_tensor_read_u32(heads[s], head, counts[s]);
        for (uint32_t r = 0; r < counts[s] && ok; r++) {
            h3_target_row_class kind = h3_bridge_row_class(&dit->bridge, s, r,
                dit->latent_h, dit->latent_w, dit->audio_t);
            uint32_t expected = h3_dit_schedule_class_row(dit->schedule, step, kind);
            ok = expected != UINT32_MAX && head[r] == expected &&
                 blocks[starts[s] + r] == expected * H3_DIT_MODALITIES + (s ? 2u : 0u);
        }
        free(head);
    }
    free(blocks);
    if (!ok) fail(error, error_size, "bridge core/head class audit failed at step %d", step);
    else {
        fprintf(stderr, "h3cli: bridge core audit step=%d evaluated=%d heads=current classes=",
            step, evaluated_core);
        for (int c = 0; c < H3_TARGET_ROW_CLASSES; c++) if (dit->bridge.active[c])
            fprintf(stderr, " %d:%u", c, h3_dit_schedule_class_row(dit->schedule,
                step, (h3_target_row_class)c));
        fputc('\n', stderr);
    }
    return ok;
}

static int encode_forward(h3_dit *dit, int step, int begin, int submit,
                          int disable_command_split, char *error,
                          size_t error_size) {
#define OP(call, label) do {                                                    \
    int control=strstr(label,"begin")||strstr(label,"submit")||strstr(label,"continue"); \
    h3_profile_span span={0};                                                 \
    if(!control)span=h3_profile_component_begin(dit->gpu,label,H3_DIT_BLOCKS,step); \
    if(span.started<0)return gpu_op(dit,0,error,error_size,label);               \
    int call_ok=(call);                                                       \
    int profile_ok=control?1:h3_profile_component_end(dit->gpu,span,label,H3_DIT_BLOCKS,step); \
    if(!gpu_op(dit,call_ok&&profile_ok,error,error_size,label))return 0;          \
} while (0)
    dit->last_executed_blocks=0;
    if (begin) OP(h3_gpu_begin(dit->gpu), "begin DiT forward");
    size_t video_offset = 0;
    size_t audio_offset = 0;
    if (dit->fused_patch_pack) {
        if (dit->video_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->video_input, dit->video_patch_w,
                dit->video_patch_b, dit->video_projection_map, dit->sequence,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "project mapped video sources");
        if (dit->audio_projection_map)
            OP(h3_gpu_patch_linear_bf16_map(
                dit->gpu, dit->hidden, dit->audio_input, dit->audio_patch_w,
                dit->audio_patch_b, dit->audio_projection_map, dit->sequence,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "project mapped audio sources");
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                if (!dit->video_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->video_input,
                        video_offset * VIDEO_PATCH, dit->video_patch_w,
                        dit->video_patch_b, (uint32_t)segment_rows,
                        VIDEO_PATCH, HIDDEN), "project packed video source");
                video_offset += segment_rows;
            } else {
                if (!dit->audio_projection_map)
                    OP(h3_gpu_patch_linear_bf16_offset(
                        dit->gpu, dit->hidden, destination, dit->audio_input,
                        audio_offset * AUDIO_CHANNELS, dit->audio_patch_w,
                        dit->audio_patch_b, (uint32_t)segment_rows,
                        AUDIO_CHANNELS, HIDDEN),
                       "project packed audio source");
                audio_offset += segment_rows;
            }
        }
    } else {
        if (dit->fused_patch_projection) {
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->video_projected, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "fused video patch projection");
            OP(h3_gpu_patch_linear_bf16(
                dit->gpu, dit->audio_projected, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "fused audio patch projection");
        } else {
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->video_projected_f32, dit->video_input,
                dit->video_patch_w, dit->video_patch_b,
                dit->video_total_rows, VIDEO_PATCH, HIDDEN),
               "video patch projection");
            OP(h3_gpu_linear_f32(
                dit->gpu, dit->audio_projected_f32, dit->audio_input,
                dit->audio_patch_w, dit->audio_patch_b,
                dit->audio_total_rows, AUDIO_CHANNELS, HIDDEN),
               "audio patch projection");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->video_projected, dit->video_projected_f32,
                dit->video_total_rows * HIDDEN), "video BF16 cast");
            OP(h3_gpu_cast_f32_to_bf16(
                dit->gpu, dit->audio_projected, dit->audio_projected_f32,
                dit->audio_total_rows * HIDDEN), "audio BF16 cast");
        }
        for (size_t index = 0; index < dit->layout.segment_count; index++) {
            const h3_segment *segment = &dit->layout.segments[index];
            size_t segment_rows = segment->stop - segment->start;
            size_t destination = segment->start * HIDDEN;
            if (segment->kind == H3_SEG_TEXT) {
                OP(h3_gpu_copy_bf16(dit->gpu, dit->hidden, destination,
                    dit->refined_text, 0, segment_rows * HIDDEN),
                   "pack refined text");
            } else if (segment->kind == H3_SEG_COND ||
                       segment->kind == H3_SEG_REF_IMAGE ||
                       segment->kind == H3_SEG_VIDEO) {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->video_projected,
                    video_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack video source");
                video_offset += segment_rows;
            } else {
                OP(h3_gpu_copy_bf16(
                    dit->gpu, dit->hidden, destination, dit->audio_projected,
                    audio_offset * HIDDEN, segment_rows * HIDDEN),
                   "pack audio source");
                audio_offset += segment_rows;
            }
        }
    }
    if (video_offset != dit->video_total_rows ||
        audio_offset != dit->audio_total_rows) {
        fail(error, error_size, "DiT segment packing did not consume row sources");
        return 0;
    }
    int evaluate_core = dit->core_reuse_interval == 1 ||
        !dit->core_residual_ready ||
        dit->core_forward_count % dit->core_reuse_interval == 0 ||
        step == h3_dit_schedule_steps(dit->schedule) - 1;
    int use_token_reduction = evaluate_core && dit->token_reduction &&
        !getenv("H3_DISABLE_TOKEN_REDUCTION");
    unsigned token_reduction_end =
        dit->token_reduction_early_steps &&
        (unsigned)step < dit->token_reduction_early_steps ?
            dit->token_reduction_early_end : dit->token_reduction_end;
    uint32_t hidden_elements = dit->sequence * HIDDEN;
    if(dit->adaptive_mode) {
        if(dit->adaptive_pending){fail(error,error_size,"adaptive forward requires commit or reset after prior execution");return 0;}
        dit->adaptive_pending=1;
        dit->adaptive_phase=dit->captured_attention==H3_ATTENTION_SUBBLOCK && step>=dit->subblock_warmup;
        OP(h3_gpu_copy_bf16(dit->gpu,dit->adaptive_scratch,0,dit->hidden,0,hidden_elements),"save adaptive probe input");
    }
    if (evaluate_core && dit->core_reuse_interval > 1)
        OP(h3_gpu_copy_bf16(dit->gpu, dit->core_input, 0, dit->hidden, 0,
                            hidden_elements), "save DiT core input");
    if (evaluate_core) {
        unsigned command_blocks = disable_command_split
            ? 0 : command_block_interval(dit);
        if (dit->ssd_streaming) command_blocks = 0;
        unsigned completed_blocks = 0;
        int carried_attention_adaln = 0;
        int carried_attention_input_quantized = 0;
        for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
            int fused_token_adaln = carried_attention_adaln;
            int fused_attention_input_quantized =
                carried_attention_input_quantized;
            carried_attention_adaln = 0;
            carried_attention_input_quantized = 0;
            if (use_token_reduction &&
                block == dit->token_reduction_begin) {
                fused_token_adaln = dit->block_active[block] &&
                    !getenv("H3_DISABLE_FUSED_TOKEN_POOL_ADALN");
                if (fused_token_adaln) {
                    if (!enter_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!enter_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (use_token_reduction && block == token_reduction_end) {
                fused_token_adaln = dit->block_active[block] &&
                    !getenv("H3_DISABLE_FUSED_TOKEN_ADALN");
                if (fused_token_adaln) {
                    if (!leave_token_reduction_adaln(
                            dit, block, step, error, error_size)) return 0;
                    fused_attention_input_quantized = 0;
                } else if (!leave_token_reduction(
                               dit, error, error_size)) return 0;
            }
            if (!dit->block_active[block]) continue;
            unsigned next_block = block + 1;
            int next_is_token_boundary = use_token_reduction &&
                (next_block == dit->token_reduction_begin ||
                 next_block == token_reduction_end);
            int fuse_next_attention =
                !getenv("H3_DISABLE_FUSED_CROSS_BLOCK_ADALN") &&
                next_block < H3_DIT_BLOCKS &&
                dit->block_active[next_block] && !next_is_token_boundary &&
                !(dit->adaptive_mode && block==0);
            h3_dit_block streamed_weight;
            h3_dit_block *weight = &dit->blocks[block];
            h3_dit_stream_job stream_job={.dit=dit,.layer=H3_DIT_BLOCKS,.slot=0};
            pthread_t stream_thread;
            int stream_started = 0;
            int block_streamed=dit->ssd_streaming && !(dit->weight_plan.resident_mask & (UINT64_C(1)<<block));
#ifndef __APPLE__
            /* A previous evaluation may have completed/paused, or skipped the
             * suffix. Prime only on demand; resident block 0 can hide this copy. */
            if(dit->ssd_streaming && dit->stream_ready_layer==H3_DIT_BLOCKS &&
               !(dit->adaptive_mode && block==0 && !block_streamed)) {
                stream_job.layer=first_active_block(dit);
                stream_job.slot=dit->stream_ready_slot^1u;
                if(block_streamed) {
                    if(!read_stream_layer(&stream_job)){fail(error,error_size,"cannot prime DiT stream: %s",stream_job.error);return 0;}
                    dit->stream_bytes+=stream_job.bytes;dit->stream_read_layers++;dit->stream_read_seconds+=stream_job.seconds;
                    dit->stream_ready_layer=stream_job.layer;dit->stream_ready_slot=stream_job.slot;
                } else {
                    int err=pthread_create(&stream_thread,NULL,read_stream_layer_thread,&stream_job);
                    if(err){fail(error,error_size,"cannot prime partial DiT stream: %s",strerror(err));return 0;}
                    stream_started=1;
                }
            }
#endif
            if (block_streamed) {
                if (dit->stream_ready_layer != block ||
                    dit->stream_ready_slot > 1) {
                    fail(error, error_size,
                         "DiT SSD stream expected block %u, has block %u",
                         block, dit->stream_ready_layer);
                    return 0;
                }
                h3_dit_block *slot =
                    &dit->stream_slots[dit->stream_ready_slot];
                streamed_weight = dit->blocks[block];
                streamed_weight.qkv = slot->qkv;
                streamed_weight.out = slot->out;
                streamed_weight.fc1 = slot->fc1;
                streamed_weight.fc2 = slot->fc2;
                weight = &streamed_weight;

                unsigned future = next_active_block(dit, block);
#ifdef __APPLE__
                if (future == H3_DIT_BLOCKS)future = first_active_block(dit);
#endif
                stream_job = (h3_dit_stream_job){
                    .dit = dit,
                    .layer = future,
                    .slot = dit->stream_ready_slot ^ 1u
                };
                int defer_prefetch=(dit->adaptive_mode && block==0) || future==H3_DIT_BLOCKS;
                int thread_error = defer_prefetch?0:pthread_create(
                    &stream_thread, NULL, read_stream_layer_thread,
                    &stream_job);
                if (thread_error) {
                    fail(error, error_size,
                         "cannot start DiT SSD prefetch for block %u: %s",
                         future, strerror(thread_error));
                    return 0;
                }
                stream_started = !defer_prefetch;
            }
            int block_ok = run_block(
                dit, block, step, weight, fused_token_adaln,
                fused_attention_input_quantized,
                fuse_next_attention, next_block,
                &carried_attention_adaln,
                &carried_attention_input_quantized,
                error, error_size);
            if (!block_ok) {
                if (stream_started) (void)pthread_join(stream_thread, NULL);
                return 0;
            }
            dit->last_executed_blocks++;
#ifndef __APPLE__
            if(block_streamed)dit->stream_ready_layer=H3_DIT_BLOCKS;
#endif
            if (getenv("H3_DEBUG_DIT_DIR")&&debug_dit_step(step)) {
                char label[64];
                if(getenv("H3_DEBUG_DIT_STEPS"))snprintf(label,sizeof(label),"step-%03d-block-%02u",step,block);
                else snprintf(label,sizeof(label),"block-%02u",block);
                int dumped=debug_dit_tensor(dit,label,dit->hidden);
                if(!getenv("H3_DEBUG_DIT_STEPS")){
                    snprintf(label,sizeof(label),"modulation-%02u",block);
                    dumped=dumped && debug_dit_tensor(dit,label,h3_dit_schedule_block(dit->schedule,block));
                }
                if(!dumped) {if(stream_started)(void)pthread_join(stream_thread,NULL);fail(error,error_size,"cannot dump DiT diagnostics");return 0;}
            }
            if(dit->adaptive_mode && block==0) {
#ifndef __APPLE__
                if(dit->adaptive_recipe==H3_ADAPTIVE_CONTINUATION_VERSION) {
                    OP(h3_gpu_adaptive_continuation_probe(dit->gpu,dit->adaptive_scratch,dit->hidden,
                        dit->adaptive_anchor,dit->adaptive_reduction,HIDDEN,&dit->adaptive_regions,
                        (int)dit->adaptive_history.ready,dit->adaptive_scores),"adaptive continuation class scores");
                } else OP(h3_gpu_adaptive_probe(dit->gpu,dit->adaptive_scratch,dit->hidden,
                    dit->adaptive_anchor,dit->adaptive_reduction,dit->sequence,HIDDEN,
                    dit->video_target_start,dit->video_rows,dit->audio_target_start,dit->audio_rows,
                    (int)dit->adaptive_history.ready,dit->adaptive_scores),"adaptive probe score");
#else
                fail(error,error_size,"adaptive cache requires CUDA");return 0;
#endif
                if(!(dit->adaptive_recipe==H3_ADAPTIVE_CONTINUATION_VERSION?
                    h3_adaptive_regions_score(&dit->adaptive_regions,dit->adaptive_scores,&dit->adaptive_score):
                    h3_adaptive_score(dit->adaptive_recipe,dit->adaptive_scores,&dit->adaptive_score))){
                    fail(error,error_size,"invalid adaptive component score");return 0;
                }
                int refresh=h3_adaptive_decide(dit->adaptive_mode,&dit->adaptive_history,step,
                    dit->sigmas.steps,dit->adaptive_warmup,dit->adaptive_phase,dit->adaptive_score,dit->adaptive_threshold,dit->adaptive_max_hits,&dit->adaptive_reason);
                if(refresh<0){fail(error,error_size,"invalid adaptive cache decision");return 0;}
                dit->adaptive_refresh=refresh;
                if(!refresh) {
                    OP(h3_gpu_add_bf16(dit->gpu,dit->hidden,dit->hidden,dit->adaptive_delta,hidden_elements),"adaptive suffix hit");
                    break;
                }
                OP(h3_gpu_copy_bf16(dit->gpu,dit->adaptive_anchor,0,dit->adaptive_scratch,0,hidden_elements),"stage adaptive anchor");
                OP(h3_gpu_copy_bf16(dit->gpu,dit->adaptive_scratch,0,dit->hidden,0,hidden_elements),"save adaptive suffix input");
                if(dit->ssd_streaming && !stream_started) {
                    if(!block_streamed && dit->stream_ready_layer==H3_DIT_BLOCKS)
                        stream_job=(h3_dit_stream_job){.dit=dit,.layer=first_active_block(dit),.slot=dit->stream_ready_slot^1u};
                    int err=stream_job.layer==H3_DIT_BLOCKS?0:pthread_create(&stream_thread,NULL,read_stream_layer_thread,&stream_job);
                    if(err){fail(error,error_size,"cannot start adaptive suffix prefetch: %s",strerror(err));return 0;}
                    stream_started=stream_job.layer!=H3_DIT_BLOCKS;
                }
            }
            completed_blocks++;
            if (command_blocks &&
                completed_blocks < dit->active_block_count &&
                completed_blocks % command_blocks == 0)
                OP(h3_gpu_continue(dit->gpu), "continue DiT command chain");
            if (stream_started) {
#ifdef __APPLE__
                int gpu_ok = gpu_op(dit,h3_gpu_submit(dit->gpu),error,error_size,"submit streamed DiT block");
#else
                int gpu_ok = gpu_op(dit,h3_gpu_continue(dit->gpu),error,error_size,"continue streamed DiT block");
#endif
                double wait_started = stream_now();
                int join_error = pthread_join(stream_thread, NULL);
                dit->stream_wait_seconds += stream_now() - wait_started;
                if (!gpu_ok) return 0;
                if (join_error) {
                    fail(error, error_size,
                         "cannot join DiT SSD prefetch: %s",
                         strerror(join_error));
                    return 0;
                }
                dit->stream_bytes += stream_job.bytes;dit->stream_read_layers++;
                dit->stream_read_seconds += stream_job.seconds;
                if (!stream_job.ok) {
                    fail(error, error_size,
                         "cannot stream DiT block %u: %s",
                         stream_job.layer, stream_job.error);
                    return 0;
                }
                dit->stream_ready_layer = stream_job.layer;
                dit->stream_ready_slot = stream_job.slot;
#ifdef __APPLE__
                OP(h3_gpu_begin(dit->gpu),"continue after streamed DiT block");
#endif
            }
        }
        if(dit->adaptive_mode && dit->adaptive_refresh)
            OP(h3_gpu_sub_bf16(dit->gpu,dit->adaptive_delta,dit->hidden,dit->adaptive_scratch,hidden_elements),"stage adaptive suffix delta");
        if (use_token_reduction &&
            token_reduction_end == H3_DIT_BLOCKS &&
            !leave_token_reduction(dit, error, error_size)) return 0;
        if (dit->core_reuse_interval > 1) {
            OP(h3_gpu_sub_bf16(dit->gpu, dit->core_residual, dit->hidden,
                               dit->core_input, hidden_elements),
               "cache DiT core residual");
            dit->core_residual_ready = 1;
        }
    } else {
        OP(h3_gpu_add_bf16(dit->gpu, dit->hidden, dit->hidden,
                           dit->core_residual, hidden_elements),
           "reuse DiT core residual");
    }
    dit->core_forward_count++;
    if(dit->adaptive_mode)dit->adaptive_pending=1;
    const h3_gpu_tensor *audio_map = dit->final_audio_maps[step];
    const h3_gpu_tensor *video_map = dit->final_video_maps[step];
    if (!audit_bridge_heads(dit, step, evaluate_core, audio_map, video_map,
                             error, error_size)) return 0;
    const h3_gpu_tensor *final = h3_dit_schedule_final(dit->schedule);
    int fused_final_head = dit->bf16_final &&
        !getenv("H3_DISABLE_FUSED_FINAL_HEAD") &&
        !getenv("H3_DISABLE_FUSED_FINAL_SLICE");
    if (fused_final_head) {
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->audio_output_bf16, dit->final_audio_inverse,
            dit->hidden, (size_t)dit->audio_target_start * HIDDEN,
            dit->final_norm, final, audio_map,
            dit->final_audio_w, dit->final_audio_b, dit->audio_rows, HIDDEN,
            AUDIO_CHANNELS, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final audio AdaLN/head");
        OP(h3_gpu_adaln_linear_bf16(
            dit->gpu, dit->video_output_bf16, dit->final_video_inverse,
            dit->hidden, (size_t)dit->video_target_start * HIDDEN,
            dit->final_norm, final, video_map,
            dit->final_video_w, dit->final_video_b, dit->video_rows, HIDDEN,
            VIDEO_PATCH, FINAL_SLOTS, 0, 1, 1e-5f),
           "fused final video AdaLN/head");
    } else if (getenv("H3_DISABLE_FUSED_FINAL_SLICE")) {
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_audio_input, 0, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN,
            (size_t)dit->audio_rows * HIDDEN), "slice final audio");
        OP(h3_gpu_copy_bf16(dit->gpu, dit->final_video_input, 0, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN,
            (size_t)dit->video_rows * HIDDEN), "slice final video");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_audio_norm,
            dit->final_audio_input, dit->final_norm, final,
            audio_map, dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final audio AdaLN");
        OP(h3_gpu_adaln_bf16(dit->gpu, dit->final_video_norm,
            dit->final_video_input, dit->final_norm, final,
            video_map, dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "final video AdaLN");
    } else {
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_audio_norm, dit->hidden,
            (size_t)dit->audio_target_start * HIDDEN, dit->final_norm, final,
            audio_map, dit->audio_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final audio slice/AdaLN");
        OP(h3_gpu_adaln_bf16_offset(
            dit->gpu, dit->final_video_norm, dit->hidden,
            (size_t)dit->video_target_start * HIDDEN, dit->final_norm, final,
            video_map, dit->video_rows, HIDDEN, FINAL_SLOTS,
            0, 1, 1e-5f), "fused final video slice/AdaLN");
    }
    if (dit->bf16_final && !fused_final_head) {
        OP(h3_gpu_linear_bf16(dit->gpu, dit->audio_output_bf16,
            dit->final_audio_norm, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, HIDDEN, AUDIO_CHANNELS),
           "BF16 final audio head");
        OP(h3_gpu_linear_bf16(dit->gpu, dit->video_output_bf16,
            dit->final_video_norm, dit->final_video_w, dit->final_video_b,
            dit->video_rows, HIDDEN, VIDEO_PATCH),
           "BF16 final video head");
    } else if (!dit->bf16_final) {
        OP(sglang_boundary(dit,"final-audio-norm",dit->final_audio_norm,
            (size_t)dit->audio_rows*HIDDEN,step,0),"capture final audio norm");
        OP(sglang_boundary(dit,"final-video-norm",dit->final_video_norm,
            (size_t)dit->video_rows*HIDDEN,step,0),"capture final video norm");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_audio_f32,
            dit->final_audio_norm, dit->audio_rows * HIDDEN),
           "final audio F32 cast");
        OP(h3_gpu_cast_bf16_to_f32(dit->gpu, dit->final_video_f32,
            dit->final_video_norm, dit->video_rows * HIDDEN),
           "final video F32 cast");
        OP(final_f32_head(dit, dit->audio_output,
            dit->final_audio_f32, dit->final_audio_w, dit->final_audio_b,
            dit->audio_rows, AUDIO_CHANNELS), "final audio head");
        OP(final_f32_head(dit, dit->video_output,
            dit->final_video_f32, dit->final_video_w, dit->final_video_b,
            dit->video_rows, VIDEO_PATCH), "final video head");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->audio_output_bf16,
            dit->audio_output, dit->audio_rows * AUDIO_CHANNELS),
           "final audio output cast");
        OP(h3_gpu_cast_f32_to_bf16(dit->gpu, dit->video_output_bf16,
            dit->video_output, dit->video_rows * VIDEO_PATCH),
           "final video output cast");
    }
    int mixed=0,diagnostics=0;
#ifdef __APPLE__
    h3_backend_scope scope=h3_backend_current();
    mixed=scope.backend==H3_BACKEND_METAL&&scope.metal.precision;
    diagnostics=getenv("H3_TEST_NATIVE_RANGES")!=NULL;
#endif
    if (submit||mixed||diagnostics) OP(h3_gpu_submit(dit->gpu), "submit DiT forward");
#ifdef __APPLE__
    // Reject the complete forward before velocity can update CPU or GPU state.
    // CPU Euler already submits here; GPU Euler requires this explicit boundary.
    if(mixed) {
        OP(h3_gpu_mixed_step_finish(dit->gpu,step),"submit validation boundary for mixed attention");
    }
    if(diagnostics)OP(h3_gpu_diagnostic_finish(dit->gpu,step),"submit validation boundary for range diagnostics");
    if(!submit&&(mixed||diagnostics))OP(h3_gpu_begin(dit->gpu),"continue after attention diagnostics");
#endif
#undef OP
    return 1;
}

size_t h3_dit_video_elements(const h3_dit *dit) {
    return dit ? (size_t)VIDEO_CHANNELS * (size_t)dit->latent_t *
        (size_t)dit->latent_h * (size_t)dit->latent_w : 0;
}

size_t h3_dit_audio_elements(const h3_dit *dit) {
    return dit ? (size_t)AUDIO_CHANNELS * AUDIO_STREAMS *
        (size_t)dit->audio_t : 0;
}

int h3_dit_reset_run(h3_dit *dit,
                     const float *condition_video_rows,
                     size_t condition_video_elements,
                     const float *condition_audio_rows,
                     size_t condition_audio_elements,
                     char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit) {
        fail(error, error_size, "prepared DiT is absent");
        return 0;
    }
    size_t wanted_video =
        (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t wanted_audio =
        (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (condition_video_elements != wanted_video ||
        condition_audio_elements != wanted_audio ||
        (wanted_video && !condition_video_rows) ||
        (wanted_audio && !condition_audio_rows)) {
        fail(error, error_size, "prepared DiT condition rows do not match");
        return 0;
    }
    if ((wanted_video && !h3_gpu_tensor_write_f32_range(
             dit->video_input, 0, condition_video_rows, wanted_video)) ||
        (wanted_audio && !h3_gpu_tensor_write_f32_range(
             dit->audio_input, 0, condition_audio_rows, wanted_audio))) {
        fail(error, error_size, "cannot refresh prepared DiT conditions");
        return 0;
    }
    dit->core_forward_count = 0;
    dit->core_residual_ready = 0;
    if(!h3_adaptive_plan_admit(&dit->adaptive_plan,h3_cuda_policy_current().adaptive_cache_max_bytes,error,error_size))return 0;
    dit->adaptive_budget=h3_adaptive_budget(h3_cuda_policy_current().adaptive_cache_max_bytes);
    memset(&dit->adaptive_history,0,sizeof(dit->adaptive_history));dit->adaptive_pending=0;
    return 1;
}

int h3_dit_forward(h3_dit *dit, int step,
                   const float *video_latent, const float *audio_latent,
                   float *video_velocity, float *audio_velocity,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || step < 0 || step >= h3_dit_schedule_steps(dit->schedule) ||
        !video_latent || !audio_latent || !video_velocity || !audio_velocity) {
        fail(error, error_size, "invalid DiT forward arguments");
        return 0;
    }
    size_t video_row_elements = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_row_elements = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    float *video_rows = malloc(video_row_elements * sizeof(*video_rows));
    float *audio_rows = malloc(audio_row_elements * sizeof(*audio_rows));
    uint16_t *video_out = malloc(video_row_elements * sizeof(*video_out));
    uint16_t *audio_out = malloc(audio_row_elements * sizeof(*audio_out));
    float *video_f32 = malloc(video_row_elements * sizeof(*video_f32));
    float *audio_f32 = malloc(audio_row_elements * sizeof(*audio_f32));
    if (!video_rows || !audio_rows || !video_out || !audio_out ||
        !video_f32 || !audio_f32) {
        fail(error, error_size, "out of memory packing DiT latents");
        free(video_rows); free(audio_rows); free(video_out); free(audio_out);
        free(video_f32); free(audio_f32);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows,
        video_row_elements) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->video_input,
            (size_t)dit->video_condition_rows * VIDEO_PATCH,
            video_rows, video_row_elements) &&
        h3_gpu_tensor_write_f32_range(
            dit->audio_input,
            (size_t)dit->audio_condition_rows * AUDIO_CHANNELS,
            audio_rows, audio_row_elements);
    if (!ok) fail(error, error_size, "cannot pack/write DiT input latents");
    if (ok) ok = encode_forward(dit, step, 1, 1, 0, error, error_size);
    if(ok && dit->sglang_reference)
        ok=h3_gpu_tensor_read_f32(dit->video_output,video_f32,video_row_elements) &&
           h3_gpu_tensor_read_f32(dit->audio_output,audio_f32,audio_row_elements);
    else if (ok) ok = h3_gpu_tensor_read_bf16(dit->video_output_bf16, video_out,
                                         video_row_elements) &&
                 h3_gpu_tensor_read_bf16(dit->audio_output_bf16, audio_out,
                                         audio_row_elements);
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot read DiT output");
    if (ok && !dit->sglang_reference) {
        for (size_t index = 0; index < video_row_elements; index++) {
            uint32_t bits = (uint32_t)video_out[index] << 16;
            memcpy(&video_f32[index], &bits, sizeof(bits));
        }
        for (size_t index = 0; index < audio_row_elements; index++) {
            uint32_t bits = (uint32_t)audio_out[index] << 16;
            memcpy(&audio_f32[index], &bits, sizeof(bits));
        }
    }
    if (ok) ok = h3_dit_unpatchify_video(video_f32, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_velocity,
        h3_dit_video_elements(dit)) &&
        h3_dit_unpack_audio(audio_f32, AUDIO_CHANNELS, dit->audio_t,
                            audio_velocity, h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error)) fail(error, error_size, "cannot unpack DiT output");
    /* Bridge Euler consumes raw predictions and scales once at application.
     * Keep the original idempotent double-mask behavior only for hard mode. */
    if (ok && !dit->layout.bridge && (dit->layout.prefix.video_prefix_t || dit->layout.prefix.audio_prefix_t))
        h3_prefix_mask_velocity(dit->layout.prefix, dit->latent_t,
            dit->latent_h, dit->latent_w, dit->audio_t, video_velocity, audio_velocity);
    free(video_rows); free(audio_rows); free(video_out); free(audio_out);
    free(video_f32); free(audio_f32);
    return ok;
}

int h3_dit_placement_compatible(h3_dit *dit,int explicit_ssd) {
    if(!dit||dit->explicit_ssd_streaming!=explicit_ssd)return 0;
#ifdef __APPLE__
    return 1;
#else
    return h3_gpu_weight_plan_compatible(dit->gpu,explicit_ssd);
#endif
}
int h3_dit_get_gpu_stats(const h3_dit *dit, h3_gpu_stats *stats) {
    return dit && h3_gpu_get_stats(dit->gpu, stats);
}

int h3_dit_release_weight_cache(h3_dit *dit, char *error, size_t error_size) {
    return dit && gpu_op(dit, h3_gpu_release_weight_cache(dit->gpu),
                         error, error_size, "release optional streamed weights");
}

static float extrapolation_ratio(float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) return 0.0f;
    float denominator = last_sigma - previous_sigma;
    float ratio = denominator != 0.0f
        ? (current_sigma - last_sigma) / denominator : 0.0f;
    /* Reuse intervals are deliberately small. This guard prevents malformed
     * custom schedules from turning one cached evaluation into an explosion. */
    if (ratio < -2.0f) ratio = -2.0f;
    if (ratio > 2.0f) ratio = 2.0f;
    return ratio;
}

static void extrapolate_velocity(float *output, const float *last,
                                 const float *previous, size_t count,
                                 float current_sigma, float last_sigma,
                                 float previous_sigma, int have_previous) {
    if (!have_previous) {
        memcpy(output, last, count * sizeof(*output));
        return;
    }
    float ratio = extrapolation_ratio(current_sigma, last_sigma,
                                      previous_sigma, have_previous);
    for (size_t index = 0; index < count; index++)
        output[index] = last[index] +
                        ratio * (last[index] - previous[index]);
}

int h3_dit_reuse_schedule(int steps, int reuse_interval, uint8_t *selected,
                          size_t selected_count) {
    if (steps < 1 || reuse_interval < 1 || reuse_interval > 32 || !selected ||
        selected_count < (size_t)steps) return -1;
    memset(selected, 0, (size_t)steps);

    int count = 0;
    for (int step = 0; step < steps; step++) {
        if (reuse_interval == 1 || step == 0 || step == steps - 1 ||
            step % reuse_interval == 0) {
            selected[step] = 1;
            count++;
        }
    }
    return count;
}

static int parse_reuse_steps(int steps, uint8_t *selected) {
    const char *text = getenv("H3_REUSE_STEPS");
    if (!text || !*text) return 0;
    memset(selected, 0, (size_t)steps);
    int count = 0;
    int previous = -1;
    while (*text) {
        char *end = NULL;
        long value = strtol(text, &end, 10);
        if (end == text || value < 0 || value >= steps ||
            value <= previous) return -1;
        selected[value] = 1;
        previous = (int)value;
        count++;
        if (!*end) break;
        if (*end != ',') return -1;
        text = end + 1;
        if (!*text) return -1;
    }
    return selected[0] && selected[steps - 1] ? count : -1;
}

int h3_dit_resolve_reuse(int steps,int reuse_interval,uint8_t *selected) {
    int n=h3_dit_reuse_schedule(steps,reuse_interval,selected,H3_MAX_STEPS);
    if(n<0) return -1;
    int custom=reuse_interval>1?parse_reuse_steps(steps,selected):0;
    return custom<0?-1:custom?custom:n;
}

static int gpu_sampler_requested(const h3_dit *dit) {
    if(dit->sglang_reference||dit->layout.frozen_audio)return 0;
    if (dit->resume_sampler_mode >= 0) return dit->resume_sampler_mode;
    const char *cpu = getenv("H3_CPU_SAMPLER");
    if (cpu && *cpu && strcmp(cpu, "0")) return 0;
    const char *value = getenv("H3_GPU_SAMPLER");
    if (value) return *value && strcmp(value, "0");
    int continuation = dit->layout.prefix.video_prefix_t || dit->layout.prefix.audio_prefix_t;
    return h3_gpu_prefers_device_sampler(dit->gpu, continuation);
}

static unsigned gpu_sampler_window(void) {
    const char *value = getenv("H3_GPU_SAMPLER_WINDOW");
    if (!value || !*value) return 1;
    char *end = NULL;
    long parsed = strtol(value, &end, 10);
    return end != value && !*end && parsed >= 0 && parsed <= H3_MAX_STEPS
        ? (unsigned)parsed : 1;
}

static int ensure_previous_velocities(h3_dit *dit, char *error,
                                      size_t error_size) {
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    dit->previous_video_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->video_rows * VIDEO_PATCH);
    dit->previous_audio_velocity = h3_gpu_tensor_new_bf16(
        dit->gpu, (size_t)dit->audio_rows * AUDIO_CHANNELS);
    if (dit->previous_video_velocity && dit->previous_audio_velocity) return 1;
    free_tensor(&dit->previous_video_velocity);
    free_tensor(&dit->previous_audio_velocity);
    fail(error, error_size, "cannot allocate GPU Euler velocity cache: %s",
         h3_gpu_error(dit->gpu));
    return 0;
}

typedef struct {
    h3_gpu_tensor *classes[2], *initial[2], *strengths, *changed;
} bridge_gpu_state;

static void bridge_gpu_free(bridge_gpu_state *state) {
    for (int s = 0; s < 2; s++) {
        h3_gpu_tensor_free(state->classes[s]);
        h3_gpu_tensor_free(state->initial[s]);
    }
    h3_gpu_tensor_free(state->strengths);
    h3_gpu_tensor_free(state->changed);
}

static int bridge_gpu_init(h3_dit *dit, bridge_gpu_state *state,
    const float *video_rows, const float *audio_rows,
    char *error, size_t error_size) {
    if (!dit->layout.bridge && !dit->layout.prefix.video_prefix_t &&
        !dit->layout.prefix.audio_prefix_t) return 1;
    const uint32_t rows[] = {dit->video_rows, dit->audio_rows};
    const uint32_t widths[] = {VIDEO_PATCH, AUDIO_CHANNELS};
    const float *initial[] = {video_rows, audio_rows};
    uint32_t zero = 0;
    float masks[H3_TARGET_ROW_CLASSES] = {0};
    masks[H3_ROW_GENERATED_VIDEO] = masks[H3_ROW_GENERATED_AUDIO] = 1.0f;
    if (dit->layout.bridge) memcpy(masks, dit->bridge.class_mask, sizeof(masks));
    state->strengths = h3_gpu_tensor_from_f32(dit->gpu,
        masks, H3_TARGET_ROW_CLASSES);
    state->changed = h3_gpu_tensor_from_u32(dit->gpu, &zero, 1);
    int ok = state->strengths && state->changed;
    for (int s = 0; s < 2 && ok; s++) {
        uint32_t *classes = malloc((size_t)rows[s] * sizeof(*classes));
        if (!classes) { ok = 0; break; }
        for (uint32_t r = 0; r < rows[s]; r++) {
            classes[r] = (uint32_t)(dit->layout.bridge
                ? h3_bridge_row_class(&dit->bridge, s, r,
                    dit->latent_h, dit->latent_w, dit->audio_t)
                : h3_prefix_row_class(dit->layout.prefix, s, r,
                    dit->latent_h, dit->latent_w, dit->audio_t));
            if (classes[r] >= H3_TARGET_ROW_CLASSES) ok = 0;
        }
        if (ok) {
            state->classes[s] = h3_gpu_tensor_from_u32(dit->gpu, classes, rows[s]);
            state->initial[s] = h3_gpu_tensor_from_f32(dit->gpu, initial[s],
                (size_t)rows[s] * widths[s]);
            ok = state->classes[s] && state->initial[s];
        }
        free(classes);
    }
    if (!ok) fail(error, error_size, "cannot allocate GPU bridge classes/audit state: %s",
                  h3_gpu_error(dit->gpu));
    if (ok && getenv("H3_PROFILE"))
        fprintf(stderr, "h3cli: GPU bridge class maps %zu bytes; exact audit snapshot %zu bytes\n",
            ((size_t)rows[0] + rows[1] + H3_TARGET_ROW_CLASSES + 1) * sizeof(uint32_t),
            ((size_t)rows[0]*widths[0] + (size_t)rows[1]*widths[1]) * sizeof(float));
    return ok;
}

static int bridge_gpu_audit(h3_dit *dit, const bridge_gpu_state *state,
    int step, char *error, size_t error_size) {
    if (!state->changed) return 1;
    uint32_t changed = UINT32_MAX;
    if (!h3_gpu_tensor_read_u32(state->changed, &changed, 1) || changed) {
        fail(error, error_size, "GPU %s exact rows/class audit failed through step %d (flags %u)",
             dit->layout.bridge ? "bridge" : "hard continuation", step, changed);
        return 0;
    }
    if (getenv("H3_PROFILE") || getenv("H3_BRIDGE_DIAGNOSTICS"))
        fprintf(stderr, "h3cli: GPU %s exact video/audio bits unchanged through step %d\n",
            dit->layout.bridge ? "bridge" : "hard continuation", step);
    return 1;
}

static int denoise_euler_gpu_range(h3_dit *dit,h3_sampler_state *state,int stop,
    h3_dit_progress progress,void *progress_opaque,h3_dit_preview preview,void *preview_opaque,
    h3_preview_mode preview_mode, char *error,size_t error_size) {
    float *video_latent=state->video,*audio_latent=state->audio;
    int reuse_interval=state->reuse_interval;
    const uint8_t *selected=state->selected;
    unsigned window = gpu_sampler_window();
    int disable_command_split = window == 1 &&
#ifdef __APPLE__
        dit->sequence<8192 &&
#endif
                                getenv("H3_DIT_COMMAND_BLOCKS") == NULL;
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3cli: GPU sampler encode window is %s; internal split "
                "%s\n", window ? "bounded" : "unbounded",
                disable_command_split ? "disabled" : "enabled");

    size_t video_count = (size_t)dit->video_rows * VIDEO_PATCH;
    size_t audio_count = (size_t)dit->audio_rows * AUDIO_CHANNELS;
    size_t video_offset = (size_t)dit->video_condition_rows * VIDEO_PATCH;
    size_t audio_offset = (size_t)dit->audio_condition_rows * AUDIO_CHANNELS;
    if (video_count > UINT32_MAX || audio_count > UINT32_MAX ||
        video_offset > UINT32_MAX - video_count ||
        audio_offset > UINT32_MAX - audio_count ||
        (reuse_interval > 1 &&
         !ensure_previous_velocities(dit, error, error_size))) return 0;

    float *video_rows = malloc(video_count * sizeof(*video_rows));
    float *audio_rows = malloc(audio_count * sizeof(*audio_rows));
    if (!video_rows || !audio_rows) {
        fail(error, error_size, "out of memory packing GPU Euler latents");
        free(video_rows);
        free(audio_rows);
        return 0;
    }
    int ok = h3_dit_patchify_video(video_latent, VIDEO_CHANNELS,
        dit->latent_t, dit->latent_h, dit->latent_w, video_rows, video_count) &&
        h3_dit_pack_audio(audio_latent, AUDIO_CHANNELS, dit->audio_t,
                          audio_rows, audio_count) &&
        h3_gpu_tensor_write_f32_range(dit->video_input, video_offset,
                                      video_rows, video_count) &&
        h3_gpu_tensor_write_f32_range(dit->audio_input, audio_offset,
                                      audio_rows, audio_count);
    if (!ok) fail(error, error_size, "cannot pack/write GPU Euler latents");
    bridge_gpu_state bridge_state = {0};
    if (ok) ok = bridge_gpu_init(dit, &bridge_state, video_rows, audio_rows,
                                 error, error_size);
    if (ok && dit->on_latent_step && dit->on_latent_step(state->next_step, dit->sigmas.steps,
        video_latent, h3_dit_video_elements(dit), audio_latent,
        h3_dit_audio_elements(dit), dit->latent_opaque)) {
        fail(error, error_size, "latent callback stopped before sampling"); ok = 0;
    }

    if(ok && state->next_step && reuse_interval>1) {
        ok=state->gpu_last_video && state->gpu_last_audio && state->gpu_previous_video && state->gpu_previous_audio &&
            h3_gpu_tensor_write_bf16(dit->video_output_bf16,state->gpu_last_video,video_count) &&
            h3_gpu_tensor_write_bf16(dit->audio_output_bf16,state->gpu_last_audio,audio_count) &&
            h3_gpu_tensor_write_bf16(dit->previous_video_velocity,state->gpu_previous_video,video_count) &&
            h3_gpu_tensor_write_bf16(dit->previous_audio_velocity,state->gpu_previous_audio,audio_count);
        if(!ok) fail(error,error_size,"cannot restore native BF16 GPU velocity history");
    }
    unsigned pending_evaluations = 0;
    int command_active = 0;
    for (int step = state->next_step; step < stop && ok; step++) {
        double step_started=stream_now();uint64_t step_read_bytes=dit->stream_bytes,step_read_layers=dit->stream_read_layers;h3_gpu_stats step_stats={0};h3_gpu_get_stats(dit->gpu,&step_stats);
        /* A fresh command chain starts at a completed GPU boundary. Queued
         * windows keep their enqueue label so the CLI never times submission
         * work as a completed step or attributes a whole window to one step. */
        if (!(ok = report(progress, progress_opaque,
               command_active ? "denoise enqueue" : "denoise", step,
               dit->sigmas.steps, error, error_size))) break;
        if (!command_active) {
            ok = gpu_op(dit, h3_gpu_begin(dit->gpu), error, error_size,
                        "begin GPU Euler command chain");
            command_active = ok;
        }
        if (!ok) break;
        int evaluate = selected[step];
        if (evaluate) {
            if (state->last_evaluated >= 0 && reuse_interval > 1) {
                ok = gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_video_velocity, 0,
                    dit->video_output_bf16, 0, video_count),
                    error, error_size, "cache previous video velocity") &&
                    gpu_op(dit, h3_gpu_copy_bf16(
                    dit->gpu, dit->previous_audio_velocity, 0,
                    dit->audio_output_bf16, 0, audio_count),
                    error, error_size, "cache previous audio velocity");

            }
            if (ok) ok = encode_forward(dit, step, 0, 0,
                                        disable_command_split,
                                        error, error_size);
            if (ok) {
                state->previous_evaluated=state->last_evaluated;
                state->last_evaluated = step;
                pending_evaluations++;
            }
        }
        if (!ok) break;

        float video_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.video[step], dit->sigmas.video[state->last_evaluated],
            state->previous_evaluated >= 0
                ? dit->sigmas.video[state->previous_evaluated] : 0.0f,
            state->previous_evaluated >= 0);
        float audio_ratio = evaluate ? 0.0f : extrapolation_ratio(
            dit->sigmas.audio[step], dit->sigmas.audio[state->last_evaluated],
            state->previous_evaluated >= 0
                ? dit->sigmas.audio[state->previous_evaluated] : 0.0f,
            state->previous_evaluated >= 0);
        const h3_gpu_tensor *previous_video = reuse_interval > 1 && state->previous_evaluated >= 0
            ? dit->previous_video_velocity : dit->video_output_bf16;
        const h3_gpu_tensor *previous_audio = reuse_interval > 1 && state->previous_evaluated >= 0
            ? dit->previous_audio_velocity : dit->audio_output_bf16;
        if (bridge_state.strengths) {
            const h3_gpu_tensor *last[] = {dit->video_output_bf16, dit->audio_output_bf16};
            const h3_gpu_tensor *previous[] = {previous_video, previous_audio};
            h3_gpu_tensor *sample[] = {dit->video_input, dit->audio_input};
            const size_t offsets[] = {video_offset, audio_offset};
            const uint32_t rows[] = {dit->video_rows, dit->audio_rows};
            const uint32_t widths[] = {VIDEO_PATCH, AUDIO_CHANNELS};
            const float deltas[] = {dit->sigmas.video[step]-dit->sigmas.video[step+1],
                                    dit->sigmas.audio[step]-dit->sigmas.audio[step+1]};
            const float ratios[] = {video_ratio, audio_ratio};
            for (int s = 0; s < 2 && ok; s++) {
                ok = gpu_op(dit, h3_gpu_bridge_euler_bf16(dit->gpu, sample[s],
                    offsets[s], last[s], previous[s], bridge_state.classes[s],
                    bridge_state.strengths, rows[s], widths[s], deltas[s], ratios[s]),
                    error, error_size, "GPU bridge Euler step") &&
                    gpu_op(dit, h3_gpu_bridge_check_exact(dit->gpu, sample[s],
                    offsets[s], bridge_state.initial[s], bridge_state.classes[s],
                    bridge_state.strengths, bridge_state.changed, rows[s], widths[s]),
                    error, error_size, "GPU bridge exact-row audit");
            }
        } else ok = gpu_op(dit, h3_gpu_euler_bf16(
                dit->gpu, dit->video_input, video_offset,
                dit->video_output_bf16, previous_video, (uint32_t)video_count,
                dit->sigmas.video[step] - dit->sigmas.video[step + 1],
                video_ratio), error, error_size, "GPU video Euler step") &&
             gpu_op(dit, h3_gpu_euler_bf16(
                dit->gpu, dit->audio_input, audio_offset,
                dit->audio_output_bf16, previous_audio, (uint32_t)audio_count,
                dit->sigmas.audio[step] - dit->sigmas.audio[step + 1],
                audio_ratio), error, error_size, "GPU audio Euler step");
        if (ok && preview && preview_mode == H3_PREVIEW_DENOISED) {
            /* QKV and its temporary aliases are dead after the final head.
             * Core residuals and raw BF16 velocity histories use other buffers.
             * Write C,T,H,W directly into this scratch, avoiding both a new
             * device allocation and a separate full velocity readback. */
            ok = gpu_op(dit, h3_gpu_video_preview_bf16(dit->gpu, dit->qkv,
                dit->video_input, video_offset, dit->video_output_bf16,
                previous_video, bridge_state.classes[0], bridge_state.strengths,
                (uint32_t)dit->latent_t, (uint32_t)dit->latent_h,
                (uint32_t)dit->latent_w, dit->sigmas.video[step+1], video_ratio),
                error, error_size, "GPU denoised display preview");
        }
        if (ok && (evaluate || preview || dit->on_latent_step || h3_profile_steps_enabled())) {
            int finish = h3_profile_steps_enabled() || preview || dit->on_latent_step || step + 1 == stop ||
                         (window && pending_evaluations >= window);
            ok = gpu_op(dit, finish ? h3_gpu_submit(dit->gpu)
                                    : h3_gpu_continue(dit->gpu),
                        error, error_size,
                        finish ? "submit GPU Euler window"
                               : "continue GPU Euler command chain");
            if (ok && finish) {
                command_active = 0;
                pending_evaluations = 0;
                ok = bridge_gpu_audit(dit, &bridge_state, step + 1, error, error_size);
            }
        }
        if (ok) state->next_step=step+1;
        if(ok && getenv("H3_EXPERIMENT_TRACE")) {
            h3_gpu_stats after={0};h3_gpu_get_stats(dit->gpu,&after);
#define DELTA(field) (unsigned long long)(after.field-step_stats.field)
            fprintf(stderr,"h3_experiment {\"step\":%d,\"total\":%d,\"video_sigma\":%.9g,\"audio_sigma\":%.9g,\"evaluated\":%d,\"blocks\":%u,\"forward_count\":%u,\"wall_seconds\":%.9g,\"stream_read_bytes\":%llu,\"stream_read_layers\":%llu,\"dense_calls\":%llu,\"sparse_calls\":%llu,\"router_calls\":%llu,\"protected_calls\":%llu,\"dense_bypass\":%llu,\"selected\":%llu,\"possible\":%llu,\"attention_seconds\":%.9g,\"router_seconds\":%.9g,\"sparse_kernel_seconds\":%.9g,\"h2d_seconds\":%.9g,\"d2h_seconds\":%.9g,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"gemm_calls\":%llu,\"quant_calls\":%llu,\"attention_timing_samples\":%llu,\"attention_timing_missed\":%llu,\"device_bytes\":%llu,\"peak_tensor_bytes\":%llu,\"pinned_bytes\":%llu,\"peak_pinned_bytes\":%llu}\n",
                step,dit->sigmas.steps,dit->sigmas.video[step],dit->sigmas.audio[step],evaluate,
                evaluate?dit->last_executed_blocks:0,dit->core_forward_count,stream_now()-step_started,
                (unsigned long long)(dit->stream_bytes-step_read_bytes),(unsigned long long)(dit->stream_read_layers-step_read_layers),
                DELTA(main_dense_calls),DELTA(subblock_calls),DELTA(subblock_router_calls),DELTA(subblock_protected_calls),
                DELTA(subblock_bypass),DELTA(subblock_selected),DELTA(subblock_possible),
                after.main_attention_seconds-step_stats.main_attention_seconds,
                after.subblock_router_seconds-step_stats.subblock_router_seconds,after.subblock_kernel_seconds-step_stats.subblock_kernel_seconds,
                after.h2d_seconds-step_stats.h2d_seconds,after.d2h_seconds-step_stats.d2h_seconds,
                DELTA(h2d_bytes),DELTA(d2h_bytes),DELTA(linear_dispatches),DELTA(quant_projection_dispatches),DELTA(attention_timing_samples),DELTA(attention_timing_missed),
                (unsigned long long)after.device_bytes,(unsigned long long)after.peak_live_bytes,
                (unsigned long long)after.pinned_bytes,(unsigned long long)after.peak_pinned_bytes);
#undef DELTA
        }
        if(ok)h3_profile_step(dit->gpu,step_started,step,dit->sigmas.steps,evaluate,step_stats);
        if (ok && (preview || dit->on_latent_step)) {
            ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent,
                     h3_dit_video_elements(dit));
            if (ok && dit->on_latent_step)
                ok = h3_gpu_tensor_read_f32_range(dit->audio_input, audio_offset,
                    audio_rows, audio_count) && h3_dit_unpack_audio(audio_rows,
                    AUDIO_CHANNELS, dit->audio_t, audio_latent, h3_dit_audio_elements(dit));
            if (!ok) {
                fail(error, error_size,
                     "cannot read GPU Euler callback latent at step %d", step);
            } else if (dit->on_latent_step && dit->on_latent_step(step + 1,
                dit->sigmas.steps, video_latent, h3_dit_video_elements(dit),
                audio_latent, h3_dit_audio_elements(dit), dit->latent_opaque)) {
                fail(error, error_size, "latent callback stopped at step %d", step + 1);
                ok = 0;
            }
        }
        /* submit() above waits for completion. Report that boundary before
         * preview decoding, using the same timer as CPU-state sampling. No
         * extra synchronization is needed for progress output. */
        if (ok) ok = report(progress, progress_opaque,
               command_active ? "denoise enqueue" : "denoise", step + 1,
               dit->sigmas.steps, error, error_size);
        if (ok && preview) {
            /* State callbacks always receive x_next above. Only the display
             * callback may receive x0, using the existing host packing buffer. */
            const float *display = video_latent;
            if (ok && preview && preview_mode == H3_PREVIEW_DENOISED) {
                ok = h3_gpu_video_preview_read(dit->qkv, video_rows, video_count);
                display = video_rows;
                if (!ok) fail(error, error_size, "cannot read GPU denoised preview");
            }
            if (ok && preview && preview(step + 1, dit->sigmas.steps, display,
                                         video_count, preview_opaque)) {
                fail(error, error_size,
                     "denoising preview stopped at step %d", step + 1);
                ok = 0;
            }
        }
    }
    if (ok && command_active) {
        ok = gpu_op(dit, h3_gpu_submit(dit->gpu), error, error_size,
                    "submit GPU Euler denoise");
        if (ok) ok = bridge_gpu_audit(dit, &bridge_state, state->next_step, error, error_size);
    }
    if (ok) ok = h3_gpu_tensor_read_f32_range(
                     dit->video_input, video_offset, video_rows, video_count) &&
                 h3_gpu_tensor_read_f32_range(
                     dit->audio_input, audio_offset, audio_rows, audio_count);
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot read GPU Euler latents");
    if (ok) ok = h3_dit_unpatchify_video(
                     video_rows, VIDEO_CHANNELS, dit->latent_t, dit->latent_h,
                     dit->latent_w, video_latent, h3_dit_video_elements(dit)) &&
                 h3_dit_unpack_audio(audio_rows, AUDIO_CHANNELS, dit->audio_t,
                                     audio_latent,
                                     h3_dit_audio_elements(dit));
    if (!ok && (!error || !*error))
        fail(error, error_size, "cannot unpack GPU Euler latents");
    if(ok) ok=h3_dit_sampler_export(dit,state,0,error,error_size);
    free(video_rows);
    free(audio_rows);
    if (!ok) h3_gpu_cancel(dit->gpu);
    bridge_gpu_free(&bridge_state);
    if (ok) ok = report(progress, progress_opaque, "denoise", state->next_step,
               dit->sigmas.steps, error, error_size);
    h3_gpu_profile_mark(dit->gpu, "GPU Euler denoise");
    return ok;
}

static int denoise_euler_gpu(h3_dit *dit,float *video,float *audio,int reuse,
    h3_dit_progress progress,void *opaque,h3_dit_preview preview,void *preview_opaque,
    char *error,size_t size) {
    h3_sampler_state *s=h3_sampler_state_create(&dit->sigmas,h3_dit_video_elements(dit),h3_dit_audio_elements(dit),reuse);
    if(!s) return 0;
    s->sampler_mode=1;
    memcpy(s->video,video,s->video_elements*4); memcpy(s->audio,audio,s->audio_elements*4);
    int ok=h3_dit_resolve_reuse(s->total_steps,reuse,s->selected)>0 &&
        h3_dit_denoise_euler_range(dit,s,s->total_steps,progress,opaque,preview,preview_opaque,error,size);
    if(ok) { memcpy(video,s->video,s->video_elements*4); memcpy(audio,s->audio,s->audio_elements*4); }
    h3_sampler_state_free(s); return ok;
}

int h3_dit_denoise(h3_dit *dit, float *video_latent, float *audio_latent,
                   h3_dit_progress progress, void *progress_opaque,
                   char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid DiT denoising arguments");
        return 0;
    }
    if (dit->layout.prefix.video_prefix_t || dit->layout.prefix.audio_prefix_t) {
        fail(error,error_size,"continuation requires the Euler sampler"); return 0;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    float *video_denoised = malloc(video_count * sizeof(*video_denoised));
    float *audio_denoised = malloc(audio_count * sizeof(*audio_denoised));
    float *old_video = malloc(video_count * sizeof(*old_video));
    float *old_audio = malloc(audio_count * sizeof(*old_audio));
    float *video_next = malloc(video_count * sizeof(*video_next));
    float *audio_next = malloc(audio_count * sizeof(*audio_next));
    if (!video_velocity || !audio_velocity || !video_denoised ||
        !audio_denoised || !old_video || !old_audio || !video_next ||
        !audio_next) {
        fail(error, error_size, "out of memory allocating RES solver state");
        free(video_velocity); free(audio_velocity); free(video_denoised);
        free(audio_denoised); free(old_video); free(old_audio);
        free(video_next); free(audio_next);
        return 0;
    }
    int ok = 1;
    for (int step = 0; step < dit->sigmas.steps && ok; step++) {
        if (!(ok = report(progress, progress_opaque, "denoise", step,
               dit->sigmas.steps, error, error_size))) break;
        ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                            video_velocity, audio_velocity,
                            error, error_size);
        float sigma = dit->sigmas.video[step];
        float timestep = 1.0f - sigma;
        float sigma_from_timestep = 1.0f - timestep;
        float audio_slope = (float)h3_time_shift_slope(
            sigma, H3_VIDEO_SIGMA_SHIFT, H3_AUDIO_SIGMA_SHIFT);
        if (ok) {
            for (size_t index = 0; index < video_count; index++)
                video_denoised[index] = video_latent[index] +
                    sigma_from_timestep * video_velocity[index];
            for (size_t index = 0; index < audio_count; index++)
                audio_denoised[index] = audio_latent[index] +
                    sigma_from_timestep * audio_velocity[index] * audio_slope;
            ok = h3_res_step(video_next, video_latent, video_denoised,
                             step ? old_video : NULL, video_count,
                             dit->sigmas.video, step, dit->sigmas.steps) &&
                 h3_res_step(audio_next, audio_latent, audio_denoised,
                             step ? old_audio : NULL, audio_count,
                             dit->sigmas.video, step, dit->sigmas.steps);
            if (!ok) fail(error, error_size, "RES solver rejected step %d", step);
        }
        if (ok) {
            memcpy(video_latent, video_next,
                   video_count * sizeof(*video_latent));
            memcpy(audio_latent, audio_next,
                   audio_count * sizeof(*audio_latent));
            memcpy(old_video, video_denoised,
                   video_count * sizeof(*old_video));
            memcpy(old_audio, audio_denoised,
                   audio_count * sizeof(*old_audio));
            ok = report(progress, progress_opaque, "denoise", step + 1,
               dit->sigmas.steps, error, error_size);
        }
    }
    free(video_velocity); free(audio_velocity); free(video_denoised);
    free(audio_denoised); free(old_video); free(old_audio);
    free(video_next); free(audio_next);
    h3_gpu_profile_mark(dit->gpu, "RES denoise");
    return ok;
}

int h3_dit_denoise_euler_preview(
                         h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!dit || !video_latent || !audio_latent || reuse_interval < 1 ||
        reuse_interval > 32 ||
        dit->sigmas.steps != h3_dit_schedule_steps(dit->schedule)) {
        fail(error, error_size, "invalid Euler denoising arguments");
        return 0;
    }
    int continuation = dit->layout.prefix.video_prefix_t || dit->layout.prefix.audio_prefix_t;
    if (dit->layout.bridge && reuse_interval > 3) {
        fail(error, error_size, "bridge continuation supports --reuse 1, 2, or 3; other intervals are not validated"); return 0;
    }
    const char *custom_bridge_reuse = getenv("H3_REUSE_STEPS");
    if (dit->layout.bridge && reuse_interval > 1 && custom_bridge_reuse && *custom_bridge_reuse) {
        fail(error, error_size, "bridge continuation does not support custom H3_REUSE_STEPS schedules"); return 0;
    }
    if (dit->layout.bridge && reuse_interval > 1 && dit->core_reuse_interval > 1) {
        fail(error, error_size, "bridge velocity and core reuse cannot be combined"); return 0;
    }
    /* CUDA keeps continuation state on the device. Metal retains its existing
     * CPU continuation default; explicit options and stored modes take priority. */
    int use_gpu = gpu_sampler_requested(dit);
    if (continuation)
        H3_VERBOSE("h3cli: %s continuation uses %s Euler sampler (%s DiT)\n",
            dit->layout.bridge ? "bridge" : "hard", use_gpu ? "GPU-state F32" : "CPU F32",
            h3_gpu_backend_name(dit->gpu));
    if (use_gpu)
        return denoise_euler_gpu(dit, video_latent, audio_latent,
                                 reuse_interval, progress, progress_opaque,
                                 preview, preview_opaque,
                                 error, error_size);
    h3_sampler_state *state = h3_sampler_state_create(&dit->sigmas,
        h3_dit_video_elements(dit), h3_dit_audio_elements(dit), reuse_interval);
    if (!state) { fail(error,error_size,"out of memory creating Euler state"); return 0; }
    int custom_count = reuse_interval > 1 ? parse_reuse_steps(dit->sigmas.steps,state->selected) : 0;
    if (custom_count < 0) {
        h3_sampler_state_free(state);
        fail(error,error_size,"H3_REUSE_STEPS must be increasing and include 0 and %d",dit->sigmas.steps-1);
        return 0;
    }
    if (reuse_interval>1 && getenv("H3_PROFILE")) {
        int evaluations=0;
        for (int i=0;i<state->total_steps;i++) evaluations+=state->selected[i]!=0;
        H3_VERBOSE("h3cli: %s reuse schedule has %d evaluations\n",custom_count>0?"custom":"selected",evaluations);
    }
    memcpy(state->video,video_latent,state->video_elements*sizeof(float));
    memcpy(state->audio,audio_latent,state->audio_elements*sizeof(float));
    int ok = h3_dit_denoise_euler_range(dit,state,state->total_steps,
        progress,progress_opaque,preview,preview_opaque,error,error_size);
    memcpy(video_latent,state->video,state->video_elements*sizeof(float));
    memcpy(audio_latent,state->audio,state->audio_elements*sizeof(float));
    h3_sampler_state_free(state);
    return ok;
}

int h3_dit_denoise_euler_range(h3_dit *dit, h3_sampler_state *state,
                         int stop_after_step,
                         h3_dit_progress progress, void *progress_opaque,
                         h3_dit_preview preview, void *preview_opaque,
                         char *error, size_t error_size) {
    if (error && error_size) *error = 0;
    if(state&&!h3_test_evaluation_budget(stop_after_step-state->next_step,error,error_size))return 0;
    if (!dit || !state || state->version != H3_SAMPLE_VERSION ||
        state->next_step < 0 || stop_after_step < state->next_step ||
        stop_after_step > state->total_steps || state->total_steps != dit->sigmas.steps ||
        state->sigmas.steps != state->total_steps ||
        state->video_elements != h3_dit_video_elements(dit) ||
        state->audio_elements != h3_dit_audio_elements(dit) || !state->video || !state->audio ||
        memcmp(state->sigmas.video,dit->sigmas.video,((size_t)dit->sigmas.steps+1)*sizeof(float)) ||
        memcmp(state->sigmas.audio,dit->sigmas.audio,((size_t)dit->sigmas.steps+1)*sizeof(float))) {
        fail(error,error_size,"incompatible Euler range/state or absolute stop boundary"); return 0;
    }
    if (state->reuse_interval < 1 || state->reuse_interval > 32 ||
        (state->next_step && !state->selected[state->next_step-1] && state->last_evaluated < 0)) {
        fail(error,error_size,"invalid Euler reuse state"); return 0;
    }

    int expected_last=-1, expected_previous=-1;
    for (int step=0;step<state->total_steps;step++) {
        if (state->selected[step]>1 || (state->reuse_interval==1 && !state->selected[step])) { fail(error,error_size,"invalid Euler evaluation bitmap"); return 0; }
        if (step<state->next_step && state->selected[step]) { expected_previous=expected_last; expected_last=step; }
    }
    if (!state->selected[0] || !state->selected[state->total_steps-1] ||
        (state->reuse_interval>1 && (state->last_evaluated!=expected_last || state->previous_evaluated!=expected_previous))) {
        fail(error,error_size,"inconsistent Euler reuse evaluation history"); return 0;
    }
    h3_preview_mode preview_mode;
    if (!h3_preview_mode_parse(getenv("H3_PREVIEW_MODE"), &preview_mode,
                               error, error_size)) return 0;
    if (preview && (getenv("H3_PROFILE") || getenv("H3_PREVIEW_DIAGNOSTICS")))
        h3_log_printf("Preview mode: %s\n", h3_preview_mode_name(preview_mode));
    const char *teacher=getenv("H3_TEST_NATIVE_TEACHER_DIR");
    const char *teacher_budget=dit->sglang_reference && state->total_steps==50?"50":"6";
    if(teacher && (!*teacher || !getenv("H3_TEST_MAX_EVALUATIONS") ||
        strcmp(getenv("H3_TEST_MAX_EVALUATIONS"),teacher_budget) || state->total_steps>atoi(teacher_budget) ||
        state->sampler_mode || state->next_step || state->reuse_interval!=1 ||
        dit->core_reuse_interval!=1 || !getenv("H3_TEST_NATIVE_STEP_DIR"))) {
        fail(error,error_size,"teacher diagnostics require fresh CPU Euler, explicit evaluation budget, no reuse and step capture");return 0;
    }
    if(state->sampler_mode)
        return denoise_euler_gpu_range(dit,state,stop_after_step,progress,progress_opaque,
            preview,preview_opaque,preview_mode,error,error_size);
    if(state->upscale.stage==2 && (!dit->layout.frozen_audio||state->sampler_mode||
       !h3_upscale_audio_intact(state))) {fail(error,error_size,"invalid frozen-audio refinement state");return 0;}
    int first_step = state->next_step;
    float *video_latent = state->video, *audio_latent = state->audio;
    int reuse_interval = state->reuse_interval;
    int continuation = dit->layout.prefix.video_prefix_t || dit->layout.prefix.audio_prefix_t;
    int profile_prefix = continuation && getenv("H3_PROFILE");
    double prefix_update_seconds = 0.0;
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    float *video_velocity = malloc(video_count * sizeof(*video_velocity));
    float *audio_velocity = malloc(audio_count * sizeof(*audio_velocity));
    int audit_prefix=dit->layout.bridge || (continuation&&!dit->layout.frozen_audio&&
        (dit->adaptive_mode||dit->captured_attention==H3_ATTENTION_SUBBLOCK));
    float *initial_video = audit_prefix ? malloc(video_count * sizeof(*initial_video)) : NULL;
    float *initial_audio = audit_prefix ? malloc(audio_count * sizeof(*initial_audio)) : NULL;
    float *last_video = state->last_video_velocity;
    float *previous_video = state->previous_video_velocity;
    float *last_audio = state->last_audio_velocity;
    float *previous_audio = state->previous_audio_velocity;
    if (!video_velocity || !audio_velocity ||
        (audit_prefix && (!initial_video || !initial_audio)) ||
        (reuse_interval > 1 &&
         (!last_video || !previous_video || !last_audio || !previous_audio))) {
        fail(error, error_size, "out of memory allocating Euler velocities");
        free(video_velocity);
        free(audio_velocity);
        free(initial_video);
        free(initial_audio);
        return 0;
    }
    int ok = 1;
    if (audit_prefix) {
        memcpy(initial_video, video_latent, video_count*sizeof(*initial_video));
        memcpy(initial_audio, audio_latent, audio_count*sizeof(*initial_audio));
        if (profile_prefix) fprintf(stderr,"h3cli: bridge exact-row audit snapshot: %zu bytes\n",(video_count+audio_count)*sizeof(float));
    }
    if (dit->on_latent_step && dit->on_latent_step(state->next_step,dit->sigmas.steps,
        video_latent,video_count,audio_latent,audio_count,dit->latent_opaque)) {
        fail(error,error_size,"latent callback stopped before sampling"); ok=0;
    }
    for (int step = state->next_step; step < stop_after_step && ok; step++) {
        double step_started=stream_now();uint64_t step_read_bytes=dit->stream_bytes,step_read_layers=dit->stream_read_layers;h3_gpu_stats step_stats={0};h3_gpu_get_stats(dit->gpu,&step_stats);
        if(teacher && (step || dit->sglang_reference) && !(dit->sglang_reference?h3_test_teacher_load_sglang:h3_test_teacher_load)(teacher,step,video_latent,video_count,
            audio_latent,audio_count,error,error_size)){ok=0;break;}
        if(!debug_native_step(step,"video-input",video_latent,video_count)||
           !debug_native_step(step,"audio-input",audio_latent,audio_count)) {
            fail(error,error_size,"cannot capture teacher/trajectory inputs");ok=0;break;
        }
        if(teacher)fprintf(stderr,"h3_teacher {\"step\":%d,\"video_sigma\":%.9g,\"audio_sigma\":%.9g,\"timing_comparable\":false}\n",
            step+1,dit->sigmas.video[step],dit->sigmas.audio[step]);
        if (!(ok = report(progress, progress_opaque, "denoise", step,
               dit->sigmas.steps, error, error_size))) break;
        int evaluate = state->selected[step];
        if (evaluate) {
            ok = h3_dit_forward(dit, step, video_latent, audio_latent,
                                video_velocity, audio_velocity,
                                error, error_size);
            if (ok && reuse_interval > 1) {
                /* Cache RAW predictions before bridge Euler scales the working
                 * velocity. Reuse/extrapolation is masked once per application. */
                if (state->last_evaluated >= 0) {
                    memcpy(previous_video, last_video,
                           video_count * sizeof(*previous_video));
                    memcpy(previous_audio, last_audio,
                           audio_count * sizeof(*previous_audio));
                    state->previous_evaluated = state->last_evaluated;
                }
                memcpy(last_video, video_velocity,
                       video_count * sizeof(*last_video));
                memcpy(last_audio, audio_velocity,
                       audio_count * sizeof(*last_audio));
                state->last_evaluated = step;
            }
        } else {
            extrapolate_velocity(
                video_velocity, last_video, previous_video, video_count,
                dit->sigmas.video[step], dit->sigmas.video[state->last_evaluated],
                state->previous_evaluated >= 0
                    ? dit->sigmas.video[state->previous_evaluated] : 0.0f,
                state->previous_evaluated >= 0);
            extrapolate_velocity(
                audio_velocity, last_audio, previous_audio, audio_count,
                dit->sigmas.audio[step], dit->sigmas.audio[state->last_evaluated],
                state->previous_evaluated >= 0
                    ? dit->sigmas.audio[state->previous_evaluated] : 0.0f,
                state->previous_evaluated >= 0);
        }
        if (ok) {
            if(state->upscale.stage==2) {
                /* Audio remains joint-attention conditioning; never invoke an
                 * audio Euler/noise operation, including at sigma zero. */
                ok=(dit->sglang_reference?h3_sglang_euler:h3_euler_velocity_step)(video_latent,video_velocity,video_count,
                    dit->sigmas.video[step],dit->sigmas.video[step+1])&&h3_upscale_audio_intact(state);
                if(!ok)fail(error,error_size,"refinement update or frozen audio identity failed");
            } else if (dit->layout.bridge) {
                h3_bridge_step_stats stats;
                int diagnostics = profile_prefix || getenv("H3_BRIDGE_DIAGNOSTICS");
                double update_started = profile_prefix ? stream_now() : 0.0;
                ok = h3_bridge_euler_step(dit->layout.bridge, dit->latent_t,
                    dit->latent_h, dit->latent_w, dit->audio_t,
                    video_latent, audio_latent, video_velocity, audio_velocity,
                    dit->sigmas.video[step], dit->sigmas.video[step+1],
                    dit->sigmas.audio[step], dit->sigmas.audio[step+1], diagnostics ? &stats : NULL);
                if (ok && !h3_bridge_check_exact(dit->layout.bridge, dit->latent_t,
                    dit->latent_h, dit->latent_w, dit->audio_t, video_latent, audio_latent,
                    initial_video, initial_audio)) {
                    fail(error,error_size,"bridge exact video/audio rows changed at step %d",step+1); ok=0;
                }
                if (ok && diagnostics) {
                    fprintf(stderr,"h3cli: bridge step %d exact video/audio bits unchanged\n",step+1);
                    for (int c=0; c<H3_TARGET_ROW_CLASSES; c++) if (stats.elements[c]) {
                        double n=(double)stats.elements[c];
                        fprintf(stderr,"h3cli: bridge step %d class %d mask %.6g raw_rms %.9g scaled_rms %.9g update_rms %.9g changed %zu/%zu\n",
                            step+1,c,(double)dit->bridge.class_mask[c],sqrt(stats.raw_square[c]/n),
                            sqrt(stats.scaled_square[c]/n),sqrt(stats.update_square[c]/n),stats.changed[c],stats.elements[c]);
                    }
                }
                if (profile_prefix) prefix_update_seconds += stream_now()-update_started;
            } else if (continuation) {
                double update_started = profile_prefix ? stream_now() : 0.0;
                h3_prefix_mask_velocity(dit->layout.prefix, dit->latent_t,
                    dit->latent_h, dit->latent_w, dit->audio_t, video_velocity, audio_velocity);
                ok = h3_prefix_euler_step(dit->layout.prefix, dit->latent_t,
                    dit->latent_h, dit->latent_w, dit->audio_t,
                    video_latent, audio_latent, video_velocity, audio_velocity,
                    dit->sigmas.video[step], dit->sigmas.video[step+1],
                    dit->sigmas.audio[step], dit->sigmas.audio[step+1]);
                if (profile_prefix) prefix_update_seconds += stream_now()-update_started;
            } else ok = (dit->sglang_reference?h3_sglang_euler:h3_euler_velocity_step)(
                     video_latent, video_velocity, video_count,
                     dit->sigmas.video[step], dit->sigmas.video[step + 1]) &&
                 (dit->sglang_reference?h3_sglang_euler:h3_euler_velocity_step)(
                     audio_latent, audio_velocity, audio_count,
                     dit->sigmas.audio[step], dit->sigmas.audio[step + 1]);
            if (!ok && (!error || !*error)) fail(error, error_size,
                          "Euler solver rejected step %d", step);
        }
        if (ok) state->next_step = step + 1;
        if(ok && audit_prefix && !dit->layout.bridge) {
            size_t spatial=(size_t)dit->latent_h*dit->latent_w;
            for(size_t c=0;ok&&c<24;c++)ok=!memcmp(video_latent+c*dit->latent_t*spatial,
                initial_video+c*dit->latent_t*spatial,dit->layout.prefix.video_prefix_t*spatial*sizeof(float));
            for(size_t c=0;ok&&c<64;c++)ok=!memcmp(audio_latent+c*dit->audio_t,
                initial_audio+c*dit->audio_t,dit->layout.prefix.audio_prefix_t*sizeof(float));
            if(!ok)fail(error,error_size,"hard continuation exact video/audio rows changed at step %d",step+1);
        }
        if(ok && dit->adaptive_mode && dit->adaptive_pending) {
            if(dit->adaptive_recipe==H3_ADAPTIVE_CONTINUATION_VERSION) {
                fprintf(stderr,"h3cli: adaptive continuation step=%d mode=%s context=%d/%d recipe=4 components=",
                    step,dit->layout.bridge?"bridge":"hard",dit->layout.prefix.video_prefix_t,dit->layout.prefix.audio_prefix_t);
                for(unsigned c=0;c<H3_ADAPTIVE_REGIONS;c++)if(dit->adaptive_regions.counts[c])
                    fprintf(stderr,"%u:%u:%.9g,",c,dit->adaptive_regions.counts[c],dit->adaptive_scores[c]);
                fprintf(stderr," workspace=%zu exact=ok\n",dit->adaptive_plan.scratch_bytes);
            }
            h3_adaptive_commit(&dit->adaptive_history,step,dit->adaptive_phase,dit->adaptive_refresh);
            dit->adaptive_pending=0;
            fprintf(stderr,"h3cli: adaptive step=%d video_sigma=%.9g audio_sigma=%.9g decision=%s reason=%s score=%.9g global_score=%.9g video_score=%.9g audio_score=%.9g blocks=%d streak=%u bytes=%zu\n",
                step,dit->sigmas.video[step],dit->sigmas.audio[step],dit->adaptive_refresh?"refresh":"hit",
                dit->adaptive_reason,dit->adaptive_score,dit->adaptive_scores[0],dit->adaptive_scores[1],dit->adaptive_scores[2],
                dit->adaptive_refresh?50:1,dit->adaptive_history.streak,dit->adaptive_plan.bytes);
        }
        if(ok && getenv("H3_EXPERIMENT_TRACE")) {
            h3_gpu_stats after={0};h3_gpu_get_stats(dit->gpu,&after);
#define DELTA(field) (unsigned long long)(after.field-step_stats.field)
            fprintf(stderr,"h3_experiment {\"step\":%d,\"total\":%d,\"video_sigma\":%.9g,\"audio_sigma\":%.9g,\"evaluated\":%d,\"blocks\":%u,\"forward_count\":%u,\"wall_seconds\":%.9g,\"stream_read_bytes\":%llu,\"stream_read_layers\":%llu,\"dense_calls\":%llu,\"sparse_calls\":%llu,\"router_calls\":%llu,\"protected_calls\":%llu,\"dense_bypass\":%llu,\"selected\":%llu,\"possible\":%llu,\"attention_seconds\":%.9g,\"router_seconds\":%.9g,\"sparse_kernel_seconds\":%.9g,\"h2d_seconds\":%.9g,\"d2h_seconds\":%.9g,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"gemm_calls\":%llu,\"quant_calls\":%llu,\"attention_timing_samples\":%llu,\"attention_timing_missed\":%llu,\"device_bytes\":%llu,\"peak_tensor_bytes\":%llu,\"pinned_bytes\":%llu,\"peak_pinned_bytes\":%llu}\n",
                step,dit->sigmas.steps,dit->sigmas.video[step],dit->sigmas.audio[step],evaluate,
                evaluate?dit->last_executed_blocks:0,dit->core_forward_count,stream_now()-step_started,
                (unsigned long long)(dit->stream_bytes-step_read_bytes),(unsigned long long)(dit->stream_read_layers-step_read_layers),
                DELTA(main_dense_calls),DELTA(subblock_calls),DELTA(subblock_router_calls),DELTA(subblock_protected_calls),
                DELTA(subblock_bypass),DELTA(subblock_selected),DELTA(subblock_possible),
                after.main_attention_seconds-step_stats.main_attention_seconds,
                after.subblock_router_seconds-step_stats.subblock_router_seconds,after.subblock_kernel_seconds-step_stats.subblock_kernel_seconds,
                after.h2d_seconds-step_stats.h2d_seconds,after.d2h_seconds-step_stats.d2h_seconds,
                DELTA(h2d_bytes),DELTA(d2h_bytes),DELTA(linear_dispatches),DELTA(quant_projection_dispatches),DELTA(attention_timing_samples),DELTA(attention_timing_missed),
                (unsigned long long)after.device_bytes,(unsigned long long)after.peak_live_bytes,
                (unsigned long long)after.pinned_bytes,(unsigned long long)after.peak_pinned_bytes);
#undef DELTA
        }
        if(ok)h3_profile_step(dit->gpu,step_started,step,dit->sigmas.steps,evaluate,step_stats);
        /* Report completed work before diagnostics and preview decoding so
         * the CLI step duration measures denoising, not display overhead. */
        if (ok) ok = report(progress, progress_opaque, "denoise", step + 1,
               dit->sigmas.steps, error, error_size);
        if(ok && (!debug_native_step(step,"video-velocity",video_velocity,video_count)||
                  !debug_native_step(step,"audio-velocity",audio_velocity,audio_count)||
                  !debug_native_step(step,"video-latent",video_latent,video_count)||
                  !debug_native_step(step,"audio-latent",audio_latent,audio_count))) {
            fail(error,error_size,"cannot capture native attention step diagnostics");ok=0;
        }
        if (ok && dit->on_latent_step && dit->on_latent_step(step+1,dit->sigmas.steps,
            video_latent,video_count,audio_latent,audio_count,dit->latent_opaque)) {
            fail(error,error_size,"latent callback stopped at step %d",step+1); ok=0;
        }
        /* Raw/reused history was committed before Euler; bridge diagnostics
         * and exact x_next callbacks are complete. This workspace is now dead
         * as velocity and may hold a display estimate. Never overwrite state. */
        const float *display = video_latent;
        if (ok && preview && preview_mode == H3_PREVIEW_DENOISED) {
            h3_preview_denoised_f32(video_velocity, video_latent, video_count,
                                    dit->sigmas.video[step+1]);
            display = video_velocity;
        }
        if (ok && preview &&
            preview(step + 1, dit->sigmas.steps, display, video_count,
                    preview_opaque)) {
            fail(error, error_size, "denoising preview stopped at step %d",
                 step + 1);
            ok = 0;
        }
    }
    free(video_velocity);
    free(audio_velocity);
    free(initial_video);
    free(initial_audio);
    if (profile_prefix)
        fprintf(stderr,"h3cli: continuation CPU mask and suffix Euler updates %.6f s across %d transitions\n",
                prefix_update_seconds, state->next_step-first_step);
    h3_gpu_profile_mark(dit->gpu, "Euler denoise");
    return ok;
}

int h3_dit_denoise_euler(h3_dit *dit, float *video_latent,
                         float *audio_latent, int reuse_interval,
                         h3_dit_progress progress, void *progress_opaque,
                         char *error, size_t error_size) {
    return h3_dit_denoise_euler_preview(
        dit, video_latent, audio_latent, reuse_interval,
        progress, progress_opaque, NULL, NULL, error, error_size);
}

void h3_dit_free(h3_dit *dit) {
    double teardown_started=stream_now();
    if(dit)h3_profile_memory(dit->gpu,"DiT teardown begin",-1,-1,0);
    #ifdef __APPLE__
    if(dit)h3_ane_split_free(dit->ane);
    #endif
    if (!dit) return;
    h3_gpu_cancel(dit->gpu);
    int steps = h3_dit_schedule_steps(dit->schedule);
    if (dit->row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->row_maps[step]);
    if (dit->reduced_row_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->reduced_row_maps[step]);
    if (dit->final_audio_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_audio_maps[step]);
    if (dit->final_video_maps) for (int step = 0; step < steps; step++)
        h3_gpu_tensor_free(dit->final_video_maps[step]);
    free(dit->row_maps);
    free(dit->reduced_row_maps);
    free(dit->final_audio_maps);
    free(dit->final_video_maps);
    free_tensor(&dit->refined_text);
    free_tensor(&dit->rope_cos);
    free_tensor(&dit->rope_sin);
    free_tensor(&dit->reduced_rope_cos);
    free_tensor(&dit->reduced_rope_sin);
    free_tensor(&dit->video_patch_w); free_tensor(&dit->video_patch_b);
    free_tensor(&dit->audio_patch_w); free_tensor(&dit->audio_patch_b);
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        free_block(&dit->blocks[block]);
    free_block(&dit->stream_slots[0]);
    free_block(&dit->stream_slots[1]);
    free_tensor(&dit->final_norm);
    free_tensor(&dit->final_video_w); free_tensor(&dit->final_video_b);
    free_tensor(&dit->final_audio_w); free_tensor(&dit->final_audio_b);
#define FREE(field) free_tensor(&dit->field)
    if (dit->activation_aliases) {
        dit->attention_heads = NULL;
        dit->mod_mlp = NULL;
    }
    FREE(video_input); FREE(audio_input);
    FREE(video_projected_f32); FREE(audio_projected_f32);
    FREE(video_projected); FREE(audio_projected);
    FREE(video_projection_map); FREE(audio_projection_map); FREE(hidden);
    FREE(core_input); FREE(core_residual);
    FREE(adaptive_anchor); FREE(adaptive_delta); FREE(adaptive_scratch); FREE(adaptive_reduction);
    FREE(mod_attention); FREE(qkv); FREE(query); FREE(key); FREE(value);
    FREE(attention_heads); FREE(attention_output);
    FREE(token_pool_pairs); FREE(token_baseline_indices);
    FREE(token_expand_parents); FREE(token_original); FREE(mod_mlp); FREE(fc1);
    FREE(activated); FREE(mlp_output); FREE(int8_activation);
    FREE(int8_activation_scales); FREE(final_audio_input);
    FREE(final_video_input); FREE(final_audio_inverse);
    FREE(final_video_inverse); FREE(final_audio_norm); FREE(final_video_norm);
    FREE(final_audio_f32); FREE(final_video_f32); FREE(audio_output);
    FREE(video_output);
    FREE(audio_output_bf16); FREE(video_output_bf16);
    FREE(previous_audio_velocity); FREE(previous_video_velocity);
#undef FREE
    h3_dit_schedule_free(dit->schedule);
    if (dit->ssd_streaming && getenv("H3_PROFILE")) {
        double gib = (double)dit->stream_bytes / (1024.0 * 1024.0 * 1024.0);
        fprintf(stderr,
                "h3cli: BF16 SSD stream %.3f GiB read in %.3fs (%.3f GiB/s), "
                "unhidden wait %.3fs\n",
                gib, dit->stream_read_seconds,
                dit->stream_read_seconds > 0.0
                    ? gib / dit->stream_read_seconds : 0.0,
                dit->stream_wait_seconds);
    }
    h3_gpu_free(dit->gpu);
    h3_weight_store_free(dit->weights);
    h3_layout_free(&dit->layout);
    free(dit);
    h3_profile_memory(NULL,"DiT teardown end",-1,-1,stream_now()-teardown_started);
}

static int video_shape(int channels, int time, int height, int width,
                       size_t *latent_count, size_t *row_count) {
    if (channels < 1 || time < 1 || height < 2 || width < 2 ||
        height % 2 || width % 2) return 0;
    size_t c = (size_t)channels, t = (size_t)time;
    size_t h = (size_t)height, w = (size_t)width;
    if (c > SIZE_MAX / t || c * t > SIZE_MAX / h ||
        c * t * h > SIZE_MAX / w) return 0;
    *latent_count = c * t * h * w;
    *row_count = t * (h / 2) * (w / 2) * c * 4;
    return 1;
}

int h3_dit_patchify_video(const float *latent, int channels, int time,
                          int height, int width, float *rows,
                          size_t row_elements) {
    size_t latent_count, expected;
    if (!latent || !rows ||
        !video_shape(channels, time, height, width, &latent_count, &expected) ||
        row_elements != expected || latent_count != expected) return 0;
    size_t output = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t input = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            rows[output++] = latent[input];
                        }
    return output == row_elements;
}

int h3_dit_unpatchify_video(const float *rows, int channels, int time,
                            int height, int width, float *latent,
                            size_t latent_elements) {
    size_t expected, row_count;
    if (!rows || !latent ||
        !video_shape(channels, time, height, width, &expected, &row_count) ||
        latent_elements != expected || row_count != expected) return 0;
    size_t input = 0;
    for (int t = 0; t < time; t++)
        for (int h = 0; h < height; h += 2)
            for (int w = 0; w < width; w += 2)
                for (int c = 0; c < channels; c++)
                    for (int dh = 0; dh < 2; dh++)
                        for (int dw = 0; dw < 2; dw++) {
                            size_t output = (((size_t)c * (size_t)time +
                                (size_t)t) * (size_t)height + (size_t)(h + dh)) *
                                (size_t)width + (size_t)(w + dw);
                            latent[output] = rows[input++];
                        }
    return input == row_count;
}

int h3_dit_pack_audio(const float *latent, int channels, int time,
                      float *rows, size_t row_elements) {
    if (!latent || !rows || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        row_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t output = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t input = ((size_t)channel * 2 + (size_t)stream) *
                               (size_t)time + (size_t)t;
                rows[output++] = latent[input];
            }
    return output == row_elements;
}

int h3_dit_unpack_audio(const float *rows, int channels, int time,
                        float *latent, size_t latent_elements) {
    if (!rows || !latent || channels < 1 || time < 1 ||
        (size_t)channels > SIZE_MAX / (2 * (size_t)time) ||
        latent_elements != (size_t)channels * 2 * (size_t)time) return 0;
    size_t input = 0;
    for (int stream = 0; stream < 2; stream++)
        for (int t = 0; t < time; t++)
            for (int channel = 0; channel < channels; channel++) {
                size_t output = ((size_t)channel * 2 + (size_t)stream) *
                                (size_t)time + (size_t)t;
                latent[output] = rows[input++];
            }
    return input == latent_elements;
}

static int export_bf16(const h3_gpu_tensor *tensor,uint16_t **values,size_t elements,int initialized) {
    if(!*values) *values=calloc(elements,2);
    return *values && (!initialized || h3_gpu_tensor_read_bf16(tensor,*values,elements));
}

int h3_dit_sampler_export(h3_dit *dit,h3_sampler_state *s,int prepared,char *error,size_t size) {
    #ifdef __APPLE__
    h3_ane_split_export(dit->ane,&s->ane_decided,s->ane_rows);
    #endif
    if(!h3_gpu_synchronize(dit->gpu)) { fail(error,size,"sampler export synchronization failed"); return 0; }
    if(!s->next_step) {
        s->sampler_mode=(uint32_t)gpu_sampler_requested(dit);
    }
    s->execution_version=1; s->core_forward_count=dit->core_forward_count;
    s->core_residual_ready=(uint32_t)dit->core_residual_ready;
    s->full_sequence=dit->sequence; s->reduced_sequence=dit->reduced_sequence;
    s->reduction_enabled=(uint32_t)dit->token_reduction; s->reduction_active=(uint32_t)dit->token_reduction_active;
    s->reduction_begin=dit->token_reduction_begin; s->reduction_end=dit->token_reduction_end;
    s->reduction_early_steps=dit->token_reduction_early_steps; s->reduction_early_end=dit->token_reduction_early_end;
    s->reduction_scale=dit->token_reduction_scale;
    if(dit->adaptive_mode) {
        if(dit->adaptive_pending){fail(error,size,"cannot checkpoint an uncommitted adaptive forward");return 0;}
        s->adaptive_version=dit->adaptive_recipe;s->adaptive_history=dit->adaptive_history;
        s->adaptive_elements=dit->adaptive_plan.elements;
        s->params.adaptive_cache_max_bytes=dit->adaptive_budget;
        s->params.adaptive_cache_threshold=dit->adaptive_threshold;
        s->params.adaptive_cache_max_hits=dit->adaptive_max_hits;
        s->params.adaptive_cache_threshold_set=s->params.adaptive_cache_max_hits_set=1;
        if(!export_bf16(dit->adaptive_anchor,&s->adaptive_anchor,s->adaptive_elements,(int)s->adaptive_history.ready)||
           !export_bf16(dit->adaptive_delta,&s->adaptive_delta,s->adaptive_elements,(int)s->adaptive_history.ready))goto failed;
    }
    if(dit->core_reuse_interval>1) {
        s->core_rows=dit->sequence; s->core_columns=HIDDEN; s->core_elements=s->core_rows*HIDDEN;
        if(!export_bf16(dit->core_residual,&s->core_residual,s->core_elements,dit->core_residual_ready)) goto failed;
    }
    if(s->sampler_mode && s->reuse_interval>1) {
        if(!export_bf16(dit->video_output_bf16,&s->gpu_last_video,s->video_elements,s->last_evaluated>=0) ||
            !export_bf16(dit->audio_output_bf16,&s->gpu_last_audio,s->audio_elements,s->last_evaluated>=0) ||
            !export_bf16(dit->previous_video_velocity,&s->gpu_previous_video,s->video_elements,s->previous_evaluated>=0) ||
            !export_bf16(dit->previous_audio_velocity,&s->gpu_previous_audio,s->audio_elements,s->previous_evaluated>=0)) goto failed;
    }
    if(prepared && s->prepared.count==0) {
        s->prepared.version=1;
        if(!h3_sampler_prepared_key(s,s->prepared.key)) goto failed;
        h3_prepared_tensor *t=&s->prepared.tensors[s->prepared.count++];
        t->id=1; t->elements=h3_gpu_tensor_elements(dit->refined_text);
        if(!export_bf16(dit->refined_text,&t->values,t->elements,1) || !h3_dit_schedule_export(dit->schedule,s)) {
            /* Rebuildable acceleration data is never necessary for correctness. */
            h3_prepared_cache_free(&s->prepared);
            fprintf(stderr,"h3cli: prepared cache export unavailable; checkpoint will rebuild it\n");
        }
    }
    return 1;
failed: fail(error,size,"cannot export native sampler/core state"); return 0;
}

int h3_dit_conditioning_export(h3_dit *dit,h3_conditioning *s,char *error,size_t size) {
    if(!dit||!s||s->prepared.count||!h3_gpu_synchronize(dit->gpu)) {
        fail(error,size,"cannot export conditioning from this DiT");return 0;
    }
    h3_prepared_tensor *t=&s->prepared.tensors[s->prepared.count++];
    t->id=1;t->elements=h3_gpu_tensor_elements(dit->refined_text);
    if(!export_bf16(dit->refined_text,&t->values,t->elements,1)) {
        fail(error,size,"cannot export refined conditioning");return 0;
    }
    s->prepared.version=1;
    if(s->has_schedule){h3_sampler_state temp={0};temp.prepared=s->prepared;
        int ok=h3_dit_schedule_export(dit->schedule,&temp);s->prepared=temp.prepared;
        if(!ok){fail(error,size,"cannot export schedule conditioning");return 0;}
    }
    return 1;
}

int h3_dit_sampler_import(h3_dit *dit,const h3_sampler_state *s,char *error,size_t size) {
    if(!h3_gpu_synchronize(dit->gpu)) { fail(error,size,"sampler import synchronization failed"); return 0; }
    uint32_t mode=(uint32_t)gpu_sampler_requested(dit);
    if(mode!=s->sampler_mode || s->full_sequence!=dit->sequence || s->reduced_sequence!=dit->reduced_sequence ||
        s->reduction_enabled!=(uint32_t)dit->token_reduction || s->reduction_active ||
        s->reduction_begin!=dit->token_reduction_begin || s->reduction_end!=dit->token_reduction_end ||
        s->reduction_early_steps!=dit->token_reduction_early_steps || s->reduction_early_end!=dit->token_reduction_early_end ||
        s->reduction_scale!=dit->token_reduction_scale) {
        fail(error,size,"sampler mode or resolved token topology differs from checkpoint"); return 0;
    }
    if(dit->core_reuse_interval>1 && (s->core_rows!=dit->sequence || s->core_columns!=HIDDEN ||
        !s->core_residual || !h3_gpu_tensor_write_bf16(dit->core_residual,s->core_residual,s->core_elements))) {
        fail(error,size,"cannot restore native BF16 core residual"); return 0;
    }
    if(dit->adaptive_mode!=s->params.adaptive_cache||
       (dit->adaptive_mode&&(dit->adaptive_recipe!=s->adaptive_version||
        dit->adaptive_threshold!=s->params.adaptive_cache_threshold||dit->adaptive_max_hits!=s->params.adaptive_cache_max_hits))||
       (dit->adaptive_mode&&dit->adaptive_warmup!=h3_adaptive_warmup(s->params.adaptive_cache_warmup))||
       (dit->captured_attention==H3_ATTENTION_SUBBLOCK&&dit->subblock_warmup!=h3_subblock_warmup(s->params.subblock_warmup))){fail(error,size,"adaptive policy differs from checkpoint");return 0;}
    if(dit->adaptive_mode) {
        if(s->adaptive_elements!=dit->adaptive_plan.elements||!s->adaptive_anchor||!s->adaptive_delta||
           !h3_gpu_tensor_write_bf16(dit->adaptive_anchor,s->adaptive_anchor,s->adaptive_elements)||
           !h3_gpu_tensor_write_bf16(dit->adaptive_delta,s->adaptive_delta,s->adaptive_elements)) {
            fail(error,size,"cannot restore native adaptive state");return 0;
        }
        dit->adaptive_history=s->adaptive_history;dit->adaptive_pending=0;
    }
    dit->core_forward_count=s->core_forward_count; dit->core_residual_ready=(int)s->core_residual_ready;
    dit->token_reduction_active=0;
    return 1;
}
