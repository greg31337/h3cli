#include "src/log.h"
#include "src/sglang/sglang.h"
#include "src/denoise/attention.h"
#include "src/vae/image_vae.h"
#include "src/memory.h"
#include "src/profile.h"
#include "src/internal.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include "src/conditioning/conditioning.h"
#include "src/vae/audio_vae.h"
#include "src/host.h"
#include "src/denoise/dit.h"
#include "src/media/ffmpeg.h"
#include "src/execution.h"
#include "src/weights/quant.h"
#include "src/denoise/adaptive_cache.h"
#include "src/denoise/approximate.h"
#include "src/denoise/subblock.h"
#include "src/weights/q8.h"
#include "src/media/delivery.h"
#include "src/media/refvideo.h"
#include "src/device.h"
#include "src/platform.h"
#include "src/runtime/runtime.h"
#include "src/conditioning/multimodal.h"
#include "src/weights/safetensors.h"
#include "src/conditioning/text_encoder.h"
#include "src/conditioning/tokenizer.h"
#include "src/vae/video_encoder.h"
#include "src/vae/video_vae.h"
#include "src/conditioning/vision_encoder.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static char h3_global_error[512];

typedef struct {
    char *text;
    size_t length;
    size_t capacity;
} h3_key;

static void h3_conditioning_cache_clear(h3_ctx *ctx) {
    if (!ctx) return;
    free(ctx->conditioning_key);
    free(ctx->conditioning_values);
    free(ctx->conditioning_tags);
    h3_text_embedding diagnostics = {0};
    diagnostics.diagnostics = ctx->conditioning_diagnostics;
    h3_text_embedding_free(&diagnostics);
    ctx->conditioning_diagnostics = NULL;
    free(ctx->conditioning_video_rows);
    free(ctx->conditioning_audio_rows);
    free(ctx->conditioning_references);
    ctx->conditioning_key = NULL;
    ctx->conditioning_values = NULL;
    ctx->conditioning_tags = NULL;
    ctx->conditioning_video_rows = NULL;
    ctx->conditioning_audio_rows = NULL;
    ctx->conditioning_references = NULL;
    ctx->conditioning_tokens = 0;
    ctx->conditioning_width = 0;
    ctx->conditioning_video_elements = 0;
    ctx->conditioning_audio_elements = 0;
    ctx->conditioning_reference_count = 0;
    ctx->conditioning_present = 0;
}

void h3_cache_clear(h3_ctx *ctx) {
    if (!ctx) return;
    h3_conditioning_cache_clear(ctx);
    h3_tiny_vae_free(ctx->tiny_decoder); ctx->tiny_decoder = NULL;
    h3_dit_free(ctx->dit);
    ctx->dit = NULL;
    free(ctx->dit_key);
    ctx->dit_key = NULL;
    h3_video_vae_decoder_free(ctx->video_decoder);
    ctx->video_decoder = NULL;
    free(ctx->video_decoder_key);
    ctx->video_decoder_key = NULL;
}

void h3_cache_set_enabled(h3_ctx *ctx, int enabled) {
    if (!ctx) return;
    if (!enabled) h3_cache_clear(ctx);
    ctx->cache_enabled = enabled != 0;
}

void h3_cache_get_info(const h3_ctx *ctx, h3_cache_info *info) {
    if (!info) return;
    memset(info, 0, sizeof(*info));
    if (!ctx) return;
    if (ctx->conditioning_key) {
        info->embedding_entries = 1;
        info->embedding_bytes =
            ctx->conditioning_tokens * ctx->conditioning_width *
                sizeof(*ctx->conditioning_values) +
            ctx->conditioning_tokens * sizeof(*ctx->conditioning_tags) +
            ctx->conditioning_video_elements *
                sizeof(*ctx->conditioning_video_rows) +
            ctx->conditioning_audio_elements *
                sizeof(*ctx->conditioning_audio_rows);
        if (ctx->conditioning_diagnostics) {
            const h3_text_diagnostics *d = ctx->conditioning_diagnostics;
            info->embedding_bytes += sizeof(*d) + d->tokens * sizeof(*d->ids) +
                (d->positions ? 3 * d->tokens * sizeof(*d->positions) : 0) +
                2 * d->span_count * sizeof(*d->spans);
        }
    }
    info->prepared_dit = ctx->dit != NULL;
    info->video_decoder = ctx->video_decoder != NULL || ctx->tiny_decoder != NULL;
}

static int h3_key_append(h3_key *key, const char *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    va_list copy;
    va_copy(copy, arguments);
    int needed = vsnprintf(NULL, 0, format, copy);
    va_end(copy);
    if (needed < 0) {
        va_end(arguments);
        return 0;
    }
    size_t wanted = key->length + (size_t)needed + 1;
    if (wanted > key->capacity) {
        size_t capacity = key->capacity ? key->capacity : 256;
        while (capacity < wanted) {
            if (capacity > SIZE_MAX / 2) {
                va_end(arguments);
                return 0;
            }
            capacity *= 2;
        }
        char *grown = realloc(key->text, capacity);
        if (!grown) {
            va_end(arguments);
            return 0;
        }
        key->text = grown;
        key->capacity = capacity;
    }
    vsnprintf(key->text + key->length, key->capacity - key->length,
              format, arguments);
    va_end(arguments);
    key->length += (size_t)needed;
    return 1;
}

static int h3_key_file(h3_key *key, const char *role, const char *path) {
    if (!path) return h3_key_append(key, "|%s=none", role);
    struct stat status;
    if (stat(path, &status) != 0)
        return h3_key_append(key, "|%s=%zu:%s:missing", role,
                             strlen(path), path);
    return h3_key_append(key, "|%s=%zu:%s:%lld:%lld:%ld", role, strlen(path),
                         path, (long long)status.st_size,
                         (long long)h3_stat_mtime(&status).tv_sec,
                         h3_stat_mtime(&status).tv_nsec);
}

static int h3_key_runtime(h3_key *key,const char *role,const char *name) {
    const char *identity=h3_runtime_dependency_id(name);
    return identity?h3_key_append(key,"|%s=%s",role,identity):h3_key_file(key,role,getenv(name));
}

/* Choices captured by backend constructors belong to every prepared cache key. */
static int h3_key_execution(h3_key *key, const h3_params *params) {
    if (!h3_key_append(key, "|cuda-pipeline=1|base=%d|attention=%d|precision=%d",
        params->_arithmetic_recipe,params->cuda_attention==H3_ATTENTION_SUBBLOCK?0:params->cuda_attention,params->cuda_denoise_quant)) return 0;
    const char *runtime=h3_runtime_dependency_id("H3_PACKAGED_RUNTIME");
    if(runtime&&!h3_key_append(key,"|packaged-runtime=%s",runtime))return 0;
    if(params->_arithmetic_recipe) {
        if(!h3_key_append(key,"|sglang-reference=v%d",H3_SGLANG_VERSION) ||
           !h3_key_runtime(key,"sglang-cublas","H3_SGLANG_CUBLAS_LIBRARY") ||
           !h3_key_runtime(key,"sglang-cudnn","H3_SGLANG_CUDNN_LIBRARY") ||
           !h3_key_runtime(key,"sglang-jpeg","H3_SGLANG_JPEG_LIBRARY") ||
           !h3_key_runtime(key,"sglang-input-ffmpeg","H3_SGLANG_INPUT_FFMPEG"))return 0;
    }
    return 1;
}

char *h3_conditioning_key(const char *prompt, const h3_params *params,
                                 int render_width, int render_height,
                                 int ref2va) {
    h3_key key = {0};
    h3_qwen_gqa_scale_mode scale_mode;
    if (!h3_qwen_gqa_scale_parse(getenv("H3_QWEN_GQA_SCALE_MODE"),
                                 &scale_mode, NULL, 0)) return NULL;
    if (params->still && !h3_key_append(&key,"operation=still-v1:audio2-generated-discarded|")) goto failed;
    if (!h3_key_append(&key, "mode=%d|prompt=%zu:%s", ref2va,
                       strlen(prompt), prompt)) goto failed;
    if (!h3_key_execution(&key,params)) goto failed;
    if (!h3_key_append(&key, "|qwen-scale=%s",
                       h3_qwen_gqa_scale_name(scale_mode))) goto failed;
    if (!ref2va && !params->first_frame && !params->last_frame) return key.text;
    if (params->first_frame &&
        !h3_key_append(&key, "|first-frame-fit=cover-v1")) goto failed;
    if (!h3_key_append(&key, "|render=%dx%d|frames=%d|image-size=%d",
                       render_width, render_height, params->frames,
                       params->reference_image_size) ||
        !h3_key_file(&key, "first", params->first_frame) ||
        !h3_key_file(&key, "last", params->last_frame)) goto failed;
    if (h3_conditioning_image_geometry(params) &&
        !h3_key_append(&key,"|reference-image-geometry=%d",
            h3_conditioning_image_geometry(params))) goto failed;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (ref2va && (reference->kind == H3_REFERENCE_VIDEO ||
                       reference->kind == H3_REFERENCE_VIDEO_AUDIO) &&
            !h3_key_append(&key, "|ref2va-video-pipeline=%s",
                "released-v1")) goto failed;
        if (!h3_key_append(&key, "|ref=%d:%d", reference->kind,
                           reference->include_embedded_audio) ||
            !h3_key_file(&key, "media", reference->path) ||
            !h3_key_file(&key, "audio", reference->audio_path)) goto failed;
    }
    return key.text;
failed:
    free(key.text);
    return NULL;
}

char *h3_prepared_key(const char *conditioning,
                             const h3_params *params,
                             int render_width, int render_height) {
    h3_key key = {0};
    if((params->adaptive_cache||params->cuda_attention==H3_ATTENTION_SUBBLOCK) &&
       !h3_key_append(&key,"adaptive=%d:v%u|subblock=%.9g:v1|",params->adaptive_cache,
           params->adaptive_cache?h3_adaptive_execution_recipe(params->adaptive_cache,params->cuda_denoise_quant,params->reference_count!=0,params->continuation!=NULL):1u,
           (double)params->subblock_sparsity))return NULL;
    if((params->adaptive_cache||params->cuda_attention==H3_ATTENTION_SUBBLOCK)&&
       !h3_key_append(&key,"warmup=%d/%d|",params->adaptive_cache?h3_adaptive_warmup(params->adaptive_cache_warmup):0,
           params->cuda_attention==H3_ATTENTION_SUBBLOCK?h3_subblock_warmup(params->subblock_warmup):0)){free(key.text);return NULL;}
    if(params->adaptive_cache&&!h3_key_append(&key,"adaptive-controls=%a/%d|",
       (double)h3_adaptive_threshold(params),h3_adaptive_max_hits(params))){free(key.text);return NULL;}
    const char *bench_blocks=getenv("H3_TEST_DIT_BLOCKS");
    if(bench_blocks&&!h3_key_append(&key,"test-blocks=%s|",bench_blocks))return NULL;
    if (!h3_key_append(
            &key,
            "%s|shape=%dx%dx%d|steps=%d|layers=%d|reuse-core=%d|reduce=%d"
            "|row-fc2=%d|reference-rope=%d|ssd-streaming=%d"
            "|slow=%d%d%d%d%d%d%d%d%d%d|prefix=%d",
            conditioning, render_width, render_height, params->frames,
            params->steps, params->dit_layers, params->core_reuse,
            params->token_reduction, params->use_int8_row_fc2,
            params->use_reference_rope,
            params->ssd_streaming,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            params->continuation ? (params->continuation_context_frames ?
                params->continuation_context_frames : 39) : 0)) {
        free(key.text);
        return NULL;
    }
    if (params->cuda_attention==H3_ATTENTION_SOL) {
        h3_cuda_sol_options o=params->cuda_sol;
        if(!h3_key_append(&key,"|cuda-sol=v%d/p%d/q%d/k%d/l%d/s%d/r%d/t%a/f%a/n%a",
            H3_CUDA_SOL_VERSION,H3_CUDA_SOL_PLAN_VERSION,o.q_block,o.kv_block,o.dense_layers,o.dense_steps,o.local_radius,
            (double)o.tau,(double)o.min_exact,(double)o.dense_sigma)){free(key.text);return NULL;}
    }
    if (params->cuda_attention && !h3_key_append(&key,"|attention=%d/v%d/p%d",
        params->cuda_attention,h3_attention_execution_recipe(params->cuda_attention,params->cuda_denoise_quant),H3_ATTENTION_PLAN_VERSION)) {free(key.text);return NULL;}
    if(params->backend) {
        h3_metal_attention_options m=params->metal_attention;
        /* Preserve the version-6 native marker in existing conditioning keys. */
        if(!h3_key_append(&key,"|backend=%d/attention=%d/v%d/%d/%d/%d/%d/%d/%d/%a/%a/dtype%d/tier%d/mixedv%d/solv%d/densesteps%d/sigma%a/layout%d/v%d",
            params->backend,params->attention_mode,H3_METAL_ATTENTION_VERSION,
            1,m.candidate,m.q_block,m.kv_block,m.dense_layers,m.local_radius,
            (double)m.tau,(double)m.min_exact,m.precision,m.tier,m.precision?H3_METAL_FP16_VERSION:0,
            params->attention_mode==H3_ATTN_SOL?H3_METAL_SOL_VERSION:0,m.dense_steps,(double)m.dense_sigma,m.layout_fusion,m.layout_fusion?H3_METAL_LAYOUT_VERSION:0)){free(key.text);return NULL;}
        if(!h3_key_append(&key,"/anev%d/mode%d/rows%d/chunk%d",H3_METAL_ANE_VERSION,m.ane_mode,m.ane_rows,m.ane_chunk)){free(key.text);return NULL;}
        if(m.weight_format&&!h3_key_append(&key,"/weight%d/v%d/group%d/kernel%d",m.weight_format,H3_Q8_VERSION,H3_Q8_GROUP,m.q8_kernel)){free(key.text);return NULL;}
    }
    if (params->cuda_denoise_quant && !h3_key_append(&key,"|denoise-quant=%d/v%d/verify%d",
        params->cuda_denoise_quant,h3_quant_execution_recipe(params->cuda_denoise_quant,params->adaptive_cache,params->cuda_attention),h3_quant_verify())) {free(key.text);return NULL;}
    if (params->continuation && params->continuation_mode == H3_CONTINUE_BRIDGE &&
        !h3_key_append(&key, "|bridge=%d:%a:%d", params->bridge_video_steps,
            (double)params->bridge_max_strength, params->bridge_profile)) {
        free(key.text); return NULL;
    }
    if(params->continuation&&(params->adaptive_cache||params->cuda_attention==H3_ATTENTION_SUBBLOCK)) {
        uint8_t source[32];h3_av_state_fingerprint(params->continuation,source);
        if(!h3_key_append(&key,"|continuation-source=")){free(key.text);return NULL;}
        for(size_t i=0;i<sizeof(source);i++)if(!h3_key_append(&key,"%02x",source[i])){free(key.text);return NULL;}
    }
    return key.text;
}

static int h3_text_embedding_copy(h3_text_embedding *destination,
                                  const h3_text_embedding *source) {
    memset(destination, 0, sizeof(*destination));
    if (!source || !source->tokens || !source->width || !source->values ||
        source->tokens > SIZE_MAX / source->width) return 0;
    size_t elements = source->tokens * source->width;
    if (elements > SIZE_MAX / sizeof(*destination->values)) return 0;
    destination->values = malloc(elements * sizeof(*destination->values));
    if (source->tags)
        destination->tags = malloc(source->tokens * sizeof(*destination->tags));
    if (!destination->values || (source->tags && !destination->tags)) {
        h3_text_embedding_free(destination);
        return 0;
    }
    memcpy(destination->values, source->values,
           elements * sizeof(*destination->values));
    if (source->tags)
        memcpy(destination->tags, source->tags,
               source->tokens * sizeof(*destination->tags));
    destination->tokens = source->tokens;
    destination->width = source->width;
    destination->gpu_stats = source->gpu_stats;
    /* Cache hits must preserve checkpoint presentation diagnostics as well as
     * numerical embeddings; otherwise display-only reruns change file bytes. */
    if (source->diagnostics) {
        const h3_text_diagnostics *s = source->diagnostics;
        h3_text_diagnostics *d = calloc(1, sizeof(*d));
        destination->diagnostics = d;
        if (!d || s->tokens != source->tokens || !s->ids ||
            s->tokens > SIZE_MAX / (3 * sizeof(uint32_t)) ||
            s->span_count > SIZE_MAX / (2 * sizeof(uint64_t)) ||
            (s->span_count && !s->spans)) goto diagnostic_failure;
        d->tokens = s->tokens; d->span_count = s->span_count;
        d->ids = malloc(s->tokens * sizeof(*d->ids));
        if (s->positions) d->positions = malloc(3 * s->tokens * sizeof(*d->positions));
        if (s->span_count) d->spans = malloc(2 * s->span_count * sizeof(*d->spans));
        if (!d->ids || (s->positions && !d->positions) ||
            (s->span_count && !d->spans)) goto diagnostic_failure;
        memcpy(d->ids, s->ids, s->tokens * sizeof(*d->ids));
        if (s->positions) memcpy(d->positions, s->positions, 3 * s->tokens * sizeof(*d->positions));
        if (s->span_count) memcpy(d->spans, s->spans, 2 * s->span_count * sizeof(*d->spans));
    }
    return 1;
diagnostic_failure:
    h3_text_embedding_free(destination);
    return 0;
}

static int h3_conditioning_cache_store(
        h3_ctx *ctx, const char *key, const h3_text_embedding *text,
        const float *video, size_t video_elements,
        const float *audio, size_t audio_elements,
        const h3_layout_ref *references, size_t reference_count,
        int conditioned) {
    h3_text_embedding copy;
    if (!h3_text_embedding_copy(&copy, text)) return 0;
    float *video_copy = NULL;
    float *audio_copy = NULL;
    h3_layout_ref *reference_copy = NULL;
    char *key_copy = strdup(key);
    if (video_elements) {
        video_copy = malloc(video_elements * sizeof(*video_copy));
        if (video_copy) memcpy(video_copy, video,
                               video_elements * sizeof(*video_copy));
    }
    if (audio_elements) {
        audio_copy = malloc(audio_elements * sizeof(*audio_copy));
        if (audio_copy) memcpy(audio_copy, audio,
                               audio_elements * sizeof(*audio_copy));
    }
    if (reference_count) {
        reference_copy = malloc(reference_count * sizeof(*reference_copy));
        if (reference_copy) memcpy(reference_copy, references,
                                   reference_count * sizeof(*reference_copy));
    }
    if (!key_copy || (video_elements && !video_copy) ||
        (audio_elements && !audio_copy) ||
        (reference_count && !reference_copy)) {
        free(key_copy); free(video_copy); free(audio_copy); free(reference_copy);
        h3_text_embedding_free(&copy);
        return 0;
    }
    h3_conditioning_cache_clear(ctx);
    ctx->conditioning_key = key_copy;
    ctx->conditioning_tokens = copy.tokens;
    ctx->conditioning_width = copy.width;
    ctx->conditioning_values = copy.values;
    ctx->conditioning_tags = copy.tags;
    ctx->conditioning_diagnostics = copy.diagnostics;
    ctx->conditioning_video_rows = video_copy;
    ctx->conditioning_video_elements = video_elements;
    ctx->conditioning_audio_rows = audio_copy;
    ctx->conditioning_audio_elements = audio_elements;
    ctx->conditioning_references = reference_copy;
    ctx->conditioning_reference_count = reference_count;
    ctx->conditioning_present = conditioned;
    return 1;
}

static int h3_conditioning_cache_load(
        const h3_ctx *ctx, h3_text_embedding *text,
        float **video, size_t *video_elements,
        float **audio, size_t *audio_elements,
        h3_layout_ref **references, size_t *reference_count,
        int *conditioned) {
    h3_text_embedding source = {
        ctx->conditioning_tokens, ctx->conditioning_width,
        ctx->conditioning_values, {0}, ctx->conditioning_tags, ctx->conditioning_diagnostics};
    if (!h3_text_embedding_copy(text, &source)) return 0;
    *video = NULL;
    *audio = NULL;
    *references = NULL;
    *video_elements = ctx->conditioning_video_elements;
    *audio_elements = ctx->conditioning_audio_elements;
    *reference_count = ctx->conditioning_reference_count;
    *conditioned = ctx->conditioning_present;
    if (*video_elements) {
        *video = malloc(*video_elements * sizeof(**video));
        if (*video) memcpy(*video, ctx->conditioning_video_rows,
                           *video_elements * sizeof(**video));
    }
    if (*audio_elements) {
        *audio = malloc(*audio_elements * sizeof(**audio));
        if (*audio) memcpy(*audio, ctx->conditioning_audio_rows,
                           *audio_elements * sizeof(**audio));
    }
    if (*reference_count) {
        *references = malloc(*reference_count * sizeof(**references));
        if (*references) memcpy(*references, ctx->conditioning_references,
                                *reference_count * sizeof(**references));
    }
    if ((*video_elements && !*video) || (*audio_elements && !*audio) ||
        (*reference_count && !*references)) {
        h3_text_embedding_free(text);
        free(*video); free(*audio); free(*references);
        *video = NULL; *audio = NULL; *references = NULL;
        return 0;
    }
    return 1;
}

static void h3_augment_span(float *values, size_t count, uint64_t seed) {
    h3_rng rng;
    h3_rng_seed(&rng, seed);
    for (size_t index = 0; index < count; index++)
        values[index] = 0.999f * values[index] +
                        0.001f * h3_rng_normal(&rng);
}

int h3_augment_conditions(const h3_params *params, int ref2va,
                                 int render_width, int render_height,
                                 const h3_layout_ref *references,
                                 float *video, size_t video_elements,
                                 float *audio, size_t audio_elements) {
    size_t video_offset = 0;
    size_t audio_offset = 0;
    if (!ref2va) {
        int latent_w, latent_h;
        h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
        size_t span = (size_t)h3_video_encoder_latent_t(1) *
                      (size_t)latent_h * (size_t)latent_w / 4 * 96;
        size_t count = (size_t)(params->first_frame != NULL) +
                       (size_t)(params->last_frame != NULL);
        if (count && (span > SIZE_MAX / count || span * count != video_elements))
            return 0;
        for (size_t index = 0; index < count; index++) {
            if(params->_arithmetic_recipe) {
                if(!h3_sglang_video_condition(video+video_offset,h3_video_encoder_latent_t(1),latent_h,latent_w,
                    h3_temporal(params->frames).video_t,(int)count,params->seed))return 0;
            } else h3_augment_span(video + video_offset, span, params->seed);
            video_offset += span;
        }
        return video_offset == video_elements && audio_elements == 0;
    }
    int visual_conditions=0;
    if(params->_arithmetic_recipe)for(size_t index=0;index<params->reference_count;index++)
        visual_conditions+=references[index].kind!=H3_LAYOUT_REF_AUDIO;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_layout_ref *reference = &references[index];
        if (reference->kind != H3_LAYOUT_REF_AUDIO) {
            size_t span = (size_t)reference->latent_t *
                (size_t)reference->latent_h * (size_t)reference->latent_w /
                4 * 96;
            if (span > video_elements - video_offset) return 0;
            if(params->_arithmetic_recipe) {
                if(!h3_sglang_video_condition(video+video_offset,reference->latent_t,reference->latent_h,
                    reference->latent_w,h3_temporal(params->frames).video_t,visual_conditions,params->seed))return 0;
            } else h3_augment_span(video + video_offset, span, params->seed);
            video_offset += span;
        }
        if (reference->audio_t) {
            size_t span = (size_t)reference->audio_t * 2 * 32;
            if (span > audio_elements - audio_offset) return 0;
            if(params->_arithmetic_recipe) {
                if(!h3_sglang_audio_condition(audio+audio_offset,span,params->seed))return 0;
            } else h3_augment_span(audio + audio_offset, span, params->seed + 1);
            audio_offset += span;
        }
    }
    return video_offset == video_elements && audio_offset == audio_elements;
}

void h3_set_error(h3_ctx *ctx, const char *format, ...) {
    char *destination = ctx ? ctx->error : h3_global_error;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(destination, 512, format, arguments);
    va_end(arguments);
}

static int h3_is_file(const char *path) {
    struct stat status;
    return stat(path, &status) == 0 && S_ISREG(status.st_mode);
}

static char *h3_path(const char *root, const char *relative) {
    size_t size = strlen(root) + strlen(relative) + 2;
    char *result = malloc(size);
    if (result) snprintf(result, size, "%s/%s", root, relative);
    return result;
}

static int h3_require_file(h3_ctx *ctx, const char *relative) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving model path");
        return 0;
    }
    int exists = h3_is_file(path);
    if (!exists) h3_set_error(ctx, "missing required model file: %s", path);
    free(path);
    return exists;
}

static int h3_inventory(h3_ctx *ctx, const char *relative,
                        h3_component_info *info) {
    char *path = h3_path(ctx->model_dir, relative);
    if (!path) {
        h3_set_error(ctx, "out of memory resolving component path");
        return 0;
    }
    char detail[384];
    int ok = h3_st_inventory_dir(path, info, detail, sizeof(detail));
    if (!ok) h3_set_error(ctx, "%s", detail);
    free(path);
    return ok;
}

h3_ctx *h3_load_dir(const char *model_dir) {
    return h3_load_dir_with_lora(model_dir,NULL);
}
h3_ctx *h3_load_dir_with_lora(const char *model_dir,const h3_lora_options *options) {
    h3_global_error[0] = '\0';
    if (!model_dir || !*model_dir) {
        h3_set_error(NULL, "model directory is required");
        return NULL;
    }
    h3_ctx *ctx = calloc(1, sizeof(*ctx));
    if (!ctx) {
        h3_set_error(NULL, "out of memory creating H3 context");
        return NULL;
    }
    ctx->lora_variants[0].lease=ctx->lora_variants[1].lease=-1;
    if(!h3_lora_selection_create(options,&ctx->lora,h3_global_error,sizeof(h3_global_error))) { free(ctx); return NULL; }
    ctx->model_dir = strdup(model_dir);
    if (!ctx->model_dir) {
        h3_set_error(NULL, "out of memory copying model path");
        h3_lora_selection_free(ctx->lora); free(ctx);
        return NULL;
    }
    /* Offline LoRA assembly may intentionally contain just one mode. Startup
     * inventories available weights; the request selects its required mode. */
    char *fl_config=h3_path(ctx->model_dir,"FL2VA/transformer/config.json");
    char *ref_config=h3_path(ctx->model_dir,"Ref2VA/transformer/config.json");
    char *fl_tokenizer=h3_path(ctx->model_dir,"FL2VA/tokenizer/tokenizer.json");
    char *ref_tokenizer=h3_path(ctx->model_dir,"Ref2VA/tokenizer/tokenizer.json");
    /* Decode-only installs contain identity configs and VAEs, without a
     * generation checkpoint. Do not choose them for broad startup inventory. */
    int has_fl2va=fl_config && h3_is_file(fl_config) && fl_tokenizer && h3_is_file(fl_tokenizer);
    int has_ref2va=ref_config && h3_is_file(ref_config) && ref_tokenizer && h3_is_file(ref_tokenizer);
    int paths_ok=fl_config && ref_config && fl_tokenizer && ref_tokenizer;
    free(fl_tokenizer);free(ref_tokenizer);
    free(fl_config);free(ref_config);
    int ok=paths_ok;
    if(!paths_ok)h3_set_error(ctx,"out of memory resolving model modes");
    else if(!has_fl2va && !has_ref2va) {
        h3_set_error(ctx,"missing required model file: expected FL2VA/transformer/config.json or Ref2VA/transformer/config.json in %s",ctx->model_dir);ok=0;
    }
    const char *primary=has_fl2va?"FL2VA":"Ref2VA";
    char relative[96];
    if(ok) {
        snprintf(relative,sizeof(relative),"%s/tokenizer/tokenizer.json",primary);
        ok=h3_require_file(ctx,relative);
    }
    const char *components[]={"text_encoder","video_vae/source","audio_vae"};
    h3_component_info *inventories[]={&ctx->model.text_encoder,&ctx->model.video_vae,&ctx->model.audio_vae};
    for(size_t i=0;ok && i<3;i++) {
        snprintf(relative,sizeof(relative),"%s/%s",primary,components[i]);
        ok=h3_inventory(ctx,relative,inventories[i]);
    }
    if(ok && has_fl2va)ok=h3_inventory(ctx,"FL2VA/transformer",&ctx->model.fl2va_transformer);
    if(ok && has_ref2va)ok=h3_inventory(ctx,"Ref2VA/transformer",&ctx->model.ref2va_transformer);
    if(!ok) {
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    char device_error[256];
    if (!h3_device_query(&ctx->device, device_error, sizeof(device_error))) {
        h3_set_error(ctx, "%s", device_error);
        snprintf(h3_global_error, sizeof(h3_global_error), "%s", ctx->error);
        h3_free(ctx);
        return NULL;
    }
    return ctx;
}

void h3_free(h3_ctx *ctx) {
    if (!ctx) return;
    h3_cache_clear(ctx);
    h3_lora_variant_free(&ctx->lora_variants[0]);
    h3_lora_variant_free(&ctx->lora_variants[1]);
    h3_lora_selection_free(ctx->lora);
    free(ctx->model_dir);
    free(ctx);
}

const char *h3_last_error(const h3_ctx *ctx) {
    return ctx ? ctx->error : h3_global_error;
}

const h3_device_info *h3_device(const h3_ctx *ctx) {
    return ctx ? &ctx->device : NULL;
}

const h3_model_info *h3_model(const h3_ctx *ctx) {
    return ctx ? &ctx->model : NULL;
}

int h3_request_params_valid(const h3_params *params, const h3_device_info *device,
                            char *error, size_t error_size) {
    if (!params) {
        snprintf(error,error_size, "generation parameters are required");
        return 0;
    }
    if (!h3_still_options(params,error,error_size)) return 0;
    if (!h3_output_encoding_valid(&params->output_encoding,error,error_size)) return 0;
    if (h3_output_encoding_selected(&params->output_encoding) &&
        (params->still || params->state_only)) {
        snprintf(error,error_size,"video encoding options require video output");
        return 0;
    }
    if (params->width < 32 || params->height < 32 ||
        params->width % H3_CANVAS_MULTIPLE ||
        params->height % H3_CANVAS_MULTIPLE) {
        snprintf(error,error_size, "width and height must be multiples of 32 and at least 32");
        return 0;
    }
    if (!h3_geometry_valid(params->width,params->height,params->geometry_profile)) {
        snprintf(error,error_size, "canvas exceeds the released 768*1344 pixel limit");
        return 0;
    }
    char tile_error[512];
    if (!h3_video_vae_tile_pixels(params->height, params->width,
                                  tile_error, sizeof(tile_error))) {
        snprintf(error,error_size, "%s", tile_error);
        return 0;
    }
    if ((params->render_width == 0) != (params->render_height == 0)) {
        snprintf(error,error_size, "render width and height must be set together");
        return 0;
    }
    if (params->render_width) {
        if (params->render_width < 32 || params->render_height < 32 ||
            params->render_width % H3_CANVAS_MULTIPLE ||
            params->render_height % H3_CANVAS_MULTIPLE ||
            params->render_width > params->width ||
            params->render_height > params->height ||
            (int64_t)params->render_width * params->height !=
                (int64_t)params->render_height * params->width) {
            snprintf(error,error_size,
                "internal render canvas must be same-aspect multiples of 32 "
                "no larger than the output canvas");
            return 0;
        }
    }
    if (!params->still && (params->frames < 5 || params->frames > 362 || h3_align_frame_count(params->frames) > 362)) {
        snprintf(error,error_size, "frames must align within the released 5..362 range");
        return 0;
    }
    if (params->steps < 1 || params->steps > H3_MAX_STEPS) {
        snprintf(error,error_size, "denoising steps must be in [1, 1000]");
        return 0;
    }
    if (params->denoise_reuse < 1 || params->denoise_reuse > 3) {
        snprintf(error,error_size, "denoise reuse must be in [1, 3]");
        return 0;
    }
    if (params->dit_layers < H3_MIN_DIT_LAYERS ||
        params->dit_layers > H3_DEFAULT_DIT_LAYERS) {
        snprintf(error,error_size, "DiT layers must be in [35, 50]");
        return 0;
    }
    if (params->core_reuse < 1 || params->core_reuse > 6) {
        snprintf(error,error_size, "core reuse must be in [1, 6]");
        return 0;
    }
    if (params->token_reduction != 0 && params->token_reduction != 1) {
        snprintf(error,error_size, "token reduction must be zero or one");
        return 0;
    }
    if (params->use_int8_row_fc2 != 0 &&
        params->use_int8_row_fc2 != 1) {
        snprintf(error,error_size, "int8 row FC2 must be zero or one");
        return 0;
    }
    if (params->use_reference_rope != 0 &&
        params->use_reference_rope != 1) {
        snprintf(error,error_size, "reference RoPE must be zero or one");
        return 0;
    }
    if (params->ssd_streaming != 0 && params->ssd_streaming != 1) {
        snprintf(error,error_size, "SSD streaming must be zero or one");
        return 0;
    }
    if (params->ssd_streaming && params->use_int8_row_fc2) {
        snprintf(error,error_size, "SSD streaming uses original BF16 weights and cannot "
                         "be combined with int8 row FC2");
        return 0;
    }
    if (params->use_int8_row_fc2 && params->use_slower_bf16_mlp) {
        snprintf(error,error_size, "int8 row FC2 cannot be combined with the BF16 MLP");
        return 0;
    }
    if (params->use_int8_row_fc2 && device && !device->metal4) {
        snprintf(error,error_size, "int8 row FC2 requires an M5-class Metal 4 GPU");
        return 0;
    }
    if (params->preview_denoise != 0 && params->preview_denoise != 1) {
        snprintf(error,error_size, "denoising preview must be zero or one");
        return 0;
    }
    if (params->preview_denoise && !params->on_frame) {
        snprintf(error,error_size, "denoising preview requires a frame callback");
        return 0;
    }
    if (params->core_reuse > 1 && params->denoise_reuse > 1) {
        snprintf(error,error_size, "core reuse and denoiser reuse cannot be combined");
        return 0;
    }
    if (params->keep_continuation_prefix != 0 && params->keep_continuation_prefix != 1) {
        snprintf(error,error_size,"keep_continuation_prefix must be zero or one"); return 0;
    }
    if (!params->continuation && params->keep_continuation_prefix) {
        snprintf(error,error_size,"keeping a continuation prefix requires a continuation state"); return 0;
    }
    if (params->continuation_mode != H3_CONTINUE_HARD &&
        params->continuation_mode != H3_CONTINUE_BRIDGE) {
        snprintf(error,error_size,"unknown continuation mode; expected hard or bridge"); return 0;
    }
    if (params->continuation_mode == H3_CONTINUE_BRIDGE && !params->continuation) {
        snprintf(error,error_size,"bridge continuation requires a continuation state"); return 0;
    }
    if (params->continuation) {
        if (params->first_frame || params->last_frame) {
            snprintf(error,error_size,"continuation cannot be combined with first/last-frame anchors"); return 0;
        }
        const char *reduction = getenv("H3_TOKEN_REDUCTION");
        if ((params->continuation_mode == H3_CONTINUE_HARD && params->core_reuse != 1) ||
            params->dit_layers != 50 || params->token_reduction ||
            (reduction && *reduction && strcmp(reduction,"0"))) {
            snprintf(error,error_size,params->continuation_mode == H3_CONTINUE_BRIDGE ?
                "bridge continuation requires --layers 50 and token reduction off" :
                "continuation requires --core-reuse 1 --layers 50 and token reduction off"); return 0;
        }
        h3_denoise_prefix checked_prefix;
        if (!h3_av_state_validate_continuation(params->continuation,
            params->render_width ? params->render_width : params->width,
            params->render_height ? params->render_height : params->height,
            h3_align_frame_count(params->frames),
            params->continuation_context_frames ? params->continuation_context_frames : 39,
            NULL,&checked_prefix,error,error_size)) return 0;
        if (params->continuation_mode == H3_CONTINUE_BRIDGE) {
            h3_bridge_profile bridge;
            char warning[256];
            if (!h3_bridge_profile_build(params->continuation_context_frames ?
                params->continuation_context_frames : 39, params->bridge_video_steps,
                params->bridge_max_strength, params->bridge_profile, &bridge,
                warning, sizeof(warning), error,error_size)) return 0;
            if (*warning) fprintf(stderr, "h3cli: warning: %s\n", warning);
            if (params->core_reuse != 1 && params->core_reuse != 4 && params->core_reuse != 6) {
                snprintf(error,error_size,"bridge continuation supports --core-reuse 1, 4, or 6; other intervals are not validated"); return 0;
            }
            const char *custom_reuse = getenv("H3_REUSE_STEPS");
            if (params->denoise_reuse > 1 && custom_reuse && *custom_reuse) {
                snprintf(error,error_size,"bridge continuation does not support custom H3_REUSE_STEPS schedules"); return 0;
            }
        }
    }
    if (params->reference_count && !params->references) {
        snprintf(error,error_size, "reference_count is nonzero but references is NULL");
        return 0;
    }
    if (params->reference_count > 12) {
        snprintf(error,error_size, "Ref2VA supports at most 12 references");
        return 0;
    }
    if (params->reference_image_size != H3_REFERENCE_IMAGE_MATCH &&
        params->reference_image_size != H3_REFERENCE_IMAGE_MAX &&
        params->reference_image_size != H3_REFERENCE_IMAGE_HIGH) {
        snprintf(error,error_size, "unknown reference image sizing policy");
        return 0;
    }
    if (params->reference_count && (params->first_frame || params->last_frame)) {
        snprintf(error,error_size, "full references cannot be combined with frame anchors");
        return 0;
    }
    size_t images = 0, videos = 0, audio_inputs = 0, visual = 0;
    for (size_t index = 0; index < params->reference_count; index++) {
        const h3_reference *reference = &params->references[index];
        if (!reference->path || !*reference->path) {
            snprintf(error,error_size, "reference %zu has no input path", index + 1);
            return 0;
        }
        switch (reference->kind) {
        case H3_REFERENCE_IMAGE:
            images++; visual++;
            break;
        case H3_REFERENCE_VIDEO:
            videos++; visual++;
            if (reference->include_embedded_audio) audio_inputs++;
            break;
        case H3_REFERENCE_AUDIO:
            audio_inputs++;
            break;
        case H3_REFERENCE_VIDEO_AUDIO:
            videos++; visual++; audio_inputs++;
            if (!reference->audio_path || !*reference->audio_path) {
                snprintf(error,error_size,
                    "video+audio reference %zu has no soundtrack path",
                    index + 1);
                return 0;
            }
            break;
        default:
            snprintf(error,error_size, "reference %zu has an unknown type", index + 1);
            return 0;
        }
    }
    if (images > 9 || videos > 3 || audio_inputs > 3) {
        snprintf(error,error_size,
            "Ref2VA limits are 9 images, 3 videos, and 3 audio inputs");
        return 0;
    }
    if (params->reference_count && !visual) {
        snprintf(error,error_size, "reference audio requires an image or video reference");
        return 0;
    }
    return 1;
}

typedef struct {
    h3_ctx *ctx;
    const h3_params *params;
    int cancelled;
    char cancellation_error[512];
    double vae_load_started;
} h3_generation_progress;

static int h3_progress_emit(h3_generation_progress *state, const char *phase,
                             int completed, int total) {
    if (!state) return 0;
    if (state->cancelled) return 1;
    if (!h3_memory_check(0, phase, state->cancellation_error,
                         sizeof(state->cancellation_error))) state->cancelled = 1;
    /* Notify the user even when this boundary detects low memory. Preserve the
     * first diagnostic when both memory and the user request cancellation. */
    if (state->params->on_progress && state->params->on_progress(
            phase, completed, total, state->params->callback_opaque)) {
        if (!state->cancelled)
            snprintf(state->cancellation_error, sizeof(state->cancellation_error),
                     "generation cancelled during %s", phase);
        state->cancelled = 1;
    }
    if (state->cancelled) h3_set_error(state->ctx, "%s", state->cancellation_error);
    return state->cancelled;
}

static int h3_text_progress_bridge(int completed, int total, void *opaque) {
    return h3_progress_emit(opaque, "text encoder", completed, total);
}

static int h3_dit_progress_bridge(const char *phase, int completed, int total,
                                   void *opaque) {
    return h3_progress_emit(opaque, phase, completed, total);
}

static int h3_delivery_frame_bridge(const h3_frame *frame, void *opaque) {
    h3_generation_progress *p=opaque;
    return p->params->on_frame && p->params->on_frame(frame,p->params->callback_opaque);
}

static int h3_vae_progress_bridge(int completed, int total, void *opaque) {
    h3_generation_progress *p=opaque;
    if(p&&completed==0&&p->vae_load_started==0) {
        p->vae_load_started=h3_av_now();h3_profile_memory(NULL,"VideoVAE load begin",-1,-1,0);
    }
    if(p&&total>0&&completed==total&&p->vae_load_started!=0) {
        h3_profile_memory(NULL,"VideoVAE load end",-1,-1,h3_av_now()-p->vae_load_started);
        p->vae_load_started=0;
    }
    return h3_progress_emit(opaque, "video VAE load", completed, total);
}

static int h3_vae_decode_progress_bridge(int completed, int total, void *opaque) {
    return h3_progress_emit(opaque, "video VAE decode", completed, total);
}

static int h3_preview_decode_progress_bridge(int completed, int total, void *opaque) {
    h3_generation_progress *p=opaque;
    return h3_progress_emit(p,p->params->preview_vae?"tiny preview VAE":"preview VAE decode",completed,total);
}

static int h3_preview_vae_progress_bridge(int completed, int total,
                                           void *opaque) {
    return h3_progress_emit(opaque, "preview VAE load", completed, total);
}

static int h3_audio_vae_progress_bridge(int completed, int total,
                                         void *opaque) {
    return h3_progress_emit(opaque, "audio VAE", completed, total);
}

static int h3_audio_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    return h3_progress_emit(opaque, "audio VAE encoder", completed, total);
}

static int h3_video_encoder_progress_bridge(int completed, int total,
                                             void *opaque) {
    return h3_progress_emit(opaque, "video VAE encoder", completed, total);
}

static h3_video_vae_decoder *h3_acquire_video_decoder(
        h3_ctx *ctx, const char *key, const char *weight_directory,
        int latent_height, int latent_width, h3_video_vae_progress progress,
        void *progress_opaque, int *cached, char *error, size_t error_size) {
    *cached = 0;
    if (ctx->cache_enabled && ctx->video_decoder &&
        ctx->video_decoder_key && !strcmp(ctx->video_decoder_key, key)) {
        *cached = 1;
        H3_VERBOSE("h3cli: video VAE cache hit\n");
        return ctx->video_decoder;
    }
    if (ctx->cache_enabled) {
        h3_video_vae_decoder_free(ctx->video_decoder);
        ctx->video_decoder = NULL;
        free(ctx->video_decoder_key);
        ctx->video_decoder_key = NULL;
    }
    h3_video_vae_decoder *decoder = h3_video_vae_decoder_load(
        weight_directory, "src/metal/shaders.metal", latent_height, latent_width,
        progress, progress_opaque, error, error_size);
    if (!decoder || !ctx->cache_enabled) return decoder;
    char *key_copy = strdup(key);
    if (!key_copy) {
        fprintf(stderr, "h3cli: warning: could not retain video VAE cache key\n");
        return decoder;
    }
    ctx->video_decoder = decoder;
    ctx->video_decoder_key = key_copy;
    *cached = 1;
    H3_VERBOSE("h3cli: video VAE cache miss; decoder retained\n");
    return decoder;
}

static int h3_vision_progress_bridge(int completed, int total, void *opaque) {
    return h3_progress_emit(opaque, "Qwen vision", completed, total);
}

static uint8_t *h3_rgb_f32_to_u8(const float *rgb, size_t count) {
    uint8_t *output = malloc(count);
    if (!output) return NULL;
    for (size_t index = 0; index < count; index++) {
        float scaled = rgb[index] * 255.0f;
        if (scaled < 0.0f) scaled = 0.0f;
        if (scaled > 255.0f) scaled = 255.0f;
        output[index] = (uint8_t)lrintf(scaled);
    }
    return output;
}

typedef struct {
    h3_generation_progress *progress;
    h3_video_vae_decoder *decoder;
    h3_tiny_vae *tiny;
    int latent_t;
    int latent_h;
    int latent_w;
    int output_frames;
    int output_width;
    int output_height;
    int trim_frames;
    int failed;
} h3_live_preview;

static int h3_deliver_denoise_preview(int completed_steps, int total_steps,
                                      const float *video_latent,
                                      size_t video_elements, void *opaque) {
    h3_live_preview *preview = opaque;
    if (!preview || !preview->progress || (!preview->decoder && !preview->tiny) || !video_latent) {
        if (preview && preview->progress)
            h3_set_error(preview->progress->ctx,
                         "invalid denoising preview latent");
        if (preview) preview->failed = 1;
        return 1;
    }
    size_t expected = (size_t)24 * (size_t)preview->latent_t *
                      (size_t)preview->latent_h * (size_t)preview->latent_w;
    if (video_elements != expected) {
        h3_set_error(preview->progress->ctx,
                     "invalid denoising preview latent size");
        preview->failed = 1;
        return 1;
    }
    char detail[512];
    h3_video_frames decoded;
    memset(&decoded, 0, sizeof(decoded));
    int frame_index = ((preview->latent_t - 2) / 5 / 2) * 17 + 11;
    if (frame_index < preview->trim_frames) frame_index = preview->trim_frames;
    int decoded_ok = preview->tiny ? h3_tiny_vae_decode(preview->tiny,
            video_latent, preview->latent_t, preview->latent_h, preview->latent_w,
            frame_index, &decoded, h3_preview_decode_progress_bridge,
            preview->progress, detail, sizeof(detail)) :
        h3_video_vae_decoder_preview_progress(preview->decoder, video_latent,
            preview->latent_t, h3_preview_decode_progress_bridge,
            preview->progress, &decoded, &frame_index, detail, sizeof(detail));
    if (!decoded_ok) {
        h3_set_error(preview->progress->ctx,
                     "cannot decode denoising preview: %s", detail);
        preview->failed = 1;
        return 1;
    }
    if (frame_index < preview->trim_frames) {
        h3_video_frames_free(&decoded); return 0;
    }
    frame_index -= preview->trim_frames;
    size_t count = (size_t)decoded.width * (size_t)decoded.height * 3;
    uint8_t *rgb = h3_rgb_f32_to_u8(decoded.rgb, count);
    h3_video_frames_free(&decoded);
    if (!rgb) {
        h3_set_error(preview->progress->ctx,
                     "out of memory converting denoising preview");
        preview->failed = 1;
        return 1;
    }
    int source_width = preview->latent_w * H3_VAE_SPATIAL_RATIO;
    int source_height = preview->latent_h * H3_VAE_SPATIAL_RATIO;
    if (source_width != preview->output_width ||
        source_height != preview->output_height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(
                rgb, 1, source_width, source_height,
                preview->output_width, preview->output_height, &resized)) {
            free(rgb);
            h3_set_error(preview->progress->ctx,
                         "cannot resize denoising preview");
            preview->failed = 1;
            return 1;
        }
        free(rgb);
        rgb = resized;
    }
    h3_frame frame = {
        preview->output_width, preview->output_height,
        preview->output_width * 3, rgb, frame_index,
        preview->output_frames, completed_steps - 1, total_steps
    };
    int cancelled = preview->progress->params->on_frame(
        &frame, preview->progress->params->callback_opaque);
    free(rgb);
    if (cancelled) {
        h3_set_error(preview->progress->ctx,
                     "generation cancelled during denoising preview %d",
                     completed_steps);
        preview->failed = 1;
        return 1;
    }
    return 0;
}

typedef struct { h3_ctx *ctx; const h3_params *params; int mode; } h3_lora_progress;
static int h3_lora_progress_bridge(const char *phase,int done,int total,void *opaque) {
    h3_lora_progress *p=opaque;
    if(!strcmp(phase,"LoRA cache miss")) {
        /* Drop mapped weights before an exclusive repair/build lease. A hit
         * retains the prepared DiT and holds overlapping reader leases. */
        h3_cache_clear(p->ctx);
        h3_lora_variant_free(&p->ctx->lora_variants[p->mode]);
    }
    return p->params->on_progress?p->params->on_progress(phase,done,total,p->params->callback_opaque):0;
}
static int h3_prepare_lora(h3_ctx *ctx,int mode,const h3_params *params) {
    if(!ctx->lora)return 1;
    char *source=h3_path(ctx->model_dir,mode?"Ref2VA/transformer":"FL2VA/transformer");
    if(!source){h3_set_error(ctx,"out of memory resolving LoRA transformer");return 0;}
    h3_lora_variant next={.lease=-1};h3_lora_progress progress={ctx,params,mode};
    int ok=h3_lora_prepare(ctx->lora,source,mode,h3_lora_progress_bridge,&progress,&next,ctx->error,sizeof(ctx->error));
    free(source);
    h3_component_info inventory;
    if(ok&&!h3_st_inventory_dir(next.transformer,&inventory,ctx->error,sizeof(ctx->error))) {
        h3_lora_variant_free(&next);return 0;
    }
    if(ok){
        if(ctx->lora_variants[mode].transformer&&strcmp(ctx->lora_variants[mode].key,next.key))h3_cache_clear(ctx);
        h3_lora_variant_free(&ctx->lora_variants[mode]);ctx->lora_variants[mode]=next;
        if(mode)ctx->model.ref2va_transformer=inventory;
        else ctx->model.fl2va_transformer=inventory;
    }
    return ok;
}

static h3_result *h3_generate_state(h3_ctx *ctx, const char *prompt,
                       const h3_params *params, h3_sampler_state *resume) {
    if (!ctx) return NULL;
    ctx->error[0] = '\0';
    if (!prompt || !*prompt) {
        h3_set_error(ctx, "prompt must not be empty");
        return NULL;
    }
    if (!h3_preview_vae_options(params->preview_vae, params->preview_vae_model, ctx->error, sizeof(ctx->error))) return NULL;
    if (!resume && !h3_request_params_valid(params,&ctx->device,ctx->error,sizeof(ctx->error))) return NULL;
    if (resume && !h3_video_vae_tile_pixels(params->height, params->width,
                                            ctx->error, sizeof(ctx->error))) return NULL;
    if (!resume && getenv("H3_MPS_GQA")) {
        h3_set_error(ctx, "H3_MPS_GQA cannot enforce Qwen scaling semantics; unset H3_MPS_GQA and use H3_QWEN_GQA_SCALE_MODE");
        return NULL;
    }
    int checkpoint = resume || params->stop_after_step >= 0 || params->save_sampler_state || params->save_upscale_state;
    if (params->stop_after_step < -1 || params->stop_after_step > params->steps ||
        (resume && params->stop_after_step >= 0 && params->stop_after_step < resume->next_step)) {
        h3_set_error(ctx,"stop-after-step must be an absolute boundary within the original schedule and at or after next_step"); return NULL;
    }
    if (checkpoint && !h3_sampler_checkpoint_options(params,&ctx->device,ctx->error,sizeof(ctx->error))) return NULL;
    if (params->preview_on_stop && !checkpoint) { h3_set_error(ctx,"preview-on-stop requires a sampler checkpoint"); return NULL; }
    int has_continuation = resume ? resume->continuation : params->continuation != NULL;

    int render_width = params->render_width ? params->render_width :
                                               params->width;
    int render_height = params->render_height ? params->render_height :
                                                 params->height;
    if (!params->still && h3_align_frame_count(params->frames) < 22) {
        h3_set_error(ctx,
            "generation requires at least one trained 22-frame decoder chunk");
        return NULL;
    }
    int ref2va = resume ? resume->ref2va : params->reference_count != 0;
    if (!ref2va && !ctx->model.fl2va_transformer.files) {
        h3_set_error(ctx, "prompt-only and first/last-frame requests require the FL2VA checkpoint");
        return NULL;
    }
    if (ref2va && !ctx->model.ref2va_transformer.files) {
        h3_set_error(ctx, "ordered references require the Ref2VA checkpoint");
        return NULL;
    }
    h3_generation_progress progress = {.ctx = ctx, .params = params};
    h3_temporal_shape temporal = params->still ? (h3_temporal_shape){.frame_count=1,.video_t=1,.audio_t=2} : h3_temporal(params->frames);
    h3_denoise_prefix prefix = {0};
    h3_bridge_profile bridge;
    int bridge_mode = has_continuation && params->continuation_mode == H3_CONTINUE_BRIDGE;
    int context_frames = resume ? resume->context_frames : params->continuation ?
        (params->continuation_context_frames ? params->continuation_context_frames : 39) : 0;
    if (resume) { prefix=resume->layout.prefix; bridge=resume->bridge; }
    double signature_start = h3_av_now();
    if (!params->still && !ctx->av_signature_ready[ref2va]) {
        if (!h3_av_state_metadata_signature(ctx->model_dir,ref2va,ctx->av_signature[ref2va],
            ctx->error,sizeof(ctx->error))) return NULL;
        ctx->av_signature_ready[ref2va] = 1;
    }
    if (resume && memcmp(resume->av_signature,ctx->av_signature[ref2va],32)) {
        h3_set_error(ctx,"sampler state AV compatibility signature mismatch"); return NULL;
    }
    if (params->continuation && !h3_av_state_validate_continuation(params->continuation,
        render_width,render_height,temporal.frame_count,context_frames,
        ctx->av_signature[ref2va],&prefix,ctx->error,sizeof(ctx->error))) return NULL;
    if (getenv("H3_PROFILE")) fprintf(stderr,"h3cli: AV compatibility signature %.3f s\n",h3_av_now()-signature_start);
    if (bridge_mode && !resume) {
        double started = h3_av_now();
        if (!h3_bridge_profile_build(context_frames,params->bridge_video_steps,
            params->bridge_max_strength,params->bridge_profile,&bridge,NULL,0,ctx->error,sizeof(ctx->error))) return NULL;
        if (getenv("H3_PROFILE")) fprintf(stderr,"h3cli: bridge profile construction %.6f s, %zu bytes\n",h3_av_now()-started,sizeof(bridge));
    }
    if (has_continuation) H3_VERBOSE(
        "h3cli: continuation raw=%d frames, protected=%d frames / %d video steps / %d audio ticks (%.3f s), net-new=%d frames (%.3f s)\n"
        "h3cli: mode=%s, explicit references=%zu, prefix trimming=%s\n",
        temporal.frame_count,context_frames,prefix.video_prefix_t,prefix.audio_prefix_t,
        (double)context_frames/H3_FPS,temporal.frame_count-context_frames,
        (double)(temporal.frame_count-context_frames)/H3_FPS,
        ref2va ? "Ref2VA" : "T2VA",params->reference_count,
        params->keep_continuation_prefix ? "off" : "on");
    h3_image_vae_info image_info={0};
    if(params->still) {
        if(!h3_image_vae_inspect(params->image_vae,&image_info,ctx->error,sizeof(ctx->error)))return NULL;
        H3_VERBOSE("h3cli: single-still: video T=1, auxiliary audio T=2 (generated and discarded), dense BF16, 50 blocks\n");
        /* A still never retains a video decoder beside its own final decoder. */
        h3_video_vae_decoder_free(ctx->video_decoder);ctx->video_decoder=NULL;
        free(ctx->video_decoder_key);ctx->video_decoder_key=NULL;
        h3_tiny_vae_free(ctx->tiny_decoder);ctx->tiny_decoder=NULL;
    }
    if(!resume && !h3_prepare_lora(ctx,ref2va,params))return NULL;
    int latent_w, latent_h;
    h3_latent_canvas(render_width, render_height, &latent_w, &latent_h);
    h3_tokenizer *tokenizer = NULL;
    uint32_t *ids = NULL;
    size_t token_count = 0;
    size_t visual_capacity = ref2va ? params->reference_count :
        (size_t)(params->first_frame != NULL) +
        (size_t)(params->last_frame != NULL);
    size_t visual_count = 0;
    float **condition_pixels = NULL;
    int *condition_widths = NULL;
    int *condition_heights = NULL;
    int *condition_frames = NULL;
    int *condition_vae_frames = NULL;
    int *condition_latent_t = NULL;
    int *condition_audio_limits = NULL;
    size_t *visual_reference_indices = NULL;
    size_t *reference_visual_indices = NULL;
    h3_vision_output *vision_outputs = NULL;
    size_t vision_output_count = 0;
    h3_reference_presentation *presentations = NULL;
    double **presentation_timestamps = NULL;
    h3_layout_ref *layout_references = NULL;
    int keyframes[2] = {0, 0};
    size_t keyframe_count = 0;
    float *condition_video_rows = NULL;
    size_t condition_video_elements = 0;
    float *condition_audio_rows = NULL;
    size_t condition_audio_elements = 0;
    h3_text_embedding text;
    memset(&text, 0, sizeof(text));
    h3_layout layout;
    memset(&layout, 0, sizeof(layout));
    h3_dit *dit = NULL;
    h3_video_vae_decoder *preview_decoder = NULL;
    h3_tiny_vae *tiny_decoder = NULL;
    int tiny_is_cached = 0;
    int output_width = params->width, output_height = params->height;
    h3_presentation delivery = {.version=9, .av_metadata_identity=1, .geometry_profile=params->geometry_profile, .ref2va=ref2va,
        .render_width=render_width, .render_height=render_height,
        .width=params->width, .height=params->height,
        .trim_frames=has_continuation&&!params->keep_continuation_prefix?context_frames:0,
        .trim_samples=has_continuation&&!params->keep_continuation_prefix?prefix.audio_prefix_t*800:0,
        .fps=24, .sample_rate=32000, .codec_version=params->_arithmetic_recipe?2:1};
    if(resume&&resume->upscale.stage==2)h3_upscale_presentation(resume,&delivery);
    if(params->cuda_attention) {
        delivery.cuda_attention=params->cuda_attention;
        if(params->cuda_attention==H3_ATTENTION_SOL)delivery.cuda_sol=params->cuda_sol;
        delivery.attention_version=h3_attention_execution_recipe(params->cuda_attention,params->cuda_denoise_quant);delivery.attention_plan=h3_attention_plan(params->cuda_attention);
    }
    if(params->adaptive_cache||params->cuda_attention==H3_ATTENTION_SUBBLOCK) {
        delivery.adaptive_cache=params->adaptive_cache;
        delivery.adaptive_version=h3_adaptive_execution_recipe(params->adaptive_cache,params->cuda_denoise_quant,ref2va,has_continuation);
        delivery.adaptive_cache_threshold=h3_adaptive_threshold(params);
        delivery.adaptive_cache_max_hits=h3_adaptive_max_hits(params);
        if(params->cuda_attention==H3_ATTENTION_SUBBLOCK)delivery.subblock_sparsity=params->subblock_sparsity;
        {
            delivery.adaptive_cache_warmup=params->adaptive_cache?h3_adaptive_warmup(params->adaptive_cache_warmup):0;
            delivery.subblock_warmup=params->cuda_attention==H3_ATTENTION_SUBBLOCK?h3_subblock_warmup(params->subblock_warmup):0;
        }
    }
    if(params->cuda_denoise_quant) {
        uint8_t digest[32];
        int strict=h3_quant_verify();
        int identified=strict?
            h3_sampler_model_fingerprint_effective(ctx->model_dir,ctx->lora_variants[ref2va].transformer,ref2va,digest,ctx->error,sizeof(ctx->error)):
            h3_sampler_model_metadata_effective(ctx->model_dir,ctx->lora_variants[ref2va].transformer,ref2va,digest,ctx->error,sizeof(ctx->error));
        if(!identified)return NULL;
        delivery.quant_model_metadata=!strict;
        delivery.cuda_denoise_quant=params->cuda_denoise_quant;
        delivery.quant_version=h3_quant_execution_recipe(params->cuda_denoise_quant,params->adaptive_cache,params->cuda_attention);
        for(int i=0;i<32;i++)snprintf(delivery.quant_model_sha256+i*2,3,"%02x",digest[i]);
    }
    h3_live_preview live_preview;
    memset(&live_preview, 0, sizeof(live_preview));
    float *video = NULL, *audio = NULL;
    h3_video_frames frames;
    memset(&frames, 0, sizeof(frames));
    h3_audio_waveform waveform;
    memset(&waveform, 0, sizeof(waveform));
    uint8_t *rgb8 = NULL;
    h3_result *result = NULL;
    h3_sampler_state *sampler = resume;
    h3_conditioning *loaded_conditioning=NULL, *saved_conditioning=NULL;
    float *up_raw_video=NULL,*up_raw_audio=NULL;
    h3_upscale_record up_record={0};
    const h3_conditioning *previous_conditioning=h3_conditioning_current();
    char *disk_conditioning_identity=NULL;
    int paused = 0;
    h3_av_state *captured = NULL;
    char *conditioning_key = NULL;
    char *prepared_key = NULL;
    char *decoder_key = NULL;
    int conditioning_hit = 0;
    int conditioned = 0;
    int dit_is_cached = 0;
    int decoder_is_cached = 0;
    char *tokenizer_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/tokenizer/tokenizer.json" : "FL2VA/tokenizer/tokenizer.json");
    char *text_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/text_encoder" : "FL2VA/text_encoder");
    char *dit_path = ctx->lora_variants[ref2va].transformer ? strdup(ctx->lora_variants[ref2va].transformer) : h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/transformer" : "FL2VA/transformer");
    char *vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/video_vae/source" : "FL2VA/video_vae/source");
    char *audio_vae_path = h3_path(ctx->model_dir, ref2va ?
        "Ref2VA/audio_vae" : "FL2VA/audio_vae");
    if (!tokenizer_path || !text_path || !dit_path || !vae_path ||
        !audio_vae_path) {
        h3_set_error(ctx, "out of memory resolving generation model paths");
        goto cleanup;
    }
    for (size_t i = 0; i < params->reference_count; i++) {
        int is_video = resume ? resume->references[i].kind == H3_LAYOUT_REF_VIDEO :
            (params->references[i].kind == H3_REFERENCE_VIDEO || params->references[i].kind == H3_REFERENCE_VIDEO_AUDIO);
        if (is_video) H3_VERBOSE("h3cli: Ref2VA video %zu pipeline=%s\n", i + 1,
            "released-v1");
    }
    conditioning_key = resume ? strdup("serialized-conditioning") : h3_conditioning_key(
        prompt, params, render_width, render_height, ref2va);
    if (!conditioning_key) {
        h3_set_error(ctx, "out of memory constructing conditioning cache key");
        goto cleanup;
    }
    if(params->save_conditioning || params->load_conditioning) {
        disk_conditioning_identity=h3_conditioning_identity(ctx->model_dir,
            ctx->lora_variants[ref2va].transformer,prompt,params,
            ctx->av_signature[ref2va],&ctx->device,ctx->error,sizeof(ctx->error));
        if(!disk_conditioning_identity)goto cleanup;
        /* A long-lived context must not reuse conditioning or a prepared DiT
         * from a weaker path/mtime-only identity when disk caching is explicit. */
        uint8_t digest[32];char identity_hex[65];h3_key key={0};
        h3_sampler_hash(disk_conditioning_identity,strlen(disk_conditioning_identity),digest);
        for(int i=0;i<32;i++)snprintf(identity_hex+2*i,3,"%02x",digest[i]);
        if(!h3_key_append(&key,"%s|h3cond-sha256=%s",conditioning_key,identity_hex)) {
            free(key.text);h3_set_error(ctx,"cannot key disk conditioning identity");goto cleanup;
        }
        free(conditioning_key);conditioning_key=key.text;
    }
    prepared_key = h3_prepared_key(
        conditioning_key, params, render_width, render_height);
    if (!prepared_key) {
        h3_set_error(ctx, "out of memory constructing prepared-model cache key");
        goto cleanup;
    }
    if(ctx->lora_variants[ref2va].transformer) {
        h3_key effective_key={0};
        if(!h3_key_append(&effective_key,"%s|lora:%s",prepared_key,ctx->lora_variants[ref2va].key)) {
            h3_set_error(ctx,"out of memory keying effective transformer");goto cleanup;
        }
        free(prepared_key);prepared_key=effective_key.text;
    }
    h3_key decoder_cache_key = {0};
    if (!h3_key_append(&decoder_cache_key, "%s|%dx%d", vae_path, latent_h, latent_w) ||
        !h3_key_execution(&decoder_cache_key,params) ||
        !h3_key_append(&decoder_cache_key,"|video-decoder=default-v1|device=%s:%s:%d|graph=%s",ctx->device.backend,ctx->device.name,ctx->device.device_index,getenv("H3_FULL_VAE_CUDA_GRAPH")?getenv("H3_FULL_VAE_CUDA_GRAPH"):"0")) {
        h3_set_error(ctx, "out of memory constructing decoder cache key");
        goto cleanup;
    }
    if(ctx->cache_enabled && !params->preview_vae && !params->still) {
        uint8_t digest[32];
        int (*identity)(const char *,const char *,int,const char *,uint8_t *,char *,size_t)=h3_sampler_component_metadata;
        if(!identity(ctx->model_dir,NULL,ref2va,"video_vae",digest,ctx->error,sizeof(ctx->error))) {
            free(decoder_cache_key.text);goto cleanup;
        }
        for(size_t i=0;i<sizeof(digest);i++)if(!h3_key_append(&decoder_cache_key,"%02x",digest[i])) {
            free(decoder_cache_key.text);h3_set_error(ctx,"cannot key video decoder content");goto cleanup;
        }
    }
    decoder_key = decoder_cache_key.text;
    if (ctx->cache_enabled && ctx->video_decoder &&
        (!ctx->video_decoder_key || strcmp(ctx->video_decoder_key, decoder_key))) {
        h3_video_vae_decoder_free(ctx->video_decoder);
        ctx->video_decoder = NULL;
        free(ctx->video_decoder_key);
        ctx->video_decoder_key = NULL;
    }
    uint8_t resume_key[32]={0};
    if(resume && !h3_sampler_prepared_key(resume,resume_key)) { h3_set_error(ctx,"cannot identify prepared checkpoint state"); goto cleanup; }
    if(ctx->dit && !h3_dit_placement_compatible(ctx->dit,params->ssd_streaming)) {
        h3_dit_free(ctx->dit);ctx->dit=NULL;free(ctx->dit_key);ctx->dit_key=NULL;ctx->dit_sampler_key_ready=0;
        H3_VERBOSE("h3cli: invalidated retained DiT weight placement\n");
    }
    int resume_live=resume && ctx->cache_enabled && ctx->dit && ctx->dit_sampler_key_ready && !memcmp(resume_key,ctx->dit_sampler_key,32);
    if (ctx->cache_enabled && ctx->dit &&
        (resume ? !resume_live : (!ctx->dit_key || strcmp(ctx->dit_key, prepared_key)))) {
        h3_dit_free(ctx->dit);
        ctx->dit = NULL;
        free(ctx->dit_key);
        ctx->dit_key = NULL;
    }
    conditioning_hit = ctx->cache_enabled && ctx->conditioning_key &&
        !strcmp(ctx->conditioning_key, conditioning_key);
    if (!resume && !conditioning_hit && !h3_device_memory_fits(UINT64_C(4)<<30)) {
        h3_dit_free(ctx->dit); ctx->dit = NULL;
        free(ctx->dit_key); ctx->dit_key = NULL; ctx->dit_sampler_key_ready = 0;
        h3_video_vae_decoder_free(ctx->video_decoder); ctx->video_decoder = NULL;
        free(ctx->video_decoder_key); ctx->video_decoder_key = NULL;
        H3_VERBOSE("h3cli: evicted GPU caches to make room for conditioning\n");
    }
    char detail[512];
    if(params->save_conditioning || params->load_conditioning) {
        if(params->load_conditioning) {
            loaded_conditioning=h3_conditioning_load(params->load_conditioning,ctx->error,sizeof(ctx->error));
            if(!loaded_conditioning || !h3_conditioning_identity_matches(
                loaded_conditioning->identity,disk_conditioning_identity,ctx->error,sizeof(ctx->error)))goto cleanup;
            if(!h3_conditioning_request_matches(loaded_conditioning,params,ctx->error,sizeof(ctx->error)))goto cleanup;
            if(loaded_conditioning->reference_count!=params->reference_count ||
                loaded_conditioning->keyframe_count!=(size_t)(params->first_frame!=NULL)+(size_t)(params->last_frame!=NULL)) {
                h3_set_error(ctx,"h3cond: reference/keyframe tensor count mismatch");goto cleanup;
            }
        }
    }
    if (params->preview_vae) {
        /* Keep only the selected video decoder resident. Sampling cache keys
         * deliberately exclude this delivery policy. */
        h3_video_vae_decoder_free(ctx->video_decoder); ctx->video_decoder=NULL;
        free(ctx->video_decoder_key); ctx->video_decoder_key=NULL;
        char digest[65]; double load_begin=h3_av_now();
        if (!h3_tiny_vae_validate(h3_preview_vae_path(params->preview_vae_model),digest,detail,sizeof(detail))) {
            h3_set_error(ctx,"%s",detail); goto cleanup;
        }
        if (ctx->tiny_decoder && !h3_tiny_vae_cache_matches(ctx->tiny_decoder,digest)) {
            h3_tiny_vae_free(ctx->tiny_decoder); ctx->tiny_decoder=NULL;
        }
        int hit=ctx->tiny_decoder!=NULL;
        if (h3_progress_emit(&progress,"tiny VAE load",0,1)) goto cleanup;
        tiny_decoder=ctx->tiny_decoder?ctx->tiny_decoder:h3_tiny_vae_load(
            h3_preview_vae_path(params->preview_vae_model),detail,sizeof(detail));
        if (!tiny_decoder) { h3_set_error(ctx,"%s",detail); goto cleanup; }
        if (ctx->cache_enabled) { ctx->tiny_decoder=tiny_decoder; tiny_is_cached=1; }
        memcpy(delivery.tiny_sha256,h3_tiny_vae_digest(tiny_decoder),65);
        if (h3_progress_emit(&progress,"tiny VAE load",1,1)) goto cleanup;
        if (getenv("H3_PROFILE")) fprintf(stderr,"h3cli: tiny VAE %s load %.6f s; FP16; sha256=%s\n",
            hit?"cached":"cold",h3_av_now()-load_begin,delivery.tiny_sha256);
    } else { h3_tiny_vae_free(ctx->tiny_decoder); ctx->tiny_decoder=NULL; }
    if (resume) {
        if (!h3_text_embedding_copy(&text,&resume->text)) { h3_set_error(ctx,"cannot restore checkpoint text conditioning"); goto cleanup; }
        condition_video_elements=resume->condition_video_elements;
        condition_audio_elements=resume->condition_audio_elements;
        condition_video_rows=condition_video_elements?malloc(condition_video_elements*sizeof(float)):NULL;
        condition_audio_rows=condition_audio_elements?malloc(condition_audio_elements*sizeof(float)):NULL;
        layout_references=resume->reference_count?malloc(resume->reference_count*sizeof(*layout_references)):NULL;
        if ((condition_video_elements && !condition_video_rows) || (condition_audio_elements && !condition_audio_rows) ||
            (resume->reference_count && !layout_references)) { h3_set_error(ctx,"cannot restore checkpoint condition rows"); goto cleanup; }
        if (condition_video_elements) memcpy(condition_video_rows,resume->condition_video,condition_video_elements*sizeof(float));
        if (condition_audio_elements) memcpy(condition_audio_rows,resume->condition_audio,resume->condition_audio_elements*sizeof(float));
        if (resume->reference_count) memcpy(layout_references,resume->references,resume->reference_count*sizeof(*layout_references));
        conditioned=resume->conditioned;
        if(ctx->cache_enabled && !h3_conditioning_cache_store(ctx,conditioning_key,&text,
            condition_video_rows,condition_video_elements,condition_audio_rows,condition_audio_elements,
            layout_references,resume->reference_count,conditioned)) {
            h3_set_error(ctx,"cannot restore serialized conditioning cache"); goto cleanup;
        }
        H3_VERBOSE("h3cli: restored exact conditioning; tokenizer, text/vision and reference encoders skipped\n");
    } else if (loaded_conditioning) {
        text=loaded_conditioning->text;memset(&loaded_conditioning->text,0,sizeof(loaded_conditioning->text));
        condition_video_rows=loaded_conditioning->video;loaded_conditioning->video=NULL;
        condition_audio_rows=loaded_conditioning->audio;loaded_conditioning->audio=NULL;
        condition_video_elements=loaded_conditioning->video_elements;
        condition_audio_elements=loaded_conditioning->audio_elements;
        layout_references=loaded_conditioning->references;loaded_conditioning->references=NULL;
        conditioned=loaded_conditioning->conditioned;
        keyframe_count=loaded_conditioning->keyframe_count;
        memcpy(keyframes,loaded_conditioning->keyframes,sizeof(keyframes));
        if(ctx->cache_enabled&&!h3_conditioning_cache_store(ctx,conditioning_key,&text,
            condition_video_rows,condition_video_elements,condition_audio_rows,condition_audio_elements,
            layout_references,params->reference_count,conditioned)) {
            h3_set_error(ctx,"cannot retain loaded conditioning");goto cleanup;
        }
        H3_VERBOSE("h3cli: h3cond hit; tokenizer, text/vision and reference encoders skipped\n");
    } else if (conditioning_hit) {
        size_t cached_reference_count = 0;
        if (!h3_conditioning_cache_load(
                ctx, &text, &condition_video_rows, &condition_video_elements,
                &condition_audio_rows, &condition_audio_elements,
                &layout_references, &cached_reference_count, &conditioned) ||
            cached_reference_count != (ref2va ? params->reference_count : 0)) {
            h3_set_error(ctx, "cannot restore cached conditioning");
            goto cleanup;
        }
        if (!ref2va) {
            if (params->first_frame) keyframes[keyframe_count++] = 0;
            if (params->last_frame)
                keyframes[keyframe_count++] = temporal.frame_count - 1;
        }
        H3_VERBOSE("h3cli: conditioning cache hit\n");
    } else {
    if (visual_capacity) {
        condition_pixels = calloc(visual_capacity, sizeof(*condition_pixels));
        condition_widths = calloc(visual_capacity, sizeof(*condition_widths));
        condition_heights = calloc(visual_capacity, sizeof(*condition_heights));
        condition_frames = calloc(visual_capacity, sizeof(*condition_frames));
        condition_vae_frames = calloc(visual_capacity, sizeof(*condition_vae_frames));
        condition_latent_t = calloc(visual_capacity, sizeof(*condition_latent_t));
        condition_audio_limits = calloc(visual_capacity, sizeof(*condition_audio_limits));
        visual_reference_indices = calloc(
            visual_capacity, sizeof(*visual_reference_indices));
        reference_visual_indices = malloc(
            params->reference_count * sizeof(*reference_visual_indices));
        if (reference_visual_indices)
            for (size_t index = 0; index < params->reference_count; index++)
                reference_visual_indices[index] = SIZE_MAX;
        if (ref2va) {
            layout_references = calloc(visual_capacity,
                                       sizeof(*layout_references));
            presentations = calloc(params->reference_count,
                                   sizeof(*presentations));
            presentation_timestamps = calloc(
                params->reference_count, sizeof(*presentation_timestamps));
        }
        if (!condition_pixels || !condition_widths || !condition_heights ||
            !condition_frames || !condition_vae_frames || !condition_latent_t ||
            !condition_audio_limits || !visual_reference_indices ||
            !reference_visual_indices ||
            (ref2va && (!layout_references || !presentations ||
                        !presentation_timestamps))) {
            h3_set_error(ctx, "out of memory preparing visual references");
            goto cleanup;
        }
    }
    if (h3_progress_emit(&progress, "tokenizer", 0, 1)) goto cleanup;
    tokenizer = h3_tokenizer_load(tokenizer_path, detail, sizeof(detail));
    if (!tokenizer) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (h3_progress_emit(&progress, "tokenizer", 1, 1)) goto cleanup;
    if (progress.cancelled) goto cleanup;
    if (ref2va) {
        int total_video_frames = 0;
        size_t total_image_patches = 0;
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            if (reference->kind == H3_REFERENCE_AUDIO) {
                presentations[index].kind = H3_PRESENTATION_AUDIO;
                layout_references[index] = (h3_layout_ref){
                    H3_LAYOUT_REF_AUDIO, 0, 0, 0, 0};
                continue;
            }
            int source_width, source_height, media_width, media_height;
            if (!h3_ffprobe_visual_size(reference->path,
                                        &source_width, &source_height,
                                        detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (reference->kind == H3_REFERENCE_IMAGE) {
                h3_reference_image_shape image_shape;
                if (!h3_reference_image_resolve(source_width, source_height,
                        render_width, render_height, params->reference_image_size,
                        &image_shape, detail, sizeof(detail))) {
                    h3_set_error(ctx, "reference %zu: %s", index + 1, detail);
                    goto cleanup;
                }
                media_width = image_shape.width; media_height = image_shape.height;
                total_image_patches += image_shape.patches; /* at most nine admitted images */
                H3_VERBOSE("h3cli: reference image %zu size=%s source=%dx%d canvas=%dx%d patches=%zu image-batch-patches=%zu\n",
                    index + 1,h3_reference_image_size_name(params->reference_image_size),
                    source_width,source_height,media_width,media_height,image_shape.patches,total_image_patches);
                int pixels_ok=params->_arithmetic_recipe ?
                    h3_sglang_read_reference_image(reference->path,source_width,source_height,media_width,media_height,
                        &condition_pixels[visual_count],detail,sizeof(detail)) : h3_ffmpeg_read_image_f32(
                        reference->path, media_width, media_height,
                        H3_IMAGE_FIT_STRETCH, &condition_pixels[visual_count],
                        detail, sizeof(detail));
                if (!pixels_ok) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                condition_frames[visual_count] = 1;
                condition_vae_frames[visual_count] = 1;
                condition_latent_t[visual_count] = 1;
                presentations[index].kind = H3_PRESENTATION_IMAGE;
                presentations[index].vision_count = 1;
                vision_output_count++;
            } else {
                /* The pinned reference policy always adapts from the display
                 * ratio, including upscaling a smaller source. */
                if ((params->_arithmetic_recipe && ((double)source_width>4.0*source_height ||
                        (double)source_height>4.0*source_width)) ||
                    !(params->_arithmetic_recipe ? h3_adapt_canvas : h3_reference_video_canvas)(
                        source_width, source_height,
                        &media_width, &media_height)) {
                    h3_set_error(ctx,
                        "cannot resolve reference video %zu canvas", index + 1);
                    goto cleanup;
                }
                h3_refvideo_plan video_plan;
                int video_ok=params->_arithmetic_recipe ? h3_refvideo_read_sglang(
                        reference->path,media_width,media_height,temporal.frame_count,
                        &total_video_frames,&condition_pixels[visual_count],&video_plan,detail,sizeof(detail)) : h3_refvideo_read(
                        reference->path, media_width, media_height,
                        temporal.frame_count,
                        &total_video_frames, &condition_pixels[visual_count], &video_plan,
                        detail, sizeof(detail));
                if (!video_ok) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                condition_frames[visual_count] = video_plan.frames;
                condition_vae_frames[visual_count] = video_plan.vae_frames;
                condition_latent_t[visual_count] = video_plan.latent_t;
                condition_audio_limits[visual_count] = video_plan.soundtrack_samples;
                H3_VERBOSE("h3cli: reference video %zu: pipeline=%s normalized-frames=%d VAE-frames=%d latent-T=%d posterior-seed=%s\n",
                    index + 1, h3_refvideo_pipeline_name(video_plan.pipeline),
                    video_plan.frames, video_plan.vae_frames, video_plan.latent_t,
                    "42");
                size_t blocks = video_plan.qwen_blocks;
                if (vision_output_count > SIZE_MAX - blocks) {
                    h3_set_error(ctx, "reference video vision count overflows");
                    goto cleanup;
                }
                presentation_timestamps[index] = malloc(
                    blocks * sizeof(*presentation_timestamps[index]));
                if (!presentation_timestamps[index]) {
                    h3_set_error(ctx,
                        "out of memory allocating reference video timestamps");
                    goto cleanup;
                }
                for (size_t block = 0; block < blocks; block++) {
                    int first, second;
                    if (!h3_refvideo_qwen_pair(video_plan.frames, block, &first, &second,
                            &presentation_timestamps[index][block])) {
                        h3_set_error(ctx, "invalid Qwen reference-video sampling plan"); goto cleanup;
                    }
                }
                presentations[index].kind = H3_PRESENTATION_VIDEO;
                presentations[index].vision_count = blocks;
                presentations[index].timestamps = presentation_timestamps[index];
                vision_output_count += blocks;
            }
            condition_widths[visual_count] = media_width;
            condition_heights[visual_count] = media_height;
            visual_reference_indices[visual_count] = index;
            reference_visual_indices[index] = visual_count;
            int ref_latent_w, ref_latent_h;
            h3_latent_canvas(media_width, media_height,
                             &ref_latent_w, &ref_latent_h);
            layout_references[index] = (h3_layout_ref){
                reference->kind == H3_REFERENCE_IMAGE ?
                    H3_LAYOUT_REF_IMAGE : H3_LAYOUT_REF_VIDEO,
                condition_latent_t[visual_count],
                ref_latent_h, ref_latent_w, 0};
            visual_count++;
        }

        size_t total_audio_samples = 0;
        for (size_t index = 0; index < params->reference_count; index++) {
            const h3_reference *reference = &params->references[index];
            const char *audio_path = NULL;
            int truncate = 0;
            int max_samples = 32000 * 15;
            if (reference->kind == H3_REFERENCE_AUDIO) {
                audio_path = reference->path;
            } else if (reference->kind == H3_REFERENCE_VIDEO_AUDIO) {
                audio_path = reference->audio_path;
                truncate = 1;
            } else if (reference->kind == H3_REFERENCE_VIDEO &&
                       reference->include_embedded_audio) {
                audio_path = reference->path;
                truncate = 1;
            }
            if (!audio_path) continue;
            if (truncate) {
                size_t visual = reference_visual_indices[index];
                if (visual == SIZE_MAX) {
                    h3_set_error(ctx,
                        "video soundtrack %zu has no decoded video", index + 1);
                    goto cleanup;
                }
                max_samples = condition_audio_limits[visual];
                if (max_samples < 64000 && !params->_arithmetic_recipe) {
                    h3_set_error(ctx,
                        "video soundtrack %zu requires at least 2 seconds; "
                        "request at least 56 output frames", index + 1);
                    goto cleanup;
                }
            }
            float *pcm = NULL;
            int samples = 0;
            int audio_ok=params->_arithmetic_recipe&&truncate ?
                h3_sglang_read_soundtrack(audio_path,temporal.frame_count,&pcm,&samples,detail,sizeof(detail)) : h3_ffmpeg_read_audio_f32(
                    audio_path, max_samples, truncate, &pcm, &samples,
                    detail, sizeof(detail));
            if (!audio_ok) {
                free(pcm);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if ((size_t)samples > (size_t)32000 * 15 - total_audio_samples) {
                free(pcm);
                h3_set_error(ctx,
                    "ordered reference audio exceeds 15 seconds in total");
                goto cleanup;
            }
            h3_audio_latent latent;
            memset(&latent, 0, sizeof(latent));
            if(params->_arithmetic_recipe&&getenv("H3_TEST_SGLANG_AUDIO_INPUT_ONLY")&&
               !h3_sglang_dump("audio-input.f32",pcm,(size_t)samples*2*sizeof(float))) {
                free(pcm);h3_set_error(ctx,"cannot capture reference soundtrack input");goto cleanup;
            }
            if (!h3_audio_vae_encode(
                    audio_vae_path, "src/metal/shaders.metal", pcm, samples,
                    h3_audio_encoder_progress_bridge, &progress, &latent,
                    detail, sizeof(detail))) {
                free(pcm);
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            free(pcm);
            if (latent.channels != 32 || latent.stereo != 2 ||
                latent.length < 1 ||
                (size_t)latent.length > SIZE_MAX / 2 / 32) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "reference audio encoder produced invalid geometry");
                goto cleanup;
            }
            size_t elements = (size_t)latent.length * 2 * 32;
            if (condition_audio_elements > SIZE_MAX - elements ||
                condition_audio_elements + elements >
                    SIZE_MAX / sizeof(*condition_audio_rows)) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx, "reference audio row count overflows");
                goto cleanup;
            }
            float *grown = realloc(
                condition_audio_rows,
                (condition_audio_elements + elements) * sizeof(*grown));
            if (!grown) {
                h3_audio_latent_free(&latent);
                h3_set_error(ctx,
                    "out of memory packing reference audio conditions");
                goto cleanup;
            }
            condition_audio_rows = grown;
            float *rows = condition_audio_rows + condition_audio_elements;
            for (int stereo = 0; stereo < 2; stereo++)
                for (int time = 0; time < latent.length; time++)
                    for (int channel = 0; channel < 32; channel++) {
                        size_t source = ((size_t)channel * 2 +
                                         (size_t)stereo) * latent.length +
                                        (size_t)time;
                        size_t destination = ((size_t)stereo * latent.length +
                                              (size_t)time) * 32 +
                                             (size_t)channel;
                        rows[destination] = latent.values[source];
                    }
            condition_audio_elements += elements;
            total_audio_samples += (size_t)samples;
            layout_references[index].audio_t = latent.length;
            presentations[index].has_audio = 1;
            h3_audio_latent_free(&latent);
            if(params->_arithmetic_recipe&&getenv("H3_TEST_SGLANG_AUDIO_INPUT_ONLY")) {
                int dumped=h3_sglang_dump("condition-audio.f32",condition_audio_rows,
                                            condition_audio_elements*sizeof(float));
                h3_set_error(ctx,dumped?"test-only reference audio capture complete; no denoising executed":
                                          "cannot capture reference audio rows");
                goto cleanup;
            }
            if (progress.cancelled) goto cleanup;
        }
    } else {
        /* A shared image at a segment join must have the same framing in
         * either role: preserve aspect ratio, fill the canvas, center-crop. */
        const char *anchors[] = {params->first_frame, params->last_frame};
        for (size_t anchor = 0; anchor < 2; anchor++) {
            if (!anchors[anchor]) continue;
            keyframes[keyframe_count++] = anchor ? temporal.frame_count - 1 : 0;
            if (!h3_ffmpeg_read_image_f32(
                    anchors[anchor], render_width, render_height,
                    H3_IMAGE_FIT_COVER, &condition_pixels[visual_count],
                    detail, sizeof(detail))) {
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            condition_widths[visual_count] = render_width;
            condition_heights[visual_count] = render_height;
            condition_frames[visual_count] = 1;
            condition_vae_frames[visual_count] = 1;
            condition_latent_t[visual_count] = 1;
            visual_count++;
        }
        vision_output_count = visual_count;
    }
    if (vision_output_count) {
        vision_outputs = calloc(vision_output_count, sizeof(*vision_outputs));
        if (!vision_outputs) {
            h3_set_error(ctx, "out of memory allocating Qwen vision outputs");
            goto cleanup;
        }
    }

    if (vision_output_count) {
        size_t cursor = 0, predicted_tokens = 0;
        for (size_t visual = 0; visual < visual_count; visual++) {
            size_t ref = visual_reference_indices[visual];
            size_t blocks = ref2va ? presentations[ref].vision_count : 1;
            if (ref2va) presentations[ref].vision = &vision_outputs[cursor];
            for (size_t block = 0; block < blocks; block++, cursor++) {
                vision_outputs[cursor].tokens = (size_t)(condition_widths[visual] / 32) *
                                               (size_t)(condition_heights[visual] / 32);
            }
        }
        int counted = ref2va ? h3_multimodal_count_ref2va(tokenizer, prompt,
            presentations, params->reference_count, &predicted_tokens, detail, sizeof(detail)) :
            h3_multimodal_count_fl2va(tokenizer, prompt, vision_outputs,
                vision_output_count, &predicted_tokens, detail, sizeof(detail));
        h3_gpu *preflight_gpu = counted ? h3_gpu_create("src/metal/shaders.metal", detail, sizeof(detail)) : NULL;
        int supported = preflight_gpu && h3_gpu_gqa_causal_preflight(preflight_gpu,
            predicted_tokens, detail, sizeof(detail));
        h3_gpu_free(preflight_gpu);
        if (!supported) { h3_set_error(ctx, "%s", detail); goto cleanup; }
        if (h3_progress_emit(&progress, "reference vision preparation", 0, (int)vision_output_count))
            goto cleanup;
    }

    if (visual_count) {
        for (size_t image = 0; image < visual_count; image++) {
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            int image_latent_t = condition_latent_t[image];
            size_t rows = (size_t)image_latent_t *
                          (size_t)image_latent_h * (size_t)image_latent_w / 4;
            if (rows > SIZE_MAX / 96 ||
                condition_video_elements > SIZE_MAX - rows * 96) {
                h3_set_error(ctx, "condition row count overflows");
                goto cleanup;
            }
            condition_video_elements += rows * 96;
        }
        if (condition_video_elements > SIZE_MAX /
                                       sizeof(*condition_video_rows)) {
            h3_set_error(ctx, "condition storage size overflows");
            goto cleanup;
        }
        condition_video_rows = malloc(condition_video_elements *
                                      sizeof(*condition_video_rows));
        if (!condition_video_rows) {
            h3_set_error(ctx, "out of memory allocating visual condition rows");
            goto cleanup;
        }
        size_t condition_offset = 0;
        for (size_t image = 0; image < visual_count; image++) {
            int image_latent_w, image_latent_h;
            h3_latent_canvas(condition_widths[image], condition_heights[image],
                             &image_latent_w, &image_latent_h);
            int image_latent_t = condition_latent_t[image];
            size_t row_elements = (size_t)image_latent_t *
                                  (size_t)image_latent_h *
                                  (size_t)image_latent_w / 4 * 96;
            h3_video_latent latent;
            memset(&latent, 0, sizeof(latent));
            int released_video = ref2va && condition_frames[image] > 1;
            size_t pixel_count=(size_t)3*condition_frames[image]*condition_heights[image]*condition_widths[image];
            if(h3_sglang_requested())h3_sglang_vae_byte_pixels(condition_pixels[image],pixel_count,0);
            int encoded = released_video ? h3_ref2va_video_vae_encode(
                    vae_path, "src/metal/shaders.metal", condition_pixels[image],
                    condition_frames[image], condition_vae_frames[image],
                    condition_heights[image], condition_widths[image],
                    h3_video_encoder_progress_bridge, &progress,
                    &latent, detail, sizeof(detail)) : h3_video_vae_encode(
                    vae_path, "src/metal/shaders.metal", condition_pixels[image],
                    condition_vae_frames[image], condition_heights[image],
                    condition_widths[image],
                    h3_video_encoder_progress_bridge, &progress,
                    &latent, detail, sizeof(detail));
            if(h3_sglang_requested())h3_sglang_vae_byte_pixels(condition_pixels[image],pixel_count,1);
            if (!encoded) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "%s", detail);
                goto cleanup;
            }
            if (latent.time != image_latent_t ||
                latent.height != image_latent_h ||
                latent.width != image_latent_w) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx,
                    "visual condition VAE produced unexpected latent geometry");
                goto cleanup;
            }
            float *rows = condition_video_rows + condition_offset;
            if (!h3_dit_patchify_video(
                    latent.values, 24, image_latent_t,
                    image_latent_h, image_latent_w,
                    rows, row_elements)) {
                h3_video_latent_free(&latent);
                h3_set_error(ctx, "cannot patchify visual condition latent");
                goto cleanup;
            }
            h3_video_latent_free(&latent);
            condition_offset += row_elements;
            if (progress.cancelled) goto cleanup;
        }
        if (condition_offset != condition_video_elements) {
            h3_set_error(ctx, "visual condition packing size mismatch");
            goto cleanup;
        }
        size_t vision_cursor = 0;
        if(params->_arithmetic_recipe) {
            if(vision_output_count>128){h3_set_error(ctx,"reference vision batch exceeds 128 inputs");goto cleanup;}
            for(int video_batch=0;video_batch<2;video_batch++) {
                h3_vision_input inputs[128];h3_vision_output encoded[128]={0};
                float *owned[128]={0};size_t destinations[128],count=0,cursor=0;int ok=1;
                for(size_t image=0;image<visual_count&&ok;image++) {
                    size_t ref=visual_reference_indices[image];
                    int is_video=ref2va&&params->references[ref].kind!=H3_REFERENCE_IMAGE;
                    size_t blocks=is_video?presentations[ref].vision_count:1;
                    if(ref2va)presentations[ref].vision=&vision_outputs[cursor];
                    for(size_t block=0;block<blocks;block++,cursor++) {
                        if(is_video!=video_batch)continue;
                        const float *pixels=condition_pixels[image];
                        if(is_video) {
                            int first,second;double timestamp;
                            ok=h3_refvideo_qwen_pair(condition_frames[image],block,&first,&second,&timestamp);
                            if(ok)owned[count]=h3_refvideo_extract_pair(pixels,condition_frames[image],condition_heights[image],condition_widths[image],first,second);
                            if(!ok||!owned[count]){ok=0;break;}pixels=owned[count];
                        }
                        inputs[count]=(h3_vision_input){pixels,is_video?2:1,condition_heights[image],condition_widths[image]};
                        destinations[count++]=cursor;
                    }
                }
                if(ok&&count)ok=h3_vision_encode_batch_bf16(text_path,"src/metal/shaders.metal",inputs,count,
                    h3_vision_progress_bridge,&progress,encoded,detail,sizeof(detail));
                for(size_t i=0;i<count;i++) {
                    free(owned[i]);
                    if(ok)vision_outputs[destinations[i]]=encoded[i];else h3_vision_output_free(&encoded[i]);
                }
                if(!ok){h3_set_error(ctx,"packed reference vision: %s",*detail?detail:"cannot prepare input batch");goto cleanup;}
            }
            vision_cursor=vision_output_count;
            for(size_t image=0;image<visual_count;image++){free(condition_pixels[image]);condition_pixels[image]=NULL;}
        } else {
        for (size_t image = 0; image < visual_count; image++) {
            size_t reference_index = visual_reference_indices[image];
            if (!ref2va ||
                params->references[reference_index].kind == H3_REFERENCE_IMAGE) {
                if (!h3_vision_encode_bf16(
                        text_path, "src/metal/shaders.metal", condition_pixels[image],
                        1, condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail))) {
                    h3_set_error(ctx, "%s", detail);
                    goto cleanup;
                }
                if (ref2va)
                    presentations[reference_index].vision =
                        &vision_outputs[vision_cursor];
                vision_cursor++;
            } else {
                size_t blocks = presentations[reference_index].vision_count;
                presentations[reference_index].vision =
                    &vision_outputs[vision_cursor];
                for (size_t block = 0; block < blocks; block++) {
                    int first, second; double timestamp;
                    if (!h3_refvideo_qwen_pair(condition_frames[image], block,
                            &first, &second, &timestamp)) {
                        h3_set_error(ctx, "invalid Qwen reference-video pair"); goto cleanup;
                    }
                    float *pair = h3_refvideo_extract_pair(
                        condition_pixels[image], condition_frames[image],
                        condition_heights[image], condition_widths[image],
                        first, second);
                    if (!pair) {
                        h3_set_error(ctx,
                            "out of memory extracting Qwen video pair");
                        goto cleanup;
                    }
                    int ok = h3_vision_encode_bf16(
                        text_path, "src/metal/shaders.metal", pair, 2,
                        condition_heights[image], condition_widths[image],
                        h3_vision_progress_bridge, &progress,
                        &vision_outputs[vision_cursor], detail, sizeof(detail));
                    free(pair);
                    if (!ok) {
                        h3_set_error(ctx, "%s", detail);
                        goto cleanup;
                    }
                    vision_cursor++;
                    if (progress.cancelled) goto cleanup;
                }
            }
            free(condition_pixels[image]);
            condition_pixels[image] = NULL;
            if (progress.cancelled) goto cleanup;
        }
        }
        if (vision_cursor != vision_output_count) {
            h3_set_error(ctx, "Qwen reference vision count mismatch");
            goto cleanup;
        }
        if (h3_progress_emit(&progress, "reference vision preparation",
                             (int)vision_cursor, (int)vision_output_count)) goto cleanup;
        if (h3_progress_emit(&progress, "text encoder", 0, 50)) goto cleanup;
        int text_ok = ref2va ? h3_multimodal_encode_ref2va_bf16(
                tokenizer, text_path, "src/metal/shaders.metal", prompt,
                presentations, params->reference_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail)) :
            h3_multimodal_encode_fl2va_bf16(
                tokenizer, text_path, "src/metal/shaders.metal", prompt,
                vision_outputs, visual_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail));
        if (!text_ok) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        for (size_t image = 0; image < vision_output_count; image++)
            h3_vision_output_free(&vision_outputs[image]);
    } else {
        if (!h3_tokenizer_encode(tokenizer, prompt, 1, &ids, &token_count,
                                 detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        if (h3_progress_emit(&progress, "text encoder", 0, 50)) goto cleanup;
        if (!h3_text_encode_bf16(
                text_path, "src/metal/shaders.metal", ids, token_count,
                h3_text_progress_bridge, &progress, &text,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    conditioned = visual_count != 0 || condition_audio_elements != 0;
    if(params->_arithmetic_recipe && !h3_sglang_import_text(ids,token_count,text.values,text.width)) {
        h3_set_error(ctx,"invalid SGLang teacher text fixture or mismatched token IDs");goto cleanup;
    }
    const uint32_t *reference_tokens=ids?ids:text.diagnostics?text.diagnostics->ids:NULL;
    size_t reference_token_count=ids?token_count:text.diagnostics?text.diagnostics->tokens:0;
    if(!h3_sglang_dump("text.bf16",text.values,text.tokens*text.width*2) ||
       (reference_tokens&&!h3_sglang_dump("tokens.u32",reference_tokens,reference_token_count*sizeof(*reference_tokens)))) {
        h3_set_error(ctx,"cannot capture SGLang conditioning diagnostics");goto cleanup;
    }
    if (ctx->cache_enabled) {
        if (!h3_conditioning_cache_store(
                ctx, conditioning_key, &text,
                condition_video_rows, condition_video_elements,
                condition_audio_rows, condition_audio_elements,
                layout_references, ref2va ? params->reference_count : 0,
                conditioned))
            fprintf(stderr, "h3cli: warning: could not retain conditioning cache\n");
        else
            H3_VERBOSE("h3cli: conditioning cache miss; stored exact BF16\n");
    }
    }
    if(params->save_conditioning) {
        saved_conditioning=calloc(1,sizeof(*saved_conditioning));
        if(!saved_conditioning) {h3_set_error(ctx,"cannot allocate conditioning snapshot");goto cleanup;}
        h3_conditioning *s=saved_conditioning;
        s->identity=strdup(disk_conditioning_identity);s->conditioned=conditioned;
        s->video_elements=condition_video_elements;s->audio_elements=condition_audio_elements;
        s->reference_count=params->reference_count;s->keyframe_count=keyframe_count;
        memcpy(s->keyframes,keyframes,sizeof(keyframes));
        s->video=condition_video_elements?malloc(condition_video_elements*sizeof(float)):NULL;
        s->audio=condition_audio_elements?malloc(condition_audio_elements*sizeof(float)):NULL;
        s->references=s->reference_count?malloc(s->reference_count*sizeof(*s->references)):NULL;
        if(!s->identity||!h3_text_embedding_copy(&s->text,&text)||
            (s->video_elements&&!s->video)||(s->audio_elements&&!s->audio)||(s->reference_count&&!s->references)) {
            h3_set_error(ctx,"cannot capture raw conditioning");goto cleanup;
        }
        if(s->video_elements)memcpy(s->video,condition_video_rows,s->video_elements*sizeof(float));
        if(s->audio_elements)memcpy(s->audio,condition_audio_rows,s->audio_elements*sizeof(float));
        if(s->reference_count)memcpy(s->references,layout_references,s->reference_count*sizeof(*s->references));
        if(!s->text.diagnostics && ids && token_count) {
            s->text.diagnostics=calloc(1,sizeof(*s->text.diagnostics));
            h3_text_diagnostics *d=s->text.diagnostics;
            if(!d){h3_set_error(ctx,"cannot save tokenizer diagnostics");goto cleanup;}
            d->tokens=token_count;d->ids=malloc(token_count*4);d->positions=malloc(token_count*3*4);
            if(!d->ids||!d->positions){h3_set_error(ctx,"cannot save token positions");goto cleanup;}
            memcpy(d->ids,ids,token_count*4);
            for(size_t axis=0;axis<3;axis++)for(size_t i=0;i<token_count;i++)d->positions[axis*token_count+i]=(uint32_t)i;
        }
    }
    if(params->save_upscale_state) {
        up_raw_video=condition_video_elements?malloc(condition_video_elements*sizeof(float)):NULL;
        up_raw_audio=condition_audio_elements?malloc(condition_audio_elements*sizeof(float)):NULL;
        if((condition_video_elements&&!up_raw_video)||(condition_audio_elements&&!up_raw_audio)) {
            h3_set_error(ctx,"cannot capture raw upscale conditioning");goto cleanup;
        }
        if(up_raw_video)memcpy(up_raw_video,condition_video_rows,condition_video_elements*sizeof(float));
        if(up_raw_audio)memcpy(up_raw_audio,condition_audio_rows,condition_audio_elements*sizeof(float));
        up_record.keyframe_count=(uint32_t)keyframe_count;
        memcpy(up_record.keyframes,keyframes,sizeof(keyframes));
        for(size_t i=0;i<params->reference_count;i++)if(layout_references[i].kind!=H3_LAYOUT_REF_AUDIO) {
            if(!h3_ffprobe_visual_size(params->references[i].path,&up_record.original_width[i],
                &up_record.original_height[i],ctx->error,sizeof(ctx->error)))goto cleanup;
            up_record.semantic_width[i]=layout_references[i].latent_w*16;
            up_record.semantic_height[i]=layout_references[i].latent_h*16;
        }
    }
    if (!resume && conditioned && !h3_augment_conditions(
            params, ref2va, render_width, render_height, layout_references,
            condition_video_rows, condition_video_elements,
            condition_audio_rows, condition_audio_elements)) {
        h3_set_error(ctx, "cannot apply seeded condition augmentation");
        goto cleanup;
    }
    if (progress.cancelled) goto cleanup;

    h3_layout_spec spec = {(int)text.tokens, temporal.video_t, latent_h,
                           latent_w, temporal.audio_t, temporal.frame_count,
                           keyframes, keyframe_count,
                           layout_references,
                           ref2va ? params->reference_count : 0};
    if (resume) {
        layout=resume->layout;
        layout.segments=malloc(layout.segment_count*sizeof(*layout.segments));
        layout.positions=malloc(layout.seq_len*sizeof(*layout.positions));
        if (!layout.segments || !layout.positions) { h3_set_error(ctx,"cannot restore resolved checkpoint layout"); goto cleanup; }
        memcpy(layout.segments,resume->layout.segments,layout.segment_count*sizeof(*layout.segments));
        memcpy(layout.positions,resume->layout.positions,layout.seq_len*sizeof(*layout.positions));
        if (bridge_mode) layout.bridge=&bridge;
    } else if (!(params->still ? h3_still_layout_build(&spec, &layout, detail, sizeof(detail)) : h3_layout_build(&spec, &layout, detail, sizeof(detail)))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    layout.prefix = prefix;
    if(bridge_mode)layout.bridge=&bridge;
    if(!h3_approximate_layout_valid(params,has_continuation,context_frames,&layout,detail,sizeof(detail))) {
        h3_set_error(ctx,"%s",detail);goto cleanup;
    }
    h3_adaptive_plan adaptive_plan;
    if(!h3_adaptive_plan_recipe(params->adaptive_cache,delivery.adaptive_version,layout.seq_len,5376,params->adaptive_cache_max_bytes,
        &adaptive_plan,detail,sizeof(detail))) {h3_set_error(ctx,"%s",detail);goto cleanup;}
    if(params->adaptive_cache)H3_VERBOSE("h3cli: adaptive resources source=%s ceiling=%llu device=%zu persistent=%zu rows=%zu\n",
        resume?"restored/override":params->adaptive_cache_max_bytes?"explicit":"default",
        (unsigned long long)h3_adaptive_budget(params->adaptive_cache_max_bytes),adaptive_plan.bytes,adaptive_plan.persistent_bytes,layout.seq_len);
    if (bridge_mode) {
        layout.bridge = &bridge;
        H3_VERBOSE("h3cli: continuation mode=bridge, profile=%s, max strength=%.6g\n"
            "h3cli: bridge=%d video steps / %d frames (%.6f s); exact=%d steps / %d frames (%.6f s)\n"
            "h3cli: audio bridge ticks=[0,%d), exact=[%d,%d); audio bridge duration=%.6f s\n"
            "h3cli: video bridge masks:",h3_bridge_profile_name(bridge.type),(double)bridge.max_strength,
            bridge.video_bridge_t,bridge.bridge_frames,(double)bridge.bridge_frames/H3_FPS,
            bridge.video_exact_t,context_frames-bridge.bridge_frames,(double)(context_frames-bridge.bridge_frames)/H3_FPS,
            bridge.audio_bridge_t,bridge.audio_bridge_t,bridge.prefix.audio_prefix_t,(double)bridge.audio_bridge_t/H3_AUDIO_LATENT_FPS);
        for (int t=0;t<prefix.video_prefix_t;t++) H3_VERBOSE(" %.3g",(double)bridge.class_mask[bridge.video_classes[t]]);
        H3_VERBOSE("\n");
        size_t counts[H3_TARGET_ROW_CLASSES]={0};
        for (size_t seg=0;seg<layout.segment_count;seg++) {
            const h3_segment *s=&layout.segments[seg];
            if (s->kind!=H3_SEG_VIDEO && s->kind!=H3_SEG_AUDIO) continue;
            for (size_t row=s->start;row<s->stop;row++) {
                h3_target_row_class c=h3_bridge_row_class(&bridge,s->kind==H3_SEG_AUDIO,
                    row-s->start,latent_h,latent_w,temporal.audio_t);
                if (c>=H3_TARGET_ROW_CLASSES) { h3_set_error(ctx,"invalid bridge target row"); goto cleanup; }
                counts[c]++;
            }
        }
        for (int c=0;c<H3_TARGET_ROW_CLASSES;c++) if (counts[c])
            H3_VERBOSE("h3cli: bridge class %d mask %.6g packed rows %zu\n",c,(double)bridge.class_mask[c],counts[c]);
        if (getenv("H3_PROFILE")) fprintf(stderr,"h3cli: bridge modulation maps: %zu bytes (%zu extra versus hard); compact temporal classes only\n",
            (size_t)params->steps*(layout.seq_len+layout.img_target_rows+layout.audio_target_rows)*sizeof(uint32_t),(size_t)0);
    } else if (has_continuation) H3_VERBOSE("h3cli: continuation mode=hard\n");
    h3_sigma_schedule sigmas;
    if (resume) sigmas=resume->sigmas;
    else if (!(params->_arithmetic_recipe ? h3_sglang_schedule(params->steps,&sigmas) : h3_serving_schedule_build(params->steps, &sigmas))) {
        h3_set_error(ctx, "cannot construct the requested sigma schedule");
        goto cleanup;
    }
    float spatial_rope_scale = !params->use_reference_rope &&
        render_width == 256 && render_height == 256 ? 0.5f : 1.0f;
    int conditioning_refvideo=0;
    if(ref2va)
        for(size_t i=0;i<params->reference_count;i++)if(layout_references[i].kind==H3_LAYOUT_REF_VIDEO)conditioning_refvideo=1;
    uint8_t conditioning_schedule_key[32];
    h3_conditioning_schedule_key(&sigmas,&layout,conditioning_refvideo,conditioning_schedule_key);
    if(loaded_conditioning&&!loaded_conditioning->has_schedule&&params->conditioning_schedule&&!params->save_conditioning) {
        h3_set_error(ctx,"h3cond: cache has no schedule-specific records; regenerate with --conditioning-schedule or omit it to rebuild AdaLN");goto cleanup;
    }
    if(loaded_conditioning&&loaded_conditioning->has_schedule&&
        memcmp(loaded_conditioning->schedule_key,conditioning_schedule_key,32)) {
        if(params->conditioning_schedule){h3_set_error(ctx,"h3cond: complete sigma schedule or conditioning time classes mismatch; omit --conditioning-schedule to rebuild AdaLN");goto cleanup;}
        loaded_conditioning->has_schedule=0;
        H3_VERBOSE("h3cli: h3cond schedule miss; invariant conditioning reused, AdaLN rebuilt\n");
    }
    if(getenv("H3_PROFILE")&&*getenv("H3_PROFILE")&&strcmp(getenv("H3_PROFILE"),"0")) {
        uint8_t digest[32];
        const void *inputs[]={text.values,text.tags,condition_video_rows,condition_audio_rows,layout.positions};
        const size_t sizes[]={text.tokens*text.width*2,text.tags?text.tokens:0,condition_video_elements*4,condition_audio_elements*4,layout.seq_len*sizeof(*layout.positions)};
        const char *names[]={"raw-text","text-tags","augmented-video","augmented-audio","positions"};
        fprintf(stderr,"h3cli: conditioning-inputs sequence=%zu",layout.seq_len);
        for(size_t i=0;i<5;i++) {
            h3_sampler_hash(inputs[i],sizes[i],digest);fprintf(stderr," %s=",names[i]);
            for(size_t j=0;j<32;j++)fprintf(stderr,"%02x",digest[j]);
        }
        fputc('\n',stderr);
    }
    h3_conditioning_exchange(loaded_conditioning);
    if(!h3_sglang_import_conditions(condition_video_rows,condition_video_elements,condition_audio_rows,condition_audio_elements)) {
        h3_set_error(ctx,"cannot import bounded reference condition fixture");goto cleanup;
    }
    if(!h3_sglang_dump("positions.f64",layout.positions,layout.seq_len*sizeof(*layout.positions)) ||
       !h3_sglang_dump("sigmas-video.f32",sigmas.video,((size_t)sigmas.steps+1)*sizeof(float)) ||
       !h3_sglang_dump("sigmas-audio.f32",sigmas.audio,((size_t)sigmas.steps+1)*sizeof(float))) {
        h3_set_error(ctx,"cannot capture SGLang layout diagnostics");goto cleanup;
    }
    if(!h3_sglang_dump("condition-video.f32",condition_video_rows,condition_video_elements*sizeof(float)) ||
       !h3_sglang_dump("condition-audio.f32",condition_audio_rows,condition_audio_elements*sizeof(float))) {
        h3_set_error(ctx,"cannot capture reference condition rows");goto cleanup;
    }
    if(params->_arithmetic_recipe && getenv("H3_TEST_SGLANG_INPUTS_ONLY")) {
        h3_set_error(ctx,"test-only reference input capture complete; no denoising executed");goto cleanup;
    }
    if (checkpoint && !resume) {
        size_t nv=(size_t)24*temporal.video_t*latent_h*latent_w, na=(size_t)64*temporal.audio_t;
        sampler=h3_sampler_state_create(&sigmas,nv,na,params->denoise_reuse);
        if (!sampler || !h3_sampler_state_capture_effective(sampler,ctx->model_dir,ctx->lora_variants[ref2va].transformer,&ctx->device,params,prompt,
            &text,&layout,layout_references,ref2va?params->reference_count:0,
            condition_video_rows,condition_video_elements,condition_audio_rows,condition_audio_elements,
            conditioned,ctx->av_signature[ref2va],ctx->error,sizeof(ctx->error))) {
            if (!sampler) h3_set_error(ctx,"cannot allocate sampler state");
            goto cleanup;
        }
        if (ids && token_count) {
            sampler->token_ids=malloc(token_count*sizeof(*ids));
            if (!sampler->token_ids) { h3_set_error(ctx,"cannot capture tokenizer diagnostics"); goto cleanup; }
            memcpy(sampler->token_ids,ids,token_count*sizeof(*ids)); sampler->token_count=token_count;
        }
    }
    if (h3_progress_emit(&progress, "DiT initialization", 0, 1)) goto cleanup;
    if (resume_live || (ctx->cache_enabled && ctx->dit && ctx->dit_key &&
        !strcmp(ctx->dit_key, prepared_key))) {
        dit = ctx->dit;
        dit_is_cached = 1;
        if (!h3_dit_reset_run(
                dit, condition_video_rows, condition_video_elements,
                condition_audio_rows, condition_audio_elements,
                detail, sizeof(detail))) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        H3_VERBOSE("h3cli: prepared DiT cache hit\n");
        if (h3_progress_emit(&progress, "DiT initialization", 1, 1)) goto cleanup;
    } else if (resume) {
        dit=h3_dit_load_resume(dit_path,"src/metal/shaders.metal",resume,h3_dit_progress_bridge,&progress,detail,sizeof(detail));
    } else if (conditioned) {
        int released_refvideo = 0;
        if (ref2va)
            for (size_t i = 0; i < params->reference_count; i++)
                if (layout_references[i].kind == H3_LAYOUT_REF_VIDEO) released_refvideo = 1;
        dit = h3_dit_load_conditioned(
            dit_path, "src/metal/shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers, (unsigned)params->core_reuse,
            params->token_reduction,
            params->ssd_streaming,
            spatial_rope_scale,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            params->use_int8_row_fc2,
            condition_video_rows, condition_video_elements,
            condition_audio_rows, condition_audio_elements,
            released_refvideo,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    } else {
        dit = h3_dit_load_t2va(
            dit_path, "src/metal/shaders.metal", &text, &layout, &sigmas,
            (unsigned)params->dit_layers, (unsigned)params->core_reuse,
            params->token_reduction,
            params->ssd_streaming,
            spatial_rope_scale,
            params->use_slower_bf16_mlp,
            params->use_slower_bf16_qkv,
            params->use_slower_bf16_attention_output,
            params->use_slower_row_major_attention_output,
            params->use_slower_unfused_int8_inputs,
            params->use_slower_unfused_qkv_rope,
            params->use_slower_scalar_qkv_rms,
            params->use_slower_uncached_int8_scales,
            params->use_slower_dynamic_fc1_k,
            params->use_slower_grouped_quantizer,
            params->use_int8_row_fc2,
            h3_dit_progress_bridge, &progress, detail, sizeof(detail));
    }
    if (!dit) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (ctx->cache_enabled && !dit_is_cached) {
        char *key_copy = strdup(prepared_key);
        if (!key_copy) {
            fprintf(stderr, "h3cli: warning: could not retain prepared DiT key\n");
        } else {
            ctx->dit = dit;
            ctx->dit_key = key_copy;
            dit_is_cached = 1;
            H3_VERBOSE("h3cli: prepared DiT cache miss; model retained\n");
        }
    }
    if(sampler) {
        int ok=resume ? h3_dit_sampler_import(dit,sampler,detail,sizeof(detail)) :
            h3_dit_sampler_export(dit,sampler,0,detail,sizeof(detail));
        if(!ok) { h3_set_error(ctx,"%s",detail); goto cleanup; }
        if(dit_is_cached) ctx->dit_sampler_key_ready=h3_sampler_prepared_key(sampler,ctx->dit_sampler_key);
    } else ctx->dit_sampler_key_ready=0;
    if(saved_conditioning) {
        saved_conditioning->has_schedule=params->conditioning_schedule;
        memcpy(saved_conditioning->schedule_key,conditioning_schedule_key,32);
        if(!h3_dit_conditioning_export(dit,saved_conditioning,ctx->error,sizeof(ctx->error)) ||
            !h3_conditioning_save(saved_conditioning,params->save_conditioning,ctx->error,sizeof(ctx->error)))goto cleanup;
        fprintf(stderr,"h3cli: conditioning saved atomically to %s (%s)\n",params->save_conditioning,
            params->conditioning_schedule?"invariants + exact schedule":"invariants");
    }
    h3_conditioning_exchange(previous_conditioning);
    h3_text_embedding_free(&text);
    free(condition_video_rows);
    condition_video_rows = NULL;
    free(condition_audio_rows);
    condition_audio_rows = NULL;
    if (progress.cancelled) goto cleanup;
    if (params->preview_denoise) {
        if (!tiny_decoder) {
        if (h3_progress_emit(&progress, "preview VAE load", 0, 36)) goto cleanup;
        preview_decoder = h3_acquire_video_decoder(
            ctx, decoder_key, vae_path, latent_h, latent_w,
            h3_preview_vae_progress_bridge, &progress,
            &decoder_is_cached, detail, sizeof(detail));
        if (preview_decoder && ctx->video_decoder == preview_decoder)
            if (h3_progress_emit(&progress, "preview VAE load", 36, 36)) goto cleanup;
        if (!preview_decoder) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
        }
        live_preview.tiny = tiny_decoder;
        live_preview.progress = &progress;
        live_preview.decoder = preview_decoder;
        live_preview.latent_t = temporal.video_t;
        live_preview.latent_h = latent_h;
        live_preview.latent_w = latent_w;
        live_preview.output_frames = temporal.frame_count;
        live_preview.trim_frames = params->keep_continuation_prefix ? 0 : context_frames;
        live_preview.output_frames -= live_preview.trim_frames;
        live_preview.output_width = params->width;
        live_preview.output_height = params->height;
        if (progress.cancelled) goto cleanup;
    }
    size_t video_count = h3_dit_video_elements(dit);
    size_t audio_count = h3_dit_audio_elements(dit);
    video = malloc(video_count * sizeof(*video));
    audio = malloc(audio_count * sizeof(*audio));
    if (!video || !audio) {
        h3_set_error(ctx, "out of memory allocating joint H3 noise");
        goto cleanup;
    }
    /* The released server initializes each modality from a separate generator
     * carrying the same requested seed. */
    if (!resume) {
    h3_rng video_rng, audio_rng;
    h3_rng_seed(&video_rng, params->seed);
    h3_rng_seed(&audio_rng, params->seed);
    if(params->_arithmetic_recipe) {
        float *audio_rows=malloc(audio_count*sizeof(float));
        int ok=h3_sglang_normal(params->seed,video,video_count) && audio_rows &&
            h3_sglang_normal(params->seed,audio_rows,audio_count) &&
            h3_dit_unpack_audio(audio_rows,32,temporal.audio_t,audio,audio_count);
        free(audio_rows);
        if(!ok){h3_set_error(ctx,"SGLang CPU RNG requires x86 AVX2/FMA and valid latent geometry");goto cleanup;}
    } else {
        h3_rng_fill_normal(&video_rng, video, video_count);
        h3_rng_fill_normal(&audio_rng, audio, audio_count);
    }
    if(!h3_sglang_dump("initial-video.f32",video,video_count*sizeof(float)) ||
       !h3_sglang_dump("initial-audio.f32",audio,audio_count*sizeof(float))) {
        h3_set_error(ctx,"cannot capture initial AV noise");goto cleanup;
    }
    if (sampler) {
        sampler->video_rng=video_rng; sampler->audio_rng=audio_rng;
        sampler->video_random_count=video_count; sampler->audio_random_count=audio_count;
        sampler->original_video_noise=malloc(video_count*sizeof(float));
        sampler->original_audio_noise=malloc(audio_count*sizeof(float));
        if (!sampler->original_video_noise || !sampler->original_audio_noise) { h3_set_error(ctx,"cannot retain original AV noise"); goto cleanup; }
        memcpy(sampler->original_video_noise,video,video_count*sizeof(float));
        memcpy(sampler->original_audio_noise,audio,audio_count*sizeof(float));
    }
    if (params->continuation) {
        double copy_start = h3_av_now();
        h3_av_state_info target;
        if (!h3_av_state_shape(render_width,render_height,temporal.frame_count,&target) ||
            !(bridge_mode ? h3_av_state_insert_bridge(params->continuation,&target,&bridge,
                sigmas.video[0],sigmas.audio[0],video,audio) :
                h3_av_state_insert_prefix(params->continuation,&target,prefix,video,audio,1))) {
            h3_set_error(ctx,"cannot initialize continuation latent prefix"); goto cleanup;
        }
        if (getenv("H3_PROFILE")) fprintf(stderr,"h3cli: %s initialization %.6f s; existing target noise, no full mask tensors\n",
            bridge_mode ? "bridge AV" : "continuation prefix",h3_av_now()-copy_start);
    }
    if (sampler) { memcpy(sampler->video,video,video_count*sizeof(float)); memcpy(sampler->audio,audio,audio_count*sizeof(float)); }
    }
    h3_dit_set_latent_callback(dit,params->on_latent_step,params->callback_opaque);
    int denoise_ok = sampler ? h3_dit_denoise_euler_range(dit,sampler,
            params->stop_after_step>=0?params->stop_after_step:params->steps,
            h3_dit_progress_bridge,&progress,
            params->preview_denoise?h3_deliver_denoise_preview:NULL,
            params->preview_denoise?&live_preview:NULL,detail,sizeof(detail)) :
        h3_dit_denoise_euler_preview(
            dit, video, audio, params->denoise_reuse,
            h3_dit_progress_bridge, &progress,
            params->preview_denoise ? h3_deliver_denoise_preview : NULL,
            params->preview_denoise ? &live_preview : NULL,
            detail, sizeof(detail));
    if (!denoise_ok) {
        if (!live_preview.failed) h3_set_error(ctx, "%s", detail);
        if(sampler&&sampler->upscale.stage==2&&params->save_sampler_state) {
            char recovery[512]="audio identity differs from the saved refinement policy";
            if(h3_upscale_audio_intact(sampler)&&h3_dit_sampler_export(dit,sampler,1,recovery,sizeof(recovery))&&
               h3_sampler_state_save(sampler,params->save_sampler_state,recovery,sizeof(recovery)))
                fprintf(stderr,"h3cli: refinement recovery checkpoint saved at boundary %d\n",sampler->next_step);
            else fprintf(stderr,"h3cli: refinement recovery save failed: %s; earlier atomic checkpoint retained\n",recovery);
        }
        if (dit_is_cached) {
            ctx->dit = NULL;
            free(ctx->dit_key);
            ctx->dit_key = NULL;
            dit_is_cached = 0;
        }
        goto cleanup;
    }
    if (sampler) {
        if(!h3_dit_sampler_export(dit,sampler,1,ctx->error,sizeof(ctx->error))) goto cleanup;
        memcpy(video,sampler->video,video_count*sizeof(float)); memcpy(audio,sampler->audio,audio_count*sizeof(float));
        paused=sampler->next_step<sampler->total_steps;
        if (params->save_sampler_state) {
            if(!h3_sampler_state_save(sampler,params->save_sampler_state,ctx->error,sizeof(ctx->error)))goto cleanup;
            if(ctx->lora) {
                char *provenance=h3_lora_provenance(ctx->lora,&ctx->lora_variants[ref2va],ctx->model_dir,ref2va);
                int saved=provenance&&h3_lora_save_provenance(params->save_sampler_state,provenance,ctx->error,sizeof(ctx->error));
                free(provenance);if(!saved)goto cleanup;
                H3_VERBOSE("h3cli: sampler runtime LoRA key %s; resume with the same ordered --lora arguments\n",ctx->lora_variants[ref2va].key);
            }
        }
    }
    /* Export the complete clean source before any delivery operation can fail.
     * The serializer only borrows this view; runtime sampler buffers are intact. */
    if(params->save_upscale_state && (!sampler || paused ||
        !h3_upscale_capture(sampler,up_raw_video,up_raw_audio,&up_record,
            params->save_upscale_state,ctx->error,sizeof(ctx->error))))goto cleanup;
    if(params->still && dit_is_cached) {
        ctx->dit=NULL;free(ctx->dit_key);ctx->dit_key=NULL;ctx->dit_sampler_key_ready=0;dit_is_cached=0;
    }
    #ifdef __APPLE__
    /* Completed latents/checkpoints own everything required for delivery or
     * continuation. Production VAE must not overlap a retained transformer.
     * A paused invocation keeps its prepared cache for immediate resumption. */
    if(dit_is_cached&&!paused&&!tiny_decoder) {
        ctx->dit=NULL;free(ctx->dit_key);ctx->dit_key=NULL;ctx->dit_sampler_key_ready=0;
        dit_is_cached=0;
        H3_VERBOSE("h3cli: releasing completed DiT cache before production VAE\n");
    }
    #endif
    if (!dit_is_cached) h3_dit_free(dit);
    dit = NULL;
    if(params->still) {
        h3_still_latent *z=calloc(1,sizeof(*z));
        if(!z){h3_set_error(ctx,"cannot retain still latent");goto cleanup;}
        z->height=latent_h;z->width=latent_w;z->values=video;
        memcpy(z->compatibility_sha256,image_info.compatibility_sha256,65);
        if(params->save_still_latent && !h3_still_latent_save(params->save_still_latent,z,ctx->error,sizeof(ctx->error))) {free(z);goto cleanup;}
        h3_decode_options options={.output_path=params->output_path,.output_encoding=params->output_encoding,.on_frame=params->on_frame,
            .on_progress=params->on_progress,.callback_opaque=params->callback_opaque};
        result=h3_decode_still_values(z,params->image_vae,&options,ctx->error,sizeof(ctx->error));
        if(result) {
            result->still_latent=z;video=NULL;result->seed=params->seed;
            result->completed_steps=params->steps;result->total_steps=params->steps;
        } else free(z);
        goto cleanup;
    }

    /* A retained decoder already owns its weights/scratch. Reserve only the
     * audio/transfer working space in that case; a paused run needs no decode. */
    uint64_t decode_headroom = tiny_decoder ?
        h3_tiny_vae_memory_reserve(latent_h,latent_w,5)+(UINT64_C(1)<<30) :
        (ctx->video_decoder || preview_decoder) ? (UINT64_C(2)<<30) : (UINT64_C(12)<<30);
    if (ctx->dit && (!paused || params->preview_on_stop) &&
        !h3_device_memory_fits(decode_headroom) &&
        !h3_dit_release_weight_cache(ctx->dit, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (ctx->dit && (!paused || params->preview_on_stop) &&
        !h3_device_memory_fits(decode_headroom)) {
        h3_dit_free(ctx->dit); ctx->dit = NULL;
        free(ctx->dit_key); ctx->dit_key = NULL; ctx->dit_sampler_key_ready = 0;
        dit_is_cached = 0;
        H3_VERBOSE("h3cli: evicted prepared DiT cache to make room for decoding\n");
    }
    if (paused && !params->preview_on_stop) {
        result=calloc(1,sizeof(*result));
        if (!result) { h3_set_error(ctx,"cannot allocate paused result"); goto cleanup; }
        result->status=H3_RESULT_PAUSED; result->completed_steps=sampler->next_step; result->total_steps=sampler->total_steps;
        result->width=params->width; result->height=params->height; result->fps=H3_FPS; result->seed=params->seed;
        result->sampler_state=sampler; sampler=NULL;
        goto cleanup;
    }
    if(params->state_only) {
        captured=h3_av_state_new_profile(render_width,render_height,temporal.frame_count,params->geometry_profile,params->seed,ctx->av_signature[ref2va]);
        result=calloc(1,sizeof(*result));
        if(!captured||!result){free(result);result=NULL;h3_set_error(ctx,"cannot retain state-only result");goto cleanup;}
        memcpy(captured->video,video,video_count*sizeof(float));
        memcpy(captured->audio,audio,audio_count*sizeof(float));
        result->status=H3_RESULT_COMPLETE;result->completed_steps=params->steps;result->total_steps=params->steps;
        result->width=params->width;result->height=params->height;result->frames=temporal.frame_count;
        result->fps=H3_FPS;result->seed=params->seed;result->presentation=delivery;
        result->av_state=captured;captured=NULL;
        goto cleanup;
    }
    if (progress.cancelled) goto cleanup;
    if (!paused) {
    if (h3_progress_emit(&progress, "audio VAE", 0, 7)) goto cleanup;
    captured = calloc(1,sizeof(*captured));
    if (!captured || !h3_av_state_shape_profile(render_width,render_height,temporal.frame_count,params->geometry_profile,&captured->info)) {
        free(captured); captured = NULL;
        h3_set_error(ctx,"cannot retain final AV state"); goto cleanup;
    }
    captured->info.seed = params->seed;
    memcpy(captured->info.compatibility,ctx->av_signature[ref2va],32);
    captured->video = video; captured->audio = audio;
    if (getenv("H3_PROFILE")) fprintf(stderr,"h3cli: retained complete AV state: %zu payload bytes (ownership transfer, no copy)\n",
        (video_count+audio_count)*sizeof(float));
    if (!h3_audio_vae_decode(audio_vae_path, "src/metal/shaders.metal", audio,
                             temporal.audio_t, h3_audio_vae_progress_bridge,
                             &progress, &waveform, detail, sizeof(detail))) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (progress.cancelled) goto cleanup;
    }
    if (tiny_decoder) {
        h3_decode_options options={.output_path=params->output_path,.output_encoding=params->output_encoding,
            .on_frame=params->on_frame,.on_progress=params->on_progress,.callback_opaque=params->callback_opaque};
        if (!h3_deliver_video(tiny_decoder,NULL,video,temporal.video_t,latent_h,latent_w,
                &delivery,&waveform,&options,detail,sizeof(detail))) {
            h3_set_error(ctx,"%s",detail); goto cleanup;
        }
        frames.frames=temporal.frame_count-delivery.trim_frames;
        goto delivered;
    }
    int stream_full_vae=params->_arithmetic_recipe;
    if (!preview_decoder && (ctx->cache_enabled || stream_full_vae)) {
        if (h3_progress_emit(&progress, "video VAE load", 0, 36)) goto cleanup;
        preview_decoder = h3_acquire_video_decoder(
            ctx, decoder_key, vae_path, latent_h, latent_w,
            h3_vae_progress_bridge, &progress,
            &decoder_is_cached, detail, sizeof(detail));
        if (preview_decoder && ctx->video_decoder == preview_decoder)
            if (h3_progress_emit(&progress, "video VAE load", 36, 36)) goto cleanup;
        if (!preview_decoder) {
            h3_set_error(ctx, "%s", detail);
            goto cleanup;
        }
    }
    if(stream_full_vae){
        h3_decode_options options={.output_path=params->output_path,.output_encoding=params->output_encoding,
            .on_frame=params->on_frame?h3_delivery_frame_bridge:NULL,
            .on_progress=h3_dit_progress_bridge,.callback_opaque=&progress};
        if(!h3_deliver_video(NULL,preview_decoder,video,temporal.video_t,latent_h,latent_w,
                &delivery,&waveform,&options,detail,sizeof(detail))){
            h3_set_error(ctx,"%s",detail);goto cleanup;
        }
        frames.frames=temporal.frame_count-delivery.trim_frames;
        goto delivered;
    }
    int video_ok = preview_decoder ?
        h3_video_vae_decoder_decode_progress(
            preview_decoder, video, temporal.video_t,
            h3_vae_decode_progress_bridge, &progress, &frames,
            detail, sizeof(detail)) :
        h3_video_vae_decode_phased(
            vae_path, "src/metal/shaders.metal", video,
            temporal.video_t, latent_h, latent_w,
            h3_vae_progress_bridge, h3_vae_decode_progress_bridge, &progress, &frames,
            detail, sizeof(detail));
    if (!video_ok) {
        h3_set_error(ctx, "%s", detail);
        goto cleanup;
    }
    if (progress.cancelled) goto cleanup;
    size_t rgb_count = (size_t)frames.frames * (size_t)frames.height *
                       (size_t)frames.width * 3;
    rgb8 = h3_rgb_f32_to_u8(frames.rgb, rgb_count);
    if (!rgb8) {
        h3_set_error(ctx, "out of memory converting generated RGB frames");
        goto cleanup;
    }
    output_width = frames.width;
    output_height = frames.height;
    if (output_width != params->width || output_height != params->height) {
        uint8_t *resized = NULL;
        if (!h3_resize_rgb24_high_quality(
                rgb8, frames.frames, output_width, output_height,
                params->width, params->height, &resized)) {
            h3_set_error(ctx, "cannot resize generated RGB frames");
            goto cleanup;
        }
        free(rgb8);
        rgb8 = resized;
        output_width = params->width;
        output_height = params->height;
    }
    if (has_continuation && !params->keep_continuation_prefix) {
        int trim_samples = prefix.audio_prefix_t*800;
        if (frames.frames <= context_frames || (!paused && waveform.samples <= trim_samples)) {
            h3_set_error(ctx,"decoded continuation is shorter than protected prefix"); goto cleanup;
        }
        size_t frame_bytes = (size_t)output_width*(size_t)output_height*3;
        frames.frames -= context_frames;
        memmove(rgb8,rgb8+(size_t)context_frames*frame_bytes,(size_t)frames.frames*frame_bytes);
        int remaining = paused ? 0 : waveform.samples-trim_samples;
        for (int c=0;c<waveform.channels;c++) memmove(waveform.pcm+(size_t)c*(size_t)remaining,
            waveform.pcm+(size_t)c*(size_t)waveform.samples+(size_t)trim_samples,
            (size_t)remaining*sizeof(float));
        waveform.samples = remaining;
    }
    if (params->on_frame) {
        size_t frame_bytes = (size_t)output_width * (size_t)output_height * 3;
        for (int index = 0; index < frames.frames; index++) {
            h3_frame frame = {output_width, output_height, output_width * 3,
                              rgb8 + (size_t)index * frame_bytes,
                              /* Full delivered clips, including a pause preview,
                               * use ordinary frame delivery. The result carries
                               * the pause boundary; per-step previews use their
                               * separate callback path above. */
                              index, frames.frames, -1, 0};
            if (params->on_frame(&frame, params->callback_opaque)) {
                h3_set_error(ctx, "generation cancelled while delivering frame %d",
                             index);
                goto cleanup;
            }
        }
    }
    if (params->output_path && *params->output_path) {
        double mux_start=h3_av_now();
        if (h3_progress_emit(&progress, "FFmpeg", 0, frames.frames)) goto cleanup;
        int encoded = paused ? h3_ffmpeg_write_rgb24(params->output_path,rgb8,frames.frames,
            output_width,output_height,H3_FPS,&params->output_encoding,detail,sizeof(detail)) : h3_ffmpeg_write_av_rgb24_f32(
                params->output_path, rgb8, frames.frames, output_width,
                output_height, H3_FPS, waveform.pcm, waveform.samples,
                waveform.channels, waveform.sample_rate,
                &params->output_encoding, detail, sizeof(detail));
        if (!encoded) { h3_set_error(ctx,"%s",detail); goto cleanup; }
        if (h3_progress_emit(&progress, "FFmpeg", frames.frames, frames.frames)) goto cleanup;
        if(getenv("H3_PROFILE"))fprintf(stderr,"h3cli: FFmpeg mux wall %.6f s\n",h3_av_now()-mux_start);
    }
delivered:
    result = calloc(1, sizeof(*result));
    if (!result) {
        h3_set_error(ctx, "out of memory creating generation result");
        goto cleanup;
    }
    result->status=paused?H3_RESULT_PAUSED:H3_RESULT_COMPLETE;
    result->completed_steps=sampler?sampler->next_step:params->steps; result->total_steps=params->steps;
    if (paused) { result->sampler_state=sampler; sampler=NULL; }
    result->width = output_width;
    result->height = output_height;
    result->frames = frames.frames;
    result->fps = H3_FPS;
    result->sample_rate = waveform.sample_rate;
    result->seed = params->seed;
    result->audio_samples = waveform.samples;
    result->presentation = delivery;
    result->av_state = captured;
    captured = NULL;
    if (!paused) { video = NULL; audio = NULL; }

cleanup:
    h3_conditioning_exchange(previous_conditioning);
    h3_conditioning_free(loaded_conditioning);h3_conditioning_free(saved_conditioning);
    free(up_raw_video);free(up_raw_audio);
    free(disk_conditioning_identity);
    if(result&&ctx->lora) {
        result->lora_provenance=h3_lora_provenance(ctx->lora,&ctx->lora_variants[ref2va],ctx->model_dir,ref2va);
        if(!result->lora_provenance)fprintf(stderr,"h3cli: unable to retain informational LoRA provenance\n");
    }
    if (sampler != resume) h3_sampler_state_free(sampler);
    free(conditioning_key);
    free(prepared_key);
    free(decoder_key);
    free(tokenizer_path); free(text_path); free(dit_path); free(vae_path);
    free(audio_vae_path);
    h3_tokenizer_free(tokenizer);
    h3_tokenizer_ids_free(ids);
    for (size_t image = 0; image < visual_capacity; image++) {
        if (condition_pixels) free(condition_pixels[image]);
    }
    for (size_t image = 0; image < vision_output_count; image++)
        if (vision_outputs) h3_vision_output_free(&vision_outputs[image]);
    for (size_t index = 0; index < params->reference_count; index++)
        if (presentation_timestamps) free(presentation_timestamps[index]);
    free(condition_pixels);
    free(condition_widths);
    free(condition_heights);
    free(condition_frames);
    free(condition_vae_frames);
    free(condition_latent_t);
    free(condition_audio_limits);
    free(visual_reference_indices);
    free(reference_visual_indices);
    free(vision_outputs);
    free(presentations);
    free(presentation_timestamps);
    free(layout_references);
    free(condition_video_rows);
    free(condition_audio_rows);
    h3_text_embedding_free(&text);
    h3_layout_free(&layout);
    if (!dit_is_cached) h3_dit_free(dit);
    if (!decoder_is_cached) h3_video_vae_decoder_free(preview_decoder);
    if (!tiny_is_cached) h3_tiny_vae_free(tiny_decoder);
    if (captured) { h3_av_state_free(captured); video = NULL; audio = NULL; }
    free(video); free(audio); free(rgb8);
    h3_video_frames_free(&frames);
    h3_audio_waveform_free(&waveform);
    if (progress.cancelled || (!result && h3_memory_error(ctx->error)))
        h3_cache_clear(ctx);
    if (progress.cancelled && progress.cancellation_error[0])
        h3_set_error(ctx, "%s", progress.cancellation_error);
    return result;
}

h3_result *h3_generate_upscale_state(h3_ctx *ctx,h3_sampler_state *state,const h3_upscale_options *o) {
    h3_params p=state->params;p.output_path=o->state_only?NULL:o->delivery.output_path;p.output_encoding=o->delivery.output_encoding;p.state_only=o->state_only;
    p.stop_after_step=o->stop_after_step;p.save_sampler_state=o->save_sampler_state;
    p.on_frame=o->delivery.on_frame;p.on_progress=o->delivery.on_progress;p.callback_opaque=o->delivery.callback_opaque;
    p.on_latent_step=o->on_latent_step;
    h3_cuda_policy policy;
    if(!h3_cuda_policy_resolve(&p,ctx->device.backend,0,&policy,ctx->error,sizeof(ctx->error))||
       !h3_cuda_policy_preflight(&policy,ctx->error,sizeof(ctx->error)))return NULL;
    h3_cache_clear(ctx);ctx->dit_sampler_key_ready=0;
    h3_backend_scope old_backend=h3_backend_exchange(H3_BACKEND_SCOPE(&p));
    int old_reference=h3_sglang_exchange(p._arithmetic_recipe),old_attention=h3_attention_exchange(0);
    h3_quant_scope old_quant=h3_quant_exchange((h3_quant_scope){0});
    h3_cuda_policy old_policy=h3_cuda_policy_exchange(policy);
    h3_result *result=h3_generate_state(ctx,state->prompt,&p,state);
    h3_cache_clear(ctx);ctx->dit_sampler_key_ready=0;
    h3_cuda_policy_exchange(old_policy);h3_quant_exchange(old_quant);h3_attention_exchange(old_attention);
    h3_sglang_exchange(old_reference);h3_backend_exchange(old_backend);return result;
}

static h3_result *h3_generate_request(h3_ctx *ctx,const char *prompt,const h3_params *params) {
    if (!ctx || !params) return NULL;
    if(!h3_cuda_sol_params_valid(params,ctx->error,sizeof(ctx->error)))return NULL;
    h3_params still_params;
    if(params->still) {
        still_params=*params;
        still_params.use_slower_bf16_mlp=1;still_params.use_slower_bf16_qkv=1;
        still_params.use_slower_bf16_attention_output=1;still_params.use_reference_rope=1;
        params=&still_params;
    }
    h3_preview_mode preview_mode;
    if (!h3_preview_mode_parse(getenv("H3_PREVIEW_MODE"), &preview_mode,
                               ctx->error, sizeof(ctx->error))) return NULL;
    h3_qwen_gqa_scale_mode scale_mode;
    if (!h3_qwen_gqa_scale_parse(getenv("H3_QWEN_GQA_SCALE_MODE"),
                                 &scale_mode, ctx->error, sizeof(ctx->error))) return NULL;
    if (!params->resume_sampler_state) return h3_generate_state(ctx,prompt,params,NULL);
    h3_sampler_state *state=h3_sampler_state_load_with_budget(params->resume_sampler_state,params->adaptive_cache_max_bytes,params->adaptive_cache_max_bytes_set||params->adaptive_cache_max_bytes,ctx->error,sizeof(ctx->error));
    if (!state) return NULL;
    if(params->state_only&&state->upscale.stage!=2) {h3_set_error(ctx,"state-only resume requires an upscale refinement checkpoint");h3_sampler_state_free(state);return NULL;}
    if(state->upscale.stage==2&&(params->preview_vae||params->preview_denoise||params->preview_on_stop||ctx->lora)) {
        h3_set_error(ctx,"refinement resume requires dense full-VAE delivery without LoRA or previews");h3_sampler_state_free(state);return NULL;
    }
    if(((params->backend_set&1||params->backend)&&params->backend!=state->params.backend)||
       ((params->backend_set&2||params->attention_mode)&&params->attention_mode!=state->params.attention_mode)||
       (((params->backend_set&12)||params->backend)&&!h3_metal_options_equal(params->metal_attention,state->params.metal_attention))) {
        h3_set_error(ctx,"resume backend/attention differs from checkpoint; restart denoising to change backend");
        h3_sampler_state_free(state);return NULL;
    }
    if(!h3_backend_preflight(H3_BACKEND_SCOPE(&state->params),0,ctx->error,sizeof(ctx->error))) {
        h3_sampler_state_free(state);return NULL;
    }
    if ((params->cuda_sol_set && (state->params.cuda_attention!=H3_ATTENTION_SOL ||
         !h3_cuda_sol_options_match(params->cuda_sol,state->params.cuda_sol,params->cuda_sol_set))) ||
        ((params->cuda_attention_set || params->cuda_attention) &&
         params->cuda_attention != state->params.cuda_attention)) {
        h3_set_error(ctx,"resume attention differs from the checkpoint; restart denoising to change attention");
        h3_sampler_state_free(state);return NULL;
    }
    if(((params->adaptive_cache_set||params->adaptive_cache)&&params->adaptive_cache!=state->params.adaptive_cache)||
       (params->subblock_sparsity_set&&params->subblock_sparsity!=state->params.subblock_sparsity)||
       !h3_warmups_match(params,&state->params)||!h3_adaptive_controls_match(params,&state->params)) {
        h3_set_error(ctx,"adaptive/SubBlock policy differs from checkpoint");h3_sampler_state_free(state);return NULL;
    }
    if(!h3_attention_preflight(state->params.cuda_attention,ctx->error,sizeof(ctx->error))) {
        h3_sampler_state_free(state);return NULL;
    }
    if ((params->cuda_denoise_quant_set || params->cuda_denoise_quant) &&
        params->cuda_denoise_quant != state->params.cuda_denoise_quant) {
        h3_set_error(ctx,"resume quantization differs from the checkpoint; restart denoising to change precision");
        h3_sampler_state_free(state); return NULL;
    }
    if (!h3_quant_options(state->params.cuda_denoise_quant,params->cuda_denoise_quant_cache,ctx->error,sizeof(ctx->error)) ||
        !h3_quant_preflight(state->params.cuda_denoise_quant,ctx->error,sizeof(ctx->error))) {
        h3_sampler_state_free(state); return NULL;
    }
    if(!h3_test_evaluation_budget((params->stop_after_step>=0?params->stop_after_step:state->total_steps)-state->next_step,ctx->error,sizeof(ctx->error))) {
        h3_sampler_state_free(state);return NULL;
    }
    h3_cuda_policy resume_policy;
    if(!strcmp(ctx->device.backend,"cuda") && !state->params._arithmetic_recipe) {
        h3_set_error(ctx,"checkpoint arithmetic differs from current CUDA video; restart rendering");
        h3_sampler_state_free(state);return NULL;
    }
    h3_params policy_params=state->params;
    h3_reference policy_refs[12]={0};
    for(size_t i=0;i<state->reference_count;i++) {
        policy_refs[i].kind=state->references[i].kind==H3_LAYOUT_REF_IMAGE?H3_REFERENCE_IMAGE:
            state->references[i].kind==H3_LAYOUT_REF_VIDEO?H3_REFERENCE_VIDEO:H3_REFERENCE_AUDIO;
        policy_refs[i].include_embedded_audio=state->references[i].kind==H3_LAYOUT_REF_VIDEO&&state->references[i].audio_t>0;
    }
    policy_params.references=policy_refs;
    policy_params.preview_vae=params->preview_vae;
    if(!h3_cuda_policy_resolve(&policy_params,ctx->device.backend,ctx->lora!=NULL,&resume_policy,ctx->error,sizeof(ctx->error)) ||
       !h3_cuda_policy_preflight(&resume_policy,ctx->error,sizeof(ctx->error))) {h3_sampler_state_free(state);return NULL;}
    if(!h3_prepare_lora(ctx,state->ref2va,params)){h3_sampler_state_free(state);return NULL;}
    int compatible=h3_sampler_state_compatible_effective(state,ctx->model_dir,ctx->lora_variants[state->ref2va].transformer,&ctx->device,ctx->error,sizeof(ctx->error));
    if (!compatible) { h3_sampler_state_free(state); return NULL; }
    if(state->resume_count==UINT32_MAX) { h3_set_error(ctx,"resume count overflow"); h3_sampler_state_free(state); return NULL; }
    state->resume_count++; state->resume_step=(uint32_t)state->next_step; state->resume_format=H3_SAMPLE_VERSION;
    memcpy(state->resume_hash,state->loaded_hash,32);
    h3_params restored=state->params;
    restored.cuda_denoise_quant_cache=params->cuda_denoise_quant_cache;
    restored.state_only=params->state_only;
    restored.output_path=params->output_path; restored.output_encoding=params->output_encoding; restored.stop_after_step=params->stop_after_step;
    restored.save_sampler_state=params->save_sampler_state; restored.resume_sampler_state=params->resume_sampler_state;
    restored.preview_on_stop=params->preview_on_stop; restored.preview_denoise=params->preview_denoise;
    restored.preview_vae=params->preview_vae; restored.preview_vae_model=params->preview_vae_model;
    restored.on_frame=params->on_frame; restored.on_progress=params->on_progress;
    restored.on_latent_step=params->on_latent_step; restored.callback_opaque=params->callback_opaque;
    h3_cuda_sol_options previous_sol=h3_cuda_sol_exchange(restored.cuda_sol);
    int previous_attention=h3_attention_exchange(restored.cuda_attention);
    int previous_reference=h3_sglang_exchange(restored._arithmetic_recipe);
    h3_quant_scope previous_quant=h3_quant_exchange((h3_quant_scope){restored.cuda_denoise_quant,restored.cuda_denoise_quant_cache});
    h3_backend_scope previous_backend=h3_backend_exchange(H3_BACKEND_SCOPE(&restored));
    resume_policy.preview=restored.preview_vae;
    h3_cuda_policy previous_policy=h3_cuda_policy_exchange(resume_policy);
    h3_result *result=h3_generate_state(ctx,state->prompt,&restored,state);
    h3_cuda_policy_exchange(previous_policy);
    h3_backend_exchange(previous_backend);
    h3_attention_exchange(previous_attention);
    h3_cuda_sol_exchange(previous_sol);
    h3_quant_exchange(previous_quant);
    h3_sglang_exchange(previous_reference);
    if(result) {
        result->resume_count=state->resume_count; result->resume_step=state->resume_step; result->resume_format=state->resume_format;
        memcpy(result->resume_hash,state->resume_hash,32);
        H3_VERBOSE("h3cli: resume provenance: format=%u step=%u operations=%u checkpoint_sha256=",
            result->resume_format,result->resume_step,result->resume_count);
        for (int i = 0; i < 32; i++)
            H3_VERBOSE("%02x", result->resume_hash[i]);
        H3_VERBOSE("\n");
    }
    if (!result || result->sampler_state!=state) h3_sampler_state_free(state);
    return result;
}

h3_result *h3_generate(h3_ctx *ctx,const char *prompt,const h3_params *params) {
    if (!ctx || !params) return NULL;
    if(!h3_upscale_options_valid(params,ctx->error,sizeof(ctx->error)))return NULL;
    if(params->save_upscale_state&&ctx->lora) {h3_set_error(ctx,"upscale source capture does not support LoRA");return NULL;}
    if(!h3_output_encoding_valid(&params->output_encoding,ctx->error,sizeof(ctx->error)))return NULL;
    h3_params resolved=*params;
    if((params->save_upscale_state||params->geometry_profile)&&!strcmp(ctx->device.backend,"metal")) {
        resolved.use_slower_bf16_mlp=1;resolved.use_slower_bf16_qkv=1;resolved.use_slower_bf16_attention_output=1;
    }
    resolved._arithmetic_recipe=h3_sglang_resolve(params,ctx->device.backend,ctx->lora!=NULL);
    params=&resolved;
    if(params->_arithmetic_recipe && (getenv("H3_TEST_SGLANG_INPUT_DIR")||getenv("H3_TEST_SGLANG_CONDITION_DIR")) &&
       (ctx->cache_enabled || params->load_conditioning || params->save_conditioning)) {
        h3_set_error(ctx,"teacher text/condition replay requires uncached conditioning");return NULL;
    }
    if(params->_arithmetic_recipe && !params->resume_sampler_state && getenv("H3_TEST_SGLANG_DIR") &&
       h3_sglang_capture_step(0,params->steps)<0) {
        h3_set_error(ctx,"H3_SGLANG_CAPTURE_STEPS must be none or valid evaluation indices");return NULL;
    }
    h3_cuda_policy policy;
    if(!h3_cuda_policy_resolve(params,ctx->device.backend,ctx->lora!=NULL,&policy,ctx->error,sizeof(ctx->error)) ||
       !h3_cuda_policy_preflight(&policy,ctx->error,sizeof(ctx->error)))return NULL;
    if(!h3_cuda_sol_params_valid(params,ctx->error,sizeof(ctx->error)))return NULL;
    if((params->save_conditioning&&!*params->save_conditioning)||
       (params->load_conditioning&&!*params->load_conditioning)||
       (params->conditioning_schedule!=0&&params->conditioning_schedule!=1)||
       (params->conditioning_schedule&&!params->save_conditioning&&!params->load_conditioning)) {
        h3_set_error(ctx,"conditioning requires nonempty paths and a valid schedule-cache option");return NULL;
    }
    if(!h3_backend_preflight(H3_BACKEND_SCOPE(params),
        params->backend_set,ctx->error,sizeof(ctx->error)) ||
        (params->backend && !h3_sampler_checkpoint_options(params,&ctx->device,
            ctx->error,sizeof(ctx->error))) ||
        (!params->resume_sampler_state && !h3_test_evaluation_budget(
            params->stop_after_step>=0?params->stop_after_step:params->steps,
            ctx->error,sizeof(ctx->error))))return NULL;
    if((params->save_conditioning || params->load_conditioning) && params->resume_sampler_state) {
        h3_set_error(ctx,"conditioning cache cannot be combined with sampler resume");return NULL;
    }
    if(params->save_conditioning&&params->conditioning_schedule&&
       (params->dit_layers!=50||(getenv("H3_TEST_DIT_BLOCKS")&&
       strcmp(getenv("H3_TEST_DIT_BLOCKS"),"50")))) {
        h3_set_error(ctx,"schedule conditioning caches require all 50 blocks");return NULL;
    }
    h3_backend_scope previous_backend=h3_backend_exchange(H3_BACKEND_SCOPE(params));
    if(!h3_attention_preflight(params->cuda_attention,ctx->error,sizeof(ctx->error))) {
        h3_backend_exchange(previous_backend);return NULL;
    }
    if (!h3_quant_options(params->cuda_denoise_quant,
        params->resume_sampler_state?NULL:params->cuda_denoise_quant_cache,ctx->error,sizeof(ctx->error)) ||
        !h3_quant_preflight(params->cuda_denoise_quant,ctx->error,sizeof(ctx->error))) {
        h3_backend_exchange(previous_backend);return NULL;
    }
    if(policy.active && !params->resume_sampler_state)
        H3_VERBOSE("h3cli: CUDA execution: single-pipeline base=sglang-v%d attention=%s quant=%s decoder=%s%s\n",
            policy.base_recipe,h3_attention_name(policy.attention),h3_quant_name(policy.projection_precision),
            policy.preview?"preview":"sglang",
            policy.preview?" (delivery outside full-output SGLang parity)":"");
    h3_cuda_sol_options previous_sol=h3_cuda_sol_exchange(params->cuda_sol);
    int previous_attention=h3_attention_exchange(params->cuda_attention);
    int previous_reference=h3_sglang_exchange(params->_arithmetic_recipe);
    h3_quant_scope previous_quant=h3_quant_exchange((h3_quant_scope){params->cuda_denoise_quant,params->cuda_denoise_quant_cache});
    h3_cuda_policy previous_policy=h3_cuda_policy_exchange(policy);
    h3_result *result=h3_generate_request(ctx,prompt,params);
    h3_cuda_policy_exchange(previous_policy);
    h3_attention_exchange(previous_attention);
    h3_cuda_sol_exchange(previous_sol);
    h3_quant_exchange(previous_quant);
    h3_sglang_exchange(previous_reference);
    h3_backend_exchange(previous_backend);
    return result;
}

void h3_result_free(h3_result *result) {
    if (result) { if(result->still_latent){h3_still_latent_free(result->still_latent);free(result->still_latent);} free(result->lora_provenance); h3_av_state_free(result->av_state); h3_sampler_state_free(result->sampler_state); }
    free(result);
}
