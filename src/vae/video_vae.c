#include "src/memory.h"
#include "src/vae/image_vae.h"
#include "src/vae/video_vae.h"
#include "src/sglang/sglang.h"

#include "src/weights/weights.h"

#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static double vae_clock(void) {
    struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}

enum {
    LATENT_CHANNELS = 24,
    CHUNK_LATENT_TIME = 7,
    FIRST_CHUNK_FRAMES = 22,
    FRAME_OFFSET = 3,
    HIDDEN = 2048,
    LAYERS = 36,
    HEADS = 32,
    HEAD_DIM = 64,
    INNER = HEADS * HEAD_DIM,
    FFN = 8192,
    REGISTERS = 4,
    SUFFIX = 5,
    ROPE_HALF = 24,
    OUTPUT_PATCH = 3 * 4 * 16 * 16,
    SPATIAL_RATIO = 16,
    TILE_PIXELS = 256,
    TILE_OVERLAP_MIN = 64
};

typedef struct {
    int count;
    int length;
    int *starts;
    int *overlaps;
} tile_axis;

typedef struct {
    h3_gpu_tensor *norm1;
    h3_gpu_tensor *qkv_w;
    h3_gpu_tensor *qkv_b;
    h3_gpu_tensor *out_w;
    h3_gpu_tensor *out_b;
    h3_gpu_tensor *scale1;
    h3_gpu_tensor *norm2;
    h3_gpu_tensor *w1;
    h3_gpu_tensor *w1_b;
    h3_gpu_tensor *w2;
    h3_gpu_tensor *w2_b;
    h3_gpu_tensor *scale2;
} vae_block;

typedef struct {
    h3_gpu *gpu;
    h3_weight_store *weights;
    vae_block blocks[LAYERS];
    h3_gpu_tensor *post_w;
    h3_gpu_tensor *post_b;
    h3_gpu_tensor *embed_w;
    h3_gpu_tensor *embed_b;
    h3_gpu_tensor *registers;
    h3_gpu_tensor *norm_out_w;
    h3_gpu_tensor *norm_out_b;
    h3_gpu_tensor *proj_w;
    h3_gpu_tensor *proj_b;
    h3_gpu_tensor *latent;
    h3_gpu_tensor *post;
    h3_gpu_tensor *patch_hidden;
    h3_gpu_tensor *hidden;
    h3_gpu_tensor *norm;
    h3_gpu_tensor *qkv;
    h3_gpu_tensor *query;
    h3_gpu_tensor *key;
    h3_gpu_tensor *value;
    h3_gpu_tensor *heads;
    h3_gpu_tensor *branch;
    h3_gpu_tensor *ff1;
    h3_gpu_tensor *activated;
    h3_gpu_tensor *rope_cos;
    h3_gpu_tensor *rope_sin;
    h3_gpu_tensor *projected;
    h3_gpu_tensor *suffix_zero;
    int reuse_buffers;
    uint32_t patches;
    uint32_t sequence;
    int latent_h;
    int latent_w;
    int latent_t;
    int output_frames;
    int image_weights;
    int image_slice;
    int sglang_captured;
} vae_context;

struct h3_video_vae_decoder {
    vae_context vae;
    tile_axis y_axis;
    tile_axis x_axis;
    int latent_h;
    int latent_w;
    float latent_mean[LATENT_CHANNELS];
    float latent_std[LATENT_CHANNELS];
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(vae_context *vae, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(vae->gpu));
    return 0;
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static h3_gpu_tensor *load_f32(vae_context *vae, const char *name, int ndim,
                               const uint64_t *shape, char *error,
                               size_t error_size) {
    h3_gpu_tensor *weight=vae->image_weights ?
        h3_weight_load_f16_f32(vae->weights, vae->gpu, name, ndim, shape, error, error_size) :
        h3_weight_load_f32(vae->weights, vae->gpu, name, ndim, shape, error, error_size);
#ifndef __APPLE__
    if(weight && !vae->image_weights && h3_gpu_sglang_vae_enabled(vae->gpu) &&
       ((!strncmp(name,"decoder.transformer_blocks.",27) &&
       (strstr(name,".attn.to_qkv.") || strstr(name,".attn.to_out.") ||
        strstr(name,".ff.w1.") || strstr(name,".ff.w2."))) ||
        !strcmp(name,"post_quant_conv.weight") || !strcmp(name,"post_quant_conv.bias"))) {
        h3_gpu_tensor *half=h3_gpu_sglang_vae_weight(vae->gpu,weight);
        h3_gpu_tensor_free(weight);weight=half;
        if(!weight)fail(error,error_size,"reference VAE weight conversion: %s",h3_gpu_error(vae->gpu));
    }
#endif
    return weight;
}

static h3_gpu_tensor *f1(vae_context *vae, const char *name, uint64_t width,
                         char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return load_f32(vae, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *f2(vae_context *vae,const char *name,uint64_t rows,
                         uint64_t columns,char *error,size_t error_size) {
    uint64_t shape[]={rows,columns};
    return load_f32(vae,name,2,shape,error,error_size);
}

static void free_block(vae_block *block) {
    free_tensor(&block->norm1); free_tensor(&block->qkv_w);
    free_tensor(&block->qkv_b); free_tensor(&block->out_w);
    free_tensor(&block->out_b); free_tensor(&block->scale1);
    free_tensor(&block->norm2); free_tensor(&block->w1);
    free_tensor(&block->w1_b); free_tensor(&block->w2);
    free_tensor(&block->w2_b); free_tensor(&block->scale2);
}

static void cleanup(vae_context *vae) {
    if (!vae) return;
    h3_gpu_cancel(vae->gpu);
    for (int index = 0; index < LAYERS; index++) free_block(&vae->blocks[index]);
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers); free_tensor(&vae->norm_out_w);
    free_tensor(&vae->norm_out_b); free_tensor(&vae->proj_w);
    free_tensor(&vae->proj_b); free_tensor(&vae->latent); free_tensor(&vae->post);
    free_tensor(&vae->patch_hidden); free_tensor(&vae->hidden);
    free_tensor(&vae->norm); free_tensor(&vae->qkv); free_tensor(&vae->query);
    free_tensor(&vae->key); free_tensor(&vae->value); free_tensor(&vae->heads);
    free_tensor(&vae->branch); free_tensor(&vae->ff1);
    free_tensor(&vae->activated); free_tensor(&vae->rope_cos);
    free_tensor(&vae->rope_sin); free_tensor(&vae->projected);
    free_tensor(&vae->suffix_zero);
    h3_gpu_free(vae->gpu);
    h3_weight_store_free(vae->weights);
    memset(vae, 0, sizeof(*vae));
}

static int load_block(vae_context *vae, int index, char *error,
                      size_t error_size) {
    vae_block *block = &vae->blocks[index];
    char prefix[96], name[160];
    snprintf(prefix, sizeof(prefix), "decoder.transformer_blocks.%d.", index);
#define F1(field, suffix, width) do {                                           \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f1(vae, name, width, error, error_size);                     \
    if (!block->field) return 0;                                                \
} while (0)
#define F2(field, suffix, rows, columns) do {                                   \
    snprintf(name, sizeof(name), "%s%s", prefix, suffix);                    \
    block->field = f2(vae, name, rows, columns, error, error_size);             \
    if (!block->field) return 0;                                                \
} while (0)
    F1(norm1, "norm1.weight", HIDDEN);
    F2(qkv_w, "attn.to_qkv.weight", INNER * 3, HIDDEN);
    F1(qkv_b, "attn.to_qkv.bias", INNER * 3);
    F2(out_w, "attn.to_out.weight", HIDDEN, INNER);
    F1(out_b, "attn.to_out.bias", HIDDEN);
    F1(scale1, "scale1", HIDDEN);
    F1(norm2, "norm2.weight", HIDDEN);
    F2(w1, "ff.w1.weight", FFN * 2, HIDDEN);
    F1(w1_b, "ff.w1.bias", FFN * 2);
    F2(w2, "ff.w2.weight", HIDDEN, FFN);
    F1(w2_b, "ff.w2.bias", HIDDEN);
    F1(scale2, "scale2", HIDDEN);
#undef F1
#undef F2
    return 1;
}

static int load_input_weights(vae_context *vae, char *error,
                              size_t error_size) {
    uint64_t post_shape[] = {LATENT_CHANNELS, LATENT_CHANNELS, 1, 1, 1};
    vae->post_w = load_f32(vae, "post_quant_conv.weight", 5, post_shape,
                           error, error_size);
    vae->post_b = f1(vae, "post_quant_conv.bias", LATENT_CHANNELS,
                     error, error_size);
    vae->embed_w = f2(vae, "decoder.x_embedder.weight", HIDDEN,
                      LATENT_CHANNELS, error, error_size);
    vae->embed_b = f1(vae, "decoder.x_embedder.bias", HIDDEN,
                      error, error_size);
    uint64_t register_shape[] = {1, REGISTERS, HIDDEN};
    vae->registers = load_f32(vae, "decoder.register_tokens", 3,
                              register_shape, error, error_size);
    return vae->post_w && vae->post_b && vae->embed_w && vae->embed_b &&
           vae->registers;
}

static int load_output_weights(vae_context *vae, char *error,
                               size_t error_size) {
    vae->norm_out_w = f1(vae, "decoder.norm_out.weight", HIDDEN,
                          error, error_size);
    vae->norm_out_b = f1(vae, "decoder.norm_out.bias", HIDDEN,
                          error, error_size);
    vae->proj_w = f2(vae, "decoder.proj_out.weight", OUTPUT_PATCH, HIDDEN,
                      error, error_size);
    vae->proj_b = f1(vae, "decoder.proj_out.bias", OUTPUT_PATCH,
                      error, error_size);
    return vae->norm_out_w && vae->norm_out_b && vae->proj_w && vae->proj_b;
}

static int parse_float_array(const char *json, const char *key, float *values,
                             size_t count, char *error, size_t error_size) {
    char pattern[64];
    snprintf(pattern, sizeof(pattern), "\"%s\"", key);
    const char *cursor = strstr(json, pattern);
    if (!cursor || !(cursor = strchr(cursor + strlen(pattern), ':')) ||
        !(cursor = strchr(cursor, '['))) {
        fail(error, error_size, "video VAE config is missing %s", key);
        return 0;
    }
    cursor++;
    for (size_t index = 0; index < count; index++) {
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        errno = 0;
        char *end = NULL;
        float value = strtof(cursor, &end);
        if (errno || end == cursor || !isfinite(value)) {
            fail(error, error_size, "video VAE config has malformed %s", key);
            return 0;
        }
        values[index] = value;
        cursor = end;
        while (*cursor == ' ' || *cursor == '\n' || *cursor == '\r' ||
               *cursor == '\t') cursor++;
        if (index + 1 < count) {
            if (*cursor++ != ',') {
                fail(error, error_size, "video VAE config has short %s", key);
                return 0;
            }
        } else if (*cursor != ']') {
            fail(error, error_size, "video VAE config has long %s", key);
            return 0;
        }
    }
    return 1;
}

static int load_latent_normalization(const char *weight_directory,
                                     float *mean, float *deviation,
                                     char *error, size_t error_size) {
    size_t path_size = strlen(weight_directory) + strlen("/../config.json") + 1;
    char *path = malloc(path_size);
    if (!path) {
        fail(error, error_size, "out of memory resolving video VAE config");
        return 0;
    }
    snprintf(path, path_size, "%s/../config.json", weight_directory);
    FILE *file = fopen(path, "rb");
    if (!file || fseek(file, 0, SEEK_END)) {
        fail(error, error_size, "cannot open video VAE config %s: %s", path,
             strerror(errno));
        if (file) fclose(file);
        free(path);
        return 0;
    }
    long end = ftell(file);
    if (end < 1 || end > 1024 * 1024 || fseek(file, 0, SEEK_SET)) {
        fail(error, error_size, "invalid video VAE config %s", path);
        fclose(file);
        free(path);
        return 0;
    }
    char *json = malloc((size_t)end + 1);
    if (!json || fread(json, 1, (size_t)end, file) != (size_t)end) {
        fail(error, error_size, "cannot read video VAE config %s", path);
        free(json);
        fclose(file);
        free(path);
        return 0;
    }
    json[end] = '\0';
    fclose(file);
    free(path);
    int ok = parse_float_array(json, "latents_mean", mean, LATENT_CHANNELS,
                               error, error_size) &&
             parse_float_array(json, "latents_std", deviation,
                               LATENT_CHANNELS, error, error_size);
    free(json);
    if (ok) for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
        if (deviation[channel] <= 0.0f) {
            fail(error, error_size, "video VAE latent standard deviation is invalid");
            return 0;
        }
    }
    return ok;
}

static int prepare_input(vae_context *vae, const float *input,
                         const float *mean, const float *deviation,
                         char *error, size_t error_size) {
    size_t patch_elements = (size_t)vae->patches * LATENT_CHANNELS;
    float *rows = malloc(patch_elements * sizeof(*rows));
    if (!rows) {
        fail(error, error_size, "out of memory padding video latent");
        return 0;
    }
    size_t row = 0;
    for (int time = 0; time < (int)(vae->patches / (unsigned)(vae->latent_h * vae->latent_w)); time++) {
        int source_t = time < vae->latent_t ? time : vae->latent_t - 1;
        for (int height = 0; height < vae->latent_h; height++)
            for (int width = 0; width < vae->latent_w; width++)
                for (int channel = 0; channel < LATENT_CHANNELS; channel++) {
                    size_t source = (((size_t)channel * (size_t)vae->latent_t +
                        (size_t)source_t) *
                        (size_t)vae->latent_h + (size_t)height) *
                        (size_t)vae->latent_w + (size_t)width;
                    rows[row++] = input[source] * deviation[channel] +
                                  mean[channel];
                }
    }
    int ok=1;
    if(vae->reuse_buffers && vae->latent && h3_gpu_tensor_elements(vae->latent)==patch_elements)
        ok=h3_gpu_tensor_write_f32(vae->latent,rows,patch_elements);
    else {
        free_tensor(&vae->latent);
        vae->latent = h3_gpu_tensor_from_f32(vae->gpu, rows, patch_elements);
    }
    free(rows);
    if(!ok)fail(error,error_size,"cannot refill video VAE latent: %s",h3_gpu_error(vae->gpu));
    return ok && vae->latent != NULL;
}

static int prepare_rope(vae_context *vae, char *error, size_t error_size) {
    size_t count = (size_t)vae->sequence * ROPE_HALF;
    float *cosines = malloc(count * sizeof(*cosines));
    float *sines = malloc(count * sizeof(*sines));
    if (!cosines || !sines) {
        free(cosines); free(sines);
        fail(error, error_size, "out of memory allocating video VAE RoPE");
        return 0;
    }
    uint32_t row = 0;
    int tile_time = (int)(vae->patches / (unsigned)(vae->latent_h * vae->latent_w));
    for (int t = 0; t < tile_time; t++)
        for (int h = 0; h < vae->latent_h; h++)
            for (int w = 0; w < vae->latent_w; w++, row++) {
                float axes[] = {
                    2.0f * (((float)t + 0.5f) /
                            (float)tile_time) - 1.0f,
                    2.0f * (((float)h + 0.5f) / (float)vae->latent_h) - 1.0f,
                    2.0f * (((float)w + 0.5f) / (float)vae->latent_w) - 1.0f
                };
                for (uint32_t axis = 0; axis < 3; axis++)
                    for (uint32_t frequency = 0; frequency < 8; frequency++) {
                        float inverse = 1.0f /
                            powf(100.0f, (float)frequency * 0.125f);
                        float angle = 2.0f * 3.14159265358979323846f *
                                      axes[axis] * inverse;
                        size_t offset = (size_t)row * ROPE_HALF +
                                        axis * 8 + frequency;
                        cosines[offset] = cosf(angle);
                        sines[offset] = sinf(angle);
                    }
            }
    while (row < vae->sequence) {
        for (uint32_t index = 0; index < ROPE_HALF; index++) {
            cosines[(size_t)row * ROPE_HALF + index] = 1.0f;
            sines[(size_t)row * ROPE_HALF + index] = 0.0f;
        }
        row++;
    }
    vae->rope_cos = h3_gpu_tensor_from_f32(vae->gpu, cosines, count);
    vae->rope_sin = h3_gpu_tensor_from_f32(vae->gpu, sines, count);
    free(cosines); free(sines);
#ifndef __APPLE__
    if(vae->rope_cos && vae->rope_sin && h3_gpu_sglang_vae_enabled(vae->gpu) &&
       !h3_gpu_sglang_vae_rope(vae->gpu,vae->rope_cos,vae->rope_sin,
           tile_time,vae->latent_h,vae->latent_w,vae->sequence)) {
        fail(error,error_size,"reference VAE rotary positions: %s",h3_gpu_error(vae->gpu));return 0;
    }
#endif
    if (!vae->rope_cos || !vae->rope_sin) {
        fail(error, error_size, "cannot allocate video VAE RoPE: %s",
             h3_gpu_error(vae->gpu));
        return 0;
    }
    return 1;
}

static int allocate_activations(vae_context *vae, char *error,
                                size_t error_size) {
    vae->reuse_buffers=0;
#ifndef __APPLE__
    if(!vae->image_weights && h3_gpu_sglang_vae_enabled(vae->gpu))vae->reuse_buffers=1;
#endif
    size_t patches = vae->patches, sequence = vae->sequence;
#define F32(field, elements) (vae->field = h3_gpu_tensor_new_f32(vae->gpu, (elements)))
    h3_gpu_tensor *all[] = {
        F32(post, patches * LATENT_CHANNELS),
        F32(patch_hidden, patches * HIDDEN),
        F32(hidden, sequence * HIDDEN),
        F32(norm, sequence * HIDDEN),
        F32(qkv, sequence * INNER * 3),
        F32(query, sequence * INNER), F32(key, sequence * INNER), F32(value, sequence * INNER), F32(heads, sequence * INNER),
        F32(branch, sequence * HIDDEN), F32(ff1, sequence * FFN * 2),
        F32(activated, sequence * FFN),
        F32(projected, sequence * OUTPUT_PATCH)
    };
#undef F32
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++)
        if (!all[index]) {
            fail(error, error_size, "cannot allocate video VAE activations: %s",
                 h3_gpu_error(vae->gpu));
            return 0;
        }
    return 1;
}

/* Bounded, test-only snapshots of the first tile. Timing runs set no path. */
static int reference_vae_capture(vae_context *vae,const h3_gpu_tensor *tensor,
    const char *name,size_t count) {
    const char *directory=getenv("H3_TEST_SGLANG_DIR");
    if(!directory||!*directory||!h3_sglang_requested()||vae->sglang_captured)return 1;
    if(count>h3_gpu_tensor_elements(tensor)||count>(256u<<20)/sizeof(float))return 0;
    float *data=malloc(count*sizeof(float));char filename[128];
    snprintf(filename,sizeof(filename),"vae.%s.f32",name);
    int ok=data && h3_gpu_submit(vae->gpu) && h3_gpu_tensor_read_f32(tensor,data,count) &&
        h3_sglang_dump(filename,data,count*sizeof(float));
    free(data);return h3_gpu_begin(vae->gpu)&&ok;
}

static int run_block(vae_context *vae, int index, char *error,
                     size_t error_size) {
    vae_block *weight = &vae->blocks[index];
    uint32_t rows = vae->sequence;
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) return 0;               \
} while (0)
#define TRACE(t,n,w) do {if(index==0)OP(reference_vae_capture(vae,vae->t,n,(size_t)rows*(w)),"reference VAE capture");}while(0)
    TRACE(hidden,"input",HIDDEN);
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm1,
        rows, HIDDEN, 1e-5f), "video VAE attention norm");
    TRACE(norm,"norm1",HIDDEN);
    OP(h3_gpu_video_linear(vae->gpu, vae->qkv, vae->norm, weight->qkv_w,
        weight->qkv_b, rows, HIDDEN, INNER * 3), "video VAE QKV");
    TRACE(qkv,"qkv",INNER*3);
    OP(h3_gpu_video_qkv(vae->gpu, vae->query, vae->key, vae->value,
        vae->qkv, vae->rope_cos, vae->rope_sin, rows, HEADS, HEAD_DIM,
        ROPE_HALF, 1e-5f), "video VAE QK norm/RoPE");
    TRACE(query,"query",INNER);TRACE(key,"key",INNER);TRACE(value,"value",INNER);
    OP(h3_gpu_video_sdpa(vae->gpu, vae->heads, vae->query, vae->key, vae->value,
        rows, HEADS, HEAD_DIM, 1.0f / sqrtf((float)HEAD_DIM)),
       "video VAE attention");
    TRACE(heads,"attention",INNER);
    OP(h3_gpu_video_linear(vae->gpu, vae->branch, vae->heads, weight->out_w,
        weight->out_b, rows, INNER, HIDDEN), "video VAE attention output");
    TRACE(branch,"projection",HIDDEN);
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale1, rows, HIDDEN), "video VAE attention residual");
    TRACE(hidden,"residual1",HIDDEN);
    OP(h3_gpu_rms_norm_f32(vae->gpu, vae->norm, vae->hidden, weight->norm2,
        rows, HIDDEN, 1e-5f), "video VAE MLP norm");
    TRACE(norm,"norm2",HIDDEN);
    OP(h3_gpu_video_linear(vae->gpu, vae->ff1, vae->norm, weight->w1,
        weight->w1_b, rows, HIDDEN, FFN * 2), "video VAE MLP input");
    TRACE(ff1,"ff1",FFN*2);
    OP(h3_gpu_swiglu_f32(vae->gpu, vae->activated, vae->ff1, rows, FFN),
       "video VAE SwiGLU");
    TRACE(activated,"swiglu",FFN);
    OP(h3_gpu_video_linear(vae->gpu, vae->branch, vae->activated, weight->w2,
        weight->w2_b, rows, FFN, HIDDEN), "video VAE MLP output");
    TRACE(branch,"mlp",HIDDEN);
    OP(h3_gpu_scale_add_f32(vae->gpu, vae->hidden, vae->hidden, vae->branch,
        weight->scale2, rows, HIDDEN), "video VAE MLP residual");
    TRACE(hidden,"output",HIDDEN);
#undef TRACE
#undef OP
    return 1;
}

static int run_decoder(vae_context *vae, h3_video_vae_progress progress,
                       void *progress_opaque, char *error,
                       size_t error_size) {
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        h3_gpu_cancel(vae->gpu); h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin video VAE decoder");
    OP(h3_gpu_video_linear(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "video VAE post-quant projection");
    OP(h3_gpu_video_linear(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack video VAE suffix");
    OP(h3_gpu_submit(vae->gpu), "submit video VAE input projection");
    h3_gpu_tensor_free(zero);
    zero = NULL;
    free_tensor(&vae->post_w); free_tensor(&vae->post_b);
    free_tensor(&vae->embed_w); free_tensor(&vae->embed_b);
    free_tensor(&vae->registers);

    for (int index = 0; index < LAYERS; index++) {
        if (!h3_memory_check(0, "video VAE", error, error_size)) return 0;
        if (!load_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_begin(vae->gpu), "begin video VAE transformer block");
        if (!run_block(vae, index, error, error_size)) return 0;
        OP(h3_gpu_submit(vae->gpu), "submit video VAE transformer block");
        free_block(&vae->blocks[index]);
        if (!h3_memory_checkpoint(progress && progress(index + 1, LAYERS, progress_opaque),
                                   "video VAE", error, error_size)) return 0;
    }
    if (!load_output_weights(vae, error, error_size)) return 0;
    OP(h3_gpu_begin(vae->gpu), "begin video VAE output projection");
    OP(reference_vae_capture(vae,vae->hidden,"final-input",(size_t)vae->sequence*HIDDEN),"reference final VAE input capture");
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "video VAE output LayerNorm");
    OP(reference_vae_capture(vae,vae->norm,"final-norm",(size_t)vae->sequence*HIDDEN),"reference final VAE norm capture");
    OP(h3_gpu_video_linear(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "video VAE output projection");
    OP(reference_vae_capture(vae,vae->projected,"final-projection",(size_t)vae->sequence*OUTPUT_PATCH),"reference final VAE projection capture");
    vae->sglang_captured=1;
    OP(h3_gpu_submit(vae->gpu), "submit video VAE decoder");
    free_tensor(&vae->norm_out_w); free_tensor(&vae->norm_out_b);
    free_tensor(&vae->proj_w); free_tensor(&vae->proj_b);
#undef OP
    return 1;
}

static int load_resident_weights(vae_context *vae,
                                 h3_video_vae_progress progress,
                                 void *progress_opaque, char *error,
                                 size_t error_size) {
    if (!h3_memory_checkpoint(progress && progress(0, LAYERS, progress_opaque),
                               "video VAE load", error, error_size)) return 0;
    if (!load_input_weights(vae, error, error_size)) return 0;
    for (int index = 0; index < LAYERS; index++) {
        if (!h3_memory_check(0, "video VAE", error, error_size)) return 0;
        if (!load_block(vae, index, error, error_size)) return 0;
        if (!h3_memory_checkpoint(progress && progress(index + 1, LAYERS, progress_opaque),
                                   "video VAE", error, error_size)) return 0;
    }
    return load_output_weights(vae, error, error_size);
}

/* Tiled decoding keeps every VAE weight resident for the duration of the phase.
 * This uses about 9 GiB after the DiT has been retired, avoids rereading 9 GiB
 * per spatial tile, and permits one command-buffer chain per tile. */
static int run_resident_tile(vae_context *vae, h3_video_vae_progress progress,
                             void *progress_opaque, char *error, size_t error_size) {
    if (!h3_memory_checkpoint(progress && progress(0, LAYERS, progress_opaque),
                               "video VAE decode", error, error_size)) return 0;
    float zeros[HIDDEN];
    memset(zeros, 0, sizeof(zeros));
    h3_gpu_tensor *zero = vae->suffix_zero;
    if(!zero)zero=h3_gpu_tensor_from_f32(vae->gpu, zeros, HIDDEN);
    if(vae->reuse_buffers)vae->suffix_zero=zero;
    if (!zero) {
        fail(error, error_size, "cannot allocate video VAE suffix token");
        return 0;
    }
#define OP(call, label) do {                                                    \
    if (!gpu_op(vae, (call), error, error_size, label)) {                       \
        if(h3_gpu_video_graph_abort(vae->gpu))return run_resident_tile(vae,progress,progress_opaque,error,error_size); \
        h3_gpu_cancel(vae->gpu); if(!vae->reuse_buffers)h3_gpu_tensor_free(zero); return 0;                                    \
    }                                                                           \
} while (0)
    OP(h3_gpu_begin(vae->gpu), "begin tiled video VAE decoder");
    int graph=h3_gpu_video_graph_begin(vae->gpu);
    if(graph<0){h3_gpu_cancel(vae->gpu);return gpu_op(vae,0,error,error_size,"VAE graph");}
    if(graph==2){
        if(!gpu_op(vae,h3_gpu_submit(vae->gpu),error,error_size,"VAE graph replay")){
            h3_gpu_cancel(vae->gpu);return 0;
        }
        return h3_memory_checkpoint(progress&&progress(LAYERS,LAYERS,progress_opaque),
                                    "VAE graph replay",error,error_size);
    }
    OP(h3_gpu_video_linear(vae->gpu, vae->post, vae->latent, vae->post_w,
        vae->post_b, vae->patches, LATENT_CHANNELS, LATENT_CHANNELS),
       "tiled video VAE post-quant projection");
    OP(h3_gpu_video_linear(vae->gpu, vae->patch_hidden, vae->post, vae->embed_w,
        vae->embed_b, vae->patches, LATENT_CHANNELS, HIDDEN),
       "tiled video VAE latent embedding");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden, 0, vae->patch_hidden, 0,
        (size_t)vae->patches * HIDDEN), "pack tiled video VAE patches");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)vae->patches * HIDDEN, vae->registers, 0,
        REGISTERS * HIDDEN), "pack tiled video VAE registers");
    OP(h3_gpu_copy_f32(vae->gpu, vae->hidden,
        (size_t)(vae->patches + REGISTERS) * HIDDEN, zero, 0, HIDDEN),
       "pack tiled video VAE suffix");
    for (int index = 0; index < LAYERS; index++) {
        if (!h3_memory_check(0, "video VAE decode", error, error_size)) {
            h3_gpu_cancel(vae->gpu);
            if(!vae->reuse_buffers)h3_gpu_tensor_free(zero);
            return 0;
        }
        if (!run_block(vae, index, error, error_size)) {
            if(h3_gpu_video_graph_abort(vae->gpu))
                return run_resident_tile(vae,progress,progress_opaque,error,error_size);
            h3_gpu_cancel(vae->gpu);
            if(!vae->reuse_buffers)h3_gpu_tensor_free(zero);
            return 0;
        }
        if (!h3_memory_checkpoint(progress && graph!=1 && progress(index + 1, LAYERS, progress_opaque),
                                  "video VAE decode", error, error_size)) {
            h3_gpu_cancel(vae->gpu);
            if(!vae->reuse_buffers)h3_gpu_tensor_free(zero);
            return 0;
        }
    }
    OP(reference_vae_capture(vae,vae->hidden,"final-input",(size_t)vae->sequence*HIDDEN),"reference final VAE input capture");
    OP(h3_gpu_layer_norm_f32(vae->gpu, vae->norm, vae->hidden,
        vae->norm_out_w, vae->norm_out_b, vae->sequence, HIDDEN, 1e-5f),
       "tiled video VAE output LayerNorm");
    OP(reference_vae_capture(vae,vae->norm,"final-norm",(size_t)vae->sequence*HIDDEN),"reference final VAE norm capture");
    OP(h3_gpu_video_linear(vae->gpu, vae->projected, vae->norm, vae->proj_w,
        vae->proj_b, vae->sequence, HIDDEN, OUTPUT_PATCH),
       "tiled video VAE output projection");
    OP(reference_vae_capture(vae,vae->projected,"final-projection",(size_t)vae->sequence*OUTPUT_PATCH),"reference final VAE projection capture");
    vae->sglang_captured=1;
    int captured=graph==1;
    graph=h3_gpu_video_graph_end(vae->gpu);
    if(!graph)return run_resident_tile(vae,progress,progress_opaque,error,error_size);
    if(graph<0){h3_gpu_cancel(vae->gpu);return gpu_op(vae,0,error,error_size,"VAE graph finalization");}
    OP(h3_gpu_submit(vae->gpu), "submit tiled video VAE decoder");
    if(!vae->reuse_buffers)h3_gpu_tensor_free(zero);
#undef OP
    if(captured)return h3_memory_checkpoint(progress&&progress(LAYERS,LAYERS,progress_opaque),
                                            "VAE captured tile",error,error_size);
    return 1;
}

static int reference_video(const vae_context *vae) {
#ifndef __APPLE__
    return !vae->image_weights && h3_gpu_sglang_vae_enabled(vae->gpu);
#else
    (void)vae;return 0;
#endif
}

/* The oracle blends normalized decoder samples, then applies torchvision's
 * inverse Normalize and clamp once. Clipping individual tiles before blending
 * changes overlap pixels even when every transformer operation is identical. */
static void reference_video_finish(const vae_context *vae,float *rgb,size_t n) {
    if(!reference_video(vae))return;
    static const float inverse_mean[]={ (float)(-.485/.229),(float)(-.456/.224),(float)(-.406/.225) };
    static const float inverse_std[]={ (float)(1./.229),(float)(1./.224),(float)(1./.225) };
    for(size_t i=0;i<n;i++){
        float value=(rgb[i]-inverse_mean[i%3])/inverse_std[i%3];
        rgb[i]=value<0?0:value>1?1:value;
    }
}

static int unpack_frame_range(vae_context *vae, int first_frame,
                              int frame_count, h3_video_frames *output,
                              char *error, size_t error_size) {
    double start=vae_clock();
    if (first_frame < 0 || frame_count < 1 ||
        first_frame > vae->output_frames - frame_count) {
        fail(error, error_size, "invalid video VAE output frame range");
        return 0;
    }
    size_t projected_count = (size_t)vae->sequence * OUTPUT_PATCH;
    float *rows = NULL;
    int pixel_h = vae->latent_h * 16;
    int pixel_w = vae->latent_w * 16;
    size_t rgb_count = (size_t)frame_count * (size_t)pixel_h *
                       (size_t)pixel_w * 3;
    float *rgb = malloc(rgb_count * sizeof(*rgb));
    if(rgb && !vae->image_weights && reference_video(vae)) {
        int status=h3_gpu_video_unpack(vae->gpu,vae->projected,rgb,vae->latent_h,
            vae->latent_w,vae->output_frames,first_frame,frame_count);
        if(status<0){free(rgb);fail(error,error_size,"VAE GPU unpack: %s",h3_gpu_error(vae->gpu));return 0;}
        if(status>0){
            output->frames=frame_count;output->height=pixel_h;output->width=pixel_w;output->rgb=rgb;
            if(getenv("H3_PROFILE"))fprintf(stderr,"h3cli: video VAE GPU unpack/copy=%.6fs frames=%d\n",vae_clock()-start,frame_count);
            return h3_gpu_get_stats(vae->gpu,&output->gpu_stats);
        }
    }
    rows=malloc(projected_count*sizeof(*rows));
    if (!rows || !rgb) {
        free(rows); free(rgb);
        fail(error, error_size, "out of memory unpacking video VAE frames");
        return 0;
    }
    if (!h3_gpu_tensor_read_f32(vae->projected, rows, projected_count)) {
        free(rows); free(rgb);
        fail(error, error_size, "cannot read video VAE output patches");
        return 0;
    }
    double unpack_start=vae_clock();
    static const float mean[] = {0.485f, 0.456f, 0.406f};
    static const float deviation[] = {0.229f, 0.224f, 0.225f};
    int raw=reference_video(vae);
    for (int output_frame = 0; output_frame < frame_count; output_frame++) {
        int frame = first_frame + output_frame;
        int decoded_t = vae->image_weights ? vae->image_slice : frame + FRAME_OFFSET;
        if (!vae->image_weights && vae->output_frames == FIRST_CHUNK_FRAMES && frame >= 17)
            decoded_t += 3;
        int patch_t = decoded_t / 4;
        int within_t = decoded_t % 4;
        for (int y = 0; y < pixel_h; y++) {
            int patch_y = y / 16, within_y = y % 16;
            for (int x = 0; x < pixel_w; x++) {
                int patch_x = x / 16, within_x = x % 16;
                size_t patch = ((size_t)patch_t * (size_t)vae->latent_h +
                    (size_t)patch_y) * (size_t)vae->latent_w +
                    (size_t)patch_x;
                for (int channel = 0; channel < 3; channel++) {
                    size_t component = ((((size_t)channel * 4 +
                        (size_t)within_t) * 16 + (size_t)within_y) * 16 +
                        (size_t)within_x);
                    float value = raw ? rows[patch * OUTPUT_PATCH + component] :
                        rows[patch * OUTPUT_PATCH + component] * deviation[channel] + mean[channel];
                    if (vae->image_weights && !isfinite(value)) {
                        free(rows); free(rgb); fail(error, error_size, "nonfinite image VAE output"); return 0;
                    }
                    if (!vae->image_weights && !raw) {
                        if (value < 0.0f) value = 0.0f;
                        if (value > 1.0f) value = 1.0f;
                    }
                    size_t destination = (((size_t)output_frame *
                        (size_t)pixel_h +
                        (size_t)y) * (size_t)pixel_w + (size_t)x) * 3 +
                        (size_t)channel;
                    rgb[destination] = value;
                }
            }
        }
    }
    free(rows);
    output->frames = frame_count;
    output->height = pixel_h;
    output->width = pixel_w;
    output->rgb = rgb;
    if(getenv("H3_PROFILE"))fprintf(stderr,"h3cli: video VAE delivery readback=%.6fs unpack=%.6fs frames=%d\n",unpack_start-start,vae_clock()-unpack_start,frame_count);
    return h3_gpu_get_stats(vae->gpu, &output->gpu_stats);
}

static int unpack_frames(vae_context *vae, h3_video_frames *output,
                         char *error, size_t error_size) {
    return unpack_frame_range(vae, 0, vae->output_frames, output,
                              error, error_size);
}

static void tile_axis_free(tile_axis *axis) {
    if (!axis) return;
    free(axis->starts);
    free(axis->overlaps);
    memset(axis, 0, sizeof(*axis));
}

static int tile_count_for_extent(int extent, int tile_pixels) {
    if (extent <= tile_pixels) return 1;
    int count = (extent + tile_pixels - 1) / tile_pixels;
    while (tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) < extent)
        count++;
    return count;
}

static int auto_tile_pixels(int pixel_height, int pixel_width) {
    int best = TILE_PIXELS;
    uint64_t best_score = UINT64_MAX;
    for (int pixels = TILE_PIXELS; pixels <= 320;
         pixels += SPATIAL_RATIO) {
        uint64_t tiles = (uint64_t)tile_count_for_extent(pixel_height, pixels) *
            (uint64_t)tile_count_for_extent(pixel_width, pixels);
        /* The resident VAE is dominated by linears and activation traffic;
         * measured tile cost follows area more closely than cubic sequence
         * growth at these shapes. Attention is still included in each tile. */
        uint64_t score = tiles * (uint64_t)pixels * (uint64_t)pixels;
        if (score < best_score) {
            best = pixels;
            best_score = score;
        }
    }
    return best;
}

int h3_video_vae_tile_pixels(int pixel_height, int pixel_width,
                             char *error, size_t error_size) {
    const char *value = getenv("H3_VAE_TILE_PIXELS");
    const char *policy = "default";
    int selected = TILE_PIXELS;
    if (value) {
        if (!strcmp(value, "auto")) {
            selected = auto_tile_pixels(pixel_height, pixel_width);
            policy = "auto";
        } else {
            char *end = NULL;
            long pixels = strtol(value, &end, 10);
            if (*value && end && !*end && pixels >= TILE_PIXELS &&
                pixels <= 320 && pixels % SPATIAL_RATIO == 0) {
                selected = (int)pixels;
                policy = "explicit override";
            } else {
                fail(error, error_size, "Invalid H3_VAE_TILE_PIXELS=%s. Supported "
                     "VideoVAE decoder tile range is 256..320 pixels in multiples "
                     "of 16 (or auto). The recommended/released default is 256.", value);
                return 0;
            }
        }
    }
    if (getenv("H3_PROFILE")) {
        fprintf(stderr, "h3cli: VideoVAE tile: %d px (%s)\n", selected, policy);
        if (selected > TILE_PIXELS && !strcmp(policy, "explicit override"))
            fprintf(stderr, "h3cli: VideoVAE: %d px decoder tiles; "
                    "reconstruction may differ from the released "
                    "MiniMax-H3 256 px configuration\n", selected);
    }
    return selected;
}

static int tile_axis_build(int extent, int tile_pixels, tile_axis *axis,
                           char *error,
                           size_t error_size) {
    memset(axis, 0, sizeof(*axis));
    if (extent < 1 || extent % SPATIAL_RATIO) {
        fail(error, error_size, "video VAE tile extent must be a multiple of %d",
             SPATIAL_RATIO);
        return 0;
    }
    if (extent <= tile_pixels) {
        axis->count = 1;
        axis->length = extent;
        axis->starts = calloc(1, sizeof(*axis->starts));
        if (!axis->starts) {
            fail(error, error_size, "out of memory constructing VAE tile plan");
            return 0;
        }
        return 1;
    }
    int count = tile_count_for_extent(extent, tile_pixels);
    axis->starts = calloc((size_t)count, sizeof(*axis->starts));
    axis->overlaps = malloc((size_t)(count - 1) * sizeof(*axis->overlaps));
    if (!axis->starts || !axis->overlaps) {
        tile_axis_free(axis);
        fail(error, error_size, "out of memory constructing VAE tile plan");
        return 0;
    }
    for (int index = 0; index < count - 1; index++)
        axis->overlaps[index] = TILE_OVERLAP_MIN;
    int remaining = tile_pixels * count - TILE_OVERLAP_MIN * (count - 1) -
                    extent;
    for (int unit = 0; unit < remaining / SPATIAL_RATIO; unit++)
        axis->overlaps[unit % (count - 1)] += SPATIAL_RATIO;
    for (int index = 1; index < count; index++)
        axis->starts[index] = axis->starts[index - 1] + tile_pixels -
                              axis->overlaps[index - 1];
    axis->count = count;
    axis->length = tile_pixels;
    return 1;
}

static float *extract_latent_tile(const float *latent, int full_t,
                                  int full_h, int full_w, int start_t,
                                  int start_y, int start_x, int tile_t,
                                  int tile_h, int tile_w, char *error,
                                  size_t error_size) {
    size_t count = (size_t)LATENT_CHANNELS * (size_t)tile_t *
                   (size_t)tile_h * (size_t)tile_w;
    float *tile = malloc(count * sizeof(*tile));
    if (!tile) {
        fail(error, error_size, "out of memory extracting video VAE tile");
        return NULL;
    }
    for (int channel = 0; channel < LATENT_CHANNELS; channel++)
        for (int time = 0; time < tile_t; time++)
            for (int y = 0; y < tile_h; y++) {
                size_t source = (((size_t)channel * (size_t)full_t +
                    (size_t)(start_t + time)) * (size_t)full_h +
                    (size_t)(start_y + y)) * (size_t)full_w +
                    (size_t)start_x;
                size_t destination = (((size_t)channel * (size_t)tile_t +
                    (size_t)time) * (size_t)tile_h + (size_t)y) *
                    (size_t)tile_w;
                memcpy(tile + destination, latent + source,
                       (size_t)tile_w * sizeof(*tile));
            }
    return tile;
}

static int stitch_tiles(float **tiles, const tile_axis *y_axis,
                        const tile_axis *x_axis, int frame_count, int reference,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    double start=vae_clock();
    int full_h = y_axis->starts[y_axis->count - 1] + y_axis->length;
    int full_w = x_axis->starts[x_axis->count - 1] + x_axis->length;
    size_t count = (size_t)frame_count * (size_t)full_h *
                   (size_t)full_w * 3;
    float *rgb = malloc(count * sizeof(*rgb));
    if (!rgb) {
        fail(error, error_size, "out of memory stitching video VAE tiles");
        return 0;
    }
    for (int tile_y = 0; tile_y < y_axis->count; tile_y++)
        for (int tile_x = 0; tile_x < x_axis->count; tile_x++) {
            int index = tile_y * x_axis->count + tile_x;
            const float *current = tiles[index];
            const float *above = tile_y ? tiles[index - x_axis->count] : NULL;
            const float *left = tile_x ? tiles[index - 1] : NULL;
            int overlap_y = tile_y ? y_axis->overlaps[tile_y - 1] : 0;
            int overlap_x = tile_x ? x_axis->overlaps[tile_x - 1] : 0;
            int keep_h = y_axis->length -
                (tile_y + 1 < y_axis->count ? y_axis->overlaps[tile_y] : 0);
            int keep_w = x_axis->length -
                (tile_x + 1 < x_axis->count ? x_axis->overlaps[tile_x] : 0);
            for (int frame = 0; frame < frame_count; frame++)
                for (int y = 0; y < keep_h; y++)
                    for (int x = 0; x < keep_w; x++)
                        for (int channel = 0; channel < 3; channel++) {
                            size_t local = (((size_t)frame *
                                (size_t)y_axis->length + (size_t)y) *
                                (size_t)x_axis->length + (size_t)x) * 3 +
                                (size_t)channel;
                            float value = current[local];
                            if (above && y < overlap_y) {
                                size_t top = (((size_t)frame *
                                    (size_t)y_axis->length +
                                    (size_t)(y_axis->length - overlap_y + y)) *
                                    (size_t)x_axis->length + (size_t)x) * 3 +
                                    (size_t)channel;
                                /* Torch scalar division multiplies by the
                                 * rounded reciprocal; tensor division does not. */
                                float alpha = reference ? (float)y * (1.0f / (float)overlap_y) :
                                                          (float)y / (float)overlap_y;
                                value = above[top] * (1.0f - alpha) +
                                        value * alpha;
                            }
                            if (left && x < overlap_x) {
                                size_t prior = (((size_t)frame *
                                    (size_t)y_axis->length + (size_t)y) *
                                    (size_t)x_axis->length +
                                    (size_t)(x_axis->length - overlap_x + x)) *
                                    3 + (size_t)channel;
                                float alpha = reference ? (float)x * (1.0f / (float)overlap_x) :
                                                          (float)x / (float)overlap_x;
                                value = left[prior] * (1.0f - alpha) +
                                        value * alpha;
                            }
                            size_t destination = (((size_t)frame *
                                (size_t)full_h +
                                (size_t)(y_axis->starts[tile_y] + y)) *
                                (size_t)full_w +
                                (size_t)(x_axis->starts[tile_x] + x)) * 3 +
                                (size_t)channel;
                            rgb[destination] = value;
                        }
        }
    output->frames = frame_count;
    output->height = full_h;
    output->width = full_w;
    output->rgb = rgb;
    if(getenv("H3_PROFILE"))fprintf(stderr,"h3cli: video VAE stitching wall=%.6fs tiles=%d frames=%d\n",vae_clock()-start,y_axis->count*x_axis->count,frame_count);
    return 1;
}

typedef struct {
    h3_video_vae_progress callback;
    void *opaque;
    int offset, total;
} decode_progress;

static int decode_progress_init(decode_progress *state,
                                h3_video_vae_progress callback, void *opaque,
                                int chunks, int tiles_y, int tiles_x,
                                char *error, size_t error_size) {
    int total = LAYERS;
    const int factors[] = {chunks, tiles_y, tiles_x};
    for (size_t i = 0; i < sizeof(factors) / sizeof(*factors); i++) {
        if (factors[i] < 1 || factors[i] > INT32_MAX / total) {
            fail(error, error_size, "video VAE progress work count overflows");
            return 0;
        }
        total *= factors[i];
    }
    *state = (decode_progress){callback, opaque, 0, total};
    return 1;
}

static int decode_progress_block(int completed, int total, void *opaque) {
    decode_progress *state = opaque;
    (void)total;
    int current = state->offset + completed;
    /* GPU submission, unpacking and stitching still follow the final layer. */
    if (current == state->total) current--;
    return state->callback && state->callback(current, state->total, state->opaque);
}

static int decode_progress_finish(decode_progress *state,
                                  char *error, size_t error_size) {
    return h3_memory_checkpoint(state->callback && state->callback(
        state->total, state->total, state->opaque),
        "video VAE decode", error, error_size);
}

static int decoder_decode_chunk(h3_video_vae_decoder *decoder,
                                const float *normalized_latent,
                                int latent_time, int chunk,
                                int selected_frame,
                                decode_progress *progress,
                                h3_video_frames *output,
                                char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (!decoder || !normalized_latent || !output || latent_time < 7 ||
        chunk < 0 || chunk > (latent_time - CHUNK_LATENT_TIME) / 5 ||
        selected_frame < -1 || selected_frame >= FIRST_CHUNK_FRAMES) {
        fail(error, error_size, "invalid resident video VAE chunk arguments");
        return 0;
    }
    int tile_count = decoder->y_axis.count * decoder->x_axis.count;
    float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
    if (!tiles) {
        fail(error, error_size, "out of memory retaining video VAE tiles");
        return 0;
    }
    int frame_count = selected_frame >= 0 ? 1 : FIRST_CHUNK_FRAMES;
    int ok = 1;
    int device_stitch=0;
#ifndef __APPLE__
    device_stitch=h3_gpu_sglang_stitch_begin(decoder->vae.gpu,
        decoder->y_axis.length,decoder->x_axis.length,decoder->y_axis.count,
        decoder->x_axis.count,decoder->y_axis.starts,decoder->x_axis.starts,
        selected_frame>=0?selected_frame:0,frame_count);
    if(device_stitch<0){free(tiles);return gpu_op(&decoder->vae,0,error,error_size,"reference spatial staging");}
#endif
    for (int tile_y = 0; tile_y < decoder->y_axis.count && ok; tile_y++)
        for (int tile_x = 0; tile_x < decoder->x_axis.count && ok; tile_x++) {
            if (!(ok = h3_memory_check(0, "video VAE decode", error, error_size))) break;
            float *input = extract_latent_tile(
                normalized_latent, latent_time, decoder->latent_h,
                decoder->latent_w, chunk * 5,
                decoder->y_axis.starts[tile_y] / SPATIAL_RATIO,
                decoder->x_axis.starts[tile_x] / SPATIAL_RATIO,
                CHUNK_LATENT_TIME, decoder->vae.latent_h,
                decoder->vae.latent_w, error, error_size);
            if (!input) {
                ok = 0;
                break;
            }
            if(!decoder->vae.reuse_buffers)free_tensor(&decoder->vae.latent);
            ok = prepare_input(&decoder->vae, input,
                               decoder->latent_mean, decoder->latent_std,
                               error, error_size) &&
                 run_resident_tile(&decoder->vae, decode_progress_block, progress, error, error_size);
            free(input);
            if (!ok) break;
            progress->offset += LAYERS;
#ifndef __APPLE__
            if(device_stitch){
                ok=gpu_op(&decoder->vae,h3_gpu_sglang_stitch_tile(decoder->vae.gpu,
                    decoder->vae.projected,tile_y*decoder->x_axis.count+tile_x),error,error_size,"reference spatial tile");
                continue;
            }
#endif
            h3_video_frames tile;
            memset(&tile, 0, sizeof(tile));
            ok = selected_frame >= 0 ?
                unpack_frame_range(&decoder->vae, selected_frame, 1, &tile,
                                   error, error_size) :
                unpack_frames(&decoder->vae, &tile, error, error_size);
            if (ok) {
                int index = tile_y * decoder->x_axis.count + tile_x;
                tiles[index] = tile.rgb;
            }
        }
    if(ok&&device_stitch){
#ifndef __APPLE__
        output->frames=frame_count;output->height=decoder->latent_h*16;output->width=decoder->latent_w*16;
        size_t n=(size_t)frame_count*output->height*output->width*3;
        output->rgb=malloc(n*sizeof(float));
        ok=output->rgb&&gpu_op(&decoder->vae,h3_gpu_sglang_stitch_read(decoder->vae.gpu,output->rgb),error,error_size,"reference spatial readback");
        if(!output->rgb)fail(error,error_size,"out of memory retaining stitched reference frames");
#endif
    }else if (ok) ok = stitch_tiles(tiles, &decoder->y_axis, &decoder->x_axis,
                              frame_count, reference_video(&decoder->vae), output, error, error_size);
    for (int index = 0; index < tile_count; index++) free(tiles[index]);
    free(tiles);
    if (!ok) { h3_gpu_cancel(decoder->vae.gpu); h3_video_frames_free(output); }
    return ok;
}

h3_video_vae_decoder *h3_video_vae_decoder_load(
                        const char *weight_directory,
                        const char *shader_source_path,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || latent_height < 1 ||
        latent_width < 1 || latent_height > INT32_MAX / SPATIAL_RATIO ||
        latent_width > INT32_MAX / SPATIAL_RATIO ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX) {
        fail(error, error_size, "invalid resident video VAE arguments");
        return NULL;
    }
    int tile_pixels = h3_video_vae_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO, error, error_size);
    if (!tile_pixels) return NULL;
    if (!h3_memory_check(0, "video VAE load", error, error_size)) return NULL;
    h3_video_vae_decoder *decoder = calloc(1, sizeof(*decoder));
    if (!decoder) {
        fail(error, error_size, "out of memory creating resident video VAE");
        return NULL;
    }
    decoder->latent_h = latent_height;
    decoder->latent_w = latent_width;

    int ok = load_latent_normalization(
        weight_directory, decoder->latent_mean, decoder->latent_std,
        error, error_size) &&
        tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                        &decoder->y_axis, error, error_size) &&
        tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                        &decoder->x_axis, error, error_size);
    if (ok && getenv("H3_PROFILE"))
        fprintf(stderr, "h3cli: resident video VAE tiles %dx%d at %d pixels\n",
                decoder->x_axis.count, decoder->y_axis.count, tile_pixels);
    vae_context *vae = &decoder->vae;
    if (ok) {
        vae->latent_h = decoder->y_axis.length / SPATIAL_RATIO;
        vae->latent_w = decoder->x_axis.length / SPATIAL_RATIO;
        vae->latent_t = CHUNK_LATENT_TIME;
        vae->output_frames = FIRST_CHUNK_FRAMES;
        vae->patches = (uint32_t)(CHUNK_LATENT_TIME * vae->latent_h *
                                  vae->latent_w);
        vae->sequence = vae->patches + SUFFIX;
        vae->weights = h3_weight_store_open(weight_directory,
                                             error, error_size);
        if (vae->weights)
            vae->gpu = h3_gpu_create(shader_source_path, error, error_size);
        if (vae->gpu)
            h3_gpu_profile_set_label(vae->gpu, "resident video VAE decoder");
        ok = vae->weights && vae->gpu &&
             gpu_op(vae,h3_gpu_video_vae_configure(vae->gpu,0)>=0,error,error_size,"configure full VAE") &&
             load_resident_weights(vae, progress, progress_opaque,
                                   error, error_size) &&
             prepare_rope(vae, error, error_size) &&
             allocate_activations(vae, error, error_size);
    }
    if (!ok) {
        h3_video_vae_decoder_free(decoder);
        return NULL;
    }
    return decoder;
}

int h3_video_vae_decoder_preview_progress(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output, int *output_frame_index,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output || !output_frame_index ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid video VAE preview arguments");
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    int chunk = chunks / 2;
    int local_frame = FIRST_CHUNK_FRAMES / 2;
    int output_frames = chunks * 17 + 5;
    int global_frame = chunk * 17 + local_frame;
    if (global_frame >= output_frames) global_frame = output_frames - 1;
    decode_progress work;
    if (!decode_progress_init(&work, progress, progress_opaque, 1,
                             decoder->y_axis.count, decoder->x_axis.count,
                             error, error_size)) return 0;
    int ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, local_frame, &work, output,
                                  error, error_size);
    if(ok)reference_video_finish(&decoder->vae,output->rgb,(size_t)output->frames*output->height*output->width*3);
    if (ok) ok = decode_progress_finish(&work, error, error_size);
    if (ok) *output_frame_index = global_frame;
    else h3_video_frames_free(output);
    return ok;
}

int h3_video_vae_decoder_decode_progress(h3_video_vae_decoder *decoder,
                        const float *normalized_latent, int latent_time,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!decoder || !normalized_latent || !output ||
        latent_time < CHUNK_LATENT_TIME || (latent_time - 2) % 5) {
        fail(error, error_size, "invalid resident video VAE decode arguments");
        return 0;
    }
    if (!h3_memory_check(0, "video VAE decode", error, error_size)) return 0;
    int chunks = (latent_time - 2) / 5;
    decode_progress work;
    if (!decode_progress_init(&work, progress, progress_opaque, chunks,
                             decoder->y_axis.count, decoder->x_axis.count,
                             error, error_size)) return 0;
    int output_frames = chunks * 17 + 5;
    int pixel_h = decoder->latent_h * SPATIAL_RATIO;
    int pixel_w = decoder->latent_w * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = malloc(output_elements * sizeof(*final_rgb));
    float *temporal_overlap = malloc(5 * frame_elements *
                                     sizeof(*temporal_overlap));
    if (!final_rgb || !temporal_overlap) {
        free(final_rgb);
        free(temporal_overlap);
        fail(error, error_size, "out of memory retaining decoded video frames");
        return 0;
    }
    int ok = 1;
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        ok = decoder_decode_chunk(decoder, normalized_latent, latent_time,
                                  chunk, -1, &work, &decoded, error, error_size);
        if (!ok) break;
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        reference_video_finish(&decoder->vae,output->rgb,output_elements);
        ok = h3_gpu_get_stats(decoder->vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    if (ok) ok = decode_progress_finish(&work, error, error_size);
    if (!ok) h3_video_frames_free(output);
    return ok;
}

int h3_video_vae_decoder_stream_progress(h3_video_vae_decoder *decoder,
    const float *latent, int time, h3_video_batch_callback callback, void *opaque,
    h3_video_vae_progress progress, void *progress_opaque, char *error, size_t size) {
    if (!decoder || !latent || !callback || time < 7 || time > 107 || (time-2)%5) {
        fail(error,size,"invalid streaming video VAE arguments"); return 0;
    }
    int chunks=(time-2)/5;
    decode_progress work;
    if(!decode_progress_init(&work,progress,progress_opaque,chunks,
        decoder->y_axis.count,decoder->x_axis.count,error,size))return 0;
    size_t frame=(size_t)decoder->latent_h*(size_t)decoder->latent_w*256*3;
    float *overlap=malloc(5*frame*sizeof(float));
    if(!overlap){fail(error,size,"out of memory retaining VAE overlap");return 0;}
    int ok=1;
    for(int chunk=0;chunk<chunks&&ok;chunk++){
        h3_video_frames decoded={0};
        ok=decoder_decode_chunk(decoder,latent,time,chunk,-1,&work,&decoded,error,size);
        if(ok&&chunk)for(int f=0;f<5;f++){
            float alpha=(float)f/5.0f;
            for(size_t p=0;p<frame;p++){
                size_t i=(size_t)f*frame+p;
                decoded.rgb[i]=overlap[i]*(1.0f-alpha)+decoded.rgb[i]*alpha;
            }
        }
        if(ok){
            memcpy(overlap,decoded.rgb+17*frame,5*frame*sizeof(float));
            int count=chunk+1==chunks?22:17;
            reference_video_finish(&decoder->vae,decoded.rgb,(size_t)count*frame);
            for(size_t i=0;i<(size_t)count*frame;i++)if(!isfinite(decoded.rgb[i])){
                fail(error,size,"non-finite streaming video VAE output");ok=0;break;}
            if(ok&&callback(decoded.rgb,chunk*17,count,decoded.width,decoded.height,opaque)){
                fail(error,size,"video VAE frame delivery cancelled");ok=0;}
        }
        h3_video_frames_free(&decoded);
    }
    free(overlap);
    if(ok)ok=decode_progress_finish(&work,error,size);
    return ok;
}

void h3_video_vae_decoder_free(h3_video_vae_decoder *decoder) {
    if (!decoder) return;
    cleanup(&decoder->vae);
    tile_axis_free(&decoder->y_axis);
    tile_axis_free(&decoder->x_axis);
    free(decoder);
}

static int decode_chunked(const char *weight_directory,
                          const char *shader_source_path,
                          const float *normalized_latent, int latent_time,
                          int latent_height, int latent_width,
                          const float *latent_mean, const float *latent_std,
                          int tile_pixels,
                          h3_video_vae_progress load_progress,
                          h3_video_vae_progress progress, void *progress_opaque,
                          h3_video_frames *output, char *error,
                          size_t error_size) {
    tile_axis y_axis, x_axis;
    memset(&y_axis, 0, sizeof(y_axis));
    memset(&x_axis, 0, sizeof(x_axis));
    int ok = tile_axis_build(latent_height * SPATIAL_RATIO, tile_pixels,
                             &y_axis,
                             error, error_size) &&
             tile_axis_build(latent_width * SPATIAL_RATIO, tile_pixels,
                             &x_axis,
                             error, error_size);
    if (!ok) {
        tile_axis_free(&y_axis); tile_axis_free(&x_axis);
        return 0;
    }
    int chunks = (latent_time - 2) / 5;
    decode_progress work;
    if (!decode_progress_init(&work, progress, progress_opaque, chunks,
                             y_axis.count, x_axis.count, error, error_size)) {
        tile_axis_free(&y_axis); tile_axis_free(&x_axis);
        return 0;
    }
    if (getenv("H3_PROFILE"))
        fprintf(stderr, "h3cli: video VAE tiles %dx%d at %d pixels\n",
                x_axis.count, y_axis.count, tile_pixels);
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = y_axis.length / SPATIAL_RATIO;
    vae.latent_w = x_axis.length / SPATIAL_RATIO;
    vae.latent_t = CHUNK_LATENT_TIME;
    vae.output_frames = FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * vae.latent_h * vae.latent_w);
    vae.sequence = vae.patches + SUFFIX;
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (vae.weights)
        vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    ok = vae.weights && vae.gpu &&
         gpu_op(&vae,h3_gpu_video_vae_configure(vae.gpu,0)>=0,error,error_size,"configure full VAE") &&
         load_resident_weights(&vae, load_progress, progress_opaque,
                               error, error_size) &&
         prepare_rope(&vae, error, error_size) &&
         allocate_activations(&vae, error, error_size);
    int tile_count = y_axis.count * x_axis.count;
    int output_frames = chunks * 17 + 5;
    int pixel_h = latent_height * SPATIAL_RATIO;
    int pixel_w = latent_width * SPATIAL_RATIO;
    size_t frame_elements = (size_t)pixel_h * (size_t)pixel_w * 3;
    size_t output_elements = (size_t)output_frames * frame_elements;
    float *final_rgb = ok ? malloc(output_elements * sizeof(*final_rgb)) : NULL;
    float *temporal_overlap = ok ? malloc(5 * frame_elements *
                                          sizeof(*temporal_overlap)) : NULL;
    if (ok && (!final_rgb || !temporal_overlap)) {
        fail(error, error_size, "out of memory retaining decoded video frames");
        ok = 0;
    }
    for (int chunk = 0; chunk < chunks && ok; chunk++) {
        float **tiles = calloc((size_t)tile_count, sizeof(*tiles));
        if (!tiles) {
            fail(error, error_size, "out of memory retaining video VAE tiles");
            ok = 0;
            break;
        }
        for (int tile_y = 0; tile_y < y_axis.count && ok; tile_y++)
            for (int tile_x = 0; tile_x < x_axis.count && ok; tile_x++) {
                float *input = extract_latent_tile(normalized_latent,
                    latent_time, latent_height, latent_width, chunk * 5,
                    y_axis.starts[tile_y] / SPATIAL_RATIO,
                    x_axis.starts[tile_x] / SPATIAL_RATIO,
                    CHUNK_LATENT_TIME, vae.latent_h, vae.latent_w,
                    error, error_size);
                if (!input) {
                    ok = 0;
                    break;
                }
                if(!vae.reuse_buffers)free_tensor(&vae.latent);
                ok = prepare_input(&vae, input, latent_mean, latent_std,
                                   error, error_size) &&
                     run_resident_tile(&vae, decode_progress_block, &work, error, error_size);
                free(input);
                if (!ok) break;
                work.offset += LAYERS;
                h3_video_frames tile;
                memset(&tile, 0, sizeof(tile));
                ok = unpack_frames(&vae, &tile, error, error_size);
                if (ok) {
                    int index = tile_y * x_axis.count + tile_x;
                    tiles[index] = tile.rgb;
                }
            }
        h3_video_frames decoded;
        memset(&decoded, 0, sizeof(decoded));
        if (ok) ok = stitch_tiles(tiles, &y_axis, &x_axis,
                                  FIRST_CHUNK_FRAMES, reference_video(&vae), &decoded,
                                  error, error_size);
        for (int index = 0; index < tile_count; index++) free(tiles[index]);
        free(tiles);
        if (!ok) {
            h3_video_frames_free(&decoded);
            break;
        }
        if (chunk) for (int frame = 0; frame < 5; frame++) {
            float alpha = (float)frame / 5.0f;
            size_t base = (size_t)frame * frame_elements;
            for (size_t index = 0; index < frame_elements; index++)
                decoded.rgb[base + index] = temporal_overlap[base + index] *
                    (1.0f - alpha) + decoded.rgb[base + index] * alpha;
        }
        memcpy(final_rgb + (size_t)chunk * 17 * frame_elements,
               decoded.rgb, 17 * frame_elements * sizeof(*final_rgb));
        memcpy(temporal_overlap, decoded.rgb + 17 * frame_elements,
               5 * frame_elements * sizeof(*temporal_overlap));
        h3_video_frames_free(&decoded);
    }
    if (ok) {
        memcpy(final_rgb + (size_t)chunks * 17 * frame_elements,
               temporal_overlap, 5 * frame_elements * sizeof(*final_rgb));
        output->frames = output_frames;
        output->height = pixel_h;
        output->width = pixel_w;
        output->rgb = final_rgb;
        final_rgb = NULL;
        reference_video_finish(&vae,output->rgb,output_elements);
        ok = h3_gpu_get_stats(vae.gpu, &output->gpu_stats);
    }
    free(final_rgb);
    free(temporal_overlap);
    cleanup(&vae);
    tile_axis_free(&y_axis); tile_axis_free(&x_axis);
    if (ok) ok = decode_progress_finish(&work, error, error_size);
    return ok;
}

void h3_video_frames_free(h3_video_frames *frames) {
    if (!frames) return;
    free(frames->rgb);
    memset(frames, 0, sizeof(*frames));
}

int h3_video_vae_decode_phased(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress load_progress,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    if (error && error_size) error[0] = '\0';
    if (!weight_directory || !shader_source_path || !normalized_latent ||
        !output || latent_time < 2 ||
        (latent_time != 2 && (latent_time < CHUNK_LATENT_TIME ||
                              (latent_time - 2) % 5)) ||
        latent_height < 1 || latent_width < 1 ||
        latent_height > INT32_MAX / SPATIAL_RATIO ||
        latent_width > INT32_MAX / SPATIAL_RATIO ||
        (uint64_t)(unsigned)latent_height * (uint64_t)(unsigned)latent_width >
            UINT32_MAX ||
        (uint32_t)(latent_height * latent_width) >
            (UINT32_MAX - SUFFIX) / CHUNK_LATENT_TIME) {
        fail(error, error_size, "invalid first-chunk video VAE arguments");
        return 0;
    }
    int tile_pixels = h3_video_vae_tile_pixels(
        latent_height * SPATIAL_RATIO, latent_width * SPATIAL_RATIO, error, error_size);
    if (!tile_pixels) return 0;
    int chunked = latent_time > CHUNK_LATENT_TIME ||
        latent_height > tile_pixels / SPATIAL_RATIO ||
        latent_width > tile_pixels / SPATIAL_RATIO;
    h3_video_vae_progress initial = chunked ? load_progress : progress;
    if (!h3_memory_checkpoint(initial && initial(0, LAYERS, progress_opaque),
                               "video VAE decode", error, error_size)) return 0;
    vae_context vae;
    memset(&vae, 0, sizeof(vae));
    vae.latent_h = latent_height;
    vae.latent_w = latent_width;
    vae.latent_t = latent_time;
    vae.output_frames = latent_time == 2 ? 5 : FIRST_CHUNK_FRAMES;
    vae.patches = (uint32_t)(CHUNK_LATENT_TIME * latent_height * latent_width);
    vae.sequence = vae.patches + SUFFIX;
    float latent_mean[LATENT_CHANNELS], latent_std[LATENT_CHANNELS];
    if (!load_latent_normalization(weight_directory, latent_mean, latent_std,
                                   error, error_size)) return 0;

    if (latent_time == 2 &&
        (latent_height > TILE_PIXELS / SPATIAL_RATIO ||
         latent_width > TILE_PIXELS / SPATIAL_RATIO)) {
        fail(error, error_size,
             "the two-token diagnostic VAE path supports one spatial tile");
        return 0;
    }
    if (chunked) {
        int ok = decode_chunked(weight_directory, shader_source_path,
                                normalized_latent, latent_time, latent_height,
                                latent_width, latent_mean, latent_std,
                                tile_pixels, load_progress, progress,
                                progress_opaque, output, error, error_size);
        if (!ok) h3_video_frames_free(output);
        return ok;
    }
    vae.weights = h3_weight_store_open(weight_directory, error, error_size);
    if (!vae.weights) return 0;
    vae.gpu = h3_gpu_create(shader_source_path, error, error_size);
    if (vae.gpu)
        h3_gpu_profile_set_label(vae.gpu, "video VAE decoder");
    decode_progress work;
    /* The streaming decoder runs one tile and one temporal chunk. */
    decode_progress_init(&work, progress, progress_opaque, 1, 1, 1, NULL, 0);
    int ok = vae.gpu &&
        gpu_op(&vae,h3_gpu_video_vae_configure(vae.gpu,0)>=0,error,error_size,"configure full VAE") &&
        load_input_weights(&vae, error, error_size) &&
        prepare_input(&vae, normalized_latent, latent_mean, latent_std,
                      error, error_size) &&
        prepare_rope(&vae, error, error_size) &&
        allocate_activations(&vae, error, error_size) &&
        run_decoder(&vae, decode_progress_block, &work, error, error_size) &&
        unpack_frames(&vae, output, error, error_size);
    if(ok)reference_video_finish(&vae,output->rgb,(size_t)output->frames*output->height*output->width*3);
    if (ok) ok = decode_progress_finish(&work, error, error_size);
    if (!ok) h3_video_frames_free(output);
    cleanup(&vae);
    return ok;
}

int h3_video_vae_decode(const char *weight_directory,
                        const char *shader_source_path,
                        const float *normalized_latent, int latent_time,
                        int latent_height, int latent_width,
                        h3_video_vae_progress progress, void *progress_opaque,
                        h3_video_frames *output,
                        char *error, size_t error_size) {
    return h3_video_vae_decode_phased(weight_directory, shader_source_path,
        normalized_latent, latent_time, latent_height, latent_width,
        progress, progress, progress_opaque, output, error, error_size);
}

int h3_video_vae_decoder_decode(h3_video_vae_decoder *decoder,
        const float *latent, int time, h3_video_frames *output,
        char *error, size_t error_size) {
    return h3_video_vae_decoder_decode_progress(decoder, latent, time,
        NULL, NULL, output, error, error_size);
}

int h3_video_vae_decoder_preview(h3_video_vae_decoder *decoder,
        const float *latent, int time, h3_video_frames *output, int *frame_index,
        char *error, size_t error_size) {
    return h3_video_vae_decoder_preview_progress(decoder, latent, time,
        NULL, NULL, output, frame_index, error, error_size);
}

/* The image role owns its own context; it can never enter the video cache. */
struct h3_image_vae_decoder {
    h3_video_vae_decoder core;
    h3_image_vae_info info;
};
const h3_image_vae_info *h3_image_vae_info_get(const h3_image_vae_decoder *d) { return d ? &d->info : NULL; }
void h3_image_vae_free(h3_image_vae_decoder *d) {
    if (!d) return;
    cleanup(&d->core.vae); tile_axis_free(&d->core.y_axis); tile_axis_free(&d->core.x_axis); free(d);
}
h3_image_vae_decoder *h3_image_vae_load(const char *path, const char *shader,
    int h, int w, h3_video_vae_progress progress, void *opaque, char *error, size_t size) {
    if (error && size) *error = 0;
    if (!path || !*path || !shader || !*shader || !h3_still_geometry(h,w,error,size)) return NULL;
    const char *override = getenv("H3_VAE_TILE_PIXELS");
    if (override && *override && strcmp(override,"256")) {
        fail(error,size,"image VAE requires the released 256/64 tile profile; unset H3_VAE_TILE_PIXELS"); return NULL;
    }
    if (!h3_memory_checkpoint(progress && progress(0,LAYERS,opaque),"image VAE load",error,size)) return NULL;
    h3_image_vae_decoder *d = calloc(1,sizeof(*d));
    if (!d) { fail(error,size,"out of memory creating image decoder"); return NULL; }
    h3_video_vae_decoder *core = &d->core; vae_context *v = &core->vae;
    int ok = h3_image_vae_inspect(path,&d->info,error,size) &&
        h3_memory_check(UINT64_C(11000000000),"image VAE resident load",error,size) &&
        tile_axis_build(h*16,256,&core->y_axis,error,size) && tile_axis_build(w*16,256,&core->x_axis,error,size);
    if (ok) {
        core->latent_h=h; core->latent_w=w;
        memcpy(core->latent_mean,d->info.mean,sizeof(core->latent_mean));
        memcpy(core->latent_std,d->info.deviation,sizeof(core->latent_std));
        v->image_weights=1;v->image_slice=d->info.output_slice;v->latent_t=1;v->output_frames=1;
        v->latent_h=core->y_axis.length/16;v->latent_w=core->x_axis.length/16;
        v->patches=(uint32_t)(v->latent_h*v->latent_w);v->sequence=v->patches+SUFFIX;
        v->weights=h3_weight_store_open_file(path,error,size);
        if(v->weights)v->gpu=h3_gpu_create(shader,error,size);
        if(v->gpu)h3_gpu_profile_set_label(v->gpu,"image VAE decoder F16-to-F32 T=1");
        ok=v->weights && v->gpu && load_resident_weights(v,progress,opaque,error,size) &&
            prepare_rope(v,error,size) && allocate_activations(v,error,size);
    }
    if (!ok) { h3_image_vae_free(d);return NULL; }
    return d;
}
int h3_image_vae_decode(h3_image_vae_decoder *d, const float *z,
    h3_video_vae_progress progress, void *opaque, h3_video_frames *out, char *error, size_t size) {
    if (error && size)
        *error = 0;
    if (out)
        memset(out, 0, sizeof(*out));
    if(!d || !z || !out || d->info.output_slice<0 || d->info.output_slice>3 ||
        d->core.vae.latent_t!=1 || !d->core.vae.image_weights) {
        fail(error,size,"invalid direct image decode arguments");return 0;
    }
    h3_video_vae_decoder *c=&d->core;vae_context *v=&c->vae;
    size_t count=(size_t)24*c->latent_h*c->latent_w;
    for(size_t i=0;i<count;i++)if(!isfinite(z[i])) {fail(error,size,"nonfinite still latent");return 0;}
    decode_progress work;
    if(!decode_progress_init(&work,progress,opaque,1,c->y_axis.count,c->x_axis.count,error,size))return 0;
    int tiles=c->y_axis.count*c->x_axis.count;
    float **rgb=calloc((size_t)tiles,sizeof(*rgb));
    if(!rgb){fail(error,size,"out of memory assembling still");return 0;}
    int ok=1;
    for(int y=0;ok && y<c->y_axis.count;y++)for(int x=0;ok && x<c->x_axis.count;x++) {
        if(!(ok=h3_memory_checkpoint(progress && progress(work.offset,work.total,opaque),"image VAE decode",error,size)))break;
        float *input=extract_latent_tile(z,1,c->latent_h,c->latent_w,0,c->y_axis.starts[y]/16,c->x_axis.starts[x]/16,1,v->latent_h,v->latent_w,error,size);
        ok=input && prepare_input(v,input,c->latent_mean,c->latent_std,error,size) && run_resident_tile(v,decode_progress_block,&work,error,size);
        free(input);work.offset+=LAYERS;
        h3_video_frames tile={0};
        if(ok)ok=unpack_frame_range(v,0,1,&tile,error,size);
        if(ok)rgb[y*c->x_axis.count+x]=tile.rgb;
    }
    if(ok)ok=stitch_tiles(rgb,&c->y_axis,&c->x_axis,1,0,out,error,size) &&
        h3_gpu_get_stats(v->gpu,&out->gpu_stats) && decode_progress_finish(&work,error,size);
    for (int i = 0; i < tiles; i++)
        free(rgb[i]);
    free(rgb);
    if(!ok){h3_gpu_cancel(v->gpu);h3_video_frames_free(out);}return ok;
}
