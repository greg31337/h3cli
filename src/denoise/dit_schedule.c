#include "src/memory.h"
#include "src/sglang/sglang.h"
#include "src/denoise/dit_schedule.h"
#include "src/sampling/sampler_state.h"

#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

enum {
    TIME_INPUT = 256,
    TIME_HIDDEN = 5376,
    BLOCK_OUTPUT = H3_DIT_MODALITIES * H3_DIT_ADALN_SLOTS * H3_DIT_HIDDEN,
    FINAL_OUTPUT = 2 * H3_DIT_HIDDEN
};

struct h3_dit_schedule {
    h3_gpu *gpu;
    int steps;
    uint32_t time_rows;
    float *times;
    uint32_t *video_rows;
    uint32_t *audio_rows;
    uint32_t *visual_condition_rows;
    uint32_t *audio_condition_rows;
    h3_bridge_profile *bridge;
    uint32_t *bridge_rows;
    h3_gpu_tensor *blocks[H3_DIT_BLOCKS];
    h3_gpu_tensor *final;
};

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int gpu_op(h3_gpu *gpu, int ok, char *error, size_t error_size,
                  const char *operation) {
    if (ok) return 1;
    fail(error, error_size, "%s: %s", operation, h3_gpu_error(gpu));
    return 0;
}

static h3_gpu_tensor *weight_f32_1d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t width,
                                    char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_f32(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_f32_2d(const h3_weight_store *store, h3_gpu *gpu,
                                    const char *name, uint64_t rows,
                                    uint64_t columns, char *error,
                                    size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_f32(store, gpu, name, 2, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_1d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t width,
                                     char *error, size_t error_size) {
    uint64_t shape[] = {width};
    return h3_weight_load_bf16(store, gpu, name, 1, shape, error, error_size);
}

static h3_gpu_tensor *weight_bf16_2d(const h3_weight_store *store, h3_gpu *gpu,
                                     const char *name, uint64_t rows,
                                     uint64_t columns, char *error,
                                     size_t error_size) {
    uint64_t shape[] = {rows, columns};
    return h3_weight_load_bf16(store, gpu, name, 2, shape, error, error_size);
}

static void free_tensor(h3_gpu_tensor **tensor) {
    h3_gpu_tensor_free(*tensor);
    *tensor = NULL;
}

static int prepare_rows(h3_dit_schedule *schedule,
                        const h3_sigma_schedule *sigmas,
                        int visual_condition, int audio_condition,
                        float **features_out, char *error,
                        size_t error_size) {
    schedule->steps = sigmas->steps;
    schedule->video_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->video_rows));
    schedule->audio_rows = calloc((size_t)sigmas->steps,
                                  sizeof(*schedule->audio_rows));
    if (visual_condition)
        schedule->visual_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->visual_condition_rows));
    if (audio_condition)
        schedule->audio_condition_rows = calloc(
            (size_t)sigmas->steps, sizeof(*schedule->audio_condition_rows));
    if (!schedule->video_rows || !schedule->audio_rows ||
        (visual_condition && !schedule->visual_condition_rows) ||
        (audio_condition && !schedule->audio_condition_rows)) {
        fail(error, error_size, "out of memory allocating timestep row maps");
        return 0;
    }
    uint32_t count = 0;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (video == audio) {
            schedule->video_rows[step] = count;
            schedule->audio_rows[step] = count++;
        } else if (video < audio) {
            schedule->video_rows[step] = count++;
            schedule->audio_rows[step] = count++;
        } else {
            schedule->audio_rows[step] = count++;
            schedule->video_rows[step] = count++;
        }
    }
    uint32_t visual_condition_row = UINT32_MAX;
    uint32_t audio_condition_row = UINT32_MAX;
    if (visual_condition) visual_condition_row = count++;
    if (audio_condition) audio_condition_row = count++;
    for (int step = 0; step < sigmas->steps; step++) {
        float video = 1.0f - sigmas->video[step];
        float audio = 1.0f - sigmas->audio[step];
        if (visual_condition)
            schedule->visual_condition_rows[step] = video >= 0.999f ?
                schedule->video_rows[step] : visual_condition_row;
        if (audio_condition)
            schedule->audio_condition_rows[step] = audio >= 1.0f ?
                schedule->audio_rows[step] : audio_condition_row;
    }
    schedule->time_rows = count;
    if (!count || count > UINT32_MAX / TIME_INPUT) {
        fail(error, error_size, "invalid number of timestep rows");
        return 0;
    }
    float *times = calloc(count, sizeof(*times));
    float *features = malloc((size_t)count * TIME_INPUT * sizeof(*features));
    if (!times || !features) {
        free(times);
        free(features);
        fail(error, error_size, "out of memory allocating timestep features");
        return 0;
    }
    for (int step = 0; step < sigmas->steps; step++) {
        times[schedule->video_rows[step]] = 1.0f - sigmas->video[step];
        times[schedule->audio_rows[step]] = 1.0f - sigmas->audio[step];
    }
    if (visual_condition) times[visual_condition_row] = 0.999f;
    if (audio_condition) times[audio_condition_row] = 1.0f;
    for (uint32_t row = 0; row < count; row++) {
        for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
            float frequency = expf(-logf(10000.0f) *
                                   (float)index / (float)(TIME_INPUT / 2));
            float angle = times[row] * frequency;
            features[(size_t)row * TIME_INPUT + index] = cosf(angle);
            features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] =
                sinf(angle);
        }
    }
    schedule->times = times;
    *features_out = features;
    return 1;
}

h3_dit_schedule *h3_dit_schedule_plan(const h3_sigma_schedule *sigmas,
    int visual_condition, int audio_condition, char *error, size_t error_size) {
    if (!sigmas || sigmas->steps<1 || sigmas->steps>H3_MAX_STEPS) return NULL;
    h3_dit_schedule *schedule=calloc(1,sizeof(*schedule));
    float *features=NULL;
    if (schedule && !prepare_rows(schedule,sigmas,visual_condition,audio_condition,
        &features,error,error_size)) { h3_dit_schedule_free(schedule); schedule=NULL; }
    free(features);
    return schedule;
}

float h3_dit_schedule_timestep(const h3_dit_schedule *schedule, uint32_t row) {
    return schedule && row<schedule->time_rows ? schedule->times[row] : NAN;
}

uint32_t h3_dit_schedule_class_row(const h3_dit_schedule *s, int step,
    h3_target_row_class kind) {
    if (!s || step < 0 || step >= s->steps) return UINT32_MAX;
    switch (kind) {
    case H3_ROW_GENERATED_VIDEO: return s->video_rows[step];
    case H3_ROW_GENERATED_AUDIO: return s->audio_rows[step];
    case H3_ROW_PRESERVED_VIDEO: return s->visual_condition_rows ? s->visual_condition_rows[step] : UINT32_MAX;
    case H3_ROW_PRESERVED_AUDIO: return s->audio_condition_rows ? s->audio_condition_rows[step] : UINT32_MAX;
    default:
        return s->bridge_rows && kind >= H3_ROW_BRIDGE_VIDEO_FIRST && kind < H3_TARGET_ROW_CLASSES ?
            s->bridge_rows[(size_t)step * H3_TARGET_ROW_CLASSES + (size_t)kind] : UINT32_MAX;
    }
}

h3_dit_schedule *h3_dit_schedule_plan_bridge(const h3_sigma_schedule *sigmas,
    const h3_bridge_profile *p, char *error, size_t error_size) {
    if (!h3_bridge_profile_valid(p)) {
        fail(error, error_size, "invalid bridge temporal profile");
        return NULL;
    }
    if (!sigmas || sigmas->steps < 1 || sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid bridge sigma schedule");
        return NULL;
    }
    for (int step = 0; step < sigmas->steps; step++) {
        if (!isfinite(h3_bridge_class_sigma(p, H3_ROW_GENERATED_VIDEO,
            sigmas->video[step], sigmas->audio[step]))) {
            fail(error, error_size, "bridge sigmas must be finite and in [0,1]");
            return NULL;
        }
    }
    /* The original time rows are a stable prefix. References and text retain
     * precisely the same indices, values and tags as the hard schedule. */
    h3_dit_schedule *s = h3_dit_schedule_plan(sigmas, 1, 1, error, error_size);
    if (!s) return NULL;
    size_t count = (size_t)s->steps * H3_TARGET_ROW_CLASSES;
    s->bridge = malloc(sizeof(*s->bridge));
    s->bridge_rows = malloc(count * sizeof(*s->bridge_rows));
    float *times = realloc(s->times, ((size_t)s->time_rows + count) * sizeof(*times));
    if (times) s->times = times;
    if (!s->bridge || !s->bridge_rows || !times) {
        fail(error, error_size, "out of memory planning bridge time rows");
        h3_dit_schedule_free(s);
        return NULL;
    }
    *s->bridge = *p;
    for (size_t i = 0; i < count; i++) s->bridge_rows[i] = UINT32_MAX;
    for (int step = 0; step < s->steps; step++) {
        for (int c = H3_ROW_BRIDGE_VIDEO_FIRST; c < H3_TARGET_ROW_CLASSES; c++) {
            if (!p->active[c]) continue;
            float time = h3_bridge_class_timestep(p, (h3_target_row_class)c,
                sigmas->video[step], sigmas->audio[step]);
            uint32_t row = UINT32_MAX;
            /* Share equal timestep vectors across modalities/classes at this
             * step, including generated and existing condition-strength rows. */
            for (int previous = 0; previous < c; previous++) {
                uint32_t candidate = h3_dit_schedule_class_row(s, step, (h3_target_row_class)previous);
                if (candidate != UINT32_MAX && s->times[candidate] == time) { row = candidate; break; }
            }
            if (row == UINT32_MAX) { row = s->time_rows++; s->times[row] = time; }
            s->bridge_rows[(size_t)step * H3_TARGET_ROW_CLASSES + (size_t)c] = row;
        }
    }
    return s;
}

static h3_gpu_tensor *time_embeddings(const h3_weight_store *weights,
                                      h3_gpu *gpu, uint32_t rows,
                                      const float *features, const float *times,int released_refvideo, char *error,
                                      size_t error_size) {
    h3_gpu_tensor *input = h3_gpu_tensor_from_f32(
        gpu, features, (size_t)rows * TIME_INPUT);
    h3_gpu_tensor *time_values=NULL;
#ifndef __APPLE__
    if(h3_sglang_requested()&&times)time_values=h3_gpu_tensor_from_f32(gpu,times,rows);
#else
    (void)times;
#endif
    h3_gpu_tensor *in_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_in.weight", TIME_HIDDEN, TIME_INPUT,
        error, error_size);
    h3_gpu_tensor *in_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_in.bias", TIME_HIDDEN, error, error_size);
    h3_gpu_tensor *out_w = weight_f32_2d(weights, gpu,
        "time_embedder.proj_out.weight", H3_DIT_TIME_DIM, TIME_HIDDEN,
        error, error_size);
    h3_gpu_tensor *out_b = weight_f32_1d(weights, gpu,
        "time_embedder.proj_out.bias", H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *hidden = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *activated = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * TIME_HIDDEN);
    h3_gpu_tensor *output = h3_gpu_tensor_new_f32(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *bf16 = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *silu = h3_gpu_tensor_new_bf16(
        gpu, (size_t)rows * H3_DIT_TIME_DIM);
    h3_gpu_tensor *result = NULL;
    h3_gpu_tensor *all[] = {input, in_w, in_b, out_w, out_b, hidden,
                            activated, output, bf16, silu};
    for (size_t index = 0; index < sizeof(all) / sizeof(*all); index++) {
        if (!all[index]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate timestep tensors: %s",
                     h3_gpu_error(gpu));
            goto cleanup;
        }
    }
    if (!gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin timestep embedding") ||
#ifndef __APPLE__
        (h3_sglang_requested()&&times&&!gpu_op(gpu,
            time_values&&h3_gpu_sglang_time_features(gpu,input,time_values,rows),
            error,error_size,"reference timestep features")) ||
#endif
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, hidden, input, in_w, in_b, rows,
                                       TIME_INPUT, TIME_HIDDEN),
                error, error_size, "timestep input projection") ||
        !gpu_op(gpu, h3_gpu_silu_f32(gpu, activated, hidden,
                                     rows * TIME_HIDDEN),
                error, error_size, "timestep SiLU") ||
        !gpu_op(gpu, h3_gpu_linear_f32(gpu, output, activated, out_w, out_b,
                                       rows, TIME_HIDDEN, H3_DIT_TIME_DIM),
                error, error_size, "timestep output projection")) goto cleanup;
    /* The released AdaLN activates the F32 time embedding before casting to
     * the BF16 projection. Keep the existing rounding for Metal and still routes. */
    if (released_refvideo || h3_sglang_requested()) {
        if (!gpu_op(gpu, h3_gpu_silu_f32(gpu, output, output,
                        rows * H3_DIT_TIME_DIM),
                    error, error_size, "released timestep AdaLN SiLU") ||
            !gpu_op(gpu, h3_gpu_cast_f32_to_bf16(gpu, silu, output,
                        rows * H3_DIT_TIME_DIM),
                    error, error_size, "released timestep BF16 cast")) goto cleanup;
    } else if (!gpu_op(gpu, h3_gpu_cast_f32_to_bf16(
                        gpu, bf16, output, rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep BF16 cast") ||
        !gpu_op(gpu, h3_gpu_silu_bf16(gpu, silu, bf16,
                                      rows * H3_DIT_TIME_DIM),
                error, error_size, "timestep AdaLN SiLU")) goto cleanup;
    if (!gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit timestep embedding")) {
        goto cleanup;
    }
    result = silu;
    silu = NULL;
cleanup:
    free_tensor(&time_values);
    free_tensor(&input);
    free_tensor(&in_w);
    free_tensor(&in_b);
    free_tensor(&out_w);
    free_tensor(&out_b);
    free_tensor(&hidden);
    free_tensor(&activated);
    free_tensor(&output);
    free_tensor(&bf16);
    free_tensor(&silu);
    return result;
}

static h3_dit_schedule *precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition, const h3_bridge_profile *bridge,
    int released_refvideo,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!weights || !gpu || !sigmas || sigmas->steps < 1 ||
        sigmas->steps > H3_MAX_STEPS) {
        fail(error, error_size, "invalid AdaLN schedule arguments");
        return NULL;
    }
    if (!h3_memory_checkpoint(progress && progress(0, (int)H3_DIT_BLOCKS, progress_opaque),
                               "precompute AdaLN", error, error_size)) return NULL;
    h3_dit_schedule *schedule = bridge ?
        h3_dit_schedule_plan_bridge(sigmas, bridge, error, error_size) : calloc(1, sizeof(*schedule));
    if (!schedule) {
        fail(error, error_size, "out of memory creating AdaLN schedule");
        return NULL;
    }
    schedule->gpu = gpu;
    float *features = NULL;
    if (bridge) {
        features = malloc((size_t)schedule->time_rows * TIME_INPUT * sizeof(*features));
        if (!features) { fail(error, error_size, "out of memory preparing bridge features"); goto failed; }
        for (uint32_t row = 0; row < schedule->time_rows; row++) {
            for (uint32_t index = 0; index < TIME_INPUT / 2; index++) {
                float frequency = expf(-logf(10000.0f) * (float)index / (float)(TIME_INPUT / 2));
                float angle = schedule->times[row] * frequency;
                features[(size_t)row * TIME_INPUT + index] = cosf(angle);
                features[(size_t)row * TIME_INPUT + TIME_INPUT / 2 + index] = sinf(angle);
            }
        }
    } else if (!prepare_rows(schedule, sigmas, visual_condition, audio_condition,
                             &features, error, error_size)) goto failed;
#ifndef __APPLE__
    /* T2VA has one shared timestep initially, then two modality timesteps.
     * Combining all rows changes the FP32 projection kernel and BF16 ties. */
    if(h3_sglang_requested()&&!visual_condition&&!audio_condition&&!bridge) {
        uint32_t groups[H3_MAX_STEPS];
        for(int step=0;step<schedule->steps;step++)
            groups[step]=schedule->video_rows[step]==schedule->audio_rows[step]?1u:2u;
        if(!h3_gpu_sglang_batch_groups(gpu,groups,(size_t)schedule->steps))goto failed;
    }
#endif
    h3_gpu_tensor *time = time_embeddings(weights, gpu, schedule->time_rows,
                                           features, schedule->times,released_refvideo, error, error_size);
    free(features);
    features = NULL;
    if (!time) goto failed;

    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (!h3_memory_check(0, "precompute AdaLN", error, error_size)) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
        char weight_name[128], bias_name[128], operation[128];
        snprintf(weight_name, sizeof(weight_name),
                 "blocks.%u.adaln_proj.linear.weight", block);
        snprintf(bias_name, sizeof(bias_name),
                 "blocks.%u.adaln_proj.linear.bias", block);
        h3_gpu_tensor *weight = weight_bf16_2d(
            weights, gpu, weight_name, BLOCK_OUTPUT, H3_DIT_TIME_DIM,
            error, error_size);
        h3_gpu_tensor *bias = weight_bf16_1d(
            weights, gpu, bias_name, BLOCK_OUTPUT, error, error_size);
        schedule->blocks[block] = h3_gpu_tensor_new_bf16(
            gpu, (size_t)schedule->time_rows * BLOCK_OUTPUT);
        if (!weight || !bias || !schedule->blocks[block]) {
            if (!error || !*error)
                fail(error, error_size, "cannot allocate AdaLN block %u: %s",
                     block, h3_gpu_error(gpu));
            free_tensor(&weight);
            free_tensor(&bias);
            h3_gpu_tensor_free(time);
            goto failed;
        }
        snprintf(operation, sizeof(operation), "AdaLN block %u", block);
        int ok = gpu_op(gpu, h3_gpu_begin(gpu), error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_linear_bf16(
                gpu, schedule->blocks[block], time, weight, bias,
                schedule->time_rows, H3_DIT_TIME_DIM, BLOCK_OUTPUT),
                error, error_size, operation) &&
            gpu_op(gpu, h3_gpu_submit(gpu), error, error_size, operation);
        free_tensor(&weight);
        free_tensor(&bias);
        if (!ok) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
        if (!h3_memory_checkpoint(progress && progress((int)block + 1,
                (int)H3_DIT_BLOCKS, progress_opaque), "precompute AdaLN", error, error_size)) {
            h3_gpu_tensor_free(time);
            goto failed;
        }
    }

    h3_gpu_tensor *final_w = weight_bf16_2d(
        weights, gpu, "final_layer.adaln_proj.linear.weight",
        FINAL_OUTPUT, H3_DIT_TIME_DIM, error, error_size);
    h3_gpu_tensor *final_b = weight_bf16_1d(
        weights, gpu, "final_layer.adaln_proj.linear.bias",
        FINAL_OUTPUT, error, error_size);
    schedule->final = h3_gpu_tensor_new_bf16(
        gpu, (size_t)schedule->time_rows * FINAL_OUTPUT);
    if (!final_w || !final_b || !schedule->final ||
        !gpu_op(gpu, h3_gpu_begin(gpu), error, error_size,
                "begin final AdaLN") ||
        !gpu_op(gpu, h3_gpu_linear_bf16(
            gpu, schedule->final, time, final_w, final_b, schedule->time_rows,
            H3_DIT_TIME_DIM, FINAL_OUTPUT), error, error_size,
            "final AdaLN projection") ||
        !gpu_op(gpu, h3_gpu_submit(gpu), error, error_size,
                "submit final AdaLN")) {
        if ((!error || !*error) && (!final_w || !final_b || !schedule->final))
            fail(error, error_size, "cannot allocate final AdaLN tensors: %s",
                 h3_gpu_error(gpu));
        free_tensor(&final_w);
        free_tensor(&final_b);
        h3_gpu_tensor_free(time);
        goto failed;
    }
    free_tensor(&final_w);
    free_tensor(&final_b);
    h3_gpu_tensor_free(time);
#ifndef __APPLE__
    if(h3_sglang_requested())h3_gpu_sglang_batch_groups(gpu,NULL,0);
#endif
    return schedule;

failed:
#ifndef __APPLE__
    if(h3_sglang_requested())h3_gpu_sglang_batch_groups(gpu,NULL,0);
#endif
    free(features);
    h3_dit_schedule_free(schedule);
    return NULL;
}

h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition, int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    return precompute(weights, gpu, sigmas, visual_condition, audio_condition,
        NULL, 0, progress, progress_opaque, error, error_size);
}

h3_dit_schedule *h3_dit_schedule_precompute_refvideo(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition, int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    return precompute(weights, gpu, sigmas, visual_condition, audio_condition,
        NULL, 1, progress, progress_opaque, error, error_size);
}

h3_dit_schedule *h3_dit_schedule_precompute_bridge(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, const h3_bridge_profile *profile,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size) {
    if (!h3_bridge_profile_valid(profile)) {
        fail(error, error_size, "invalid bridge modulation profile"); return NULL;
    }
    return precompute(weights, gpu, sigmas, 1, 1, profile, 0,
        progress, progress_opaque, error, error_size);
}

void h3_dit_schedule_free(h3_dit_schedule *schedule) {
    if (!schedule) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++)
        h3_gpu_tensor_free(schedule->blocks[block]);
    h3_gpu_tensor_free(schedule->final);
    free(schedule->video_rows);
    free(schedule->audio_rows);
    free(schedule->visual_condition_rows);
    free(schedule->audio_condition_rows);
    free(schedule->bridge);
    free(schedule->bridge_rows);
    free(schedule->times);
    free(schedule);
}

int h3_dit_schedule_steps(const h3_dit_schedule *schedule) {
    return schedule ? schedule->steps : 0;
}

uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule) {
    return schedule ? schedule->time_rows : 0;
}

uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->video_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step) {
    return schedule && step >= 0 && step < schedule->steps ?
        schedule->audio_rows[step] : UINT32_MAX;
}

uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->visual_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->visual_condition_rows[step] :
        UINT32_MAX;
}

uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step) {
    return schedule && schedule->audio_condition_rows && step >= 0 &&
        step < schedule->steps ? schedule->audio_condition_rows[step] :
        UINT32_MAX;
}

const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block) {
    return schedule && block < H3_DIT_BLOCKS ? schedule->blocks[block] : NULL;
}

double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block) {
    if (!schedule || block >= H3_DIT_BLOCKS || !schedule->blocks[block])
        return -1.0;
    size_t count = (size_t)schedule->time_rows * BLOCK_OUTPUT;
    uint16_t *values = malloc(count * sizeof(*values));
    if (!values || !h3_gpu_tensor_read_bf16(schedule->blocks[block], values,
                                             count)) {
        free(values);
        return -1.0;
    }
    double total = 0.0;
    size_t samples = 0;
    for (uint32_t row = 0; row < schedule->time_rows; row++)
        for (uint32_t modality = 0; modality < H3_DIT_MODALITIES; modality++)
            for (uint32_t slot = 2; slot <= 5; slot += 3) {
                size_t base = ((size_t)row * H3_DIT_MODALITIES *
                               H3_DIT_ADALN_SLOTS +
                               (size_t)modality * H3_DIT_ADALN_SLOTS + slot) *
                              H3_DIT_HIDDEN;
                for (uint32_t column = 0; column < H3_DIT_HIDDEN; column++) {
                    uint32_t bits = (uint32_t)values[base + column] << 16;
                    float value;
                    memcpy(&value, &bits, sizeof(value));
                    total += fabs((double)value);
                }
                samples += H3_DIT_HIDDEN;
            }
    free(values);
    return samples ? total / (double)samples : -1.0;
}

void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count) {
    if (!schedule || !active_blocks || count != H3_DIT_BLOCKS) return;
    for (unsigned block = 0; block < H3_DIT_BLOCKS; block++) {
        if (active_blocks[block]) continue;
        h3_gpu_tensor_free(schedule->blocks[block]);
        schedule->blocks[block] = NULL;
    }
}

const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule) {
    return schedule ? schedule->final : NULL;
}

int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count) {
    if (!schedule || step < 0 || step >= schedule->steps || !layout || !rows ||
        row_count != layout->seq_len || !layout->segments ||
        (text_tags && text_tag_count != (size_t)layout->signature[0])) return 0;
    if (schedule->bridge &&
        (layout->prefix.video_prefix_t != schedule->bridge->prefix.video_prefix_t ||
         layout->prefix.audio_prefix_t != schedule->bridge->prefix.audio_prefix_t ||
         layout->signature[1] < layout->prefix.video_prefix_t ||
         layout->signature[4] < layout->prefix.audio_prefix_t)) return 0;
    size_t text_index = 0;
    for (size_t seg_index = 0; seg_index < layout->segment_count; seg_index++) {
        const h3_segment *segment = &layout->segments[seg_index];
        if (segment->start > segment->stop || segment->stop > row_count)
            return 0;
        uint32_t time_row;
        uint32_t tag;
        switch (segment->kind) {
        case H3_SEG_TEXT:
            time_row = schedule->video_rows[step];
            for (size_t row = segment->start; row < segment->stop; row++) {
                uint32_t text_tag = text_tags ? text_tags[text_index] : 1u;
                if (text_tag >= H3_DIT_MODALITIES) return 0;
                rows[row] = time_row * H3_DIT_MODALITIES + text_tag;
                text_index++;
            }
            continue;
        case H3_SEG_COND:
        case H3_SEG_REF_IMAGE:
            if (!schedule->visual_condition_rows) return 0;
            time_row = schedule->visual_condition_rows[step];
            tag = 0;
            break;
        case H3_SEG_REF_AUDIO:
            if (!schedule->audio_condition_rows) return 0;
            time_row = schedule->audio_condition_rows[step];
            tag = 2;
            break;
        case H3_SEG_AUDIO:
            time_row = schedule->audio_rows[step];
            tag = 2;
            break;
        case H3_SEG_VIDEO:
            time_row = schedule->video_rows[step];
            tag = 0;
            break;
        default:
            return 0;
        }
        uint32_t modulation = time_row * H3_DIT_MODALITIES + tag;
        if (!(layout->prefix.video_prefix_t || layout->prefix.audio_prefix_t) ||
            (segment->kind != H3_SEG_VIDEO && segment->kind != H3_SEG_AUDIO)) {
            for (size_t row = segment->start; row < segment->stop; row++)
                rows[row] = modulation;
            continue;
        }
        for (size_t row = segment->start; row < segment->stop; row++) {
            uint32_t selected = modulation;
            if (schedule->bridge) {
                h3_target_row_class kind = h3_bridge_row_class(schedule->bridge,
                    segment->kind == H3_SEG_AUDIO, row-segment->start,
                    layout->signature[2], layout->signature[3], layout->signature[4]);
                uint32_t time = h3_dit_schedule_class_row(schedule, step, kind);
                if (time == UINT32_MAX) return 0;
                rows[row] = time * H3_DIT_MODALITIES + tag;
                continue;
            }
            if (segment->kind == H3_SEG_VIDEO || segment->kind == H3_SEG_AUDIO) {
                h3_target_row_class kind = h3_prefix_row_class(layout->prefix,
                    segment->kind == H3_SEG_AUDIO, row-segment->start,
                    layout->signature[2], layout->signature[3], layout->signature[4]);
                if (kind == H3_ROW_PRESERVED_VIDEO) {
                    if (!schedule->visual_condition_rows) return 0;
                    selected = schedule->visual_condition_rows[step]*H3_DIT_MODALITIES;
                } else if (kind == H3_ROW_PRESERVED_AUDIO) {
                    if (!schedule->audio_condition_rows) return 0;
                    selected = schedule->audio_condition_rows[step]*H3_DIT_MODALITIES+2;
                }
            }
            rows[row] = selected;
        }
    }
    return text_index == (size_t)layout->signature[0];
}

/* These payloads contain derived BF16 words only, never model/runtime objects. */
int h3_dit_schedule_export(const h3_dit_schedule *schedule,h3_sampler_state *state) {
    for(unsigned i=0;i<=H3_DIT_BLOCKS;i++) {
        const h3_gpu_tensor *source=i==H3_DIT_BLOCKS?schedule->final:schedule->blocks[i];
        if(!source) continue;
        if(state->prepared.count==H3_PREPARED_MAX) return 0;
        h3_prepared_tensor *t=&state->prepared.tensors[state->prepared.count++];
        t->id=i==H3_DIT_BLOCKS?2:100+i;
        t->elements=h3_gpu_tensor_elements(source); t->values=malloc(t->elements*2);
        if(!t->values || !h3_gpu_tensor_read_bf16(source,t->values,t->elements)) return 0;
    }
    return 1;
}
h3_dit_schedule *h3_dit_schedule_import(h3_gpu *gpu,const h3_sampler_state *state) {
    char error[128];
    h3_dit_schedule *s=state->layout.bridge ?
        h3_dit_schedule_plan_bridge(&state->sigmas,state->layout.bridge,error,sizeof(error)) :
        h3_dit_schedule_plan(&state->sigmas,
            state->layout.img_cond_rows!=0 || state->layout.prefix.video_prefix_t!=0,
            state->layout.audio_cond_rows!=0 || state->layout.prefix.audio_prefix_t!=0,error,sizeof(error));
    if(!s) return NULL;
    s->gpu=gpu;
    for(unsigned i=0;i<=H3_DIT_BLOCKS;i++) {
        size_t n=(size_t)s->time_rows*(i==H3_DIT_BLOCKS?FINAL_OUTPUT:BLOCK_OUTPUT);
        const h3_prepared_tensor *t=h3_prepared_find(&state->prepared,i==H3_DIT_BLOCKS?2:100+i,n);
        if(!t) { h3_dit_schedule_free(s); return NULL; }
        h3_gpu_tensor *tensor=h3_gpu_tensor_from_bf16(gpu,t->values,n);
        if(!tensor) { h3_dit_schedule_free(s); return NULL; }
        if(i==H3_DIT_BLOCKS) s->final=tensor; else s->blocks[i]=tensor;
    }
    return s;
}
