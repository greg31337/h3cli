#include "src/sampling/bridge.h"

#include <math.h>
#include <stdio.h>
#include <string.h>

static int fail(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error, size, "%s", message);
    return 0;
}

const char *h3_bridge_profile_name(h3_bridge_profile_type type) {
    switch (type) {
    case H3_BRIDGE_STEPPED: return "stepped";
    case H3_BRIDGE_LINEAR: return "linear";
    case H3_BRIDGE_EASE_OUT: return "ease-out";
    }
    return NULL;
}

float h3_bridge_strength(h3_bridge_profile_type type, float maximum, double time) {
    if (!h3_bridge_profile_name(type) || !isfinite(maximum) || maximum < 0 ||
        maximum > 1 || !isfinite(time) || time < 0) return NAN;
    if (time >= 1) return 0;
    double strength;
    if (type == H3_BRIDGE_STEPPED) {
        static const double levels[] = {1, 1, .8, .8, .6, .4, .2, .1};
        strength = levels[(int)(time * 8)];
    } else if (type == H3_BRIDGE_LINEAR) strength = 1 - time;
    else strength = 1 - time * time;
    return (float)((double)maximum * strength);
}

static uint8_t classify(const h3_bridge_profile *p, int audio, double time) {
    float strength = h3_bridge_strength(p->type, p->max_strength, time);
    int level = p->max_strength > 0 ?
        (int)floor((double)strength / p->max_strength * (double)H3_BRIDGE_LEVELS + .5) : 0;
    if (!level) return (uint8_t)(audio ? H3_ROW_PRESERVED_AUDIO : H3_ROW_PRESERVED_VIDEO);
    if (level > H3_BRIDGE_LEVELS) level = H3_BRIDGE_LEVELS;
    if (p->class_mask[H3_ROW_BRIDGE_VIDEO_FIRST + level - 1] == 0)
        return (uint8_t)(audio ? H3_ROW_PRESERVED_AUDIO : H3_ROW_PRESERVED_VIDEO);
    if (p->max_strength == 1 && level == H3_BRIDGE_LEVELS)
        return (uint8_t)(audio ? H3_ROW_GENERATED_AUDIO : H3_ROW_GENERATED_VIDEO);
    return (uint8_t)((audio ? H3_ROW_BRIDGE_AUDIO_FIRST : H3_ROW_BRIDGE_VIDEO_FIRST) + level - 1);
}

int h3_bridge_profile_build(int context_frames, int video_steps, float maximum,
    h3_bridge_profile_type type, h3_bridge_profile *profile,
    char *warning, size_t warning_size, char *error, size_t error_size) {
    if (error && error_size) *error = 0;
    if (warning && warning_size) *warning = 0;
    h3_bridge_profile p = {0};
    if (!profile || context_frames > 362 ||
        !h3_continuation_context(context_frames, &p.prefix))
        return fail(error, error_size, "bridge context must be 39 + 51*k frames within 39..362");
    if (video_steps < 1 || video_steps >= p.prefix.video_prefix_t)
        return fail(error, error_size, "bridge steps must be positive and shorter than the video context; retain at least one exact row");
    if (!isfinite(maximum) || maximum < 0 || maximum > 1)
        return fail(error, error_size, "bridge maximum strength must be finite and in [0,1]");
    if (!h3_bridge_profile_name(type))
        return fail(error, error_size, "unknown bridge profile; expected stepped, linear or ease-out");
    p.context_frames = context_frames;
    p.video_bridge_t = video_steps;
    p.video_exact_t = p.prefix.video_prefix_t - video_steps;
    p.bridge_frames = h3_video_time_boundary_frames(video_steps);
    /* Round the rational frame-time boundary to the nearest 40 Hz tick.
     * A tie goes toward the later tick. The complete prefix stays AV-exact. */
    p.audio_bridge_t = (p.bridge_frames * H3_AUDIO_LATENT_FPS + H3_FPS / 2) / H3_FPS;
    p.audio_exact_t = p.prefix.audio_prefix_t - p.audio_bridge_t;
    p.type = type;
    p.max_strength = maximum;
    p.class_mask[H3_ROW_GENERATED_VIDEO] = p.class_mask[H3_ROW_GENERATED_AUDIO] = 1;
    for (int i = 0; i < H3_BRIDGE_LEVELS; i++) {
        float mask = (float)((double)maximum * (i + 1) / (double)H3_BRIDGE_LEVELS);
        p.class_mask[H3_ROW_BRIDGE_VIDEO_FIRST + i] = mask;
        p.class_mask[H3_ROW_BRIDGE_AUDIO_FIRST + i] = mask;
    }
    p.active[H3_ROW_GENERATED_VIDEO] = p.active[H3_ROW_GENERATED_AUDIO] = 1;
    for (int t = 0; t < p.prefix.video_prefix_t; t++) {
        double time = (double)h3_video_time_boundary_frames(t) / p.bridge_frames;
        p.video_classes[t] = classify(&p, 0, time);
        p.active[p.video_classes[t]] = 1;
    }
    for (int t = 0; t < p.prefix.audio_prefix_t; t++) {
        double time = (double)t * H3_FPS / (H3_AUDIO_LATENT_FPS * p.bridge_frames);
        p.audio_classes[t] = t < p.audio_bridge_t ? classify(&p, 1, time) : H3_ROW_PRESERVED_AUDIO;
        p.active[p.audio_classes[t]] = 1;
    }
    for (int c = 0; c < H3_TARGET_ROW_CLASSES; c++) p.class_count += p.active[c];
    if (p.video_exact_t < 2 && warning && warning_size)
        snprintf(warning, warning_size, "bridge retains only one exact video row; at least two are recommended");
    *profile = p;
    return 1;
}

int h3_bridge_profile_valid(const h3_bridge_profile *p) {
    if (!p) return 0;
    h3_bridge_profile expected;
    if (!h3_bridge_profile_build(p->context_frames, p->video_bridge_t,
        p->max_strength, p->type, &expected, NULL, 0, NULL, 0)) return 0;
    /* Compare fields, not structure padding. Also reject corrupt class ids
     * before consumers use them to index a modulation table. */
    return p->prefix.video_prefix_t == expected.prefix.video_prefix_t &&
        p->prefix.audio_prefix_t == expected.prefix.audio_prefix_t &&
        p->video_exact_t == expected.video_exact_t && p->bridge_frames == expected.bridge_frames &&
        p->audio_bridge_t == expected.audio_bridge_t && p->audio_exact_t == expected.audio_exact_t &&
        p->class_count == expected.class_count &&
        !memcmp(p->active, expected.active, sizeof(p->active)) &&
        !memcmp(p->class_mask, expected.class_mask, sizeof(p->class_mask)) &&
        !memcmp(p->video_classes, expected.video_classes, sizeof(p->video_classes)) &&
        !memcmp(p->audio_classes, expected.audio_classes, sizeof(p->audio_classes));
}

h3_target_row_class h3_bridge_row_class(const h3_bridge_profile *p,
    int audio, size_t row, int latent_h, int latent_w, int audio_t) {
    if (!p || p->prefix.video_prefix_t < 0 || p->prefix.video_prefix_t > H3_BRIDGE_MAX_VIDEO_T ||
        p->prefix.audio_prefix_t < 0 || p->prefix.audio_prefix_t > H3_BRIDGE_MAX_AUDIO_T)
        return H3_TARGET_ROW_CLASSES;
    if (audio) {
        if (audio_t < p->prefix.audio_prefix_t || audio_t < 1 || row >= (size_t)audio_t * 2)
            return H3_TARGET_ROW_CLASSES;
        size_t t = row % (size_t)audio_t;
        return t < (size_t)p->prefix.audio_prefix_t ?
            (h3_target_row_class)p->audio_classes[t] : H3_ROW_GENERATED_AUDIO;
    }
    if (latent_h < 2 || latent_w < 2 || latent_h % 2 || latent_w % 2) return H3_TARGET_ROW_CLASSES;
    size_t spatial = (size_t)(latent_h / 2) * (size_t)(latent_w / 2);
    size_t t = row / spatial;
    return t < (size_t)p->prefix.video_prefix_t ?
        (h3_target_row_class)p->video_classes[t] : H3_ROW_GENERATED_VIDEO;
}

float h3_bridge_class_sigma(const h3_bridge_profile *p,
    h3_target_row_class kind, float sigma_v, float sigma_a) {
    if (!p || kind < 0 || kind >= H3_TARGET_ROW_CLASSES ||
        !isfinite(sigma_v) || sigma_v < 0 || sigma_v > 1 ||
        !isfinite(sigma_a) || sigma_a < 0 || sigma_a > 1) return NAN;
    int audio = kind == H3_ROW_GENERATED_AUDIO || kind == H3_ROW_PRESERVED_AUDIO ||
        kind >= H3_ROW_BRIDGE_AUDIO_FIRST;
    return p->class_mask[kind] * (audio ? sigma_a : sigma_v);
}

float h3_bridge_class_timestep(const h3_bridge_profile *p,
    h3_target_row_class kind, float sigma_v, float sigma_a) {
    float sigma = h3_bridge_class_sigma(p, kind, sigma_v, sigma_a);
    if (!isfinite(sigma)) return NAN;
    /* Preserve the existing exact-video condition floor literally. Applying
     * 0.999 to fractional rows would falsely label noisy bridge latents as
     * almost clean. Fractional rows use 1 - mask * stream_sigma. */
    if (kind < H3_ROW_BRIDGE_VIDEO_FIRST) return h3_prefix_timestep(kind, sigma_v, sigma_a);
    return 1 - sigma;
}

int h3_bridge_mask_velocity(const h3_bridge_profile *p, int video_t,
    int latent_h, int latent_w, int audio_t, float *video, float *audio) {
    if (!h3_bridge_profile_valid(p) || !video || !audio ||
        video_t < p->prefix.video_prefix_t || video_t > H3_BRIDGE_MAX_VIDEO_T ||
        audio_t < p->prefix.audio_prefix_t || audio_t > H3_BRIDGE_MAX_AUDIO_T ||
        latent_h < 1 || latent_w < 1 || (int64_t)latent_h * latent_w > H3_MAX_PIXELS / 256) return 0;
    size_t hw = (size_t)latent_h * (size_t)latent_w;
    for (size_t c = 0; c < 24; c++) for (int t = 0; t < p->prefix.video_prefix_t; t++) {
        float mask = p->class_mask[p->video_classes[t]];
        float *row = video + (c * (size_t)video_t + (size_t)t) * hw;
        if (mask == 0) memset(row, 0, hw * sizeof(*row));
        else if (mask != 1) for (size_t i = 0; i < hw; i++) row[i] *= mask;
    }
    for (size_t c = 0; c < 64; c++) for (int t = 0; t < p->prefix.audio_prefix_t; t++) {
        float mask = p->class_mask[p->audio_classes[t]];
        float *value = audio + c * (size_t)audio_t + (size_t)t;
        if (mask == 0) *value = 0;
        else if (mask != 1) *value *= mask;
    }
    return 1;
}

float h3_flow_mix(float clean, float noise, float sigma) {
    if (!isfinite(sigma) || sigma < 0 || sigma > 1) return NAN;
    if (sigma == 0) return clean;
    if (sigma == 1) return noise;
    volatile float signal = (1 - sigma) * clean;
    volatile float perturbation = sigma * noise;
    return signal + perturbation;
}

static int step_row(float *sample, const float *velocity, size_t count,
    float sigma, float next, int kind, h3_bridge_step_stats *stats) {
    if (!stats) return h3_euler_velocity_step(sample, velocity, count, sigma, next);
    float before[H3_MAX_PIXELS / 256];
    if (count > sizeof(before) / sizeof(*before)) return 0;
    memcpy(before, sample, count * sizeof(*before));
    if (!h3_euler_velocity_step(sample, velocity, count, sigma, next)) return 0;
    for (size_t i = 0; i < count; i++) {
        double delta = (double)sample[i] - before[i];
        stats->update_square[kind] += delta * delta;
        stats->changed[kind] += memcmp(sample+i, before+i, sizeof(*sample)) != 0;
    }
    return 1;
}

int h3_bridge_euler_step(const h3_bridge_profile *p, int video_t,
    int latent_h, int latent_w, int audio_t, float *video, float *audio,
    float *vv, float *av, float sv, float nv, float sa, float na,
    h3_bridge_step_stats *stats) {
    if (!h3_bridge_profile_valid(p) || !video || !audio || !vv || !av ||
        video_t < p->prefix.video_prefix_t || video_t > H3_BRIDGE_MAX_VIDEO_T ||
        audio_t < p->prefix.audio_prefix_t || audio_t > H3_BRIDGE_MAX_AUDIO_T ||
        latent_h < 1 || latent_w < 1 || (int64_t)latent_h * latent_w > H3_MAX_PIXELS / 256 ||
        !isfinite(sv) || !isfinite(nv) || !isfinite(sa) || !isfinite(na) ||
        !(sv > nv) || !(sa > na) || nv < 0 || na < 0 || sv > 1 || sa > 1) return 0;
    size_t hw = (size_t)latent_h * (size_t)latent_w;
    if (stats) {
        memset(stats, 0, sizeof(*stats));
        for (int stream = 0; stream < 2; stream++) {
            size_t channels = stream ? 64 : 24, length = (size_t)(stream ? audio_t : video_t);
            size_t spatial = stream ? 1 : hw;
            int prefix = stream ? p->prefix.audio_prefix_t : p->prefix.video_prefix_t;
            for (size_t c = 0; c < channels; c++) for (size_t t = 0; t < length; t++) {
                int kind = t < (size_t)prefix ? (stream ? p->audio_classes[t] : p->video_classes[t]) :
                    stream ? H3_ROW_GENERATED_AUDIO : H3_ROW_GENERATED_VIDEO;
                stats->elements[kind] += spatial;
                if (p->class_mask[kind] == 0) continue;
                const float *v = (stream ? av : vv) + (c * length + t) * spatial;
                for (size_t i = 0; i < spatial; i++) stats->raw_square[kind] += (double)v[i] * v[i];
            }
        }
    }
    if (!h3_bridge_mask_velocity(p, video_t, latent_h, latent_w, audio_t, vv, av)) return 0;
    for (int stream = 0; stream < 2; stream++) {
        size_t channels = stream ? 64 : 24, length = (size_t)(stream ? audio_t : video_t);
        size_t spatial = stream ? 1 : hw;
        int prefix = stream ? p->prefix.audio_prefix_t : p->prefix.video_prefix_t;
        for (size_t c = 0; c < channels; c++) for (size_t t = 0; t < length; t++) {
            int kind = t < (size_t)prefix ? (stream ? p->audio_classes[t] : p->video_classes[t]) :
                stream ? H3_ROW_GENERATED_AUDIO : H3_ROW_GENERATED_VIDEO;
            if (p->class_mask[kind] == 0) continue;
            size_t offset = (c * length + t) * spatial;
            float *sample = (stream ? audio : video) + offset;
            const float *velocity = (stream ? av : vv) + offset;
            if (stats) for (size_t i = 0; i < spatial; i++)
                stats->scaled_square[kind] += (double)velocity[i] * velocity[i];
            if (!step_row(sample, velocity, spatial, stream ? sa : sv, stream ? na : nv, kind, stats)) return 0;
        }
    }
    return 1;
}

int h3_bridge_check_exact(const h3_bridge_profile *p, int video_t,
    int latent_h, int latent_w, int audio_t, const float *video, const float *audio,
    const float *initial_video, const float *initial_audio) {
    if (!h3_bridge_profile_valid(p) || !video || !audio || !initial_video || !initial_audio ||
        video_t < p->prefix.video_prefix_t || audio_t < p->prefix.audio_prefix_t ||
        video_t > H3_BRIDGE_MAX_VIDEO_T || audio_t > H3_BRIDGE_MAX_AUDIO_T ||
        latent_h < 1 || latent_w < 1 || (int64_t)latent_h * latent_w > H3_MAX_PIXELS / 256) return 0;
    size_t hw = (size_t)latent_h * (size_t)latent_w;
    for (size_t c = 0; c < 24; c++) for (int t = 0; t < p->prefix.video_prefix_t; t++) {
        if (p->class_mask[p->video_classes[t]] != 0) continue;
        size_t offset = (c * (size_t)video_t + (size_t)t) * hw;
        if (memcmp(video+offset, initial_video+offset, hw * sizeof(*video))) return 0;
    }
    for (size_t c = 0; c < 64; c++) for (int t = 0; t < p->prefix.audio_prefix_t; t++) {
        if (p->class_mask[p->audio_classes[t]] != 0) continue;
        size_t offset = c * (size_t)audio_t + (size_t)t;
        if (memcmp(audio+offset, initial_audio+offset, sizeof(*audio))) return 0;
    }
    return 1;
}
