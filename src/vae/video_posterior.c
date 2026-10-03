#include "src/vae/video_posterior.h"
#include "src/media/refvideo.h"

#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Match PyTorch's separate F32 tensor operations, including rounding before
 * the add. Fusing mean + std*epsilon can cross an FP16 rounding boundary. */
#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif

int h3_video_chunk_plan_build(int frames, h3_video_chunk_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (frames < 5 || frames > 360 || !h3_ref2va_video_latent_t(frames)) return 0;
    plan->vae_frames = frames;
    plan->padding_frames = (17 - frames % 17) % 17;
    plan->padded_frames = frames + plan->padding_frames;
    plan->chunks = plan->padded_frames / 17;
    plan->raw_time = plan->chunks * 5;
    plan->time = plan->raw_time - 3;
    return plan->time == h3_ref2va_video_latent_t(frames);
}

float *h3_video_chunk_extract(const float *pixels, int source_frames,
    int height, int width, const h3_video_chunk_plan *plan, int chunk) {
    h3_video_chunk_plan expected;
    if (!pixels || !plan || !h3_video_chunk_plan_build(plan->vae_frames, &expected) ||
        memcmp(plan, &expected, sizeof(expected)) || source_frames < plan->vae_frames ||
        height < 1 || width < 1 || chunk < 0 || chunk >= plan->chunks) return NULL;
    size_t area = (size_t)height * (size_t)width;
    if (area > SIZE_MAX / 3 / sizeof(float) / (size_t)source_frames ||
        area > SIZE_MAX / 51 / sizeof(float)) return NULL;
    float *output = malloc(51 * area * sizeof(float));
    if (!output) return NULL;
    for (int c = 0; c < 3; c++) for (int t = 0; t < 17; t++) {
        int source = chunk * 17 + t;
        if (source >= plan->vae_frames) source = plan->vae_frames - 1;
        memcpy(output + ((size_t)c * 17 + t) * area,
            pixels + ((size_t)c * source_frames + source) * area, area * sizeof(float));
    }
    return output;
}

int h3_video_moments_join(const h3_video_moments *chunks, int count,
    int frames, h3_video_moments *output, char *error, size_t error_size) {
    if (output) memset(output, 0, sizeof(*output));
    h3_video_chunk_plan plan;
    const char *message = "invalid released temporal moment chunks";
    if (!output || !chunks || !h3_video_chunk_plan_build(frames, &plan) || count != plan.chunks)
        goto failed;
    int h = chunks[0].height, w = chunks[0].width;
    if (h < 1 || w < 1) goto failed;
    size_t area = (size_t)h * (size_t)w;
    if (area > SIZE_MAX / 48 / sizeof(float) / (size_t)plan.raw_time) goto failed;
    for (int i = 0; i < count; i++)
        if (!chunks[i].values || chunks[i].time != 5 || chunks[i].height != h || chunks[i].width != w)
            goto failed;
    size_t raw_span = (size_t)plan.raw_time * area, final_span = (size_t)plan.time * area;
    float *values = malloc(48 * raw_span * sizeof(float));
    if (!values) { message = "out of memory concatenating video moments"; goto failed; }
    for (size_t c = 0; c < 48; c++) for (int i = 0; i < count; i++)
        memcpy(values + c * raw_span + (size_t)i * 5 * area,
            chunks[i].values + c * 5 * area, 5 * area * sizeof(float));
    /* Compact each channel after concatenation; trailing capacity is unused. */
    for (size_t c = 0; c < 48; c++)
        memmove(values + c * final_span, values + c * raw_span, final_span * sizeof(float));
    output->values = values; output->time = plan.time; output->height = h; output->width = w;
    return 1;
failed:
    if (error && error_size) snprintf(error, error_size, "%s", message);
    return 0;
}

/* MT19937 and PyTorch's CPU contiguous-normal recipe. See the upstream source
 * references and applicable MT19937/PyTorch notices in THIRD_PARTY_NOTICES.md. */
typedef struct { uint32_t state[624]; size_t index; } posterior_rng;

static void seed42(posterior_rng *rng) {
    rng->state[0] = 42;
    for (uint32_t i = 1; i < 624; i++)
        rng->state[i] = UINT32_C(1812433253) * (rng->state[i-1] ^ (rng->state[i-1] >> 30)) + i;
    rng->index = 624;
}

static uint32_t random32(posterior_rng *rng) {
    if (rng->index == 624) {
        for (size_t i = 0; i < 624; i++) {
            uint32_t joined = (rng->state[i] & UINT32_C(0x80000000)) |
                (rng->state[(i+1)%624] & UINT32_C(0x7fffffff));
            rng->state[i] = rng->state[(i+397)%624] ^ (joined >> 1) ^
                ((joined & 1) ? UINT32_C(0x9908b0df) : 0);
        }
        rng->index = 0;
    }
    uint32_t value = rng->state[rng->index++];
    value ^= value >> 11;
    value ^= (value << 7) & UINT32_C(0x9d2c5680);
    value ^= (value << 15) & UINT32_C(0xefc60000);
    return value ^ (value >> 18);
}

static float uniform_float(posterior_rng *rng) {
    return (float)(random32(rng) & UINT32_C(0xffffff)) * 0x1p-24f;
}

static double uniform_double(posterior_rng *rng) {
    uint64_t value = (uint64_t)random32(rng) << 32;
    value |= random32(rng);
    return (double)(value & UINT64_C(0x1fffffffffffff)) * 0x1p-53;
}

static void normal16(float *data) {
    for (size_t j = 0; j < 8; j++) {
        float radius = sqrtf(-2.0f * logf(1.0f - data[j]));
        float theta = (float)(6.283185307179586476925286766559 * (double)data[j+8]);
        data[j] = radius * cosf(theta);
        data[j+8] = radius * sinf(theta);
    }
}

int h3_video_posterior_epsilon(float *values, size_t count) {
    if (!values || !count || count > SIZE_MAX / sizeof(float)) return 0;
    posterior_rng rng; seed42(&rng);
    if (count < 16) {
        for (size_t i = 0; i < count; i += 2) {
            double u1 = uniform_double(&rng), u2 = uniform_double(&rng);
            double radius = sqrt(-2.0 * log1p(-u2));
            double theta = 6.283185307179586476925286766559 * u1;
            values[i] = (float)(radius * cos(theta));
            if (i+1 < count) values[i+1] = (float)(radius * sin(theta));
        }
    } else {
        for (size_t i = 0; i < count; i++) values[i] = uniform_float(&rng);
        for (size_t i = 0; i <= count-16; i += 16) normal16(values+i);
        if (count % 16) {
            float *tail = values + count - 16;
            for (size_t i = 0; i < 16; i++) tail[i] = uniform_float(&rng);
            normal16(tail);
        }
    }
    return 1;
}

int h3_video_posterior_sample(const float *moments, const float *epsilon,
    size_t count, float *sample) {
    if (!moments || !epsilon || !sample || !count || count > SIZE_MAX / 2 / sizeof(float)) return 0;
    for (size_t i = 0; i < count; i++) {
        float logvar = moments[count+i];
        if (logvar < -30.0f) logvar = -30.0f;
        if (logvar > 20.0f) logvar = 20.0f;
        float std = expf(0.5f * logvar);
        float noise = std * epsilon[i];
        sample[i] = moments[i] + noise;
    }
    return 1;
}

uint16_t h3_video_f32_to_f16(float value) {
    uint32_t bits; memcpy(&bits, &value, sizeof(bits));
    uint32_t sign = (bits >> 16) & 0x8000, magnitude = bits & 0x7fffffff;
    if (magnitude >= 0x7f800000) return (uint16_t)(sign | (magnitude == 0x7f800000 ? 0x7c00 : 0x7e00));
    if (magnitude >= 0x477ff000) return (uint16_t)(sign | 0x7c00);
    if (magnitude < 0x33000000) return (uint16_t)sign;
    uint32_t exponent = magnitude >> 23;
    uint32_t mantissa = magnitude & 0x7fffff;
    if (exponent < 113) {
        mantissa |= 0x800000;
        unsigned shift = 126 - exponent;
        uint32_t rounded = mantissa >> shift;
        uint32_t rest = mantissa & ((UINT32_C(1) << shift) - 1);
        uint32_t half = UINT32_C(1) << (shift - 1);
        rounded += rest > half || (rest == half && (rounded & 1));
        return (uint16_t)(sign | rounded);
    }
    uint32_t rounded = magnitude + 0xfff + ((magnitude >> 13) & 1);
    return (uint16_t)(sign | ((rounded - 0x38000000) >> 13));
}

float h3_video_f16_to_f32(uint16_t value) {
    uint32_t sign = ((uint32_t)value & 0x8000) << 16;
    uint32_t exponent = ((uint32_t)value >> 10) & 31, mantissa = value & 1023, bits;
    if (!exponent) {
        if (!mantissa) bits = sign;
        else {
            int exp = -14;
            while (!(mantissa & 1024)) { mantissa <<= 1; exp--; }
            bits = sign | (uint32_t)(exp + 127) << 23 | (mantissa & 1023) << 13;
        }
    } else if (exponent == 31) bits = sign | 0x7f800000 | mantissa << 13;
    else bits = sign | (exponent + 112) << 23 | mantissa << 13;
    float result; memcpy(&result, &bits, sizeof(result)); return result;
}

int h3_video_posterior_normalize(const float *sample, size_t span,
    const float mean[24], const float deviation[24], float *normalized) {
    if (!sample || !mean || !deviation || !normalized || !span || span > SIZE_MAX / 24 / sizeof(float)) return 0;
    for (int c = 0; c < 24; c++) if (!isfinite(mean[c]) || !isfinite(deviation[c]) || deviation[c] <= 0) return 0;
    for (size_t c = 0; c < 24; c++) for (size_t i = 0; i < span; i++) {
        size_t index = c * span + i;
        float rounded = h3_video_f16_to_f32(h3_video_f32_to_f16(sample[index]));
        normalized[index] = (rounded - mean[c]) / deviation[c];
    }
    return 1;
}
