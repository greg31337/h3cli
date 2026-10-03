#include "src/media/refvideo.h"
#include "src/media/ffmpeg.h"
#include "src/host.h"

#include <limits.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *h3_refvideo_pipeline_name(h3_refvideo_pipeline pipeline) {
    switch (pipeline) {
    case H3_REFVIDEO_NONE: return "none";
    case H3_REFVIDEO_RELEASED_V1: return "released-v1";
    default: return "unsupported";
    }
}

int h3_ref2va_video_vae_frames(int normalized_frames) {
    return normalized_frames < 5 ? 0 :
        normalized_frames - (normalized_frames - 5) % 17;
}

int h3_ref2va_video_latent_t(int vae_frames) {
    return vae_frames < 5 || (vae_frames - 5) % 17 ? 0 :
        h3_video_latent_t(vae_frames);
}

size_t h3_refvideo_qwen_blocks(int frames) {
    size_t samples = frames > 0 ? ((size_t)frames + 11) / 12 : 0;
    return (samples + 1) / 2;
}

int h3_refvideo_qwen_pair(int frames, size_t block, int *first, int *second,
    double *timestamp) {
    if (!first || !second || !timestamp || block >= h3_refvideo_qwen_blocks(frames))
        return 0;
    size_t samples = ((size_t)frames + 11) / 12;
    size_t a = 2 * block, b = a + 1 < samples ? a + 1 : a;
    *first = (int)(a * 12); *second = (int)(b * 12);
    *timestamp = ((double)a + (double)b) / 4.0;
    return 1;
}

int h3_refvideo_plan_build(int frames, h3_refvideo_plan *plan) {
    if (!plan) return 0;
    memset(plan, 0, sizeof(*plan));
    if (frames < 5 || frames > INT_MAX - 3) return 0;
    int64_t samples = ((int64_t)frames * 32000 + H3_FPS / 2) / H3_FPS;
    if (samples > INT_MAX) return 0;
    plan->frames = frames;
    plan->vae_frames = h3_ref2va_video_vae_frames(frames);
    plan->latent_t = h3_ref2va_video_latent_t(plan->vae_frames);
    plan->soundtrack_samples = (int)samples;
    if (plan->soundtrack_samples > 32000 * 15)
        plan->soundtrack_samples = 32000 * 15;
    plan->qwen_blocks = h3_refvideo_qwen_blocks(frames);
    plan->pipeline = H3_REFVIDEO_RELEASED_V1;
    return 1;
}

int h3_refvideo_validate_duration(int frames, int *total_frames,
    char *error, size_t error_size) {
    if (error && error_size) *error = 0;
    const char *message = NULL;
    if (!total_frames || frames < 5)
        message = "invalid reference-video duration arguments";
    else if (frames < 2 * H3_FPS || frames > 15 * H3_FPS)
        message = "released-v1 reference videos require 2–15 seconds at normalized 24 fps";
    else if (*total_frames < 0 || *total_frames > 15 * H3_FPS ||
             frames > 15 * H3_FPS - *total_frames)
        message = "released-v1 reference videos exceed 15 seconds in total";
    if (message) {
        if (error && error_size) snprintf(error, error_size, "%s", message);
        return 0;
    }
    *total_frames += frames;
    return 1;
}

static int read_video(const char *path, int width, int height, int max_frames,
    int *total_frames, float **pixels, h3_refvideo_plan *plan,
    char *error, size_t error_size,int reference) {
    if (pixels) *pixels = NULL;
    if (plan) memset(plan, 0, sizeof(*plan));
    if (!pixels || !plan || !total_frames) {
        if (error && error_size) snprintf(error, error_size, "invalid reference-video preparation arguments");
        return 0;
    }
    /* One frame beyond the released limit detects overlength clips without
     * allocating their entire duration. The request's shorter cap still wins. */
    int cap = max_frames > 15 * H3_FPS + 1 ? 15 * H3_FPS + 1 : max_frames;
    int frames = 0;
    int ok = h3_ffmpeg_read_normalized_video_f32(path, width, height, cap,
        pixels, &frames, error, error_size);
    if (!ok) return 0;
    if (!h3_refvideo_plan_build(frames, plan)) {
        if (error && error_size) snprintf(error, error_size, "invalid reference-video frame geometry");
        ok = 0;
    } else if(reference) {
        ok=frames>=5&&frames<=15*H3_FPS&&*total_frames>=0&&*total_frames<=15*H3_FPS-frames;
        if(ok)*total_frames+=frames;
        else if(error&&error_size)snprintf(error,error_size,"SGLang reference videos require at least five frames and at most 15 seconds total");
    } else ok = h3_refvideo_validate_duration(frames, total_frames, error, error_size);
    if (!ok) { free(*pixels); *pixels = NULL; memset(plan, 0, sizeof(*plan)); }
    return ok;
}

int h3_refvideo_read(const char *path,int width,int height,int max_frames,int *total_frames,
    float **pixels,h3_refvideo_plan *plan,char *error,size_t error_size) {
    return read_video(path,width,height,max_frames,total_frames,pixels,plan,error,error_size,0);
}
int h3_refvideo_read_sglang(const char *path,int width,int height,int max_frames,int *total_frames,
    float **pixels,h3_refvideo_plan *plan,char *error,size_t error_size) {
    return read_video(path,width,height,max_frames,total_frames,pixels,plan,error,error_size,1);
}

float *h3_refvideo_extract_pair(const float *pixels, int frames,
    int height, int width, int first, int second) {
    if (!pixels || frames < 1 || height < 1 || width < 1 || first < 0 ||
        second < 0 || first >= frames || second >= frames) return NULL;
    size_t area = (size_t)height * (size_t)width;
    if (area > SIZE_MAX / 6 || 6 * area > SIZE_MAX / sizeof(float)) return NULL;
    if ((size_t)frames > SIZE_MAX / 3 / area / sizeof(float)) return NULL;
    float *pair = malloc(6 * area * sizeof(*pair));
    if (!pair) return NULL;
    const int times[2] = {first, second};
    for (int time = 0; time < 2; time++)
        for (int channel = 0; channel < 3; channel++) {
            size_t source = ((size_t)channel * (size_t)frames + (size_t)times[time]) * area;
            size_t destination = ((size_t)time * 3 + (size_t)channel) * area;
            memcpy(pair + destination, pixels + source, area * sizeof(*pair));
        }
    return pair;
}
