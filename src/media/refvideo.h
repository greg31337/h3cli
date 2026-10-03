#ifndef H3_REFVIDEO_H
#define H3_REFVIDEO_H

#include <stddef.h>

/* Persisted values: never reuse a number for a different conditioning recipe. */
typedef enum {
    H3_REFVIDEO_NONE = 0,
    H3_REFVIDEO_RELEASED_V1 = 2
} h3_refvideo_pipeline;

typedef struct {
    int frames;
    int vae_frames;
    int latent_t;
    int soundtrack_samples;
    size_t qwen_blocks;
    h3_refvideo_pipeline pipeline;
} h3_refvideo_plan;

const char *h3_refvideo_pipeline_name(h3_refvideo_pipeline pipeline);
int h3_ref2va_video_vae_frames(int normalized_frames);
int h3_ref2va_video_latent_t(int vae_frames);
int h3_refvideo_plan_build(int frames, h3_refvideo_plan *plan);
/* Duration accounting uses normalized frames, before VAE snap-down. */
int h3_refvideo_validate_duration(int frames, int *total_frames,
    char *error, size_t error_size);
/* width/height are the resolved reference canvas. Owns no returned memory. */
int h3_refvideo_read(const char *path, int width, int height, int max_frames,
    int *total_frames, float **pixels, h3_refvideo_plan *plan,
    char *error, size_t error_size);
/* Pinned SGLang accepts short clips; preserve the existing released/fast
 * duration contract in h3_refvideo_read. Both remain bounded at 15 seconds. */
int h3_refvideo_read_sglang(const char *path,int width,int height,int max_frames,
    int *total_frames,float **pixels,h3_refvideo_plan *plan,char *error,size_t error_size);
size_t h3_refvideo_qwen_blocks(int frames);
int h3_refvideo_qwen_pair(int frames, size_t block, int *first, int *second,
    double *timestamp);
/* Borrow channel-major [3,T,H,W], return owned time-major [2,3,H,W]. */
float *h3_refvideo_extract_pair(const float *pixels, int frames,
    int height, int width, int first, int second);

#endif
