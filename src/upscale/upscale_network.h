#ifndef H3_UPSCALE_NETWORK_H
#define H3_UPSCALE_NETWORK_H
#include "src/h3.h"
#include "src/gpu.h"
typedef struct h3_upscale_model h3_upscale_model;
typedef int (*h3_upscale_trace)(const char *name,const float *ncthw,
    int channels,int time,int height,int width,void *opaque);
typedef struct {
    uint64_t weights_bytes,workspace_bytes,peak_gpu_bytes;
    double load_seconds,forward_seconds;
    uint64_t convolution_tiles,convolutions;
} h3_upscale_stats;
const char *h3_upscale_artifact_sha256(void);
h3_upscale_model *h3_upscale_model_load(const char *path,char *error,size_t size);
void h3_upscale_model_free(h3_upscale_model *model);
/* Caller owns the returned F32 NCTHW. No input buffer is mutated. Exact
 * temporal length; arbitrary spatial enlargement is private to retargeting. */
float *h3_upscale_volume(h3_upscale_model *model,const float *input,
    int time,int height,int width,int target_height,int target_width,
    h3_progress_callback progress,void *opaque,h3_upscale_trace trace,
    void *trace_opaque,char *error,size_t size);
int h3_upscale_model_stats(const h3_upscale_model *model,h3_upscale_stats *stats);
/* Bounded numeric qualification of the production layer functions. Inputs
 * are recorded BF16-valued NCTHW, with the fixed scale embedding supplied.
 * out_blocks.0 includes the spatial interpolation of in_blocks.17. */
float *h3_upscale_probe_layer(h3_upscale_model *model,const char *name,
    const float *input,const float embedding[64],int time,int height,int width,
    char *error,size_t size);
#endif
