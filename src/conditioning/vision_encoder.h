#ifndef H3_VISION_ENCODER_H
#define H3_VISION_ENCODER_H

#include "src/gpu.h"

#include <stddef.h>
#include <stdint.h>

#ifdef __cplusplus
extern "C" {
#endif

#define H3_VISION_OUTPUT_WIDTH 5120u
#define H3_VISION_DEEPSTACKS 3u

typedef struct {
    int grid_h;
    int grid_w;
    size_t tokens;
    uint16_t *merged;
    uint16_t *deepstack[H3_VISION_DEEPSTACKS];
    h3_gpu_stats gpu_stats;
} h3_vision_output;

/* Progress returns 0 to continue, nonzero to cancel at this boundary. */
typedef int (*h3_vision_progress)(int completed_layers, int total_layers,
                                   void *opaque);

/* Encode one image or one two-frame video block. Pixels are F32 [T,3,H,W] in
 * [0,1], T is 1 or 2, and H/W are multiples of 32. The output owns BF16 Qwen
 * presentation rows and three same-shaped deepstack additions. */
int h3_vision_encode_bf16(const char *weight_directory,
                          const char *shader_source_path,
                          const float *pixels, int frames,
                          int height, int width,
                          h3_vision_progress progress, void *progress_opaque,
                          h3_vision_output *output,
                          char *error, size_t error_size);
void h3_vision_output_free(h3_vision_output *output);
typedef struct { const float *pixels; int frames,height,width; } h3_vision_input;
/* Conservative live-memory envelope: inputs/activations, merger/deepstacks,
 * packed projection scratch, one layer's weights and context workspaces.
 * Source pixels already belong to the caller's process footprint. A zero
 * budget means geometry/index validation only, not a promise of admission. */
typedef struct {
    uint32_t rows, groups[128];
    uint64_t host_bytes, device_bytes;
} h3_vision_plan;
int h3_vision_plan_create(const h3_vision_input *inputs, size_t count,
    uint64_t host_budget, uint64_t device_budget, h3_vision_plan *plan,
    char *error, size_t error_size);
/* Reference-only packed vision GEMMs, with independent attention per image or
 * video pair. The output array has count entries, each owning its tensors. */
int h3_vision_encode_batch_bf16(const char *weights,const char *shader,
    const h3_vision_input *inputs,size_t count,h3_vision_progress progress,void *opaque,
    h3_vision_output *outputs,char *error,size_t error_size);

#ifdef __cplusplus
}
#endif

#endif
