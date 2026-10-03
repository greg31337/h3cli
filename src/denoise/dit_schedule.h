#ifndef H3_DIT_SCHEDULE_H
#define H3_DIT_SCHEDULE_H

#include "src/gpu.h"
#include "src/host.h"
#include "src/sampling/bridge.h"
#include "src/weights/weights.h"

#include <stddef.h>
#include <stdint.h>

#define H3_DIT_BLOCKS 50u
#define H3_DIT_HIDDEN 5376u
#define H3_DIT_TIME_DIM 2688u
#define H3_DIT_MODALITIES 3u
#define H3_DIT_ADALN_SLOTS 6u

typedef struct h3_dit_schedule h3_dit_schedule;
struct h3_sampler_state;
int h3_dit_schedule_export(const h3_dit_schedule *schedule,
    struct h3_sampler_state *state);
h3_dit_schedule *h3_dit_schedule_import(h3_gpu *gpu,
    const struct h3_sampler_state *state);

/* Progress returns 0 to continue, nonzero to cancel at this boundary. */
typedef int (*h3_dit_schedule_progress)(int completed_blocks,
                                         int total_blocks, void *opaque);

/* Materialize every per-step AdaLN value. This intentionally submits one
 * projection at a time, so a 498 MiB block projection is released before the
 * next is loaded. */
h3_dit_schedule *h3_dit_schedule_precompute(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition,
    int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
h3_dit_schedule *h3_dit_schedule_precompute_bridge(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, const h3_bridge_profile *profile,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
/* Released video references: activate time embeddings in F32, then cast for
 * AdaLN. Other routes retain historical BF16-before-SiLU operation ordering. */
h3_dit_schedule *h3_dit_schedule_precompute_refvideo(
    const h3_weight_store *weights, h3_gpu *gpu,
    const h3_sigma_schedule *sigmas, int visual_condition, int audio_condition,
    h3_dit_schedule_progress progress, void *progress_opaque,
    char *error, size_t error_size);
void h3_dit_schedule_free(h3_dit_schedule *schedule);
/* Host-only schedule planning, also used to verify row semantics without
 * loading weights. It has no block/final tensors. */
h3_dit_schedule *h3_dit_schedule_plan(const h3_sigma_schedule *sigmas,
    int visual_condition, int audio_condition, char *error, size_t error_size);
/* Host-only bridge planning; precompute_bridge materializes these rows. */
h3_dit_schedule *h3_dit_schedule_plan_bridge(const h3_sigma_schedule *sigmas,
    const h3_bridge_profile *profile, char *error, size_t error_size);
uint32_t h3_dit_schedule_class_row(const h3_dit_schedule *schedule, int step,
    h3_target_row_class kind);
float h3_dit_schedule_timestep(const h3_dit_schedule *schedule, uint32_t row);

int h3_dit_schedule_steps(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_time_rows(const h3_dit_schedule *schedule);
uint32_t h3_dit_schedule_video_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_row(const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_visual_condition_row(
    const h3_dit_schedule *schedule, int step);
uint32_t h3_dit_schedule_audio_condition_row(
    const h3_dit_schedule *schedule, int step);
const h3_gpu_tensor *h3_dit_schedule_block(const h3_dit_schedule *schedule,
                                           unsigned block);
double h3_dit_schedule_gate_score(const h3_dit_schedule *schedule,
                                  unsigned block);
void h3_dit_schedule_prune(h3_dit_schedule *schedule,
                           const uint8_t *active_blocks, size_t count);
const h3_gpu_tensor *h3_dit_schedule_final(const h3_dit_schedule *schedule);

/* Build the row map consumed by the fused AdaLN/gate kernels. text_tags may be
 * NULL (all tag 1), or one tag per text row. Qwen vision presentation spans use
 * tag 0. Segment kinds select target/condition timesteps and modality tags. */
int h3_dit_schedule_row_map(const h3_dit_schedule *schedule, int step,
                            const h3_layout *layout,
                            const uint8_t *text_tags, size_t text_tag_count,
                            uint32_t *rows, size_t row_count);

#endif
