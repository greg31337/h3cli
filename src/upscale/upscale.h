#ifndef H3_UPSCALE_H
#define H3_UPSCALE_H
#include "src/h3.h"
#include "src/host.h"
#include "src/upscale/upscale_network.h"

/* Required, versioned section in clean-source/refinement containers. No
 * pointers: the state owns the raw arrays in its ordinary condition slots.
 * Semantic policy 1 freezes the source Qwen view and its token/position grid. */
typedef struct {
    uint32_t version, stage, semantic_policy, metadata_identity;
    int source_width, source_height;
    int keyframes[2];
    uint32_t keyframe_count;
    int original_width[12], original_height[12];
    int semantic_width[12], semantic_height[12];
    uint8_t audio_hash[32];
    uint32_t recipe,refinement_steps;
    float sigma;
    uint64_t seed;
    uint8_t parent_hash[32],artifact_hash[32],noise_hash[32],condition_hash[32];
} h3_upscale_record;

struct h3_upscale_source { h3_sampler_state *state; char *path; };
struct h3_upscale_plan {
    h3_upscale_plan_info info;
    h3_layout_ref references[12];
    size_t reference_count,video_condition_elements,audio_condition_elements;
    int keyframes[2];size_t keyframe_count;
    uint8_t source_identity[32];
};
typedef struct {
    float *video,*audio;
    size_t video_elements,audio_elements;
    h3_layout layout;
} h3_upscale_conditions;
void h3_upscale_conditions_free(h3_upscale_conditions *conditions);
/* model=NULL explicitly selects the bilinear comparison method; never a fallback. */
int h3_upscale_retarget(const h3_upscale_source *source,const h3_upscale_plan *plan,
    h3_upscale_model *model,h3_upscale_conditions *conditions,
    h3_progress_callback progress,void *opaque,char *error,size_t size);
float *h3_upscale_bilinear(const float *input,int time,int height,int width,
    int target_height,int target_width,char *error,size_t size);
int h3_upscale_schedule(int steps,float sigma,h3_sigma_schedule *schedule);
int h3_upscale_request_valid(const h3_upscale_options *options,char *error,size_t size);
int h3_upscale_audio_intact(const h3_sampler_state *state);
void h3_upscale_condition_hash(const h3_sampler_state *state,uint8_t hash[32]);
void h3_upscale_presentation(const h3_sampler_state *state,h3_presentation *presentation);
typedef struct {
    float *video;
    h3_upscale_conditions conditions;
    uint8_t parent_hash[32],artifact_hash[32];
    int recipe;
    double load_seconds,forward_seconds,retarget_seconds;
    /* Optional borrowed initialized checkpoint for a fixed comparison's shared
     * target noise. Ordinary fresh jobs leave this NULL and draw their noise. */
    const h3_sampler_state *noise_source;
} h3_upscale_transfer;
void h3_upscale_transfer_free(h3_upscale_transfer *transfer);
h3_upscale_transfer *h3_upscale_transfer_create(const h3_upscale_source *source,
    const h3_upscale_plan *plan,const char *model,int recipe,h3_progress_callback progress,
    void *opaque,char *error,size_t size);
h3_sampler_state *h3_upscale_initialize(const h3_upscale_source *source,const h3_upscale_plan *plan,
    const h3_upscale_transfer *transfer,const h3_upscale_options *options,const char *model_dir,
    const h3_device_info *device,char *error,size_t size);
/* Experiment runner may retain one transfer for several independent K jobs. */
h3_result *h3_upscale_execute(h3_ctx *ctx,const h3_upscale_source *source,const h3_upscale_plan *plan,
    const h3_upscale_transfer *transfer,const h3_upscale_options *options);
int h3_upscale_paths_alias(const char *a,const char *b);
int h3_upscale_options_valid(const h3_params *params, char *error, size_t size);
int h3_upscale_record_valid(const h3_sampler_state *s, char *error, size_t size);
int h3_upscale_capture(const h3_sampler_state *s, const float *raw_video,
    const float *raw_audio, const h3_upscale_record *record, const char *path,
    char *error, size_t size);
/* Shared bounded serializer; source mode requires its own magic and clean stage. */
h3_sampler_state *h3_upscale_file_load(const char *path, char *error, size_t size);
int h3_upscale_file_save(const h3_sampler_state *s, const char *path, char *error, size_t size);
#endif
