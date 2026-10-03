#ifndef H3_CONDITIONING_H
#define H3_CONDITIONING_H
#include "src/sampling/sampler_state.h"
#define H3_CONDITIONING_VERSION 1
/* Raw reference rows precede seeded augmentation. Only prepared AdaLN records
 * depend on the schedule. Runtime objects, generated latents and noise are absent. */
typedef struct {
    char *identity;
    h3_text_embedding text;
    float *video, *audio;
    size_t video_elements, audio_elements;
    h3_layout_ref *references;
    size_t reference_count;
    int conditioned, keyframes[2];
    size_t keyframe_count;
    h3_prepared_cache prepared;
    uint8_t schedule_key[32];
    int has_schedule;
} h3_conditioning;
/* Nonzero only for image policies whose fresh-cache identity needs the shared
 * geometry version. Prepared sampler state keeps its recorded geometry. */
int h3_conditioning_image_geometry(const h3_params *params);
char *h3_conditioning_identity(const char *model_dir,const char *transformer,
    const char *prompt,const h3_params *params,const uint8_t av_signature[32],const h3_device_info *device,
    char *error,size_t size);
int h3_conditioning_identity_matches(const char *stored,const char *requested,
    char *error,size_t size);
int h3_conditioning_request_matches(const h3_conditioning *state,const h3_params *params,
    char *error,size_t size);
void h3_conditioning_schedule_key(const h3_sigma_schedule *sigmas,
    const h3_layout *layout,int released_refvideo,uint8_t key[32]);
int h3_conditioning_save(const h3_conditioning *state,const char *path,
    char *error,size_t size);
h3_conditioning *h3_conditioning_load(const char *path,char *error,size_t size);
void h3_conditioning_free(h3_conditioning *state);
/* Scoped only around DiT construction; owns nothing, supports nested calls. */
const h3_conditioning *h3_conditioning_current(void);
const h3_conditioning *h3_conditioning_exchange(const h3_conditioning *state);
#endif
