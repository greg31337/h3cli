#ifndef H3_SGLANG_H
#define H3_SGLANG_H
#include "src/host.h"
#include "src/h3.h"
#ifdef __cplusplus
extern "C" {
#endif
/* Shared SGLang arithmetic identity. Zero is a Metal/still or component scope. */
/* v4 matches reference soundtrack input decoding and the FP32 audio encoder.
 * Reject exploratory v1-v3 conditioning/sampler reuse: even sub-ULP encoder
 * drift can cross BF16 ties in the subsequent FP32 patch projection. */
#define H3_SGLANG_VERSION 4
int h3_sglang_exchange(int enabled);
/* Shared base predicate: true for CUDA video arithmetic. */
int h3_sglang_requested(void);
int h3_sglang_exact_requested(void);
/* Derive from the backend and operation before constructing caches/models. */
int h3_sglang_resolve(const h3_params *params,const char *backend,int lora_active);
/* Pinned Ref2VA policy: upscale to short edge 2048, nearest-even 32px grid. */
int h3_sglang_reference_image_canvas(int width,int height,int *out_width,int *out_height);
int h3_sglang_resize_rgb(const uint8_t *in,int iw,int ih,int ow,int oh,uint8_t **out);
int h3_sglang_read_reference_image(const char *path,int iw,int ih,int ow,int oh,float **out,char *error,size_t size);
int h3_sglang_read_soundtrack(const char *path,int max_frames,float **pcm,int *samples,char *error,size_t size);
int h3_sglang_schedule(int evaluations, h3_sigma_schedule *schedule);
int h3_sglang_euler(float *state, const float *velocity, size_t count,
                    float sigma, float next);
/* Matches the pinned PyTorch CPU contiguous FP32 normal stream, n >= 16.
 * No random draws occur after initialization for the eta=0 sampler. */
int h3_sglang_normal(uint64_t seed, float *out, size_t count);
/* Packed visual conditions restart the pinned CPU RNG per condition. */
int h3_sglang_video_condition(float *rows,int time,int height,int width,
                              int target_time,int visual_conditions,uint64_t seed);
/* Audio reference timestep is 1.0: validate and retain clean rows, no draw. */
int h3_sglang_audio_condition(float *rows,size_t elements,uint64_t seed);
/* Owned FFmpeg byte-derived pixels only. VAE uses GPU reciprocal scaling;
 * Qwen uses CPU division. Restore exactly before reusing the pixels for Qwen.
 * This adapter is not for arbitrary floating-point image API inputs. */
void h3_sglang_vae_byte_pixels(float *pixels,size_t count,int restore);
/* Test-only lossless snapshots; absent environment means no allocation or I/O. */
int h3_sglang_dump(const char *name,const void *data,size_t bytes);
/* -1 is an invalid test capture selection; 0/1 are unselected/selected. */
int h3_sglang_capture_step(int step,int total);
int h3_sglang_import_text(const uint32_t *tokens,size_t rows,uint16_t *text,size_t width);
int h3_sglang_import_conditions(float *video,size_t video_elements,float *audio,size_t audio_elements);
#ifdef __cplusplus
}
#endif
#endif
