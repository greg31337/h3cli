#ifndef H3_GPU_H
#define H3_GPU_H

#include <stddef.h>
#include <stdint.h>
#include "src/weights/residency.h"
#define H3_VISION_PATCH_CHUNK_ROWS 32768u /* 512 MiB of padded BF16 input. */
#ifdef __cplusplus
extern "C" {
#endif

typedef struct h3_gpu h3_gpu;
typedef struct h3_gpu_tensor h3_gpu_tensor;
typedef struct h3_gpu_event h3_gpu_event;

typedef enum {
    H3_GPU_F32 = 0,
    H3_GPU_BF16,
    H3_GPU_I8,
    H3_GPU_U32,
    H3_GPU_F16
} h3_gpu_dtype;

typedef enum { H3_GPU_HOST_VISIBLE, H3_GPU_DEVICE_ONLY, H3_GPU_HOST_PINNED }
    h3_gpu_storage;
h3_gpu_tensor *h3_gpu_tensor_alloc(h3_gpu *gpu, size_t elements,
                                   h3_gpu_dtype dtype, h3_gpu_storage storage);
/* Synchronizes host-visible storage; device-only tensors return NULL. */
void *h3_gpu_tensor_contents(h3_gpu_tensor *tensor);
h3_gpu_event *h3_gpu_event_new(h3_gpu *gpu);
int h3_gpu_event_record(h3_gpu *gpu, h3_gpu_event *event);
int h3_gpu_event_wait(h3_gpu *gpu, h3_gpu_event *event);
void h3_gpu_event_free(h3_gpu_event *event);
/* 0 resident, 1 streamed, -1 invalid/insufficient device memory. Metal keeps
 * its existing residency policy. CUDA includes workspace and safety reserve. */
int h3_gpu_plan_weights(h3_gpu *gpu, uint64_t weights, uint64_t activations);
#ifndef __APPLE__
int h3_gpu_plan_bf16_weights(h3_gpu *gpu,uint64_t active_mask,uint64_t block_bytes,
    uint64_t future_bytes,int force_stream,int retry_cap,h3_weight_plan *plan);
int h3_gpu_weight_plan_compatible(h3_gpu *gpu,int force_stream);
#endif
/* Quantized descriptors are only loaded explicitly by repeated DiT blocks.
 * Their BF16 projection interfaces dispatch native narrow GEMMs. */
int h3_gpu_quant_configure(h3_gpu *gpu,int mode,const char *cache,
                           uint64_t bf16_weights,uint64_t activations,int force_stream);
h3_gpu_tensor *h3_gpu_quant_load(h3_gpu *gpu,const char *path,uint64_t offset,
                                 uint32_t rows,uint32_t columns);
/* Release optional copies of streamed weights before another component loads.
 * Preserves tensors, prepared model state and future cache admission. */
int h3_gpu_release_weight_cache(h3_gpu *gpu);
/* Only main DiT blocks use this explicit domain. Other SDPA APIs are unchanged. */
int h3_gpu_dit_attention_configure(h3_gpu *gpu,int mode);
int h3_gpu_dit_sdpa_bf16(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t seq,uint32_t heads,
    uint32_t dim,float scale,int head_major,unsigned block,int step);
#ifndef __APPLE__
#include "src/denoise/sol.h"
#include "src/denoise/adaptive_cache.h"
int h3_gpu_adaptive_continuation_probe(h3_gpu *gpu,h3_gpu_tensor *probe,
    const h3_gpu_tensor *output,const h3_gpu_tensor *anchor,h3_gpu_tensor *scratch,
    unsigned columns,const h3_adaptive_regions *regions,int ready,float scores[H3_ADAPTIVE_REGIONS]);
int h3_gpu_adaptive_probe(h3_gpu *gpu,h3_gpu_tensor *probe,
    const h3_gpu_tensor *output,const h3_gpu_tensor *anchor,h3_gpu_tensor *scratch,
    unsigned rows,unsigned columns,unsigned video_start,unsigned video_rows,
    unsigned audio_start,unsigned audio_rows,int ready,float scores[3]);
int h3_gpu_dit_sol_layout(h3_gpu *gpu,const h3_sol_layout *layout);
int h3_gpu_dit_attention_noise(h3_gpu *gpu,int step,float video_sigma,float audio_sigma);
#endif
#ifdef __APPLE__
#include "src/backend.h"
#include "src/denoise/sol.h"
/* Load original BF16 weights into bounded, grouped-Q8 resident storage.
 * Linear BF16 dispatch recognizes this descriptor; only DiT opts into it. */
h3_gpu_tensor *h3_gpu_q8_load(h3_gpu *gpu,const char *path,uint64_t offset,
                              uint32_t rows,uint32_t columns);
/* Device-wide Metal reservations, including between component contexts. */
uint64_t h3_gpu_device_allocated_bytes(void);
/* M1 microbenchmark primitive. Layout 0 is [sequence,head,128], 1 is
 * [head,sequence,128]. Both layouts are read/written directly, without copies.
 * tile is 0..3, or -1 for the production selection. Requires an active command. */
int h3_gpu_native_sdpa_bf16(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t seq,uint32_t heads,uint32_t dim,float scale,
    int input_head_major,int output_head_major,int tile);
int h3_gpu_native_sdpa_candidate_b(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t seq,uint32_t heads,float scale,int input_head_major,int output_head_major,int query_block);
/* Binds the absolute sampler step and both schedule sigmas before DiT attention. */
int h3_gpu_native_noise(h3_gpu *gpu,int step,float video_sigma,float audio_sigma);
int h3_gpu_native_attention_layout(h3_gpu *gpu,const h3_sol_layout *layout);
int h3_gpu_native_sol_bf16(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t seq,uint32_t heads,float scale,int input_head_major,int output_head_major,
    h3_metal_attention_options options,unsigned block,int step);
/* Call at a completed step boundary; reports per-layer router counters. */
void h3_gpu_native_sol_report(h3_gpu *gpu,int step);
#endif

typedef struct {
    uint64_t q8_weights, q8_source_bytes, q8_storage_bytes, q8_dispatches;
    double q8_load_seconds;
    uint64_t layout_prepares, layout_inplace_packs, layout_scan_replaced_bytes, layout_partial_bytes;
    uint64_t allocated_bytes;
    uint64_t live_bytes;
    uint64_t peak_live_bytes;
    uint64_t tensor_allocations;
    uint64_t direct_dispatches;
    uint64_t mps_linear_dispatches;
    uint64_t mps_conv_dispatches;
    uint64_t mps_sdpa_dispatches;
    uint64_t native_attention_dispatches;
    uint64_t metal_current_allocated_bytes;
    uint64_t blit_copies;
    uint64_t submissions;
    double command_encode_seconds;
    double command_wait_seconds;
    /* Root MTLCommandBuffer timestamps; MPSGraph may schedule child buffers,
     * so command_wait_seconds is the complete turnaround measurement. */
    double gpu_seconds;
    uint64_t linear_dispatches, conv_dispatches, attention_dispatches;
    uint64_t h2d_bytes, d2h_bytes, d2d_bytes, streamed_bytes;
    uint64_t resident_weight_bytes, stream_slot_bytes, resident_block_mask;
    unsigned resident_blocks, active_weight_blocks;
    uint64_t device_bytes, managed_bytes, pinned_bytes, peak_pinned_bytes;
    uint64_t stream_stalls;
    double transfer_seconds, stream_wait_seconds, linear_seconds, attention_seconds;
    double conv_seconds;
    double h2d_seconds, d2h_seconds, slot_wait_seconds, normalization_seconds;
    /* Optional CUDA streamed-weight cache. Included in live/device bytes. */
    uint64_t weight_cache_bytes, weight_cache_hits, weight_cache_hit_bytes;
    uint64_t weight_cache_misses, weight_cache_evictions;
    double quant_conversion_seconds;
    uint64_t quant_clipped_values,quant_zeroed_values;
    uint64_t quant_projection_dispatches,quant_cache_hits,quant_cache_misses;
    uint64_t sage2_attention_dispatches,sage3_attention_dispatches,attention_workspace_bytes;
    uint64_t inflight_peak,command_retirements,bounded_waits,pressure_drains;
    uint64_t native_scratch_bytes,native_scratch_allocations,native_scratch_reuses,native_pipeline_count;
    uint64_t main_dense_calls,subblock_calls,subblock_bypass,subblock_protected_calls;
    uint64_t subblock_selected,subblock_possible,subblock_router_calls;
    uint64_t attention_timing_samples,attention_timing_missed;
    double main_attention_seconds,subblock_router_seconds,subblock_kernel_seconds;
} h3_gpu_stats;

h3_gpu *h3_gpu_create(const char *shader_source_path,
                     char *error, size_t error_size);
#ifndef __APPLE__
/* Constructor-time component distinction for the CUDA recipe. */
void h3_gpu_sglang_text_encoder(h3_gpu *gpu);
void h3_gpu_sglang_audio_decoder(h3_gpu *gpu);
void h3_gpu_sglang_audio_encoder(h3_gpu *gpu);
void h3_gpu_sglang_video_encoder(h3_gpu *gpu);
int h3_gpu_sglang_audio_average(h3_gpu *gpu,h3_gpu_tensor *sum,
    const h3_gpu_tensor *last,uint32_t count);
/* Optional, process-bounded pinned host staging for streamed DiT weights.
 * Admission failure falls back to existing bounded file streaming. */
int h3_gpu_sglang_preload_weight(h3_gpu *gpu,const char *path,uint64_t offset,size_t elements);
int h3_gpu_sglang_vae_enabled(const h3_gpu *gpu);
int h3_gpu_sglang_posterior(h3_gpu *gpu,const float *moments,const float *epsilon,
                            float *sample,size_t count);
int h3_gpu_sglang_vision_prepare(h3_gpu *gpu,h3_gpu_tensor *position,
    const h3_gpu_tensor *table,h3_gpu_tensor *cosine,h3_gpu_tensor *sine,uint32_t height,uint32_t width);
int h3_gpu_sglang_vision_patch(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *weight,const h3_gpu_tensor *bias,uint32_t rows,uint32_t k,uint32_t n);
h3_gpu_tensor *h3_gpu_sglang_vae_weight(h3_gpu *gpu,const h3_gpu_tensor *weight);
int h3_gpu_sglang_vae_rope(h3_gpu *gpu,h3_gpu_tensor *cosine,h3_gpu_tensor *sine,
    uint32_t time,uint32_t height,uint32_t width,uint32_t rows);
/* Preserve the oracle's per-evaluation GEMM batch sizes while precomputing
 * a schedule. An empty list restores ordinary reference GEMM dispatch. */
int h3_gpu_sglang_batch_groups(h3_gpu *gpu,const uint32_t *counts,size_t count);
int h3_gpu_sglang_vision_groups(h3_gpu *gpu,const uint32_t *counts,size_t count);
int h3_gpu_sglang_vision_prepare_range(h3_gpu *gpu,h3_gpu_tensor *position,
    const h3_gpu_tensor *table,h3_gpu_tensor *cosine,h3_gpu_tensor *sine,
    uint32_t height,uint32_t width,uint32_t offset);
int h3_gpu_sglang_time_features(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *times,uint32_t rows);
int h3_gpu_sglang_rope_trig(h3_gpu *gpu,h3_gpu_tensor *cosine,h3_gpu_tensor *sine,
    const h3_gpu_tensor *angles,uint32_t count);
int h3_gpu_sglang_head_f32(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *weight,const h3_gpu_tensor *bias,uint32_t rows,uint32_t k,uint32_t n,uint32_t packed_rows);
#endif
void h3_gpu_free(h3_gpu *gpu);
/* Drop uncommitted commands and wait for already committed work before cleanup. */
void h3_gpu_cancel(h3_gpu *gpu);
typedef struct {
    size_t device_bytes, static_bytes, alignment_bytes;
    size_t max_sequence;
    size_t direct_max_sequence; /* Metal full-row scratch boundary; tiled above. */
} h3_gqa_limits;
/* Qwen only. Q/K/V storage stays BF16; reference scales the FP32 QK score. */
typedef enum {
    H3_QWEN_GQA_REFERENCE,
    H3_QWEN_GQA_SCALED_Q,
    H3_QWEN_GQA_LEGACY
} h3_qwen_gqa_scale_mode;
int h3_qwen_gqa_scale_parse(const char *value, h3_qwen_gqa_scale_mode *mode,
                            char *error, size_t error_size);
const char *h3_qwen_gqa_scale_name(h3_qwen_gqa_scale_mode mode);
int h3_gpu_gqa_causal_limits(h3_gpu *gpu, h3_gqa_limits *limits);
size_t h3_gpu_gqa_causal_max_sequence(h3_gpu *gpu);
int h3_gpu_gqa_causal_preflight(h3_gpu *gpu, size_t sequence,
                               char *error, size_t error_size);
int h3_gpu_is_m5(const h3_gpu *gpu);
/* Backend defaults and recoverable device-allocation pressure. */
const char *h3_gpu_backend_name(const h3_gpu *gpu);
int h3_gpu_prefers_device_sampler(const h3_gpu *gpu, int continuation);
int h3_gpu_should_retry_streaming(const h3_gpu *gpu);
int h3_gpu_has_nax_mlp(const h3_gpu *gpu);
int h3_gpu_has_int8_mlp(const h3_gpu *gpu);

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements);
h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements);
/* Allocate shared Metal storage and pread BF16 payload directly into it. */
/* Storage diagnostic; logical tensor bytes are unchanged by mapping padding. */
int h3_gpu_tensor_is_file_mapped(const h3_gpu_tensor *tensor);
h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *gpu, const char *path,
                                       uint64_t file_offset, size_t elements);
h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *gpu, const char *path,
                                      uint64_t file_offset, size_t elements);
/* Fill an existing shared BF16 buffer from a file. The tensor and its
 * accounting are unchanged, so this may run on an I/O thread while another
 * tensor is in flight on the GPU. */
int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size);
/* As above, but ask Darwin to avoid retaining a second copy in the file cache.
 * Intended for large sequential weight streams whose destination is the only
 * useful resident copy. */
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *tensor, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size);
void h3_gpu_tensor_free(h3_gpu_tensor *tensor);
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor);
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor);
int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements);
int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements);
int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements);
int h3_gpu_tensor_read_u32(const h3_gpu_tensor *tensor, uint32_t *values,
                           size_t elements);
int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements);
int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements);
int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements);
int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements);

int h3_gpu_begin(h3_gpu *gpu);
/* Commit the current command buffer without waiting, then continue encoding on
 * the same ordered queue. h3_gpu_submit() waits and validates the whole chain. */
int h3_gpu_continue(h3_gpu *gpu);
int h3_gpu_submit(h3_gpu *gpu);
/* Completed-boundary export barrier; safe when no command is active. */
int h3_gpu_synchronize(h3_gpu *gpu);
const char *h3_gpu_error(const h3_gpu *gpu);
int h3_gpu_get_stats(const h3_gpu *gpu, h3_gpu_stats *stats);
/* Optional benchmark labels. With H3_PROFILE set, marks and context teardown
 * print wall time alongside command-buffer GPU time and allocation counters. */
void h3_gpu_profile_set_label(h3_gpu *gpu, const char *label);
void h3_gpu_profile_mark(h3_gpu *gpu, const char *phase);

/* Full video decoder only; never configure a DiT, audio or image context.
 * CUDA uses the SGLang FP16/FP32 path; Metal retains its FP32 path. */
/* Reserved zero argument preserves the frozen parity test ABI; no mode selection. */
int h3_gpu_video_vae_configure(h3_gpu *gpu, int reserved);
/* Begin after h3_gpu_begin: 0 ordinary, 1 capturing, 2 replayed, -1 error.
 * End returns 0 for a safe ordinary retry, 1 success, -1 error. */
int h3_gpu_video_graph_begin(h3_gpu *gpu);
int h3_gpu_video_graph_end(h3_gpu *gpu);
int h3_gpu_video_graph_abort(h3_gpu *gpu);
int h3_gpu_video_linear(h3_gpu *gpu, h3_gpu_tensor *output,
    const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
    const h3_gpu_tensor *bias, uint32_t rows, uint32_t k, uint32_t n);
int h3_gpu_video_sdpa(h3_gpu *gpu, h3_gpu_tensor *output,
    const h3_gpu_tensor *q, const h3_gpu_tensor *k, const h3_gpu_tensor *v,
    uint32_t rows, uint32_t heads, uint32_t dim, float scale);
/* Returns 0 on unsupported backend, -1 on failure, 1 after writing RGB. */
int h3_gpu_video_unpack(h3_gpu *gpu, const h3_gpu_tensor *projected,
    float *rgb, int latent_h, int latent_w, int output_frames,
    int first_frame, int frames);
#ifndef __APPLE__
/* Reference-only bounded spatial stitching: 0 unsupported, -1 error, 1 ready.
 * Retains one temporal chunk, with no CPU readback between spatial tiles. */
int h3_gpu_sglang_stitch_begin(h3_gpu *gpu, int tile_h, int tile_w,
    int count_y, int count_x, const int *starts_y, const int *starts_x,
    int first_frame, int frames);
int h3_gpu_sglang_stitch_tile(h3_gpu *gpu, const h3_gpu_tensor *projected, int tile);
int h3_gpu_sglang_stitch_read(h3_gpu *gpu, float *rgb);
#endif

int h3_gpu_linear_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim);
int h3_gpu_silu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_cast_f32_to_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_cast_bf16_to_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_copy_bf16(h3_gpu *gpu, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements);
int h3_gpu_copy_f32(h3_gpu *gpu, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements);
int h3_gpu_rms_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon);
int h3_gpu_adaln_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_gate_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon);
int h3_gpu_sdpa_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_swiglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width);
int h3_gpu_scale_add_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width);
int h3_gpu_layer_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon);
int h3_gpu_video_qkv_rope_f32(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);
int h3_gpu_video_qkv(h3_gpu *gpu, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon);

/* H3 AudioVAE uses time-major [batch,length,channels] activations and stores
 * Conv1d/ConvTranspose1d weights in PyTorch OIK/IOK order respectively. */
int h3_gpu_conv1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation);
int h3_gpu_conv1d_stride_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation);
int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding);
int h3_gpu_weight_norm_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner);
int h3_gpu_add_scaled_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements);
int h3_gpu_alias_free_snake_f32(
                          h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels);
int h3_gpu_snake1d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels);
int h3_gpu_audio_qkv_split_f32(h3_gpu *gpu,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim);
int h3_gpu_sdpa_causal_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query,
                       const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence, uint32_t heads,
                       uint32_t head_dim, float scale);
int h3_gpu_audio_attention_pool_f32(h3_gpu *gpu,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim);
int h3_gpu_geglu_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements);
int h3_gpu_clip_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum);

/* Visual-VAE encoder tensors use channels-last [B,T,H,W,C] storage. Spatial
 * padding reflects pixels while temporal front padding is zero-filled. */
int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after);
int h3_gpu_conv3d_f32(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width);
int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon);

/* Portable BF16 storage path. Arithmetic accumulates in F32 and rounds at
 * operation boundaries, matching the released checkpoint's compute dtype. */
int h3_gpu_linear_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim);
int h3_gpu_mlp_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim);
/* M5 Metal 4 paired FC1/SwiGLU plus direct FC2 path. Available
 * only when the context was created with H3_NAX=mlp. */
int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim);
/* M5 Metal 4 int8 MLP. Weights use one F32 scale per output
 * channel; activations are quantized dynamically with one F32 scale per row. */
int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns);
int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales);
/* Consume SDPA's native [head,row,dimension] BF16 layout without a full
 * BF16 transpose, gathering directly into the projection's row-major int8. */
int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim);
int h3_gpu_mlp_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         h3_gpu_tensor *activated,
                         h3_gpu_tensor *quantized_activation,
                         h3_gpu_tensor *activation_scales,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *fc1_weight,
                         const h3_gpu_tensor *fc1_scales,
                         const h3_gpu_tensor *fc2_weight,
                         const h3_gpu_tensor *fc2_scales,
                         const h3_gpu_tensor *fc1_bf16,
                         const h3_gpu_tensor *fc2_bf16, uint32_t rows,
                         uint32_t input_dim, uint32_t hidden_dim,
                         uint32_t output_dim,
                         int use_slower_grouped_quantizer,
                         int use_slower_dynamic_fc1_k,
                         int use_int8_row_fc2,
                         int input_is_quantized);
int h3_gpu_silu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements);
int h3_gpu_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon);
int h3_gpu_layer_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon);
int h3_gpu_gelu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate);
int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half);
int h3_gpu_adaln_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_bf16_offset(h3_gpu *gpu, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon);
int h3_gpu_adaln_linear_bf16(
                      h3_gpu *gpu, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon);
int h3_gpu_gate_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot);
int h3_gpu_gate_adaln_bf16(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon);
int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *gpu, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *quantized_output,
                     h3_gpu_tensor *quantized_scales,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t padded_rows, uint32_t width, uint32_t slots,
                     uint32_t gate_slot, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon);
int h3_gpu_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon);
/* H3 checkpoint QKV rows are [head, q/k/v, dimension], unlike the
 * conventional [q/k/v, head, dimension] layout accepted above. */
int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon);
/* Project grouped H3 QKV and apply its exact Q/K norm/RoPE boundary. Metal 4
 * may route projections directly into the attention layout; other devices
 * retain the ordinary two calls. */
int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon);
int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *gpu,
                                 h3_gpu_tensor *query,
                                 h3_gpu_tensor *key,
                                 h3_gpu_tensor *value,
                                 h3_gpu_tensor *quantized_input,
                                 h3_gpu_tensor *input_scales,
                                 const h3_gpu_tensor *input,
                                 const h3_gpu_tensor *weight,
                                 const h3_gpu_tensor *weight_scales,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t rows, uint32_t input_dim,
                                 uint32_t heads, uint32_t head_dim,
                                 uint32_t rope_half, float epsilon,
                                 int input_is_quantized,
                                 int use_slower_unfused_qkv_rope,
                                 int use_slower_scalar_qkv_rms,
                                 int use_slower_uncached_int8_scales);
int h3_gpu_sdpa_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
/* Preserve SDPA's native [head,row,dimension] output for an immediately
 * following layout-aware projection. */
int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *gpu, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale);
int h3_gpu_swiglu_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width);
int h3_gpu_embedding_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width);
int h3_gpu_text_qk_rope_bf16(h3_gpu *gpu,
                             h3_gpu_tensor *query_output,
                             h3_gpu_tensor *key_output,
                             const h3_gpu_tensor *query_input,
                             const h3_gpu_tensor *key_input,
                             const h3_gpu_tensor *q_norm,
                             const h3_gpu_tensor *k_norm,
                             const h3_gpu_tensor *rope_cos,
                             const h3_gpu_tensor *rope_sin,
                             uint32_t sequence, uint32_t query_heads,
                             uint32_t kv_heads, uint32_t head_dim,
                             float epsilon);
int h3_gpu_head_rms_norm_bf16(h3_gpu *gpu, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon);
int h3_gpu_rope_text_bf16(h3_gpu *gpu, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim);
int h3_gpu_gqa_causal_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale);
int h3_gpu_add_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
int h3_gpu_sub_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements);
int h3_gpu_token_pool_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width);
int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *input, size_t input_offset,
                           h3_gpu_tensor *original, size_t original_offset,
                           h3_gpu_tensor *baseline, size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t input_rows, uint32_t rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents, uint32_t rows,
                           uint32_t reduced_rows, uint32_t baseline_rows,
                           uint32_t width,
                           uint32_t exact_prefix_rows,
                           float update_scale);
int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *gpu, h3_gpu_tensor *residual,
                           h3_gpu_tensor *output,
                           const h3_gpu_tensor *original,
                           size_t original_offset,
                           const h3_gpu_tensor *reduced,
                           const h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *parents,
                           const h3_gpu_tensor *norm_weight,
                           const h3_gpu_tensor *modulation,
                           const h3_gpu_tensor *row_map,
                           uint32_t rows, uint32_t reduced_rows,
                           uint32_t baseline_rows, uint32_t width,
                           uint32_t exact_prefix_rows, float update_scale,
                           uint32_t slots, uint32_t shift_slot,
                           uint32_t scale_slot, float epsilon);
/* Apply one Euler step to an F32 sample range from BF16 velocity caches:
 * sample += delta * (last + ratio * (last - previous)). */
int h3_gpu_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio);
/* Destructively reuse dead activation scratch as F32 C,T,H,W preview storage.
 * Scratch must be disjoint from sample/history/maps and have at least
 * 24*time*height*width*sizeof(float) bytes; its declared dtype is irrelevant.
 * The caller owns its lifetime and submits before readback. No allocation or
 * velocity readback occurs. Optional row classes/strengths match bridge Euler. */
int h3_gpu_video_preview_bf16(h3_gpu *gpu, h3_gpu_tensor *scratch,
    const h3_gpu_tensor *sample, size_t sample_offset,
    const h3_gpu_tensor *last, const h3_gpu_tensor *previous,
    const h3_gpu_tensor *row_classes, const h3_gpu_tensor *strengths,
    uint32_t time, uint32_t height, uint32_t width, float sigma_next, float ratio);
int h3_gpu_video_preview_read(const h3_gpu_tensor *scratch, float *latent,
                              size_t elements);
/* Bridge Euler uses one U32 class per packed target row and a compact F32
 * strength table. Caches remain raw: extrapolate, scale once, then update.
 * Zero-strength rows are not written, including signed zeros/NaN payloads. */
int h3_gpu_bridge_euler_bf16(h3_gpu *gpu, h3_gpu_tensor *sample,
    size_t sample_offset, const h3_gpu_tensor *last,
    const h3_gpu_tensor *previous, const h3_gpu_tensor *row_classes,
    const h3_gpu_tensor *strengths, uint32_t rows, uint32_t row_width,
    float delta, float ratio);
/* OR a nonzero bit into a one-element U32 flag if any zero-strength target
 * element differs from its initial F32 snapshot. Does not clear the flag. */
int h3_gpu_bridge_check_exact(h3_gpu *gpu, const h3_gpu_tensor *sample,
    size_t sample_offset, const h3_gpu_tensor *initial,
    const h3_gpu_tensor *row_classes, const h3_gpu_tensor *strengths,
    h3_gpu_tensor *changed, uint32_t rows, uint32_t row_width);
int h3_gpu_silu_mul_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements);

#ifdef __cplusplus
}
#endif
#endif
