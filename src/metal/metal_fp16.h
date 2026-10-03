#ifndef H3_METAL_FP16_H
#define H3_METAL_FP16_H
#include "src/gpu.h"
#include "src/backend.h"
#ifdef __cplusplus
extern "C" {
#endif
/* M2 dense adapter; T139 admits hybrid testing, not Reference.
 * BF16 interface, power-of-two scaled FP16 Q/K/V and P matrix operands;
 * FP32 QK/PV accumulators, softmax and output. BF16/FP32 per-head recovery.
 * Recipe 4: Metal 3.1 safe math; explicit fast::exp2 from pinned Steel.
 * SOL recipe 1 uses these exact operands with BF16-centroid/FP32 approximate
 * contributions; recovery heads are fully dense. No model state conversions. */
typedef struct {
    float q_max, k_max, v_max, q_min, k_min, v_min;
    int32_t q_exponent, k_exponent, v_exponent;
    uint32_t flags; /* 1 invalid input/score range, 2 FP32 recovery */
    float output_max;
    uint32_t output_invalid;
    float score_error_bound, value_error_bound;
    uint32_t q_underflows,k_underflows,v_underflows;
} h3_metal_fp16_range;
/* Range scans and conversion/commit kernels use uint element indices. Leave
 * room for the final 256-lane scan increment and rounded-up dispatch. Sequence
 * length alone is not a limit: large video/conditioning packs exceed 100K rows.
 * Fused preparation additionally bounds the three interleaved QKV operands. */
static inline int h3_metal_fp16_shape_valid(uint32_t sequence,uint32_t heads) {
    return sequence && heads && heads<=256 &&
        (uint64_t)sequence*heads*128<=UINT32_MAX-255u;
}
/* Requires an active command and distinct BF16 output. Includes range scans,
 * conversions, both conditional kernels and validated BF16 output commit.
 * Invalid heads prevent the entire output commit. Call range_report after
 * submission, before accepting results. Scratch is owned by the GPU context.
 * tile 0 = 32x16, 1 = 64x32. Layout 0 = row-major, 1 = head-major. */
int h3_gpu_mixed_sdpa(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t sequence,uint32_t heads,float scale,int input_layout,int output_layout,int tile);
int h3_gpu_mixed_sol(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t sequence,uint32_t heads,float scale,int input_layout,int output_layout,
    h3_metal_attention_options options,unsigned block,int step);
/* M2E: prepare owned head-major BF16 buffers and their range records together.
 * The subsequent prepared attention consumes this single-use preparation.
 * Dense packing may overwrite Q/K/V with private FP16/recovery operands;
 * callers must not read them as BF16 until the next prepare. SOL retains BF16
 * inputs for summaries. Original raw projection, weights and RoPE are read-only. */
int h3_gpu_mixed_prepare(h3_gpu *gpu,h3_gpu_tensor *q,h3_gpu_tensor *k,h3_gpu_tensor *v,
    const h3_gpu_tensor *qkv,const h3_gpu_tensor *qn,const h3_gpu_tensor *kn,
    const h3_gpu_tensor *cos,const h3_gpu_tensor *sin,uint32_t sequence,uint32_t heads,
    uint32_t dimension,uint32_t rope_half,float epsilon,float scale);
int h3_gpu_mixed_prepared_attention(h3_gpu *gpu,h3_gpu_tensor *out,
    h3_gpu_tensor *q,h3_gpu_tensor *k,h3_gpu_tensor *v,uint32_t sequence,uint32_t heads,
    float scale,int output_layout,h3_metal_attention_options options,int routed,unsigned block,int step);
int h3_gpu_mixed_range_report(h3_gpu *gpu,h3_metal_fp16_range *ranges,uint32_t heads);
/* Archive each block before scratch reuse; finish after forward submission and
 * before accepting velocity or mutating sampler state. Earlier failures cannot
 * be hidden by a subsequent successful block. */
int h3_gpu_mixed_record(h3_gpu *gpu,uint32_t heads,unsigned block,int step);
int h3_gpu_mixed_step_finish(h3_gpu *gpu,int step);
/* Opt-in diagnostics, guarded by H3_TEST_NATIVE_RANGES and the six-evaluation
 * budget. Reductions stay on the GPU until the existing completed-forward wait. */
int h3_gpu_diagnostic_range(h3_gpu *gpu,const h3_gpu_tensor *input,uint32_t count,
    uint32_t width,uint32_t stride,uint32_t offset,unsigned block,unsigned stage);
int h3_gpu_diagnostic_scores(h3_gpu *gpu,const h3_gpu_tensor *q,const h3_gpu_tensor *k,
    uint32_t sequence,uint32_t heads,float scale,int major,unsigned block);
int h3_gpu_diagnostic_finish(h3_gpu *gpu,int step);
#ifdef __cplusplus
}
#endif
#endif
