#ifndef H3_BACKEND_H
#define H3_BACKEND_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
typedef enum { H3_BACKEND_MPSGRAPH_REFERENCE = 0, H3_BACKEND_METAL = 1 } h3_backend;
typedef enum { H3_ATTN_DENSE = 0, H3_ATTN_SOL = 1 } h3_attention_mode;
typedef enum { H3_WEIGHT_BF16 = 0, H3_WEIGHT_Q8 = 1 } h3_weight_format;
/* Version 6 adds the optional QKV ANE split recipe and persisted decisions.
 * Version 5 added explicit adapter/fused layout identity; SOL remains recipe 1,
 * while retaining dense FP16 recipe 4. Any incompatible mixed-recipe
 * change must also advance this serialized attention version.
 * All native paths remain opt-in hybrids with MPSGraph linears.
 * Candidate 0 retains pinned Steel; 1 uses the H3 routed/register-Q adapter. */
#define H3_METAL_ATTENTION_VERSION 6
#define H3_METAL_ANE_VERSION 1
#define H3_METAL_LAYOUT_VERSION 1
#define H3_METAL_SOL_VERSION 1
#define H3_METAL_FP16_VERSION 4
typedef struct {
    int candidate, q_block, kv_block, dense_layers, local_radius;
    float tau, min_exact;
    int precision; /* 0 BF16 diagnostic, 1 scaled FP16 / FP32 accumulation */
    int tier; /* 0 diagnostic, 1 Reference candidate, 2 Preview candidate */
    int dense_steps; /* Absolute zero-based schedule evaluations below this are dense. */
    float dense_sigma; /* -1 disables; max(video,audio) >= threshold is dense. */
    int layout_fusion; /* 0 retained adapter, 1 norm/RoPE range fusion and owned-buffer packing. */
    int ane_mode; /* 0 GPU-only, 1 serial offload, 2 static, 3 bounded dynamic. */
    int ane_rows, ane_chunk; /* QKV row budget and fixed compiled chunk size. */
    h3_weight_format weight_format; /* Independent of attention dtype/routing. */
    int q8_kernel; /* 0 bounded Metal dequant + MPSGraph, 1 direct simdgroup. */
} h3_metal_attention_options;
#define H3_METAL_ATTENTION_DEFAULT {0,32,64,1,1,1.0f,0.0f,0,0,0,-1.0f,0,0,4096,512,H3_WEIGHT_BF16,0}
typedef struct {
    h3_backend backend;
    h3_attention_mode attention;
    h3_metal_attention_options metal;
} h3_backend_scope;
#define H3_BACKEND_SCOPE(p) ((h3_backend_scope){(p)->backend,(p)->attention_mode,(p)->metal_attention})
const char *h3_metal_candidate_name(int candidate);
const char *h3_metal_precision_name(int precision);
const char *h3_metal_tier_name(int tier);
const char *h3_weight_format_name(h3_weight_format format);
int h3_weight_format_parse(const char *text, h3_weight_format *format);
int h3_metal_options_valid(h3_metal_attention_options options, char *error, size_t size);
int h3_metal_options_equal(h3_metal_attention_options a, h3_metal_attention_options b);
/* NULL means routing is allowed; otherwise describes the dense override. */
const char *h3_metal_sol_dense_reason(h3_metal_attention_options options, unsigned block,
    int absolute_step, float video_sigma, float audio_sigma);
const char *h3_backend_name(h3_backend backend);
const char *h3_attention_mode_name(h3_attention_mode mode);
int h3_backend_parse(const char *text, h3_backend *backend);
int h3_attention_mode_parse(const char *text, h3_attention_mode *mode);
int h3_backend_preflight(h3_backend_scope scope, int explicit_selection,
                         char *error, size_t size);
h3_backend_scope h3_backend_current(void);
h3_backend_scope h3_backend_exchange(h3_backend_scope scope);
/* Benchmark/test entry points set this ceiling. Ordinary generation is unchanged. */
int h3_test_evaluation_budget(int evaluations, char *error, size_t size);
#ifdef __cplusplus
}
#endif
#endif
