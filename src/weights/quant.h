#ifndef H3_QUANT_H
#define H3_QUANT_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
#define H3_QUANT_VERSION 2
#define H3_QUANT_ADAPTIVE_VERSION 3
#define H3_QUANT_SUBBLOCK_VERSION 4
enum { H3_QUANT_OFF=0, H3_QUANT_FP8=1, H3_QUANT_NVFP4=2 };
typedef struct { int mode; const char *cache; } h3_quant_scope;
const char *h3_quant_name(int mode);
int h3_quant_parse(const char *text, int *mode);
int h3_quant_options(int mode,const char *cache,char *error,size_t size);
/* Optional diagnostic: H3_QUANT_VERIFY=1 selects the strict content cache. */
int h3_quant_verify(void);
/* Recipe 2: repeated projections on shared arithmetic 4; unchanged SM120
 * tensorwide E4M3 / VEC16 UE4M3 packing, BF16 output, FP32 accumulation.
 * The packing ID versions kernel/layout/scaling changes independently of attention.
 * Non-projection matrices retain their original BF16/FP32 policy.
 * Future per-matrix exceptions belong here and require a recipe version bump. */
int h3_quant_projection(const char *name,int requested);
/* Execution recipe 3 keeps adaptive block 0 in BF16. Packed matrices still
 * use recipe 2; their bytes/scales and cache keys are unchanged. */
unsigned h3_quant_recipe(int mode,int adaptive_cache);
/* Recipe 4 binds unchanged all-block projection arithmetic to SubBlock. */
unsigned h3_quant_execution_recipe(int mode,int adaptive_cache,int attention);
int h3_quant_projection_policy(const char *name,int requested,int adaptive_cache);
h3_quant_scope h3_quant_exchange(h3_quant_scope next);
h3_quant_scope h3_quant_current(void);
/* Runtime/library capability check, before encoders/model allocations. */
int h3_quant_preflight(int mode,char *error,size_t size);
#ifdef __cplusplus
}
#endif
#endif
