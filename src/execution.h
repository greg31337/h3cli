#ifndef H3_EXECUTION_H
#define H3_EXECUTION_H
#include "src/h3.h"
#include "src/weights/residency.h"
#ifdef __cplusplus
extern "C" {
#endif
/* A resolved request is immutable for the lifetime of its GPU components.
 * active=0 is a non-CUDA/image/deferred-resume scope. There is one CUDA pipeline. */
typedef struct {
    int active;
    h3_weight_options weights;
    int weights_captured;
    int base_recipe, attention, projection_precision;
    int preview;
    int adaptive_cache;
    float subblock_sparsity;
    int adaptive_cache_warmup, subblock_warmup;
    uint64_t adaptive_cache_max_bytes;
    float adaptive_cache_threshold;
    int adaptive_cache_max_hits;
} h3_cuda_policy;
int h3_cuda_policy_resolve(const h3_params *params,const char *backend,int lora,
                          h3_cuda_policy *out,char *error,size_t size);
int h3_cuda_policy_preflight(const h3_cuda_policy *policy,char *error,size_t size);
h3_cuda_policy h3_cuda_policy_exchange(h3_cuda_policy policy);
h3_cuda_policy h3_cuda_policy_current(void);

#ifdef __cplusplus
}
#endif
#endif
