#include "src/execution.h"
#include "src/denoise/approximate.h"
#include "src/denoise/adaptive_cache.h"
#include "src/sglang/sglang.h"
#include "src/denoise/attention.h"
#include "src/weights/quant.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static _Thread_local h3_cuda_policy request_policy;
h3_cuda_policy h3_cuda_policy_exchange(h3_cuda_policy policy) {
    h3_cuda_policy old=request_policy;request_policy=policy;return old;
}
h3_cuda_policy h3_cuda_policy_current(void) { return request_policy; }
static int policy_error(char *error,size_t size,const char *why) {
    if (error && size)
        snprintf(error, size, "%s", why);
    return 0;
}
int h3_cuda_policy_resolve(const h3_params *p,const char *backend,int lora,
                          h3_cuda_policy *out,char *error,size_t size) {
    if(!p||!out)return policy_error(error,size,"missing CUDA policy parameters");
    *out=(h3_cuda_policy){0};
    if(!h3_approximate_params_valid(p,lora,error,size))return 0;
    int cuda=backend&&!strcmp(backend,"cuda");
    if(!cuda) {
        if(p->cuda_attention||p->cuda_denoise_quant||p->adaptive_cache)
            return policy_error(error,size,"CUDA execution options require the CUDA backend");
        return 1;
    }
    h3_weight_options weights;
    if(!h3_weight_options_read(&weights,p->ssd_streaming,error,size))return 0;
    out->weights=weights;out->weights_captured=1;
    (void)lora;
    if(p->still) {
        /* Image generation retains its own decoder/shape restrictions, not a fast mode. */
        if(p->cuda_attention||p->cuda_denoise_quant)
            return policy_error(error,size,"CUDA still does not support video attention/precision options");
        return 1;
    }
    if(p->use_int8_row_fc2)
        return policy_error(error,size,"INT8 row FC2 is Metal-only; use CUDA projection precision options");
    if(!h3_attention_options(p->cuda_attention,error,size)||
       !h3_quant_options(p->cuda_denoise_quant,p->cuda_denoise_quant_cache,error,size))return 0;
    *out=(h3_cuda_policy){.active=1,.weights=weights,.weights_captured=1,.base_recipe=H3_SGLANG_VERSION,
        .attention=p->cuda_attention,.projection_precision=p->cuda_denoise_quant,
        .preview=p->preview_vae,.adaptive_cache=p->adaptive_cache,
        .subblock_sparsity=p->subblock_sparsity,
        .adaptive_cache_warmup=p->adaptive_cache_warmup,.subblock_warmup=p->subblock_warmup,
        .adaptive_cache_max_bytes=p->adaptive_cache?h3_adaptive_budget(p->adaptive_cache_max_bytes):0,
        .adaptive_cache_threshold=h3_adaptive_threshold(p),.adaptive_cache_max_hits=h3_adaptive_max_hits(p)};
    return 1;
}
int h3_cuda_policy_preflight(const h3_cuda_policy *p,char *error,size_t size) {
    if(!p||!p->active)return 1;
#if defined(__APPLE__) || !defined(H3_CUDA_USE_SGLANG_FLASH) || !defined(H3_CUDA_USE_CUDNN)
    return policy_error(error,size,"the CUDA pipeline requires CUDA_SGLANG=1 CUDA_CUDNN=1; no legacy fallback is available");
#else
    return h3_adaptive_preflight(p->adaptive_cache,error,size)&&h3_attention_preflight(p->attention,error,size)&&h3_quant_preflight(p->projection_precision,error,size);
#endif
}
