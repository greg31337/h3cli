#include "src/sglang/sglang.h"
/* Linux single-device BF16 backend. Shared C owns model and sampler policy. */
#include "src/gpu.h"
#include "src/device.h"
#include "src/cuda/cuda_dispatch.h"
#include "src/execution.h"
#include "src/memory.h"
#include "src/weights/quant.h"
#include "src/cuda/cuda_cudnn.h"
#include "src/denoise/attention.h"
#include "src/cuda/cuda_sage.h"
#include "src/cuda/cuda_sol.h"
#include "src/cuda/cuda_subblock.h"
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <cublasLt.h>
#include <cublas_v2.h>
#include <algorithm>
#include <atomic>
#include <chrono>
#include <cmath>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cerrno>
#include <climits>
#include <map>
#include <memory>
#include <set>
#include <mutex>
#include <string>
#include <tuple>
#include <vector>
#include <fcntl.h>
#include <unistd.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include "src/cuda/cuda_sglang_blas.cuh"
#ifdef H3_CUDA_USE_SGLANG_FLASH
#include "src/cuda/cuda_sglang_flash.h"
#endif
using gemm_key=std::tuple<unsigned,unsigned,unsigned,int,int,int>;
struct cuda_gemm_plan {
    cublasLtMatmulDesc_t operation=nullptr;
    cublasLtMatrixLayout_t a=nullptr,b=nullptr,c=nullptr;
    cublasLtMatmulAlgo_t algorithm={};
    ~cuda_gemm_plan() {
        if(a)cublasLtMatrixLayoutDestroy(a);if(b)cublasLtMatrixLayoutDestroy(b);
        if(c)cublasLtMatrixLayoutDestroy(c);if(operation)cublasLtMatmulDescDestroy(operation);
    }
};
struct cuda_timing { cudaEvent_t begin=nullptr,end=nullptr; int category=0; };
/* Include file identity, size and nanosecond modification/change times so a
 * replaced checkpoint cannot reuse a previous file's weights. */
using weight_cache_key=std::tuple<uint64_t,uint64_t,int64_t,int64_t,int64_t,int64_t,int64_t,uint64_t,size_t,int>;
struct cached_weight { void *data=nullptr; size_t bytes=0; };
struct host_weight : cached_weight {
    void *mapping=nullptr; size_t mapping_bytes=0, pinned_bytes=0;
};
#ifdef H3_CUDA_EXACT_TEST
/* Fault injection belongs only to the standalone mutable test translation unit. */
static bool exact_test_register_failure=false,exact_test_unregister_failure=false;
#endif
/* Process-wide host staging cap also bounds simultaneous contexts;
 * contents and file identities stay local. */
static std::mutex sglang_host_mutex;
static uint64_t sglang_host_bytes=0;
static constexpr uint64_t sglang_host_limit=40ull<<30;
struct h3_gpu {
    cudaStream_t compute=nullptr, copy=nullptr;
    cublasLtHandle_t lt=nullptr;
    cublasHandle_t upscale_blas=nullptr;
    size_t attention_budget=64u<<20;
    void *workspace=nullptr;
    size_t workspace_bytes=32u<<20;
    void *bounce[2]={nullptr,nullptr};
    cudaEvent_t bounce_free[2]={nullptr,nullptr};
    bool bounce_used[2]={false,false};
    size_t bounce_bytes=16u<<20;
    int device=0;
    bool experiment_timing=false;
    bool active=false, profiling=false, allocation_failed=false, convolution_gemm=false;
    bool sglang_reference=false;
    bool exact_mapped_weights=false,exact_fused_vae_casts=false;
    uint64_t exact_mapped_weight_entries=0,exact_fused_vae_cast_count=0;
    uint64_t exact_copied_weights=0,exact_separate_vae_casts=0;
    h3_cuda_policy policy={};
    bool sglang_text=false;
    bool sglang_vae=false;
    bool sglang_audio=false;
    bool sglang_audio_encoder=false;
    bool sglang_encoder=false;
    std::set<std::tuple<unsigned,unsigned,unsigned>> sglang_audio_activation_shapes;
    void *sglang_conv_cudnn=nullptr;
    h3_gpu_tensor *sglang_conv_input=nullptr,*sglang_conv_output=nullptr,*sglang_conv_workspace=nullptr;
    std::vector<unsigned> sglang_batch_groups;
    std::vector<unsigned> sglang_vision_groups;
    unsigned sglang_heuristic_rows=0;
    std::unique_ptr<h3_sglang_blas> sglang_blas;
    h3_gpu_tensor *sglang_lse=nullptr;
    h3_gpu_tensor *sglang_patch=nullptr;
    h3_gpu_tensor *sglang_audio_math=nullptr,*sglang_audio_scores=nullptr;
    h3_gpu_tensor *sglang_vae_input=nullptr,*sglang_vae_output=nullptr;
    h3_gpu_tensor *sglang_vae_tiles=nullptr,*sglang_vae_canvas=nullptr;
    struct stitch_plan { int h,w,ny,nx,first,frames,height,width,ys[16],xs[16]; } sglang_stitch={};
    int sglang_stitch_next=0;
    int attention_mode=0;
    bool dit_attention_configured=false;
    uint64_t main_dense_calls=0;
    void *sage=nullptr;
    h3_gpu_tensor *sage_workspace=nullptr;
    h3_sage_stats sage_stats={};
    void *subblock=nullptr;
    h3_gpu_tensor *subblock_workspace=nullptr;
    h3_cuda_subblock_stats subblock_stats={};
    std::vector<std::pair<unsigned,unsigned>> subblock_protected;
    uint64_t subblock_bypass=0,subblock_reported=0,subblock_protected_calls=0;
    void *sol=nullptr;
    h3_gpu_tensor *sol_workspace=nullptr;
    h3_cuda_sol_stats sol_stats={};
    h3_cuda_sol_options sol_options=H3_CUDA_SOL_DEFAULT;
    int sol_step=-1;float sol_video=0,sol_audio=0;
    uint64_t sol_bypass=0,sol_reported=0;

    std::atomic<bool> deferred_error{false};
    int architecture=0;
    h3_cuda_dispatch dispatch={};
    char error[1024]={0};
    std::string label="CUDA context";
    h3_gpu_stats stats={};
    h3_weight_options weight_options={};
    h3_weight_plan weight_plan={};
    double start=0;
    std::recursive_mutex mutex;
    std::mutex copy_mutex;
    std::mutex timing_mutex;
    std::map<gemm_key,cublasLtMatmulAlgo_t> algorithms;
    std::map<gemm_key,std::unique_ptr<cuda_gemm_plan>> conv_plans;
    std::vector<h3_gpu_tensor*> accessed;
    std::vector<cuda_timing> timings;
    std::set<std::tuple<int,unsigned,unsigned,unsigned,unsigned,int>> attention_shapes;
    uint64_t skipped_release_events=0;
    std::map<weight_cache_key,host_weight> sglang_host_weights;
    uint64_t sglang_host_hits=0,sglang_host_misses=0,sglang_host_read_bytes=0;
    void *vae_pinned_rgb=nullptr;size_t vae_pinned_bytes=0;
    cudaGraphExec_t vae_graph=nullptr;
    bool vae_warm=false,vae_capturing=false,vae_graph_disabled=false;
    uint64_t vae_replays=0;
    h3_gpu_tensor *conv_columns=nullptr;
    h3_gpu_tensor *vae_q=nullptr,*vae_k=nullptr,*vae_v=nullptr;
    h3_gpu_tensor *vae_rgb=nullptr,*vae_fault=nullptr;
    int quant_mode=0;
    int quant_verify=0;
    bool quant_streaming=false,quant_diagnostics=false;
    std::string quant_cache;
    h3_gpu_tensor *quant_activation=nullptr,*quant_output=nullptr,*quant_stream=nullptr,*quant_fault=nullptr;
    std::map<std::tuple<unsigned,unsigned,unsigned,int>,std::unique_ptr<cuda_gemm_plan>> quant_plans;
    uint64_t quant_calls=0,quant_cache_hits=0,quant_cache_misses=0;
};
void h3_gpu_sglang_text_encoder(h3_gpu *g) { if(g && g->sglang_reference)g->sglang_text=true; }
void h3_gpu_sglang_audio_decoder(h3_gpu *g) {if(g && g->sglang_reference)g->sglang_audio=true;}
void h3_gpu_sglang_audio_encoder(h3_gpu *g) {if(g && g->sglang_reference)g->sglang_audio_encoder=true;}
void h3_gpu_sglang_video_encoder(h3_gpu *g) {if(g && g->sglang_reference)g->sglang_encoder=true;}
int h3_gpu_sglang_batch_groups(h3_gpu *g,const uint32_t *counts,size_t count) {
    if(!g||!g->sglang_reference||count>1000||(count&&!counts))return 0;
    for(size_t i=0;i<count;i++)if(!counts[i]||counts[i]>4)return 0;
    g->sglang_batch_groups.clear();
    if(count)g->sglang_batch_groups.assign(counts,counts+count);
    return 1;
}
struct h3_gpu_tensor {
    h3_gpu *owner=nullptr;
    void *data=nullptr;
    size_t elements=0,bytes=0;
    h3_gpu_dtype dtype=H3_GPU_F32;
    h3_gpu_storage storage=H3_GPU_DEVICE_ONLY;
    cudaEvent_t ready=nullptr;
    cudaEvent_t released=nullptr;
    bool ready_recorded=false;
    mutable bool compute_ready=false;
    bool release_recorded=false;
    bool used=false, streamed=false;
    int quant_mode=0;
    unsigned quant_k=0,quant_n=0;
    size_t quant_scale=0,quant_global=0;
    std::vector<unsigned char> quant_host;
};
struct h3_gpu_event { cudaEvent_t value=nullptr; int device=0; bool recorded=false; };
static double now() {
    return std::chrono::duration<double>(std::chrono::steady_clock::now().time_since_epoch()).count();
}
static int h3_gpu_set_error(h3_gpu *g,const char *fmt,...) {
    if(g){va_list a;va_start(a,fmt);vsnprintf(g->error,sizeof(g->error),fmt,a);va_end(a);}return 0;
}
static int checked(h3_gpu *g,cudaError_t e,const char *op,size_t bytes=0) {
    if(e==cudaSuccess)return 1;
    if(g)g->deferred_error=true;
    size_t available=0,total=0;cudaMemGetInfo(&available,&total);
    return h3_gpu_set_error(g,"CUDA %s: %s; requested %zu bytes; free/total %.3f/%.3f GiB (device %d)",op,cudaGetErrorString(e),bytes,available/1073741824.,total/1073741824.,g?g->device:0);
}
static int selected(h3_gpu *g) { return g && checked(g,cudaSetDevice(g->device),"select device"); }
static int launch_ready(h3_gpu *g) {
    return selected(g) && (g->active || h3_gpu_set_error(g,"CUDA operation requires h3_gpu_begin"));
}
static int launch_status(h3_gpu *g,const char *op) {
    g->stats.direct_dispatches++;
    int ok=checked(g,cudaGetLastError(),op);
    for(auto *t:g->accessed) {
        /* Reference resident tensors remain on the compute stream and cudaFree
         * protects retirement. Only reusable streamed slots need a copy-stream
         * release fence. Fast's existing condition is preserved. */
        if(g->sglang_reference && !t->streamed){g->skipped_release_events++;continue;}
        if(!checked(g,cudaEventRecord(t->released,g->compute),"tensor release event"))ok=0;
        else t->release_recorded=true;
    }
    g->accessed.clear();return ok && !g->deferred_error;
}
static size_t item_size(h3_gpu_dtype dtype) {return dtype==H3_GPU_I8?1:(dtype==H3_GPU_BF16||dtype==H3_GPU_F16)?2:4;}
static void wait_tensor_ready(const h3_gpu_tensor *t);
static void *tensor_pointer(const h3_gpu_tensor *t,size_t offset=0) {
    if(!t)return nullptr;
    auto *g=t->owner;
    const_cast<h3_gpu_tensor*>(t)->used=true;
    if(t->ready_recorded)wait_tensor_ready(t);
    if(std::find(g->accessed.begin(),g->accessed.end(),t)==g->accessed.end())g->accessed.push_back(const_cast<h3_gpu_tensor*>(t));
    return (char*)t->data+offset;
}
/* Event timings are collected after the existing submission synchronization;
 * profiling introduces no device-wide fence or dependency between streams. */
struct profile_scope {
    h3_gpu *gpu;cuda_timing timing;cudaStream_t stream;
    profile_scope(h3_gpu *g,int category,cudaStream_t target=nullptr,bool force=false):gpu(g),stream(target?target:(g?g->compute:nullptr)) {
        if(!g||category<0||(!g->profiling&&!force&&!(g->experiment_timing&&(category==4||category==5))))return;
        timing.category=category;
        if(cudaEventCreate(&timing.begin)!=cudaSuccess || cudaEventCreate(&timing.end)!=cudaSuccess) {
            if(timing.begin)cudaEventDestroy(timing.begin);if(timing.end)cudaEventDestroy(timing.end);
            if(category==8)g->stats.attention_timing_missed++;
            timing={};return;
        }
        cudaEventRecord(timing.begin,stream);
    }
    ~profile_scope(){if(timing.begin){cudaEventRecord(timing.end,stream);std::lock_guard<std::mutex> lock(gpu->timing_mutex);gpu->timings.push_back(timing);}}
};
static void wait_tensor_ready(const h3_gpu_tensor *t) {
    if(t->owner->sglang_reference && t->compute_ready)return;
    profile_scope timing(t->owner,3);
    if(checked(t->owner,cudaStreamWaitEvent(t->owner->compute,t->ready,0),"wait tensor ready"))t->compute_ready=true;
}
static void collect_timings(h3_gpu *g) {
    std::lock_guard<std::mutex> lock(g->timing_mutex);
    size_t pending=0;
    for(auto &t:g->timings) {
        if(cudaEventQuery(t.end)==cudaErrorNotReady){g->timings[pending++]=t;continue;}
        float ms=0;if(cudaEventElapsedTime(&ms,t.begin,t.end)==cudaSuccess) {
            double seconds=ms/1000.;
            if(t.category<=2 || t.category==6)g->stats.gpu_seconds+=seconds;
            if(t.category==0)g->stats.linear_seconds+=seconds;
            else if(t.category==1)g->stats.attention_seconds+=seconds;
            else if(t.category==2)g->stats.conv_seconds+=seconds;
            else if(t.category==3)g->stats.slot_wait_seconds+=seconds;
            else if(t.category==4)g->stats.h2d_seconds+=seconds;
            else if(t.category==5)g->stats.d2h_seconds+=seconds;
            else if(t.category==6)g->stats.normalization_seconds+=seconds;
            else if(t.category==7)g->stats.quant_conversion_seconds+=seconds;
            else if(t.category==8){g->stats.main_attention_seconds+=seconds;g->stats.attention_timing_samples++;}
        }
        cudaEventDestroy(t.begin);cudaEventDestroy(t.end);
    }
    g->timings.resize(pending);
}
static int h3_gpu_require_elements(h3_gpu *g,const h3_gpu_tensor *t,size_t n,const char *name) {
    return (g && t && t->owner==g && t->elements>=n) || h3_gpu_set_error(g,"%s: need %zu elements, have %zu",name,n,t?t->elements:0);
}
static int h3_gpu_require_bf16(h3_gpu *g,const h3_gpu_tensor *t,size_t n,const char *name) {
    return h3_gpu_require_elements(g,t,n,name) && (t->dtype==H3_GPU_BF16 || h3_gpu_set_error(g,"%s requires BF16",name));
}
static int bridge_ranges(const h3_gpu_tensor *sample,size_t offset,const h3_gpu_tensor *rows,const h3_gpu_tensor *strengths,uint32_t count,uint32_t width) {
    size_t n=(size_t)count*width;
    return sample && rows && strengths && count && width && n<=UINT32_MAX && offset<=UINT32_MAX-n && sample->dtype==H3_GPU_F32 && offset<=sample->elements && n<=sample->elements-offset && rows->dtype==H3_GPU_U32 && rows->elements>=count && strengths->dtype==H3_GPU_F32 && strengths->elements && strengths->elements<=UINT32_MAX;
}
#include "src/cuda/cuda_kernels.cuh"
#include "src/cuda/cuda_sglang.cuh"
#include "src/cuda/cuda_reference_wrappers.cuh"

h3_gpu *h3_gpu_create(const char *shader,char *error,size_t size) {
    (void)shader;
    h3_device_info info;
    if(!h3_device_query(&info,error,size))return nullptr;
    const char *nax=getenv("H3_NAX"), *int8=getenv("H3_INT8");
    if((nax && *nax && strcmp(nax,"0")) ||
       (int8 && *int8 && strcmp(int8,"0"))) {
        snprintf(error,size,"Metal TensorOps/INT8 flags are unsupported by the CUDA BF16 backend");return nullptr;
    }
    h3_gpu *g=new(std::nothrow) h3_gpu;
    if(!g){snprintf(error,size,"out of memory creating CUDA context");return nullptr;}
    g->device=info.device_index;g->start=now();
    g->architecture=info.cuda_compute_major*10+info.cuda_compute_minor;
    g->profiling=getenv("H3_PROFILE")!=nullptr;
    g->experiment_timing=getenv("H3_EXPERIMENT_TIMING")&&strcmp(getenv("H3_EXPERIMENT_TIMING"),"0");
    g->policy=h3_cuda_policy_current();
    if(g->policy.weights_captured)g->weight_options=g->policy.weights;
    else if(!h3_weight_options_read(&g->weight_options,0,error,size)){delete g;return nullptr;}
    g->sglang_reference=(g->policy.active?g->policy.base_recipe==H3_SGLANG_VERSION:
        h3_sglang_requested());
    g->exact_mapped_weights=g->sglang_reference && !getenv("H3_TEST_EXACT_NO_MMAP");
    g->exact_fused_vae_casts=g->sglang_reference && getenv("H3_TEST_EXACT_VAE_FUSION") &&
        !strcmp(getenv("H3_TEST_EXACT_VAE_FUSION"),"1");
    g->dispatch=h3_cuda_dispatch_select(g->architecture,0);
    int ok=checked(g,cudaStreamCreateWithFlags(&g->compute,cudaStreamNonBlocking),"create compute stream") &&
        checked(g,cudaStreamCreateWithFlags(&g->copy,cudaStreamNonBlocking),"create copy stream");
    if(ok && cublasLtCreate(&g->lt)!=CUBLAS_STATUS_SUCCESS)ok=h3_gpu_set_error(g,"create cuBLASLt handle failed");
    if(ok && g->sglang_reference) {
#ifndef H3_CUDA_USE_SGLANG_FLASH
        ok=h3_gpu_set_error(g,"SGLang reference requires a CUDA_SGLANG=1 build");
#else
        fprintf(stderr,"h3cli: SGLang CUDA reference recipe v%d; attention=%s; production unqualified\n",
            H3_SGLANG_VERSION,"pinned dense FlashAttention BF16/FP32");
#endif
    }
    if(ok && g->workspace_bytes)ok=checked(g,cudaMalloc(&g->workspace,g->workspace_bytes),"GEMM workspace",g->workspace_bytes);
    if(ok && g->sglang_reference) {
        g->sglang_blas=std::make_unique<h3_sglang_blas>();
        ok=g->sglang_blas->initialize(g->compute,g->workspace,g->workspace_bytes,g->error,sizeof(g->error));
    }
    for(int i=0;ok&&i<2;i++) {
        ok=checked(g,cudaHostAlloc(&g->bounce[i],g->bounce_bytes,cudaHostAllocDefault),"stream bounce allocation",g->bounce_bytes) &&
            checked(g,cudaEventCreateWithFlags(&g->bounce_free[i],cudaEventDisableTiming),"bounce reuse event");
    }
    if(!ok){snprintf(error,size,"%s",g->error);h3_gpu_free(g);return nullptr;}
    g->stats.pinned_bytes=g->stats.peak_pinned_bytes=2*g->bounce_bytes;
    static std::mutex log_mutex; static std::vector<int> logged;
    {std::lock_guard<std::mutex> lock(log_mutex);
     if(std::find(logged.begin(),logged.end(),g->device)==logged.end()) {
        fprintf(stderr,"h3cli: CUDA device %d: %s, SM%d%d, %.2f/%.2f GiB free/total, runtime %d driver %d; BF16/F32; %s; weight mode %s\n",g->device,info.name,info.cuda_compute_major,info.cuda_compute_minor,info.free_device_memory/1073741824.,info.device_memory/1073741824.,info.cuda_runtime_version,info.cuda_driver_version,g->dispatch.name,getenv("H3_CUDA_WEIGHT_MODE")?getenv("H3_CUDA_WEIGHT_MODE"):"auto");
        logged.push_back(g->device);
     }}
    return g;
}
/* Caller owns copy_mutex then mutex, or exclusive teardown ownership. */
static int clear_sglang_host_weights(h3_gpu *g) {
    if(g->sglang_host_weights.empty())return 1;
    if(!checked(g,cudaStreamSynchronize(g->copy),"reference host weights release"))return 0;
    std::lock_guard<std::mutex> lock(sglang_host_mutex);
    for(auto entry=g->sglang_host_weights.begin();entry!=g->sglang_host_weights.end();) {
        auto &w=entry->second;
        if(w.mapping) {
#ifdef H3_CUDA_EXACT_TEST
            if(exact_test_unregister_failure) {exact_test_unregister_failure=false;
                return h3_gpu_set_error(g,"injected mapped weights unregister failure");}
#endif
            if(!checked(g,cudaHostUnregister(w.mapping),"mapped weights unregister"))return 0;
            munmap(w.mapping,w.mapping_bytes);
        } else if(!checked(g,cudaFreeHost(w.data),"reference host weights free"))return 0;
        g->stats.pinned_bytes-=w.pinned_bytes;sglang_host_bytes-=w.pinned_bytes;
        /* Erase each successful release so retry after a later failure cannot
         * double-free it or subtract its budget twice. */
        entry=g->sglang_host_weights.erase(entry);
    }
    g->sglang_host_weights.clear();return 1;
}
int h3_gpu_release_weight_cache(h3_gpu *g) {
    if(!selected(g))return 0;
    std::lock_guard<std::mutex> io(g->copy_mutex);
    std::lock_guard<std::recursive_mutex> lock(g->mutex);
    return clear_sglang_host_weights(g);
}
void h3_gpu_free(h3_gpu *g) {
    if(!g)return;selected(g);
    if(g->vae_capturing){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(g->compute,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);g->vae_capturing=false;}
    if(g->compute)cudaStreamSynchronize(g->compute);
    if(g->copy)cudaStreamSynchronize(g->copy);
    collect_timings(g);
    h3_gpu_profile_mark(g,"complete");
    if(g->dit_attention_configured)fprintf(stderr,"h3cli: main DiT attention counters requested=%s dense=%llu sage2=%llu sage3=%llu sol=%llu protected_bypass=%llu workspace=%zu\n",
        h3_attention_name(g->attention_mode),(unsigned long long)g->main_dense_calls,
        (unsigned long long)g->sage_stats.calls[1],(unsigned long long)g->sage_stats.calls[2],
        (unsigned long long)g->sol_stats.calls,(unsigned long long)g->sol_bypass,
        (g->sage_workspace?g->sage_workspace->bytes:0)+(g->sol_workspace?g->sol_workspace->bytes:0)+(g->subblock_workspace?g->subblock_workspace->bytes:0));
    if(g->sglang_reference)fprintf(stderr,"h3cli: shared exact substitutions %s: mode=%s mapped_weight_entries=%llu copied_weight_entries=%llu fused_vae_qkv_casts=%llu separate_vae_qkv_casts=%llu\n",
        g->label.c_str(),"single",(unsigned long long)g->exact_mapped_weight_entries,
        (unsigned long long)g->exact_copied_weights,(unsigned long long)g->exact_fused_vae_cast_count,
        (unsigned long long)g->exact_separate_vae_casts);
    if(g->sglang_reference)fprintf(stderr,"h3cli: reference allocation counters %s: peak_device_tensor_bytes=%llu peak_pinned_bytes=%llu host_weight_entries=%zu host_weight_read_bytes=%llu H2D_bytes=%llu D2H_bytes=%llu\n",
        g->label.c_str(),(unsigned long long)g->stats.peak_live_bytes,(unsigned long long)g->stats.peak_pinned_bytes,
        g->sglang_host_weights.size(),(unsigned long long)g->sglang_host_read_bytes,
        (unsigned long long)g->stats.h2d_bytes,(unsigned long long)g->stats.d2h_bytes);
    if(g->dit_attention_configured)fprintf(stderr,"h3cli: projection counters requested=%s recipe=%d native_calls=%llu cache_hits=%llu prepared=%llu\n",
        h3_quant_name(g->quant_mode),h3_quant_execution_recipe(g->quant_mode,g->policy.adaptive_cache,g->policy.attention),
        (unsigned long long)g->quant_calls,(unsigned long long)g->quant_cache_hits,
        (unsigned long long)g->quant_cache_misses);
    clear_sglang_host_weights(g);
    g->conv_plans.clear();
    g->quant_plans.clear();
    if(g->vae_graph)cudaGraphExecDestroy(g->vae_graph);
    if(g->vae_pinned_rgb)cudaFreeHost(g->vae_pinned_rgb);
    h3_gpu_tensor_free(g->conv_columns);
    for(auto *t:{g->vae_q,g->vae_k,g->vae_v,g->vae_rgb,g->vae_fault})h3_gpu_tensor_free(t);
    h3_gpu_tensor_free(g->quant_activation);h3_gpu_tensor_free(g->quant_output);
    h3_gpu_tensor_free(g->quant_stream);h3_gpu_tensor_free(g->quant_fault);
    if(g->lt)cublasLtDestroy(g->lt);
    if(g->upscale_blas)cublasDestroy(g->upscale_blas);
    h3_cuda_subblock_free(g->subblock);g->subblock=nullptr;
    h3_gpu_tensor_free(g->subblock_workspace);
    h3_cuda_sol_free(g->sol);g->sol=nullptr;
    if(g->sol_workspace)h3_gpu_tensor_free(g->sol_workspace);
    h3_sage_free(g->sage);g->sage=nullptr;
    if(g->sage_workspace)h3_gpu_tensor_free(g->sage_workspace);
    h3_cudnn_reference_free(g->sglang_conv_cudnn);
    h3_gpu_tensor_free(g->sglang_conv_input);
    h3_gpu_tensor_free(g->sglang_conv_output);
    h3_gpu_tensor_free(g->sglang_conv_workspace);
    g->sglang_blas.reset();
    h3_gpu_tensor_free(g->sglang_lse);
    h3_gpu_tensor_free(g->sglang_patch);
    h3_gpu_tensor_free(g->sglang_audio_math);h3_gpu_tensor_free(g->sglang_audio_scores);
    h3_gpu_tensor_free(g->sglang_vae_input);
    h3_gpu_tensor_free(g->sglang_vae_output);
    h3_gpu_tensor_free(g->sglang_vae_tiles);
    h3_gpu_tensor_free(g->sglang_vae_canvas);
    if(g->workspace)cudaFree(g->workspace);
    for(int i=0;i<2;i++){if(g->bounce_free[i])cudaEventDestroy(g->bounce_free[i]);if(g->bounce[i])cudaFreeHost(g->bounce[i]);}
    if(g->compute)cudaStreamDestroy(g->compute);
    if(g->copy)cudaStreamDestroy(g->copy);
    delete g;
}
int h3_gpu_begin(h3_gpu *g) {
    if(!selected(g))return 0;
    if(g->active)return h3_gpu_set_error(g,"CUDA command stream already active");
    g->error[0]=0;g->deferred_error=false;
    if(g->vae_fault && !checked(g,cudaMemsetAsync(g->vae_fault->data,0,sizeof(unsigned),g->compute),"reset VAE finite flag"))return 0;
    g->active=true;return 1;
}
int h3_gpu_continue(h3_gpu *g) { return launch_ready(g) && checked(g,cudaPeekAtLastError(),"continue"); }
int h3_gpu_synchronize(h3_gpu *g) {
    if(!selected(g))return 0;
    double t=now();int ok=checked(g,cudaStreamSynchronize(g->compute),"compute synchronization");
    if(ok && g->vae_fault) {
        unsigned fault=0;
        ok=checked(g,cudaMemcpy(&fault,g->vae_fault->data,sizeof(fault),cudaMemcpyDeviceToHost),"VAE finite check");
        if(ok&&fault)ok=h3_gpu_set_error(g,"nonfinite full VAE mixed-precision operand or output");
    }
    if(ok && g->quant_fault) {
        uint64_t values[3]={};
        ok=checked(g,cudaMemcpy(values,g->quant_fault->data,g->quant_diagnostics?sizeof(values):sizeof(unsigned),cudaMemcpyDeviceToHost),"quantization finite check");
        if(ok&&values[0])ok=h3_gpu_set_error(g,"nonfinite input encountered in quantized DiT projection");
        if(g->quant_diagnostics){g->stats.quant_clipped_values=values[1];g->stats.quant_zeroed_values=values[2];}
    }
    if(ok&&g->sage)ok=h3_sage_collect(g->sage,&g->sage_stats,g->error,sizeof(g->error));
    if(ok&&g->sol) {
        ok=h3_cuda_sol_collect(g->sol,&g->sol_stats,g->error,sizeof(g->error));
        if(ok&&g->sol_stats.calls!=g->sol_reported) {
            auto &s=g->sol_stats;g->sol_reported=s.calls;
            fprintf(stderr,"h3cli: SOL counters step=%d calls=%llu dense_bypass=%llu exact=%llu approximate=%llu protected=%llu local=%llu floor=%llu recovery=%llu tail=%llu workspace=%llu reserved=%zu\n",
                g->sol_step,(unsigned long long)s.calls,(unsigned long long)g->sol_bypass,
                (unsigned long long)s.pairs[0],(unsigned long long)s.pairs[1],(unsigned long long)s.pairs[2],(unsigned long long)s.pairs[3],
                (unsigned long long)s.pairs[4],(unsigned long long)s.pairs[5],(unsigned long long)s.pairs[6],(unsigned long long)s.workspace_bytes,g->sol_workspace->bytes);
        }
    }
    if(ok&&g->subblock) {
        ok=h3_cuda_subblock_collect(g->subblock,&g->subblock_stats,g->error,sizeof(g->error));
        if(ok&&g->subblock_stats.calls!=g->subblock_reported) {
            auto &s=g->subblock_stats;g->subblock_reported=s.calls;
            fprintf(stderr,"h3cli: subblock step=%d calls=%llu dense_bypass=%llu protected_calls=%llu selected=%llu possible=%llu protected_queries=%llu added_protected=%llu workspace=%llu pool_seconds=%.9g route_seconds=%.9g kernel_seconds=%.9g output_seconds=%.9g profiled=%llu unprofiled=%llu\n",
                g->sol_step,(unsigned long long)s.calls,(unsigned long long)g->subblock_bypass,(unsigned long long)g->subblock_protected_calls,
                (unsigned long long)s.pairs[0],(unsigned long long)s.pairs[1],(unsigned long long)s.pairs[2],(unsigned long long)s.pairs[3],
                (unsigned long long)s.workspace_bytes,s.pool_seconds,s.route_seconds,s.kernel_seconds,s.output_seconds,
                (unsigned long long)s.profiled_regions,(unsigned long long)s.unprofiled_regions);
        }
    }
    g->stats.command_wait_seconds+=now()-t;if(ok)collect_timings(g);return ok && !g->deferred_error;
}
int h3_gpu_submit(h3_gpu *g) {
    if(!launch_ready(g))return 0;
    int ok=h3_gpu_synchronize(g);g->active=false;g->stats.submissions++;return ok;
}
void h3_gpu_cancel(h3_gpu *g) { if(g){
    if(g->vae_capturing){cudaGraph_t abandoned=nullptr;cudaStreamEndCapture(g->compute,&abandoned);if(abandoned)cudaGraphDestroy(abandoned);g->vae_capturing=false;g->vae_graph_disabled=true;}
    h3_gpu_synchronize(g);
    if(g->vae_fault && cudaMemsetAsync(g->vae_fault->data,0,sizeof(unsigned),g->compute)==cudaSuccess)cudaStreamSynchronize(g->compute);
    if(g->quant_fault && cudaMemsetAsync(g->quant_fault->data,0,g->quant_fault->bytes,g->compute)==cudaSuccess)cudaStreamSynchronize(g->compute);
    if(g->sage&&!h3_sage_reset(g->sage,g->error,sizeof(g->error)))g->deferred_error=true;
    if(g->subblock&&!h3_cuda_subblock_reset(g->subblock,g->error,sizeof(g->error)))g->deferred_error=true;
    if(g->sol&&!h3_cuda_sol_reset(g->sol,g->error,sizeof(g->error)))g->deferred_error=true;
    g->sglang_vision_groups.clear();
    g->active=false;} }
const char *h3_gpu_error(const h3_gpu *g) {return g?g->error:"missing CUDA context";}
int h3_gpu_get_stats(const h3_gpu *g,h3_gpu_stats *s) {if(!g||!s)return 0;*s=g->stats;s->quant_projection_dispatches=g->quant_calls;s->quant_cache_hits=g->quant_cache_hits;s->quant_cache_misses=g->quant_cache_misses;s->sage2_attention_dispatches=g->sage_stats.calls[1];s->sage3_attention_dispatches=g->sage_stats.calls[2];s->attention_workspace_bytes=(g->sage_workspace?g->sage_workspace->bytes:0)+(g->sol_workspace?g->sol_workspace->bytes:0) +(g->subblock_workspace?g->subblock_workspace->bytes:0);
    s->main_dense_calls=g->main_dense_calls;s->subblock_calls=g->subblock_stats.calls;
    s->subblock_bypass=g->subblock_bypass;s->subblock_protected_calls=g->subblock_protected_calls;
    s->subblock_selected=g->subblock_stats.pairs[0];s->subblock_possible=g->subblock_stats.pairs[1];
    s->subblock_router_calls=g->subblock_stats.router_calls;
    s->subblock_router_seconds=g->subblock_stats.pool_seconds+g->subblock_stats.route_seconds;
    s->subblock_kernel_seconds=g->subblock_stats.kernel_seconds;return 1;}
void h3_gpu_profile_set_label(h3_gpu *g,const char *label){if(g&&label)g->label=label;}
void h3_gpu_profile_mark(h3_gpu *g,const char *phase) {
    if(!g||!getenv("H3_PROFILE"))return;
    h3_gpu_stats complete_stats;h3_gpu_get_stats(g,&complete_stats);
    if(g->stats.active_weight_blocks)fprintf(stderr,"h3cli: weight placement %s: resident=%u/%u resident_bytes=%llu slot_bytes=%llu pinned_bytes=%llu uploaded_bytes=%llu host_hits=%llu logical_host_read_bytes=%llu\n",
        phase,g->stats.resident_blocks,g->stats.active_weight_blocks,
        (unsigned long long)g->stats.resident_weight_bytes,(unsigned long long)g->stats.stream_slot_bytes,
        (unsigned long long)g->stats.pinned_bytes,(unsigned long long)g->stats.streamed_bytes,
        (unsigned long long)g->sglang_host_hits,(unsigned long long)g->sglang_host_read_bytes);
    if(g->sglang_vae)fprintf(stderr,"h3cli: full VAE profile: graph_replays=%llu pinned_output=%zu\n",
        (unsigned long long)g->vae_replays,g->vae_pinned_bytes);
    if(g->quant_mode)fprintf(stderr,"h3cli: quant profile %s / %s: mode=%s recipe=%d native_calls=%llu cache_hits=%llu prepared=%llu conversion=%.6fs (included in GEMM) stream=%s\n",
        g->label.c_str(),phase,h3_quant_name(g->quant_mode),h3_quant_execution_recipe(g->quant_mode,g->policy.adaptive_cache,g->policy.attention),
        (unsigned long long)g->quant_calls,(unsigned long long)g->quant_cache_hits,
        (unsigned long long)g->quant_cache_misses,g->stats.quant_conversion_seconds,g->quant_streaming?"compressed":"resident");
    if(g->quant_diagnostics)fprintf(stderr,"h3cli: quant packing diagnostics (weights and activations): clipped=%llu nonzero-to-zero=%llu\n",
        (unsigned long long)g->stats.quant_clipped_values,(unsigned long long)g->stats.quant_zeroed_values);
    fprintf(stderr,"h3cli: CUDA profile %s / %s: wall %.3fs peak %.3f GiB, GEMM %llu attention %llu conv %llu, H2D %.3f GiB D2H %.3f GiB stream %.3f GiB copy %.3fs wait %.3fs\n",g->label.c_str(),phase,now()-g->start,complete_stats.peak_live_bytes/1073741824.,(unsigned long long)g->stats.linear_dispatches,(unsigned long long)g->stats.attention_dispatches,(unsigned long long)g->stats.conv_dispatches,g->stats.h2d_bytes/1073741824.,g->stats.d2h_bytes/1073741824.,g->stats.streamed_bytes/1073741824.,g->stats.transfer_seconds,g->stats.command_wait_seconds);
    if(g->sage)fprintf(stderr,"h3cli: Sage profile attention=%s recipe=%d plan=%d sage2=%llu sage3=%llu groups=%llu workspace=%llu reduction=%.6f pack=%.6f correction=%.6f kernel=%.6f output=%.6f seconds\n",
        h3_attention_name(g->attention_mode),H3_ATTENTION_VERSION,H3_ATTENTION_PLAN_VERSION,
        (unsigned long long)g->sage_stats.calls[1],(unsigned long long)g->sage_stats.calls[2],
        (unsigned long long)g->sage_stats.head_groups,(unsigned long long)g->sage_stats.workspace_bytes,
        g->sage_stats.reduction_seconds,g->sage_stats.pack_seconds,g->sage_stats.correction_seconds,
        g->sage_stats.kernel_seconds,g->sage_stats.output_seconds);
    if(g->sol)fprintf(stderr,"h3cli: SOL profile finite=%.6f summary=%.6f router=%.6f fused_exact_centroid_merge=%.6f output=%.6f profiled_regions=%llu unprofiled_regions=%llu seconds\n",
        g->sol_stats.finite_seconds,g->sol_stats.summary_seconds,g->sol_stats.route_seconds,g->sol_stats.fused_seconds,g->sol_stats.output_seconds,
        (unsigned long long)g->sol_stats.profiled_regions,(unsigned long long)g->sol_stats.unprofiled_regions);
    fprintf(stderr,"h3cli: CUDA category seconds: GEMM %.6f attention %.6f conv %.6f; pinned peak %.3f GiB; bounce stalls %llu / %.6fs\n",g->stats.linear_seconds,g->stats.attention_seconds,g->stats.conv_seconds,g->stats.peak_pinned_bytes/1073741824.,(unsigned long long)g->stats.stream_stalls,g->stats.stream_wait_seconds);
    fprintf(stderr,"h3cli: CUDA transfer events: H2D %.6fs (%.3f GiB/s), D2H %.6fs; uncovered upload wait %.6fs; normalization/elementwise %.6fs; source read/upload wall %.6fs\n",g->stats.h2d_seconds,g->stats.h2d_seconds?g->stats.h2d_bytes/1073741824./g->stats.h2d_seconds:0.0,g->stats.d2h_seconds,g->stats.slot_wait_seconds,g->stats.normalization_seconds,g->stats.transfer_seconds);
    if(g->sglang_reference)fprintf(stderr,"h3cli: reference host weights: entries=%zu hits=%llu misses=%llu file_read=%.3f GiB pinned=%.3f GiB process_cap=40 GiB\n",
        g->sglang_host_weights.size(),(unsigned long long)g->sglang_host_hits,
        (unsigned long long)g->sglang_host_misses,g->sglang_host_read_bytes/1073741824.,g->stats.pinned_bytes/1073741824.);
}
int h3_gpu_is_m5(const h3_gpu*) {return 0;}
const char *h3_gpu_backend_name(const h3_gpu*) {return "CUDA";}
int h3_gpu_prefers_device_sampler(const h3_gpu *g,int) {return g!=nullptr;}
int h3_gpu_should_retry_streaming(const h3_gpu *g) {
    return g&&g->allocation_failed&&g->weight_options.mode==H3_WEIGHTS_AUTO;
}
int h3_gpu_has_nax_mlp(const h3_gpu*) {return 0;}
int h3_gpu_has_int8_mlp(const h3_gpu*) {return 0;}
static uint64_t weight_available(h3_gpu *g,uint64_t available) {
    uint64_t limit=g->weight_options.capacity_limit;
    return limit?std::min(available,limit>g->stats.live_bytes?limit-g->stats.live_bytes:0):available;
}
int h3_gpu_plan_weights(h3_gpu *g,uint64_t weights,uint64_t activations) {
    if(!selected(g))return -1;
    std::lock_guard<std::mutex> io(g->copy_mutex);
    std::lock_guard<std::recursive_mutex> lock(g->mutex);
    size_t available,total;
    if(!checked(g,cudaMemGetInfo(&available,&total),"weight planner"))return -1;
    uint64_t reserve=std::max<uint64_t>(1ull<<30,total/10),free=weight_available(g,available);
    bool fit=activations<=free && reserve<=free-activations && weights<=free-activations-reserve;
    int mode=g->weight_options.mode;
    if(mode==H3_WEIGHTS_RESIDENT&&!fit){h3_gpu_set_error(g,"resident weights need %.2f GiB plus %.2f GiB activations/reserve; %.2f GiB free",weights/1073741824.,activations/1073741824.+reserve/1073741824.,free/1073741824.);return -1;}
    if(mode==H3_WEIGHTS_STREAM)fit=false;
    if(activations>free||reserve>free-activations){h3_gpu_set_error(g,"insufficient CUDA capacity for streamed activations/reserve");return -1;}
    fprintf(stderr,"h3cli: CUDA packed weight planner: requested=%s effective=%s weights=%llu future=%llu reserve=%llu free=%llu live=%llu capacity_cap=%llu\n",
        h3_weight_mode_name(mode),fit?"resident":"stream",(unsigned long long)weights,(unsigned long long)activations,
        (unsigned long long)reserve,(unsigned long long)free,(unsigned long long)g->stats.live_bytes,(unsigned long long)g->weight_options.capacity_limit);
    return fit?0:1;
}
int h3_gpu_plan_bf16_weights(h3_gpu *g,uint64_t mask,uint64_t block,uint64_t future,
    int force_stream,int retry_cap,h3_weight_plan *plan) {
    if(!selected(g)||!plan)return 0;
    size_t available,total;
    if(!checked(g,cudaMemGetInfo(&available,&total),"BF16 weight planner"))return 0;
    h3_weight_options options=g->weight_options;
    if(force_stream&&options.mode==H3_WEIGHTS_RESIDENT)return h3_gpu_set_error(g,"--ssd-streaming conflicts with CUDA resident weight mode");
    if(force_stream)options.mode=H3_WEIGHTS_STREAM;
    if(retry_cap>=0&&(options.max_resident<0||retry_cap<options.max_resident))options.max_resident=retry_cap;
    uint64_t reserve=std::max<uint64_t>(1ull<<30,total/10);
    int ok=h3_weight_plan_build(plan,options,mask,block,available,g->stats.live_bytes,future,reserve,g->error,sizeof(g->error));
    fprintf(stderr,"h3cli: CUDA BF16 weight planner: requested=%s effective=%s resident=%u/%u mask=%llx resident_bytes=%llu slots=%llu future=%llu reserve=%llu free=%llu live=%llu count_cap=%d capacity_cap=%llu retry_cap=%d%s\n",
        h3_weight_mode_name(options.mode),ok?(plan->effective_mode==H3_WEIGHTS_AUTO?"partial":h3_weight_mode_name(plan->effective_mode)):"rejected",
        plan->resident_count,plan->active_count,(unsigned long long)plan->resident_mask,(unsigned long long)plan->resident_bytes,
        (unsigned long long)plan->slot_bytes,(unsigned long long)future,(unsigned long long)reserve,(unsigned long long)plan->free_bytes,
        (unsigned long long)g->stats.live_bytes,options.max_resident,(unsigned long long)options.capacity_limit,retry_cap,ok?"":g->error);
    if(ok){g->weight_plan=*plan;g->stats.resident_weight_bytes=plan->resident_bytes;g->stats.stream_slot_bytes=plan->slot_bytes;
        g->stats.resident_blocks=plan->resident_count;g->stats.active_weight_blocks=plan->active_count;g->stats.resident_block_mask=plan->resident_mask;}
    return ok;
}
int h3_gpu_weight_plan_compatible(h3_gpu *g,int force_stream) {
    if(!g)return 0;
    h3_weight_options requested;char error[128];h3_cuda_policy policy=h3_cuda_policy_current();
    if(policy.weights_captured)requested=policy.weights;
    else if(!h3_weight_options_read(&requested,force_stream,error,sizeof(error)))return 0;
    h3_weight_options old=g->weight_options;
    if(force_stream){requested.mode=H3_WEIGHTS_STREAM;old.mode=H3_WEIGHTS_STREAM;}
    if(!h3_weight_options_equal(&requested,&old))return 0;
    /* A constrained automatic plan must reconsider capacity on each request:
     * external allocations may disappear as well as grow. Full and explicit
     * streamed contexts can be retained subject to the normal headroom check. */
    if(old.mode==H3_WEIGHTS_AUTO && (g->quant_streaming ||
       (g->weight_plan.active_count && g->weight_plan.resident_count<g->weight_plan.active_count)))return 0;
    size_t free,total;
    return selected(g)&&cudaMemGetInfo(&free,&total)==cudaSuccess&&free>=std::max<uint64_t>(1ull<<30,total/10);
}
h3_gpu_tensor *h3_gpu_tensor_alloc(h3_gpu *g,size_t n,h3_gpu_dtype dtype,h3_gpu_storage storage) {
    if(!selected(g) || dtype<H3_GPU_F32 || dtype>H3_GPU_F16 || storage<H3_GPU_HOST_VISIBLE || storage>H3_GPU_HOST_PINNED || n>SIZE_MAX/item_size(dtype))return nullptr;
    std::lock_guard<std::mutex> io(g->copy_mutex);
    std::lock_guard<std::recursive_mutex> lock(g->mutex);
    auto *t=new(std::nothrow) h3_gpu_tensor;
    if(!t)return nullptr;
    t->owner=g;t->elements=n;t->bytes=n*item_size(dtype);t->dtype=dtype;t->storage=storage;
    size_t bytes=std::max<size_t>(t->bytes,1);
    uint64_t budget=g->weight_options.allocation_limit;
    if(budget && (g->stats.live_bytes>budget || bytes>budget-g->stats.live_bytes)) {
        g->allocation_failed=true;h3_gpu_set_error(g,"CUDA tensor allocation exceeds injected test budget");delete t;return nullptr;
    }
    cudaError_t e=storage==H3_GPU_HOST_VISIBLE ? cudaMallocManaged(&t->data,bytes) : storage==H3_GPU_HOST_PINNED ? cudaHostAlloc(&t->data,bytes,cudaHostAllocMapped) : cudaMalloc(&t->data,bytes);
    if(!checked(g,e,"tensor allocation",bytes)){g->allocation_failed=e==cudaErrorMemoryAllocation;cudaGetLastError();delete t;return nullptr;}
    if(!checked(g,cudaEventCreateWithFlags(&t->ready,cudaEventDisableTiming),"tensor upload event")){
        if(storage==H3_GPU_HOST_PINNED)cudaFreeHost(t->data);else cudaFree(t->data);delete t;return nullptr;
    }
    if(!checked(g,cudaEventCreateWithFlags(&t->released,cudaEventDisableTiming),"tensor release event")) {
        cudaEventDestroy(t->ready);if(storage==H3_GPU_HOST_PINNED)cudaFreeHost(t->data);else cudaFree(t->data);delete t;return nullptr;
    }
    g->stats.allocated_bytes+=t->bytes;g->stats.live_bytes+=t->bytes;g->stats.tensor_allocations++;
    g->stats.peak_live_bytes=std::max(g->stats.peak_live_bytes,g->stats.live_bytes);
    if(storage==H3_GPU_DEVICE_ONLY)g->stats.device_bytes+=t->bytes;
    else if(storage==H3_GPU_HOST_VISIBLE)g->stats.managed_bytes+=t->bytes;
    else {g->stats.pinned_bytes+=t->bytes;g->stats.peak_pinned_bytes=std::max(g->stats.peak_pinned_bytes,g->stats.pinned_bytes);}
    return t;
}
void *h3_gpu_tensor_contents(h3_gpu_tensor *t){
    if(!t||t->storage==H3_GPU_DEVICE_ONLY||!h3_gpu_synchronize(t->owner))return nullptr;
    if(t->ready_recorded&&!checked(t->owner,cudaEventSynchronize(t->ready),"host-visible tensor ready"))return nullptr;
    return t->data;
}
void h3_gpu_tensor_free(h3_gpu_tensor *t) {
    if(!t)return;auto *g=t->owner;selected(g);
    g->accessed.erase(std::remove(g->accessed.begin(),g->accessed.end(),t),g->accessed.end());
    // cudaFree's synchronization protects tensors retired immediately after enqueue.
    if(t->ready_recorded)cudaEventSynchronize(t->ready);
    if(t->storage==H3_GPU_HOST_PINNED){h3_gpu_synchronize(g);cudaFreeHost(t->data);}else cudaFree(t->data);
    if(t->ready)cudaEventDestroy(t->ready);
    if(t->released)cudaEventDestroy(t->released);
    {std::lock_guard<std::recursive_mutex> lock(g->mutex);g->stats.live_bytes-=t->bytes;
     if(t->storage==H3_GPU_DEVICE_ONLY)g->stats.device_bytes-=t->bytes;else if(t->storage==H3_GPU_HOST_VISIBLE)g->stats.managed_bytes-=t->bytes;else g->stats.pinned_bytes-=t->bytes;}
    delete t;
}
size_t h3_gpu_tensor_elements(const h3_gpu_tensor *t){return t?t->elements:0;}
h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *t){return t?t->dtype:H3_GPU_F32;}
int h3_gpu_tensor_is_file_mapped(const h3_gpu_tensor*){return 0;}
static int transfer(h3_gpu_tensor *t,size_t offset,void *values,size_t n,h3_gpu_dtype dtype,bool write) {
    if(!t || t->dtype!=dtype || offset>t->elements || n>t->elements-offset || (!values&&n))return 0;
    auto *g=t->owner;if(!h3_gpu_synchronize(g))return 0;
    if(t->ready_recorded && !checked(g,cudaEventSynchronize(t->ready),"wait tensor upload"))return 0;
    size_t bytes=n*item_size(dtype);double started=now();
    void *p=(char*)t->data+offset*item_size(dtype);
    profile_scope timing(g,write?4:5);
    int ok=checked(g,cudaMemcpyAsync(write?p:values,write?values:p,bytes,write?cudaMemcpyHostToDevice:cudaMemcpyDeviceToHost,g->compute),write?"tensor host upload":"tensor readback",bytes) && checked(g,cudaStreamSynchronize(g->compute),"host transfer completion");
    {std::lock_guard<std::recursive_mutex> guard(g->mutex);
     if(write)g->stats.h2d_bytes+=bytes;else g->stats.d2h_bytes+=bytes;
     g->stats.transfer_seconds+=now()-started;}
    return ok;
}
#define TRANSFERS(label,type,dtype) \
int h3_gpu_tensor_read_##label(const h3_gpu_tensor*t,type*v,size_t n){return transfer(const_cast<h3_gpu_tensor*>(t),0,v,n,dtype,false);} \
int h3_gpu_tensor_write_##label(h3_gpu_tensor*t,const type*v,size_t n){return transfer(t,0,(void*)v,n,dtype,true);} \
int h3_gpu_tensor_write_##label##_range(h3_gpu_tensor*t,size_t off,const type*v,size_t n){return transfer(t,off,(void*)v,n,dtype,true);}
TRANSFERS(f32,float,H3_GPU_F32)
TRANSFERS(bf16,uint16_t,H3_GPU_BF16)
int h3_gpu_tensor_read_u32(const h3_gpu_tensor*t,uint32_t*v,size_t n){return transfer(const_cast<h3_gpu_tensor*>(t),0,v,n,H3_GPU_U32,false);}
int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor*t,size_t off,float*v,size_t n){return transfer(const_cast<h3_gpu_tensor*>(t),off,v,n,H3_GPU_F32,false);}
#define NEW(label,dtype) h3_gpu_tensor*h3_gpu_tensor_new_##label(h3_gpu*g,size_t n){return h3_gpu_tensor_alloc(g,n,dtype,H3_GPU_DEVICE_ONLY);}
NEW(f32,H3_GPU_F32) NEW(bf16,H3_GPU_BF16) NEW(i8,H3_GPU_I8)
#define FROM(label,type,dtype) h3_gpu_tensor*h3_gpu_tensor_from_##label(h3_gpu*g,const type*v,size_t n){auto*t=h3_gpu_tensor_alloc(g,n,dtype,H3_GPU_DEVICE_ONLY);if(t&&!transfer(t,0,(void*)v,n,dtype,true)){h3_gpu_tensor_free(t);t=nullptr;}return t;}
FROM(f32,float,H3_GPU_F32) FROM(bf16,uint16_t,H3_GPU_BF16) FROM(u32,uint32_t,H3_GPU_U32)
/* copy_mutex held. Failure to admit optional staging falls back to the normal
 * bounded bounce buffers. Read errors still fail closed. No model hashes. */
static int sglang_host_weight(h3_gpu *g,int fd,uint64_t offset,size_t bytes,
                             h3_gpu_dtype dtype,void **data) {
    *data=nullptr;if(!g->sglang_reference||!bytes)return 1;
    struct stat st;
    if(fstat(fd,&st)||!S_ISREG(st.st_mode)||offset>(uint64_t)st.st_size||bytes>(uint64_t)st.st_size-offset)
        return h3_gpu_set_error(g,"invalid reference weight file/range");
    weight_cache_key key=std::make_tuple(st.st_dev,st.st_ino,st.st_size,
        st.st_mtim.tv_sec,st.st_mtim.tv_nsec,st.st_ctim.tv_sec,st.st_ctim.tv_nsec,offset,bytes,(int)dtype);
    auto found=g->sglang_host_weights.find(key);
    if(found!=g->sglang_host_weights.end()) {*data=found->second.data;g->sglang_host_hits++;return 1;}
    g->sglang_host_misses++;
    void *mapping=nullptr;size_t mapping_bytes=0,pinned_bytes=0;
    long page_size=sysconf(_SC_PAGESIZE);
    if(page_size<=0)return 1;
    size_t page=(size_t)page_size,delta=offset%page;
    if(bytes>SIZE_MAX-delta || bytes+delta>SIZE_MAX-(page-1))return 1;
    size_t mapped_charge=((bytes+delta+page-1)/page)*page;
    size_t copied_charge=((bytes+page-1)/page)*page;
    auto release=[&]() {
        if(mapping){cudaHostUnregister(mapping);munmap(mapping,mapping_bytes);}
        else cudaFreeHost(*data);
        *data=nullptr;
    };
    {
        std::lock_guard<std::mutex> lock(sglang_host_mutex);
        char reason[512];
        size_t admission=g->exact_mapped_weights?mapped_charge:copied_charge;
        if(admission>sglang_host_limit-sglang_host_bytes ||
           !h3_memory_check(admission+(16ull<<30),"optional reference host weights",reason,sizeof(reason)))return 1;
        /* Shared exact loading can pin the file-backed pages directly. This removes the
         * allocation/zeroing/pread copy while preserving the streamed slots,
         * metadata keys, 40 GiB process cap and copy-stream release fence.
         * Unsupported filesystems/registration fall back to the shared path. */
        if(g->exact_mapped_weights) {
            {
                size_t length=bytes+delta;
                /* CUDA on this driver cannot register read-only pages. A
                 * private writable mapping permits pinning/COW but cannot
                 * write back to the model. We only read this staging area. */
                void *candidate=mmap(nullptr,length,PROT_READ|PROT_WRITE,MAP_PRIVATE,fd,(off_t)(offset-delta));
                if(candidate!=MAP_FAILED) {
                    cudaError_t registration;
#ifdef H3_CUDA_EXACT_TEST
                    registration=exact_test_register_failure?cudaErrorNotSupported:
                        cudaHostRegister(candidate,length,cudaHostRegisterDefault);
#else
                    registration=cudaHostRegister(candidate,length,cudaHostRegisterDefault);
#endif
                    if(registration==cudaSuccess) {
                        mapping=candidate;mapping_bytes=length;*data=(char*)mapping+delta;
                    } else {cudaGetLastError();munmap(candidate,length);}
                }
            }
        }
        cudaError_t status=mapping?cudaSuccess:cudaHostAlloc(data,bytes,cudaHostAllocDefault);
        if(status!=cudaSuccess){cudaGetLastError();*data=nullptr;return 1;}
        pinned_bytes=mapping?mapped_charge:copied_charge;
        sglang_host_bytes+=pinned_bytes;
    }
    size_t got=mapping?bytes:0;
    while(got<bytes) {
        ssize_t n=pread(fd,(char*)*data+got,bytes-got,(off_t)(offset+got));
        if(n<0&&errno==EINTR)continue;
        if(n<=0) {
            int saved=errno;release();
            {std::lock_guard<std::mutex> lock(sglang_host_mutex);sglang_host_bytes-=pinned_bytes;}
            return h3_gpu_set_error(g,"read reference host weight: %s",n?strerror(saved):"unexpected EOF");
        }
        got+=(size_t)n;
    }
    /* Reject a concurrent replacement/edit while filling. A later stat change
     * creates a distinct key and cannot hit the old immutable entry. */
    struct stat after;
    if(fstat(fd,&after)||st.st_size!=after.st_size||
       st.st_mtim.tv_sec!=after.st_mtim.tv_sec||st.st_mtim.tv_nsec!=after.st_mtim.tv_nsec||
       st.st_ctim.tv_sec!=after.st_ctim.tv_sec||st.st_ctim.tv_nsec!=after.st_ctim.tv_nsec) {
        release();
        {std::lock_guard<std::mutex> lock(sglang_host_mutex);sglang_host_bytes-=pinned_bytes;}
        return h3_gpu_set_error(g,"reference weight changed while loading");
    }
    g->sglang_host_weights.emplace(key,host_weight{{*data,bytes},mapping,mapping_bytes,pinned_bytes});
    if(mapping)g->exact_mapped_weight_entries++;else g->exact_copied_weights++;
    g->stats.pinned_bytes+=pinned_bytes;g->sglang_host_read_bytes+=bytes;
    g->stats.peak_pinned_bytes=std::max(g->stats.peak_pinned_bytes,g->stats.pinned_bytes);
    return 1;
}
int h3_gpu_sglang_preload_weight(h3_gpu *g,const char *path,uint64_t offset,size_t elements) {
    if(!g||!g->sglang_reference||!path||elements>INT64_MAX/2||offset>INT64_MAX-elements*2||!selected(g))return 0;
    int fd=open(path,O_RDONLY|O_CLOEXEC);if(fd<0)return h3_gpu_set_error(g,"open reference weight %s: %s",path,strerror(errno));
    std::lock_guard<std::mutex> lock(g->copy_mutex);void *data=nullptr;
    int ok=sglang_host_weight(g,fd,offset,elements*2,H3_GPU_BF16,&data);close(fd);return ok;
}
static int file_into(h3_gpu_tensor *t,const char *path,uint64_t offset,size_t n,h3_gpu_dtype dtype,bool streaming,char *error,size_t error_size) {
    if(!t||!path||t->dtype!=dtype||n>t->elements||offset>INT64_MAX||n>INT64_MAX/item_size(dtype)||n*item_size(dtype)>INT64_MAX-offset)return 0;
    auto *g=t->owner;if(!selected(g))return 0;
    int fd=open(path,O_RDONLY|O_CLOEXEC);
    if(fd<0){snprintf(error,error_size,"%s: %s",path,strerror(errno));return 0;}
    std::lock_guard<std::mutex> lock(g->copy_mutex);
    /* Resident tensors need no per-kernel copy-stream release event. Before
     * their FIRST overwrite/transition into a streaming slot, drain prior
     * compute reads once. Every streamed generation keeps its original fence. */
    if(g->sglang_reference && t->used && !t->streamed &&
       !checked(g,cudaStreamSynchronize(g->compute),"resident tensor overwrite fence")){close(fd);return 0;}
    if(streaming)t->streamed=true;
    if(t->release_recorded && !checked(g,cudaStreamWaitEvent(g->copy,t->released,0),"stream slot release")){close(fd);return 0;}
    size_t bytes=n*item_size(dtype),chunk=g->bounce_bytes;
    double started=now();int ok=1;unsigned slot=0;
    auto ready=[&]() {
        int success=checked(g,cudaEventRecord(t->ready,g->copy),"weight ready event");
        t->ready_recorded=success;t->compute_ready=false;return success;
    };
    if(streaming && g->sglang_reference) {
        void *data=nullptr;ok=sglang_host_weight(g,fd,offset,bytes,dtype,&data);
        if(ok && data) {
            {profile_scope timing(g,4,g->copy);
             ok=checked(g,cudaMemcpyAsync(t->data,data,bytes,cudaMemcpyHostToDevice,g->copy),"reference cached host weight H2D",bytes);}
            if(ok)ok=ready();
            g->stats.h2d_bytes+=bytes;g->stats.streamed_bytes+=bytes;g->stats.transfer_seconds+=now()-started;
        }
        if(!ok || data) {
            close(fd);
            if(!ok&&error&&error_size&&error!=g->error)snprintf(error,error_size,"%s",g->error);
            return ok;
        }
    }
    /* Opt in because network filesystems and pinning limits vary. A failed
     * registration falls back to the bounded pread bounce buffers. */
    const char *registered=getenv("H3_CUDA_REGISTER_WEIGHTS");
    if(bytes && registered && !strcmp(registered,"1")) {
        size_t page=(size_t)sysconf(_SC_PAGESIZE),delta=(size_t)(offset%page);
        struct stat st;
        if(bytes<=SIZE_MAX-delta && !fstat(fd,&st) && offset+bytes<=(uint64_t)st.st_size) {
            size_t length=bytes+delta;
            void *mapping=mmap(nullptr,length,PROT_READ,MAP_PRIVATE,fd,(off_t)(offset-delta));
            if(mapping!=MAP_FAILED) {
                cudaError_t pinned=cudaHostRegister(mapping,length,cudaHostRegisterReadOnly);
                if(pinned==cudaSuccess) {
                    {profile_scope timing(g,4,g->copy);
                    ok=checked(g,cudaMemcpyAsync(t->data,(char*)mapping+delta,bytes,cudaMemcpyHostToDevice,g->copy),"registered weight H2D",bytes);
                    }
                    if(ok)ok=ready();
                    if(ok)ok=checked(g,cudaEventSynchronize(t->ready),"registered source release");
                    else cudaStreamSynchronize(g->copy);
                    cudaHostUnregister(mapping);munmap(mapping,length);close(fd);
                    {std::lock_guard<std::recursive_mutex> guard(g->mutex);
                     g->stats.peak_pinned_bytes=std::max<uint64_t>(g->stats.peak_pinned_bytes,g->stats.pinned_bytes+length);
                     g->stats.h2d_bytes+=bytes;if(streaming)g->stats.streamed_bytes+=bytes;g->stats.transfer_seconds+=now()-started;}
                    if(!ok&&error&&error_size&&error!=g->error)snprintf(error,error_size,"%s",g->error);
                    return ok;
                }
                cudaGetLastError();munmap(mapping,length);
            }
        }
    }
    for(size_t done=0;ok&&done<bytes;) {
        void *bounce=g->bounce[slot];
        if(g->bounce_used[slot]) {
            double waiting=now();
            if(cudaEventQuery(g->bounce_free[slot])==cudaErrorNotReady)g->stats.stream_stalls++;
            if(!checked(g,cudaEventSynchronize(g->bounce_free[slot]),"bounce buffer reuse")){ok=0;break;}
            g->stats.stream_wait_seconds+=now()-waiting;
        }
        size_t want=std::min(chunk,bytes-done),got=0;
        while(got<want){ssize_t r=pread(fd,(char*)bounce+got,want-got,(off_t)(offset+done+got));if(r<0&&errno==EINTR)continue;if(r<=0){ok=h3_gpu_set_error(g,"read weight %s at %llu: %s",path,(unsigned long long)(offset+done+got),r?strerror(errno):"unexpected EOF");break;}got+=(size_t)r;}
        if(ok){profile_scope timing(g,4,g->copy);ok=checked(g,cudaMemcpyAsync((char*)t->data+done,bounce,want,cudaMemcpyHostToDevice,g->copy),"weight H2D",want) && checked(g,cudaEventRecord(g->bounce_free[slot],g->copy),"bounce release");}
        g->bounce_used[slot]=true;slot^=1;done+=want;
    }
    if(ok)ok=ready();
    close(fd);
    {std::lock_guard<std::recursive_mutex> guard(g->mutex);g->stats.h2d_bytes+=bytes;if(streaming)g->stats.streamed_bytes+=bytes;g->stats.transfer_seconds+=now()-started;}
    if(!ok&&error&&error_size&&error!=g->error)snprintf(error,error_size,"%s",g->error);
    return ok;
}
int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor*t,const char*p,uint64_t off,size_t n,char*e,size_t s){return file_into(t,p,off,n,H3_GPU_BF16,false,e,s);}
int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor*t,const char*p,uint64_t off,size_t n,char*e,size_t s){return file_into(t,p,off,n,H3_GPU_BF16,true,e,s);}
static h3_gpu_tensor *load_file(h3_gpu*g,const char*p,uint64_t off,size_t n,h3_gpu_dtype dtype){auto*t=h3_gpu_tensor_alloc(g,n,dtype,H3_GPU_DEVICE_ONLY);if(t&&!file_into(t,p,off,n,dtype,false,g->error,sizeof(g->error))){h3_gpu_tensor_free(t);t=nullptr;}return t;}
h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu*g,const char*p,uint64_t off,size_t n){return load_file(g,p,off,n,H3_GPU_BF16);}
h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu*g,const char*p,uint64_t off,size_t n){return load_file(g,p,off,n,H3_GPU_F32);}
static int copy_tensor(h3_gpu*g,h3_gpu_tensor*d,size_t dst,const h3_gpu_tensor*s,size_t src,size_t n,h3_gpu_dtype dtype){
    if(!launch_ready(g)||!d||!s||d->dtype!=dtype||s->dtype!=dtype||dst>d->elements||n>d->elements-dst||src>s->elements||n>s->elements-src)return 0;
    size_t z=item_size(dtype);
    {std::lock_guard<std::recursive_mutex> guard(g->mutex);
     g->stats.d2d_bytes+=n*z;g->stats.blit_copies++;}
    int ok=checked(g,cudaMemcpyAsync(tensor_pointer(d,dst*z),tensor_pointer(s,src*z),n*z,cudaMemcpyDeviceToDevice,g->compute),"tensor D2D",n*z);
    return launch_status(g,"D2D release")&&ok;
}
int h3_gpu_copy_f32(h3_gpu*g,h3_gpu_tensor*d,size_t dst,const h3_gpu_tensor*s,size_t src,size_t n){return copy_tensor(g,d,dst,s,src,n,H3_GPU_F32);}
int h3_gpu_copy_bf16(h3_gpu*g,h3_gpu_tensor*d,size_t dst,const h3_gpu_tensor*s,size_t src,size_t n){return copy_tensor(g,d,dst,s,src,n,H3_GPU_BF16);}
h3_gpu_event *h3_gpu_event_new(h3_gpu*g){if(!selected(g))return nullptr;auto*e=new(std::nothrow)h3_gpu_event;if(!e)return nullptr;e->device=g->device;if(!checked(g,cudaEventCreateWithFlags(&e->value,cudaEventDisableTiming),"create event")){delete e;return nullptr;}return e;}
int h3_gpu_event_record(h3_gpu*g,h3_gpu_event*e){if(!selected(g)||!e||g->device!=e->device)return 0;return e->recorded=checked(g,cudaEventRecord(e->value,g->compute),"record event");}
int h3_gpu_event_wait(h3_gpu*g,h3_gpu_event*e){return selected(g)&&e&&e->recorded&&g->device==e->device&&checked(g,cudaStreamWaitEvent(g->compute,e->value,0),"wait event");}
void h3_gpu_event_free(h3_gpu_event*e){if(e){cudaSetDevice(e->device);cudaEventDestroy(e->value);delete e;}}

#include "src/cuda/cuda_quant.cuh"

/* cuBLASLt sees the row-major H3 projection as W^T X in column-major order.
 * Compute is explicitly FP32; TF32/FP16/quantized substitutions are never used. */
static int vae_conv_gemm(h3_gpu*,void*,const void*,const void*,void*,unsigned,unsigned,unsigned);
int h3_gpu_sglang_time_features(h3_gpu*g,h3_gpu_tensor*out,const h3_gpu_tensor*times,uint32_t rows) {
    if(!g||!g->sglang_reference||!launch_ready(g)||!rows||rows>4000||!out||!times||
       out->dtype!=H3_GPU_F32||times->dtype!=H3_GPU_F32||out->elements<(size_t)rows*256||times->elements<rows)return 0;
    h3_sg_time_features<<<rows,256,0,g->compute>>>((float*)tensor_pointer(out),(const float*)tensor_pointer(times),rows);
    return launch_status(g,"reference timestep features");
}
int h3_gpu_sglang_rope_trig(h3_gpu *g,h3_gpu_tensor *cosine,h3_gpu_tensor *sine,
    const h3_gpu_tensor *angles,uint32_t count) {
    if(!g||!g->sglang_reference||!launch_ready(g)||!count||!cosine||!sine||!angles||
       cosine==sine||cosine->dtype!=H3_GPU_BF16||sine->dtype!=H3_GPU_BF16||angles->dtype!=H3_GPU_F32||
       cosine->elements<count||sine->elements<count||angles->elements<count)return 0;
    h3_sg_rope_trig<<<((size_t)count+255)/256,256,0,g->compute>>>((ushort*)tensor_pointer(cosine),
        (ushort*)tensor_pointer(sine),(const float*)tensor_pointer(angles),count);
    return launch_status(g,"reference CUDA rotary sine/cosine");
}
int h3_gpu_sglang_vision_groups(h3_gpu *g,const uint32_t *counts,size_t count) {
    if(!g||!g->sglang_reference)return 0;
    g->sglang_vision_groups.clear();
    if(!count&&!counts)return 1;
    if(!count||count>128||!counts)return h3_gpu_set_error(g,"invalid reference vision group count");
    uint64_t total=0;
    for(size_t i=0;i<count;i++) {
        if(!counts[i]||counts[i]%4||counts[i]>UINT32_MAX/4304-total)
            return h3_gpu_set_error(g,"reference vision groups exceed element index range or merge alignment");
        total+=counts[i];
    }
    g->sglang_vision_groups.assign(counts,counts+count);return 1;
}
int h3_gpu_sglang_vision_prepare_range(h3_gpu *g,h3_gpu_tensor *position,const h3_gpu_tensor *table,
    h3_gpu_tensor *cosine,h3_gpu_tensor *sine,uint32_t height,uint32_t width,uint32_t offset) {
    size_t rows=(size_t)height*width;
    if(!g||!g->sglang_reference||height<2||width<2||height>1024||width>1024||(height%2)||(width%2)||
       !position||!table||!cosine||!sine||position->dtype!=H3_GPU_BF16||table->dtype!=H3_GPU_BF16||
       cosine->dtype!=H3_GPU_F32||sine->dtype!=H3_GPU_F32||position->elements<(rows+offset)*1152||table->elements<48*48*1152||
       cosine->elements<(rows+offset)*36||sine->elements<(rows+offset)*36||!h3_gpu_begin(g))return 0;
    h3_sg_vision_positions<<<rows,256,0,g->compute>>>((ushort*)tensor_pointer(position)+(size_t)offset*1152,(const ushort*)tensor_pointer(table),
        (float*)tensor_pointer(cosine)+(size_t)offset*36,(float*)tensor_pointer(sine)+(size_t)offset*36,height,width);
    int ok=launch_status(g,"reference vision positions")&&h3_gpu_submit(g);if(!ok)h3_gpu_cancel(g);return ok;
}
int h3_gpu_sglang_vision_prepare(h3_gpu *g,h3_gpu_tensor *p,const h3_gpu_tensor *t,
    h3_gpu_tensor *c,h3_gpu_tensor *s,uint32_t h,uint32_t w) {
    return h3_gpu_sglang_vision_prepare_range(g,p,t,c,s,h,w,0);
}
__global__ static void sglang_posterior(float *out,const float *moments,const float *epsilon,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float logvar=fminf(20.f,fmaxf(-30.f,moments[n+i]));
    float std=expf(__fmul_rn(.5f,logvar));
    out[i]=h3_sg_half(__fadd_rn(moments[i],__fmul_rn(std,epsilon[i])));
}
int h3_gpu_sglang_posterior(h3_gpu *g,const float *moments,const float *epsilon,float *sample,size_t n) {
    if(!g||!g->sglang_reference||!moments||!epsilon||!sample||!n||n>(64u<<20)/sizeof(float))return 0;
    auto *m=h3_gpu_tensor_from_f32(g,moments,2*n),*e=h3_gpu_tensor_from_f32(g,epsilon,n);
    auto *o=h3_gpu_tensor_new_f32(g,n);int ok=m&&e&&o&&h3_gpu_begin(g);
    if(ok){
        sglang_posterior<<<(n+255)/256,256,0,g->compute>>>((float*)tensor_pointer(o),(float*)tensor_pointer(m),(float*)tensor_pointer(e),n);
        ok=launch_status(g,"reference encoder posterior")&&h3_gpu_submit(g)&&h3_gpu_tensor_read_f32(o,sample,n);
    }
    if(!ok)h3_gpu_cancel(g);
    h3_gpu_tensor_free(m);h3_gpu_tensor_free(e);h3_gpu_tensor_free(o);return ok;
}
static int gemm(h3_gpu*g,h3_gpu_tensor*out,size_t out_offset,const h3_gpu_tensor*in,size_t in_offset,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t rows,uint32_t k,uint32_t n,h3_gpu_dtype dtype) {
    if(w&&w->quant_mode) {
        if(dtype!=H3_GPU_BF16||out_offset||in_offset||b)return h3_gpu_set_error(g,"unsupported quantized projection offset/bias/type");
        return quant_gemm(g,out,in,w,rows,k,n);
    }
    size_t size=item_size(dtype);
    if(!launch_ready(g)||!rows||!k||!n||!in||!out||in->dtype!=dtype||out->dtype!=dtype||!w||w->dtype!=dtype||in_offset>in->elements||(size_t)rows*k>in->elements-in_offset||out_offset>out->elements||(size_t)rows*n>out->elements-out_offset||w->elements<(size_t)n*k||(b&&(b->dtype!=dtype||b->elements<n)))return h3_gpu_set_error(g,"linear %ux%ux%u: invalid tensor shape/type",rows,n,k);
    void *x=tensor_pointer(in,in_offset*size),*weight=tensor_pointer(w),*y=tensor_pointer(out,out_offset*size),*bias=tensor_pointer(b);
    linear_args args={rows,k,n,b?1u:0u};
    profile_scope timing(g,g->convolution_gemm?-1:0);
    /* Match the pinned torch BF16 F.linear route, including its permitted
     * BF16 partial reductions. cuBLASLt heuristics choose a different
     * reduction on real Qwen tensors even with identical operands. */
    if(g->sglang_reference && !g->convolution_gemm) {
        float alpha=1,beta=0;
        cudaDataType_t data=dtype==H3_GPU_BF16?CUDA_R_16BF:CUDA_R_32F;
        size_t total=0;for(unsigned count:g->sglang_batch_groups)total+=count;
        if(total && total!=rows)return h3_gpu_set_error(g,"reference timestep batch geometry mismatch");
        size_t groups=total?g->sglang_batch_groups.size():1,offset=0;
        for(size_t group=0;group<groups;group++) {
            unsigned count=total?g->sglang_batch_groups[group]:rows;
            void *part_x=(char*)x+offset*k*size,*part_y=(char*)y+offset*n*size;
            cublasStatus_t status=b?g->sglang_blas->biased(part_y,part_x,weight,bias,count,k,n,data,g->workspace,g->workspace_bytes,g->compute,g->sglang_heuristic_rows):
                g->sglang_blas->gemm(g->sglang_blas->handle,CUBLAS_OP_T,CUBLAS_OP_N,
                n,count,k,&alpha,weight,data,k,part_x,data,k,
                &beta,part_y,data,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
            if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,
                "reference cuBLAS linear %ux%ux%u failed (%d; CUDA %s)",count,n,k,(int)status,cudaGetErrorString(cudaGetLastError()));
            offset+=count;g->stats.linear_dispatches++;
        }
        return launch_status(g,"reference cuBLAS linear");
    }
    // Small projections avoid heuristic/descriptor overhead. Reference mode
    // retains this straightforward FP32-accumulating path for diagnostics.
    if(!g->sglang_reference && ((uint64_t)rows*k*n<=32768 || rows<32)) {
        dim3 grid((n+15)/16,(rows+15)/16),block(16,16);
        if(dtype==H3_GPU_F32)h3_linear_f32_tiled<<<grid,block,0,g->compute>>>((float*)x,(float*)weight,(float*)bias,(float*)y,args);
        else h3_linear_bf16<<<grid,block,0,g->compute>>>((ushort*)x,(ushort*)weight,(ushort*)bias,(ushort*)y,args);
        g->stats.linear_dispatches++;return launch_status(g,"small linear");
    }
    if(g->convolution_gemm &&  !getenv("H3_TEST_VAE_CONV_NO_REUSE")) {
        int result=vae_conv_gemm(g,y,x,weight,bias,rows,k,n);
        if(result<0)return 0;
        if(result>0){g->stats.linear_dispatches++;return launch_status(g,"prepared FP32 convolution GEMM");}
    }
    cublasLtMatmulDesc_t operation=nullptr;cublasLtMatrixLayout_t a=nullptr,bb=nullptr,c=nullptr;
    cublasLtMatmulPreference_t preference=nullptr;cublasStatus_t status;
    cudaDataType_t data=dtype==H3_GPU_BF16?CUDA_R_16BF:CUDA_R_32F;
    cublasComputeType_t compute=CUBLAS_COMPUTE_32F;
    cublasOperation_t transpose=CUBLAS_OP_T;
    float alpha=1,beta=0;
    gemm_key key={rows,n,k,(int)dtype,b?1:0,0};
    cublasLtMatmulAlgo_t algorithm;
#define LT(call) do{status=(call);if(status!=CUBLAS_STATUS_SUCCESS)goto done;}while(0)
    LT(cublasLtMatmulDescCreate(&operation,compute,CUDA_R_32F));
    LT(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose)));
    if(b){cublasLtEpilogue_t ep=CUBLASLT_EPILOGUE_BIAS;LT(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_EPILOGUE,&ep,sizeof(ep)));LT(cublasLtMatmulDescSetAttribute(operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias)));}
    LT(cublasLtMatrixLayoutCreate(&a,data,k,n,k));LT(cublasLtMatrixLayoutCreate(&bb,data,k,rows,k));LT(cublasLtMatrixLayoutCreate(&c,data,n,rows,n));
    {
        auto found=g->algorithms.find(key);
        if(found!=g->algorithms.end())algorithm=found->second;
        else {
            LT(cublasLtMatmulPreferenceCreate(&preference));
            LT(cublasLtMatmulPreferenceSetAttribute(preference,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&g->workspace_bytes,sizeof(g->workspace_bytes)));
            cublasLtMatmulHeuristicResult_t result[8];int count=0;
            LT(cublasLtMatmulAlgoGetHeuristic(g->lt,operation,a,bb,c,c,preference,8,result,&count));
            if(!count){status=CUBLAS_STATUS_NOT_SUPPORTED;goto done;}
            algorithm=result[0].algo;g->algorithms.emplace(key,algorithm);
        }
    }
    LT(cublasLtMatmul(g->lt,operation,&alpha,weight,a,x,bb,&beta,y,c,y,c,&algorithm,g->workspace,g->workspace_bytes,g->compute));
    g->stats.linear_dispatches++;
done:
    if(preference)cublasLtMatmulPreferenceDestroy(preference);
    if(a)cublasLtMatrixLayoutDestroy(a);if(bb)cublasLtMatrixLayoutDestroy(bb);if(c)cublasLtMatrixLayoutDestroy(c);if(operation)cublasLtMatmulDescDestroy(operation);
#undef LT
    if(status!=CUBLAS_STATUS_SUCCESS){size_t free_bytes=0,total_bytes=0;cudaMemGetInfo(&free_bytes,&total_bytes);g->accessed.clear();return h3_gpu_set_error(g,"cuBLASLt linear %ux%ux%u dtype %d: status %d; output %zu bytes, free/total %.3f/%.3f GiB",rows,n,k,(int)dtype,(int)status,(size_t)rows*n*size,free_bytes/1073741824.,total_bytes/1073741824.);}
    return launch_status(g,"cuBLASLt linear");
}
int h3_gpu_linear_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t m,uint32_t k,uint32_t n){return gemm(g,o,0,i,0,w,b,m,k,n,H3_GPU_F32);}
int h3_gpu_sglang_head_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t m,uint32_t k,uint32_t n,uint32_t packed_rows) {
    if(!g||!g->sglang_reference||!b||k!=5376||(n!=32&&n!=96)||packed_rows<m||packed_rows>1000000||packed_rows%64)return 0;
    unsigned previous=g->sglang_heuristic_rows;g->sglang_heuristic_rows=packed_rows;
    int ok=gemm(g,o,0,i,0,w,b,m,k,n,H3_GPU_F32);g->sglang_heuristic_rows=previous;return ok;
}
int h3_gpu_linear_bf16(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t m,uint32_t k,uint32_t n){return gemm(g,o,0,i,0,w,b,m,k,n,H3_GPU_BF16);}
__global__ static void sglang_patch_bias(ushort *out,const ushort *bias,size_t count,unsigned width) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<count)
        out[i]=h3_f32_to_bf16(h3_bf16_to_f32(out[i])+h3_bf16_to_f32(bias[i%width]));
}
__global__ static void sglang_vision_patch_pack(ushort *out,const ushort *in,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    size_t row=i/8192,pixel=(i%8192)/16,channel=i%16;
    out[i]=channel<3?in[row*1536+channel*512+pixel]:0;
}
int h3_gpu_sglang_vision_patch(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *weight,const h3_gpu_tensor *bias,uint32_t rows,uint32_t k,uint32_t n) {
    if(!g||!g->sglang_reference||!rows||rows>UINT32_MAX/4304||!bias||bias->dtype!=H3_GPU_BF16||bias->elements<n||k!=1536||n!=1152||
       !launch_ready(g)||!h3_gpu_require_elements(g,in,(size_t)rows*k,"vision patches")||in->dtype!=H3_GPU_BF16||
       !h3_gpu_require_elements(g,weight,(size_t)n*k,"vision filter")||weight->dtype!=H3_GPU_BF16||
       !h3_gpu_require_elements(g,out,(size_t)rows*n,"vision output")||out->dtype!=H3_GPU_BF16)return 0;
    /* Match the pinned Conv3d's NDHWC, padded-16-channel reduction order.
     * This single patch projection uses full FP32 partial reductions; ordinary
     * reference linears keep their independently qualified BF16 reduction rule.
     * Bound temporary packing to 512 MiB input plus 18 MiB filter. */
    uint32_t chunk_rows=std::min(rows,H3_VISION_PATCH_CHUNK_ROWS);
    h3_gpu_tensor *x=h3_gpu_tensor_new_bf16(g,(size_t)chunk_rows*8192),
        *w=h3_gpu_tensor_new_bf16(g,(size_t)n*8192);
    if(!x||!w){h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);return 0;}
    size_t nw=(size_t)n*8192;
    sglang_vision_patch_pack<<<(nw+255)/256,256,0,g->compute>>>((ushort*)tensor_pointer(w),(const ushort*)tensor_pointer(weight),nw);
    int ok=launch_status(g,"reference vision patch packing");
    cublasStatus_t status=g->sglang_blas->math(g->sglang_blas->handle,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION);
    if(status!=CUBLAS_STATUS_SUCCESS)ok=h3_gpu_set_error(g,"reference vision reduction mode failed: %d",(int)status);
    for(uint32_t offset=0;ok&&offset<rows;) {
        uint32_t batch=std::min(chunk_rows,rows-offset);
        size_t ni=(size_t)batch*8192;
        sglang_vision_patch_pack<<<(ni+255)/256,256,0,g->compute>>>((ushort*)tensor_pointer(x),
            (const ushort*)tensor_pointer(in)+(size_t)offset*k,ni);
        ok=launch_status(g,"reference vision input packing") &&
            gemm(g,out,(size_t)offset*n,x,0,w,nullptr,batch,8192,n,H3_GPU_BF16);
        offset+=batch;
    }
    status=g->sglang_blas->math(g->sglang_blas->handle,CUBLAS_DEFAULT_MATH);
    if(status!=CUBLAS_STATUS_SUCCESS)ok=h3_gpu_set_error(g,"reference vision reduction restore failed: %d",(int)status);
    h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);if(!ok)return 0;
    size_t count=(size_t)rows*n;
    sglang_patch_bias<<<(count+255)/256,256,0,g->compute>>>((ushort*)tensor_pointer(out),(const ushort*)tensor_pointer(bias),count,n);
    return launch_status(g,"reference vision patch separate bias");
}
static int sglang_patch(h3_gpu*g,h3_gpu_tensor*o,size_t oo,const h3_gpu_tensor*i,size_t io,
    const h3_gpu_tensor*w,const h3_gpu_tensor*b,const h3_gpu_tensor*map,uint32_t m,uint32_t k,uint32_t n) {
    if(!m||!k||!n||!b||i->dtype!=H3_GPU_F32||w->dtype!=H3_GPU_F32||b->dtype!=H3_GPU_F32||b->elements<n)
        return h3_gpu_set_error(g,"reference patch projection requires nonempty FP32 operands and bias");
    /* The FP32 result is only a staging buffer before BF16 scatter. A full
     * 1344x768/362-frame result exceeds 2 GiB even without reference images.
     * Keep the 1 GiB bound by splitting independent rows, never the K reduction.
     * Align batch starts to preserve cuBLAS pointer alignment and choose the
     * algorithm with the original row count, as for the reference heads. */
    const size_t max_elements=size_t(1)<<28;
    uint32_t batch_rows=(uint32_t)std::min<size_t>(m,max_elements/n);
    if(!batch_rows)return h3_gpu_set_error(g,"reference patch projection row exceeds 1 GiB scratch");
    if(batch_rows<m && batch_rows>=128)batch_rows-=batch_rows%128;
    size_t count=(size_t)batch_rows*n;
    if(!g->sglang_patch||g->sglang_patch->elements<count) {
        if(g->sglang_patch&&!h3_gpu_synchronize(g))return 0;
        h3_gpu_tensor_free(g->sglang_patch);g->sglang_patch=nullptr;
        g->sglang_patch=h3_gpu_tensor_new_f32(g,count);if(!g->sglang_patch)return 0;
    }
    for(uint32_t row=0;row<m;) {
        uint32_t rows=std::min(batch_rows,m-row);
        cublasStatus_t status=g->sglang_blas->biased(tensor_pointer(g->sglang_patch),
            tensor_pointer(i,(io+(size_t)row*k)*4),tensor_pointer(w),tensor_pointer(b),
            rows,k,n,CUDA_R_32F,g->workspace,g->workspace_bytes,g->compute,batch_rows<m?m:0);
        if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,
            "reference patch cuBLASLt rows %u..%u of %u failed (%d)",row,row+rows,m,(int)status);
        count=(size_t)rows*n;
        h3_sg_patch_store<<<(count+255)/256,256,0,g->compute>>>((float*)tensor_pointer(g->sglang_patch),
            (ushort*)tensor_pointer(o,(oo+(map?0:(size_t)row*n))*2),
            (const uint*)tensor_pointer(map,(size_t)row*sizeof(uint)),rows,n);
        g->stats.linear_dispatches++;
        if(!launch_status(g,"reference FP32 patch projection and BF16 scatter"))return 0;
        row+=rows;
    }
    return 1;
}
int h3_gpu_patch_linear_bf16_offset(h3_gpu*g,h3_gpu_tensor*o,size_t oo,const h3_gpu_tensor*i,size_t io,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t m,uint32_t k,uint32_t n){
    if(!launch_ready(g)||!i||!w||!o||i->dtype!=H3_GPU_F32||w->dtype!=H3_GPU_F32||o->dtype!=H3_GPU_BF16||io>i->elements||(size_t)m*k>i->elements-io||oo>o->elements||(size_t)m*n>o->elements-oo||w->elements<(size_t)k*n||(b&&(b->dtype!=H3_GPU_F32||b->elements<n)))return 0;
    if(g->sglang_reference)return sglang_patch(g,o,oo,i,io,w,b,nullptr,m,k,n);
    linear_args a={m,k,n,b?1u:0u};
    h3_linear_f32_tiled_bf16<<<dim3((n+15)/16,(m+15)/16),dim3(16,16),0,g->compute>>>((float*)tensor_pointer(i,io*4),(float*)tensor_pointer(w),(float*)tensor_pointer(b),(ushort*)tensor_pointer(o,oo*2),a);
    g->stats.linear_dispatches++;return launch_status(g,"patch linear");
}
int h3_gpu_patch_linear_bf16(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t m,uint32_t k,uint32_t n){return h3_gpu_patch_linear_bf16_offset(g,o,0,i,0,w,b,m,k,n);}
int h3_gpu_patch_linear_bf16_map(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,const h3_gpu_tensor*map,uint32_t output_rows,uint32_t m,uint32_t k,uint32_t n){
    if(!launch_ready(g)||!map||map->dtype!=H3_GPU_U32||map->elements<m||!h3_gpu_require_bf16(g,o,(size_t)output_rows*n,"patch map output")||!h3_gpu_require_elements(g,i,(size_t)m*k,"patch map input")||!h3_gpu_require_elements(g,w,(size_t)n*k,"patch map weight"))return 0;
    if(g->sglang_reference)return sglang_patch(g,o,0,i,0,w,b,map,m,k,n);
    linear_args a={m,k,n,b?1u:0u};
    h3_linear_f32_tiled_bf16_map<<<dim3((n+15)/16,(m+15)/16),dim3(16,16),0,g->compute>>>((float*)tensor_pointer(i),(float*)tensor_pointer(w),(float*)tensor_pointer(b),(ushort*)tensor_pointer(o),a,(uint*)tensor_pointer(map));
    g->stats.linear_dispatches++;return launch_status(g,"mapped patch linear");
}
int h3_gpu_mlp_bf16(h3_gpu*g,h3_gpu_tensor*out,const h3_gpu_tensor*in,const h3_gpu_tensor*w1,const h3_gpu_tensor*w2,uint32_t rows,uint32_t width,uint32_t hidden,uint32_t output_dim){
    auto*fused=h3_gpu_tensor_new_bf16(g,(size_t)rows*hidden*2);auto*activated=h3_gpu_tensor_new_bf16(g,(size_t)rows*hidden);
    int ok=fused&&activated&&h3_gpu_linear_bf16(g,fused,in,w1,nullptr,rows,width,hidden*2)&&h3_gpu_swiglu_bf16(g,activated,fused,rows,hidden)&&h3_gpu_linear_bf16(g,out,activated,w2,nullptr,rows,hidden,output_dim);
    h3_gpu_tensor_free(fused);h3_gpu_tensor_free(activated);return ok;
}
int h3_gpu_adaln_bf16(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*m,const h3_gpu_tensor*r,uint32_t rows,uint32_t width,uint32_t slots,uint32_t shift,uint32_t scale,float eps){return h3_gpu_adaln_bf16_offset(g,o,i,0,w,m,r,rows,width,slots,shift,scale,eps);}
int h3_gpu_adaln_linear_bf16(h3_gpu*g,h3_gpu_tensor*o,h3_gpu_tensor*inverse,const h3_gpu_tensor*i,size_t offset,const h3_gpu_tensor*norm,const h3_gpu_tensor*mod,const h3_gpu_tensor*map,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t rows,uint32_t width,uint32_t n,uint32_t slots,uint32_t shift,uint32_t scale,float eps){
    (void)inverse;auto*tmp=h3_gpu_tensor_new_bf16(g,(size_t)rows*width);
    int ok=tmp&&h3_gpu_adaln_bf16_offset(g,tmp,i,offset,norm,mod,map,rows,width,slots,shift,scale,eps)&&h3_gpu_linear_bf16(g,o,tmp,w,b,rows,width,n);h3_gpu_tensor_free(tmp);return ok;
}
int h3_gpu_gate_adaln_bf16(h3_gpu*g,h3_gpu_tensor*res,h3_gpu_tensor*out,const h3_gpu_tensor*old,const h3_gpu_tensor*branch,const h3_gpu_tensor*norm,const h3_gpu_tensor*gate,const h3_gpu_tensor*mod,const h3_gpu_tensor*map,uint32_t rows,uint32_t width,uint32_t slots,uint32_t gate_slot,uint32_t shift,uint32_t scale,float eps){
    return h3_gpu_gate_bf16(g,res,old,branch,gate,map,rows,width,slots,gate_slot)&&h3_gpu_adaln_bf16(g,out,res,norm,mod,map,rows,width,slots,shift,scale,eps);
}
static int qkv_rope(h3_gpu*g,h3_gpu_tensor*q,h3_gpu_tensor*k,h3_gpu_tensor*v,const h3_gpu_tensor*input,const h3_gpu_tensor*qw,const h3_gpu_tensor*kw,const h3_gpu_tensor*c,const h3_gpu_tensor*s,uint32_t seq,uint32_t heads,uint32_t dim,uint32_t half,float eps,bool grouped){
    size_t count=(size_t)seq*heads*dim;
    if(!launch_ready(g)||!seq||!heads||!dim||half*2>dim||!h3_gpu_require_bf16(g,input,count*3,"QKV")||!h3_gpu_require_bf16(g,q,count,"Q")||!h3_gpu_require_bf16(g,k,count,"K")||!h3_gpu_require_bf16(g,v,count,"V")||!h3_gpu_require_bf16(g,qw,dim,"Q norm")||!h3_gpu_require_bf16(g,kw,dim,"K norm")||!h3_gpu_require_bf16(g,c,(size_t)seq*half,"RoPE cos")||!h3_gpu_require_bf16(g,s,(size_t)seq*half,"RoPE sin"))return 0;
    qkv_args a={seq,heads,dim,half,grouped?1u:0u,eps};
    if(g->sglang_reference && dim==128) {
        profile_scope timing(g,6);
        h3_sg_qkv_rope<<<((size_t)seq*heads+3)/4,128,0,g->compute>>>((ushort*)tensor_pointer(input),(ushort*)tensor_pointer(qw),(ushort*)tensor_pointer(kw),(ushort*)tensor_pointer(c),(ushort*)tensor_pointer(s),(ushort*)tensor_pointer(q),(ushort*)tensor_pointer(k),(ushort*)tensor_pointer(v),a);
        return launch_status(g,"SGLang QKV norm/RoPE rounding");
    }
    h3_qkv_rope_bf16<<<dim3((dim+7)/8,(heads+3)/4,(seq+3)/4),dim3(8,4,4),0,g->compute>>>((ushort*)tensor_pointer(input),(ushort*)tensor_pointer(qw),(ushort*)tensor_pointer(kw),(ushort*)tensor_pointer(c),(ushort*)tensor_pointer(s),(ushort*)tensor_pointer(q),(ushort*)tensor_pointer(k),(ushort*)tensor_pointer(v),a);
    return launch_status(g,"QKV norm/RoPE");
}
#define QKV_API(name,grouped) int name(h3_gpu*g,h3_gpu_tensor*q,h3_gpu_tensor*k,h3_gpu_tensor*v,const h3_gpu_tensor*i,const h3_gpu_tensor*qw,const h3_gpu_tensor*kw,const h3_gpu_tensor*c,const h3_gpu_tensor*s,uint32_t seq,uint32_t heads,uint32_t dim,uint32_t half,float eps){return qkv_rope(g,q,k,v,i,qw,kw,c,s,seq,heads,dim,half,eps,grouped);}
QKV_API(h3_gpu_qkv_rope_bf16,false)
QKV_API(h3_gpu_grouped_qkv_rope_bf16,true)
int h3_gpu_grouped_qkv_linear_rope_bf16(h3_gpu*g,h3_gpu_tensor*q,h3_gpu_tensor*k,h3_gpu_tensor*v,h3_gpu_tensor*qkv,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*qw,const h3_gpu_tensor*kw,const h3_gpu_tensor*c,const h3_gpu_tensor*s,uint32_t rows,uint32_t width,uint32_t heads,uint32_t dim,uint32_t half,float eps){return h3_gpu_linear_bf16(g,qkv,i,w,nullptr,rows,width,heads*dim*3)&&h3_gpu_grouped_qkv_rope_bf16(g,q,k,v,qkv,qw,kw,c,s,rows,heads,dim,half,eps);}

#include "src/cuda/cuda_attention.cuh"
/* Soundtrack encoder only: bounded PyTorch math-SDPA, with separately scaled
 * FP32 operands, NN/TN batched GEMMs and causal softmax. Video attention
 * never uses these buffers or this dispatch. */
static int sglang_audio_attention(h3_gpu *g,float *out,const float *q,const float *k,const float *v,
                                  unsigned batch,unsigned seq,unsigned heads,unsigned dim,float scale) {
    if(!seq||seq>600||heads!=8||dim!=256||batch!=2||scale<=0||!g->sglang_blas)
        return h3_gpu_set_error(g,"reference audio math attention exceeds its bounded shape");
    size_t count=(size_t)batch*seq*heads*dim,scores=(size_t)batch*heads*seq*seq;
    if(count>(64u<<20)/16||scores>(64u<<20)/4)
        return h3_gpu_set_error(g,"reference audio math attention exceeds 64 MiB scratch per buffer");
    auto reserve=[&](h3_gpu_tensor **buffer,size_t n) {
        if(*buffer&&(*buffer)->elements>=n)return true;
        if(*buffer&&!h3_gpu_synchronize(g))return false;
        h3_gpu_tensor_free(*buffer);*buffer=h3_gpu_tensor_new_f32(g,n);return *buffer!=nullptr;
    };
    if(!reserve(&g->sglang_audio_math,4*count)||!reserve(&g->sglang_audio_scores,scores))return 0;
    float *packed=(float*)tensor_pointer(g->sglang_audio_math),*prob=(float*)tensor_pointer(g->sglang_audio_scores);
    h3_sg_audio_math_pack<<<(count+255)/256,256,0,g->compute>>>(packed,q,k,v,seq,heads,dim,count,sqrtf(scale));
    if(!launch_status(g,"reference audio attention packing"))return 0;
    float one=1,zero=0;
    auto status=g->sglang_blas->sgemm_batched(g->sglang_blas->handle,CUBLAS_OP_T,CUBLAS_OP_N,
        seq,seq,dim,&one,packed+count,dim,(long long)seq*dim,packed,dim,(long long)seq*dim,
        &zero,prob,seq,(long long)seq*seq,batch*heads);
    if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,"reference audio QK GEMM failed: %d",(int)status);
    h3_sg_audio_math_softmax<<<batch*heads*seq,32,0,g->compute>>>(prob,seq);
    if(!launch_status(g,"reference audio causal softmax"))return 0;
    status=g->sglang_blas->sgemm_batched(g->sglang_blas->handle,CUBLAS_OP_N,CUBLAS_OP_N,
        dim,seq,seq,&one,packed+count*2,dim,(long long)seq*dim,prob,seq,(long long)seq*seq,
        &zero,packed+count*3,dim,(long long)seq*dim,batch*heads);
    if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,"reference audio PV GEMM failed: %d",(int)status);
    h3_sg_audio_math_unpack<<<(count+255)/256,256,0,g->compute>>>(out,packed+count*3,seq,heads,dim,count);
    g->stats.attention_dispatches++;return launch_status(g,"reference audio math attention");
}
static int attention(h3_gpu*g,h3_gpu_tensor*out,const h3_gpu_tensor*q,const h3_gpu_tensor*k,const h3_gpu_tensor*v,uint32_t batch,uint32_t seq,uint32_t heads,uint32_t kv,uint32_t dim,float scale,h3_gpu_dtype dtype,bool causal,bool head_major,int mode=0){
    size_t qn=(size_t)batch*seq*heads*dim,kn=(size_t)batch*seq*kv*dim;
    if(!launch_ready(g)||!batch||batch>65535||!seq||seq>INT32_MAX||!heads||heads>65535||!kv||heads%kv||!dim||dim>256||!std::isfinite(scale)||!h3_gpu_require_elements(g,q,qn,"attention Q")||!h3_gpu_require_elements(g,k,kn,"attention K")||!h3_gpu_require_elements(g,v,kn,"attention V")||!h3_gpu_require_elements(g,out,qn,"attention output")||q->dtype!=dtype||k->dtype!=dtype||v->dtype!=dtype||out->dtype!=dtype)return h3_gpu_set_error(g,"attention: invalid batch/sequence/heads/dimension/type (%u,%u,%u,%u)",batch,seq,heads,dim);
    /* MPSGraph's BF16 SDPA casts its scale constant to the operand dtype.
     * Qwen causal GQA is a different contract and keeps its explicit F32 scale.
     * Softmax statistics and value accumulation remain F32 in both cases. */
    if(dtype==H3_GPU_BF16 && !causal && !g->sglang_reference) {
        uint32_t bits;memcpy(&bits,&scale,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000u;memcpy(&scale,&bits,4);
    }
    void *op=tensor_pointer(out),*qp=tensor_pointer(q),*kp=tensor_pointer(k),*vp=tensor_pointer(v);
    profile_scope timing(g,1);
    if(g->sglang_audio_encoder&&dtype==H3_GPU_F32&&causal&&kv==heads&&!head_major)
        return sglang_audio_attention(g,(float*)op,(float*)qp,(float*)kp,(float*)vp,batch,seq,heads,dim,scale);
    bool first_shape=g->profiling && g->attention_shapes.emplace((int)dtype,batch,seq,heads,dim,(int)head_major+2*(int)causal).second;
    if(g->profiling && first_shape)
        fprintf(stderr,"h3cli: attention shape phase=%s mode=%s dtype=%d B=%u S=%u H=%u KV=%u D=%u causal=%d head_major=%d\n",g->label.c_str(),"single",(int)dtype,batch,seq,heads,kv,dim,(int)causal,(int)head_major);
#ifdef H3_CUDA_USE_SGLANG_FLASH
    if(g->sglang_reference && dtype==H3_GPU_BF16 && (dim==128||dim==72)) {
        size_t count=(size_t)batch*seq*heads;
        if(count>g->attention_budget/sizeof(float))return h3_gpu_set_error(g,"reference attention LSE exceeds bounded scratch budget");
        if(!g->sglang_lse || g->sglang_lse->elements<count) {
            /* The previous launch may still reference the old allocation. */
            if(g->sglang_lse && !h3_gpu_synchronize(g))return 0;
            h3_gpu_tensor_free(g->sglang_lse);g->sglang_lse=nullptr;
            g->sglang_lse=h3_gpu_tensor_new_f32(g,count);
            if(!g->sglang_lse)return 0;
        }
        if(!g->sglang_vision_groups.empty()) {
            size_t total=0;for(unsigned n:g->sglang_vision_groups)total+=n;
            if(total!=seq||batch!=1||head_major||causal||dim!=72||heads!=kv)return h3_gpu_set_error(g,"invalid packed reference vision attention");
            size_t offset=0;
            for(unsigned n:g->sglang_vision_groups) {
                size_t elements=offset*heads*dim;
                int status=h3_sglang_flash((ushort*)op+elements,(ushort*)qp+elements,(ushort*)kp+elements,(ushort*)vp+elements,
                    (float*)tensor_pointer(g->sglang_lse),1,n,heads,kv,dim,scale,0,0,g->compute,g->error,sizeof(g->error));
                if(status<=0){g->deferred_error=true;return 0;}
                offset+=n;g->stats.attention_dispatches++;
            }
            return launch_status(g,"reference packed vision FlashAttention");
        }
        int status=h3_sglang_flash(op,qp,kp,vp,(float*)tensor_pointer(g->sglang_lse),
            batch,seq,heads,kv,dim,scale,causal,head_major,g->compute,g->error,sizeof(g->error));
        if(status<0){g->deferred_error=true;return 0;}
        if(status>0){g->stats.attention_dispatches++;return launch_status(g,"reference pinned FlashAttention");}
        return h3_gpu_set_error(g,"unsupported reference FlashAttention shape");
    }
#endif
    if(dtype==H3_GPU_BF16 && !causal && seq>=512 && dim%16==0) {
        if(g->dispatch.attention==H3_CUDA_SM90_ATTENTION && dim==128)
            attention_bf16_fixed128<512><<<dim3((seq+15)/16,heads,batch),512,0,g->compute>>>((ushort*)op,(ushort*)qp,(ushort*)kp,(ushort*)vp,seq,heads,scale,head_major);
        else if(g->dispatch.attention==H3_CUDA_SM120_ATTENTION && dim==128)
            attention_bf16_fixed128<128><<<dim3((seq+15)/16,heads,batch),128,0,g->compute>>>((ushort*)op,(ushort*)qp,(ushort*)kp,(ushort*)vp,seq,heads,scale,head_major);
        else
            attention_bf16_tiled<<<dim3((seq+15)/16,heads,batch),128,0,g->compute>>>((ushort*)op,(ushort*)qp,(ushort*)kp,(ushort*)vp,seq,heads,dim,scale,head_major);
    }
    else if(dtype==H3_GPU_BF16)attention_online<<<dim3(seq,heads,batch),128,0,g->compute>>>((ushort*)op,(ushort*)qp,(ushort*)kp,(ushort*)vp,seq,heads,kv,dim,scale,causal,head_major,mode);
    else attention_online<<<dim3(seq,heads,batch),128,0,g->compute>>>((float*)op,(float*)qp,(float*)kp,(float*)vp,seq,heads,kv,dim,scale,causal,head_major,mode);
    g->stats.attention_dispatches++;return launch_status(g,"online softmax attention");
}
int h3_gpu_sdpa_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*q,const h3_gpu_tensor*k,const h3_gpu_tensor*v,uint32_t seq,uint32_t heads,uint32_t dim,float scale){return attention(g,o,q,k,v,1,seq,heads,heads,dim,scale,H3_GPU_F32,false,false);}
int h3_gpu_sdpa_bf16(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*q,const h3_gpu_tensor*k,const h3_gpu_tensor*v,uint32_t seq,uint32_t heads,uint32_t dim,float scale){return attention(g,o,q,k,v,1,seq,heads,heads,dim,scale,H3_GPU_BF16,false,false);}
int h3_gpu_sdpa_bf16_head_major_output(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*q,const h3_gpu_tensor*k,const h3_gpu_tensor*v,uint32_t seq,uint32_t heads,uint32_t dim,float scale){return attention(g,o,q,k,v,1,seq,heads,heads,dim,scale,H3_GPU_BF16,false,true);}
int h3_gpu_sdpa_causal_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*q,const h3_gpu_tensor*k,const h3_gpu_tensor*v,uint32_t batch,uint32_t seq,uint32_t heads,uint32_t dim,float scale){return attention(g,o,q,k,v,batch,seq,heads,heads,dim,scale,H3_GPU_F32,true,false);}
int h3_qwen_gqa_scale_parse(const char*value,h3_qwen_gqa_scale_mode*mode,char*error,size_t size){
    if(!mode)return 0;
    if(!value||!*value||!strcmp(value,"reference"))*mode=H3_QWEN_GQA_REFERENCE;
    else if(!strcmp(value,"scaled-q"))*mode=H3_QWEN_GQA_SCALED_Q;
    else if(!strcmp(value,"legacy"))*mode=H3_QWEN_GQA_LEGACY;
    else {if(error&&size)snprintf(error,size,"invalid H3_QWEN_GQA_SCALE_MODE: %s",value);return 0;}
    return 1;
}
const char *h3_qwen_gqa_scale_name(h3_qwen_gqa_scale_mode mode){switch(mode){case H3_QWEN_GQA_REFERENCE:return "reference";case H3_QWEN_GQA_SCALED_Q:return "scaled-q";case H3_QWEN_GQA_LEGACY:return "legacy";}return "unknown";}
int h3_gpu_gqa_causal_limits(h3_gpu*g,h3_gqa_limits*limits){if(!g||!limits)return 0;cudaDeviceProp prop;if(!checked(g,cudaGetDeviceProperties(&prop,g->device),"attention limits"))return 0;limits->device_bytes=prop.sharedMemPerBlock;limits->static_bytes=16*sizeof(float)+12;limits->alignment_bytes=4;limits->max_sequence=INT32_MAX;limits->direct_max_sequence=INT32_MAX;return 1;}
size_t h3_gpu_gqa_causal_max_sequence(h3_gpu*g){return g?INT32_MAX:0;}
int h3_gpu_gqa_causal_preflight(h3_gpu*g,size_t seq,char*error,size_t size){
    h3_qwen_gqa_scale_mode mode;
    if(!h3_qwen_gqa_scale_parse(getenv("H3_QWEN_GQA_SCALE_MODE"),&mode,error,size))return 0;
    if(getenv("H3_MPS_GQA")){if(error&&size)snprintf(error,size,"H3_MPS_GQA is a Metal-only option; unset it for CUDA");return 0;}
    if(g&&seq&&seq<=INT32_MAX)return 1;
    if(error&&size)snprintf(error,size,"invalid CUDA GQA sequence: %zu",seq);return 0;
}

int h3_gpu_gqa_causal_bf16(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*q,const h3_gpu_tensor*k,const h3_gpu_tensor*v,uint32_t seq,uint32_t heads,uint32_t kv,uint32_t dim,float scale){h3_qwen_gqa_scale_mode mode;if(!h3_qwen_gqa_scale_parse(getenv("H3_QWEN_GQA_SCALE_MODE"),&mode,g->error,sizeof(g->error)))return 0;return attention(g,o,q,k,v,1,seq,heads,kv,dim,scale,H3_GPU_BF16,true,false,(int)mode);}

#include "src/cuda/cuda_conv.cuh"
#include "src/cuda/cuda_sglang_convolution.cuh"
static int conv1d(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t batch,uint32_t length,uint32_t ci,uint32_t co,uint32_t kernel,uint32_t stride,uint32_t padding,uint32_t dilation,bool transpose){
    profile_scope timing(g,2);
    if(!batch||!length||!ci||!co||!kernel||!stride||!dilation||!launch_ready(g))return 0;
    int64_t length_out=transpose?(int64_t)(length-1)*stride-2*(int64_t)padding+kernel:((int64_t)length+2*(int64_t)padding-(int64_t)dilation*(kernel-1)-1)/(int64_t)stride+1;
    if(length_out<=0||length_out>UINT32_MAX)return 0;uint32_t n=(uint32_t)length_out;
    if(!h3_gpu_require_elements(g,o,(size_t)batch*n*co,"Conv1D output")||!h3_gpu_require_elements(g,i,(size_t)batch*length*ci,"Conv1D input")||!h3_gpu_require_elements(g,w,(size_t)ci*co*kernel,"Conv1D weights")||(b&&!h3_gpu_require_elements(g,b,co,"Conv1D bias")))return 0;
    size_t count=(size_t)batch*n*co;
    if(g->sglang_audio_encoder)return sg_audio_encoder_conv(g,o,i,w,b,batch,length,ci,co,kernel,stride,padding,dilation,n,transpose);
    if(g->sglang_audio)return sg_audio_conv(g,o,i,w,b,batch,length,ci,co,kernel,stride,padding,dilation,n,transpose);
    conv1d_kernel<<<(unsigned)((count+255)/256),256,0,g->compute>>>((float*)tensor_pointer(o),(float*)tensor_pointer(i),(float*)tensor_pointer(w),(float*)tensor_pointer(b),batch,length,ci,co,kernel,stride,padding,dilation,n,transpose);
    g->stats.conv_dispatches++;return launch_status(g,"Conv1D");
}
int h3_gpu_conv1d_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t batch,uint32_t length,uint32_t ci,uint32_t co,uint32_t kernel,uint32_t padding,uint32_t dilation){return conv1d(g,o,i,w,b,batch,length,ci,co,kernel,1,padding,dilation,false);}
int h3_gpu_conv1d_stride_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t batch,uint32_t length,uint32_t ci,uint32_t co,uint32_t kernel,uint32_t stride,uint32_t padding,uint32_t dilation){return conv1d(g,o,i,w,b,batch,length,ci,co,kernel,stride,padding,dilation,false);}
int h3_gpu_conv_transpose1d_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t batch,uint32_t length,uint32_t ci,uint32_t co,uint32_t kernel,uint32_t stride,uint32_t padding){return conv1d(g,o,i,w,b,batch,length,ci,co,kernel,stride,padding,1,true);}
int h3_gpu_conv3d_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t batch,uint32_t d,uint32_t h,uint32_t width,uint32_t ci,uint32_t co,uint32_t kd,uint32_t kh,uint32_t kw,uint32_t sd,uint32_t sh,uint32_t sw){
    profile_scope timing(g,2);
    if(!launch_ready(g)||!batch||!ci||!co||!kd||!kh||!kw||!sd||!sh||!sw||d<kd||h<kh||width<kw)return 0;
    uint32_t od=(d-kd)/sd+1,oh=(h-kh)/sh+1,ow=(width-kw)/sw+1;size_t count=(size_t)batch*od*oh*ow*co;
    if(!h3_gpu_require_elements(g,o,count,"Conv3D output")||!h3_gpu_require_elements(g,i,(size_t)batch*d*h*width*ci,"Conv3D input")||!h3_gpu_require_elements(g,w,(size_t)co*ci*kd*kh*kw,"Conv3D weights")||(b&&!h3_gpu_require_elements(g,b,co,"Conv3D bias")))return 0;
    if(o->dtype!=H3_GPU_F32||i->dtype!=H3_GPU_F32||w->dtype!=H3_GPU_F32||(b&&b->dtype!=H3_GPU_F32))return h3_gpu_set_error(g,"Conv3D requires F32 tensors");
    if(g->sglang_encoder){
        unsigned shape[12]={batch,d,h,width,ci,co,kd,kh,kw,sd,sh,sw};
        return sg_encoder_conv(g,o,i,w,b,shape,od*oh*ow);
    }
    size_t positions=count/co,filter_size=(size_t)ci*kd*kh*kw;
    /* Small shapes are faster in the direct kernel. The measured large VAE
     * shapes amortize packing/SGEMM dispatch; cap packing scratch at 32 MiB. */
    if(co>=16 && filter_size>=16 && positions>=32 &&
       filter_size<=(32u<<20)/sizeof(float) &&
       count>=((UINT64_C(1)<<28)+filter_size-1)/filter_size) {
        uint32_t filter=(uint32_t)filter_size;
        uint32_t tile=(uint32_t)std::min<size_t>(1024,std::max<size_t>(1,(32u<<20)/(filter_size*4)));
        bool reuse=!getenv("H3_TEST_VAE_CONV_NO_REUSE");
        size_t column_count=(size_t)tile*filter;
        if(reuse && (!g->conv_columns || g->conv_columns->elements<column_count)) {
            h3_gpu_tensor_free(g->conv_columns);g->conv_columns=h3_gpu_tensor_new_f32(g,column_count);
        }
        auto *columns=reuse?g->conv_columns:h3_gpu_tensor_new_f32(g,column_count);if(!columns)return 0;
        int ok=1;g->convolution_gemm=true;
        for(size_t first=0;ok&&first<positions;first+=tile) {
            uint32_t rows=(uint32_t)std::min<size_t>(tile,positions-first);
            conv3d_columns<<<(unsigned)(((size_t)rows*filter+255)/256),256,0,g->compute>>>(
                (float*)tensor_pointer(columns),(float*)tensor_pointer(i),first,rows,
                d,h,width,ci,kd,kh,kw,sd,sh,sw,od,oh,ow,filter);
            ok=launch_status(g,"Conv3D im2col") && gemm(g,o,first*co,columns,0,w,b,rows,filter,co,H3_GPU_F32);
        }
        g->convolution_gemm=false;if(!reuse)h3_gpu_tensor_free(columns);
        g->stats.conv_dispatches++;return ok;
    }
    conv3d_kernel<<<(unsigned)((count+255)/256),256,0,g->compute>>>((float*)tensor_pointer(o),(float*)tensor_pointer(i),(float*)tensor_pointer(w),(float*)tensor_pointer(b),batch,d,h,width,ci,co,kd,kh,kw,sd,sh,sw,od,oh,ow);
    g->stats.conv_dispatches++;return launch_status(g,"Conv3D");
}
int h3_gpu_vae_encoder_group_norm_silu_f32(h3_gpu*g,h3_gpu_tensor*o,const h3_gpu_tensor*i,const h3_gpu_tensor*w,const h3_gpu_tensor*b,uint32_t batch,uint32_t d,uint32_t h,uint32_t width,uint32_t c,uint32_t groups,float eps){
    size_t n=(size_t)batch*d*h*width*c;
    if(!launch_ready(g)||!groups||c%groups||!h3_gpu_require_elements(g,o,n,"group norm output")||!h3_gpu_require_elements(g,i,n,"group norm input")||!h3_gpu_require_elements(g,w,c,"group norm weight")||!h3_gpu_require_elements(g,b,c,"group norm bias"))return 0;
    vae_encoder_norm_args args={batch,d,h,width,c,groups,eps};
    if(g->sglang_encoder){
        unsigned threads=(size_t)h*width*(c/groups)<512?32:512;
        h3_sg_encoder_norm<<<groups*d*batch,threads,0,g->compute>>>((float*)tensor_pointer(i),(float*)tensor_pointer(w),(float*)tensor_pointer(b),(float*)tensor_pointer(o),args);
        return launch_status(g,"reference encoder GroupNorm/SiLU");
    }
    h3_vae_encoder_group_norm_silu_f32<<<dim3(groups*d*batch),256,0,g->compute>>>((float*)tensor_pointer(i),(float*)tensor_pointer(w),(float*)tensor_pointer(b),(float*)tensor_pointer(o),args);
    return launch_status(g,"VAE group norm SiLU");
}
int h3_gpu_token_pool_adaln_bf16(h3_gpu*g,h3_gpu_tensor*res,h3_gpu_tensor*out,const h3_gpu_tensor*in,size_t io,h3_gpu_tensor*original,size_t oo,h3_gpu_tensor*baseline,size_t bo,const h3_gpu_tensor*indices,const h3_gpu_tensor*pairs,const h3_gpu_tensor*norm,const h3_gpu_tensor*mod,const h3_gpu_tensor*map,uint32_t input_rows,uint32_t rows,uint32_t base_rows,uint32_t width,uint32_t slots,uint32_t shift,uint32_t scale,float eps){
    return h3_gpu_token_pool_bf16(g,res,in,io,original,oo,baseline,bo,indices,pairs,input_rows,rows,base_rows,width)&&h3_gpu_adaln_bf16(g,out,res,norm,mod,map,rows,width,slots,shift,scale,eps);
}
int h3_gpu_token_expand_adaln_bf16(h3_gpu*g,h3_gpu_tensor*res,h3_gpu_tensor*out,const h3_gpu_tensor*original,size_t oo,const h3_gpu_tensor*reduced,const h3_gpu_tensor*baseline,size_t bo,const h3_gpu_tensor*indices,const h3_gpu_tensor*parents,const h3_gpu_tensor*norm,const h3_gpu_tensor*mod,const h3_gpu_tensor*map,uint32_t rows,uint32_t reduced_rows,uint32_t base_rows,uint32_t width,uint32_t prefix,float update,uint32_t slots,uint32_t shift,uint32_t scale,float eps){
    return h3_gpu_token_expand_delta_bf16(g,res,original,oo,reduced,baseline,bo,indices,parents,rows,reduced_rows,base_rows,width,prefix,update)&&h3_gpu_adaln_bf16(g,out,res,norm,mod,map,rows,width,slots,shift,scale,eps);
}
int h3_gpu_video_preview_bf16(h3_gpu*g,h3_gpu_tensor*scratch,const h3_gpu_tensor*sample,size_t offset,const h3_gpu_tensor*last,const h3_gpu_tensor*previous,const h3_gpu_tensor*rows,const h3_gpu_tensor*strengths,uint32_t time,uint32_t height,uint32_t width,float sigma,float ratio){
    if(!launch_ready(g)||!time||height<2||width<2||height%2||width%2||(uint64_t)time*height>UINT32_MAX/24u/width||!std::isfinite(sigma)||sigma<0||sigma>1||!std::isfinite(ratio))return 0;
    uint32_t n=24u*time*height*width;
    if(!scratch||scratch->bytes/sizeof(float)<n||!sample||sample->dtype!=H3_GPU_F32||offset>UINT32_MAX||n>UINT32_MAX-offset||offset>sample->elements||n>sample->elements-offset||!h3_gpu_require_bf16(g,last,n,"preview last")||!h3_gpu_require_bf16(g,previous,n,"preview previous")||(!!rows!=!!strengths)||(rows&&!bridge_ranges(sample,offset,rows,strengths,n/96,96)))return 0;
    const h3_gpu_tensor*inputs[]={sample,last,previous,rows,strengths};for(auto*t:inputs)if(scratch==t)return 0;
    h3_video_preview_args a={(uint32_t)offset,n,time,height,width,rows?1u:0u,strengths?(uint32_t)strengths->elements:0u,sigma,ratio};
    h3_video_preview_bf16<<<(n+255)/256,256,0,g->compute>>>((float*)tensor_pointer(scratch),(float*)tensor_pointer(sample),(ushort*)tensor_pointer(last),(ushort*)tensor_pointer(previous),(uint*)tensor_pointer(rows),(float*)tensor_pointer(strengths),a);return launch_status(g,"video preview");
}
int h3_gpu_video_preview_read(const h3_gpu_tensor*t,float*out,size_t n){
    if(!t||!out||n>t->bytes/4||!h3_gpu_synchronize(t->owner))return 0;
    int ok=checked(t->owner,cudaMemcpy(out,t->data,n*4,cudaMemcpyDeviceToHost),"preview readback",n*4);if(ok)t->owner->stats.d2h_bytes+=n*4;return ok;
}

int h3_gpu_mlp_nax_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    (void)output;
    (void)activated;
    (void)input;
    (void)fc1_weight;
    (void)fc2_weight;
    (void)rows;
    (void)input_dim;
    (void)hidden_dim;
    (void)output_dim;
    return h3_gpu_set_error(gpu, "h3_gpu_mlp_nax_bf16: Metal-only path; use original BF16 CUDA weights");
}

int h3_gpu_quantize_weight_int8(h3_gpu *gpu, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    (void)output;
    (void)scales;
    (void)input;
    (void)rows;
    (void)columns;
    return h3_gpu_set_error(gpu, "h3_gpu_quantize_weight_int8: Metal-only path; use original BF16 CUDA weights");
}

int h3_gpu_linear_int8_bf16(h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    (void)output;
    (void)quantized_input;
    (void)input_scales;
    (void)input;
    (void)weight;
    (void)weight_scales;
    (void)rows;
    (void)input_dim;
    (void)output_dim;
    (void)use_slower_uncached_int8_scales;
    return h3_gpu_set_error(gpu, "h3_gpu_linear_int8_bf16: Metal-only path; use original BF16 CUDA weights");
}

int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *gpu, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) {
    (void)output;
    (void)quantized_input;
    (void)input_scales;
    (void)input;
    (void)weight;
    (void)weight_scales;
    (void)rows;
    (void)heads;
    (void)head_dim;
    (void)output_dim;
    return h3_gpu_set_error(gpu, "h3_gpu_linear_int8_head_major_bf16: Metal-only path; use original BF16 CUDA weights");
}

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
                         int input_is_quantized) {
    (void)output;
    (void)activated;
    (void)quantized_activation;
    (void)activation_scales;
    (void)input;
    (void)fc1_weight;
    (void)fc1_scales;
    (void)fc2_weight;
    (void)fc2_scales;
    (void)fc1_bf16;
    (void)fc2_bf16;
    (void)rows;
    (void)input_dim;
    (void)hidden_dim;
    (void)output_dim;
    (void)use_slower_grouped_quantizer;
    (void)use_slower_dynamic_fc1_k;
    (void)use_int8_row_fc2;
    (void)input_is_quantized;
    return h3_gpu_set_error(gpu, "h3_gpu_mlp_int8_bf16: Metal-only path; use original BF16 CUDA weights");
}

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
                     uint32_t scale_slot, float epsilon) {
    (void)gated_residual;
    (void)quantized_output;
    (void)quantized_scales;
    (void)residual;
    (void)branch;
    (void)norm_weight;
    (void)gate_modulation;
    (void)norm_modulation;
    (void)row_map;
    (void)rows;
    (void)padded_rows;
    (void)width;
    (void)slots;
    (void)gate_slot;
    (void)shift_slot;
    (void)scale_slot;
    (void)epsilon;
    return h3_gpu_set_error(gpu, "h3_gpu_gate_adaln_quantize_int8: Metal-only path; use original BF16 CUDA weights");
}

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
                                 int use_slower_uncached_int8_scales) {
    (void)query;
    (void)key;
    (void)value;
    (void)quantized_input;
    (void)input_scales;
    (void)input;
    (void)weight;
    (void)weight_scales;
    (void)q_norm;
    (void)k_norm;
    (void)rope_cos;
    (void)rope_sin;
    (void)rows;
    (void)input_dim;
    (void)heads;
    (void)head_dim;
    (void)rope_half;
    (void)epsilon;
    (void)input_is_quantized;
    (void)use_slower_unfused_qkv_rope;
    (void)use_slower_scalar_qkv_rms;
    (void)use_slower_uncached_int8_scales;
    return h3_gpu_set_error(gpu, "h3_gpu_grouped_qkv_linear_rope_int8: Metal-only path; use original BF16 CUDA weights");
}

#include "src/cuda/cuda_attention_capture.cuh"
#include "src/cuda/cuda_adaptive_cache.cuh"
int h3_gpu_dit_attention_configure(h3_gpu *g,int mode) {
    if(!g)return 0;
    if(g->dit_attention_configured)return h3_gpu_set_error(g,"attention context is already configured");
    if(g->policy.active && mode!=g->policy.attention)
        return h3_gpu_set_error(g,"attention differs from captured CUDA request policy");
    if(!h3_attention_preflight(mode,g->error,sizeof(g->error)))return 0;
    if(!mode){g->dit_attention_configured=true;return 1;}
    if(mode==H3_ATTENTION_SUBBLOCK) {
        g->subblock_workspace=h3_gpu_tensor_alloc(g,H3_ATTENTION_WORKSPACE_BYTES,H3_GPU_I8,H3_GPU_DEVICE_ONLY);
        if(!g->subblock_workspace)return 0;
        g->subblock=h3_cuda_subblock_create(g->subblock_workspace->data,H3_ATTENTION_WORKSPACE_BYTES,g->compute,g->profiling||g->experiment_timing,g->error,sizeof(g->error));
        if(!g->subblock){h3_gpu_tensor_free(g->subblock_workspace);g->subblock_workspace=nullptr;return 0;}
        g->attention_mode=mode;g->dit_attention_configured=true;
        fprintf(stderr,"h3cli: DiT attention=subblock recipe=%u plan=1 sparsity=%.9g warmup=%d probe=dense workspace_limit=%zu\n",h3_attention_execution_recipe(mode,g->policy.projection_precision),g->policy.subblock_sparsity,h3_subblock_warmup(g->policy.subblock_warmup),H3_ATTENTION_WORKSPACE_BYTES);
        return 1;
    }
    if(mode==H3_ATTENTION_SOL) {
        g->sol_options=h3_cuda_sol_current();
        if(!h3_cuda_sol_options_valid(g->sol_options,g->error,sizeof(g->error)))return 0;
        g->sol_workspace=h3_gpu_tensor_alloc(g,H3_CUDA_SOL_WORKSPACE_BYTES,H3_GPU_I8,H3_GPU_DEVICE_ONLY);
        if(!g->sol_workspace)return 0;
        g->sol=h3_cuda_sol_create(g->sol_workspace->data,H3_CUDA_SOL_WORKSPACE_BYTES,g->compute,g->profiling,g->error,sizeof(g->error));
        if(!g->sol){h3_gpu_tensor_free(g->sol_workspace);g->sol_workspace=nullptr;return 0;}
        g->attention_mode=mode;g->dit_attention_configured=true;
        fprintf(stderr,"h3cli: DiT attention=sol recipe=%d plan=%d workspace_limit=%zu q=%d kv=%d min_exact=%.9g dense_steps=%d dense_layers=%d tau=%.9g local_radius=%d dense_sigma=%.9g\n",
            H3_CUDA_SOL_VERSION,H3_CUDA_SOL_PLAN_VERSION,H3_CUDA_SOL_WORKSPACE_BYTES,g->sol_options.q_block,g->sol_options.kv_block,g->sol_options.min_exact,g->sol_options.dense_steps,g->sol_options.dense_layers,g->sol_options.tau,g->sol_options.local_radius,g->sol_options.dense_sigma);
        return 1;
    }
    g->sage_workspace=h3_gpu_tensor_alloc(g,H3_ATTENTION_WORKSPACE_BYTES,H3_GPU_I8,H3_GPU_DEVICE_ONLY);
    if(!g->sage_workspace)return 0;
    g->sage=h3_sage_create(g->sage_workspace->data,H3_ATTENTION_WORKSPACE_BYTES,g->compute,g->profiling,g->error,sizeof(g->error));
    if(!g->sage){h3_gpu_tensor_free(g->sage_workspace);g->sage_workspace=nullptr;return 0;}
    g->attention_mode=mode;g->dit_attention_configured=true;
    fprintf(stderr,"h3cli: DiT attention=%s recipe=%d plan=%d workspace_limit=%zu\n",h3_attention_name(mode),H3_ATTENTION_VERSION,H3_ATTENTION_PLAN_VERSION,H3_ATTENTION_WORKSPACE_BYTES);
    return 1;
}
int h3_gpu_dit_sol_layout(h3_gpu *g,const h3_sol_layout *layout) {
    if(!g||!layout)return 0;
    const char *capture=getenv("H3_TEST_ATTENTION_CAPTURE_DIR");
    if(capture&&*capture) {
        std::string path=std::string(capture)+"/layout.bin";
        FILE *f=fopen(path.c_str(),"wb");if(!f)return h3_gpu_set_error(g,"cannot capture SOL layout");
        uint32_t header[]={layout->sequence,layout->query_blocks,layout->key_blocks};
        bool ok=fwrite(header,sizeof(header),1,f)==1&&fwrite(layout->query,sizeof(h3_sol_block),layout->query_blocks,f)==layout->query_blocks&&fwrite(layout->key,sizeof(h3_sol_block),layout->key_blocks,f)==layout->key_blocks;
        if(fclose(f))ok=false;if(!ok)return h3_gpu_set_error(g,"cannot write SOL layout");
    }
    if(g->subblock) {
        if(!h3_cuda_subblock_configure(g->subblock,layout,56,g->policy.subblock_sparsity,g->error,sizeof(g->error)))return 0;
        for(unsigned b=0;b<layout->query_blocks;) {
            if(!layout->query[b].protect){b++;continue;}
            unsigned first=b*64;while(b<layout->query_blocks&&layout->query[b].protect)b++;
            g->subblock_protected.emplace_back(first,std::min(layout->sequence,b*64)-first);
        }
    }
    return !g->sol||h3_cuda_sol_configure(g->sol,g->sol_options,layout,56,g->error,sizeof(g->error));
}
int h3_gpu_dit_attention_noise(h3_gpu *g,int step,float video,float audio) {
    if(!g)return 0;if(g->attention_mode!=H3_ATTENTION_SOL&&g->attention_mode!=H3_ATTENTION_SUBBLOCK)return 1;
    if(step<0||!std::isfinite(video)||!std::isfinite(audio)||video<0||video>1||audio<0||audio>1)
        return h3_gpu_set_error(g,"invalid CUDA SOL absolute step/noise");
    g->sol_step=step;g->sol_video=video;g->sol_audio=audio;return 1;
}
int h3_gpu_dit_sdpa_bf16(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t seq,uint32_t heads,
    uint32_t dim,float scale,int head_major,unsigned block,int step) {
    if(!g)return 0;
    /* Main DiT boundary: Q/K/V are sequence-major post-normalization/RoPE
     * BF16; only the projection output may be head-major. The scale remains
     * FP32 on the shared base. All four engines validate the same contract. */
    size_t n=(size_t)seq*heads*dim;
    if(!launch_ready(g)||!seq||seq>INT32_MAX/128||heads!=56||dim!=128||
       !std::isfinite(scale)||scale<=0||(head_major!=0&&head_major!=1)||block>=50||step<0||
       !h3_gpu_require_elements(g,q,n,"DiT attention Q")||!h3_gpu_require_elements(g,k,n,"DiT attention K")||
       !h3_gpu_require_elements(g,v,n,"DiT attention V")||!h3_gpu_require_elements(g,out,n,"DiT attention output")||
       q->dtype!=H3_GPU_BF16||k->dtype!=H3_GPU_BF16||v->dtype!=H3_GPU_BF16||out->dtype!=H3_GPU_BF16||
       out==q||out==k||out==v)
        return h3_gpu_set_error(g,"DiT attention requires disjoint BF16 output, S>0, H=56, D=128, positive FP32 scale and valid block/step/layout");
    if(!attention_capture(g,q,k,v,seq,heads,dim,block,step))return 0;
    profile_scope experiment_timer(g,8,nullptr,g->experiment_timing);
    auto dense=[&]() {int ok=attention(g,out,q,k,v,1,seq,heads,heads,dim,scale,H3_GPU_BF16,false,head_major!=0);if(ok)g->main_dense_calls++;return ok;};
    if(!g->attention_mode)return dense();
    if(g->attention_mode==H3_ATTENTION_SUBBLOCK) {
        if(step!=g->sol_step)return h3_gpu_set_error(g,"SubBlock absolute step was not bound");
        const char *reason=h3_subblock_dense_reason(seq,block,step,g->policy.subblock_sparsity,g->policy.subblock_warmup);
        if(reason){g->subblock_bypass++;return dense();}
#ifdef H3_CUDA_USE_SGLANG_FLASH
        if(!g->sglang_reference)return h3_gpu_set_error(g,"SubBlock main DiT requires the shared SGLang pipeline");
        size_t lse_count=size_t(seq)*heads;
        if(!g->sglang_lse||g->sglang_lse->elements<lse_count) {
            if(g->sglang_lse&&!h3_gpu_synchronize(g))return 0;
            h3_gpu_tensor_free(g->sglang_lse);g->sglang_lse=h3_gpu_tensor_new_f32(g,lse_count);
            if(!g->sglang_lse)return 0;
        }
        if(!h3_cuda_subblock_run(g->subblock,tensor_pointer(out),tensor_pointer(q),tensor_pointer(k),tensor_pointer(v),
            seq,heads,scale,head_major!=0,false,true,g->error,sizeof(g->error))){g->deferred_error=true;return 0;}
        for(auto range:g->subblock_protected) {
            if(h3_sglang_flash_query_range(tensor_pointer(out),tensor_pointer(q),tensor_pointer(k),tensor_pointer(v),
                (float*)tensor_pointer(g->sglang_lse),seq,heads,range.first,range.second,scale,head_major,g->compute,g->error,sizeof(g->error))!=1){g->deferred_error=true;return 0;}
            g->subblock_protected_calls++;
        }
        g->stats.attention_dispatches++;return launch_status(g,"SubBlock main DiT attention");
#else
        return h3_gpu_set_error(g,"SubBlock requires CUDA_SGLANG=1");
#endif
    }

    if(g->attention_mode==H3_ATTENTION_SOL) {
        if(step!=g->sol_step)return h3_gpu_set_error(g,"CUDA SOL absolute step was not bound");
        const char *reason=h3_cuda_sol_dense_reason(g->sol_options,block,step,g->sol_video,g->sol_audio);
        if(reason) {
            g->sol_bypass++;
            fprintf(stderr,"h3cli: SOL policy step=%d block=%u effective=dense reason=%s\n",step,block,reason);
            return dense();
        }
    }
    if(!g->sglang_reference) {
    uint32_t bits;memcpy(&bits,&scale,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000u;memcpy(&scale,&bits,4);
    }
    if(g->profiling&&g->attention_shapes.emplace((int)H3_GPU_BF16,1,seq,heads,dim,head_major).second)
        fprintf(stderr,"h3cli: attention shape phase=main-DiT requested=%s effective=%s recipe=%d plan=%d S=%u H=%u D=%u head_major=%d\n",
            h3_attention_name(g->attention_mode),h3_attention_name(g->attention_mode),H3_ATTENTION_VERSION,H3_ATTENTION_PLAN_VERSION,seq,heads,dim,head_major);
    profile_scope timing(g,1);
    if(g->attention_mode==H3_ATTENTION_SOL) {
        if(!h3_cuda_sol_run(g->sol,tensor_pointer(out),tensor_pointer(q),tensor_pointer(k),tensor_pointer(v),seq,heads,scale,head_major!=0,false,g->error,sizeof(g->error))){g->deferred_error=true;return 0;}
        g->stats.attention_dispatches++;return launch_status(g,"SOL main DiT attention");
    }
    if(!h3_sage_run(g->sage,g->attention_mode,tensor_pointer(out),tensor_pointer(q),tensor_pointer(k),tensor_pointer(v),seq,heads,scale,head_major!=0,g->error,sizeof(g->error))) {
        g->deferred_error=true;return 0;
    }
    g->stats.attention_dispatches++;
    return launch_status(g,"Sage main DiT attention");
}

#include "src/cuda/cuda_vae.cuh"

#include "src/upscale/upscale_cuda.cuh"
