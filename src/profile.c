#include "src/profile.h"
#include "src/backend.h"
#include "src/memory.h"
#include "src/weights/q8.h"
#include "src/sglang/sglang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#ifdef __APPLE__
#include <os/log.h>
#include <os/signpost.h>
#include <dispatch/dispatch.h>
static os_log_t trace_log;
static dispatch_once_t trace_once;
static void make_log(void *unused){(void)unused;trace_log=os_log_create("org.h3.native-metal","DiT");}
static os_log_t log_handle(void){dispatch_once_f(&trace_once,NULL,make_log);return trace_log;}
#endif
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
static int enabled(const char *name){const char *value=getenv(name);return value&&*value&&strcmp(value,"0");}
void h3_profile_memory(h3_gpu *gpu,const char *phase,int step,int block,double seconds) {
    if(!enabled("H3_PROFILE"))return;
    h3_memory_snapshot m;h3_memory_sample(&m);h3_gpu_stats g={0};
    if(gpu)h3_gpu_get_stats(gpu,&g);
#ifdef __APPLE__
    else g.metal_current_allocated_bytes=h3_gpu_device_allocated_bytes();
#endif
    if(gpu)fprintf(stderr,"h3_lifetime {\"phase\":\"%s\",\"step\":%d,\"block\":%d,\"inflight_peak\":%llu,\"retired_commands\":%llu,\"bounded_waits\":%llu,\"pressure_drains\":%llu,\"scratch_bytes\":%llu,\"scratch_allocations\":%llu,\"scratch_reuses\":%llu,\"native_pipelines\":%llu}\n",
        phase,step,block,(unsigned long long)g.inflight_peak,(unsigned long long)g.command_retirements,
        (unsigned long long)g.bounded_waits,(unsigned long long)g.pressure_drains,(unsigned long long)g.native_scratch_bytes,
        (unsigned long long)g.native_scratch_allocations,(unsigned long long)g.native_scratch_reuses,
        (unsigned long long)g.native_pipeline_count);
    fprintf(stderr,"h3_memory {\"phase\":\"%s\",\"step\":%d,\"block\":%d,\"seconds\":%.6f,\"footprint_bytes\":%llu,\"resident_bytes\":%llu,\"process_compressed_bytes\":%llu,\"system_compressed_bytes\":%llu,\"system_wired_bytes\":%llu,\"available_bytes\":%llu,\"limit_bytes\":%llu,\"metal_allocated_bytes\":%llu,\"tensor_live_bytes\":%llu,\"tensor_allocations\":%llu}\n",
        phase,step,block,seconds,(unsigned long long)m.physical_footprint,(unsigned long long)m.resident,
        (unsigned long long)m.process_compressed,(unsigned long long)m.system_compressed,(unsigned long long)m.system_wired,
        (unsigned long long)m.available,(unsigned long long)h3_memory_limit_bytes(&m),
        (unsigned long long)g.metal_current_allocated_bytes,(unsigned long long)g.live_bytes,(unsigned long long)g.tensor_allocations);
}
static int fence_component(const char *name){
    return enabled("H3_PROFILE_COMPONENTS") ||
        (enabled("H3_PROFILE_REGIONS") && !strncmp(name,"DiT complete ",13));
}
int h3_profile_steps_enabled(void){
#ifdef __APPLE__
    return enabled("H3_PROFILE");
#else
    return h3_sglang_requested() && enabled("H3_PROFILE");
#endif
}
h3_profile_span h3_profile_component_begin(h3_gpu *gpu,const char *name,unsigned block,int step){
    h3_profile_span s={0};
    if(!enabled("H3_PROFILE")&&!enabled("H3_METAL_SIGNPOSTS"))return s;
    if(fence_component(name)&&(!h3_gpu_submit(gpu)||!h3_gpu_begin(gpu))){s.started=-1;return s;}
    s.started=now();
    h3_gpu_get_stats(gpu,&s.before);
#ifdef __APPLE__
    os_log_t log=log_handle();s.signpost=os_signpost_id_generate(log);
    os_signpost_interval_begin(log,s.signpost,"DiT component","%{public}s block=%u step=%d",name,block,step);
#else
    (void)name;(void)block;(void)step;
#endif
    return s;
}
int h3_profile_component_end(h3_gpu *gpu,h3_profile_span s,const char *name,unsigned block,int step){
    if (s.started == 0)
        return 1;
    if (s.started < 0)
        return 0;
    int ok = 1;
    /* Explicit diagnostic fences; never compare this mode with ordinary B5. */
    if(fence_component(name)){
        ok=h3_gpu_submit(gpu);
        h3_gpu_stats after={0};h3_gpu_get_stats(gpu,&after);
        fprintf(stderr,"h3_component {\"name\":\"%s\",\"block\":%u,\"step\":%d,\"wall_seconds\":%.9f,\"fenced\":true,\"submissions\":%llu,\"dispatches\":%llu,\"mps_attention\":%llu,\"mps_linears\":%llu,\"blit_copies\":%llu,\"explicit_d2d_bytes\":%llu,\"encode_seconds\":%.9f,\"wait_seconds\":%.9f,\"root_gpu_seconds\":%.9f,\"live_bytes\":%llu,\"peak_live_bytes\":%llu}\n",
            name,block,step+1,now()-s.started,(unsigned long long)(after.submissions-s.before.submissions),
            (unsigned long long)(after.direct_dispatches-s.before.direct_dispatches),
            (unsigned long long)(after.mps_sdpa_dispatches-s.before.mps_sdpa_dispatches),
            (unsigned long long)(after.mps_linear_dispatches-s.before.mps_linear_dispatches),
            (unsigned long long)(after.blit_copies-s.before.blit_copies),(unsigned long long)(after.d2d_bytes-s.before.d2d_bytes),
            after.command_encode_seconds-s.before.command_encode_seconds,after.command_wait_seconds-s.before.command_wait_seconds,
            after.gpu_seconds-s.before.gpu_seconds,(unsigned long long)after.live_bytes,(unsigned long long)after.peak_live_bytes);
        if(ok)ok=h3_gpu_begin(gpu);
    }
#ifdef __APPLE__
    os_signpost_interval_end(log_handle(),s.signpost,"DiT component","%{public}s block=%u step=%d",name,block,step);
#endif
    return ok;
}
void h3_profile_step(h3_gpu *gpu,double started,int step,int total,int evaluated,h3_gpu_stats before){
    if(!h3_profile_steps_enabled())return;
    h3_gpu_stats after={0};h3_gpu_get_stats(gpu,&after);h3_memory_snapshot m;h3_memory_sample(&m);
#ifndef __APPLE__
    /* Shared CUDA diagnostics; event-instrumented runs cannot qualify time. */
    fprintf(stderr,"h3_cuda_reference_step {\"step\":%d,\"total\":%d,\"evaluated\":%s,\"wall_seconds\":%.9f,\"gemm_seconds\":%.9f,\"attention_seconds\":%.9f,\"elementwise_seconds\":%.9f,\"h2d_seconds\":%.9f,\"d2h_seconds\":%.9f,\"upload_wait_seconds\":%.9f,\"h2d_bytes\":%llu,\"d2h_bytes\":%llu,\"live_device_bytes\":%llu,\"peak_tensor_bytes\":%llu,\"pinned_bytes\":%llu,\"tensor_allocations\":%llu,\"resident_bytes\":%llu,\"swap_used_bytes\":%llu}\n",
        step+1,total,evaluated?"true":"false",now()-started,
        after.linear_seconds-before.linear_seconds,after.attention_seconds-before.attention_seconds,
        after.normalization_seconds-before.normalization_seconds,after.h2d_seconds-before.h2d_seconds,
        after.d2h_seconds-before.d2h_seconds,after.slot_wait_seconds-before.slot_wait_seconds,
        (unsigned long long)(after.h2d_bytes-before.h2d_bytes),
        (unsigned long long)(after.d2h_bytes-before.d2h_bytes),
        (unsigned long long)after.device_bytes,(unsigned long long)after.peak_live_bytes,
        (unsigned long long)after.pinned_bytes,(unsigned long long)after.tensor_allocations,
        (unsigned long long)m.resident,(unsigned long long)m.swap_used);
    return;
#endif
    h3_backend_scope scope=h3_backend_current();
    fprintf(stderr,"h3_weights {\"step\":%d,\"format\":\"%s\",\"source\":\"bf16\",\"recipe\":%d,\"group_size\":%d,\"linear_path\":\"%s\",\"activation_dtype\":\"bf16\",\"accumulation\":\"%s\",\"weights\":%llu,\"source_bytes\":%llu,\"storage_bytes\":%llu,\"load_seconds\":%.9f,\"q8_dispatches\":%llu,\"persistent_cache\":false}\n",
        step+1,h3_weight_format_name(scope.metal.weight_format),scope.metal.weight_format?H3_Q8_VERSION:0,
        scope.metal.weight_format?H3_Q8_GROUP:0,scope.metal.weight_format?(scope.metal.q8_kernel?"metal-simdgroup-q8":"metal-dequant-mpsgraph"):"baseline",
        scope.metal.weight_format?(scope.metal.q8_kernel?"fp32":"mpsgraph-internal"):"baseline",(unsigned long long)after.q8_weights,
        (unsigned long long)after.q8_source_bytes,(unsigned long long)after.q8_storage_bytes,
        after.q8_load_seconds,(unsigned long long)(after.q8_dispatches-before.q8_dispatches));
#ifdef __APPLE__
    fprintf(stderr,"h3_recipe {\"step\":%d,\"tier\":\"%s\",\"attention_oracle\":%s,\"weight_storage\":\"%s\",\"state_dtype\":\"bf16/fp32\",\"attention_storage\":\"%s\",\"matrix_operands\":\"%s\",\"accumulators\":\"%s\",\"routing\":\"%s\",\"device_path\":\"%s\",\"math\":\"%s\",\"mixed_recipe\":%d,\"layout_fusion\":%d,\"layout_recipe\":%d,\"ane_recipe\":%d,\"ane_mode\":%d,\"qkv_shard_arithmetic\":\"%s\"}\n",
        step+1,scope.backend?h3_metal_tier_name(scope.metal.tier):"reference-plus",scope.backend?"false":"true",scope.metal.weight_format?"q8":"bf16-source",
        scope.backend?h3_metal_precision_name(scope.metal.precision):"bf16",
        scope.backend?(scope.metal.precision?"fp16-with-bf16/fp32-recovery":"fp32"):"mpsgraph-internal",
        scope.backend?"fp32":"mpsgraph-internal",h3_attention_mode_name(scope.attention),
        scope.backend?"metal-simdgroup":"mpsgraph",
        scope.backend?"safe/explicit-fast-exp2":"mpsgraph",scope.backend&&scope.metal.precision?H3_METAL_FP16_VERSION:0,
        scope.metal.layout_fusion,scope.metal.layout_fusion?H3_METAL_LAYOUT_VERSION:0,scope.metal.ane_mode?H3_METAL_ANE_VERSION:0,scope.metal.ane_mode,
        scope.metal.weight_format?(scope.metal.q8_kernel?"q8-weights/bf16-reconstruction/fp32-accumulation":"q8-weights/bf16-dequant-mpsgraph"):scope.metal.ane_mode?"scaled-fp16-ANE-matmul/partial-add;bf16-output;gpu-recovery":"mpsgraph");
#endif
    fprintf(stderr,"h3_layout {\"step\":%d,\"prepares\":%llu,\"inplace_packs\":%llu,\"scan_replaced_bytes\":%llu,\"partial_buffer_bytes_written\":%llu}\n",step+1,
        (unsigned long long)(after.layout_prepares-before.layout_prepares),
        (unsigned long long)(after.layout_inplace_packs-before.layout_inplace_packs),
        (unsigned long long)(after.layout_scan_replaced_bytes-before.layout_scan_replaced_bytes),
        (unsigned long long)(after.layout_partial_bytes-before.layout_partial_bytes));
    fprintf(stderr,"h3_execution {\"step\":%d,\"composition\":\"%s\",\"status\":\"%s\",\"dense_candidate\":\"%s\",\"recipe\":%d,\"q_block\":%d,\"kv_block\":%d,\"tau\":%.9g,\"dense_layers\":%d,\"dense_steps\":%d,\"dense_sigma\":%.9g,\"sol_recipe\":%d,\"local_radius\":%d,\"min_exact\":%.9g,\"encode_seconds\":%.9f,\"wait_seconds\":%.9f,\"root_gpu_seconds\":%.9f,\"dispatches\":%llu,\"blit_copies\":%llu,\"explicit_d2d_bytes\":%llu}\n",
        step+1,scope.backend?"hybrid":"mpsgraph",scope.backend?"unqualified":"reference",
        scope.backend?h3_metal_candidate_name(scope.metal.candidate):"mpsgraph",scope.backend?H3_METAL_ATTENTION_VERSION:0,
        scope.metal.q_block,scope.metal.kv_block,(double)scope.metal.tau,scope.metal.dense_layers,
        scope.metal.dense_steps,(double)scope.metal.dense_sigma,scope.attention==H3_ATTN_SOL?H3_METAL_SOL_VERSION:0,
        scope.metal.local_radius,(double)scope.metal.min_exact,
        after.command_encode_seconds-before.command_encode_seconds,after.command_wait_seconds-before.command_wait_seconds,
        after.gpu_seconds-before.gpu_seconds,(unsigned long long)(after.direct_dispatches-before.direct_dispatches),
        (unsigned long long)(after.blit_copies-before.blit_copies),(unsigned long long)(after.d2d_bytes-before.d2d_bytes));
#ifdef __APPLE__
    if(scope.attention==H3_ATTN_SOL)h3_gpu_native_sol_report(gpu,step);
#endif
    fprintf(stderr,"h3_step {\"step\":%d,\"total\":%d,\"evaluated\":%s,\"wall_seconds\":%.9f,\"backend\":\"%s\",\"attention\":\"%s\",\"fenced\":true,\"component_fences\":%s,\"submissions\":%llu,\"mps_attention\":%llu,\"mps_linears\":%llu,\"native_attention\":%llu,\"metal_current_allocated_bytes\":%llu,\"metal_live_bytes\":%llu,\"metal_peak_bytes\":%llu,\"resident_bytes\":%llu,\"physical_footprint_bytes\":%llu,\"process_compressed_bytes\":%llu,\"system_compressed_bytes\":%llu,\"swap_used_bytes\":%llu,\"process_memory_valid\":%s,\"system_memory_valid\":%s,\"swap_valid\":%s}\n",
        step+1,total,evaluated?"true":"false",now()-started,h3_backend_name(scope.backend),h3_attention_mode_name(scope.attention),
        enabled("H3_PROFILE_COMPONENTS")?"true":"false",(unsigned long long)(after.submissions-before.submissions),
        (unsigned long long)(after.mps_sdpa_dispatches-before.mps_sdpa_dispatches),(unsigned long long)(after.mps_linear_dispatches-before.mps_linear_dispatches),
        (unsigned long long)(after.native_attention_dispatches-before.native_attention_dispatches),
        (unsigned long long)after.metal_current_allocated_bytes,(unsigned long long)after.live_bytes,(unsigned long long)after.peak_live_bytes,
        (unsigned long long)m.resident,(unsigned long long)m.physical_footprint,(unsigned long long)m.process_compressed,
        (unsigned long long)m.system_compressed,(unsigned long long)m.swap_used,
        m.process_valid?"true":"false",m.system_valid?"true":"false",m.swap_valid?"true":"false");
}
