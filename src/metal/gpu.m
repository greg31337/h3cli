#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShaders/MetalPerformanceShaders.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>

#include "src/gpu.h"
#include "src/execution.h"
#include "src/metal/metal_fp16.h"
#include "src/metal/shader.h"
#include "src/weights/safetensors.h"
#include "src/backend.h"
#include "src/profile.h"
#include "src/memory.h"
#include "src/weights/q8.h"
#include "src/metal/native_attention.inc"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <time.h>
#include <unistd.h>

@class H3GPU;
@interface H3Tensor : NSObject
@property(nonatomic, strong) id<MTLBuffer> buffer;
@property(nonatomic) size_t elements;
@property(nonatomic) size_t bytes;
@property(nonatomic) BOOL fileMapped;
@property(nonatomic) h3_gpu_dtype dtype;
@property(nonatomic) uint32_t q8Rows, q8Columns;
@property(nonatomic) size_t q8ScaleOffset;
@property(nonatomic, weak) H3GPU *owner;
@property(nonatomic, strong) MPSGraphTensorData *graphData;
@property(nonatomic, strong) NSArray<NSNumber *> *graphDataShape;
@property(nonatomic) MPSDataType graphDataType;
@end
@implementation H3Tensor
@end

@interface H3SDPA : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *query;
@property(nonatomic, strong) MPSGraphTensor *key;
@property(nonatomic, strong) MPSGraphTensor *value;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3SDPA
@end

@interface H3Linear : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *weightShape;
@property(nonatomic, strong) NSArray<NSNumber *> *biasShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3Linear
@end

@interface H3MLP : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *fc1Weight;
@property(nonatomic, strong) MPSGraphTensor *fc2Weight;
@property(nonatomic, strong) MPSGraphTensor *fc1Bias;
@property(nonatomic, strong) MPSGraphTensor *fc2Bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc1Shape;
@property(nonatomic, strong) NSArray<NSNumber *> *fc2Shape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3MLP
@end

@interface H3Conv : NSObject
@property(nonatomic, strong) MPSGraph *graph;
@property(nonatomic, strong) MPSGraphTensor *input;
@property(nonatomic, strong) MPSGraphTensor *weight;
@property(nonatomic, strong) MPSGraphTensor *bias;
@property(nonatomic, strong) MPSGraphTensor *output;
@property(nonatomic, strong) NSArray<NSNumber *> *inputShape;
@property(nonatomic, strong) NSArray<NSNumber *> *weightShape;
@property(nonatomic, strong) NSArray<NSNumber *> *biasShape;
@property(nonatomic, strong) NSArray<NSNumber *> *outputShape;
@end
@implementation H3Conv
@end

/* One context-owned arena: typed, immutable-shape buffer slots are reused in
 * command order. Separate Metal buffers preserve resource hazard tracking. */
@interface H3ScratchArena : NSObject
@property(nonatomic,strong) NSMutableDictionary<NSString *,NSValue *> *slots;
@end
@implementation H3ScratchArena
- (instancetype)init {if((self=[super init]))_slots=[NSMutableDictionary new];return self;}
- (void)dealloc {for(NSValue *entry in _slots.allValues)h3_gpu_tensor_free(entry.pointerValue);}
@end

@interface H3GPU : NSObject
@property(nonatomic, strong) id<MTLDevice> device;
@property(nonatomic, strong) id<MTLCommandQueue> queue;
@property(nonatomic, strong) id<MTLLibrary> library;
@property(nonatomic, strong) id<MTLLibrary> nativeAttentionLibrary;
@property(nonatomic, strong) NSMutableDictionary<NSString *, id<MTLComputePipelineState>> *nativeAttentionPipelines;
@property(nonatomic, strong) H3ScratchArena *nativeScratch;
@property(nonatomic, strong) NSMutableIndexSet *solUsed;
@property(nonatomic, strong) NSMutableIndexSet *fp16Used;
@property(nonatomic) uint32_t fp16Heads;
@property(nonatomic) BOOL fp16Prepared;
@property(nonatomic) uint32_t fp16PreparedSequence,fp16PreparedHeads;
@property(nonatomic) float fp16PreparedScale;
@property(nonatomic,strong) id<MTLBuffer> fp16PreparedQ,fp16PreparedK,fp16PreparedV;
@property(nonatomic, strong) NSMutableIndexSet *diagnosticRanges;
@property(nonatomic, strong) NSMutableIndexSet *diagnosticScores;
@property(nonatomic) uint32_t diagnosticHeads;
@property(nonatomic) h3_backend_scope nativeScope;
@property(nonatomic) BOOL solNoiseSet;
@property(nonatomic) int solNoiseStep;
@property(nonatomic) float solVideoSigma,solAudioSigma;
@property(nonatomic) uint32_t solSequence,solQueryBlocks,solKeyBlocks,solProtectedRows;
@property(nonatomic, strong) id<MTLCommandBuffer> command;
@property(nonatomic, strong) NSMutableArray<id<MTLCommandBuffer>> *inflightCommands;
@property(nonatomic, strong) NSDictionary<NSString *, id<MTLComputePipelineState>> *pipelines;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3SDPA *> *sdpaCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3Linear *> *linearCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3MLP *> *mlpCache;
@property(nonatomic, strong) NSMutableDictionary<NSString *, H3Conv *> *convCache;
@property(nonatomic, strong) MPSCommandBuffer *mpsCommand;
@property(nonatomic) BOOL reuseMPSCommandDefault;
@property(nonatomic, copy) NSString *lastError;
@property(nonatomic) h3_gpu_stats stats;
@property(nonatomic, copy) NSString *profileLabel;
@property(nonatomic) BOOL tensorOpsEnabled;
@property(nonatomic) NSUInteger tensorOpsMode;
@property(nonatomic) BOOL headMajorSDPAInputs;
@property(nonatomic) h3_gpu_stats profileStartStats;
@property(nonatomic) h3_gpu_stats profileMarkStats;
@property(nonatomic) double profileStartWall;
@property(nonatomic) double profileMarkWall;
@property(nonatomic) double commandStartWall;
@end
@implementation H3GPU
@end

static H3GPU *GPU(h3_gpu *gpu) {
    return (__bridge H3GPU *)gpu;
}

static H3Tensor *TENSOR(const h3_gpu_tensor *tensor) {
    return (__bridge H3Tensor *)(void *)tensor;
}

static MPSGraphTensorData *h3_gpu_graph_data(const h3_gpu_tensor *tensor,
                                             NSArray<NSNumber *> *shape,
                                             MPSDataType data_type,
                                             int stable) {
    H3Tensor *object = TENSOR(tensor);
    if (!stable || getenv("H3_DISABLE_GRAPH_DATA_CACHE"))
        return [[MPSGraphTensorData alloc] initWithMTLBuffer:object.buffer
            shape:shape dataType:data_type];
    if (object.graphData && object.graphDataShape == shape &&
        object.graphDataType == data_type) return object.graphData;
    MPSGraphTensorData *data = [[MPSGraphTensorData alloc]
        initWithMTLBuffer:object.buffer shape:shape dataType:data_type];
    if (!object.graphData) {
        object.graphData = data;
        object.graphDataShape = shape;
        object.graphDataType = data_type;
    }
    return data;
}

static MPSCommandBuffer *h3_gpu_mps_command(H3GPU *gpu) {
    const char *override = getenv("H3_REUSE_MPS_COMMAND");
    BOOL reuse = override ? (*override && strcmp(override, "0")) :
                            gpu.reuseMPSCommandDefault;
    if (!reuse)
        return [MPSCommandBuffer commandBufferWithCommandBuffer:gpu.command];
    if (!gpu.mpsCommand || gpu.mpsCommand.rootCommandBuffer != gpu.command)
        gpu.mpsCommand =
            [MPSCommandBuffer commandBufferWithCommandBuffer:gpu.command];
    return gpu.mpsCommand;
}

static double h3_gpu_now(void) {
    struct timespec time;
    if (clock_gettime(CLOCK_MONOTONIC, &time) != 0) return 0.0;
    return (double)time.tv_sec + (double)time.tv_nsec * 1e-9;
}

static int h3_gpu_profile_enabled(void) {
    const char *value = getenv("H3_PROFILE");
    return value && *value && strcmp(value, "0");
}

static uint64_t h3_gpu_counter_delta(uint64_t value, uint64_t start) {
    return value >= start ? value - start : 0;
}

static void h3_gpu_profile_emit(H3GPU *gpu, NSString *phase,
                                h3_gpu_stats start, double wall_start) {
    if (!h3_gpu_profile_enabled()) return;
    h3_gpu_stats value = gpu.stats;
    double wall = h3_gpu_now() - wall_start;
    NSString *label = gpu.profileLabel ? gpu.profileLabel : @"Metal context";
    fprintf(stderr,
        "h3cli profile: %-24s %-14s wall=%8.3fs encode=%7.3fs "
        "wait=%8.3fs root-gpu=%7.3fs "
        "peak=%7.3fGiB alloc=%7.3fGiB submissions=%llu "
        "direct=%llu linear=%llu conv=%llu attention=%llu\n",
        label.UTF8String, phase.UTF8String, wall,
        value.command_encode_seconds - start.command_encode_seconds,
        value.command_wait_seconds - start.command_wait_seconds,
        value.gpu_seconds - start.gpu_seconds,
        (double)value.peak_live_bytes / (1024.0 * 1024.0 * 1024.0),
        (double)h3_gpu_counter_delta(value.allocated_bytes,
                                     start.allocated_bytes) /
            (1024.0 * 1024.0 * 1024.0),
        (unsigned long long)h3_gpu_counter_delta(value.submissions,
                                                 start.submissions),
        (unsigned long long)h3_gpu_counter_delta(value.direct_dispatches,
                                                 start.direct_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_linear_dispatches,
                                                 start.mps_linear_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_conv_dispatches,
                                                 start.mps_conv_dispatches),
        (unsigned long long)h3_gpu_counter_delta(value.mps_sdpa_dispatches,
                                                 start.mps_sdpa_dispatches));
}

static void h3_gpu_set_error(H3GPU *gpu, NSString *format, ...) {
    va_list arguments;
    va_start(arguments, format);
    gpu.lastError = [[NSString alloc] initWithFormat:format arguments:arguments];
    va_end(arguments);
}

static int h3_gpu_require_command(H3GPU *gpu) {
    if (!gpu || !gpu.command) {
        if (gpu) h3_gpu_set_error(gpu, @"h3_gpu_begin() was not called");
        return 0;
    }
    char error[512];
    if(!h3_memory_check_gpu(0,gpu.device.currentAllocatedSize,"Metal encoding",error,sizeof(error))) {
        h3_gpu_set_error(gpu,@"%s",error);return 0;
    }
    return 1;
}

static int h3_gpu_memory_reserve(H3GPU *gpu,uint64_t bytes,const char *phase) {
    char error[512];
    if(h3_memory_check_gpu(bytes,gpu.device.currentAllocatedSize,phase,error,sizeof(error)))return 1;
    /* A queued graph owns transient resources until completion. Reclaim that
     * work before rejecting a reservation, but never submit already unsafe
     * work. This is a pressure boundary, not a per-operation CPU barrier. */
    if(gpu.command && h3_memory_check_gpu(0,gpu.device.currentAllocatedSize,phase,NULL,0)) {
        if(!h3_gpu_submit((__bridge h3_gpu *)gpu)||!h3_gpu_begin((__bridge h3_gpu *)gpu))return 0;
        h3_gpu_stats stats=gpu.stats;stats.pressure_drains++;gpu.stats=stats;
    }
    if(!h3_memory_check_gpu(bytes,gpu.device.currentAllocatedSize,phase,error,sizeof(error))) {
        h3_gpu_set_error(gpu,@"%s",error);return 0;
    }
    return 1;
}

static int h3_gpu_require_elements(H3GPU *gpu, const h3_gpu_tensor *tensor,
                                   size_t elements, NSString *label) {
    if (!tensor || TENSOR(tensor).elements < elements) {
        h3_gpu_set_error(gpu, @"%@ tensor is absent or too small", label);
        return 0;
    }
    return 1;
}

static id<MTLComputePipelineState> h3_gpu_pipeline(H3GPU *gpu, NSString *name) {
    id<MTLComputePipelineState> pipeline = gpu.pipelines[name];
    if (!pipeline) h3_gpu_set_error(gpu, @"missing Metal pipeline %@", name);
    return pipeline;
}

static int h3_gpu_dispatch_1d(H3GPU *gpu, NSString *name, uint32_t count,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        NSUInteger width = MIN((NSUInteger)256, pipeline.maxTotalThreadsPerThreadgroup);
        [encoder dispatchThreads:MTLSizeMake(count, 1, 1)
           threadsPerThreadgroup:MTLSizeMake(width, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_2d(H3GPU *gpu, NSString *name, uint32_t width,
                              uint32_t height,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        NSUInteger x = 16;
        NSUInteger y = MIN((NSUInteger)16, pipeline.maxTotalThreadsPerThreadgroup / x);
        [encoder dispatchThreads:MTLSizeMake(width, height, 1)
           threadsPerThreadgroup:MTLSizeMake(x, y, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_3d(H3GPU *gpu, NSString *name, MTLSize grid,
                              void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        [encoder dispatchThreads:grid threadsPerThreadgroup:MTLSizeMake(8, 4, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_dispatch_rows(H3GPU *gpu, NSString *name, uint32_t rows,
                                void (^bindings)(id<MTLComputeCommandEncoder>)) {
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    NSUInteger maximum = MIN((NSUInteger)256,
                             pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger threads = 1;
    while (threads * 2 <= maximum) threads *= 2;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        bindings(encoder);
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

h3_gpu *h3_gpu_create(const char *shader_source_path,
                      char *error, size_t error_size) {
    @autoreleasepool {
        char *resolved = h3_shader_resolve(shader_source_path, error, error_size);
        if (!resolved) return NULL;
        NSString *path = [NSString stringWithUTF8String:resolved];
        free(resolved);
        H3GPU *gpu = [[H3GPU alloc] init];
        gpu.profileLabel = @"Metal context";
        gpu.profileStartWall = h3_gpu_now();
        gpu.profileMarkWall = gpu.profileStartWall;
        gpu.device = MTLCreateSystemDefaultDevice();
        gpu.queue = [gpu.device newCommandQueue];
        gpu.reuseMPSCommandDefault = YES;
        gpu.inflightCommands = [NSMutableArray array];
        gpu.sdpaCache = [NSMutableDictionary dictionary];
        gpu.linearCache = [NSMutableDictionary dictionary];
        gpu.mlpCache = [NSMutableDictionary dictionary];
        gpu.convCache = [NSMutableDictionary dictionary];
        if (!gpu.device || !gpu.queue) {
            if (error && error_size) snprintf(error, error_size, "cannot initialize Metal");
            return NULL;
        }
        if (getenv("H3_DEBUG_GPU_MEMORY")) {
            fprintf(stderr, "h3cli: Metal live allocation at GPU startup: "
                    "%.3f GiB\n", (double)gpu.device.currentAllocatedSize /
                    (1024.0 * 1024.0 * 1024.0));
        }
        const char *source_path = path.UTF8String;
        NSError *libraryError = nil;
        NSString *source = [NSString stringWithContentsOfFile:path
                                                     encoding:NSUTF8StringEncoding
                                                        error:&libraryError];
        if (source) {
            MTLCompileOptions *options = [[MTLCompileOptions alloc] init];
            options.mathMode = MTLMathModeSafe;
            const char *nax = getenv("H3_NAX");
            BOOL m5 = [gpu.device.name rangeOfString:@"M5"].location !=
                      NSNotFound;
            BOOL wantsTensorOps =
                m5 && (!nax || !*nax || strcmp(nax, "0") != 0);
            if (wantsTensorOps)
                options.preprocessorMacros = @{ @"H3_METAL_HAS_TENSOR": @"1" };
            gpu.library = [gpu.device newLibraryWithSource:source
                                                   options:options
                                                     error:&libraryError];
            gpu.tensorOpsEnabled = gpu.library && wantsTensorOps;
            if (gpu.tensorOpsEnabled) {
                const char *mode = nax && *nax ? nax : "qkv-attn";
                gpu.tensorOpsMode = !strcmp(mode, "attn") ? 2u :
                                    !strcmp(mode, "qkv-attn") ? 3u :
                                    !strcmp(mode, "qkv") ? 4u :
                                    !strcmp(mode, "mlp") ? 5u : 1u;
            }
            if (!gpu.library && wantsTensorOps) {
                /* TensorOps is optional: an older runtime must retain the
                 * ordinary MPSGraph/direct Metal implementation. */
                if (getenv("H3_NAX_DIAGNOSTIC"))
                    fprintf(stderr, "h3cli: TensorOps compile failed: %s\n",
                            libraryError.localizedDescription.UTF8String);
                options.preprocessorMacros = @{};
                libraryError = nil;
                gpu.library = [gpu.device newLibraryWithSource:source
                                                       options:options
                                                         error:&libraryError];
                gpu.tensorOpsEnabled = NO;
                gpu.tensorOpsMode = 0;
            }
        }
        if (!gpu.library) {
            if (error && error_size) {
                const char *description = libraryError.localizedDescription.UTF8String;
                snprintf(error, error_size, "cannot compile %s: %s",
                         source_path, description ? description : "unknown error");
            }
            return NULL;
        }
        NSMutableArray<NSString *> *names = [@[
            @"h3_linear_f32", @"h3_linear_f32_tiled",
            @"h3_linear_f32_tiled_bf16", @"h3_silu_f32",
            @"h3_linear_f32_tiled_bf16_map",
            @"h3_cast_f32_to_bf16",
            @"h3_cast_bf16_to_f32",
            @"h3_rms_norm_f32",
            @"h3_scale_add_f32", @"h3_layer_norm_f32",
            @"h3_video_qkv_rope_f32",
            @"h3_adaln_f32", @"h3_gate_f32", @"h3_qkv_rope_f32",
            @"h3_swiglu_f32", @"h3_linear_bf16", @"h3_linear_q8_bf16", @"h3_q8_dequant", @"h3_silu_bf16",
            @"h3_rms_norm_bf16", @"h3_adaln_bf16", @"h3_gate_bf16",
            @"h3_rms_inverse_bf16", @"h3_adaln_linear_bf16",
            @"h3_gate_adaln_bf16", @"h3_gate_adaln_bf16_exact_simd",
            @"h3_qkv_rope_bf16", @"h3_qkv_rope_bf16_coop",
            @"h3_qkv_rope_bf16_coop_uncached", @"h3_qkv_rope_bf16_coop_ranges",
            @"h3_swiglu_bf16",
            @"h3_layer_norm_bf16", @"h3_gelu_bf16",
            @"h3_vision_qkv_rope_bf16",
            @"h3_embedding_bf16", @"h3_text_qk_rope_bf16",
            @"h3_head_rms_norm_bf16", @"h3_rope_text_bf16",
            @"h3_gqa_causal_bf16_reference", @"h3_gqa_causal_bf16_scaled_q",
            @"h3_gqa_causal_bf16_legacy",
            @"h3_gqa_causal_bf16_reference_tiled", @"h3_gqa_causal_bf16_scaled_q_tiled",
            @"h3_gqa_causal_bf16_legacy_tiled", @"h3_add_bf16", @"h3_sub_bf16",
            @"h3_vision_attention_pad", @"h3_vision_attention_unpad",
            @"h3_token_pool_bf16", @"h3_token_pool_adaln_bf16",
            @"h3_token_expand_delta_bf16",
            @"h3_token_expand_adaln_bf16",
            @"h3_euler_bf16", @"h3_video_preview_bf16", @"h3_bridge_euler_bf16",
            @"h3_bridge_check_exact", @"h3_silu_mul_bf16",
            @"h3_weight_norm_f32", @"h3_add_scaled_f32",
            @"h3_alias_free_snake_f32", @"h3_snake1d_f32",
            @"h3_audio_qkv_split_f32", @"h3_audio_attention_pool_f32",
            @"h3_geglu_f32", @"h3_clip_f32",
            @"h3_vae_encoder_pad_f32",
            @"h3_vae_encoder_group_norm_silu_f32"
        ] mutableCopy];
        if (gpu.tensorOpsEnabled) {
            [names addObject:@"h3_linear_bf16_nax_r128"];
            [names addObject:@"h3_linear_bf16_nax_r128_morton"];
            [names addObject:@"h3_linear_bf16_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_bf16_nax_r128_morton4"];
            [names addObject:@"h3_qk_rope_bf16_nax_inplace"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128_morton"];
            [names addObject:@"h3_fc1_swiglu_bf16_nax_r128_morton4"];
            [names addObject:@"h3_quantize_bf16_int8_rows"];
            [names addObject:@"h3_quantize_bf16_int8_rows_scalar"];
            [names addObject:
                @"h3_quantize_bf16_int8_head_major_to_rows_cached"];
            [names addObject:@"h3_quantize_bf16_int8_groups"];
            [names addObject:@"h3_quantize_bf16_int8_groups_scalar"];
            [names addObject:@"h3_quantize_bf16_int8_groups_scalar128"];
            [names addObject:
                @"h3_quantize_bf16_int8_groups_scalar128_cached"];
            [names addObject:
                @"h3_qkv_project_split_int8_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4"];
            [names addObject:
                @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128_k5376"];
            [names addObject:@"h3_fc1_swiglu_int8_nax_r128_full_k5376"];
            [names addObject:@"h3_fc1_swiglu_int8_local_nax_r128"];
            [names addObject:@"h3_linear_int8_nax_r128"];
            [names addObject:
                @"h3_linear_int8_nax_r128_full_k14336"];
            [names addObject:
                @"h3_linear_int8_nax_r128x256_full_k14336"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128"];
            [names addObject:@"h3_linear_int8_local_scales_nax_r128_k7168"];
            [names addObject:@"h3_gate_adaln_quantize_int8"];
            [names addObject:@"h3_gate_adaln_quantize_int8_scalar"];
            [names addObject:@"h3_linear_int8_grouped_nax_r128x64"];
            [names addObject:
                @"h3_linear_int8_grouped_local_nax_r128x64"];
            [names addObject:
                @"h3_linear_int8_grouped_local_nax_r128x128"];
        }
        NSMutableDictionary *pipelines = [NSMutableDictionary dictionary];
        for (NSString *name in names) {
            id<MTLFunction> function = [gpu.library newFunctionWithName:name];
            NSError *pipelineError = nil;
            id<MTLComputePipelineState> pipeline =
                function ? [gpu.device newComputePipelineStateWithFunction:function
                                                                       error:&pipelineError] : nil;
            if (!pipeline) {
                if (error && error_size) {
                    const char *description = pipelineError.localizedDescription.UTF8String;
                    snprintf(error, error_size, "cannot build %s: %s", name.UTF8String,
                             description ? description : "function missing");
                }
                return NULL;
            }
            pipelines[name] = pipeline;
        }
        gpu.pipelines = pipelines;
        return (__bridge_retained h3_gpu *)gpu;
    }
}

void h3_gpu_cancel(h3_gpu *opaque) {
    if (!opaque) return;
    H3GPU *gpu = GPU(opaque);
    @autoreleasepool {
        gpu.command = nil;
        gpu.mpsCommand = nil;
        gpu.fp16Prepared=NO;gpu.fp16PreparedQ=nil;gpu.fp16PreparedK=nil;gpu.fp16PreparedV=nil;
        gpu.headMajorSDPAInputs=NO;
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands)
            [pending waitUntilCompleted];
        [gpu.inflightCommands removeAllObjects];
    }
}

void h3_gpu_free(h3_gpu *gpu) {
    if (!gpu) return;
    h3_gpu_cancel(gpu);
    @autoreleasepool {
        H3GPU *object = CFBridgingRelease(gpu);
        object.nativeScratch=nil;
        h3_gpu_stats released=object.stats;released.native_scratch_bytes=0;object.stats=released;
        object.nativeAttentionPipelines=nil;
        object.nativeAttentionLibrary=nil;
        h3_gpu_profile_emit(object, @"total", object.profileStartStats,
                            object.profileStartWall);
        id<MTLDevice> device = object.device;
        NSUInteger before = device.currentAllocatedSize;
        object.command = nil;
        object.inflightCommands = nil;
        object.sdpaCache = nil;
        object.linearCache = nil;
        object.mlpCache = nil;
        object.convCache = nil;
        object.pipelines = nil;
        object.library = nil;
        object.queue = nil;
        object.device = nil;
        if (getenv("H3_DEBUG_GPU_MEMORY")) {
            fprintf(stderr, "h3cli: Metal live allocation at GPU teardown: "
                    "%.3f GiB\n", (double)before /
                    (1024.0 * 1024.0 * 1024.0));
        }
    }
}

int h3_gpu_is_m5(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return [gpu.device.name rangeOfString:@"M5"].location != NSNotFound;
}

int h3_gpu_has_nax_mlp(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled && gpu.tensorOpsMode == 5;
}

int h3_gpu_has_int8_mlp(const h3_gpu *opaque) {
    if (!opaque) return 0;
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    return gpu.tensorOpsEnabled;
}

int h3_gpu_quant_configure(h3_gpu *gpu,int mode,const char *cache,
                           uint64_t weights,uint64_t activations,int force_stream) {
    (void)gpu;(void)cache;(void)weights;(void)activations;(void)force_stream;return mode==0;
}
h3_gpu_tensor *h3_gpu_quant_load(h3_gpu *gpu,const char *path,uint64_t offset,
                                 uint32_t rows,uint32_t columns) {
    (void)gpu;(void)path;(void)offset;(void)rows;(void)columns;return NULL;
}

static h3_gpu_tensor *h3_gpu_tensor_new(h3_gpu *opaque, const void *values,
                                        size_t elements, size_t item_size,
                                        h3_gpu_dtype dtype) {
    @autoreleasepool {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    if(!h3_gpu_memory_reserve(gpu,bytes,"Metal buffer allocation"))return NULL;
    H3Tensor *tensor = [[H3Tensor alloc] init];
    tensor.elements = elements;
    tensor.bytes = bytes;
    tensor.dtype = dtype;
    tensor.buffer = [gpu.device newBufferWithLength:MAX(bytes, (size_t)1)
                                            options:MTLResourceStorageModeShared];
    if (!tensor.buffer) {
        h3_gpu_set_error(gpu, @"cannot allocate %zu-byte Metal buffer", bytes);
        return NULL;
    }
    tensor.owner = gpu;
    if (values && bytes) memcpy(tensor.buffer.contents, values, bytes);
    h3_gpu_stats stats = gpu.stats;
    stats.allocated_bytes += bytes;
    stats.live_bytes += bytes;
    if (stats.live_bytes > stats.peak_live_bytes)
        stats.peak_live_bytes = stats.live_bytes;
    stats.tensor_allocations++;
    gpu.stats = stats;
    return (__bridge_retained h3_gpu_tensor *)tensor;
    }
}

h3_gpu_tensor *h3_gpu_tensor_new_f32(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_new_bf16(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_new_i8(h3_gpu *gpu, size_t elements) {
    return h3_gpu_tensor_new(gpu, NULL, elements, sizeof(int8_t), H3_GPU_I8);
}

h3_gpu_tensor *h3_gpu_tensor_from_f32(h3_gpu *gpu, const float *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(float), H3_GPU_F32);
}

h3_gpu_tensor *h3_gpu_tensor_from_bf16(h3_gpu *gpu, const uint16_t *values,
                                       size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(uint16_t), H3_GPU_BF16);
}

h3_gpu_tensor *h3_gpu_tensor_from_u32(h3_gpu *gpu, const uint32_t *values,
                                      size_t elements) {
    return h3_gpu_tensor_new(gpu, values, elements, sizeof(uint32_t), H3_GPU_U32);
}

static h3_gpu_tensor *h3_gpu_tensor_load_file(h3_gpu *opaque, const char *path,
                                              uint64_t file_offset,
                                              size_t elements,
                                              size_t item_size,
                                              h3_gpu_dtype dtype,
                                              const char *label) {
    @autoreleasepool {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !path || !*path || file_offset > INT64_MAX ||
        elements > SIZE_MAX / item_size) return NULL;
    size_t bytes = elements * item_size;
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) return NULL;
    const char *zero_copy = getenv("H3_ZERO_COPY_WEIGHTS");
    int transformer_weight = strstr(path, "/transformer/") != NULL;
    int m5 = [gpu.device.name rangeOfString:@"M5"].location != NSNotFound;
    int map_weight = bytes &&
        ((zero_copy && !strcmp(zero_copy, "1")) ||
         (transformer_weight &&
          ((zero_copy && !strcmp(zero_copy, "transformer")) ||
           (!zero_copy && m5))));
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        h3_gpu_set_error(gpu, @"cannot open %s: %s", path, strerror(errno));
        return NULL;
    }
    struct stat status;
    if (fstat(descriptor, &status) || status.st_size < 0 ||
        !h3_st_file_range(file_offset, bytes, (uint64_t)status.st_size)) {
        h3_gpu_set_error(gpu, @"%s: tensor range at %llu (%zu bytes) exceeds file",
                         path, (unsigned long long)file_offset, bytes);
        close(descriptor);
        return NULL;
    }
    size_t map_bytes = 0;
    char reason[128];
    if (map_weight && !h3_st_zero_copy_plan(file_offset, bytes,
            dtype == H3_GPU_BF16 ? H3_DTYPE_BF16 : H3_DTYPE_F32,
            (uint64_t)status.st_size, (size_t)getpagesize(),
            gpu.device.maxBufferLength, &map_bytes, reason, sizeof(reason))) {
        if (getenv("H3_PROFILE"))
            fprintf(stderr, "h3cli: %s: tensor at %llu (%zu bytes): zero-copy disabled; "
                    "per-tensor copied fallback (%s)\n", path,
                    (unsigned long long)file_offset, bytes, reason);
        map_weight = 0;
    }
    if (map_weight) @autoreleasepool {
        if(!h3_gpu_memory_reserve(gpu,map_bytes,"mapped Metal weight")){close(descriptor);return NULL;}
        void *mapping = mmap(NULL, map_bytes, PROT_READ | PROT_WRITE,
                             MAP_PRIVATE, descriptor, (off_t)file_offset);
        id<MTLBuffer> buffer = nil;
        if (mapping != MAP_FAILED) {
            buffer = [gpu.device newBufferWithBytesNoCopy:mapping length:map_bytes
                options:MTLResourceStorageModeShared
                deallocator:^(void *pointer, NSUInteger length) {
                    if (getenv("H3_PROFILE")) fprintf(stderr,"h3cli: released %zu-byte weight mapping\n", (size_t)length);
                    munmap(pointer, length);
                }];
            if (!buffer) munmap(mapping, map_bytes);
        }
        if (!buffer) {
            if (getenv("H3_PROFILE"))
                fprintf(stderr, "h3cli: %s: tensor at %llu: per-tensor copied fallback "
                        "(mapping allocation unavailable)\n", path,
                        (unsigned long long)file_offset);
        } else {
            close(descriptor);
            H3Tensor *tensor = [[H3Tensor alloc] init];
            tensor.elements = elements;
            tensor.bytes = bytes;
            tensor.dtype = dtype;
            tensor.buffer = buffer;
            tensor.fileMapped = YES;
            tensor.owner = gpu;
            h3_gpu_stats stats = gpu.stats;
            stats.allocated_bytes += bytes;
            stats.live_bytes += bytes;
            if (stats.live_bytes > stats.peak_live_bytes)
                stats.peak_live_bytes = stats.live_bytes;
            stats.tensor_allocations++;
            gpu.stats = stats;
            return (__bridge_retained h3_gpu_tensor *)tensor;
        }
    }
    h3_gpu_tensor *opaque_tensor = h3_gpu_tensor_new(
        opaque, NULL, elements, item_size, dtype);
    if (!opaque_tensor) { close(descriptor); return NULL; }
    H3Tensor *tensor = TENSOR(opaque_tensor);
    size_t remaining = bytes;
    size_t completed = 0;
    while (remaining) {
        size_t request = MIN(remaining, (size_t)SSIZE_MAX);
        ssize_t count = pread(descriptor,
                              (unsigned char *)tensor.buffer.contents + completed,
                              request, (off_t)(file_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            int detail = count < 0 ? errno : 0;
            h3_gpu_set_error(gpu, @"cannot read %s payload from %s: %s", label,
                             path, detail ? strerror(detail) :
                                            "unexpected end of file");
            close(descriptor);
            h3_gpu_tensor_free(opaque_tensor);
            return NULL;
        }
        completed += (size_t)count;
        remaining -= (size_t)count;
    }
    close(descriptor);
    return opaque_tensor;
    }
}

int h3_gpu_tensor_is_file_mapped(const h3_gpu_tensor *tensor) {
    return tensor && TENSOR(tensor).fileMapped;
}

h3_gpu_tensor *h3_gpu_tensor_load_bf16(h3_gpu *opaque, const char *path,
                                       uint64_t file_offset, size_t elements) {
    return h3_gpu_tensor_load_file(opaque, path, file_offset, elements,
                                   sizeof(uint16_t), H3_GPU_BF16, "BF16");
}

h3_gpu_tensor *h3_gpu_tensor_load_f32(h3_gpu *opaque, const char *path,
                                      uint64_t file_offset, size_t elements) {
    return h3_gpu_tensor_load_file(opaque, path, file_offset, elements,
                                   sizeof(float), H3_GPU_F32, "F32");
}

static int h3_gpu_tensor_read_file_bf16_mode(
                                 h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 int uncached,
                                 char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (!opaque || !path || !*path ||
        TENSOR(opaque).dtype != H3_GPU_BF16 ||
        elements != TENSOR(opaque).elements ||
        elements > SIZE_MAX / sizeof(uint16_t) || file_offset > INT64_MAX) {
        if (error && error_size)
            snprintf(error, error_size, "invalid BF16 file read request");
        return 0;
    }
    size_t bytes = elements * sizeof(uint16_t);
    if ((uint64_t)bytes > (uint64_t)INT64_MAX - file_offset) {
        if (error && error_size)
            snprintf(error, error_size, "BF16 file read range overflows");
        return 0;
    }
    int descriptor = open(path, O_RDONLY | O_CLOEXEC);
    if (descriptor < 0) {
        if (error && error_size)
            snprintf(error, error_size, "cannot open %s: %s", path,
                     strerror(errno));
        return 0;
    }
    struct stat status;
    if (fstat(descriptor, &status) || status.st_size < 0 ||
        !h3_st_file_range(file_offset, bytes, (uint64_t)status.st_size)) {
        if (error && error_size)
            snprintf(error, error_size, "%s: BF16 tensor range exceeds file", path);
        close(descriptor);
        return 0;
    }
#ifdef F_NOCACHE
    if (uncached) (void)fcntl(descriptor, F_NOCACHE, 1);
#else
    (void)uncached;
#endif
    unsigned char *destination = TENSOR(opaque).buffer.contents;
    size_t completed = 0;
    while (completed < bytes) {
        size_t request = MIN(bytes - completed, (size_t)SSIZE_MAX);
        ssize_t count = pread(descriptor, destination + completed, request,
                              (off_t)(file_offset + completed));
        if (count < 0 && errno == EINTR) continue;
        if (count <= 0) {
            int detail = count < 0 ? errno : 0;
            if (error && error_size) {
                snprintf(error, error_size, "cannot read BF16 payload from %s: %s",
                         path, detail ? strerror(detail) :
                                        "unexpected end of file");
            }
            close(descriptor);
            return 0;
        }
        completed += (size_t)count;
    }
    close(descriptor);
    return 1;
}

int h3_gpu_tensor_read_file_bf16(h3_gpu_tensor *opaque, const char *path,
                                 uint64_t file_offset, size_t elements,
                                 char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(
        opaque, path, file_offset, elements, 0, error, error_size);
}

int h3_gpu_tensor_stream_file_bf16(h3_gpu_tensor *opaque, const char *path,
                                   uint64_t file_offset, size_t elements,
                                   char *error, size_t error_size) {
    return h3_gpu_tensor_read_file_bf16_mode(
        opaque, path, file_offset, elements, 1, error, error_size);
}

void h3_gpu_tensor_free(h3_gpu_tensor *tensor) {
    if (!tensor) return;
    @autoreleasepool {
        H3Tensor *object = CFBridgingRelease(tensor);
        H3GPU *owner = object.owner;
        if (owner) {
            h3_gpu_stats stats = owner.stats;
            stats.live_bytes = stats.live_bytes >= object.bytes ?
                stats.live_bytes - object.bytes : 0;
            owner.stats = stats;
        }
        [object.buffer setPurgeableState:MTLPurgeableStateEmpty];
        object.graphData=nil;
        object.graphDataShape=nil;
        object.buffer = nil;
    }
}

size_t h3_gpu_tensor_elements(const h3_gpu_tensor *tensor) {
    return tensor ? TENSOR(tensor).elements : 0;
}

h3_gpu_dtype h3_gpu_tensor_dtype(const h3_gpu_tensor *tensor) {
    return tensor ? TENSOR(tensor).dtype : H3_GPU_F32;
}

int h3_gpu_tensor_read_f32(const h3_gpu_tensor *tensor, float *values,
                           size_t elements) {
    return h3_gpu_tensor_read_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_read_f32_range(const h3_gpu_tensor *tensor,
                                 size_t source_offset, float *values,
                                 size_t elements) {
    @autoreleasepool {
        if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_F32 ||
            source_offset > TENSOR(tensor).elements ||
            elements > TENSOR(tensor).elements - source_offset) return 0;
        const unsigned char *source = TENSOR(tensor).buffer.contents;
        memcpy(values, source + source_offset * sizeof(float),
               elements * sizeof(float));
        return 1;
    }
}

int h3_gpu_tensor_read_u32(const h3_gpu_tensor *tensor, uint32_t *values,
                           size_t elements) {
    @autoreleasepool {
        if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_U32 ||
            elements > TENSOR(tensor).elements) return 0;
        memcpy(values, TENSOR(tensor).buffer.contents, elements * sizeof(*values));
        return 1;
    }
}

int h3_gpu_tensor_read_bf16(const h3_gpu_tensor *tensor, uint16_t *values,
                            size_t elements) {
    @autoreleasepool {
        if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_BF16 ||
            elements > TENSOR(tensor).elements) return 0;
        memcpy(values, TENSOR(tensor).buffer.contents, elements * sizeof(uint16_t));
        return 1;
    }
}

int h3_gpu_tensor_write_f32(h3_gpu_tensor *tensor, const float *values,
                            size_t elements) {
    return h3_gpu_tensor_write_f32_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_f32_range(h3_gpu_tensor *tensor,
                                  size_t destination_offset,
                                  const float *values, size_t elements) {
    @autoreleasepool {
        if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_F32 ||
            destination_offset > TENSOR(tensor).elements ||
            elements > TENSOR(tensor).elements - destination_offset) return 0;
        unsigned char *destination = TENSOR(tensor).buffer.contents;
        memcpy(destination + destination_offset * sizeof(float), values,
               elements * sizeof(float));
        return 1;
    }
}

int h3_gpu_tensor_write_bf16(h3_gpu_tensor *tensor, const uint16_t *values,
                             size_t elements) {
    return h3_gpu_tensor_write_bf16_range(tensor, 0, values, elements);
}

int h3_gpu_tensor_write_bf16_range(h3_gpu_tensor *tensor,
                                   size_t destination_offset,
                                   const uint16_t *values, size_t elements) {
    @autoreleasepool {
        if (!tensor || !values || TENSOR(tensor).dtype != H3_GPU_BF16 ||
            destination_offset > TENSOR(tensor).elements ||
            elements > TENSOR(tensor).elements - destination_offset) return 0;
        unsigned char *destination = TENSOR(tensor).buffer.contents;
        memcpy(destination + destination_offset * sizeof(uint16_t), values,
               elements * sizeof(uint16_t));
        return 1;
    }
}

int h3_gpu_begin(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || gpu.command || gpu.inflightCommands.count) return 0;
    gpu.lastError = nil;
    @autoreleasepool {
        gpu.command = [gpu.queue commandBuffer];
    }
    if (!gpu.command) {
        h3_gpu_set_error(gpu, @"cannot create Metal command buffer");
        return 0;
    }
    gpu.commandStartWall = h3_gpu_now();
    return 1;
}

/* Completed commands retain their MPSGraph temporaries until the last owner
 * releases them. Keep one previous batch overlapped with the batch being
 * encoded; never accumulate every command in a denoising step. */
static int h3_gpu_retire_commands(H3GPU *gpu) {
    while(gpu.inflightCommands.count) {
        id<MTLCommandBuffer> oldest=gpu.inflightCommands[0];
        BOOL finished=oldest.status==MTLCommandBufferStatusCompleted||oldest.status==MTLCommandBufferStatusError;
        if(!finished&&gpu.inflightCommands.count<2)break;
        h3_gpu_stats stats=gpu.stats;
        if(!finished) {
            double started=h3_gpu_now();[oldest waitUntilCompleted];
            stats.command_wait_seconds+=h3_gpu_now()-started;stats.bounded_waits++;
        }
        if(oldest.status==MTLCommandBufferStatusError) {
            h3_gpu_set_error(gpu,@"Metal command failed: %@",oldest.error.localizedDescription);
            return 0;
        }
        if(oldest.GPUEndTime>=oldest.GPUStartTime)stats.gpu_seconds+=oldest.GPUEndTime-oldest.GPUStartTime;
        stats.command_retirements++;gpu.stats=stats;
        [gpu.inflightCommands removeObjectAtIndex:0];
    }
    return 1;
}

int h3_gpu_continue(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !gpu.command) return 0;
    if(!h3_gpu_require_command(gpu))return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = gpu.command;
        gpu.command = nil;
        gpu.mpsCommand = nil;
        double commit_time = h3_gpu_now();
        [command commit];
        [gpu.inflightCommands addObject:command];
        h3_gpu_stats stats = gpu.stats;
        stats.submissions++;
        stats.inflight_peak=MAX(stats.inflight_peak,gpu.inflightCommands.count);
        stats.command_encode_seconds += commit_time - gpu.commandStartWall;
        gpu.stats = stats;
        if(!h3_gpu_retire_commands(gpu))return 0;
        gpu.command = [gpu.queue commandBuffer];
    }
    if (!gpu.command) {
        h3_gpu_set_error(gpu, @"cannot continue Metal command chain");
        return 0;
    }
    gpu.commandStartWall = h3_gpu_now();
    return 1;
}

int h3_gpu_submit(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !gpu.command) return 0;
    if(!h3_gpu_require_command(gpu))return 0;
    @autoreleasepool {
        id<MTLCommandBuffer> command = gpu.command;
        gpu.command = nil;
        gpu.mpsCommand = nil;
        double commit_time = h3_gpu_now();
        [command commit];
        [gpu.inflightCommands addObject:command];
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands)
            [pending waitUntilCompleted];
        double complete_time = h3_gpu_now();
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands) {
            if (pending.status == MTLCommandBufferStatusError) {
                h3_gpu_set_error(gpu, @"Metal command failed: %@",
                                 pending.error.localizedDescription);
                [gpu.inflightCommands removeAllObjects];
                return 0;
            }
        }
        h3_gpu_stats stats = gpu.stats;
        stats.submissions++;
        stats.inflight_peak=MAX(stats.inflight_peak,gpu.inflightCommands.count);
        stats.command_retirements+=gpu.inflightCommands.count;
        stats.command_encode_seconds += commit_time - gpu.commandStartWall;
        stats.command_wait_seconds += complete_time - commit_time;
        for (id<MTLCommandBuffer> pending in gpu.inflightCommands) {
            if (pending.GPUEndTime >= pending.GPUStartTime)
                stats.gpu_seconds += pending.GPUEndTime - pending.GPUStartTime;
        }
        gpu.stats = stats;
        [gpu.inflightCommands removeAllObjects];
    }
    return 1;
}

int h3_gpu_synchronize(h3_gpu *opaque) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu) return 0;
    return gpu.command ? h3_gpu_submit(opaque) : 1;
}

const char *h3_gpu_error(const h3_gpu *opaque) {
    H3GPU *gpu = GPU((h3_gpu *)(void *)opaque);
    const char *message = gpu.lastError.UTF8String;
    return message ? message : "unknown Metal error";
}

uint64_t h3_gpu_device_allocated_bytes(void) {
    @autoreleasepool { return MTLCreateSystemDefaultDevice().currentAllocatedSize; }
}

int h3_gpu_get_stats(const h3_gpu *opaque, h3_gpu_stats *stats) {
    if (!opaque || !stats) return 0;
    *stats = GPU((h3_gpu *)(void *)opaque).stats;
    stats->metal_current_allocated_bytes = GPU((h3_gpu *)(void *)opaque).device.currentAllocatedSize;
    stats->linear_dispatches = stats->mps_linear_dispatches;
    stats->conv_dispatches = stats->mps_conv_dispatches;
    stats->attention_dispatches = stats->mps_sdpa_dispatches + stats->native_attention_dispatches;
    return 1;
}

void h3_gpu_profile_set_label(h3_gpu *opaque, const char *label) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !label || !*label) return;
    gpu.profileLabel = [NSString stringWithUTF8String:label];
}

void h3_gpu_profile_mark(h3_gpu *opaque, const char *phase) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu || !phase || !*phase || !h3_gpu_profile_enabled()) return;
    h3_gpu_profile_emit(gpu, [NSString stringWithUTF8String:phase],
                        gpu.profileMarkStats, gpu.profileMarkWall);
    gpu.profileMarkStats = gpu.stats;
    gpu.profileMarkWall = h3_gpu_now();
}

typedef struct { uint32_t rows, input_dim, output_dim, has_bias; } linear_args;
typedef struct { uint32_t rows, columns; float clip; } int8_quant_args;
typedef struct {
    uint32_t rows, padded_rows, heads, head_dim;
    float clip;
} int8_head_major_quant_args;
typedef struct { uint32_t rows, columns, group_size, groups; }
    int8_group_quant_args;
typedef struct { uint32_t rows, width; float epsilon; } norm_args;
typedef struct {
    uint32_t rows, width, slots, shift_slot, scale_slot;
    float epsilon;
} adaln_args;
typedef struct { uint32_t rows, width, slots, gate_slot; } gate_args;
typedef struct {
    uint32_t sequence, heads, head_dim, rope_half, grouped;
    float epsilon;
} qkv_args;
typedef struct { uint32_t outer, inner; } weight_norm_args;
typedef struct { uint32_t elements; float left_scale, right_scale; }
    add_scaled_args;
typedef struct { uint32_t batch, length, channels; } audio_activation_args;
typedef struct { uint32_t batch, length, heads, head_dim; } audio_qkv_args;
typedef struct {
    uint32_t batch, length, heads, head_dim, output_dim;
} audio_pool_args;
typedef struct { uint32_t elements; float minimum, maximum; } clip_args;
typedef struct {
    uint32_t batch, depth, height, width, channels, depth_front;
    uint32_t height_before, height_after, width_before, width_after;
} vae_encoder_pad_args;
typedef struct {
    uint32_t batch, depth, height, width, channels, groups;
    float epsilon;
} vae_encoder_norm_args;
typedef struct { uint32_t rows, width; } swiglu_args;
typedef struct { uint32_t elements, approximate; } gelu_bf16_args;
typedef struct { uint32_t tokens, vocab_size, width; } embedding_args;
typedef struct {
    uint32_t sequence, query_heads, kv_heads, head_dim;
    float epsilon;
} text_rope_args;
typedef struct { uint32_t sequence, heads, head_dim; float epsilon; } head_norm_args;
typedef struct { uint32_t sequence, query_heads, kv_heads, head_dim; } text_rope_inplace_args;
typedef struct {
    uint32_t sequence, query_heads, kv_heads, head_dim;
    float scale;
} gqa_args;
typedef struct { uint32_t sample_offset, elements; float delta, ratio; }
    euler_args;
typedef struct {
    uint32_t sample_offset, elements, row_width, classes;
    float delta, ratio;
} bridge_euler_args;

static int h3_gpu_linear_mps(H3GPU *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim,
                             MPSDataType dataType);

int h3_gpu_linear_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t input_dim, uint32_t output_dim) {
    @autoreleasepool {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"linear input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"linear weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"linear output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim, @"linear bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    if (!getenv("H3_SCALAR_PATCH") && rows >= 16 && output_dim == 5376 &&
        (input_dim == 32 || input_dim == 96)) {
        linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
        const h3_gpu_tensor *bias_buffer = bias ? bias : input;
        if (!h3_gpu_require_command(gpu)) return 0;
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, @"h3_linear_f32_tiled");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
            h3_gpu_set_error(gpu,
                             @"device cannot dispatch the F32 16x16 tile");
            return 0;
        }
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                      (rows + 15) / 16, 1)
                     threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
            [encoder endEncoding];
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
        return 1;
    }
    if (rows >= 32 && input_dim >= 256 && output_dim >= 256 &&
        h3_gpu_linear_mps(gpu, output, input, weight, bias, rows,
                          input_dim, output_dim, MPSDataTypeFloat32)) return 1;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    return h3_gpu_dispatch_2d(gpu, @"h3_linear_f32", output_dim, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
    }
}

int h3_gpu_patch_linear_bf16_offset(
                             h3_gpu *opaque, h3_gpu_tensor *output,
                             size_t output_offset,
                             const h3_gpu_tensor *input, size_t input_offset,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96)) {
        h3_gpu_set_error(gpu, @"unsupported fused patch projection shape");
        return 0;
    }
    if (input_offset > SIZE_MAX - input_count ||
        output_offset > SIZE_MAX - output_count ||
        input_offset > SIZE_MAX / sizeof(float) ||
        output_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"fused patch projection offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_elements(gpu, input, input_offset + input_count,
                                 @"patch projection input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"patch projection weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_offset + output_count,
                                 @"patch projection output") ||
        TENSOR(output).dtype != H3_GPU_BF16 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim,
                                           @"patch projection bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_linear_f32_tiled_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch the fused F32/BF16 tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer
                       offset:input_offset * sizeof(float) atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer
                       offset:output_offset * sizeof(uint16_t) atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_patch_linear_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    return h3_gpu_patch_linear_bf16_offset(
        opaque, output, 0, input, 0, weight, bias, rows, input_dim,
        output_dim);
}

int h3_gpu_patch_linear_bf16_map(
                             h3_gpu *opaque, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias,
                             const h3_gpu_tensor *row_map,
                             uint32_t output_rows, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)output_rows * output_dim;
    if (output_dim != 5376 || (input_dim != 32 && input_dim != 96)) {
        h3_gpu_set_error(gpu, @"unsupported mapped patch projection shape");
        return 0;
    }
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"mapped patch input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"mapped patch weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"mapped patch output") ||
        TENSOR(output).dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"mapped patch row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_dim,
                                           @"mapped patch bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_linear_f32_tiled_bf16_map");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch the mapped patch tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:5];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_silu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"SiLU input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"SiLU output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_cast_f32_to_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"cast input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"cast output") ||
        TENSOR(output).dtype != H3_GPU_BF16) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_f32_to_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_cast_bf16_to_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                            const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, input, elements, @"cast input") ||
        TENSOR(input).dtype != H3_GPU_BF16 ||
        !h3_gpu_require_elements(gpu, output, elements, @"cast output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_cast_bf16_to_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_copy_bf16(h3_gpu *opaque, h3_gpu_tensor *destination,
                     size_t destination_offset,
                     const h3_gpu_tensor *source, size_t source_offset,
                     size_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_command(gpu) || !destination || !source ||
        TENSOR(destination).dtype != H3_GPU_BF16 ||
        TENSOR(source).dtype != H3_GPU_BF16 ||
        source_offset > TENSOR(source).elements ||
        elements > TENSOR(source).elements - source_offset ||
        destination_offset > TENSOR(destination).elements ||
        elements > TENSOR(destination).elements - destination_offset ||
        elements > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"invalid BF16 blit range");
        return 0;
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> encoder = [gpu.command blitCommandEncoder];
        [encoder copyFromBuffer:TENSOR(source).buffer
                   sourceOffset:source_offset * sizeof(uint16_t)
                       toBuffer:TENSOR(destination).buffer
              destinationOffset:destination_offset * sizeof(uint16_t)
                           size:elements * sizeof(uint16_t)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.blit_copies++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_copy_f32(h3_gpu *opaque, h3_gpu_tensor *destination,
                    size_t destination_offset,
                    const h3_gpu_tensor *source, size_t source_offset,
                    size_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_command(gpu) || !destination || !source ||
        TENSOR(destination).dtype != H3_GPU_F32 ||
        TENSOR(source).dtype != H3_GPU_F32 ||
        source_offset > TENSOR(source).elements ||
        elements > TENSOR(source).elements - source_offset ||
        destination_offset > TENSOR(destination).elements ||
        elements > TENSOR(destination).elements - destination_offset ||
        elements > SIZE_MAX / sizeof(float)) {
        h3_gpu_set_error(gpu, @"invalid F32 blit range");
        return 0;
    }
    @autoreleasepool {
        id<MTLBlitCommandEncoder> encoder = [gpu.command blitCommandEncoder];
        [encoder copyFromBuffer:TENSOR(source).buffer
                   sourceOffset:source_offset * sizeof(float)
                       toBuffer:TENSOR(destination).buffer
              destinationOffset:destination_offset * sizeof(float)
                           size:elements * sizeof(float)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.blit_copies++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_rms_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *weight, uint32_t rows,
                        uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"RMSNorm input") ||
        !h3_gpu_require_elements(gpu, weight, width, @"RMSNorm weight") ||
        !h3_gpu_require_elements(gpu, output, count, @"RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_rms_norm_f32", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_adaln_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t shift_slot,
                     uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"AdaLN input") ||
        !h3_gpu_require_elements(gpu, norm_weight, width, @"AdaLN norm") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"AdaLN row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"AdaLN output") ||
        !modulation || shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_adaln_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_gate_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *residual,
                    const h3_gpu_tensor *branch,
                    const h3_gpu_tensor *modulation,
                    const h3_gpu_tensor *row_map, uint32_t rows,
                    uint32_t width, uint32_t slots, uint32_t gate_slot) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, @"gate residual") ||
        !h3_gpu_require_elements(gpu, branch, count, @"gate branch") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"gate row map") ||
        !h3_gpu_require_elements(gpu, output, count, @"gate output") ||
        !modulation || gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, @"h3_gate_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                        h3_gpu_tensor *key, h3_gpu_tensor *value,
                        const h3_gpu_tensor *qkv,
                        const h3_gpu_tensor *q_norm,
                        const h3_gpu_tensor *k_norm,
                        const h3_gpu_tensor *rope_cos,
                        const h3_gpu_tensor *rope_sin, uint32_t sequence,
                        uint32_t heads, uint32_t head_dim,
                        uint32_t rope_half, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"QKV input") ||
        !h3_gpu_require_elements(gpu, q_norm, head_dim, @"Q norm") ||
        !h3_gpu_require_elements(gpu, k_norm, head_dim, @"K norm") ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"RoPE cosine") ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"RoPE sine") ||
        !h3_gpu_require_elements(gpu, query, count, @"query") ||
        !h3_gpu_require_elements(gpu, key, count, @"key") ||
        !h3_gpu_require_elements(gpu, value, count, @"value") ||
        rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

static H3SDPA *h3_gpu_sdpa_graph(H3GPU *gpu, uint32_t batch,
                                 uint32_t sequence,
                                 uint32_t heads, uint32_t head_dim, float scale,
                                 MPSDataType dataType, int causal,
                                 int headMajor, int outputHeadMajor) {
    @autoreleasepool {
        NSString *cacheKey = [NSString stringWithFormat:
                              @"%u:%u:%u:%u:%u:%.9g:%d:%d:%d",
                              (unsigned)dataType, batch, sequence, heads,
                              head_dim, scale, causal, headMajor,
                              outputHeadMajor];
        H3SDPA *cached = gpu.sdpaCache[cacheKey];
        if (cached) return cached;
        MPSGraph *graph = [[MPSGraph alloc] init];
        NSArray<NSNumber *> *rowMajorShape =
            @[@(batch), @(sequence), @(heads), @(head_dim)];
        NSArray<NSNumber *> *headMajorShape =
            @[@(batch), @(heads), @(sequence), @(head_dim)];
        NSArray<NSNumber *> *outputShape = outputHeadMajor ?
            headMajorShape : rowMajorShape;
        NSArray<NSNumber *> *inputShape = headMajor ?
            headMajorShape : rowMajorShape;
        MPSGraphTensor *q = [graph placeholderWithShape:inputShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *k = [graph placeholderWithShape:inputShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *v = [graph placeholderWithShape:inputShape
                                               dataType:dataType name:nil];
        MPSGraphTensor *qt = headMajor ? q :
            [graph transposeTensor:q dimension:1 withDimension:2 name:nil];
        MPSGraphTensor *kt = headMajor ? k :
            [graph transposeTensor:k dimension:1 withDimension:2 name:nil];
        MPSGraphTensor *vt = headMajor ? v :
            [graph transposeTensor:v dimension:1 withDimension:2 name:nil];
        if (![graph respondsToSelector:@selector(scaledDotProductAttentionWithQueryTensor:keyTensor:valueTensor:scale:name:)]) {
            h3_gpu_set_error(gpu, @"native MPSGraph SDPA is unavailable");
            return nil;
        }
        MPSGraphTensor *attention;
        if (causal) {
            size_t mask_count = (size_t)sequence * sequence;
            float *mask_values = malloc(mask_count * sizeof(*mask_values));
            if (!mask_values) return nil;
            for (uint32_t row = 0; row < sequence; row++)
                for (uint32_t column = 0; column < sequence; column++)
                    mask_values[(size_t)row * sequence + column] =
                        column <= row ? 0.0f : -INFINITY;
            NSData *mask_data = [NSData dataWithBytesNoCopy:mask_values
                length:mask_count * sizeof(*mask_values) freeWhenDone:YES];
            MPSGraphTensor *mask = [graph constantWithData:mask_data
                shape:@[@1, @1, @(sequence), @(sequence)]
                dataType:MPSDataTypeFloat32];
            attention = [graph
                scaledDotProductAttentionWithQueryTensor:qt keyTensor:kt
                valueTensor:vt maskTensor:mask scale:scale name:nil];
        } else {
            attention = [graph scaledDotProductAttentionWithQueryTensor:qt
                keyTensor:kt valueTensor:vt scale:scale name:nil];
        }
        H3SDPA *result = [[H3SDPA alloc] init];
        result.graph = graph;
        result.query = q;
        result.key = k;
        result.value = v;
        result.output = outputHeadMajor ? attention :
            [graph transposeTensor:attention dimension:1 withDimension:2
             name:nil];
        result.inputShape = inputShape;
        result.outputShape = outputShape;
        gpu.sdpaCache[cacheKey] = result;
        return result;
    }
}

/* Large D=72 vision SDPA can stall inside MPSGraph. Pad only the head dimension
 * for the existing bounded dense BF16 kernel, preserving the original scale
 * and complete image sequence. Complete consumers before freeing scratch:
 * tensor release makes Metal storage purgeable, even when commands retain it. */
static int h3_gpu_large_vision_attention(h3_gpu *opaque,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t sequence,uint32_t heads,float scale) {
    H3GPU *gpu=GPU(opaque);
    size_t count=(size_t)sequence*heads*128;
    if(count>UINT32_MAX)return 0;
    const h3_gpu_tensor *inputs[]={q,k,v};
    h3_gpu_tensor *padded[4]={NULL,NULL,NULL,NULL};
    int ok=1;
    for(int i=0;i<4&&ok;i++)ok=(padded[i]=h3_gpu_tensor_new_bf16(opaque,count))!=NULL;
    for(int i=0;i<3&&ok;i++) {
        uint32_t n=(uint32_t)count;
        const h3_gpu_tensor *input=inputs[i];h3_gpu_tensor *output=padded[i];
        ok=h3_gpu_dispatch_1d(gpu,@"h3_vision_attention_pad",n,^(id<MTLComputeCommandEncoder> e){
            [e setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [e setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [e setBytes:&n length:sizeof(n) atIndex:2];
        });
    }
    if(ok)ok=h3_gpu_native_sdpa_bf16(opaque,padded[3],padded[0],padded[1],padded[2],
                                    sequence,heads,128,scale,0,0,-1);
    if(ok) {
        uint32_t n=sequence*heads*72;
        h3_gpu_tensor *input=padded[3];
        ok=h3_gpu_dispatch_1d(gpu,@"h3_vision_attention_unpad",n,^(id<MTLComputeCommandEncoder> e){
            [e setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [e setBuffer:TENSOR(out).buffer offset:0 atIndex:1];
            [e setBytes:&n length:sizeof(n) atIndex:2];
        });
    }
    if(ok)ok=h3_gpu_submit(opaque);
    if(!ok)h3_gpu_cancel(opaque);
    for(int i=0;i<4;i++)h3_gpu_tensor_free(padded[i]);
    return ok && h3_gpu_begin(opaque);
}

static int h3_gpu_sdpa(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                       const h3_gpu_tensor *value, uint32_t batch,
                       uint32_t sequence,
                       uint32_t heads, uint32_t head_dim, float scale,
                       h3_gpu_dtype tensor_dtype, MPSDataType mps_dtype,
                       int causal, int outputHeadMajor) {
    @autoreleasepool {
    H3GPU *gpu = GPU(opaque);
    int headMajor = gpu.headMajorSDPAInputs &&
        tensor_dtype == H3_GPU_BF16 && batch == 1 && !causal;
    gpu.headMajorSDPAInputs = NO;
    size_t count = (size_t)batch * sequence * heads * head_dim;
    if (!batch || !sequence || !heads || !head_dim ||
        !h3_gpu_require_command(gpu) ||
        !h3_gpu_require_elements(gpu, query, count, @"SDPA query") ||
        !h3_gpu_require_elements(gpu, key, count, @"SDPA key") ||
        !h3_gpu_require_elements(gpu, value, count, @"SDPA value") ||
        !h3_gpu_require_elements(gpu, output, count, @"SDPA output")) return 0;
    if (TENSOR(query).dtype != tensor_dtype || TENSOR(key).dtype != tensor_dtype ||
        TENSOR(value).dtype != tensor_dtype || TENSOR(output).dtype != tensor_dtype) {
        h3_gpu_set_error(gpu, @"SDPA tensor dtype mismatch");
        return 0;
    }
    if(batch==1&&sequence>32768&&head_dim==72&&tensor_dtype==H3_GPU_BF16&&
       !causal&&!headMajor&&!outputHeadMajor)
        return h3_gpu_large_vision_attention(opaque,output,query,key,value,sequence,heads,scale);
    if(count>UINT64_MAX/24||!h3_gpu_memory_reserve(gpu,(uint64_t)count*24,
                                               "MPSGraph attention workspace"))return 0;
    H3SDPA *cache = h3_gpu_sdpa_graph(gpu, batch, sequence, heads, head_dim,
                                      scale, mps_dtype, causal, headMajor,
                                      outputHeadMajor);
    if (!cache) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *(^data)(const h3_gpu_tensor *) =
            ^MPSGraphTensorData *(const h3_gpu_tensor *tensor) {
                return h3_gpu_graph_data(tensor, cache.inputShape,
                                         mps_dtype, 0);
            };
        NSDictionary *feeds = @{
            cache.query: data(query), cache.key: data(key), cache.value: data(value)
        };
        MPSGraphTensorData *outputData = h3_gpu_graph_data(
            output, cache.outputShape, mps_dtype, 0);
        NSDictionary *results = @{cache.output: outputData};
        @try {
            [cache.graph encodeToCommandBuffer:command feeds:feeds targetOperations:nil
                             resultsDictionary:results executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph SDPA failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_sdpa_dispatches++;
    gpu.stats = stats;
    return 1;
    }
}

int h3_gpu_sdpa_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t sequence,
                    uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_F32, MPSDataTypeFloat32, 0, 0);
}

int h3_gpu_sdpa_causal_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                    const h3_gpu_tensor *value, uint32_t batch,
                    uint32_t sequence, uint32_t heads, uint32_t head_dim,
                    float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, batch, sequence,
                       heads, head_dim, scale, H3_GPU_F32,
                       MPSDataTypeFloat32, 1, 0);
}

int h3_gpu_sdpa_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_BF16, MPSDataTypeBFloat16, 0,
                       0);
}

int h3_gpu_sdpa_bf16_head_major_output(
                     h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *query, const h3_gpu_tensor *key,
                     const h3_gpu_tensor *value, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim, float scale) {
    return h3_gpu_sdpa(opaque, output, query, key, value, 1, sequence, heads,
                       head_dim, scale, H3_GPU_BF16, MPSDataTypeBFloat16, 0,
                       1);
}

int h3_gpu_swiglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *fused, uint32_t rows,
                      uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, fused, (size_t)rows * width * 2, @"SwiGLU input") ||
        !h3_gpu_require_elements(gpu, output, (size_t)rows * width, @"SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_scale_add_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *residual,
                         const h3_gpu_tensor *branch,
                         const h3_gpu_tensor *scale, uint32_t rows,
                         uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, residual, count, @"scale-add residual") ||
        TENSOR(residual).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, branch, count, @"scale-add branch") ||
        TENSOR(branch).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, scale, width, @"scale-add scale") ||
        TENSOR(scale).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"scale-add output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_scale_add_f32", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(scale).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_layer_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *bias, uint32_t rows,
                          uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_elements(gpu, input, count, @"LayerNorm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, width, @"LayerNorm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, width, @"LayerNorm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"LayerNorm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_layer_norm_f32", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_video_qkv_rope_f32(h3_gpu *opaque, h3_gpu_tensor *query,
                              h3_gpu_tensor *key, h3_gpu_tensor *value,
                              const h3_gpu_tensor *qkv,
                              const h3_gpu_tensor *rope_cos,
                              const h3_gpu_tensor *rope_sin,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, uint32_t rope_half,
                              float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_elements(gpu, qkv, count * 3, @"video QKV") ||
        TENSOR(qkv).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_cos, rope_count, @"video RoPE cosine") ||
        TENSOR(rope_cos).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin, rope_count, @"video RoPE sine") ||
        TENSOR(rope_sin).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, @"video query") ||
        TENSOR(query).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, key, count, @"video key") ||
        TENSOR(key).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, value, count, @"video value") ||
        TENSOR(value).dtype != H3_GPU_F32 || rope_half * 2 > head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_video_qkv_rope_f32",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

static H3Conv *h3_gpu_conv_graph(H3GPU *gpu, uint32_t batch,
                                 uint32_t length, uint32_t input_channels,
                                 uint32_t output_channels, uint32_t kernel,
                                 uint32_t stride, uint32_t padding,
                                 uint32_t dilation, uint32_t output_length,
                                 int transpose, int has_bias) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:
            @"%d:%u:%u:%u:%u:%u:%u:%u:%u:%d", transpose, batch, length,
            input_channels, output_channels, kernel, stride, padding,
            dilation, has_bias];
        H3Conv *cached = gpu.convCache[key];
        if (cached) return cached;

        H3Conv *conv = [[H3Conv alloc] init];
        conv.graph = [[MPSGraph alloc] init];
        conv.inputShape = @[@(batch), @1, @(length), @(input_channels)];
        conv.weightShape = transpose ?
            @[@(input_channels), @(output_channels), @1, @(kernel)] :
            @[@(output_channels), @(input_channels), @1, @(kernel)];
        conv.biasShape = @[@1, @1, @1, @(output_channels)];
        conv.outputShape = @[@(batch), @1, @(output_length),
                             @(output_channels)];
        conv.input = [conv.graph placeholderWithShape:conv.inputShape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        conv.weight = [conv.graph placeholderWithShape:conv.weightShape
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
        MPSGraphConvolution2DOpDescriptor *descriptor =
            [MPSGraphConvolution2DOpDescriptor
                descriptorWithStrideInX:stride strideInY:1
                dilationRateInX:dilation dilationRateInY:1 groups:1
                paddingLeft:padding paddingRight:padding
                paddingTop:0 paddingBottom:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNHWC
                weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
        MPSGraphTensor *result = transpose ?
            [conv.graph convolutionTranspose2DWithSourceTensor:conv.input
                 weightsTensor:conv.weight outputShape:conv.outputShape
                 descriptor:descriptor name:nil] :
            [conv.graph convolution2DWithSourceTensor:conv.input
                 weightsTensor:conv.weight descriptor:descriptor name:nil];
        if (has_bias) {
            conv.bias = [conv.graph placeholderWithShape:conv.biasShape
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
            result = [conv.graph additionWithPrimaryTensor:result
                                           secondaryTensor:conv.bias name:nil];
        }
        conv.output = result;
        gpu.convCache[key] = conv;
        return conv;
    }
}

static int h3_gpu_conv_mps(H3GPU *gpu, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t batch,
                           uint32_t length, uint32_t input_channels,
                           uint32_t output_channels, uint32_t kernel,
                           uint32_t stride, uint32_t padding,
                           uint32_t dilation, uint32_t output_length,
                           int transpose) {
    @autoreleasepool {
    if (!h3_gpu_require_command(gpu)) return 0;
    if(!h3_gpu_memory_reserve(gpu,(TENSOR(input).bytes+(uint64_t)TENSOR(output).bytes)*2,
                             "MPSGraph convolution workspace"))return 0;
    H3Conv *conv = h3_gpu_conv_graph(
        gpu, batch, length, input_channels, output_channels, kernel, stride,
        padding, dilation, output_length, transpose, bias != NULL);
    if (!conv) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(input).buffer shape:conv.inputShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *weight_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(weight).buffer shape:conv.weightShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:conv.outputShape
            dataType:MPSDataTypeFloat32];
        NSMutableDictionary *feeds = [@{conv.input: input_data,
                                         conv.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:TENSOR(bias).buffer shape:conv.biasShape
                dataType:MPSDataTypeFloat32];
            feeds[conv.bias] = bias_data;
        }
        NSDictionary *results = @{conv.output: output_data};
        @try {
            [conv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph Conv1d failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches++;
    gpu.stats = stats;
    return 1;
    }
}

static H3Conv *h3_gpu_conv3d_graph(
        H3GPU *gpu, uint32_t batch, uint32_t depth, uint32_t height,
        uint32_t width, uint32_t input_channels, uint32_t output_channels,
        uint32_t kernel_depth, uint32_t kernel_height, uint32_t kernel_width,
        uint32_t stride_depth, uint32_t stride_height, uint32_t stride_width,
        uint32_t output_depth, uint32_t output_height, uint32_t output_width,
        int has_bias) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:
            @"3:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%u:%d", batch, depth,
            height, width, input_channels, output_channels, kernel_depth,
            kernel_height, kernel_width, stride_depth, stride_height,
            stride_width, has_bias];
        H3Conv *cached = gpu.convCache[key];
        if (cached) return cached;
        H3Conv *conv = [[H3Conv alloc] init];
        conv.graph = [[MPSGraph alloc] init];
        conv.inputShape = @[@(batch), @(depth), @(height), @(width),
                            @(input_channels)];
        conv.weightShape = @[@(output_channels), @(input_channels),
                             @(kernel_depth), @(kernel_height), @(kernel_width)];
        conv.biasShape = @[@1, @1, @1, @1, @(output_channels)];
        conv.outputShape = @[@(batch), @(output_depth), @(output_height),
                             @(output_width), @(output_channels)];
        conv.input = [conv.graph placeholderWithShape:conv.inputShape
                                              dataType:MPSDataTypeFloat32
                                                  name:nil];
        conv.weight = [conv.graph placeholderWithShape:conv.weightShape
                                               dataType:MPSDataTypeFloat32
                                                   name:nil];
        MPSGraphConvolution3DOpDescriptor *descriptor =
            [MPSGraphConvolution3DOpDescriptor
                descriptorWithStrideInX:stride_width
                strideInY:stride_height strideInZ:stride_depth
                dilationRateInX:1 dilationRateInY:1 dilationRateInZ:1 groups:1
                paddingLeft:0 paddingRight:0 paddingTop:0 paddingBottom:0
                paddingFront:0 paddingBack:0
                paddingStyle:MPSGraphPaddingStyleExplicit
                dataLayout:MPSGraphTensorNamedDataLayoutNDHWC
                weightsLayout:MPSGraphTensorNamedDataLayoutOIDHW];
        MPSGraphTensor *result = [conv.graph
            convolution3DWithSourceTensor:conv.input weightsTensor:conv.weight
            descriptor:descriptor name:nil];
        if (has_bias) {
            conv.bias = [conv.graph placeholderWithShape:conv.biasShape
                                                dataType:MPSDataTypeFloat32
                                                    name:nil];
            result = [conv.graph additionWithPrimaryTensor:result
                                           secondaryTensor:conv.bias name:nil];
        }
        conv.output = result;
        gpu.convCache[key] = conv;
        return conv;
    }
}

int h3_gpu_conv3d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t input_channels, uint32_t output_channels,
                      uint32_t kernel_depth, uint32_t kernel_height,
                      uint32_t kernel_width, uint32_t stride_depth,
                      uint32_t stride_height, uint32_t stride_width) {
    @autoreleasepool {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !depth || !height || !width || !input_channels ||
        !output_channels || !kernel_depth || !kernel_height || !kernel_width ||
        !stride_depth || !stride_height || !stride_width ||
        depth < kernel_depth || height < kernel_height || width < kernel_width)
        return 0;
    uint32_t output_depth = (depth - kernel_depth) / stride_depth + 1;
    uint32_t output_height = (height - kernel_height) / stride_height + 1;
    uint32_t output_width = (width - kernel_width) / stride_width + 1;
    size_t input_count = (size_t)batch * depth * height * width * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels *
                          kernel_depth * kernel_height * kernel_width;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"Conv3d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"Conv3d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"Conv3d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"Conv3d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32)) ||
        !h3_gpu_require_command(gpu)) return 0;
    if(!h3_gpu_memory_reserve(gpu,(TENSOR(input).bytes+(uint64_t)TENSOR(output).bytes)*2,
                             "MPSGraph Conv3d workspace"))return 0;
    H3Conv *conv = h3_gpu_conv3d_graph(
        gpu, batch, depth, height, width, input_channels, output_channels,
        kernel_depth, kernel_height, kernel_width, stride_depth, stride_height,
        stride_width, output_depth, output_height, output_width, bias != NULL);
    if (!conv) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(input).buffer shape:conv.inputShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *weight_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(weight).buffer shape:conv.weightShape
            dataType:MPSDataTypeFloat32];
        MPSGraphTensorData *output_data = [[MPSGraphTensorData alloc]
            initWithMTLBuffer:TENSOR(output).buffer shape:conv.outputShape
            dataType:MPSDataTypeFloat32];
        NSMutableDictionary *feeds = [@{conv.input: input_data,
                                         conv.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = [[MPSGraphTensorData alloc]
                initWithMTLBuffer:TENSOR(bias).buffer shape:conv.biasShape
                dataType:MPSDataTypeFloat32];
            feeds[conv.bias] = bias_data;
        }
        NSDictionary *results = @{conv.output: output_data};
        @try {
            [conv.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph Conv3d failed: %@",
                             exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_conv_dispatches++;
    gpu.stats = stats;
    return 1;
    }
}

int h3_gpu_conv1d_stride_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding,
                      uint32_t dilation) {
    H3GPU *gpu = GPU(opaque);
    uint64_t effective = (uint64_t)dilation * (kernel - 1) + 1;
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || !dilation || (uint64_t)length + 2 * padding < effective)
        return 0;
    uint32_t output_length = (uint32_t)(((uint64_t)length + 2 * padding -
                                         effective) / stride + 1);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)output_channels * input_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count, @"Conv1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count, @"Conv1d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count, @"Conv1d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"Conv1d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    return h3_gpu_conv_mps(gpu, output, input, weight, bias, batch, length,
        input_channels, output_channels, kernel, stride, padding, dilation,
        output_length, 0);
}

int h3_gpu_conv1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t padding, uint32_t dilation) {
    return h3_gpu_conv1d_stride_f32(opaque, output, input, weight, bias,
        batch, length, input_channels, output_channels, kernel, 1, padding,
        dilation);
}

int h3_gpu_conv_transpose1d_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t length, uint32_t input_channels,
                      uint32_t output_channels, uint32_t kernel,
                      uint32_t stride, uint32_t padding) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !length || !input_channels || !output_channels || !kernel ||
        !stride || (uint64_t)(length - 1) * stride + kernel < 2 * padding)
        return 0;
    uint32_t output_length = (uint32_t)((uint64_t)(length - 1) * stride +
                                        kernel - 2 * padding);
    size_t input_count = (size_t)batch * length * input_channels;
    size_t weight_count = (size_t)input_channels * output_channels * kernel;
    size_t output_count = (size_t)batch * output_length * output_channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"ConvTranspose1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, weight_count,
                                 @"ConvTranspose1d weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"ConvTranspose1d output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        (bias && (!h3_gpu_require_elements(gpu, bias, output_channels,
                                           @"ConvTranspose1d bias") ||
                  TENSOR(bias).dtype != H3_GPU_F32))) return 0;
    return h3_gpu_conv_mps(gpu, output, input, weight, bias, batch, length,
        input_channels, output_channels, kernel, stride, padding, 1,
        output_length, 1);
}

int h3_gpu_weight_norm_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *vector,
                           const h3_gpu_tensor *magnitude,
                           uint32_t outer, uint32_t inner) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)outer * inner;
    if (!outer || !inner ||
        !h3_gpu_require_elements(gpu, vector, count, @"weight-norm vector") ||
        TENSOR(vector).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, magnitude, outer,
                                 @"weight-norm magnitude") ||
        TENSOR(magnitude).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"weight-norm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    weight_norm_args args = {outer, inner};
    return h3_gpu_dispatch_1d(gpu, @"h3_weight_norm_f32", outer,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(vector).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(magnitude).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_add_scaled_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *left,
                          const h3_gpu_tensor *right, float left_scale,
                          float right_scale, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, left, elements, @"scaled-add left") ||
        TENSOR(left).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, right, elements, @"scaled-add right") ||
        TENSOR(right).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"scaled-add output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    add_scaled_args args = {elements, left_scale, right_scale};
    return h3_gpu_dispatch_1d(gpu, @"h3_add_scaled_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_alias_free_snake_f32(
                          h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *input,
                          const h3_gpu_tensor *alpha_log,
                          const h3_gpu_tensor *beta_log,
                          const h3_gpu_tensor *upsample_filter,
                          const h3_gpu_tensor *downsample_filter,
                          uint32_t batch, uint32_t length,
                          uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels ||
        !h3_gpu_require_elements(gpu, input, count, @"Snake input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"Snake output") ||
        TENSOR(output).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha_log, channels, @"Snake alpha") ||
        TENSOR(alpha_log).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, beta_log, channels, @"Snake beta") ||
        TENSOR(beta_log).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, upsample_filter, 12,
                                 @"Snake upsample filter") ||
        TENSOR(upsample_filter).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, downsample_filter, 12,
                                 @"Snake downsample filter") ||
        TENSOR(downsample_filter).dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_3d(gpu, @"h3_alias_free_snake_f32",
        MTLSizeMake(channels, length, batch),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(alpha_log).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(beta_log).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(upsample_filter).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(downsample_filter).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_snake1d_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *alpha, uint32_t batch,
                       uint32_t length, uint32_t channels) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * length * channels;
    if (!batch || !length || !channels || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, input, count, @"Snake1d input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, alpha, channels, @"Snake1d alpha") ||
        TENSOR(alpha).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count, @"Snake1d output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    audio_activation_args args = {batch, length, channels};
    return h3_gpu_dispatch_1d(gpu, @"h3_snake1d_f32", (uint32_t)count,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(alpha).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_audio_qkv_split_f32(h3_gpu *opaque,
                       h3_gpu_tensor *query, h3_gpu_tensor *key,
                       h3_gpu_tensor *value, const h3_gpu_tensor *qkv,
                       const h3_gpu_tensor *q_bias,
                       const h3_gpu_tensor *k_bias,
                       const h3_gpu_tensor *v_bias, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t width = (size_t)heads * head_dim;
    size_t count = (size_t)batch * length * width;
    if (!batch || !length || !heads || !head_dim || count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, qkv, count * 3, @"audio QKV") ||
        TENSOR(qkv).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, q_bias, width, @"audio Q bias") ||
        !h3_gpu_require_elements(gpu, k_bias, width, @"audio K bias") ||
        !h3_gpu_require_elements(gpu, v_bias, width, @"audio V bias") ||
        TENSOR(q_bias).dtype != H3_GPU_F32 ||
        TENSOR(k_bias).dtype != H3_GPU_F32 ||
        TENSOR(v_bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, query, count, @"audio query") ||
        !h3_gpu_require_elements(gpu, key, count, @"audio key") ||
        !h3_gpu_require_elements(gpu, value, count, @"audio value") ||
        TENSOR(query).dtype != H3_GPU_F32 || TENSOR(key).dtype != H3_GPU_F32 ||
        TENSOR(value).dtype != H3_GPU_F32) return 0;
    audio_qkv_args args = {batch, length, heads, head_dim};
    return h3_gpu_dispatch_1d(gpu, @"h3_audio_qkv_split_f32",
        (uint32_t)count, ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_bias).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(v_bias).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:6];
            [encoder setBytes:&args length:sizeof(args) atIndex:7];
        });
}

int h3_gpu_audio_attention_pool_f32(h3_gpu *opaque,
                       h3_gpu_tensor *output,
                       const h3_gpu_tensor *attended, uint32_t batch,
                       uint32_t length, uint32_t heads,
                       uint32_t head_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)batch * length * heads * head_dim;
    size_t output_count = (size_t)batch * length * output_dim;
    if (!batch || !length || !heads || !head_dim || !output_dim ||
        head_dim % output_dim || output_count > UINT32_MAX ||
        !h3_gpu_require_elements(gpu, attended, input_count,
                                 @"audio attended values") ||
        TENSOR(attended).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"audio pooled values") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    audio_pool_args args = {batch, length, heads, head_dim, output_dim};
    return h3_gpu_dispatch_1d(gpu, @"h3_audio_attention_pool_f32",
        (uint32_t)output_count, ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(attended).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_geglu_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *gate,
                     const h3_gpu_tensor *linear, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_elements(gpu, gate, elements, @"GeGLU gate") ||
        !h3_gpu_require_elements(gpu, linear, elements, @"GeGLU linear") ||
        !h3_gpu_require_elements(gpu, output, elements, @"GeGLU output") ||
        TENSOR(gate).dtype != H3_GPU_F32 ||
        TENSOR(linear).dtype != H3_GPU_F32 ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_geglu_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(gate).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(linear).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_clip_f32(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t elements,
                    float minimum, float maximum) {
    H3GPU *gpu = GPU(opaque);
    if (!(minimum <= maximum) ||
        !h3_gpu_require_elements(gpu, input, elements, @"clip input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, elements, @"clip output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    clip_args args = {elements, minimum, maximum};
    return h3_gpu_dispatch_1d(gpu, @"h3_clip_f32", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vae_encoder_pad_f32(
                    h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input, uint32_t batch,
                    uint32_t depth, uint32_t height, uint32_t width,
                    uint32_t channels, uint32_t depth_front,
                    uint32_t height_before, uint32_t height_after,
                    uint32_t width_before, uint32_t width_after) {
    H3GPU *gpu = GPU(opaque);
    if (!batch || !depth || height < 2 || width < 2 || !channels ||
        height_before >= height || height_after >= height ||
        width_before >= width || width_after >= width) return 0;
    uint32_t output_depth = depth + depth_front;
    uint32_t output_height = height + height_before + height_after;
    uint32_t output_width = width + width_before + width_after;
    size_t input_count = (size_t)batch * depth * height * width * channels;
    size_t output_count = (size_t)batch * output_depth * output_height *
                          output_width * channels;
    if (!h3_gpu_require_elements(gpu, input, input_count,
                                 @"VAE encoder pad input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, output_count,
                                 @"VAE encoder pad output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    vae_encoder_pad_args args = {
        batch, depth, height, width, channels, depth_front,
        height_before, height_after, width_before, width_after
    };
    return h3_gpu_dispatch_3d(gpu, @"h3_vae_encoder_pad_f32",
        MTLSizeMake(channels, output_width,
                    (NSUInteger)batch * output_depth * output_height),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vae_encoder_group_norm_silu_f32(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t batch,
                      uint32_t depth, uint32_t height, uint32_t width,
                      uint32_t channels, uint32_t groups, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)batch * depth * height * width * channels;
    if (!batch || !depth || !height || !width || !channels || !groups ||
        channels % groups || !(epsilon > 0.0f) ||
        !h3_gpu_require_elements(gpu, input, count,
                                 @"VAE encoder norm input") ||
        TENSOR(input).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, weight, channels,
                                 @"VAE encoder norm weight") ||
        TENSOR(weight).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, bias, channels,
                                 @"VAE encoder norm bias") ||
        TENSOR(bias).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, output, count,
                                 @"VAE encoder norm output") ||
        TENSOR(output).dtype != H3_GPU_F32) return 0;
    vae_encoder_norm_args args = {
        batch, depth, height, width, channels, groups, epsilon
    };
    uint64_t rows = (uint64_t)batch * depth * groups;
    if (rows > UINT32_MAX) return 0;
    return h3_gpu_dispatch_rows(
        gpu, @"h3_vae_encoder_group_norm_silu_f32", (uint32_t)rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

static int h3_gpu_require_bf16(H3GPU *gpu, const h3_gpu_tensor *tensor,
                               size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_BF16) {
        h3_gpu_set_error(gpu, @"%@ tensor is not BF16", label);
        return 0;
    }
    return 1;
}

static int h3_gpu_require_i8(H3GPU *gpu, const h3_gpu_tensor *tensor,
                             size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_I8) {
        h3_gpu_set_error(gpu, @"%@ tensor is not int8", label);
        return 0;
    }
    return 1;
}

static int h3_gpu_require_f32(H3GPU *gpu, const h3_gpu_tensor *tensor,
                              size_t elements, NSString *label) {
    if (!h3_gpu_require_elements(gpu, tensor, elements, label)) return 0;
    if (TENSOR(tensor).dtype != H3_GPU_F32) {
        h3_gpu_set_error(gpu, @"%@ tensor is not F32", label);
        return 0;
    }
    return 1;
}

static H3Linear *h3_gpu_linear_graph(H3GPU *gpu, uint32_t rows,
                                     uint32_t input_dim, uint32_t output_dim,
                                     int has_bias, MPSDataType dataType) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%d:%u", rows,
                         input_dim, output_dim, has_bias, (unsigned)dataType];
        H3Linear *cached = gpu.linearCache[key];
        if (cached) return cached;

        H3Linear *linear = [[H3Linear alloc] init];
        linear.graph = [[MPSGraph alloc] init];
        linear.inputShape = @[@1, @(rows), @(input_dim)];
        linear.weightShape = @[@1, @(output_dim), @(input_dim)];
        linear.biasShape = @[@1, @1, @(output_dim)];
        linear.outputShape = @[@1, @(rows), @(output_dim)];
        linear.input = [linear.graph placeholderWithShape:linear.inputShape
                                                 dataType:dataType name:nil];
        linear.weight = [linear.graph placeholderWithShape:linear.weightShape
                                                  dataType:dataType name:nil];
        MPSGraphTensor *transposed =
            [linear.graph transposeTensor:linear.weight dimension:1
                            withDimension:2 name:nil];
        MPSGraphTensor *output =
            [linear.graph matrixMultiplicationWithPrimaryTensor:linear.input
                                                secondaryTensor:transposed name:nil];
        if (has_bias) {
            linear.bias = [linear.graph placeholderWithShape:linear.biasShape
                                                    dataType:dataType name:nil];
            output = [linear.graph additionWithPrimaryTensor:output
                                             secondaryTensor:linear.bias name:nil];
        }
        linear.output = [linear.graph castTensor:output toType:dataType name:nil];
        gpu.linearCache[key] = linear;
        return linear;
    }
}

static int h3_gpu_linear_mps(H3GPU *gpu, h3_gpu_tensor *output,
                             const h3_gpu_tensor *input,
                             const h3_gpu_tensor *weight,
                             const h3_gpu_tensor *bias, uint32_t rows,
                             uint32_t input_dim, uint32_t output_dim,
                             MPSDataType dataType) {
    if (!h3_gpu_require_command(gpu)) return 0;
    uint64_t temporary=(uint64_t)rows*(input_dim+(uint64_t)output_dim)*4;
    if(!h3_gpu_memory_reserve(gpu,temporary,"MPSGraph linear workspace"))return 0;
    H3Linear *linear = h3_gpu_linear_graph(gpu, rows, input_dim, output_dim,
                                           bias != NULL, dataType);
    if (!linear) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *input_data = h3_gpu_graph_data(
            input, linear.inputShape, dataType, 0);
        MPSGraphTensorData *weight_data = h3_gpu_graph_data(
            weight, linear.weightShape, dataType, 1);
        MPSGraphTensorData *output_data = h3_gpu_graph_data(
            output, linear.outputShape, dataType, 0);
        NSMutableDictionary *feeds = [@{linear.input: input_data,
                                         linear.weight: weight_data} mutableCopy];
        if (bias) {
            MPSGraphTensorData *bias_data = h3_gpu_graph_data(
                bias, linear.biasShape, dataType, 1);
            feeds[linear.bias] = bias_data;
        }
        NSDictionary *results = @{linear.output: output_data};
        @try {
            [linear.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph linear failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_linear_dispatches++;
    gpu.stats = stats;
    return 1;
}

#include "src/metal/q8_host.inc"

int h3_gpu_linear_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *input,
                       const h3_gpu_tensor *weight,
                       const h3_gpu_tensor *bias, uint32_t rows,
                       uint32_t input_dim, uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    if (weight && TENSOR(weight).q8Columns)
        return h3_gpu_linear_q8(gpu,output,input,weight,bias,rows,input_dim,output_dim);
    size_t input_count = (size_t)rows * input_dim;
    size_t weight_count = (size_t)output_dim * input_dim;
    size_t output_count = (size_t)rows * output_dim;
    if (!h3_gpu_require_bf16(gpu, input, input_count, @"linear input") ||
        !h3_gpu_require_bf16(gpu, weight, weight_count, @"linear weight") ||
        !h3_gpu_require_bf16(gpu, output, output_count, @"linear output") ||
        (bias && !h3_gpu_require_bf16(gpu, bias, output_dim, @"linear bias"))) return 0;
    const char *splitRowsValue = getenv("H3_NAX_SPLIT_ROWS");
    BOOL autoSplitRows = gpu.tensorOpsEnabled && rows > 2048 && rows <= 3072 &&
        (gpu.tensorOpsMode == 2 || gpu.tensorOpsMode == 3 ||
         gpu.tensorOpsMode == 4);
    BOOL splitRows = rows > 2048 &&
        (autoSplitRows || (splitRowsValue && *splitRowsValue));
    BOOL specializedRows = rows <= 2048 || splitRows;
    BOOL naxShape = gpu.tensorOpsMode == 1 ||
        (specializedRows && gpu.tensorOpsMode == 2 &&
         input_dim == 7168 && output_dim == 5376) ||
        (specializedRows && gpu.tensorOpsMode == 3 &&
         ((input_dim == 7168 && output_dim == 5376) ||
          (input_dim == 5376 && output_dim == 21504))) ||
        (specializedRows && gpu.tensorOpsMode == 4 &&
         input_dim == 5376 && output_dim == 21504);
    if (gpu.tensorOpsEnabled && naxShape && !bias && rows >= 128 &&
        !getenv("H3_DISABLE_NAX_LINEAR") &&
        (input_dim % 32) == 0 && (output_dim % 64) == 0) {
        if (!h3_gpu_require_command(gpu)) return 0;
        uint32_t column_tiles = (output_dim + 63) / 64;
        BOOL morton4 = (rows <= 2048 || splitRows) &&
                       !(column_tiles % 4) &&
                       getenv("H3_DISABLE_NAX_LINEAR_MORTON4") == NULL;
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, morton4 ? @"h3_linear_bf16_nax_r128_morton4" :
                           @"h3_linear_bf16_nax_r128");
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
            h3_gpu_set_error(gpu, @"device cannot dispatch M5 BF16 TensorOps");
            return 0;
        }
        uint32_t splitAt = splitRows ? 2048 : rows;
        if (splitRows) {
            unsigned long requested = splitRowsValue ?
                strtoul(splitRowsValue, NULL, 10) : 0;
            if (requested >= 128 && requested <= 2048)
                splitAt = (uint32_t)requested;
        }
        uint32_t dispatches = 0;
        for (uint32_t rowOffset = 0; rowOffset < rows;
             rowOffset += splitAt) @autoreleasepool {
            uint32_t chunkRows = MIN(splitAt, rows - rowOffset);
            linear_args args = {chunkRows, input_dim, output_dim, 0};
            uint32_t row_tiles = (chunkRows + 127) / 128;
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(input).buffer
                         offset:(NSUInteger)rowOffset * input_dim *
                                sizeof(uint16_t) atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer
                         offset:(NSUInteger)rowOffset * output_dim *
                                sizeof(uint16_t) atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
            if (morton4) {
                [encoder dispatchThreadgroups:
                    MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            } else {
                [encoder dispatchThreadgroups:
                    MTLSizeMake(row_tiles, column_tiles, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            }
            [encoder endEncoding];
            dispatches++;
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches += dispatches;
        gpu.stats = stats;
        return 1;
    }
    if (rows >= 32 && input_dim >= 256 && output_dim >= 256 &&
        h3_gpu_linear_mps(gpu, output, input, weight, bias, rows,
                          input_dim, output_dim,
                          MPSDataTypeBFloat16)) return 1;
    linear_args args = {rows, input_dim, output_dim, bias ? 1u : 0u};
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    if (!h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu,
                                                           @"h3_linear_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch the 16x16 BF16 linear tile");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((output_dim + 15) / 16,
                                                  (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static H3MLP *h3_gpu_mlp_graph(H3GPU *gpu, uint32_t rows,
                               uint32_t input_dim, uint32_t hidden_dim,
                               uint32_t output_dim) {
    @autoreleasepool {
        NSString *key = [NSString stringWithFormat:@"%u:%u:%u:%u", rows,
                         input_dim, hidden_dim, output_dim];
        H3MLP *cached = gpu.mlpCache[key];
        if (cached) return cached;

        H3MLP *mlp = [[H3MLP alloc] init];
        mlp.graph = [[MPSGraph alloc] init];
        mlp.inputShape = @[@1, @(rows), @(input_dim)];
        mlp.fc1Shape = @[@1, @(hidden_dim * 2), @(input_dim)];
        mlp.fc2Shape = @[@1, @(output_dim), @(hidden_dim)];
        mlp.outputShape = @[@1, @(rows), @(output_dim)];
        mlp.input = [mlp.graph placeholderWithShape:mlp.inputShape
                                           dataType:MPSDataTypeBFloat16 name:nil];
        mlp.fc1Weight = [mlp.graph placeholderWithShape:mlp.fc1Shape
                                               dataType:MPSDataTypeBFloat16 name:nil];
        mlp.fc2Weight = [mlp.graph placeholderWithShape:mlp.fc2Shape
                                               dataType:MPSDataTypeBFloat16 name:nil];
        MPSGraphTensor *fc1Transposed =
            [mlp.graph transposeTensor:mlp.fc1Weight dimension:1
                         withDimension:2 name:nil];
        MPSGraphTensor *fused =
            [mlp.graph matrixMultiplicationWithPrimaryTensor:mlp.input
                                             secondaryTensor:fc1Transposed name:nil];
        NSArray<MPSGraphTensor *> *halves =
            [mlp.graph splitTensor:fused numSplits:2 axis:2 name:nil];
        MPSGraphTensor *sigmoid = [mlp.graph sigmoidWithTensor:halves[0]
                                                              name:nil];
        MPSGraphTensor *silu =
            [mlp.graph multiplicationWithPrimaryTensor:halves[0]
                                       secondaryTensor:sigmoid name:nil];
        MPSGraphTensor *activated =
            [mlp.graph multiplicationWithPrimaryTensor:silu
                                       secondaryTensor:halves[1] name:nil];
        MPSGraphTensor *fc2Transposed =
            [mlp.graph transposeTensor:mlp.fc2Weight dimension:1
                         withDimension:2 name:nil];
        MPSGraphTensor *result =
            [mlp.graph matrixMultiplicationWithPrimaryTensor:activated
                                             secondaryTensor:fc2Transposed name:nil];
        mlp.output = [mlp.graph castTensor:result
                                    toType:MPSDataTypeBFloat16 name:nil];
        gpu.mlpCache[key] = mlp;
        return mlp;
    }
}

int h3_gpu_mlp_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *input,
                    const h3_gpu_tensor *fc1_weight,
                    const h3_gpu_tensor *fc2_weight, uint32_t rows,
                    uint32_t input_dim, uint32_t hidden_dim,
                    uint32_t output_dim) {
    @autoreleasepool {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"MLP input") ||
        !h3_gpu_require_bf16(gpu, fc1_weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             @"MLP fc1 weight") ||
        !h3_gpu_require_bf16(gpu, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             @"MLP fc2 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"MLP output") ||
        !h3_gpu_require_command(gpu)) return 0;
    if(!h3_gpu_memory_reserve(gpu,(uint64_t)rows*(input_dim+(uint64_t)hidden_dim*3+output_dim)*4,
                             "MPSGraph MLP workspace"))return 0;
    H3MLP *mlp = h3_gpu_mlp_graph(gpu, rows, input_dim, hidden_dim,
                                  output_dim);
    if (!mlp) return 0;
    @autoreleasepool {
        MPSCommandBuffer *command = h3_gpu_mps_command(gpu);
        MPSGraphTensorData *inputData = h3_gpu_graph_data(
            input, mlp.inputShape, MPSDataTypeBFloat16, 0);
        MPSGraphTensorData *fc1Data = h3_gpu_graph_data(
            fc1_weight, mlp.fc1Shape, MPSDataTypeBFloat16, 1);
        MPSGraphTensorData *fc2Data = h3_gpu_graph_data(
            fc2_weight, mlp.fc2Shape, MPSDataTypeBFloat16, 1);
        MPSGraphTensorData *outputData = h3_gpu_graph_data(
            output, mlp.outputShape, MPSDataTypeBFloat16, 0);
        NSDictionary *feeds = @{mlp.input: inputData,
                                mlp.fc1Weight: fc1Data,
                                mlp.fc2Weight: fc2Data};
        NSDictionary *results = @{mlp.output: outputData};
        @try {
            [mlp.graph encodeToCommandBuffer:command feeds:feeds
                targetOperations:nil resultsDictionary:results
                executionDescriptor:nil];
        } @catch (NSException *exception) {
            h3_gpu_set_error(gpu, @"MPSGraph MLP failed: %@", exception.reason);
            return 0;
        }
        gpu.command = command.rootCommandBuffer;
    }
    h3_gpu_stats stats = gpu.stats;
    stats.mps_linear_dispatches += 2;
    gpu.stats = stats;
    return 1;
    }
}

static int h3_gpu_fc1_swiglu_nax_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                                      const h3_gpu_tensor *input,
                                      const h3_gpu_tensor *weight,
                                      uint32_t rows, uint32_t input_dim,
                                      uint32_t hidden_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled ||
        (gpu.tensorOpsMode != 5 && !h3_gpu_has_int8_mlp(opaque)) ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"NAX FC1 input") ||
        !h3_gpu_require_bf16(gpu, weight,
                             (size_t)hidden_dim * 2 * input_dim,
                             @"NAX FC1 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * hidden_dim,
                             @"NAX FC1 output") ||
        !rows || (input_dim % 32) || (hidden_dim % 64) ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL morton = getenv("H3_DISABLE_NAX_MORTON") == NULL &&
                  getenv("H3_DISABLE_NAX_MORTON_FC1") == NULL;
    uint32_t row_tiles = (rows + 127) / 128;
    uint32_t column_tiles = (hidden_dim + 63) / 64;
    BOOL morton4 = morton && !(column_tiles % 4) &&
                   getenv("H3_DISABLE_NAX_MORTON4") == NULL &&
                   getenv("H3_DISABLE_NAX_MORTON4_FC1") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, morton4 ? @"h3_fc1_swiglu_bf16_nax_r128_morton4" :
        morton ? @"h3_fc1_swiglu_bf16_nax_r128_morton" :
                 @"h3_fc1_swiglu_bf16_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
        h3_gpu_set_error(gpu, @"device cannot dispatch fused M5 FC1/SwiGLU");
        return 0;
    }
    linear_args args = {rows, input_dim, hidden_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder setThreadgroupMemoryLength:128 * 64 * 2 * sizeof(uint16_t)
                                   atIndex:0];
        if (morton4) {
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (morton) {
            uint32_t row_codes = 1, column_codes = 1;
            while (row_codes < row_tiles) row_codes <<= 1;
            while (column_codes < column_tiles) column_codes <<= 1;
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_codes * column_codes, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:
                MTLSizeMake(row_tiles, column_tiles, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_mlp_nax_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *activated,
                        const h3_gpu_tensor *input,
                        const h3_gpu_tensor *fc1_weight,
                        const h3_gpu_tensor *fc2_weight, uint32_t rows,
                        uint32_t input_dim, uint32_t hidden_dim,
                        uint32_t output_dim) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled || gpu.tensorOpsMode != 5 || rows < 128 ||
        (hidden_dim % 32) || (output_dim % 64) ||
        !h3_gpu_require_bf16(gpu, fc2_weight,
                             (size_t)output_dim * hidden_dim,
                             @"NAX FC2 weight") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"NAX FC2 output") ||
        !h3_gpu_fc1_swiglu_nax_bf16(opaque, activated, input, fc1_weight,
                                     rows, input_dim, hidden_dim)) return 0;
    BOOL morton = getenv("H3_DISABLE_NAX_MORTON") == NULL &&
                  getenv("H3_DISABLE_NAX_MORTON_FC2") == NULL;
    uint32_t row_tiles = (rows + 127) / 128;
    uint32_t column_tiles = (output_dim + 63) / 64;
    BOOL morton4 = morton && !(column_tiles % 4) &&
                   getenv("H3_DISABLE_NAX_MORTON4") == NULL &&
                   getenv("H3_DISABLE_NAX_MORTON4_FC2") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, morton4 ? @"h3_linear_bf16_nax_r128_morton4" :
        morton ? @"h3_linear_bf16_nax_r128_morton" :
                 @"h3_linear_bf16_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 128) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 BF16 TensorOps FC2");
        return 0;
    }
    linear_args args = {rows, hidden_dim, output_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(activated).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc2_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        if (morton4) {
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_tiles * column_tiles, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else if (morton) {
            uint32_t row_codes = 1, column_codes = 1;
            while (row_codes < row_tiles) row_codes <<= 1;
            while (column_codes < column_tiles) column_codes <<= 1;
            [encoder dispatchThreadgroups:
                MTLSizeMake((NSUInteger)row_codes * column_codes, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        } else {
            [encoder dispatchThreadgroups:
                MTLSizeMake(row_tiles, column_tiles, 1)
                 threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
        }
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_rows(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t dispatch_rows, uint32_t columns,
                        float clip, NSString *label) {
    H3GPU *gpu = GPU(opaque);
    if (!gpu.tensorOpsEnabled || !rows || dispatch_rows < rows || !columns ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns, label) ||
        !h3_gpu_require_i8(gpu, output,
                           (size_t)dispatch_rows * columns,
                           @"int8 quantized output") ||
        !h3_gpu_require_f32(gpu, scales, dispatch_rows,
                            @"int8 quantization scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL vector_quantizer =
        [label isEqualToString:@"int8 MLP FC2 input"] ||
        getenv("H3_INT8_VECTOR_QUANT");
    NSString *kernel = vector_quantizer ?
        @"h3_quantize_bf16_int8_rows" :
        @"h3_quantize_bf16_int8_rows_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, kernel);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 int8 quantizer");
        return 0;
    }
    int8_quant_args args = {rows, columns, clip};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(dispatch_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_head_major_rows(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t padded_rows, uint32_t heads,
                        uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t columns = (size_t)heads * head_dim;
    if (!gpu.tensorOpsEnabled || !rows || padded_rows < rows ||
        heads != 56 || head_dim != 128 ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns,
                             @"head-major int8 linear input") ||
        !h3_gpu_require_i8(gpu, output, (size_t)padded_rows * columns,
                           @"head-major int8 quantized output") ||
        !h3_gpu_require_f32(gpu, scales, padded_rows,
                            @"head-major int8 quantization scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_quantize_bf16_int8_head_major_to_rows_cached");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(
            gpu, @"device cannot dispatch M5 head-major int8 quantizer");
        return 0;
    }
    int8_head_major_quant_args args = {
        rows, padded_rows, heads, head_dim, 1.0f
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_quantize_bf16_int8_groups(
                        h3_gpu *opaque, h3_gpu_tensor *output,
                        h3_gpu_tensor *scales,
                        const h3_gpu_tensor *input, uint32_t rows,
                        uint32_t dispatch_rows, uint32_t columns,
                        uint32_t group_size,
                        int use_slower_grouped_quantizer) {
    H3GPU *gpu = GPU(opaque);
    if (!group_size || (group_size % 4)) return 0;
    uint32_t groups = (columns + group_size - 1) / group_size;
    if (!gpu.tensorOpsEnabled || !rows || dispatch_rows < rows || !columns ||
        (columns % group_size) ||
        !h3_gpu_require_bf16(gpu, input, (size_t)rows * columns,
                             @"grouped int8 input") ||
        !h3_gpu_require_i8(gpu, output,
                           (size_t)dispatch_rows * columns,
                           @"grouped int8 output") ||
        !h3_gpu_require_f32(gpu, scales,
                            (size_t)dispatch_rows * groups,
                            @"grouped int8 scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    BOOL scalar128 = !use_slower_grouped_quantizer && rows <= 2048;
    const char *quantizer_override = getenv("H3_INT8_GROUP_QUANT_128");
    if (quantizer_override)
        scalar128 = *quantizer_override && strcmp(quantizer_override, "0");
    BOOL cached128 = scalar128 && group_size == 1024;
    const char *cache_override = getenv("H3_INT8_GROUP_QUANT_128_CACHE");
    if (cache_override)
        cached128 = cached128 && *cache_override &&
            strcmp(cache_override, "0");
    NSString *kernel = getenv("H3_INT8_VECTOR_QUANT") ?
        @"h3_quantize_bf16_int8_groups" : cached128 ?
        @"h3_quantize_bf16_int8_groups_scalar128_cached" : scalar128 ?
        @"h3_quantize_bf16_int8_groups_scalar128" :
        @"h3_quantize_bf16_int8_groups_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, kernel);
    NSUInteger threads = scalar128 ? 128u : 256u;
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch grouped M5 int8 quantizer");
        return 0;
    }
    int8_group_quant_args args = {
        rows, columns, group_size, groups
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(scales).buffer offset:0 atIndex:2];
        [encoder setBytes:&args length:sizeof(args) atIndex:3];
        [encoder dispatchThreadgroups:MTLSizeMake(dispatch_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_quantize_weight_int8(h3_gpu *opaque, h3_gpu_tensor *output,
                                h3_gpu_tensor *scales,
                                const h3_gpu_tensor *input, uint32_t rows,
                                uint32_t columns) {
    return h3_gpu_quantize_bf16_int8_rows(
        opaque, output, scales, input, rows, rows, columns,
        1.0f, @"BF16 weight to quantize");
}

static int h3_gpu_linear_int8_bf16_layout(
                            h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales,
                            BOOL headMajorInput, uint32_t heads,
                            uint32_t headDim) {
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    if (!gpu.tensorOpsEnabled || rows < 128 || input_dim % 128 ||
        output_dim % 128 ||
        !h3_gpu_require_i8(gpu, weight, (size_t)output_dim * input_dim,
                           @"int8 linear weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, output_dim,
                            @"int8 linear weight scales") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * output_dim,
                             @"int8 linear output") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (headMajorInput) {
        if (use_slower_uncached_int8_scales || heads * headDim != input_dim ||
            !h3_gpu_quantize_bf16_int8_head_major_rows(
                opaque, quantized_input, input_scales, input, rows,
                padded_rows, heads, headDim)) return 0;
    } else if (!h3_gpu_quantize_bf16_int8_rows(
                   opaque, quantized_input, input_scales, input, rows,
                   padded_rows, input_dim, 1.0f,
                   @"int8 linear input")) return 0;
    BOOL local_scales = !use_slower_uncached_int8_scales &&
        getenv("H3_DISABLE_INT8_LOCAL_SCALES") == NULL;
    BOOL known_linear = local_scales && rows <= 2048 && input_dim == 7168 &&
        output_dim == 5376 &&
        getenv("H3_DISABLE_INT8_LINEAR_KNOWN") == NULL;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, known_linear ? @"h3_linear_int8_local_scales_nax_r128_k7168" :
             local_scales ? @"h3_linear_int8_local_scales_nax_r128" :
                            @"h3_linear_int8_nax_r128");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"int8 M5 linear projection is unavailable");
        return 0;
    }
    linear_args args = {rows, input_dim, output_dim, 0};
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        NSUInteger groups = (NSUInteger)(padded_rows / 128u) *
                            (output_dim / 128u);
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_linear_int8_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t input_dim,
                            uint32_t output_dim,
                            int use_slower_uncached_int8_scales) {
    return h3_gpu_linear_int8_bf16_layout(
        opaque, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, input_dim, output_dim,
        use_slower_uncached_int8_scales, NO, 0, 0);
}

int h3_gpu_linear_int8_head_major_bf16(
                            h3_gpu *opaque, h3_gpu_tensor *output,
                            h3_gpu_tensor *quantized_input,
                            h3_gpu_tensor *input_scales,
                            const h3_gpu_tensor *input,
                            const h3_gpu_tensor *weight,
                            const h3_gpu_tensor *weight_scales,
                            uint32_t rows, uint32_t heads,
                            uint32_t head_dim, uint32_t output_dim) {
    return h3_gpu_linear_int8_bf16_layout(
        opaque, output, quantized_input, input_scales, input, weight,
        weight_scales, rows, heads * head_dim, output_dim, 0, YES, heads,
        head_dim);
}

int h3_gpu_mlp_int8_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
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
    H3GPU *gpu = GPU(opaque);
    uint32_t padded_rows = (rows + 127u) & ~127u;
    uint32_t fc2_scale_groups = hidden_dim / 1024u;
    size_t activation_capacity = (size_t)padded_rows *
        MAX(input_dim, hidden_dim);
    const char *stage = getenv("H3_INT8_MLP_STAGE");
    BOOL int8_fc1 = !stage || (strcmp(stage, "fc2") &&
                               strcmp(stage, "bf16"));
    BOOL int8_fc2 = !stage || (strcmp(stage, "fc1") &&
                               strcmp(stage, "bf16"));
    if (!gpu.tensorOpsEnabled || rows < 128 || rows > UINT32_MAX - 127u ||
        (input_dim % 128) || (hidden_dim % 128) || (output_dim % 128) ||
        !h3_gpu_require_i8(gpu, quantized_activation, activation_capacity,
                           @"int8 MLP activation") ||
        !h3_gpu_require_f32(gpu, activation_scales,
                            (size_t)padded_rows * MAX(fc2_scale_groups, 1u),
                            @"int8 MLP activation scales") ||
        !h3_gpu_require_i8(gpu, fc1_weight,
                           (size_t)hidden_dim * 2 * input_dim,
                           @"int8 MLP FC1 weight") ||
        !h3_gpu_require_f32(gpu, fc1_scales, (size_t)hidden_dim * 2,
                            @"int8 MLP FC1 scales") ||
        !h3_gpu_require_i8(gpu, fc2_weight,
                           (size_t)output_dim * hidden_dim,
                           @"int8 MLP FC2 weight") ||
        !h3_gpu_require_f32(gpu, fc2_scales, output_dim,
                            @"int8 MLP FC2 scales") ||
        (!int8_fc1 && !h3_gpu_require_bf16(
            gpu, fc1_bf16, (size_t)hidden_dim * 2 * input_dim,
            @"diagnostic BF16 MLP FC1 weight")) ||
        (!int8_fc2 && !h3_gpu_require_bf16(
            gpu, fc2_bf16, (size_t)output_dim * hidden_dim,
            @"diagnostic BF16 MLP FC2 weight")) ||
        (!input_is_quantized &&
         !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                              @"int8 MLP input")) ||
        !h3_gpu_require_bf16(gpu, activated,
                             (size_t)rows * hidden_dim,
                             @"int8 MLP activated") ||
        !h3_gpu_require_bf16(gpu, output,
                             (size_t)rows * output_dim,
                             @"int8 MLP output") ||
        !h3_gpu_require_command(gpu)) return 0;

    id<MTLComputePipelineState> fc1 = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128");
    id<MTLComputePipelineState> fc1_known = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128_k5376");
    id<MTLComputePipelineState> fc1_full = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_nax_r128_full_k5376");
    id<MTLComputePipelineState> fc1_local = h3_gpu_pipeline(
        gpu, @"h3_fc1_swiglu_int8_local_nax_r128");
    id<MTLComputePipelineState> fc2 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128");
    id<MTLComputePipelineState> fc2_full = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128_full_k14336");
    id<MTLComputePipelineState> fc2_full_n256 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_nax_r128x256_full_k14336");
    id<MTLComputePipelineState> fc2_grouped = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_nax_r128x64");
    id<MTLComputePipelineState> fc2_grouped_local = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_local_nax_r128x64");
    id<MTLComputePipelineState> fc2_grouped_local128 = h3_gpu_pipeline(
        gpu, @"h3_linear_int8_grouped_local_nax_r128x128");
    if (!fc1 || !fc1_known || !fc1_full || !fc1_local || !fc2 ||
        !fc2_full || !fc2_full_n256 ||
        fc1.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_known.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_full.maxTotalThreadsPerThreadgroup < 256 ||
        fc1_local.maxTotalThreadsPerThreadgroup < 256 ||
        fc2.maxTotalThreadsPerThreadgroup < 256 ||
        fc2_full.maxTotalThreadsPerThreadgroup < 256 ||
        fc2_full_n256.maxTotalThreadsPerThreadgroup < 512 || !fc2_grouped ||
        fc2_grouped.maxTotalThreadsPerThreadgroup < 128 ||
        !fc2_grouped_local ||
        fc2_grouped_local.maxTotalThreadsPerThreadgroup < 256 ||
        !fc2_grouped_local128 ||
        fc2_grouped_local128.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu, @"device cannot dispatch M5 int8 MLP");
        return 0;
    }
    float activation_clip = 1.0f;
    const char *clip_value = getenv("H3_INT8_ACTIVATION_CLIP");
    if (clip_value) {
        float parsed = strtof(clip_value, NULL);
        if (parsed >= 0.1f && parsed <= 1.0f) activation_clip = parsed;
    }
    BOOL grouped_fc2 = int8_fc2 && !use_int8_row_fc2;
    BOOL row_fc2_n256 = use_int8_row_fc2 && rows <= 2048;
    BOOL grouped_fc2_local = grouped_fc2 &&
        getenv("H3_INT8_GROUP_FC2_THREADGROUP") == NULL;
    BOOL grouped_fc2_local128 = grouped_fc2 &&
        getenv("H3_INT8_GROUP_FC2_LOCAL128") != NULL;
    BOOL int8_fc1_local = int8_fc1 &&
        getenv("H3_INT8_FC1_LOCAL") != NULL;
    BOOL int8_fc1_known = int8_fc1 && input_dim == 5376 &&
        !use_slower_dynamic_fc1_k;
    const char *known_override = getenv("H3_INT8_FC1_KNOWN");
    if (known_override)
        int8_fc1_known = int8_fc1 && input_dim == 5376 &&
            strcmp(known_override, "0") != 0;
    BOOL int8_fc1_full = int8_fc1_known &&
        getenv("H3_DISABLE_FC1_FULL_K") == NULL;
    uint32_t row_tiles = padded_rows / 128;
    if (input_is_quantized && !int8_fc1) {
        h3_gpu_set_error(gpu,
            @"prequantized MLP input requires the int8 FC1 path");
        return 0;
    }
    if (int8_fc1 && !input_is_quantized &&
        !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_activation, activation_scales, input, rows,
            padded_rows, input_dim, activation_clip,
            @"int8 MLP input")) return 0;
    if (int8_fc1) @autoreleasepool {
        linear_args fc1_args = {rows, input_dim, hidden_dim, 0};
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:int8_fc1_local ? fc1_local :
            int8_fc1_full ? fc1_full :
            int8_fc1_known ? fc1_known : fc1];
        [encoder setBuffer:TENSOR(quantized_activation).buffer
                         offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc1_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(activation_scales).buffer
                         offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(fc1_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(activated).buffer offset:0 atIndex:4];
        [encoder setBytes:&fc1_args length:sizeof(fc1_args) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake((NSUInteger)row_tiles * (hidden_dim / 128), 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    } else if (!h3_gpu_fc1_swiglu_nax_bf16(
                   opaque, activated, input, fc1_bf16,
                   rows, input_dim, hidden_dim)) {
        return 0;
    }
    if (int8_fc1) {
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
    }

    if (int8_fc2 && grouped_fc2 &&
        !h3_gpu_quantize_bf16_int8_groups(
            opaque, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, 1024,
            use_slower_grouped_quantizer)) return 0;
    if (int8_fc2 && !grouped_fc2 &&
        !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_activation, activation_scales, activated, rows,
            padded_rows, hidden_dim, activation_clip,
            @"int8 MLP FC2 input")) return 0;
    if (int8_fc2) @autoreleasepool {
        linear_args fc2_args = {rows, hidden_dim, output_dim, 0};
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:grouped_fc2_local128 ?
            fc2_grouped_local128 : grouped_fc2_local ?
            fc2_grouped_local : grouped_fc2 ? fc2_grouped :
            row_fc2_n256 ? fc2_full_n256 :
            hidden_dim == 14336u && output_dim == 5376u ? fc2_full : fc2];
        [encoder setBuffer:TENSOR(quantized_activation).buffer
                         offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(fc2_weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(activation_scales).buffer
                         offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(fc2_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
        [encoder setBytes:&fc2_args length:sizeof(fc2_args) atIndex:5];
        [encoder dispatchThreadgroups:
            MTLSizeMake((NSUInteger)row_tiles *
                        (output_dim /
                         (row_fc2_n256 ? 256u :
                          grouped_fc2 && !grouped_fc2_local128 ?
                          64u : 128u)), 1, 1)
                 threadsPerThreadgroup:
                    MTLSizeMake(row_fc2_n256 ? 512u :
                        grouped_fc2_local ? 256u :
                        grouped_fc2 ? 128u : 256u, 1, 1)];
        [encoder endEncoding];
    } else if (!h3_gpu_linear_bf16(
                   opaque, output, activated, fc2_bf16, NULL,
                   rows, hidden_dim, output_dim)) {
        return 0;
    }
    if (int8_fc2) {
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
    }
    return 1;
}

int h3_gpu_silu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, elements, @"SiLU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"SiLU output")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:2];
        });
}

int h3_gpu_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *input,
                         const h3_gpu_tensor *weight, uint32_t rows,
                         uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"RMSNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, @"RMSNorm weight") ||
        !h3_gpu_require_bf16(gpu, output, count, @"RMSNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_rms_norm_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_layer_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           const h3_gpu_tensor *weight,
                           const h3_gpu_tensor *bias, uint32_t rows,
                           uint32_t width, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, input, count, @"LayerNorm input") ||
        !h3_gpu_require_bf16(gpu, weight, width, @"LayerNorm weight") ||
        !h3_gpu_require_bf16(gpu, bias, width, @"LayerNorm bias") ||
        !h3_gpu_require_bf16(gpu, output, count, @"LayerNorm output")) return 0;
    norm_args args = {rows, width, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_layer_norm_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(bias).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

int h3_gpu_gelu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *input, uint32_t elements,
                     int approximate) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, input, elements, @"GELU input") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"GELU output")) return 0;
    gelu_bf16_args args = {elements, approximate ? 1u : 0u};
    return h3_gpu_dispatch_1d(gpu, @"h3_gelu_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_vision_qkv_rope_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *query,
                     h3_gpu_tensor *key, h3_gpu_tensor *value,
                     const h3_gpu_tensor *qkv,
                     const h3_gpu_tensor *rope_cos,
                     const h3_gpu_tensor *rope_sin, uint32_t sequence,
                     uint32_t heads, uint32_t head_dim,
                     uint32_t rope_half) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, @"vision QKV") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                              @"vision RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                              @"vision RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, @"vision query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"vision key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"vision value") ||
        rope_half * 2 != head_dim) return 0;
    qkv_args args = {sequence, heads, head_dim, rope_half, 0, 0.0f};
    return h3_gpu_dispatch_3d(gpu, @"h3_vision_qkv_rope_bf16",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_adaln_bf16_offset(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (input_offset > SIZE_MAX - count ||
        input_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"AdaLN input offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, input, input_offset + count,
                             @"AdaLN input") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width, @"AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, @"AdaLN modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, @"AdaLN output") ||
        shift_slot >= slots || scale_slot >= slots) return 0;
    adaln_args args = {rows, width, slots, shift_slot, scale_slot, epsilon};
    return h3_gpu_dispatch_rows(gpu, @"h3_adaln_bf16", rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer
                         offset:input_offset * sizeof(uint16_t) atIndex:0];
            [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_adaln_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                      const h3_gpu_tensor *input,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map, uint32_t rows,
                      uint32_t width, uint32_t slots, uint32_t shift_slot,
                      uint32_t scale_slot, float epsilon) {
    return h3_gpu_adaln_bf16_offset(
        opaque, output, input, 0, norm_weight, modulation, row_map, rows,
        width, slots, shift_slot, scale_slot, epsilon);
}

int h3_gpu_adaln_linear_bf16(
                      h3_gpu *opaque, h3_gpu_tensor *output,
                      h3_gpu_tensor *inverse,
                      const h3_gpu_tensor *input, size_t input_offset,
                      const h3_gpu_tensor *norm_weight,
                      const h3_gpu_tensor *modulation,
                      const h3_gpu_tensor *row_map,
                      const h3_gpu_tensor *weight,
                      const h3_gpu_tensor *bias, uint32_t rows,
                      uint32_t width, uint32_t output_dim, uint32_t slots,
                      uint32_t shift_slot, uint32_t scale_slot,
                      float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t input_count = (size_t)rows * width;
    size_t weight_count = (size_t)output_dim * width;
    size_t output_count = (size_t)rows * output_dim;
    if (input_offset > SIZE_MAX - input_count ||
        input_offset > SIZE_MAX / sizeof(uint16_t)) {
        h3_gpu_set_error(gpu, @"fused final head input offset is out of range");
        return 0;
    }
    if (!h3_gpu_require_bf16(gpu, input, input_offset + input_count,
                             @"fused final head input") ||
        !h3_gpu_require_elements(gpu, inverse, rows,
                                 @"fused final head inverse RMS") ||
        TENSOR(inverse).dtype != H3_GPU_F32 ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused final head norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused final head modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused final head row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, weight, weight_count,
                             @"fused final head weight") ||
        !h3_gpu_require_bf16(gpu, output, output_count,
                             @"fused final head output") ||
        (bias && !h3_gpu_require_bf16(gpu, bias, output_dim,
                                      @"fused final head bias")) ||
        shift_slot >= slots || scale_slot >= slots) return 0;
    norm_args rms_args = {rows, width, epsilon};
    if (!h3_gpu_dispatch_rows(
            gpu, @"h3_rms_inverse_bf16", rows,
            ^(id<MTLComputeCommandEncoder> encoder) {
                [encoder setBuffer:TENSOR(input).buffer
                             offset:input_offset * sizeof(uint16_t) atIndex:0];
                [encoder setBuffer:TENSOR(inverse).buffer offset:0 atIndex:1];
                [encoder setBytes:&rms_args length:sizeof(rms_args) atIndex:2];
            })) return 0;
    typedef struct {
        uint32_t rows, width, output_dim, slots, shift_slot, scale_slot;
        uint32_t has_bias;
    } adaln_linear_args;
    adaln_linear_args args = {
        rows, width, output_dim, slots, shift_slot, scale_slot,
        bias ? 1u : 0u
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_adaln_linear_bf16");
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
                         @"device cannot dispatch fused final BF16 head");
        return 0;
    }
    const h3_gpu_tensor *bias_buffer = bias ? bias : input;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer
                     offset:input_offset * sizeof(uint16_t) atIndex:0];
        [encoder setBuffer:TENSOR(inverse).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(bias_buffer).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:7];
        [encoder setBytes:&args length:sizeof(args) atIndex:8];
        [encoder dispatchThreadgroups:
            MTLSizeMake((output_dim + 15) / 16, (rows + 15) / 16, 1)
                 threadsPerThreadgroup:MTLSizeMake(16, 16, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gate_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)rows * width;
    if (!h3_gpu_require_bf16(gpu, residual, count, @"gate residual") ||
        !h3_gpu_require_bf16(gpu, branch, count, @"gate branch") ||
        !h3_gpu_require_bf16(gpu, modulation, 1, @"gate modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows, @"gate row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, count, @"gate output") ||
        gate_slot >= slots) return 0;
    gate_args args = {rows, width, slots, gate_slot};
    return h3_gpu_dispatch_2d(gpu, @"h3_gate_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_gate_adaln_bf16(
                     h3_gpu *opaque, h3_gpu_tensor *gated_residual,
                     h3_gpu_tensor *output,
                     const h3_gpu_tensor *residual,
                     const h3_gpu_tensor *branch,
                     const h3_gpu_tensor *norm_weight,
                     const h3_gpu_tensor *gate_modulation,
                     const h3_gpu_tensor *norm_modulation,
                     const h3_gpu_tensor *row_map, uint32_t rows,
                     uint32_t width, uint32_t slots, uint32_t gate_slot,
                     uint32_t shift_slot, uint32_t scale_slot,
                     float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    if (!rows || !width || width > 5376 || gate_slot >= slots ||
        shift_slot >= slots || scale_slot >= slots ||
        elements > UINT32_MAX ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused gate AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, branch, elements,
                             @"fused gate AdaLN branch") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused gate AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, gate_modulation, 1,
                             @"fused gate AdaLN gate modulation") ||
        !h3_gpu_require_bf16(gpu, norm_modulation, 1,
                             @"fused gate AdaLN norm modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused gate AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, gated_residual, elements,
                             @"fused gate AdaLN gated residual") ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused gate AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot;
        float epsilon;
    } gate_adaln_args;
    gate_adaln_args args = {
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon
    };
    NSString *pipeline_name = getenv("H3_DISABLE_EXACT_SIMD_GATE_ADALN") ?
        @"h3_gate_adaln_bf16" : @"h3_gate_adaln_bf16_exact_simd";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, pipeline_name);
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused gate AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(gate_modulation).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(norm_modulation).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(gated_residual).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:7];
        [encoder setBytes:&args length:sizeof(args) atIndex:8];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_gate_adaln_quantize_int8(
                     h3_gpu *opaque, h3_gpu_tensor *gated_residual,
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
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t padded_elements = (size_t)padded_rows * width;
    if (!gpu.tensorOpsEnabled || !rows || padded_rows < rows || !width ||
        width > 5376 || gate_slot >= slots || shift_slot >= slots ||
        scale_slot >= slots || elements > UINT32_MAX ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused int8 gate AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, branch, elements,
                             @"fused int8 gate AdaLN branch") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused int8 gate AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, gate_modulation, 1,
                             @"fused int8 gate modulation") ||
        !h3_gpu_require_bf16(gpu, norm_modulation, 1,
                             @"fused int8 norm modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused int8 gate AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, gated_residual, elements,
                             @"fused int8 gated residual") ||
        !h3_gpu_require_i8(gpu, quantized_output, padded_elements,
                           @"fused int8 AdaLN output") ||
        !h3_gpu_require_f32(gpu, quantized_scales, padded_rows,
                            @"fused int8 AdaLN scales") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t rows, width, slots, gate_slot, shift_slot, scale_slot;
        float epsilon;
    } gate_adaln_args;
    gate_adaln_args args = {
        rows, width, slots, gate_slot, shift_slot, scale_slot, epsilon
    };
    NSString *pipeline_name = width == 5376 &&
        getenv("H3_DISABLE_VECTOR_GATE_ADALN") == NULL ?
        @"h3_gate_adaln_quantize_int8" :
        @"h3_gate_adaln_quantize_int8_scalar";
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, pipeline_name);
    if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < 256) {
        h3_gpu_set_error(gpu,
            @"fused M5 int8 gate AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(branch).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(gate_modulation).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(norm_modulation).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(gated_residual).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(quantized_output).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(quantized_scales).buffer offset:0 atIndex:8];
        [encoder setBytes:&args length:sizeof(args) atIndex:9];
        [encoder dispatchThreadgroups:MTLSizeMake(padded_rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

static int h3_gpu_qkv_rope_bf16_layout(h3_gpu *opaque, h3_gpu_tensor *query,
                                       h3_gpu_tensor *key,
                                       h3_gpu_tensor *value,
                                       const h3_gpu_tensor *qkv,
                                       const h3_gpu_tensor *q_norm,
                                       const h3_gpu_tensor *k_norm,
                                       const h3_gpu_tensor *rope_cos,
                                       const h3_gpu_tensor *rope_sin,
                                       uint32_t sequence, uint32_t heads,
                                       uint32_t head_dim, uint32_t rope_half,
                                       uint32_t grouped, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t inner = (size_t)heads * head_dim;
    size_t count = (size_t)sequence * inner;
    size_t rope_count = (size_t)sequence * rope_half;
    if (!h3_gpu_require_bf16(gpu, qkv, count * 3, @"QKV input") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, @"RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, @"RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, count, @"query") ||
        !h3_gpu_require_bf16(gpu, key, count, @"key") ||
        !h3_gpu_require_bf16(gpu, value, count, @"value") ||
        rope_half * 2 > head_dim) return 0;
    /* Bit 1 selects direct head-major output without an intermediate repack. */
    BOOL native=grouped&&gpu.nativeScope.backend==H3_BACKEND_METAL;
    qkv_args args = {sequence, heads, head_dim, rope_half, grouped|(native?2u:0u), epsilon};
    if(native)gpu.headMajorSDPAInputs=YES;
    if (head_dim == 128 && !(heads % 4) &&
        !getenv("H3_DISABLE_COOP_QKV")) {
        if (!h3_gpu_require_command(gpu)) return 0;
        NSString *pipeline_name = getenv("H3_DISABLE_CACHED_QKV") ?
            @"h3_qkv_rope_bf16_coop_uncached" :
            @"h3_qkv_rope_bf16_coop";
        id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
            gpu, pipeline_name);
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup < head_dim) {
            h3_gpu_set_error(gpu,
                @"cooperative QKV/RoPE needs a 128-thread threadgroup");
            return 0;
        }
        @autoreleasepool {
            id<MTLComputeCommandEncoder> encoder =
                [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:pipeline];
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
            [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, sequence, 1)
                     threadsPerThreadgroup:MTLSizeMake(head_dim, 1, 1)];
            [encoder endEncoding];
        }
        h3_gpu_stats stats = gpu.stats;
        stats.direct_dispatches++;
        gpu.stats = stats;
        return 1;
    }
    return h3_gpu_dispatch_3d(gpu, @"h3_qkv_rope_bf16",
        MTLSizeMake(head_dim, heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(qkv).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

int h3_gpu_qkv_rope_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                         h3_gpu_tensor *key, h3_gpu_tensor *value,
                         const h3_gpu_tensor *qkv,
                         const h3_gpu_tensor *q_norm,
                         const h3_gpu_tensor *k_norm,
                         const h3_gpu_tensor *rope_cos,
                         const h3_gpu_tensor *rope_sin, uint32_t sequence,
                         uint32_t heads, uint32_t head_dim,
                         uint32_t rope_half, float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        opaque, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 0, epsilon);
}

int h3_gpu_grouped_qkv_rope_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                                 h3_gpu_tensor *key, h3_gpu_tensor *value,
                                 const h3_gpu_tensor *qkv,
                                 const h3_gpu_tensor *q_norm,
                                 const h3_gpu_tensor *k_norm,
                                 const h3_gpu_tensor *rope_cos,
                                 const h3_gpu_tensor *rope_sin,
                                 uint32_t sequence, uint32_t heads,
                                 uint32_t head_dim, uint32_t rope_half,
                                 float epsilon) {
    return h3_gpu_qkv_rope_bf16_layout(
        opaque, query, key, value, qkv, q_norm, k_norm, rope_cos, rope_sin,
        sequence, heads, head_dim, rope_half, 1, epsilon);
}

int h3_gpu_grouped_qkv_linear_rope_bf16(
                                 h3_gpu *opaque,
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
                                 uint32_t rope_half, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    gpu.headMajorSDPAInputs = NO;
    uint32_t inner = heads * head_dim;
    BOOL candidate = gpu.nativeScope.backend!=H3_BACKEND_METAL && gpu.tensorOpsEnabled && rows >= 64 && rows <= 2048 &&
        input_dim == 5376 && heads == 56 && head_dim == 128 &&
        rope_half == 48 &&
        (gpu.tensorOpsMode == 1 || gpu.tensorOpsMode == 3 ||
         gpu.tensorOpsMode == 4) &&
        !getenv("H3_DISABLE_NAX_LINEAR") &&
        !getenv("H3_DISABLE_SPLIT_NAX_QKV");
    if (!candidate) {
        if(!h3_gpu_linear_bf16(opaque,qkv,input,weight,NULL,rows,input_dim,inner*3))return 0;
        if(gpu.nativeScope.backend==H3_BACKEND_METAL&&gpu.nativeScope.metal.layout_fusion)
            return h3_gpu_mixed_prepare(opaque,query,key,value,qkv,q_norm,k_norm,rope_cos,rope_sin,
                rows,heads,head_dim,rope_half,epsilon,1.0f/sqrtf((float)head_dim));
        return h3_gpu_grouped_qkv_rope_bf16(opaque,query,key,value,qkv,q_norm,k_norm,rope_cos,
            rope_sin,rows,heads,head_dim,rope_half,epsilon);
    }
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    if (!h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                             @"split QKV projection input") ||
        !h3_gpu_require_bf16(gpu, weight,
                             (size_t)inner * 3 * input_dim,
                             @"split QKV projection weight") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim,
                             @"split Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim,
                             @"split K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                             @"split RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                             @"split RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, projected, @"split query") ||
        !h3_gpu_require_bf16(gpu, key, projected, @"split key") ||
        !h3_gpu_require_bf16(gpu, value, projected, @"split value") ||
        !h3_gpu_require_command(gpu)) return 0;
    uint32_t headMajor = getenv("H3_DISABLE_HEAD_MAJOR_SDPA") == NULL;
    typedef struct {
        uint32_t rows, input_dim, heads, head_dim, rope_half, head_major;
        float epsilon;
    } qkv_project_rope_args;
    qkv_project_rope_args args = {
        rows, input_dim, heads, head_dim, rope_half, headMajor, epsilon
    };
    id<MTLComputePipelineState> projection = h3_gpu_pipeline(
        gpu, @"h3_qkv_project_split_bf16_nax_r128_morton4");
    id<MTLComputePipelineState> rope = h3_gpu_pipeline(
        gpu, @"h3_qk_rope_bf16_nax_inplace");
    const NSUInteger threads = 128;
    if (!projection || !rope ||
        projection.maxTotalThreadsPerThreadgroup < threads ||
        rope.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"split M5 QKV projection/RoPE needs 128 threads");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:projection];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:4];
        [encoder setBytes:&args length:sizeof(args) atIndex:5];
        NSUInteger groups = (NSUInteger)((rows + 127) / 128) * heads * 6;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
        encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:rope];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
        [encoder setBytes:&args length:sizeof(args) atIndex:6];
        [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, rows, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += 2;
    gpu.stats = stats;
    gpu.headMajorSDPAInputs = headMajor != 0;
    return 1;
}

int h3_gpu_grouped_qkv_linear_rope_int8(
                                 h3_gpu *opaque,
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
    H3GPU *gpu = GPU(opaque);
    gpu.headMajorSDPAInputs = NO;
    uint32_t inner = heads * head_dim;
    uint32_t padded_rows = (rows + 127u) & ~127u;
    size_t projected = (size_t)rows * inner;
    size_t rope_count = (size_t)rows * rope_half;
    if (!gpu.tensorOpsEnabled || rows < 128 || input_dim % 128 ||
        head_dim != 128 || heads % 4 || rope_half != 48 ||
        !h3_gpu_require_i8(gpu, weight,
                           (size_t)inner * 3 * input_dim,
                           @"int8 QKV projection weight") ||
        !h3_gpu_require_f32(gpu, weight_scales, inner * 3,
                            @"int8 QKV weight scales") ||
        !h3_gpu_require_i8(gpu, quantized_input,
                           (size_t)padded_rows * input_dim,
                           @"int8 QKV input") ||
        !h3_gpu_require_f32(gpu, input_scales, padded_rows,
                            @"int8 QKV input scales") ||
        (!input_is_quantized &&
         !h3_gpu_require_bf16(gpu, input, (size_t)rows * input_dim,
                              @"BF16 QKV input")) ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"int8 Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"int8 K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count,
                             @"int8 RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count,
                             @"int8 RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query, projected, @"int8 query") ||
        !h3_gpu_require_bf16(gpu, key, projected, @"int8 key") ||
        !h3_gpu_require_bf16(gpu, value, projected, @"int8 value") ||
        !h3_gpu_require_command(gpu)) return 0;
    if (!input_is_quantized && !h3_gpu_quantize_bf16_int8_rows(
            opaque, quantized_input, input_scales, input, rows, padded_rows,
            input_dim, 1.0f, @"int8 QKV input")) return 0;
    BOOL fused_rope = !use_slower_unfused_qkv_rope &&
        getenv("H3_DISABLE_FUSED_INT8_QKV_ROPE") == NULL;
    BOOL local_scales = fused_rope && rows > 2048 &&
        !use_slower_uncached_int8_scales &&
        getenv("H3_DISABLE_QKV_LOCAL_SCALES") == NULL;
    BOOL full_k = fused_rope && input_dim == 5376u &&
        getenv("H3_DISABLE_QKV_FULL_K") == NULL;
    id<MTLComputePipelineState> projection = h3_gpu_pipeline(
        gpu, local_scales && full_k ?
            @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_k5376_morton4" :
            local_scales ?
            @"h3_qkv_project_split_int8_rope_local_scales_nax_r128_morton4" :
            full_k ?
            @"h3_qkv_project_split_int8_rope_nax_r128_k5376_morton4" :
            fused_rope ?
            @"h3_qkv_project_split_int8_rope_nax_r128_morton4" :
            @"h3_qkv_project_split_int8_nax_r128_morton4");
    id<MTLComputePipelineState> rope = fused_rope ? nil : h3_gpu_pipeline(
        gpu, @"h3_qk_rope_bf16_nax_inplace");
    if (!projection || projection.maxTotalThreadsPerThreadgroup < 256 ||
        (!fused_rope &&
         (!rope || rope.maxTotalThreadsPerThreadgroup < 128))) {
        h3_gpu_set_error(gpu, @"int8 M5 QKV projection is unavailable");
        return 0;
    }
    typedef struct {
        uint32_t rows, input_dim, heads, head_dim, rope_half, head_major;
        float epsilon;
    } qkv_project_rope_args;
    /* The historical layout field now carries fused-epilogue mode bits:
     * bit 1 selects ordered BF16x4 RMS loads, bit 2 packs norm/RoPE by four. */
    uint32_t rms_mode = 1u;
    if (fused_rope && rows <= 2048 && !use_slower_scalar_qkv_rms &&
        getenv("H3_DISABLE_VECTOR_QKV_RMS") == NULL)
        rms_mode |= 2u;
    if (fused_rope && getenv("H3_DISABLE_VECTOR_QKV_ROPE") == NULL)
        rms_mode |= 4u;
    qkv_project_rope_args args = {
        rows, input_dim, heads, head_dim, rope_half, rms_mode, epsilon
    };
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:projection];
        [encoder setBuffer:TENSOR(quantized_input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(input_scales).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(weight_scales).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:6];
        if (fused_rope) {
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:7];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:8];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:9];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:10];
            [encoder setBytes:&args length:sizeof(args) atIndex:11];
        } else {
            [encoder setBytes:&args length:sizeof(args) atIndex:7];
        }
        NSUInteger groups = (NSUInteger)(padded_rows / 128u) * heads * 3u;
        [encoder dispatchThreadgroups:MTLSizeMake(groups, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(256, 1, 1)];
        [encoder endEncoding];
        if (!fused_rope) {
            encoder = [gpu.command computeCommandEncoder];
            [encoder setComputePipelineState:rope];
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
            [encoder dispatchThreadgroups:MTLSizeMake(heads / 4, rows, 1)
                     threadsPerThreadgroup:MTLSizeMake(128, 1, 1)];
            [encoder endEncoding];
        }
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches += fused_rope ? 1 : 2;
    gpu.stats = stats;
    gpu.headMajorSDPAInputs = YES;
    return 1;
}

int h3_gpu_swiglu_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                       const h3_gpu_tensor *fused, uint32_t rows,
                       uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, fused, (size_t)rows * width * 2,
                              @"SwiGLU input") ||
        !h3_gpu_require_bf16(gpu, output, (size_t)rows * width,
                              @"SwiGLU output")) return 0;
    swiglu_args args = {rows, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_swiglu_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(fused).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_embedding_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                          const h3_gpu_tensor *weight,
                          const h3_gpu_tensor *token_ids, uint32_t tokens,
                          uint32_t vocab_size, uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, weight, (size_t)vocab_size * width,
                              @"embedding weight") ||
        !h3_gpu_require_elements(gpu, token_ids, tokens, @"token IDs") ||
        TENSOR(token_ids).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, (size_t)tokens * width,
                              @"embedding output")) return 0;
    embedding_args args = {tokens, vocab_size, width};
    return h3_gpu_dispatch_2d(gpu, @"h3_embedding_bf16", width, tokens,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(token_ids).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

int h3_gpu_text_qk_rope_bf16(h3_gpu *opaque,
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
                             float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query_input, query_count, @"text query") ||
        !h3_gpu_require_bf16(gpu, key_input, key_count, @"text key") ||
        !h3_gpu_require_bf16(gpu, q_norm, head_dim, @"text Q norm") ||
        !h3_gpu_require_bf16(gpu, k_norm, head_dim, @"text K norm") ||
        !h3_gpu_require_bf16(gpu, rope_cos, rope_count, @"text RoPE cosine") ||
        !h3_gpu_require_bf16(gpu, rope_sin, rope_count, @"text RoPE sine") ||
        !h3_gpu_require_bf16(gpu, query_output, query_count, @"text query output") ||
        !h3_gpu_require_bf16(gpu, key_output, key_count, @"text key output")) return 0;
    text_rope_args args = {sequence, query_heads, kv_heads, head_dim, epsilon};
    return h3_gpu_dispatch_3d(gpu, @"h3_text_qk_rope_bf16",
        MTLSizeMake(head_dim, query_heads, sequence),
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(query_input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key_input).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(q_norm).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(k_norm).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(rope_cos).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(rope_sin).buffer offset:0 atIndex:5];
            [encoder setBuffer:TENSOR(query_output).buffer offset:0 atIndex:6];
            [encoder setBuffer:TENSOR(key_output).buffer offset:0 atIndex:7];
            [encoder setBytes:&args length:sizeof(args) atIndex:8];
        });
}

int h3_gpu_head_rms_norm_bf16(h3_gpu *opaque, h3_gpu_tensor *tensor,
                              const h3_gpu_tensor *weight,
                              uint32_t sequence, uint32_t heads,
                              uint32_t head_dim, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t count = (size_t)sequence * heads * head_dim;
    if (!h3_gpu_require_bf16(gpu, tensor, count, @"head norm tensor") ||
        !h3_gpu_require_bf16(gpu, weight, head_dim, @"head norm weight")) return 0;
    head_norm_args args = {sequence, heads, head_dim, epsilon};
    return h3_gpu_dispatch_2d(gpu, @"h3_head_rms_norm_bf16", sequence, heads,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(tensor).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
            [encoder setBytes:&args length:sizeof(args) atIndex:2];
        });
}

int h3_gpu_rope_text_bf16(h3_gpu *opaque, h3_gpu_tensor *query,
                          h3_gpu_tensor *key,
                          const h3_gpu_tensor *rope_cos_f32,
                          const h3_gpu_tensor *rope_sin_f32,
                          uint32_t sequence, uint32_t query_heads,
                          uint32_t kv_heads, uint32_t head_dim) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t key_count = (size_t)sequence * kv_heads * head_dim;
    size_t rope_count = (size_t)sequence * (head_dim / 2);
    if (head_dim % 2 || !kv_heads || query_heads % kv_heads ||
        !h3_gpu_require_bf16(gpu, query, query_count, @"RoPE query") ||
        !h3_gpu_require_bf16(gpu, key, key_count, @"RoPE key") ||
        !h3_gpu_require_elements(gpu, rope_cos_f32, rope_count, @"RoPE cosine") ||
        TENSOR(rope_cos_f32).dtype != H3_GPU_F32 ||
        !h3_gpu_require_elements(gpu, rope_sin_f32, rope_count, @"RoPE sine") ||
        TENSOR(rope_sin_f32).dtype != H3_GPU_F32) return 0;
    text_rope_inplace_args args = {sequence, query_heads, kv_heads, head_dim};
    uint32_t maximum_heads = query_heads > kv_heads ? query_heads : kv_heads;
    return h3_gpu_dispatch_2d(gpu, @"h3_rope_text_bf16", sequence, maximum_heads,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(rope_cos_f32).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(rope_sin_f32).buffer offset:0 atIndex:3];
            [encoder setBytes:&args length:sizeof(args) atIndex:4];
        });
}

const char *h3_qwen_gqa_scale_name(h3_qwen_gqa_scale_mode mode) {
    switch (mode) {
        case H3_QWEN_GQA_REFERENCE: return "reference";
        case H3_QWEN_GQA_SCALED_Q: return "scaled-q";
        case H3_QWEN_GQA_LEGACY: return "legacy";
    }
    return "invalid";
}

int h3_qwen_gqa_scale_parse(const char *value, h3_qwen_gqa_scale_mode *mode,
                            char *error, size_t error_size) {
    if (!mode) return 0;
    /* Production contract: BF16 storage, FP32 QK accumulation, then FP32 scale.
     * Validated against CPU/official Qwen and full M4 generation fixtures. */
    if (!value || !*value) *mode = H3_QWEN_GQA_REFERENCE;
    else if (!strcmp(value, "reference")) *mode = H3_QWEN_GQA_REFERENCE;
    else if (!strcmp(value, "scaled-q")) *mode = H3_QWEN_GQA_SCALED_Q;
    else if (!strcmp(value, "legacy")) *mode = H3_QWEN_GQA_LEGACY;
    else {
        if (error && error_size) snprintf(error, error_size,
            "invalid H3_QWEN_GQA_SCALE_MODE '%s'; expected reference, scaled-q, or legacy", value);
        return 0;
    }
    return 1;
}

static NSString *gqa_pipeline_name(H3GPU *gpu) {
    h3_qwen_gqa_scale_mode mode;
    char error[512];
    if (!h3_qwen_gqa_scale_parse(getenv("H3_QWEN_GQA_SCALE_MODE"), &mode,
                                 error, sizeof(error))) {
        h3_gpu_set_error(gpu, @"%s", error);
        return nil;
    }
    /* MPSGraph's opaque SDPA cannot promise any of the three arithmetic
     * contracts and used to bypass the direct kernel's sequence preflight. */
    if (getenv("H3_MPS_GQA")) {
        h3_gpu_set_error(gpu, @"H3_MPS_GQA cannot enforce Qwen scaling semantics; unset H3_MPS_GQA and use H3_QWEN_GQA_SCALE_MODE");
        return nil;
    }
    switch (mode) {
        case H3_QWEN_GQA_REFERENCE: return @"h3_gqa_causal_bf16_reference";
        case H3_QWEN_GQA_SCALED_Q: return @"h3_gqa_causal_bf16_scaled_q";
        case H3_QWEN_GQA_LEGACY: return @"h3_gqa_causal_bf16_legacy";
    }
    return nil;
}

/* Metal requires each dynamic threadgroup allocation to be a multiple of 16.
 * Keep both the preflight and dispatch tied to this one allocation formula. */
static size_t gqa_score_bytes(size_t sequence) {
    if (sequence > (SIZE_MAX - 15) / sizeof(float)) return SIZE_MAX;
    return (sequence * sizeof(float) + 15) & ~(size_t)15;
}

int h3_gpu_gqa_causal_limits(h3_gpu *opaque, h3_gqa_limits *limits) {
    if (!opaque || !limits) return 0;
    H3GPU *gpu = GPU(opaque);
    NSString *name = gqa_pipeline_name(gpu);
    if (!name) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    limits->device_bytes = gpu.device.maxThreadgroupMemoryLength;
    limits->static_bytes = pipeline.staticThreadgroupMemoryLength;
    limits->alignment_bytes = 16;
    size_t available = limits->device_bytes > limits->static_bytes ?
        limits->device_bytes - limits->static_bytes : 0;
    limits->direct_max_sequence = (available & ~(size_t)15) / sizeof(float);
    /* Qwen's 64 x 128 query rows use uint32 element addressing. */
    limits->max_sequence = UINT32_MAX / 8192;
    return 1;
}

size_t h3_gpu_gqa_causal_max_sequence(h3_gpu *gpu) {
    h3_gqa_limits limits;
    return h3_gpu_gqa_causal_limits(gpu, &limits) ? limits.max_sequence : 0;
}

int h3_gpu_gqa_causal_preflight(h3_gpu *gpu, size_t sequence,
                               char *error, size_t error_size) {
    h3_gqa_limits limits;
    if (!h3_gpu_gqa_causal_limits(gpu, &limits)) {
        if (error && error_size) snprintf(error, error_size,
            "cannot query Metal causal-GQA sequence limit: %s",
            gpu ? h3_gpu_error(gpu) : "missing GPU");
        return 0;
    }
    if (sequence && sequence <= limits.max_sequence)
        return 1;
    if (error && error_size) snprintf(error, error_size,
        "Qwen causal-GQA preflight: requested sequence %zu, active GPU maximum %zu "
        "(device threadgroup limit %zu bytes, static %zu bytes, alignment %zu). "
        "Qwen query element index range exceeded.",
        sequence, limits.max_sequence, limits.device_bytes,
        limits.static_bytes, limits.alignment_bytes);
    return 0;
}

int h3_gpu_gqa_causal_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *query,
                           const h3_gpu_tensor *key,
                           const h3_gpu_tensor *value,
                           uint32_t sequence, uint32_t query_heads,
                           uint32_t kv_heads, uint32_t head_dim,
                           float scale) {
    H3GPU *gpu = GPU(opaque);
    size_t query_count = (size_t)sequence * query_heads * head_dim;
    size_t kv_count = (size_t)sequence * kv_heads * head_dim;
    if (!sequence || !query_heads || !kv_heads || !head_dim || query_count > UINT32_MAX ||
        query_heads % kv_heads || head_dim > 128 ||
        !h3_gpu_require_bf16(gpu, query, query_count, @"GQA query") ||
        !h3_gpu_require_bf16(gpu, key, kv_count, @"GQA key") ||
        !h3_gpu_require_bf16(gpu, value, kv_count, @"GQA value") ||
        !h3_gpu_require_bf16(gpu, output, query_count, @"GQA output") ||
        !h3_gpu_require_command(gpu)) return 0;
    size_t score_bytes = gqa_score_bytes(sequence);
    NSString *name = gqa_pipeline_name(gpu);
    if (!name) return 0;
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(gpu, name);
    if (!pipeline) return 0;
    char preflight_error[512];
    if (!h3_gpu_gqa_causal_preflight(opaque, sequence, preflight_error,
                                     sizeof(preflight_error))) {
        h3_gpu_set_error(gpu, @"%s", preflight_error);
        return 0;
    }
    h3_gqa_limits limits;
    if (!h3_gpu_gqa_causal_limits(opaque,&limits)) return 0;
    BOOL tiled=sequence>limits.direct_max_sequence;
    if (tiled) {
        name=[name stringByAppendingString:@"_tiled"];
        pipeline=h3_gpu_pipeline(gpu,name);
        if (!pipeline || pipeline.maxTotalThreadsPerThreadgroup<128) return 0;
        score_bytes=512*sizeof(float);
    }
    NSUInteger maximum_threads = MIN((NSUInteger)128,
                                     pipeline.maxTotalThreadsPerThreadgroup);
    NSUInteger threads = 1;
    while (threads * 2 <= maximum_threads) threads *= 2;
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder = [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(query).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(key).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(value).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
        gqa_args args = {sequence, query_heads, kv_heads, head_dim, scale};
        [encoder setBytes:&args length:sizeof(args) atIndex:4];
        [encoder setThreadgroupMemoryLength:score_bytes atIndex:0];
        [encoder dispatchThreadgroups:MTLSizeMake(sequence, query_heads, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_add_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, left, elements, @"add left") ||
        !h3_gpu_require_bf16(gpu, right, elements, @"add right") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"add output")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_add_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_sub_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                    const h3_gpu_tensor *left, const h3_gpu_tensor *right,
                    uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, left, elements, @"subtract left") ||
        !h3_gpu_require_bf16(gpu, right, elements, @"subtract right") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"subtract output"))
        return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_sub_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(left).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(right).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

int h3_gpu_token_pool_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                           const h3_gpu_tensor *input,
                           size_t input_offset,
                           h3_gpu_tensor *original,
                           size_t original_offset,
                           h3_gpu_tensor *baseline,
                           size_t baseline_offset,
                           const h3_gpu_tensor *baseline_indices,
                           const h3_gpu_tensor *pairs, uint32_t input_rows,
                           uint32_t rows, uint32_t baseline_rows,
                           uint32_t width) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows ||
        baseline_rows > rows || !width ||
        elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !input || TENSOR(input).dtype != H3_GPU_BF16 ||
        input_offset > TENSOR(input).elements ||
        input_elements > TENSOR(input).elements - input_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        input_elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements, @"token pool output") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, rows,
                                 @"token pool baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, pairs, (size_t)rows * 2,
                                 @"token pool pairs") ||
        TENSOR(pairs).dtype != H3_GPU_U32) return 0;
    typedef struct {
        uint32_t input_offset, original_offset, baseline_offset;
        uint32_t rows, width;
    } token_pool_args;
    token_pool_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width
    };
    return h3_gpu_dispatch_2d(gpu, @"h3_token_pool_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(pairs).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(baseline_indices).buffer
                          offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_token_pool_adaln_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *residual,
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
                           uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t input_elements = (size_t)input_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!input_rows || !rows || rows > input_rows || !width || width > 5376 ||
        baseline_rows > rows || shift_slot >= slots || scale_slot >= slots ||
        elements > UINT32_MAX || input_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - input_offset ||
        original_offset > UINT32_MAX ||
        input_elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !input || TENSOR(input).dtype != H3_GPU_BF16 ||
        input_offset > TENSOR(input).elements ||
        input_elements > TENSOR(input).elements - input_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        input_elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused token pool residual") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, rows,
                                 @"fused token pool baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, pairs, (size_t)rows * 2,
                                 @"fused token pool pairs") ||
        TENSOR(pairs).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused token pool norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused token pool modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused token pool row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused token pool AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t input_offset, original_offset, baseline_offset;
        uint32_t rows, width, slots, shift_slot, scale_slot;
        float epsilon;
    } token_pool_adaln_args;
    token_pool_adaln_args args = {
        (uint32_t)input_offset, (uint32_t)original_offset,
        (uint32_t)baseline_offset, rows, width, slots, shift_slot,
        scale_slot, epsilon
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_token_pool_adaln_bf16");
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused token pool AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(pairs).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(baseline_indices).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:8];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_token_expand_delta_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *output,
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
                           float update_scale) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows ||
        baseline_rows > reduced_rows || !width ||
        exact_prefix_rows > reduced_rows || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"token expand output") ||
        !h3_gpu_require_bf16(gpu, reduced, reduced_elements,
                             @"token expand reduced") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, reduced_rows,
                                 @"token expand baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, parents, rows,
                                 @"token expand parents") ||
        TENSOR(parents).dtype != H3_GPU_U32) return 0;
    typedef struct {
        uint32_t original_offset, baseline_offset;
        uint32_t rows, width, exact_prefix_rows;
        float update_scale;
    } token_expand_args;
    token_expand_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset,
        rows, width, exact_prefix_rows, update_scale
    };
    return h3_gpu_dispatch_2d(gpu, @"h3_token_expand_delta_bf16", width, rows,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(reduced).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(baseline_indices).buffer
                          offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(parents).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_token_expand_adaln_bf16(
                           h3_gpu *opaque, h3_gpu_tensor *residual,
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
                           uint32_t scale_slot, float epsilon) {
    H3GPU *gpu = GPU(opaque);
    size_t elements = (size_t)rows * width;
    size_t reduced_elements = (size_t)reduced_rows * width;
    size_t baseline_elements = (size_t)baseline_rows * width;
    if (!rows || !reduced_rows || reduced_rows > rows || !width ||
        width > 5376 || baseline_rows > reduced_rows ||
        exact_prefix_rows > reduced_rows || shift_slot >= slots ||
        scale_slot >= slots || elements > UINT32_MAX ||
        reduced_elements > UINT32_MAX || original_offset > UINT32_MAX ||
        elements > UINT32_MAX - original_offset ||
        baseline_offset > UINT32_MAX ||
        baseline_elements > UINT32_MAX - baseline_offset ||
        !original || TENSOR(original).dtype != H3_GPU_BF16 ||
        original_offset > TENSOR(original).elements ||
        elements > TENSOR(original).elements - original_offset ||
        !h3_gpu_require_bf16(gpu, reduced, reduced_elements,
                             @"fused token AdaLN reduced input") ||
        !baseline || TENSOR(baseline).dtype != H3_GPU_BF16 ||
        baseline_offset > TENSOR(baseline).elements ||
        baseline_elements > TENSOR(baseline).elements - baseline_offset ||
        !h3_gpu_require_elements(gpu, baseline_indices, reduced_rows,
                                 @"fused token AdaLN baseline indices") ||
        TENSOR(baseline_indices).dtype != H3_GPU_U32 ||
        !h3_gpu_require_elements(gpu, parents, rows,
                                 @"fused token AdaLN parents") ||
        TENSOR(parents).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, residual, elements,
                             @"fused token AdaLN residual") ||
        !h3_gpu_require_bf16(gpu, norm_weight, width,
                             @"fused token AdaLN norm") ||
        !h3_gpu_require_bf16(gpu, modulation, 1,
                             @"fused token AdaLN modulation") ||
        !h3_gpu_require_elements(gpu, row_map, rows,
                                 @"fused token AdaLN row map") ||
        TENSOR(row_map).dtype != H3_GPU_U32 ||
        !h3_gpu_require_bf16(gpu, output, elements,
                             @"fused token AdaLN output") ||
        !h3_gpu_require_command(gpu)) return 0;
    typedef struct {
        uint32_t original_offset, baseline_offset, rows, width;
        uint32_t exact_prefix_rows, slots, shift_slot, scale_slot;
        float update_scale, epsilon;
    } token_expand_adaln_args;
    token_expand_adaln_args args = {
        (uint32_t)original_offset, (uint32_t)baseline_offset, rows, width,
        exact_prefix_rows, slots, shift_slot, scale_slot,
        update_scale, epsilon
    };
    id<MTLComputePipelineState> pipeline = h3_gpu_pipeline(
        gpu, @"h3_token_expand_adaln_bf16");
    if (!pipeline) return 0;
    const NSUInteger threads = 256;
    if (pipeline.maxTotalThreadsPerThreadgroup < threads) {
        h3_gpu_set_error(gpu,
            @"fused token AdaLN needs a 256-thread threadgroup");
        return 0;
    }
    @autoreleasepool {
        id<MTLComputeCommandEncoder> encoder =
            [gpu.command computeCommandEncoder];
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(original).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(reduced).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(baseline).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(baseline_indices).buffer offset:0 atIndex:3];
        [encoder setBuffer:TENSOR(parents).buffer offset:0 atIndex:4];
        [encoder setBuffer:TENSOR(residual).buffer offset:0 atIndex:5];
        [encoder setBuffer:TENSOR(norm_weight).buffer offset:0 atIndex:6];
        [encoder setBuffer:TENSOR(modulation).buffer offset:0 atIndex:7];
        [encoder setBuffer:TENSOR(row_map).buffer offset:0 atIndex:8];
        [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:9];
        [encoder setBytes:&args length:sizeof(args) atIndex:10];
        [encoder dispatchThreadgroups:MTLSizeMake(rows, 1, 1)
                 threadsPerThreadgroup:MTLSizeMake(threads, 1, 1)];
        [encoder endEncoding];
    }
    h3_gpu_stats stats = gpu.stats;
    stats.direct_dispatches++;
    gpu.stats = stats;
    return 1;
}

int h3_gpu_euler_bf16(h3_gpu *opaque, h3_gpu_tensor *sample,
                      size_t sample_offset, const h3_gpu_tensor *last,
                      const h3_gpu_tensor *previous, uint32_t elements,
                      float delta, float ratio) {
    H3GPU *gpu = GPU(opaque);
    if (!sample || TENSOR(sample).dtype != H3_GPU_F32 ||
        sample_offset > TENSOR(sample).elements ||
        elements > TENSOR(sample).elements - sample_offset ||
        sample_offset > UINT32_MAX || elements > UINT32_MAX - sample_offset ||
        !h3_gpu_require_bf16(gpu, last, elements, @"Euler last velocity") ||
        !h3_gpu_require_bf16(gpu, previous, elements,
                             @"Euler previous velocity")) return 0;
    euler_args args = {(uint32_t)sample_offset, elements, delta, ratio};
    return h3_gpu_dispatch_1d(gpu, @"h3_euler_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(sample).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(last).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(previous).buffer offset:0 atIndex:2];
            [encoder setBytes:&args length:sizeof(args) atIndex:3];
        });
}

static int bridge_ranges(const h3_gpu_tensor *sample, size_t offset,
    const h3_gpu_tensor *row_classes, const h3_gpu_tensor *strengths,
    uint32_t rows, uint32_t width) {
    return rows && width && rows <= UINT32_MAX / width && sample &&
        TENSOR(sample).dtype == H3_GPU_F32 && offset <= UINT32_MAX &&
        rows * width <= UINT32_MAX - offset &&
        offset <= TENSOR(sample).elements &&
        rows * width <= TENSOR(sample).elements - offset &&
        row_classes && TENSOR(row_classes).dtype == H3_GPU_U32 &&
        TENSOR(row_classes).elements >= rows && strengths &&
        TENSOR(strengths).dtype == H3_GPU_F32 &&
        TENSOR(strengths).elements > 0 &&
        TENSOR(strengths).elements <= UINT32_MAX;
}

int h3_gpu_bridge_euler_bf16(h3_gpu *opaque, h3_gpu_tensor *sample,
    size_t sample_offset, const h3_gpu_tensor *last,
    const h3_gpu_tensor *previous, const h3_gpu_tensor *row_classes,
    const h3_gpu_tensor *strengths, uint32_t rows, uint32_t row_width,
    float delta, float ratio) {
    H3GPU *gpu = GPU(opaque);
    if (!bridge_ranges(sample, sample_offset, row_classes, strengths, rows,
                       row_width) || !isfinite(delta) || !(delta > 0.0f) || !isfinite(ratio) ||
        !h3_gpu_require_bf16(gpu, last, rows * row_width, @"bridge last velocity") ||
        !h3_gpu_require_bf16(gpu, previous, rows * row_width,
                             @"bridge previous velocity")) return 0;
    bridge_euler_args args = {(uint32_t)sample_offset, rows * row_width,
        row_width, (uint32_t)TENSOR(strengths).elements, delta, ratio};
    return h3_gpu_dispatch_1d(gpu, @"h3_bridge_euler_bf16", args.elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(sample).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(last).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(previous).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(row_classes).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(strengths).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_video_preview_bf16(h3_gpu *opaque, h3_gpu_tensor *scratch,
    const h3_gpu_tensor *sample, size_t offset,
    const h3_gpu_tensor *last, const h3_gpu_tensor *previous,
    const h3_gpu_tensor *row_classes, const h3_gpu_tensor *strengths,
    uint32_t time, uint32_t height, uint32_t width, float sigma, float ratio) {
    H3GPU *gpu = GPU(opaque);
    if (!time || height < 2 || width < 2 || height % 2 || width % 2 ||
        (uint64_t)time * height > UINT32_MAX / 24u / width ||
        !isfinite(sigma) || sigma < 0 || sigma > 1 || !isfinite(ratio)) return 0;
    uint32_t elements = 24u * time * height * width;
    if (!scratch || TENSOR(scratch).bytes / sizeof(float) < elements ||
        !sample || TENSOR(sample).dtype != H3_GPU_F32 || offset > UINT32_MAX ||
        elements > UINT32_MAX - offset || offset > TENSOR(sample).elements ||
        elements > TENSOR(sample).elements - offset ||
        !h3_gpu_require_bf16(gpu, last, elements, @"preview last velocity") ||
        !h3_gpu_require_bf16(gpu, previous, elements, @"preview previous velocity") ||
        (!!row_classes != !!strengths) ||
        (row_classes && !bridge_ranges(sample, offset, row_classes, strengths,
                                       elements / 96, 96))) return 0;
    const h3_gpu_tensor *inputs[] = {sample, last, previous, row_classes, strengths};
    for (size_t i = 0; i < sizeof(inputs)/sizeof(*inputs); i++)
        if (inputs[i] && TENSOR(scratch).buffer == TENSOR(inputs[i]).buffer) return 0;
    struct {
        uint32_t offset, elements, time, height, width, masked, classes;
        float sigma, ratio;
    } args = {(uint32_t)offset, elements, time, height, width, row_classes != NULL,
        strengths ? (uint32_t)TENSOR(strengths).elements : 0, sigma, ratio};
    return h3_gpu_dispatch_1d(gpu, @"h3_video_preview_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(scratch).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(sample).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(last).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(previous).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(row_classes ? row_classes : sample).buffer offset:0 atIndex:4];
            [encoder setBuffer:TENSOR(strengths ? strengths : sample).buffer offset:0 atIndex:5];
            [encoder setBytes:&args length:sizeof(args) atIndex:6];
        });
}

int h3_gpu_video_preview_read(const h3_gpu_tensor *scratch, float *latent,
                              size_t elements) {
    if (!scratch || !latent || elements > TENSOR(scratch).bytes / sizeof(float)) return 0;
    memcpy(latent, TENSOR(scratch).buffer.contents, elements * sizeof(float));
    return 1;
}

int h3_gpu_bridge_check_exact(h3_gpu *opaque, const h3_gpu_tensor *sample,
    size_t sample_offset, const h3_gpu_tensor *initial,
    const h3_gpu_tensor *row_classes, const h3_gpu_tensor *strengths,
    h3_gpu_tensor *changed, uint32_t rows, uint32_t row_width) {
    H3GPU *gpu = GPU(opaque);
    if (!bridge_ranges(sample, sample_offset, row_classes, strengths, rows,
                       row_width) || !initial ||
        TENSOR(initial).dtype != H3_GPU_F32 ||
        TENSOR(initial).elements < rows * row_width || !changed ||
        TENSOR(changed).dtype != H3_GPU_U32 ||
        TENSOR(changed).elements != 1) return 0;
    bridge_euler_args args = {(uint32_t)sample_offset, rows * row_width,
        row_width, (uint32_t)TENSOR(strengths).elements, 0, 0};
    return h3_gpu_dispatch_1d(gpu, @"h3_bridge_check_exact", args.elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(sample).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(initial).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(row_classes).buffer offset:0 atIndex:2];
            [encoder setBuffer:TENSOR(strengths).buffer offset:0 atIndex:3];
            [encoder setBuffer:TENSOR(changed).buffer offset:0 atIndex:4];
            [encoder setBytes:&args length:sizeof(args) atIndex:5];
        });
}

int h3_gpu_silu_mul_bf16(h3_gpu *opaque, h3_gpu_tensor *output,
                         const h3_gpu_tensor *gate,
                         const h3_gpu_tensor *up, uint32_t elements) {
    H3GPU *gpu = GPU(opaque);
    if (!h3_gpu_require_bf16(gpu, gate, elements, @"SiLU gate") ||
        !h3_gpu_require_bf16(gpu, up, elements, @"SiLU up") ||
        !h3_gpu_require_bf16(gpu, output, elements, @"SiLU product")) return 0;
    return h3_gpu_dispatch_1d(gpu, @"h3_silu_mul_bf16", elements,
        ^(id<MTLComputeCommandEncoder> encoder) {
            [encoder setBuffer:TENSOR(gate).buffer offset:0 atIndex:0];
            [encoder setBuffer:TENSOR(up).buffer offset:0 atIndex:1];
            [encoder setBuffer:TENSOR(output).buffer offset:0 atIndex:2];
            [encoder setBytes:&elements length:sizeof(elements) atIndex:3];
        });
}

/* Backend-neutral allocation/fence extensions retain Metal's ordered shared
 * storage. They do not introduce a second command queue or change dispatch. */
h3_gpu_tensor *h3_gpu_tensor_alloc(h3_gpu *gpu, size_t elements,
                                   h3_gpu_dtype dtype, h3_gpu_storage storage) {
    (void)storage;
    if (dtype < H3_GPU_F32 || dtype > H3_GPU_F16) return NULL;
    size_t bytes = dtype == H3_GPU_I8 ? 1 : (dtype == H3_GPU_BF16 || dtype == H3_GPU_F16) ? 2 : 4;
    return h3_gpu_tensor_new(gpu, NULL, elements, bytes, dtype);
}
void *h3_gpu_tensor_contents(h3_gpu_tensor *tensor) {
    if (!tensor || !h3_gpu_synchronize((__bridge h3_gpu *)TENSOR(tensor).owner)) return NULL;
    return TENSOR(tensor).buffer.contents;
}
struct h3_gpu_event { int recorded; };
h3_gpu_event *h3_gpu_event_new(h3_gpu *gpu) {
    return gpu ? calloc(1,sizeof(h3_gpu_event)) : NULL;
}
int h3_gpu_event_record(h3_gpu *gpu,h3_gpu_event *event) {
    if(!gpu || !event)return 0;
    event->recorded=1;return 1;
}
int h3_gpu_event_wait(h3_gpu *gpu,h3_gpu_event *event) {
    return gpu && event && event->recorded;
}
void h3_gpu_event_free(h3_gpu_event *event) { free(event); }
int h3_gpu_plan_weights(h3_gpu *gpu,uint64_t weights,uint64_t activations) {
    if(!gpu)return -1;
    if(weights>UINT64_MAX-activations) {
        h3_gpu_set_error(GPU(gpu),@"memory safety: Metal resident plan overflows");return -1;
    }
    return h3_gpu_memory_reserve(GPU(gpu),weights+activations,
                                  "Metal resident weights and activations")?0:-1;
}
const char *h3_gpu_backend_name(const h3_gpu *gpu) { (void)gpu; return "Metal"; }
int h3_gpu_prefers_device_sampler(const h3_gpu *gpu, int continuation) {
    return !continuation && h3_gpu_is_m5(gpu);
}
int h3_gpu_should_retry_streaming(const h3_gpu *gpu) { (void)gpu; return 0; }
int h3_gpu_release_weight_cache(h3_gpu *gpu) { return gpu != NULL; }

int h3_gpu_dit_attention_configure(h3_gpu *gpu,int mode) {
    if(!gpu)return 0;
    GPU(gpu).nativeScope=h3_backend_current();
    return mode==0;
}
int h3_gpu_native_noise(h3_gpu *gpu,int step,float video_sigma,float audio_sigma) {
    H3GPU *g=GPU(gpu);if(!g)return 0;
    if(step<0||!isfinite(video_sigma)||!isfinite(audio_sigma)||video_sigma<0||video_sigma>1||audio_sigma<0||audio_sigma>1) {
        h3_gpu_set_error(g,@"Invalid absolute SOL step/noise policy input");return 0;
    }
    g.solNoiseSet=YES;g.solNoiseStep=step;g.solVideoSigma=video_sigma;g.solAudioSigma=audio_sigma;return 1;
}
int h3_gpu_dit_sdpa_bf16(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t seq,uint32_t heads,
    uint32_t dim,float scale,int head_major,unsigned block,int step) {
    const char *capture=getenv("H3_TEST_NATIVE_QKV");
    const char *blocks=getenv("H3_TEST_NATIVE_QKV_BLOCKS");int capture_block=block==0;
    if(blocks){capture_block=0;const char *cursor=blocks;while(*cursor){char *end=NULL;unsigned long n=strtoul(cursor,&end,10);if(end==cursor)break;if(n==block)capture_block=1;if(*end!=',')break;cursor=end+1;}}
    char selected_path[4096];
    if(capture&&block) {
        if(snprintf(selected_path,sizeof(selected_path),"%s.block%u",capture,block)>=(int)sizeof(selected_path)) {
            h3_gpu_set_error(GPU(gpu),@"QKV capture path too long");return 0;
        }
        capture=selected_path;
    }
    if(capture&&capture_block&&step==0) {
        if(!getenv("H3_TEST_MAX_EVALUATIONS")||strcmp(getenv("H3_TEST_MAX_EVALUATIONS"),"6")) {
            h3_gpu_set_error(GPU(gpu),@"QKV capture requires the six-evaluation test guard");return 0;
        }
        if(!h3_gpu_submit(gpu))return 0;
        FILE *f=fopen(capture,"wbx");
        if(!f){h3_gpu_set_error(GPU(gpu),@"Cannot create QKV diagnostic (destination must be new)");return 0;}
        const h3_gpu_tensor *inputs[]={q,k,v};int ok=1;
        for(unsigned i=0;i<3&&ok;i++) {
            const uint16_t *p=h3_gpu_tensor_contents((h3_gpu_tensor *)inputs[i]);
            if(!p){ok=0;break;}
            if(GPU(gpu).headMajorSDPAInputs) {
                for(uint32_t row=0;row<seq&&ok;row++)for(uint32_t h=0;h<heads&&ok;h++)
                    ok=fwrite(p+((size_t)h*seq+row)*dim,2,dim,f)==dim;
            } else {size_t n=(size_t)seq*heads*dim;ok=fwrite(p,2,n,f)==n;}
        }
        if(fclose(f))ok=0;
        if(!ok){h3_gpu_set_error(GPU(gpu),@"Cannot write QKV diagnostic");return 0;}
        fprintf(stderr,"h3cli: captured real QKV sequence=%u heads=%u dim=%u at %s\n",seq,heads,dim,capture);
        if(!h3_gpu_begin(gpu))return 0;
    }
    h3_backend_scope scope=GPU(gpu).nativeScope;
    if(getenv("H3_TEST_NATIVE_RANGES") && !h3_gpu_diagnostic_scores(gpu,q,k,seq,heads,scale,
        GPU(gpu).headMajorSDPAInputs,block))return 0;
    if(scope.backend==H3_BACKEND_METAL) {
        if(dim!=128) {
            h3_gpu_set_error(GPU(gpu),@"Native attention requires D=128");return 0;
        }
        int input_major=GPU(gpu).headMajorSDPAInputs;
        GPU(gpu).headMajorSDPAInputs=NO;
        int routed=0;
        if(scope.attention==H3_ATTN_SOL) {
            H3GPU *g=GPU(gpu);
            if(!g.solNoiseSet||g.solNoiseStep!=step) {h3_gpu_set_error(g,@"Missing current absolute SOL noise policy");return 0;}
            const char *reason=h3_metal_sol_dense_reason(scope.metal,block,step,g.solVideoSigma,g.solAudioSigma);
            routed=reason==NULL;
            if(h3_profile_steps_enabled())fprintf(stderr,"h3_sol_policy {\"step\":%d,\"block\":%u,\"video_sigma\":%.9g,\"audio_sigma\":%.9g,\"decision\":\"%s\"}\n",
                step+1,block,g.solVideoSigma,g.solAudioSigma,reason?reason:"routed");
        }
        if(scope.metal.precision) {
            if(scope.metal.precision!=1||scope.metal.candidate!=1) {
                h3_gpu_set_error(GPU(gpu),@"Mixed attention requires FP16 and steel-routed");return 0;
            }
            int ok=scope.metal.layout_fusion?
                h3_gpu_mixed_prepared_attention(gpu,out,(h3_gpu_tensor *)q,(h3_gpu_tensor *)k,(h3_gpu_tensor *)v,
                    seq,heads,scale,head_major,scope.metal,routed,block,step):
                routed?h3_gpu_mixed_sol(gpu,out,q,k,v,seq,heads,scale,input_major,head_major,scope.metal,block,step):
                h3_gpu_mixed_sdpa(gpu,out,q,k,v,seq,heads,scale,input_major,head_major,0);
            if(!ok)return 0;
            return h3_gpu_mixed_record(gpu,heads,block,step);
        }
        if(routed)
            return h3_gpu_native_sol_bf16(gpu,out,q,k,v,seq,heads,scale,input_major,head_major,scope.metal,block,step);
        if(scope.metal.candidate)
            return h3_gpu_native_sdpa_candidate_b(gpu,out,q,k,v,seq,heads,scale,input_major,head_major,32);
        return h3_gpu_native_sdpa_bf16(gpu,out,q,k,v,seq,heads,dim,scale,input_major,head_major,-1);
    }
    return head_major?h3_gpu_sdpa_bf16_head_major_output(gpu,out,q,k,v,seq,heads,dim,scale):
        h3_gpu_sdpa_bf16(gpu,out,q,k,v,seq,heads,dim,scale);
}

/* The kernel is pinned in third_party/mlx-attention. Its online softmax,
 * QK/PV accumulators and running output are float32; storage is BF16. */
typedef struct {
    int B,H,D,qL,kL,gqa_factor;
    float scale;
    int NQ,NK,NQ_aligned,NK_aligned,qL_rem,kL_rem,qL_off;
    int64_t Q_strides[3],K_strides[3],V_strides[3],O_strides[3];
} h3_native_attn_params;
_Static_assert(sizeof(h3_native_attn_params)==152,"Metal attention ABI");
_Static_assert(offsetof(h3_native_attn_params,Q_strides)==56,"Metal attention strides ABI");

int h3_gpu_native_sdpa_bf16(h3_gpu *opaque,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t seq,uint32_t heads,uint32_t dim,float scale,
    int input_head_major,int output_head_major,int tile) {
    H3GPU *gpu=GPU(opaque);
    static const int tiles[4][3]={{32,32,128},{32,64,128},{64,32,256},{16,64,64}};
    if(tile==-1)tile=2;
    if(!gpu)return 0;
    if(!seq||seq>10000000||!heads||heads>INT_MAX/128||dim!=128||
        !isfinite(scale)||scale<=0||tile<0||tile>3||
        (input_head_major!=0&&input_head_major!=1)||
        (output_head_major!=0&&output_head_major!=1)) {
        h3_gpu_set_error(gpu,@"Native attention requires BF16 D=128, positive shape/scale, layout 0/1 and tile 0..3");return 0;
    }
    const size_t count=(size_t)seq*heads*dim;
    if(!h3_gpu_require_command(gpu)||
       !h3_gpu_require_elements(gpu,q,count,@"native Q")||
       !h3_gpu_require_elements(gpu,k,count,@"native K")||
       !h3_gpu_require_elements(gpu,v,count,@"native V")||
       !h3_gpu_require_elements(gpu,out,count,@"native O"))return 0;
    if(TENSOR(q).dtype!=H3_GPU_BF16||TENSOR(k).dtype!=H3_GPU_BF16||
       TENSOR(v).dtype!=H3_GPU_BF16||TENSOR(out).dtype!=H3_GPU_BF16||
       TENSOR(out).buffer==TENSOR(q).buffer||TENSOR(out).buffer==TENSOR(k).buffer||
       TENSOR(out).buffer==TENSOR(v).buffer) {
        h3_gpu_set_error(gpu,@"Native attention requires distinct BF16 output storage");return 0;
    }
    @autoreleasepool {
        NSError *error=nil;
        if(!gpu.nativeAttentionLibrary) {
            MTLCompileOptions *options=[MTLCompileOptions new];
            options.languageVersion=MTLLanguageVersion3_1;
            options.mathMode=MTLMathModeSafe;
            NSString *source=[[NSString alloc] initWithBytes:h3_native_attention_source
                length:sizeof(h3_native_attention_source)-1 encoding:NSUTF8StringEncoding];
            gpu.nativeAttentionLibrary=[gpu.device newLibraryWithSource:source options:options error:&error];
            if(!gpu.nativeAttentionLibrary){h3_gpu_set_error(gpu,@"Native attention compilation: %@",error);return 0;}
            gpu.nativeAttentionPipelines=[NSMutableDictionary new];
        }
        const int bq=tiles[tile][0],bk=tiles[tile][1],threads=tiles[tile][2];
        BOOL aq=seq%(uint32_t)bq==0,ak=seq%(uint32_t)bk==0,no=NO;
        NSString *name=[NSString stringWithFormat:@"h3_native_attention_%d",tile];
        NSString *key=[NSString stringWithFormat:@"%@:%d:%d",name,aq,ak];
        id<MTLComputePipelineState> pipeline=gpu.nativeAttentionPipelines[key];
        if(!pipeline) {
            MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
            [constants setConstantValue:&aq type:MTLDataTypeBool atIndex:200];
            [constants setConstantValue:&ak type:MTLDataTypeBool atIndex:201];
            for(NSUInteger i=300;i<=302;i++)[constants setConstantValue:&no type:MTLDataTypeBool atIndex:i];
            id<MTLFunction> function=[gpu.nativeAttentionLibrary newFunctionWithName:name constantValues:constants error:&error];
            if(function)pipeline=[gpu.device newComputePipelineStateWithFunction:function error:&error];
            if(!pipeline){h3_gpu_set_error(gpu,@"Native attention pipeline: %@",error);return 0;}
            if(pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<(NSUInteger)threads||
                pipeline.staticThreadgroupMemoryLength>gpu.device.maxThreadgroupMemoryLength){
                h3_gpu_set_error(gpu,@"Native attention tile exceeds device limits");return 0;
            }
            gpu.nativeAttentionPipelines[key]=pipeline;
        }
        h3_native_attn_params p={.B=1,.H=(int)heads,.D=128,.qL=(int)seq,.kL=(int)seq,
            .gqa_factor=1,.scale=scale,.NQ=((int)seq+bq-1)/bq,.NK=((int)seq+bk-1)/bk,
            .NQ_aligned=(int)seq/bq,.NK_aligned=(int)seq/bk,.qL_rem=(int)seq%bq,.kL_rem=(int)seq%bk};
        p.Q_strides[0]=(int64_t)count;p.Q_strides[1]=input_head_major?(int64_t)seq*dim:dim;
        p.Q_strides[2]=input_head_major?dim:(int64_t)heads*dim;
        memcpy(p.K_strides,p.Q_strides,sizeof(p.Q_strides));memcpy(p.V_strides,p.Q_strides,sizeof(p.Q_strides));
        p.O_strides[0]=(int64_t)count;p.O_strides[1]=output_head_major?(int64_t)seq*dim:dim;
        p.O_strides[2]=output_head_major?dim:(int64_t)heads*dim;
        id<MTLComputeCommandEncoder> encoder=[gpu.command computeCommandEncoder];
        if(!encoder){h3_gpu_set_error(gpu,@"Cannot create native attention encoder");return 0;}
        encoder.label=@"H3 native dense BF16 attention";
        [encoder setComputePipelineState:pipeline];
        [encoder setBuffer:TENSOR(q).buffer offset:0 atIndex:0];
        [encoder setBuffer:TENSOR(k).buffer offset:0 atIndex:1];
        [encoder setBuffer:TENSOR(v).buffer offset:0 atIndex:2];
        [encoder setBuffer:TENSOR(out).buffer offset:0 atIndex:3];
        [encoder setBytes:&p length:sizeof(p) atIndex:4];
        [encoder dispatchThreadgroups:MTLSizeMake((NSUInteger)p.NQ,heads,1)
            threadsPerThreadgroup:MTLSizeMake(32,(NSUInteger)threads/32,1)];
        [encoder endEncoding];
        h3_gpu_stats stats=gpu.stats;stats.native_attention_dispatches++;stats.direct_dispatches++;gpu.stats=stats;
    }
    return 1;
}

#include "src/metal/native_attention_host.inc"
#include "src/metal/fp16_attention_host.inc"
#include "src/metal/native_diagnostics_host.inc"

#include "src/metal/metal_vae_host.inc"

#include "src/upscale/upscale_metal.inc"
