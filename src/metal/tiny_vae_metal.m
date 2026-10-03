/* Native MPSGraph TAEH3 decoder. Architecture attribution in src/vae/tiny_vae.c. */
#include "src/vae/tiny_vae_internal.h"
#import <Foundation/Foundation.h>
#import <Metal/Metal.h>
#import <MetalPerformanceShadersGraph/MetalPerformanceShadersGraph.h>
#include <stdio.h>
#include <stdlib.h>
#include "src/sampling/av_state.h"

@interface H3TinyPlan : NSObject
@property(nonatomic,strong) MPSGraph *graph;
@property(nonatomic,strong) MPSGraphTensor *input;
@property(nonatomic,strong) MPSGraphTensor *output;
@property(nonatomic,strong) NSArray<MPSGraphTensor *> *historyInputs;
@property(nonatomic,strong) NSArray<MPSGraphTensor *> *historyOutputs;
@end
@implementation H3TinyPlan
@end
@interface H3TinyMetal : NSObject
@property(nonatomic,strong) id<MTLDevice> device;
@property(nonatomic,strong) id<MTLCommandQueue> queue;
@property(nonatomic,strong) NSMutableDictionary<NSString *,NSData *> *weights;
@property(nonatomic,strong) NSMutableDictionary<NSString *,NSArray<NSNumber *> *> *shapes;
@property(nonatomic,strong) NSMutableDictionary<NSString *,H3TinyPlan *> *plans;
@property(nonatomic,strong) NSArray<MPSGraphTensorData *> *history;
@property(nonatomic,strong) id<MTLBuffer> inputBuffer;
@property(nonatomic) double computeSeconds, transferSeconds;
@property(nonatomic) NSUInteger peakBytes, planBuilds;
@end
@implementation H3TinyMetal
@end
static int error_message(char *e,size_t n,NSString *s){if(e&&n)snprintf(e,n,"%s",s.UTF8String);return 0;}
static MPSGraphTensor *conv(H3TinyMetal *d,MPSGraph *g,MPSGraphTensor *x,NSString *name){
    NSString *wk=[name stringByAppendingString:@".weight"],*bk=[name stringByAppendingString:@".bias"];
    NSArray<NSNumber *> *shape=d.shapes[wk];NSUInteger k=shape[2].unsignedIntegerValue;
    MPSGraphTensor *weight=[g constantWithData:d.weights[wk] shape:shape dataType:MPSDataTypeFloat16];
    MPSGraphConvolution2DOpDescriptor *desc=[MPSGraphConvolution2DOpDescriptor
        descriptorWithStrideInX:1 strideInY:1 dilationRateInX:1 dilationRateInY:1 groups:1
        paddingLeft:k/2 paddingRight:k/2 paddingTop:k/2 paddingBottom:k/2
        paddingStyle:MPSGraphPaddingStyleExplicit dataLayout:MPSGraphTensorNamedDataLayoutNHWC
        weightsLayout:MPSGraphTensorNamedDataLayoutOIHW];
    x=[g convolution2DWithSourceTensor:x weightsTensor:weight descriptor:desc name:nil];
    if(d.weights[bk])x=[g additionWithPrimaryTensor:x secondaryTensor:
        [g constantWithData:d.weights[bk] shape:d.shapes[bk] dataType:MPSDataTypeFloat16] name:nil];
    return x;
}
static H3TinyPlan *plan(H3TinyMetal *d,int count,int height,int width){
    NSString *key=[NSString stringWithFormat:@"%d:%d:%d",count,height,width];
    H3TinyPlan *p=d.plans[key];if(p)return p;
    /* Only full and tail batch plans are retained. Geometry switches do not
     * grow a persistent graph/weight cache without bound. */
    if(d.plans.count>=2)[d.plans removeAllObjects];
    d.planBuilds++;p=[H3TinyPlan new];MPSGraph *g=[MPSGraph new];p.graph=g;
    p.input=[g placeholderWithShape:@[@(count),@(height),@(width),@24] dataType:MPSDataTypeFloat32 name:@"normalized_latents"];
    MPSGraphTensor *x=[g castTensor:p.input toType:MPSDataTypeFloat16 name:nil];
    MPSGraphTensor *three=[g constantWithScalar:3 dataType:MPSDataTypeFloat16];
    x=[g multiplicationWithPrimaryTensor:[g tanhWithTensor:[g divisionWithPrimaryTensor:x secondaryTensor:three name:nil] name:nil] secondaryTensor:three name:nil];
    x=[g reLUWithTensor:conv(d,g,x,@"decoder.1") name:nil];
    NSMutableArray *hi=[NSMutableArray array],*ho=[NSMutableArray array];
    int n=count,h=height,w=width;
    const int ids[3][3]={{3,4,5},{9,10,11},{15,16,17}},channels[]={256,128,64};
    for(int stage=0;stage<3;stage++){
        int c=channels[stage];
        for(int j=0;j<3;j++){
            MPSGraphTensor *mem=[g placeholderWithShape:@[@1,@(h),@(w),@(c)] dataType:MPSDataTypeFloat16 name:nil];
            [hi addObject:mem];
            [ho addObject:[g sliceTensor:x dimension:0 start:n-1 length:1 name:nil]];
            MPSGraphTensor *past=mem;
            if(n>1)past=[g concatTensors:@[mem,[g sliceTensor:x dimension:0 start:0 length:n-1 name:nil]] dimension:0 name:nil];
            MPSGraphTensor *y=[g concatTensors:@[x,past] dimension:3 name:nil];
            for(int layer=0;layer<3;layer++){
                y=conv(d,g,y,[NSString stringWithFormat:@"decoder.%d.conv.%d",ids[stage][j],2*layer]);
                if(layer<2)y=[g reLUWithTensor:y name:nil];
            }
            x=[g reLUWithTensor:[g additionWithPrimaryTensor:x secondaryTensor:y name:nil] name:nil];
        }
        x=[g reshapeTensor:x withShape:@[@(n),@(h),@1,@(w),@1,@(c)] name:nil];
        x=[g broadcastTensor:x toShape:@[@(n),@(h),@2,@(w),@2,@(c)] name:nil];
        h*=2;w*=2;
        x=[g reshapeTensor:x withShape:@[@(n),@(h),@(w),@(c)] name:nil];
        int grow=stage==0?7:stage==1?13:19,stride=stage?2:1;
        x=conv(d,g,x,[NSString stringWithFormat:@"decoder.%d.conv",grow]);
        x=[g reshapeTensor:x withShape:@[@(n),@(h),@(w),@(stride),@(c)] name:nil];
        x=[g transposeTensor:x permutation:@[@0,@3,@1,@2,@4] name:nil];n*=stride;
        x=[g reshapeTensor:x withShape:@[@(n),@(h),@(w),@(c)] name:nil];
        x=conv(d,g,x,[NSString stringWithFormat:@"decoder.%d",grow+1]);
    }
    x=conv(d,g,[g reLUWithTensor:x name:nil],@"decoder.22");
    x=[g reshapeTensor:x withShape:@[@(n),@(h),@(w),@3,@2,@2] name:nil];
    x=[g transposeTensor:x permutation:@[@0,@1,@4,@2,@5,@3] name:nil];
    x=[g reshapeTensor:x withShape:@[@(n),@(h*2),@(w*2),@3] name:nil];
    x=[g clampWithTensor:x minValueTensor:[g constantWithScalar:0 dataType:MPSDataTypeFloat16]
        maxValueTensor:[g constantWithScalar:1 dataType:MPSDataTypeFloat16] name:nil];
    p.output=[g castTensor:x toType:MPSDataTypeFloat32 name:nil];
    p.historyInputs=hi;p.historyOutputs=ho;d.plans[key]=p;return p;
}
void *h3_tiny_backend_load(const char *path,char *error,size_t size){@autoreleasepool{
    h3_st_header header={0};
    @try{
        H3TinyMetal *d=[H3TinyMetal new];d.device=MTLCreateSystemDefaultDevice();d.queue=[d.device newCommandQueue];
        if(!d.queue){error_message(error,size,@"cannot initialize Metal for preview VAE");return NULL;}
        d.weights=[NSMutableDictionary dictionary];d.shapes=[NSMutableDictionary dictionary];d.plans=[NSMutableDictionary dictionary];
        if(!h3_st_read_header(path,&header,error,size))return NULL;
        int ok=1;
        for(size_t i=0;i<header.tensor_count&&ok;i++){
            const h3_st_tensor *v=&header.tensors[i];if(strncmp(v->name,"decoder.",8))continue;
            NSMutableData *data=[NSMutableData dataWithLength:(NSUInteger)(v->data_end-v->data_begin)];
            if(!data||!h3_st_read_data(&header,v,data.mutableBytes,data.length,error,size)){ok=0;break;}
            NSString *name=[NSString stringWithUTF8String:v->name];NSMutableArray *shape=[NSMutableArray array];
            for(int j=0;j<v->ndim;j++)[shape addObject:@(v->shape[j])];d.weights[name]=data;d.shapes[name]=shape;
        }
        if(!ok)return NULL;
        return (__bridge_retained void *)d;
    }@catch(NSException *e){error_message(error,size,[NSString stringWithFormat:@"preview VAE Metal load: %@",e.reason]);return NULL;}
    @finally{h3_st_free_header(&header);}
}}
void h3_tiny_backend_free(void *backend){if(backend){@autoreleasepool{id value=(__bridge_transfer id)backend;(void)value;}}}
void h3_tiny_backend_reset(void *backend){if(backend){
    H3TinyMetal *d=(__bridge H3TinyMetal *)backend;d.history=nil;
    d.computeSeconds=d.transferSeconds=0;d.planBuilds=0;d.peakBytes=d.device.currentAllocatedSize;
}}
void h3_tiny_backend_profile(void *backend){if(backend&&getenv("H3_PROFILE")){
    H3TinyMetal *d=(__bridge H3TinyMetal *)backend;
    fprintf(stderr,"h3cli: tiny Metal FP16: compute=%.6f transfer=%.6f s device_allocated_peak=%lu plans=%lu built=%lu\n",
        d.computeSeconds,d.transferSeconds,(unsigned long)d.peakBytes,(unsigned long)d.plans.count,(unsigned long)d.planBuilds);
}}
int h3_tiny_backend_chunk(void *backend,const float *input,int t,int h,int w,float **output,char *error,size_t size){@autoreleasepool{
    *output=NULL;float *pixels=NULL;
    @try{
        H3TinyMetal *d=(__bridge H3TinyMetal *)backend;H3TinyPlan *p=plan(d,t,h,w);
        NSUInteger bytes=(NSUInteger)t*(NSUInteger)h*(NSUInteger)w*24*sizeof(float);
        if(d.inputBuffer.length<bytes)d.inputBuffer=[d.device newBufferWithLength:bytes options:MTLResourceStorageModeShared];
        id<MTLBuffer> buffer=d.inputBuffer;
        if(!buffer)return error_message(error,size,@"cannot allocate Metal preview input");
        memcpy(buffer.contents,input,bytes);
        NSMutableDictionary *feeds=[NSMutableDictionary dictionary];
        feeds[p.input]=[[MPSGraphTensorData alloc] initWithMTLBuffer:buffer shape:p.input.shape dataType:MPSDataTypeFloat32];
        for(NSUInteger i=0;i<p.historyInputs.count;i++){
            MPSGraphTensor *tensor=p.historyInputs[i];MPSGraphTensorData *data=d.history?d.history[i]:nil;
            if(!data){NSUInteger elements=1;for(NSNumber *n in tensor.shape)elements*=n.unsignedIntegerValue;
                id<MTLBuffer> zero=[d.device newBufferWithLength:elements*2 options:MTLResourceStorageModeShared];
                if(!zero)return error_message(error,size,@"cannot allocate Metal preview history");
                memset(zero.contents,0,elements*2);
                data=[[MPSGraphTensorData alloc] initWithMTLBuffer:zero shape:tensor.shape dataType:MPSDataTypeFloat16];}
            feeds[tensor]=data;
        }
        NSArray *targets=[p.historyOutputs arrayByAddingObject:p.output];
        double begin=h3_av_now();
        NSDictionary *results=[p.graph runWithMTLCommandQueue:d.queue feeds:feeds targetTensors:targets targetOperations:nil];
        d.computeSeconds+=h3_av_now()-begin;
        NSMutableArray *history=[NSMutableArray array];for(MPSGraphTensor *tensor in p.historyOutputs){
            if(!results[tensor])return error_message(error,size,@"Metal preview omitted history output");[history addObject:results[tensor]];}
        MPSGraphTensorData *rgb=results[p.output];if(!rgb)return error_message(error,size,@"Metal preview omitted RGB output");
        size_t elements=(size_t)t*4*(size_t)h*16*(size_t)w*16*3;
        pixels=malloc(elements*sizeof(float));if(!pixels)return error_message(error,size,@"cannot allocate preview RGB batch");
        begin=h3_av_now();[rgb.mpsndarray readBytes:pixels strideBytes:nil];d.transferSeconds+=h3_av_now()-begin;
        d.peakBytes=MAX(d.peakBytes,d.device.currentAllocatedSize);d.history=history;*output=pixels;pixels=NULL;return 1;
    }@catch(NSException *e){free(pixels);return error_message(error,size,[NSString stringWithFormat:@"preview VAE Metal execution: %@",e.reason]);}
}}
