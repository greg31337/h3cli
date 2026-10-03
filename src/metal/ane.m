#import <Foundation/Foundation.h>
#import <CoreML/CoreML.h>
#import <CoreVideo/CoreVideo.h>
#import <CommonCrypto/CommonDigest.h>
#include "src/metal/ane.h"
#include <math.h>
#include <sys/stat.h>
#include <sys/sysctl.h>
#include <time.h>
#include <unistd.h>
#include <mach/mach.h>

@interface H3ANE : NSObject
@property(nonatomic,strong) MLModel *model;
@property(nonatomic,strong) MLMultiArray *x, *w, *y;
@property(nonatomic,strong) MLDictionaryFeatureProvider *inputs;
@property(nonatomic,strong) MLPredictionOptions *prediction;
@property(nonatomic) int chunk, k, n;
@property(nonatomic) BOOL disabled;
@end
@implementation H3ANE
@end

static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}
static void memory(uint64_t *wired,uint64_t *footprint) {
    vm_statistics64_data_t vm={0};mach_msg_type_number_t count=HOST_VM_INFO64_COUNT;
    mach_port_t host=mach_host_self();vm_size_t page=0;host_page_size(host,&page);
    if(host_statistics64(host,HOST_VM_INFO64,(host_info64_t)&vm,&count)==KERN_SUCCESS)*wired=(uint64_t)vm.wire_count*page;
    mach_port_deallocate(mach_task_self(),host);
    task_vm_info_data_t task={0};count=TASK_VM_INFO_COUNT;
    if(task_info(mach_task_self(),TASK_VM_INFO,(task_info_t)&task,&count)==KERN_SUCCESS)*footprint=task.phys_footprint;
}
static BOOL memory_growth(uint64_t wired,uint64_t footprint,h3_ane_stats *stats) {
    uint64_t w=0,f=0;memory(&w,&f);
    if(stats){stats->system_wired_bytes=w;stats->process_footprint_bytes=f;}
    /* CoreML's driver allocations are OS-owned. This is a measured admission
     * guard, not a hard quota on system allocations belonging to other apps. */
    return (w<=wired||w-wired<=4ull*1024*1024*1024)&&
           (f<=footprint||f-footprint<=4ull*1024*1024*1024);
}
static int fail(char *error,size_t size,NSString *message) {
    if(error&&size)snprintf(error,size,"%s",message ? message.UTF8String : "CoreML failure");
    return 0;
}
static NSString *digest(NSData *data) {
    if(!data||data.length>UINT32_MAX)return nil;
    unsigned char hash[CC_SHA256_DIGEST_LENGTH];
    CC_SHA256(data.bytes,(CC_LONG)data.length,hash);
    char s[65];for(unsigned i=0;i<32;i++)snprintf(s+2*i,3,"%02x",hash[i]);
    return [NSString stringWithUTF8String:s];
}
static NSString *file_hash(NSString *path) {
    return digest([NSData dataWithContentsOfFile:path options:NSDataReadingMappedIfSafe error:nil]);
}
static MLMultiArray *array(int rows,int cols) {
    CVPixelBufferRef buffer=NULL;
    NSDictionary *attrs=@{(id)kCVPixelBufferIOSurfacePropertiesKey:@{}};
    if(CVPixelBufferCreate(kCFAllocatorDefault,(size_t)cols,(size_t)rows,
        kCVPixelFormatType_OneComponent16Half,(__bridge CFDictionaryRef)attrs,&buffer)!=kCVReturnSuccess)return nil;
    MLMultiArray *a=[[MLMultiArray alloc] initWithPixelBuffer:buffer shape:@[@(rows),@(cols)]];
    CVPixelBufferRelease(buffer);
    return a;
}
static void count_plan(MLComputePlan *plan,MLModelStructureProgramBlock *block,int *ane,int *total) {
    for(MLModelStructureProgramOperation *op in block.operations) {
        if([op.operatorName isEqualToString:@"matmul"]||[op.operatorName hasSuffix:@".matmul"]) {
            (*total)++;
            if([[plan computeDeviceUsageForMLProgramOperation:op].preferredComputeDevice isKindOfClass:MLNeuralEngineComputeDevice.class])(*ane)++;
        }
        for(MLModelStructureProgramBlock *child in op.blocks)count_plan(plan,child,ane,total);
    }
}

h3_ane *h3_ane_create(int chunk,int k,int n,int tile,const char *cache_directory,
                     h3_ane_stats *stats,char *error,size_t error_size) {
    @autoreleasepool {
        double started=now();if(stats)memset(stats,0,sizeof(*stats));
        uint64_t initialWired=0,initialFootprint=0;memory(&initialWired,&initialFootprint);
        if(chunk<1||chunk>4096||k<1||k>32768||n<1||n>65536||tile<1||tile>4096||
           (uint64_t)k*(uint64_t)n>268435456) {
            fail(error,error_size,@"ANE shape exceeds bounded graph/storage limits");return NULL;
        }
        if(@available(macOS 14.4,*)) {} else {
            fail(error,error_size,@"ANE verification requires macOS 14.4 or later");return NULL;
        }
        NSFileManager *fm=NSFileManager.defaultManager;
        NSString *root=cache_directory?[NSString stringWithUTF8String:cache_directory]:@"outputs/h3-ane-cache";
        root=root.stringByStandardizingPath;
        NSError *e=nil;
        if(![fm createDirectoryAtPath:root withIntermediateDirectories:YES attributes:@{NSFilePosixPermissions:@0700} error:&e]) {
            fail(error,error_size,e.localizedDescription);return NULL;
        }
        struct stat st;
        if(lstat(root.fileSystemRepresentation,&st)||!S_ISDIR(st.st_mode)||st.st_uid!=getuid()||(st.st_mode&0077)) {
            fail(error,error_size,@"ANE cache must be a private directory owned by this user");return NULL;
        }
        char hardware[128]={0},osbuild[128]={0};size_t len=sizeof(hardware);
        sysctlbyname("hw.model",hardware,&len,NULL,0);len=sizeof(osbuild);sysctlbyname("kern.osversion",osbuild,&len,NULL,0);
        /* Runtime-weight graphs contain no checkpoint data. Their compiled
         * identity is shape/recipe/device-specific; each invocation supplies
         * the current original/folded weights, so stale weight reuse is impossible. */
        NSString *identity=[NSString stringWithFormat:@"h3-ane-qkv-v1/vpipe-f34e2cc3a3adae759eea254419f436f5b7800057/fp16-scaled/runtime-oi/%d/%d/%d/%d/%s/%s/%s",
            chunk,k,n,tile,hardware,osbuild,__clang_version__];
        NSString *key=digest([identity dataUsingEncoding:NSUTF8StringEncoding]);
        NSString *path=[root stringByAppendingPathComponent:[key stringByAppendingString:@".mlmodelc"]];
        NSString *manifestPath=[path stringByAppendingPathComponent:@"h3-manifest.json"];
        NSData *manifestData=[NSData dataWithContentsOfFile:manifestPath];
        id decoded=manifestData?[NSJSONSerialization JSONObjectWithData:manifestData options:0 error:nil]:nil;
        NSDictionary *manifest=[decoded isKindOfClass:NSDictionary.class]?decoded:nil;
        BOOL hit=[manifest[@"identity"] isEqual:identity]&&
            [manifest[@"mil_sha256"] isEqual:file_hash([path stringByAppendingPathComponent:@"model.mil"])]&&
            [manifest[@"description_sha256"] isEqual:file_hash([path stringByAppendingPathComponent:@"coremldata.bin"])];
        if([fm fileExistsAtPath:path]&&!hit) {
            fail(error,error_size,@"ANE cache integrity/recipe mismatch; choose an empty cache directory");return NULL;
        }
        if(!hit) {
            NSUInteger count=0;uint64_t bytes=0;
            for(NSURL *url in [fm enumeratorAtURL:[NSURL fileURLWithPath:root] includingPropertiesForKeys:@[NSURLFileSizeKey] options:0 errorHandler:nil]) {
                NSNumber *size=nil;[url getResourceValue:&size forKey:NSURLFileSizeKey error:nil];bytes+=size.unsignedLongLongValue;count++;
            }
            if(bytes>64*1024*1024||count>512) {fail(error,error_size,@"ANE application graph cache is full (64 MiB/512 entries)");return NULL;}
            NSString *stage=[root stringByAppendingPathComponent:[@".stage-" stringByAppendingString:NSUUID.UUID.UUIDString]];
            if(!h3_ane_emit(stage.fileSystemRepresentation,chunk,k,n,tile,error,error_size)) {
                [fm removeItemAtPath:stage error:nil];return NULL;
            }
            NSDictionary *m=@{@"identity":identity,@"mil_sha256":file_hash([stage stringByAppendingPathComponent:@"model.mil"]),
                @"description_sha256":file_hash([stage stringByAppendingPathComponent:@"coremldata.bin"]),
                @"weights":@"runtime BF16 source converted under recipe 1; no weights cached"};
            NSData *data=[NSJSONSerialization dataWithJSONObject:m options:NSJSONWritingSortedKeys error:&e];
            BOOL ok=[data writeToFile:[stage stringByAppendingPathComponent:@"h3-manifest.json"] options:NSDataWritingAtomic error:&e]&&
                [fm moveItemAtPath:stage toPath:path error:&e];
            if(!ok){[fm removeItemAtPath:stage error:nil];fail(error,error_size,e.localizedDescription);return NULL;}
        }
        MLModelConfiguration *config=[MLModelConfiguration new];config.computeUnits=MLComputeUnitsCPUAndNeuralEngine;
        NSURL *url=[NSURL fileURLWithPath:path];
        MLModel *model=[MLModel modelWithContentsOfURL:url configuration:config error:&e];
        if(!model){fail(error,error_size,e.localizedDescription);return NULL;}
        __block MLComputePlan *plan=nil;__block NSError *planError=nil;
        dispatch_semaphore_t ready=dispatch_semaphore_create(0);
        [MLComputePlan loadContentsOfURL:url configuration:config completionHandler:^(MLComputePlan *p,NSError *problem){plan=p;planError=problem;dispatch_semaphore_signal(ready);}];
        if(dispatch_semaphore_wait(ready,dispatch_time(DISPATCH_TIME_NOW,60*NSEC_PER_SEC))) {
            fail(error,error_size,@"ANE compute-plan verification timed out");return NULL;
        }
        if(!plan){fail(error,error_size,planError.localizedDescription);return NULL;}
        int ane=0,total=0;
        for(MLModelStructureProgramFunction *f in plan.modelStructure.program.functions.allValues)count_plan(plan,f.block,&ane,&total);
        if(stats){stats->ane_matmuls=ane;stats->total_matmuls=total;stats->cache_hit=hit;stats->load_seconds=now()-started;}
        if(!total||ane!=total){fail(error,error_size,[NSString stringWithFormat:@"CoreML plans %d/%d matmuls on ANE; refusing CPU-only/mixed fallback",ane,total]);return NULL;}
        H3ANE *a=[H3ANE new];a.model=model;a.chunk=chunk;a.k=k;a.n=n;
        a.x=array(chunk,k);a.w=array(n,k);a.y=array(chunk,n);
        if(!a.x||!a.w||!a.y){fail(error,error_size,@"ANE IOSurface allocation failed");return NULL;}
        a.inputs=[[MLDictionaryFeatureProvider alloc] initWithDictionary:@{@"x":a.x,@"w":a.w} error:&e];
        a.prediction=[MLPredictionOptions new];a.prediction.outputBackings=@{@"out":a.y};
        if(!a.inputs){fail(error,error_size,e.localizedDescription);return NULL;}
        /* Every newly loaded executable must pass a sparse, exactly
         * representable numerical check. Loading an undocumented format is
         * insufficient evidence that its tensor layout is still compatible. */
        uint16_t *wx=calloc((size_t)k*n,2),*xx=calloc((size_t)k,2),*yy=calloc((size_t)n,2);
        if(!wx||!xx||!yy){free(wx);free(xx);free(yy);fail(error,error_size,@"ANE format self-test allocation failed");return NULL;}
        int selected=k/2;
        xx[selected]=0x3fc0; /* BF16 1.5 */
        for(int j=0;j<n;j++)wx[(size_t)j*k+((size_t)j*37%(size_t)k)]=0x3f00; /* BF16 .5 */
        int valid=h3_ane_predict((__bridge h3_ane *)a,xx,wx,yy,1,NULL,error,error_size);
        for(int j=0;j<n&&valid;j++)if(yy[j]!=((size_t)j*37%(size_t)k==(size_t)selected?0x3f40:0))valid=0;
        free(wx);free(xx);free(yy);
        if(!valid){fail(error,error_size,@"ANE compiled-format numerical self-test failed");return NULL;}
        if(!memory_growth(initialWired,initialFootprint,stats)) {
            fail(error,error_size,@"ANE graph exceeded the 4 GiB measured memory-growth guard");return NULL;
        }
        if(stats)stats->memory_bytes=(size_t)(a.x.count+a.w.count+a.y.count)*2;
        if(stats)stats->load_seconds=now()-started;
        return (__bridge_retained h3_ane *)a;
    }
}
void h3_ane_free(h3_ane *ane){if(ane){H3ANE *a=CFBridgingRelease(ane);(void)a;}}
static float unbf(uint16_t x){uint32_t bits=(uint32_t)x<<16;float f;memcpy(&f,&bits,4);return f;}
static uint16_t bf(float f){uint32_t bits;memcpy(&bits,&f,4);bits+=0x7fff+((bits>>16)&1);return (uint16_t)(bits>>16);}
static double weight_max(const uint16_t *w,size_t count) {
    double results[8]={0};double *maxima=results;
    dispatch_apply(8,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t part){
        double m=0;for(size_t i=count*part/8;i<count*(part+1)/8;i++) {
            float f=unbf(w[i]);if(!isfinite(f)){m=INFINITY;break;}m=fmax(m,fabs(f));
        }maxima[part]=m;
    });
    double m=0;for(unsigned i=0;i<8;i++)m=fmax(m,results[i]);return m;
}

int h3_ane_predict(h3_ane *opaque,const uint16_t *x,const uint16_t *w,uint16_t *y,
                   int rows,h3_ane_stats *stats,char *error,size_t error_size) {
    @autoreleasepool {
        H3ANE *a=(__bridge H3ANE *)opaque;
        if(!a||a.disabled||!x||!w||!y||rows<1||rows>16384||
           ((uint64_t)a.k*a.n+(uint64_t)a.chunk*(a.k+a.n)+(uint64_t)rows*a.n*2)*2>2ull*1024*1024*1024)
            return fail(error,error_size,@"invalid ANE prediction arguments or 2 GiB workspace limit exceeded");
        uint64_t initialWired=0,initialFootprint=0;memory(&initialWired,&initialFootprint);
        double started=now(),maxw=weight_max(w,(size_t)a.k*a.n);
        if(!isfinite(maxw))return fail(error,error_size,@"nonfinite BF16 ANE weight");
        /* Power-of-two operand scaling bounds every reduction, including
         * tiled partial sums, below FP16 overflow. FP32 restores output scale.
         * Conversion error is screened separately; no saturation is allowed. */
        int we=maxw>0?ilogb(maxw):0;
        double ws=scalbn(1.,-we);double input_bound=8192./((double)a.k*fmax(maxw*ws,1.));
        if(CVPixelBufferLockBaseAddress(a.w.pixelBuffer,0)!=kCVReturnSuccess)return fail(error,error_size,@"ANE weight lock failed");
        size_t wstride=CVPixelBufferGetBytesPerRow(a.w.pixelBuffer)/2;
        _Float16 *wh=CVPixelBufferGetBaseAddress(a.w.pixelBuffer);
        double errw=0,normw=0,errors[8]={0},norms[8]={0};double *pe=errors,*pn=norms;
        dispatch_apply(8,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t part){
            double er=0,norm=0;
            for(size_t r=(size_t)a.n*part/8;r<(size_t)a.n*(part+1)/8;r++) {
                for(int c=0;c<a.k;c++) {
                    double f=unbf(w[r*a.k+c])*ws;_Float16 h=(_Float16)f;
                    wh[r*wstride+c]=h;double d=(double)h-f;er+=d*d;norm+=f*f;
                }
            }
            pe[part]=er;pn[part]=norm;
        });
        for(unsigned i=0;i<8;i++){errw+=errors[i];normw+=norms[i];}
        CVPixelBufferUnlockBaseAddress(a.w.pixelBuffer,0);
        if(errw>1e-8*normw)return fail(error,error_size,@"ANE weight conversion exceeds range policy");
        uint16_t *result=malloc((size_t)rows*a.n*2);double *scales=malloc((size_t)a.chunk*sizeof(*scales));
        if(!result||!scales){free(result);free(scales);return fail(error,error_size,@"ANE result allocation failed");}
        int ok=1,minXe=INT_MAX,maxXe=0;double pack=now()-started,predict=0,unpack=0;double first=0,last=0;
        for(int offset=0;offset<rows&&ok;offset+=a.chunk) {
            double stamp=now();int take=MIN(a.chunk,rows-offset);
            if(CVPixelBufferLockBaseAddress(a.x.pixelBuffer,0)!=kCVReturnSuccess){ok=fail(error,error_size,@"ANE input lock failed");break;}
            size_t stride=CVPixelBufferGetBytesPerRow(a.x.pixelBuffer)/2;_Float16 *xh=CVPixelBufferGetBaseAddress(a.x.pixelBuffer);
            memset(xh,0,CVPixelBufferGetBytesPerRow(a.x.pixelBuffer)*(size_t)a.chunk);
            for(int r=0;r<take&&ok;r++) {
                double mx=0;for(int c=0;c<a.k;c++){float f=unbf(x[(size_t)(offset+r)*a.k+c]);if(!isfinite(f)){ok=0;break;}mx=fmax(mx,fabs(f));}
                int xe=mx>input_bound?(int)ceil(log2(mx/input_bound)):0;
                minXe=MIN(minXe,xe);maxXe=MAX(maxXe,xe);
                double xs=scalbn(1.,-xe);scales[r]=scalbn(1.,we+xe);
                double errx=0,normx=0;
                for(int c=0;c<a.k&&ok;c++){double f=unbf(x[(size_t)(offset+r)*a.k+c])*xs;_Float16 h=(_Float16)f;xh[(size_t)r*stride+c]=h;double d=(double)h-f;errx+=d*d;normx+=f*f;}
                if(errx>1e-8*normx)ok=0;
            }
            CVPixelBufferUnlockBaseAddress(a.x.pixelBuffer,0);pack+=now()-stamp;
            if(!ok){fail(error,error_size,@"ANE input is nonfinite or exceeds conversion range policy");break;}
            NSError *e=nil;stamp=now();if(first==0)first=stamp;
            id<MLFeatureProvider> output=[a.model predictionFromFeatures:a.inputs options:a.prediction error:&e];
            last=now();predict+=last-stamp;
            MLMultiArray *out=[output featureValueForName:@"out"].multiArrayValue;
            if(!out||out.dataType!=MLMultiArrayDataTypeFloat16||![out.shape isEqual:a.y.shape]){ok=fail(error,error_size,e?e.localizedDescription:@"ANE output interface changed");break;}
            stamp=now();size_t sy=out.strides[0].unsignedLongValue,sx=out.strides[1].unsignedLongValue;
            int validity[8]={1,1,1,1,1,1,1,1};int *pv=validity;
            [out getBytesWithHandler:^(const void *bytes,NSInteger size) {
                (void)size;const _Float16 *yh=bytes;
                dispatch_apply(8,dispatch_get_global_queue(QOS_CLASS_USER_INITIATED,0),^(size_t part){
                    for(size_t r=(size_t)take*part/8;r<(size_t)take*(part+1)/8&&pv[part];r++)for(int c=0;c<a.n;c++) {
                        float value=(float)((double)yh[r*sy+(size_t)c*sx]*scales[r]);
                        uint16_t rounded=bf(value);
                        if(!isfinite(value)||!isfinite(unbf(rounded))){pv[part]=0;break;}
                        result[((size_t)offset+r)*a.n+c]=rounded;
                    }
                });
            }];
            int finite=1;for(unsigned i=0;i<8;i++)finite&=validity[i];
            if(!finite)ok=fail(error,error_size,@"ANE output failed finite BF16 commit check");
            unpack+=now()-stamp;
        }
        if(!memory_growth(initialWired,initialFootprint,stats)) {
            a.disabled=YES;a.model=nil;a.inputs=nil;a.prediction=nil;a.x=nil;a.w=nil;a.y=nil;
            ok=fail(error,error_size,@"ANE prediction exceeded the 4 GiB measured memory-growth guard; backend released");
        }
        if(ok)memcpy(y,result,(size_t)rows*a.n*2);
        free(result);free(scales);
        if(stats){stats->pack_seconds=pack;stats->predict_seconds=predict;stats->unpack_seconds=unpack;stats->predict_begin=first;stats->predict_end=last;
            stats->weight_scale_exponent=-we;stats->input_scale_exponent_min=minXe==INT_MAX?0:-maxXe;stats->input_scale_exponent_max=minXe==INT_MAX?0:-minXe;}
        return ok;
    }
}
