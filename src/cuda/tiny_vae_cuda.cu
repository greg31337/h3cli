/* Native TAEH3 CUDA decoder; architecture attribution in src/vae/tiny_vae.c.
 * NHWC FP16 activations/weights, FP32 convolution accumulation. Each decoder
 * owns one nonblocking stream. No transformer stream or scratch is borrowed. */
extern "C" {
#include "src/vae/tiny_vae_internal.h"
#include "src/device.h"
}
#include <cuda_runtime.h>
#include <cuda_fp16.h>
#include <cublas_v2.h>
#ifdef H3_CUDA_USE_CUDNN
#include <cudnn.h>
#endif
#include <algorithm>
#include <chrono>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <map>
#include <memory>
#include <stdexcept>
#include <string>
#include <tuple>
#include <vector>

namespace {
using clock_type=std::chrono::steady_clock;
double now() { return std::chrono::duration<double>(clock_type::now().time_since_epoch()).count(); }
void check(cudaError_t s,const char *where) {
    if(s!=cudaSuccess)throw std::runtime_error(std::string("tiny CUDA ")+where+": "+cudaGetErrorString(s));
}
void blas_check(cublasStatus_t s,const char *where) {
    if(s!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error(std::string("tiny cuBLAS ")+where+": status "+std::to_string(s));
}
struct device_scope {
    int previous=-1;
    explicit device_scope(int device) { check(cudaGetDevice(&previous),"current device");check(cudaSetDevice(device),"select device"); }
    ~device_scope() { if(previous>=0)cudaSetDevice(previous); }
};
struct buffer {
    void *data=nullptr;size_t bytes=0;
    buffer()=default;buffer(const buffer&)=delete;buffer& operator=(const buffer&)=delete;
    ~buffer(){if(data)cudaFree(data);}
    half *fp16()const{return static_cast<half*>(data);}
};
struct weight {buffer value;int out=0,in=0,k=0;};
#ifdef H3_CUDA_USE_CUDNN
struct conv_plan {
    cudnnTensorDescriptor_t x=nullptr,y=nullptr;
    cudnnFilterDescriptor_t weight=nullptr;cudnnConvolutionDescriptor_t conv=nullptr;
    cudnnConvolutionFwdAlgo_t algorithm=CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_GEMM;
    size_t workspace=0;bool usable=false;
    ~conv_plan(){if(x)cudnnDestroyTensorDescriptor(x);if(y)cudnnDestroyTensorDescriptor(y);
        if(weight)cudnnDestroyFilterDescriptor(weight);if(conv)cudnnDestroyConvolutionDescriptor(conv);}
};
using conv_key=std::tuple<int,int,int,int,int,int>;
#endif
struct decoder {
    int device=0;cudaStream_t stream=nullptr;cublasHandle_t blas=nullptr;
    std::map<std::string,std::unique_ptr<weight>> weights;
    buffer activation[3],history[9],input,rgb,scratch;
    void *readback=nullptr;size_t pinned_bytes=0,allocated=0,peak=0;
    size_t workspace_limit=32u<<20;int history_h=0,history_w=0;
    double compute_seconds=0,transfer_seconds=0;
    uint64_t blas_calls=0,cudnn_calls=0,plan_builds=0;
    std::string policy="auto";
#ifdef H3_CUDA_USE_CUDNN
    cudnnHandle_t cudnn=nullptr;
    std::map<conv_key,std::unique_ptr<conv_plan>> plans;
#endif
    ~decoder(){
        cudaSetDevice(device);if(stream)cudaStreamSynchronize(stream);
#ifdef H3_CUDA_USE_CUDNN
        plans.clear();if(cudnn)cudnnDestroy(cudnn);
#endif
        if(blas)cublasDestroy(blas);if(readback)cudaFreeHost(readback);
        if(stream)cudaStreamDestroy(stream);
        // Member buffers are freed on this device after all work is drained.
    }
    void reserve(buffer &b,size_t bytes){
        if(bytes<=b.bytes)return;
        const char *limit=getenv("H3_TEST_TINY_CUDA_MAX_BYTES");
        if(limit&&allocated-b.bytes+bytes>strtoull(limit,nullptr,10))
            throw std::runtime_error("tiny CUDA injected allocation budget exceeded");
        // Reallocation is only at chunk boundaries, before any consumers.
        check(cudaStreamSynchronize(stream),"buffer reuse fence");
        if(b.data){check(cudaFree(b.data),"buffer release");allocated-=b.bytes;b.data=nullptr;b.bytes=0;}
        check(cudaMalloc(&b.data,bytes),"allocation");b.bytes=bytes;allocated+=bytes;peak=std::max(peak,allocated);
    }
};
unsigned blocks(size_t n){return (unsigned)std::min<size_t>((n+255)/256,65535);}
__global__ void transform_input(half *out,const float *in,size_t n){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x)
        out[i]=__float2half(3.f*tanhf(__half2float(__float2half(in[i]))/3.f));
}
__global__ void activation(half *out,const half *bias,const half *residual,size_t n,int c,bool relu){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x){
        half v=out[i];if(bias)v=__hadd(v,bias[i%(size_t)c]);if(residual)v=__hadd(v,residual[i]);
        float f=__half2float(v);out[i]=__float2half(relu&&f<0?0:f);
    }
}
__global__ void assemble_memory(half *out,const half *x,const half *history,size_t frame,size_t n,int c){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x){
        size_t channel=i%(size_t)(2*c),pixel=i/(size_t)(2*c),source=pixel*(size_t)c+channel%(size_t)c;
        out[i]=channel<(size_t)c?x[source]:(source<frame?history[source]:x[source-frame]);
    }
}
__global__ void upsample(half *out,const half *in,size_t n,int h,int w,int c){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x){
        size_t q=i/(size_t)c;int x=(int)(q%(size_t)(w*2));q/=(size_t)(w*2);
        int y=(int)(q%(size_t)(h*2));size_t t=q/(size_t)(h*2);
        out[i]=in[((t*(size_t)h+(size_t)(y/2))*(size_t)w+(size_t)(x/2))*(size_t)c+i%(size_t)c];
    }
}
__global__ void expand_time(half *out,const half *in,size_t n,size_t hw,int c){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x){
        size_t q=i/(size_t)c,p=q%hw,t=q/hw;
        out[i]=in[((t/2*hw+p)*2+t%2)*(size_t)c+i%(size_t)c];
    }
}
__global__ void pixel_shuffle(float *out,const half *in,size_t n,int h,int w){
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x){
        size_t q=i/3;int x=(int)(q%(size_t)(w*2));q/=(size_t)(w*2);
        int y=(int)(q%(size_t)(h*2));size_t t=q/(size_t)(h*2);
        size_t from=((t*(size_t)h+(size_t)(y/2))*(size_t)w+(size_t)(x/2))*12+(i%3)*4+(size_t)(y%2)*2+(size_t)(x%2);
        float v=__half2float(in[from]);out[i]=isnan(v)?v:fminf(1.f,fmaxf(0.f,v));
    }
}
/* Bounded NHWC patch panel: row = output pixel, column = ky,kx,input-channel.
 * The tensor-core GEMM reads this as column-major [K, pixels]. */
__global__ void im2col(half *out,const half *in,size_t count,size_t first,int h,int w,int c,int k){
    size_t inner=(size_t)k*k*c,n=count*inner;
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)blockDim.x*gridDim.x){
        size_t p=first+i/inner,col=i%inner;int channel=(int)(col%(size_t)c);col/=(size_t)c;
        int x=(int)(p%(size_t)w)+(int)(col%(size_t)k)-k/2;
        int y=(int)(p/(size_t)w%(size_t)h)+(int)(col/(size_t)k)-k/2;
        size_t t=p/((size_t)h*w);
        out[i]=x>=0&&x<w&&y>=0&&y<h?in[((t*(size_t)h+(size_t)y)*(size_t)w+(size_t)x)*(size_t)c+(size_t)channel]:__float2half(0.f);
    }
}
#ifdef H3_CUDA_USE_CUDNN
conv_plan &get_plan(decoder &d,int n,int h,int w,const weight &v){
    conv_key key={n,h,w,v.in,v.out,v.k};auto found=d.plans.find(key);if(found!=d.plans.end())return *found->second;
    if(d.plans.size()>=64){check(cudaStreamSynchronize(d.stream),"plan retirement");d.plans.clear();}
    auto p=std::make_unique<conv_plan>();d.plan_builds++;
    bool ok=cudnnCreateTensorDescriptor(&p->x)==CUDNN_STATUS_SUCCESS&&cudnnCreateTensorDescriptor(&p->y)==CUDNN_STATUS_SUCCESS&&
        cudnnCreateFilterDescriptor(&p->weight)==CUDNN_STATUS_SUCCESS&&cudnnCreateConvolutionDescriptor(&p->conv)==CUDNN_STATUS_SUCCESS;
    if(ok)ok=cudnnSetTensor4dDescriptor(p->x,CUDNN_TENSOR_NHWC,CUDNN_DATA_HALF,n,v.in,h,w)==CUDNN_STATUS_SUCCESS&&
        cudnnSetTensor4dDescriptor(p->y,CUDNN_TENSOR_NHWC,CUDNN_DATA_HALF,n,v.out,h,w)==CUDNN_STATUS_SUCCESS&&
        cudnnSetFilter4dDescriptor(p->weight,CUDNN_DATA_HALF,CUDNN_TENSOR_NHWC,v.out,v.in,v.k,v.k)==CUDNN_STATUS_SUCCESS&&
        cudnnSetConvolution2dDescriptor(p->conv,v.k/2,v.k/2,1,1,1,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT)==CUDNN_STATUS_SUCCESS&&
        cudnnSetConvolutionMathType(p->conv,CUDNN_TENSOR_OP_MATH)==CUDNN_STATUS_SUCCESS;
    if(ok){
        cudnnConvolutionFwdAlgoPerf_t candidates[8];int count=0;
        if(cudnnGetConvolutionForwardAlgorithm_v7(d.cudnn,p->x,p->weight,p->conv,p->y,8,&count,candidates)==CUDNN_STATUS_SUCCESS)
            for(int i=0;i<count;i++)if(candidates[i].status==CUDNN_STATUS_SUCCESS&&candidates[i].memory<=d.workspace_limit){
                p->algorithm=candidates[i].algo;p->workspace=candidates[i].memory;p->usable=true;break;
            }
    }
    auto &value=*p;d.plans.emplace(key,std::move(p));return value;
}
#endif
void convolution(decoder &d,const half *in,half *out,int n,int h,int w,const std::string &name,bool relu=false,const half *residual=nullptr){
    const auto &v=*d.weights.at(name+".weight");bool used=false;
#ifdef H3_CUDA_USE_CUDNN
    if(d.cudnn){
        auto &p=get_plan(d,n,h,w,v);
        if(p.usable){
            float one=1,zero=0;
            cudnnStatus_t status=cudnnConvolutionForward(d.cudnn,&one,p.x,in,p.weight,v.value.data,p.conv,p.algorithm,
                d.scratch.data,p.workspace,&zero,p.y,out);
            if(status!=CUDNN_STATUS_SUCCESS)throw std::runtime_error(std::string("tiny cuDNN convolution: ")+cudnnGetErrorString(status));
            d.cudnn_calls++;used=true;
        }else if(d.policy=="cudnn")throw std::runtime_error("tiny cuDNN: no supported convolution within workspace limit");
    }
#endif
    size_t pixels=(size_t)n*h*w;int inner=v.in*v.k*v.k;
    if(!used){
        size_t panel=v.k==1?pixels:std::min(pixels,d.scratch.bytes/((size_t)inner*sizeof(half)));
        if(!panel)throw std::runtime_error("tiny CUDA convolution scratch is too small");
        float one=1,zero=0;
        for(size_t first=0;first<pixels;first+=panel){
            size_t count=std::min(panel,pixels-first);const half *patch=in+first*(size_t)v.in;
            if(v.k!=1){patch=d.scratch.fp16();im2col<<<blocks(count*(size_t)inner),256,0,d.stream>>>(d.scratch.fp16(),in,count,first,h,w,v.in,v.k);}
            blas_check(cublasGemmEx(d.blas,CUBLAS_OP_T,CUBLAS_OP_N,v.out,(int)count,inner,&one,
                v.value.data,CUDA_R_16F,inner,patch,CUDA_R_16F,inner,&zero,out+first*(size_t)v.out,
                CUDA_R_16F,v.out,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP),"convolution");d.blas_calls++;
        }
    }
    auto bias=d.weights.find(name+".bias");const half *b=bias==d.weights.end()?nullptr:bias->second->value.fp16();
    if(b||relu||residual)activation<<<blocks(pixels*(size_t)v.out),256,0,d.stream>>>(out,b,residual,pixels*(size_t)v.out,v.out,relu);
    check(cudaGetLastError(),"convolution/activation launch");
}
void prepare(decoder &d,int t,int h,int w){
    size_t hw=(size_t)h*w;
    for(auto &b:d.activation)d.reserve(b,(size_t)t*hw*16384*sizeof(half));
    d.reserve(d.input,(size_t)t*hw*24*sizeof(float));d.reserve(d.rgb,(size_t)t*hw*3072*sizeof(float));
    d.reserve(d.scratch,d.workspace_limit);
    if(d.rgb.bytes>d.pinned_bytes){
        check(cudaStreamSynchronize(d.stream),"readback reuse");if(d.readback){check(cudaFreeHost(d.readback),"readback release");d.readback=nullptr;d.pinned_bytes=0;}
        check(cudaHostAlloc(&d.readback,d.rgb.bytes,cudaHostAllocDefault),"pinned readback allocation");d.pinned_bytes=d.rgb.bytes;
    }
    const int c[]={256,128,64};
    for(int stage=0;stage<3;stage++)for(int j=0;j<3;j++)d.reserve(d.history[stage*3+j],hw*(size_t)(1<<(stage*2))*(size_t)c[stage]*sizeof(half));
    if(d.history_h!=h||d.history_w!=w){
        for(auto &b:d.history)check(cudaMemsetAsync(b.data,0,b.bytes,d.stream),"history initialization");
        d.history_h=h;d.history_w=w;
    }
}
}

extern "C" void *h3_tiny_backend_load(const char *path,char *error,size_t size){
    h3_st_header header={0};std::unique_ptr<decoder> d;
    try{
        h3_device_info info;if(!h3_device_query(&info,error,size))return nullptr;
        d=std::make_unique<decoder>();d->device=info.device_index;
        const char *policy=getenv("H3_PREVIEW_CUDA_CONV");if(policy)d->policy=policy;
        if(d->policy!="auto"&&d->policy!="cublas"&&d->policy!="cudnn")throw std::runtime_error("H3_PREVIEW_CUDA_CONV must be auto, cublas or cudnn");
        const char *workspace=getenv("H3_PREVIEW_CUDA_WORKSPACE_MB");
        if(workspace){char *end=nullptr;unsigned long mb=strtoul(workspace,&end,10);
            if(!*workspace||*end||mb<1||mb>128)throw std::runtime_error("H3_PREVIEW_CUDA_WORKSPACE_MB must be 1..128");d->workspace_limit=(size_t)mb<<20;}
        check(cudaStreamCreateWithFlags(&d->stream,cudaStreamNonBlocking),"create stream");
        blas_check(cublasCreate(&d->blas),"create handle");blas_check(cublasSetStream(d->blas,d->stream),"bind stream");
        blas_check(cublasSetMathMode(d->blas,CUBLAS_DEFAULT_MATH),"FP32 accumulation");
#ifdef H3_CUDA_USE_CUDNN
        if(d->policy!="cublas"){
            if(cudnnCreate(&d->cudnn)!=CUDNN_STATUS_SUCCESS)d->cudnn=nullptr;
            if(d->cudnn&&cudnnSetStream(d->cudnn,d->stream)!=CUDNN_STATUS_SUCCESS){cudnnDestroy(d->cudnn);d->cudnn=nullptr;}
            if(!d->cudnn&&d->policy=="cudnn")throw std::runtime_error("tiny cuDNN initialization failed");
        }
#else
        if(d->policy=="cudnn")throw std::runtime_error("tiny cuDNN requested but this build has no cuDNN; use auto or cublas");
#endif
        if(!h3_st_read_header(path,&header,error,size))return nullptr;
        for(size_t i=0;i<header.tensor_count;i++){
            const h3_st_tensor &v=header.tensors[i];if(strncmp(v.name,"decoder.",8))continue;
            size_t bytes=(size_t)(v.data_end-v.data_begin);std::vector<uint16_t> raw(bytes/2),packed(bytes/2);
            if(!h3_st_read_data(&header,&v,raw.data(),bytes,error,size)){h3_st_free_header(&header);return nullptr;}
            auto weight=std::make_unique<struct weight>();
            if(v.ndim==4){weight->out=(int)v.shape[0];weight->in=(int)v.shape[1];weight->k=(int)v.shape[2];
                for(int o=0;o<weight->out;o++)for(int y=0;y<weight->k;y++)for(int x=0;x<weight->k;x++)for(int c=0;c<weight->in;c++)
                    packed[((size_t)o*weight->k*weight->k+(size_t)y*weight->k+(size_t)x)*weight->in+(size_t)c]=
                        raw[((size_t)o*weight->in+(size_t)c)*weight->k*weight->k+(size_t)y*weight->k+(size_t)x];
            }else packed=std::move(raw);
            d->reserve(weight->value,bytes);check(cudaMemcpyAsync(weight->value.data,packed.data(),bytes,cudaMemcpyHostToDevice,d->stream),"load weights");
            check(cudaStreamSynchronize(d->stream),"weight upload lifetime");d->weights.emplace(v.name,std::move(weight));
        }
        h3_st_free_header(&header);
        fprintf(stderr,"h3cli: tiny CUDA device=%d SM%d%d FP16/FP32-accumulate conv=%s workspace_limit=%zu\n",d->device,info.cuda_compute_major,info.cuda_compute_minor,
#ifdef H3_CUDA_USE_CUDNN
            d->cudnn?(d->policy=="cudnn"?"cuDNN required":"cuDNN (cuBLAS fallback)"):"cuBLAS",
#else
            "cuBLAS",
#endif
            d->workspace_limit);
#ifdef H3_CUDA_USE_CUDNN
        if(d->cudnn)fprintf(stderr,"h3cli: tiny cuDNN runtime=%zu\n",cudnnGetVersion());
#endif
        return d.release();
    }catch(const std::exception &e){h3_st_free_header(&header);if(error&&size)snprintf(error,size,"%s",e.what());return nullptr;}
}
extern "C" void h3_tiny_backend_free(void *opaque){
    auto *d=static_cast<decoder*>(opaque);if(!d)return;
    int previous=-1;cudaGetDevice(&previous);delete d;if(previous>=0)cudaSetDevice(previous);
}
extern "C" void h3_tiny_backend_reset(void *opaque){
    auto *d=static_cast<decoder*>(opaque);if(!d)return;
    // Chunk return (including failure) drains the stream before callbacks/reset.
    d->history_h=d->history_w=0;d->compute_seconds=d->transfer_seconds=0;
    d->blas_calls=d->cudnn_calls=d->plan_builds=0;d->peak=d->allocated;
}
extern "C" void h3_tiny_backend_profile(void *opaque){
    auto *d=static_cast<decoder*>(opaque);if(!d||!getenv("H3_PROFILE"))return;
    size_t plans=0;
#ifdef H3_CUDA_USE_CUDNN
    plans=d->plans.size();
#endif
    fprintf(stderr,"h3cli: tiny CUDA FP16: compute=%.6f transfer=%.6f s device_owned_peak=%zu pinned=%zu scratch=%zu plans=%zu built=%llu cublas=%llu cudnn=%llu\n",
        d->compute_seconds,d->transfer_seconds,d->peak,d->pinned_bytes,d->scratch.bytes,plans,
        (unsigned long long)d->plan_builds,(unsigned long long)d->blas_calls,(unsigned long long)d->cudnn_calls);
}
extern "C" int h3_tiny_backend_chunk(void *opaque,const float *input,int t,int h,int w,float **output,char *error,size_t size){
    auto &d=*static_cast<decoder*>(opaque);*output=nullptr;
    try{
        device_scope scope(d.device);prepare(d,t,h,w);double begin=now();
        size_t input_count=(size_t)t*h*w*24;
        check(cudaMemcpyAsync(d.input.data,input,input_count*sizeof(float),cudaMemcpyHostToDevice,d.stream),"input upload");
        half *x=d.activation[0].fp16(),*a=d.activation[1].fp16(),*b=d.activation[2].fp16();
        transform_input<<<blocks(input_count),256,0,d.stream>>>(x,static_cast<float*>(d.input.data),input_count);
        convolution(d,x,a,t,h,w,"decoder.1",true);std::swap(x,a);
        int n=t,hh=h,ww=w;const int ids[3][3]={{3,4,5},{9,10,11},{15,16,17}},channels[]={256,128,64};
        for(int stage=0;stage<3;stage++){
            int c=channels[stage];size_t frame=(size_t)hh*ww*c;
            for(int j=0;j<3;j++){
                auto &mem=d.history[stage*3+j];
                assemble_memory<<<blocks((size_t)n*frame*2),256,0,d.stream>>>(a,x,mem.fp16(),frame,(size_t)n*frame*2,c);
                check(cudaMemcpyAsync(mem.data,x+(size_t)(n-1)*frame,frame*sizeof(half),cudaMemcpyDeviceToDevice,d.stream),"retain temporal history");
                std::string name="decoder."+std::to_string(ids[stage][j])+".conv.";
                convolution(d,a,b,n,hh,ww,name+"0",true);convolution(d,b,a,n,hh,ww,name+"2",true);
                convolution(d,a,b,n,hh,ww,name+"4",true,x);std::swap(x,b);
            }
            upsample<<<blocks((size_t)n*frame*4),256,0,d.stream>>>(a,x,(size_t)n*frame*4,hh,ww,c);hh*=2;ww*=2;
            int grow=stage==0?7:stage==1?13:19;
            convolution(d,a,b,n,hh,ww,"decoder."+std::to_string(grow)+".conv");
            if(stage){size_t count=(size_t)n*hh*ww*c*2;
                expand_time<<<blocks(count),256,0,d.stream>>>(a,b,count,(size_t)hh*ww,c);n*=2;
                convolution(d,a,x,n,hh,ww,"decoder."+std::to_string(grow+1));
            }else convolution(d,b,x,n,hh,ww,"decoder.8");
        }
        size_t elements=(size_t)n*hh*ww*64;
        activation<<<blocks(elements),256,0,d.stream>>>(x,nullptr,nullptr,elements,64,true);
        convolution(d,x,a,n,hh,ww,"decoder.22");
        size_t rgb_count=(size_t)n*hh*ww*12;
        pixel_shuffle<<<blocks(rgb_count),256,0,d.stream>>>(static_cast<float*>(d.rgb.data),a,rgb_count,hh,ww);
        check(cudaGetLastError(),"decode launch");check(cudaStreamSynchronize(d.stream),"decode completion");
        d.compute_seconds+=now()-begin;begin=now();
        check(cudaMemcpyAsync(d.readback,d.rgb.data,rgb_count*sizeof(float),cudaMemcpyDeviceToHost,d.stream),"RGB transfer");
        check(cudaStreamSynchronize(d.stream),"RGB lifetime");d.transfer_seconds+=now()-begin;
        float *out=static_cast<float*>(malloc(rgb_count*sizeof(float)));
        if(!out)throw std::runtime_error("cannot allocate tiny CUDA RGB batch");
        memcpy(out,d.readback,rgb_count*sizeof(float));*output=out;return 1;
    }catch(const std::exception &e){
        int previous=-1;cudaGetDevice(&previous);cudaSetDevice(d.device);cudaStreamSynchronize(d.stream);
        // A handled allocation/launch failure must not poison the next request's
        // launch-status check. Fatal context errors still fail CUDA operations.
        cudaGetLastError();if(previous>=0)cudaSetDevice(previous);
        d.history_h=d.history_w=0;if(error&&size)snprintf(error,size,"%s",e.what());return 0;
    }
}
