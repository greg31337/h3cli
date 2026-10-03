/* Native cuDNN convolution plans for the shared SGLang pipeline. */
#include "src/cuda/cuda_cudnn.h"
#include <cstdint>
#include <cstring>
// cuDNN frontend v1.11 uses the CUDA 12 graph query signature. CUDA 13
// requires an explicit edge-data argument; retain the frontend's default edges.
#if CUDART_VERSION >= 13000
inline cudaError_t cudaGraphNodeGetDependentNodes(cudaGraphNode_t node,
                                                cudaGraphNode_t *nodes,
                                                size_t *count) {
    return cudaGraphNodeGetDependentNodes(node, nodes, nullptr, count);
}
#endif
#include <cudnn_frontend.h>
#include <map>
#include <memory>
#include <tuple>
#include <unordered_map>
#include <cstdio>
#include <array>
namespace fe=cudnn_frontend;
// Shared audio/video encoder convolution plans.
using reference_conv_key=std::tuple<unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,unsigned,bool>;
struct reference_conv_plan {
    cudnnTensorDescriptor_t x=nullptr,y=nullptr;
    cudnnFilterDescriptor_t w=nullptr;
    cudnnConvolutionDescriptor_t conv=nullptr;
    int algorithm=1;size_t bytes=0;
    std::shared_ptr<fe::graph::Graph> graph;
    ~reference_conv_plan(){if(x)cudnnDestroyTensorDescriptor(x);if(y)cudnnDestroyTensorDescriptor(y);
        if(w)cudnnDestroyFilterDescriptor(w);if(conv)cudnnDestroyConvolutionDescriptor(conv);}
};
struct reference_context {
    cudnnHandle_t handle=nullptr;cudaStream_t stream=nullptr;
    std::map<reference_conv_key,std::unique_ptr<reference_conv_plan>> plans;
    std::map<std::array<unsigned,12>,std::unique_ptr<reference_conv_plan>> conv3d;
};
void *h3_cudnn_reference_create(cudaStream_t stream) {
    if(cudnnGetVersion()!=92000)return nullptr;
    auto *c=new(std::nothrow)reference_context;if(!c)return nullptr;
    if(cudnnCreate(&c->handle)!=CUDNN_STATUS_SUCCESS){delete c;return nullptr;}
    if(cudnnSetStream(c->handle,stream)!=CUDNN_STATUS_SUCCESS){cudnnDestroy(c->handle);delete c;return nullptr;}
    c->stream=stream;return c;
}
void h3_cudnn_reference_free(void *opaque) {
    auto *c=(reference_context*)opaque;if(!c)return;
    cudaStreamSynchronize(c->stream);c->plans.clear();c->conv3d.clear();cudnnDestroy(c->handle);delete c;
}
int h3_cudnn_reference_conv(void *opaque,float *out,const float *in,const float *weight,
    unsigned batch,unsigned length,unsigned ci,unsigned co,unsigned kernel,
    unsigned stride,unsigned padding,unsigned dilation,bool transpose,
    void *workspace,size_t capacity,size_t *required,char *error,size_t size) {
    auto *c=(reference_context*)opaque;if(!c||!required)return 0;
    auto fail=[&](cudnnStatus_t s){snprintf(error,size,"reference audio cuDNN: %s",cudnnGetErrorString(s));return 0;};
    reference_conv_key key={batch,length,ci,co,kernel,stride,padding,dilation,transpose};
    auto found=c->plans.find(key);
    if(found==c->plans.end()){
        if(c->plans.size()>=256){snprintf(error,size,"reference audio plan limit (256) exceeded");return 0;}
        int64_t output=transpose?(int64_t)(length-1)*stride-2*(int64_t)padding+kernel:
            ((int64_t)length+2*(int64_t)padding-(int64_t)dilation*(kernel-1)-1)/stride+1;
        if(output<=0||output>INT32_MAX)return 0;
        auto p=std::make_unique<reference_conv_plan>();cudnnStatus_t s;
        if(transpose&&ci==256&&co==128&&kernel==4&&stride==2&&padding==1&&dilation==1){
            // cuDNN's legacy entry points choose engine 2 at this shape,
            // whereas the pinned oracle uses engine 1. Engine identity was
            // qualified on the retained real operation, without autotuning.
            auto graph=std::make_shared<fe::graph::Graph>();
            graph->set_io_data_type(fe::DataType_t::FLOAT).set_intermediate_data_type(fe::DataType_t::FLOAT).set_compute_data_type(fe::DataType_t::FLOAT);
            int64_t N=batch,I=ci,O=co,L=length,K=kernel,T=output;
            auto X=graph->tensor(fe::graph::Tensor_attributes().set_uid(1).set_dim({N,I,L,1}).set_stride({I*L,L,1,1}));
            auto W=graph->tensor(fe::graph::Tensor_attributes().set_uid(2).set_dim({I,O,K,1}).set_stride({O*K,K,1,1}));
            auto Y=graph->conv_dgrad(X,W,fe::graph::Conv_dgrad_attributes().set_padding({1,0}).set_stride({2,1}).set_dilation({1,1}));
            Y->set_output(true).set_uid(3).set_dim({N,O,T,1}).set_stride({O*T,T,1,1});
            int64_t bytes=0;
            if(!graph->validate().is_good()||!graph->build_operation_graph(c->handle).is_good()||
               !graph->create_execution_plan(1,{}).is_good()||!graph->build_plan_at_index(c->handle,0).is_good()||
               !graph->get_workspace_size_plan_at_index(0,bytes).is_good()||bytes<0||bytes>(512u<<20)){
                snprintf(error,size,"reference audio qualified dgrad engine 1 unavailable");return 0;
            }
            p->graph=graph;p->bytes=(size_t)bytes;
        }else{
#define REF_DN(call) do{s=(call);if(s!=CUDNN_STATUS_SUCCESS)return fail(s);}while(0)
        REF_DN(cudnnCreateTensorDescriptor(&p->x));REF_DN(cudnnCreateTensorDescriptor(&p->y));
        REF_DN(cudnnCreateFilterDescriptor(&p->w));REF_DN(cudnnCreateConvolutionDescriptor(&p->conv));
        REF_DN(cudnnSetTensor4dDescriptor(p->x,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batch,ci,length,1));
        REF_DN(cudnnSetTensor4dDescriptor(p->y,CUDNN_TENSOR_NCHW,CUDNN_DATA_FLOAT,batch,co,(int)output,1));
        REF_DN(cudnnSetFilter4dDescriptor(p->w,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,transpose?ci:co,transpose?co:ci,kernel,1));
        REF_DN(cudnnSetConvolution2dDescriptor(p->conv,padding,0,stride,1,dilation,1,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
        REF_DN(cudnnSetConvolutionMathType(p->conv,CUDNN_FMA_MATH));
        // Real-operation comparisons against pinned SGLang, with TF32 disabled.
        // The 2048->1024 pre-convolution uses FFT tiling; ordinary convs use
        // implicit precomputed GEMM. No timing-based algorithm selection.
        p->algorithm=(!transpose&&stride==1&&dilation==1&&
            ((ci==2048&&co==1024&&kernel==7)||(ci==512&&co==512&&kernel>=3)))?5:1;
        if(transpose)REF_DN(cudnnGetConvolutionBackwardDataWorkspaceSize(c->handle,p->w,p->x,p->conv,p->y,(cudnnConvolutionBwdDataAlgo_t)p->algorithm,&p->bytes));
        else REF_DN(cudnnGetConvolutionForwardWorkspaceSize(c->handle,p->x,p->w,p->conv,p->y,(cudnnConvolutionFwdAlgo_t)p->algorithm,&p->bytes));
        if(p->bytes>(512u<<20)){snprintf(error,size,"reference audio cuDNN workspace exceeds 512 MiB");return 0;}
        }
        found=c->plans.emplace(key,std::move(p)).first;
    }
    auto &p=*found->second;*required=p.bytes;
    if(!out)return 1;
    if(!in||!weight||capacity<p.bytes||(p.bytes&&!workspace))return 0;
    if(p.graph){
        std::unordered_map<int64_t,void*> pointers={{1,const_cast<float*>(in)},{2,const_cast<float*>(weight)},{3,out}};
        auto result=p.graph->execute_plan_at_index(c->handle,pointers,workspace,0);
        if(!result.is_good()){snprintf(error,size,"reference audio dgrad execution failed: %s",result.get_message().c_str());return 0;}
        return 1;
    }
    float alpha=1,beta=0;cudnnStatus_t s;
    if(transpose)s=cudnnConvolutionBackwardData(c->handle,&alpha,p.w,weight,p.x,in,p.conv,(cudnnConvolutionBwdDataAlgo_t)p.algorithm,workspace,capacity,&beta,p.y,out);
    else s=cudnnConvolutionForward(c->handle,&alpha,p.x,in,p.w,weight,p.conv,(cudnnConvolutionFwdAlgo_t)p.algorithm,workspace,capacity,&beta,p.y,out);
#undef REF_DN
    return s==CUDNN_STATUS_SUCCESS?1:fail(s);
}

int h3_cudnn_reference_conv3d(void *opaque,float *out,const float *in,const float *weight,
    const unsigned shape[12],void *workspace,size_t capacity,size_t *required,char *error,size_t size) {
    auto *c=(reference_context*)opaque;if(!c||!shape||!required)return 0;
    std::array<unsigned,12> key;std::copy(shape,shape+12,key.begin());
    for(unsigned value:key)if(!value||value>INT32_MAX)return 0;
    auto fail=[&](cudnnStatus_t s){snprintf(error,size,"reference video encoder cuDNN: %s",cudnnGetErrorString(s));return 0;};
    auto found=c->conv3d.find(key);
    if(found==c->conv3d.end()){
        if(c->conv3d.size()>=128){snprintf(error,size,"reference encoder plan limit (128) exceeded");return 0;}
        int N=shape[0],D=shape[1],H=shape[2],W=shape[3],I=shape[4],O=shape[5];
        int kd=shape[6],kh=shape[7],kw=shape[8],sd=shape[9],sh=shape[10],sw=shape[11];
        if(D<kd||H<kh||W<kw)return 0;
        int od=(D-kd)/sd+1,oh=(H-kh)/sh+1,ow=(W-kw)/sw+1;
        if((int64_t)I*D*H*W>INT32_MAX||(int64_t)O*od*oh*ow>INT32_MAX)return 0;
        auto p=std::make_unique<reference_conv_plan>();cudnnStatus_t s;
#define REF_DN(call) do{s=(call);if(s!=CUDNN_STATUS_SUCCESS)return fail(s);}while(0)
        REF_DN(cudnnCreateTensorDescriptor(&p->x));REF_DN(cudnnCreateTensorDescriptor(&p->y));
        REF_DN(cudnnCreateFilterDescriptor(&p->w));REF_DN(cudnnCreateConvolutionDescriptor(&p->conv));
        int xs[5]={N,I,D,H,W},xt[5]={I*D*H*W,D*H*W,H*W,W,1};
        int ys[5]={N,O,od,oh,ow},yt[5]={O*od*oh*ow,od*oh*ow,oh*ow,ow,1};
        int ws[5]={O,I,kd,kh,kw},pad[3]={0,0,0},step[3]={sd,sh,sw},dilation[3]={1,1,1};
        REF_DN(cudnnSetTensorNdDescriptor(p->x,CUDNN_DATA_FLOAT,5,xs,xt));
        REF_DN(cudnnSetTensorNdDescriptor(p->y,CUDNN_DATA_FLOAT,5,ys,yt));
        REF_DN(cudnnSetFilterNdDescriptor(p->w,CUDNN_DATA_FLOAT,CUDNN_TENSOR_NCHW,5,ws));
        REF_DN(cudnnSetConvolutionNdDescriptor(p->conv,3,pad,step,dilation,CUDNN_CROSS_CORRELATION,CUDNN_DATA_FLOAT));
        // Unlike the audio decoder, the pinned image/video encoder permits
        // TF32 convolution. The original FP32 checkpoint is still stored as-is.
        REF_DN(cudnnSetConvolutionMathType(p->conv,CUDNN_DEFAULT_MATH));
        p->algorithm=I==3?0:1;
        REF_DN(cudnnGetConvolutionForwardWorkspaceSize(c->handle,p->x,p->w,p->conv,p->y,(cudnnConvolutionFwdAlgo_t)p->algorithm,&p->bytes));
        /* A 17-frame temporal tile needs a larger workspace than the image
         * tile used during initial qualification. Keep an explicit bound. */
        if(p->bytes>(2ull<<30)){
            snprintf(error,size,"reference encoder workspace needs %zu bytes (limit 2 GiB), shape %dx%dx%dx%d channels %d->%d kernel %dx%dx%d",
                p->bytes,N,D,H,W,I,O,kd,kh,kw);return 0;
        }
        found=c->conv3d.emplace(key,std::move(p)).first;
#undef REF_DN
    }
    auto &p=*found->second;*required=p.bytes;if(!out)return 1;
    if(!in||!weight||capacity<p.bytes||(p.bytes&&!workspace))return 0;
    float alpha=1,beta=0;
    auto s=cudnnConvolutionForward(c->handle,&alpha,p.x,in,p.w,weight,p.conv,(cudnnConvolutionFwdAlgo_t)p.algorithm,workspace,capacity,&beta,p.y,out);
    return s==CUDNN_STATUS_SUCCESS?1:fail(s);
}
