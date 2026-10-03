/* Reference-only convolution adapters. Audio plans are bounded to 256 and
 * video encoder plans to 128; workspace is at most 512 MiB for audio and
 * 2 GiB for temporal video encoding. Each layout
 * adapter is at most 64 MiB for audio and 1 GiB for the 17-frame encoder.
 * Buffers belong to the component GPU context and its allocation counters. */
static int sg_audio_scratch(h3_gpu *g,h3_gpu_tensor **slot,size_t n,size_t limit) {
    if(!n)n=1;
    if(n>limit/sizeof(float))return h3_gpu_set_error(g,"reference audio scratch limit exceeded");
    if(*slot&&(*slot)->elements>=n)return 1;
    if(*slot&&!h3_gpu_synchronize(g))return 0;
    h3_gpu_tensor_free(*slot);*slot=h3_gpu_tensor_new_f32(g,n);
    return *slot!=nullptr;
}
__global__ static void sg_audio_to_cf(float *out,const float *in,unsigned length,unsigned channels,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned t=i%length,c=(i/length)%channels;size_t b=i/length/channels;
    out[i]=in[(b*length+t)*channels+c];
}
__global__ static void sg_audio_from_cf(float *out,const float *in,const float *bias,unsigned length,unsigned channels,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned c=i%channels,t=(i/channels)%length;size_t b=i/channels/length;
    float value=in[(b*channels+c)*length+t];out[i]=bias?value+bias[c]:value;
}
static int sg_audio_conv(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *w,const h3_gpu_tensor *bias,unsigned batch,unsigned length,
    unsigned ci,unsigned co,unsigned kernel,unsigned stride,unsigned padding,unsigned dilation,
    unsigned output_length,bool transpose) {
    if(out->dtype!=H3_GPU_F32||in->dtype!=H3_GPU_F32||w->dtype!=H3_GPU_F32||(bias&&bias->dtype!=H3_GPU_F32))
        return h3_gpu_set_error(g,"reference audio convolution requires FP32");
    if(!g->sglang_conv_cudnn)g->sglang_conv_cudnn=h3_cudnn_reference_create(g->compute);
    if(!g->sglang_conv_cudnn)return h3_gpu_set_error(g,"reference audio requires CUDA_CUDNN=1 and cuDNN 9.20");
    size_t bytes=0,ni=(size_t)batch*length*ci,no=(size_t)batch*output_length*co;
    if(!h3_cudnn_reference_conv(g->sglang_conv_cudnn,nullptr,nullptr,nullptr,batch,length,ci,co,kernel,stride,padding,dilation,transpose,
        nullptr,0,&bytes,g->error,sizeof(g->error))||
       !sg_audio_scratch(g,&g->sglang_conv_input,ni,64u<<20)||
       !sg_audio_scratch(g,&g->sglang_conv_output,no,64u<<20)||
       !sg_audio_scratch(g,&g->sglang_conv_workspace,(bytes+3)/4,512u<<20))return 0;
    float *x=(float*)tensor_pointer(g->sglang_conv_input),*y=(float*)tensor_pointer(g->sglang_conv_output);
    sg_audio_to_cf<<<(ni+255)/256,256,0,g->compute>>>(x,(const float*)tensor_pointer(in),length,ci,ni);
    if(!launch_status(g,"reference audio input layout")||!h3_cudnn_reference_conv(g->sglang_conv_cudnn,y,x,(const float*)tensor_pointer(w),
        batch,length,ci,co,kernel,stride,padding,dilation,transpose,tensor_pointer(g->sglang_conv_workspace),
        g->sglang_conv_workspace->bytes,&bytes,g->error,sizeof(g->error)))return 0;
    sg_audio_from_cf<<<(no+255)/256,256,0,g->compute>>>((float*)tensor_pointer(out),y,(const float*)tensor_pointer(bias),output_length,co,no);
    g->stats.conv_dispatches++;return launch_status(g,"reference audio cuDNN convolution");
}
/* Pinned SGLang disables cuDNN for soundtrack encoding. PyTorch's fallback
 * uses [C*K,T] columns and one NN SGEMM per stereo channel, with the bias
 * copied into C before beta=1 accumulation. Decoder cuDNN stays separate. */
__global__ static void sg_audio_encoder_columns(float *out,const float *in,
    unsigned length,unsigned ci,unsigned kernel,unsigned stride,unsigned padding,
    unsigned dilation,unsigned output_length,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned t=i%output_length,k=(i/output_length)%kernel,c=i/output_length/kernel;
    int64_t source=(int64_t)t*stride+(int64_t)k*dilation-padding;
    out[i]=source>=0&&source<length?in[(size_t)source*ci+c]:0.f;
}
__global__ static void sg_audio_encoder_bias(float *out,const float *bias,
    unsigned length,unsigned channels,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count)out[i]=bias?bias[(i/length)%channels]:0.f;
}
static int sg_audio_encoder_conv(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *w,const h3_gpu_tensor *bias,unsigned batch,unsigned length,
    unsigned ci,unsigned co,unsigned kernel,unsigned stride,unsigned padding,
    unsigned dilation,unsigned output_length,bool transpose) {
    if(transpose||out->dtype!=H3_GPU_F32||in->dtype!=H3_GPU_F32||w->dtype!=H3_GPU_F32||
       (bias&&bias->dtype!=H3_GPU_F32)||!g->sglang_blas)
        return h3_gpu_set_error(g,"reference audio encoder requires ordinary FP32 convolution");
    size_t columns=(size_t)ci*kernel*output_length,no=(size_t)batch*output_length*co;
    if(!sg_audio_scratch(g,&g->sglang_conv_workspace,columns,256u<<20)||
       !sg_audio_scratch(g,&g->sglang_conv_output,no,64u<<20))return 0;
    auto *x=(float*)tensor_pointer(g->sglang_conv_workspace);
    auto *y=(float*)tensor_pointer(g->sglang_conv_output);
    const auto *input=(const float*)tensor_pointer(in),*weight=(const float*)tensor_pointer(w);
    sg_audio_encoder_bias<<<(no+255)/256,256,0,g->compute>>>(y,(const float*)tensor_pointer(bias),output_length,co,no);
    float one=1.f;
    for(unsigned b=0;b<batch;b++) {
        sg_audio_encoder_columns<<<(columns+255)/256,256,0,g->compute>>>(x,input+(size_t)b*length*ci,
            length,ci,kernel,stride,padding,dilation,output_length,columns);
        auto status=g->sglang_blas->sgemm(g->sglang_blas->handle,CUBLAS_OP_N,CUBLAS_OP_N,
            output_length,co,ci*kernel,&one,x,output_length,weight,ci*kernel,&one,
            y+(size_t)b*output_length*co,output_length);
        if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,"reference audio encoder SGEMM failed: %d",(int)status);
    }
    sg_audio_from_cf<<<(no+255)/256,256,0,g->compute>>>((float*)tensor_pointer(out),y,nullptr,output_length,co,no);
    g->stats.conv_dispatches++;return launch_status(g,"reference audio encoder fallback convolution");
}
static int sg_encoder_conv(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *w,const h3_gpu_tensor *bias,const unsigned shape[12],unsigned positions) {
    if(!g->sglang_conv_cudnn)g->sglang_conv_cudnn=h3_cudnn_reference_create(g->compute);
    if(!g->sglang_conv_cudnn)return h3_gpu_set_error(g,"reference video encoder requires CUDA_CUDNN=1 and cuDNN 9.20");
    unsigned batch=shape[0],spatial=shape[1]*shape[2]*shape[3],ci=shape[4],co=shape[5];
    size_t bytes=0,ni=(size_t)batch*spatial*ci,no=(size_t)batch*positions*co;
    if(!h3_cudnn_reference_conv3d(g->sglang_conv_cudnn,nullptr,nullptr,nullptr,shape,nullptr,0,&bytes,g->error,sizeof(g->error))||
       !sg_audio_scratch(g,&g->sglang_conv_input,ni,1024u<<20)||
       !sg_audio_scratch(g,&g->sglang_conv_output,no,1024u<<20)||
       !sg_audio_scratch(g,&g->sglang_conv_workspace,(bytes+3)/4,2ull<<30))return 0;
    float *x=(float*)tensor_pointer(g->sglang_conv_input),*y=(float*)tensor_pointer(g->sglang_conv_output);
    sg_audio_to_cf<<<(ni+255)/256,256,0,g->compute>>>(x,(const float*)tensor_pointer(in),spatial,ci,ni);
    if(!launch_status(g,"reference encoder input layout")||!h3_cudnn_reference_conv3d(g->sglang_conv_cudnn,y,x,(const float*)tensor_pointer(w),shape,
        tensor_pointer(g->sglang_conv_workspace),g->sglang_conv_workspace->bytes,&bytes,g->error,sizeof(g->error)))return 0;
    sg_audio_from_cf<<<(no+255)/256,256,0,g->compute>>>((float*)tensor_pointer(out),y,(const float*)tensor_pointer(bias),positions,co,no);
    g->stats.conv_dispatches++;return launch_status(g,"reference encoder cuDNN convolution");
}
__global__ static void sg_audio_average(float *out,const float *last,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i<count)out[i]=(out[i]+last[i])*(1.f/3.f);
}
int h3_gpu_sglang_audio_average(h3_gpu *g,h3_gpu_tensor *sum,const h3_gpu_tensor *last,uint32_t count) {
    if(!g)return 0;
    if(!g->sglang_audio)return h3_gpu_add_scaled_f32(g,sum,sum,last,1.f/3.f,1.f/3.f,count);
    if(!launch_ready(g)||!count||!h3_gpu_require_elements(g,sum,count,"audio average")||
       !h3_gpu_require_elements(g,last,count,"audio average input")||sum->dtype!=H3_GPU_F32||last->dtype!=H3_GPU_F32)return 0;
    sg_audio_average<<<(count+255)/256,256,0,g->compute>>>((float*)tensor_pointer(sum),(const float*)tensor_pointer(last),count);
    return launch_status(g,"reference audio sum then average");
}
