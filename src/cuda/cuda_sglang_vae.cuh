/* Pinned SGLang full decoder. FP16 exists only at the audited block
 * boundaries; residuals, norms, embeddings and delivery remain FP32.
 * Scratch is decoder-owned, reused, and bounded to 256 MiB per tensor. */
static int sg_vae_scratch(h3_gpu *g,h3_gpu_tensor **slot,size_t n,h3_gpu_dtype dtype) {
    if(n>(256u<<20)/item_size(dtype))return h3_gpu_set_error(g,"reference VAE scratch limit exceeded");
    if(*slot && (*slot)->elements>=n && (*slot)->dtype==dtype)return 1;
    if(*slot && !h3_gpu_synchronize(g))return 0;
    return vae_scratch(g,slot,n,dtype);
}
__global__ static void sg_vae_cast_half(__half *out,const float *in,size_t n,unsigned *fault) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){out[i]=__float2half_rn(in[i]);if(!isfinite(in[i])||!isfinite(__half2float(out[i])))atomicOr(fault,1u);}
}
__global__ static void sg_vae_cast_float(float *out,const __half *in,const __half *bias,size_t n,unsigned width,unsigned *fault) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i<n){float x=__half2float(in[i]);out[i]=bias?h3_sg_half(x+__half2float(bias[i%width])):x;if(!isfinite(out[i]))atomicOr(fault,1u);}
}
/* Same range check and rounding as three separate reference conversions. */
__global__ static void exact_vae_cast_qkv(__half *q,__half *k,__half *v,
    const float *iq,const float *ik,const float *iv,size_t n,unsigned *fault) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    float a=iq[i],b=ik[i],c=iv[i];
    __half x=__float2half_rn(a),y=__float2half_rn(b),z=__float2half_rn(c);
    q[i]=x;k[i]=y;v[i]=z;
    if(!isfinite(a)||!isfinite(b)||!isfinite(c)||!isfinite(__half2float(x))||
       !isfinite(__half2float(y))||!isfinite(__half2float(z)))atomicOr(fault,1u);
}
int h3_gpu_sglang_vae_enabled(const h3_gpu *g) {return g&&g->sglang_vae;}
__global__ static void sg_vae_round_float(float *out,size_t n,unsigned *fault) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i<n){out[i]=h3_sg_half(out[i]);if(!isfinite(out[i]))atomicOr(fault,1u);}
}
__global__ static void sg_vae_rope(float *cosine,float *sine,unsigned time,unsigned height,unsigned width,unsigned rows) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*24)return;
    unsigned row=i/24,axis=(i%24)/8,freq=i%8;
    float c=1,s=0;
    if(row<time*height*width){
        unsigned coordinate=axis==0?row/(height*width):axis==1?(row/width)%height:row%width;
        unsigned size=axis==0?time:axis==1?height:width;
        float position=h3_sg_half(h3_sg_half(2.f*h3_sg_half(((float)coordinate+.5f)/(float)size))-1.f);
        float angle=h3_sg_half(6.28318530717958647692f*position)*(1.f/powf(100.f,(float)freq*.125f));
        c=h3_sg_half(cosf(angle));s=h3_sg_half(sinf(angle));
    }
    cosine[i]=c;sine[i]=s;
}
int h3_gpu_sglang_vae_rope(h3_gpu *g,h3_gpu_tensor *cosine,h3_gpu_tensor *sine,
    uint32_t time,uint32_t height,uint32_t width,uint32_t rows) {
    if(!g||!g->sglang_vae||!cosine||!sine||cosine->dtype!=H3_GPU_F32||sine->dtype!=H3_GPU_F32||
       !time||!height||!width||rows>16384||(uint64_t)time*height*width+5!=rows||
       cosine->elements<(size_t)rows*24||sine->elements<(size_t)rows*24)return 0;
    if(!h3_gpu_begin(g))return 0;
    sg_vae_rope<<<(rows*24+255)/256,256,0,g->compute>>>((float*)tensor_pointer(cosine),(float*)tensor_pointer(sine),time,height,width,rows);
    int ok=launch_status(g,"reference VAE rotary positions")&&h3_gpu_submit(g);if(!ok)h3_gpu_cancel(g);return ok;
}
h3_gpu_tensor *h3_gpu_sglang_vae_weight(h3_gpu *g,const h3_gpu_tensor *w) {
    if(!g||!g->sglang_vae||!w||w->dtype!=H3_GPU_F32||!w->elements||w->elements>(1u<<26))return nullptr;
    auto *result=h3_gpu_tensor_alloc(g,w->elements,H3_GPU_F16,H3_GPU_DEVICE_ONLY);
    if(!result)return nullptr;
    if(!h3_gpu_begin(g)){h3_gpu_tensor_free(result);return nullptr;}
    sg_vae_cast_half<<<(w->elements+255)/256,256,0,g->compute>>>((__half*)tensor_pointer(result),(float*)tensor_pointer(w),w->elements,(unsigned*)tensor_pointer(g->vae_fault));
    if(!launch_status(g,"reference VAE weight cast")||!h3_gpu_submit(g)){h3_gpu_cancel(g);h3_gpu_tensor_free(result);return nullptr;}
    return result;
}
static int sglang_vae_linear(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *w,const h3_gpu_tensor *b,unsigned rows,unsigned k,unsigned n) {
    if(!launch_ready(g)||!rows||rows>16384||!k||!n||!in||!out||!w||out==in||
       in->dtype!=H3_GPU_F32||out->dtype!=H3_GPU_F32||w->dtype!=H3_GPU_F16||
       (b&&(b->dtype!=H3_GPU_F16||b->elements<n))||in->elements<(size_t)rows*k||
       out->elements<(size_t)rows*n||w->elements<(size_t)k*n)return h3_gpu_set_error(g,"invalid reference VAE linear");
    profile_scope timing(g,0);
    size_t ni=(size_t)rows*k,no=(size_t)rows*n;
    if(!sg_vae_scratch(g,&g->sglang_vae_input,ni,H3_GPU_F16)||
       !sg_vae_scratch(g,&g->sglang_vae_output,no,H3_GPU_F16))return 0;
    auto *x=(__half*)tensor_pointer(g->sglang_vae_input),*y=(__half*)tensor_pointer(g->sglang_vae_output);
    auto *weight=(__half*)tensor_pointer(w),*bias=(__half*)tensor_pointer(b);
    sg_vae_cast_half<<<(ni+255)/256,256,0,g->compute>>>(x,(float*)tensor_pointer(in),ni,(unsigned*)tensor_pointer(g->vae_fault));
    /* The oracle deliberately uses a separately rounded bias for SM12.x W2. */
    bool separate=(g->architecture>=120&&k==8192&&n==2048)||(k==24&&n==24);float alpha=1,beta=0;
    auto status=b&&!separate?g->sglang_blas->biased(y,x,weight,bias,rows,k,n,CUDA_R_16F,g->workspace,g->workspace_bytes,g->compute):
        g->sglang_blas->gemm(g->sglang_blas->handle,CUBLAS_OP_T,CUBLAS_OP_N,n,rows,k,&alpha,
            weight,CUDA_R_16F,k,x,CUDA_R_16F,k,&beta,y,CUDA_R_16F,n,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,"reference VAE FP16 GEMM failed (%d)",(int)status);
    sg_vae_cast_float<<<(no+255)/256,256,0,g->compute>>>((float*)tensor_pointer(out),y,separate?bias:nullptr,no,n,(unsigned*)tensor_pointer(g->vae_fault));
    g->stats.linear_dispatches++;return launch_status(g,"reference VAE FP16 linear");
}
__global__ static void sg_vae_qkv(const float *input,const float *cosine,const float *sine,
    float *q,float *k,float *v,unsigned heads,unsigned half,float epsilon) {
    unsigned item=blockIdx.x,tid=threadIdx.x,row=item/heads;size_t base=(size_t)item*192;
    __shared__ float sums[2][4];float qs=0,ks=0;
    for(unsigned j=0;j<4;j++)if(tid*4+j<64){float x=input[base+tid*4+j],y=input[base+64+tid*4+j];qs=fmaf(x,x,qs);ks=fmaf(y,y,ks);}
    for(int d=16;d;d>>=1){qs+=__shfl_down_sync(0xffffffff,qs,d);ks+=__shfl_down_sync(0xffffffff,ks,d);}
    if(!(tid%32)){sums[0][tid/32]=qs;sums[1][tid/32]=ks;}__syncthreads();
    if(tid<2){sums[0][tid]+=sums[0][tid+2];sums[1][tid]+=sums[1][tid+2];}__syncthreads();
    if(!tid){sums[0][0]+=sums[0][1];sums[1][0]+=sums[1][1];}__syncthreads();
    if(tid>=64)return;
    float qi=rsqrtf(sums[0][0]/64.f+epsilon),ki=rsqrtf(sums[1][0]/64.f+epsilon);
    float x=h3_sg_half(input[base+tid]*qi),y=h3_sg_half(input[base+64+tid]*ki);
    if(tid<half*2){unsigned pair=tid<half?tid+half:tid-half,p=tid%half;
        float xp=h3_sg_half(input[base+pair]*qi),yp=h3_sg_half(input[base+64+pair]*ki);
        float c=h3_sg_half(cosine[(size_t)row*half+p]),s=h3_sg_half(sine[(size_t)row*half+p]);
        float sign=tid<half?-1.f:1.f;
        x=h3_sg_half(h3_sg_half(x*c)+sign*h3_sg_half(xp*s));
        y=h3_sg_half(h3_sg_half(y*c)+sign*h3_sg_half(yp*s));}
    q[(size_t)item*64+tid]=x;k[(size_t)item*64+tid]=y;v[(size_t)item*64+tid]=input[base+128+tid];
}
static int sglang_vae_qkv(h3_gpu *g,h3_gpu_tensor *q,h3_gpu_tensor *k,h3_gpu_tensor *v,
    const h3_gpu_tensor *input,const h3_gpu_tensor *c,const h3_gpu_tensor *s,
    unsigned rows,unsigned heads,unsigned dim,unsigned half,float epsilon) {
    size_t n=(size_t)rows*heads*dim;
    if(!launch_ready(g)||!rows||rows>16384||heads!=32||dim!=64||half!=24||!std::isfinite(epsilon)||epsilon<=0||
       !q||!k||!v||!input||!c||!s||q==k||q==v||k==v||q->dtype!=H3_GPU_F32||k->dtype!=H3_GPU_F32||
       v->dtype!=H3_GPU_F32||input->dtype!=H3_GPU_F32||c->dtype!=H3_GPU_F32||s->dtype!=H3_GPU_F32||
       q->elements<n||k->elements<n||v->elements<n||input->elements<n*3||c->elements<(size_t)rows*half||s->elements<(size_t)rows*half)
        return h3_gpu_set_error(g,"invalid reference VAE QKV");
    profile_scope timing(g,6);
    sg_vae_qkv<<<rows*heads,128,0,g->compute>>>((float*)tensor_pointer(input),(float*)tensor_pointer(c),(float*)tensor_pointer(s),
        (float*)tensor_pointer(q),(float*)tensor_pointer(k),(float*)tensor_pointer(v),heads,half,epsilon);
    return launch_status(g,"reference VAE norm/RoPE");
}
static int sglang_vae_attention(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,unsigned rows,unsigned heads,unsigned dim,float scale) {
#ifndef H3_CUDA_USE_SGLANG_FLASH
    return h3_gpu_set_error(g,"reference VAE requires CUDA_SGLANG=1");
#else
    size_t n=(size_t)rows*heads*dim;
    if(!launch_ready(g)||!rows||rows>16384||heads!=32||dim!=64||!std::isfinite(scale)||scale<=0||
       !out||!q||!k||!v||out->dtype!=H3_GPU_F32||q->dtype!=H3_GPU_F32||k->dtype!=H3_GPU_F32||v->dtype!=H3_GPU_F32||
       out->elements<n||q->elements<n||k->elements<n||v->elements<n)return h3_gpu_set_error(g,"invalid reference VAE attention");
    profile_scope timing(g,1);
    if(!sg_vae_scratch(g,&g->vae_q,n,H3_GPU_F16)||!sg_vae_scratch(g,&g->vae_k,n,H3_GPU_F16)||
       !sg_vae_scratch(g,&g->vae_v,n,H3_GPU_F16)||!sg_vae_scratch(g,&g->sglang_vae_output,n,H3_GPU_F16)||
       !sg_vae_scratch(g,&g->sglang_lse,(size_t)rows*heads,H3_GPU_F32))return 0;
    if(g->exact_fused_vae_casts) {
        exact_vae_cast_qkv<<<(n+255)/256,256,0,g->compute>>>((__half*)tensor_pointer(g->vae_q),
            (__half*)tensor_pointer(g->vae_k),(__half*)tensor_pointer(g->vae_v),
            (float*)tensor_pointer(q),(float*)tensor_pointer(k),(float*)tensor_pointer(v),n,(unsigned*)tensor_pointer(g->vae_fault));
        g->exact_fused_vae_cast_count++;
    } else {
    g->exact_separate_vae_casts++;
    sg_vae_cast_half<<<(n+255)/256,256,0,g->compute>>>((__half*)tensor_pointer(g->vae_q),(float*)tensor_pointer(q),n,(unsigned*)tensor_pointer(g->vae_fault));
    sg_vae_cast_half<<<(n+255)/256,256,0,g->compute>>>((__half*)tensor_pointer(g->vae_k),(float*)tensor_pointer(k),n,(unsigned*)tensor_pointer(g->vae_fault));
    sg_vae_cast_half<<<(n+255)/256,256,0,g->compute>>>((__half*)tensor_pointer(g->vae_v),(float*)tensor_pointer(v),n,(unsigned*)tensor_pointer(g->vae_fault));
    }
    int status=h3_sglang_flash_vae(tensor_pointer(g->sglang_vae_output),tensor_pointer(g->vae_q),tensor_pointer(g->vae_k),tensor_pointer(g->vae_v),
        (float*)tensor_pointer(g->sglang_lse),rows,heads,scale,g->compute,g->error,sizeof(g->error));
    if(status!=1)return h3_gpu_set_error(g,"reference VAE FlashAttention failed: %d",status);
    sg_vae_cast_float<<<(n+255)/256,256,0,g->compute>>>((float*)tensor_pointer(out),(__half*)tensor_pointer(g->sglang_vae_output),nullptr,n,dim,(unsigned*)tensor_pointer(g->vae_fault));
    g->stats.attention_dispatches++;return launch_status(g,"reference VAE FlashAttention");
#endif
}
