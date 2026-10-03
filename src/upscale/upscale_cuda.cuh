/* BF16 NDHWC upscaler operators; isolated from ordinary DiT/VAE dispatch. */
#include "src/upscale/upscale_gpu.h"
int h3_gpu_upscale_linear(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *weight,uint32_t rows,uint32_t inputs,uint32_t outputs) {
    if(!launch_ready(g)||!rows||!inputs||!outputs||rows>1024||inputs>13824||outputs>512||
       !h3_gpu_require_bf16(g,out,(size_t)rows*outputs,"upscale linear output")||
       !h3_gpu_require_bf16(g,in,(size_t)rows*inputs,"upscale linear input")||
       !h3_gpu_require_bf16(g,weight,(size_t)inputs*outputs,"upscale linear weights"))return 0;
    if(!g->upscale_blas) {
        if(cublasCreate(&g->upscale_blas)!=CUBLAS_STATUS_SUCCESS||
           cublasSetStream(g->upscale_blas,g->compute)!=CUBLAS_STATUS_SUCCESS||
           cublasSetMathMode(g->upscale_blas,CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION)!=CUBLAS_STATUS_SUCCESS)
            return h3_gpu_set_error(g,"cannot initialize upscaler F32 accumulation");
    }
    float alpha=1.f,beta=0.f;
    cublasStatus_t status=cublasGemmEx(g->upscale_blas,CUBLAS_OP_T,CUBLAS_OP_N,
        outputs,rows,inputs,&alpha,tensor_pointer(weight),CUDA_R_16BF,inputs,
        tensor_pointer(in),CUDA_R_16BF,inputs,&beta,tensor_pointer(out),CUDA_R_16BF,outputs,
        CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT_TENSOR_OP);
    if(status!=CUBLAS_STATUS_SUCCESS)return h3_gpu_set_error(g,"upscaler convolution GEMM failed (%d)",(int)status);
    g->stats.linear_dispatches++;return launch_status(g,"upscaler F32 convolution accumulation");
}
__global__ static void up_point(ushort *out,const ushort *in,const ushort *a,const ushort *b,
    h3_upscale_gpu_shape p,uint op) {

    uint i=blockIdx.x*blockDim.x+threadIdx.x;
    uint spatial=p.time*p.height*p.width,c=p.channels;
    if(op==1) {
        uint filter=c*p.kernel_time*p.kernel_space*p.kernel_space;
        if(i>=p.rows*filter)return;
        uint pos=p.first+i/filter,q=i%filter;
        int dx=int(q%p.kernel_space)-int(p.kernel_space/2);q/=p.kernel_space;
        int dy=int(q%p.kernel_space)-int(p.kernel_space/2);q/=p.kernel_space;
        int dz=int(q%p.kernel_time)-int(p.kernel_time/2);uint ch=q/p.kernel_time;
        int x=int(pos%p.width)+dx,y=int((pos/p.width)%p.height)+dy,z=int(pos/p.width/p.height)+dz;
        out[i]=(pos<spatial&&x>=0&&x<int(p.width)&&y>=0&&y<int(p.height)&&z>=0&&z<int(p.time))?
            in[((uint(z)*p.height+uint(y))*p.width+uint(x))*c+ch]:ushort(0);return;
    }
    if(op==4) {
        if(i>=p.time*p.out_height*p.out_width*c)return;
        uint ch=i%c,pos=i/c,ox=pos%p.out_width,oy=(pos/p.out_width)%p.out_height,z=pos/p.out_width/p.out_height;
        float xf=fmaxf(0.f,(float(ox)+.5f)*float(p.width)/float(p.out_width)-.5f);
        float yf=fmaxf(0.f,(float(oy)+.5f)*float(p.height)/float(p.out_height)-.5f);
        uint x0=uint(xf),y0=uint(yf),x1=min(x0+1,p.width-1),y1=min(y0+1,p.height-1);
        float wx=xf-float(x0),wy=yf-float(y0);
        float v00=h3_bf16_to_f32(in[((z*p.height+y0)*p.width+x0)*c+ch]);
        float v01=h3_bf16_to_f32(in[((z*p.height+y0)*p.width+x1)*c+ch]);
        float v10=h3_bf16_to_f32(in[((z*p.height+y1)*p.width+x0)*c+ch]);
        float v11=h3_bf16_to_f32(in[((z*p.height+y1)*p.width+x1)*c+ch]);
        out[i]=h3_f32_to_bf16((1.f-wy)*((1.f-wx)*v00+wx*v01)+wy*((1.f-wx)*v10+wx*v11));return;
    }
    if(i>=spatial*c)return;
    uint ch=i%c;
    if(op==2) {
        uint pos=i/c,z=pos/(p.height*p.width),xy=pos%(p.height*p.width);float value=0.f;
        for(uint k=0;k<p.kernel_time;k++) {
            int iz=int(z)+int(k)-int(p.kernel_time/2);
            if(iz>=0&&iz<int(p.time))value=fmaf(h3_bf16_to_f32(in[(uint(iz)*p.height*p.width+xy)*c+ch]),h3_bf16_to_f32(a[ch*p.kernel_time+k]),value);
        }
        out[i]=h3_f32_to_bf16(value+h3_bf16_to_f32(b[ch]));return;
    }
    if(op==3) {
        float scale=h3_bf16_to_f32(h3_f32_to_bf16(1.f+h3_bf16_to_f32(a[ch])));
        float value=h3_bf16_to_f32(h3_f32_to_bf16(h3_bf16_to_f32(in[i])*scale));
        out[i]=h3_f32_to_bf16(value+h3_bf16_to_f32(a[c+ch]));return;
    }
    if(op==5)out[i]=h3_f32_to_bf16(h3_bf16_to_f32(in[i])+h3_bf16_to_f32(a[ch]));
}
__global__ static void up_norm(ushort *out,const ushort *in,const ushort *a,const ushort *b,h3_upscale_gpu_shape p) {
    uint tid=threadIdx.x,group=blockIdx.x;
    __shared__ float shared[256];
    uint width=p.channels/32,n=p.time*p.height*p.width*width;
    float sum=0;
    for(uint j=tid;j<n;j+=256)sum+=h3_bf16_to_f32(in[(j/width)*p.channels+group*width+j%width]);
    shared[tid]=sum;__syncthreads();
    for(uint stride=128;stride;stride>>=1) {if(tid<stride)shared[tid]+=shared[tid+stride];__syncthreads();}
    float mean=shared[0]/float(n);__syncthreads();sum=0;
    for(uint j=tid;j<n;j+=256) {float d=h3_bf16_to_f32(in[(j/width)*p.channels+group*width+j%width])-mean;sum=fmaf(d,d,sum);}
    shared[tid]=sum;__syncthreads();
    for(uint stride=128;stride;stride>>=1) {if(tid<stride)shared[tid]+=shared[tid+stride];__syncthreads();}
    float inv=rsqrtf(shared[0]/float(n)+1e-5f);__syncthreads();
    for(uint j=tid;j<n;j+=256) {
        uint ch=group*width+j%width,i=(j/width)*p.channels+ch;
        out[i]=h3_f32_to_bf16((h3_bf16_to_f32(in[i])-mean)*inv*h3_bf16_to_f32(a[ch])+h3_bf16_to_f32(b[ch]));
    }
}

int h3_gpu_upscale(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *a,const h3_gpu_tensor *b,h3_upscale_gpu_op op,h3_upscale_gpu_shape p) {
    size_t n[4];if(!h3_upscale_gpu_counts(op,p,n)||!launch_ready(g))return 0;
    if(!h3_gpu_require_bf16(g,out,n[0],"upscale output")||!h3_gpu_require_bf16(g,in,n[1],"upscale input")||
       (n[2]&&!h3_gpu_require_bf16(g,a,n[2],"upscale weights"))||
       (n[3]&&!h3_gpu_require_bf16(g,b,n[3],"upscale bias"))||
       ((op==H3_UP_COLUMNS||op==H3_UP_RESIZE)&&out==in))return 0;
    if(op==H3_UP_NORM)up_norm<<<32,256,0,g->compute>>>((ushort*)tensor_pointer(out),(const ushort*)tensor_pointer(in),
        (const ushort*)tensor_pointer(a),(const ushort*)tensor_pointer(b),p);
    else up_point<<<(unsigned)((n[0]+255)/256),256,0,g->compute>>>((ushort*)tensor_pointer(out),(const ushort*)tensor_pointer(in),
        (const ushort*)tensor_pointer(a),(const ushort*)tensor_pointer(b),p,(uint)op);
    g->stats.direct_dispatches++;return launch_status(g,"native latent upscaler");
}
