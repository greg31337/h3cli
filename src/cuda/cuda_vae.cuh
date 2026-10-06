#include "src/log.h"
/* Default SGLang video VAE. Working tensors/residuals stay FP32.
 * FP16 matrix/attention operations retain the qualified rounding boundaries.
 * Plans/workspace belong to this decoder/device and are never shared with DiT. */
static int vae_scratch(h3_gpu *g,h3_gpu_tensor **slot,size_t n,h3_gpu_dtype type) {
    if(*slot && (*slot)->elements>=n && (*slot)->dtype==type)return 1;
    h3_gpu_tensor_free(*slot);*slot=nullptr;
    if(!h3_memory_check((uint64_t)n*item_size(type),"full VAE workspace",g->error,sizeof(g->error)))return 0;
    *slot=h3_gpu_tensor_alloc(g,n,type,H3_GPU_DEVICE_ONLY);return *slot!=nullptr;
}
#include "src/cuda/cuda_sglang_vae.cuh"
int h3_gpu_video_vae_configure(h3_gpu *g,int reserved) {
    if(reserved||!g||g->active)return -1;
#ifndef H3_CUDA_USE_SGLANG_FLASH
    h3_gpu_set_error(g,"full CUDA VAE requires CUDA_SGLANG=1");return -1;
#else
    /* Standalone decoder APIs also select the sole CUDA video implementation. */
    if(!g->sglang_blas) {
        g->sglang_blas=std::make_unique<h3_sglang_blas>();
        if(!g->sglang_blas->initialize(g->compute,g->workspace,g->workspace_bytes,g->error,sizeof(g->error)))return -1;
    }
    g->sglang_reference=true;g->sglang_vae=true;
    uint32_t zero=0;
    if(!g->vae_fault)g->vae_fault=h3_gpu_tensor_from_u32(g,&zero,1);
    if(!g->vae_fault)return -1;
    H3_VERBOSE("h3cli: full VAE: SGLang FP16 block matrices/attention, FP32 residuals; tile_batch=1\n");
    return 0;
#endif
}
int h3_gpu_video_linear(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *w,const h3_gpu_tensor *b,uint32_t rows,uint32_t k,uint32_t n) {
    if(g && g->sglang_vae && w && w->dtype==H3_GPU_F16)
        return sglang_vae_linear(g,out,in,w,b,rows,k,n);
    if(g && g->sglang_vae && k==24 && n==2048) {
        if(!h3_gpu_linear_f32(g,out,in,w,b,rows,k,n))return 0;
        size_t count=(size_t)rows*n;
        sg_vae_round_float<<<(count+255)/256,256,0,g->compute>>>((float*)tensor_pointer(out),count,(unsigned*)tensor_pointer(g->vae_fault));
        return launch_status(g,"reference VAE embedding cast");
    }
    return h3_gpu_linear_f32(g,out,in,w,b,rows,k,n);
}

int h3_gpu_video_sdpa(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t seq,uint32_t heads,uint32_t dim,float scale) {
    if(g && g->sglang_vae)return sglang_vae_attention(g,out,q,k,v,seq,heads,dim,scale);
    return h3_gpu_sdpa_f32(g,out,q,k,v,seq,heads,dim,scale);
}
/* Reference output must remain unclamped until spatial/temporal blending. */
__global__ static void sglang_vae_unpack(float *rgb,const float *p,size_t n,int h,int w,int total,int first,unsigned *fault) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int c=i%3,x=(i/3)%w,y=(i/(3*w))%h,frame=i/((size_t)3*w*h)+first;
    int t=frame+3;if(total==22 && frame>=17)t+=3;
    size_t patch=((size_t)(t/4)*(h/16)+y/16)*(w/16)+x/16;
    size_t component=(((c*4+t%4)*16+y%16)*16+x%16);
    float value=p[patch*3072+component];if(!isfinite(value))atomicOr(fault,1u);rgb[i]=value;
}
/* Match the already-qualified CPU blend, including its separate multiplies
 * and above-then-left order. The neighboring samples are unblended tiles. */
__global__ static void sg_spatial_stitch(float *out,const float *tiles,h3_gpu::stitch_plan p,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=n)return;
    int c=i%3,x=(i/3)%p.width,y=(i/(3*p.width))%p.height;
    size_t frame=i/((size_t)3*p.width*p.height);
    int ty=0,tx=0;
    while(ty+1<p.ny&&y>=p.ys[ty+1])ty++;
    while(tx+1<p.nx&&x>=p.xs[tx+1])tx++;
    int ly=y-p.ys[ty],lx=x-p.xs[tx];
    size_t stride=(size_t)p.frames*p.h*p.w*3,tile=(size_t)ty*p.nx+tx;
    size_t local=((frame*p.h+ly)*p.w+lx)*3+c;
    float value=tiles[tile*stride+local];
    int oy=ty?p.ys[ty-1]+p.h-p.ys[ty]:0;
    if(ty&&ly<oy){
        size_t top=((frame*p.h+p.h-oy+ly)*p.w+lx)*3+c;
        float a=(float)ly*__fdiv_rn(1.f,(float)oy);
        value=tiles[(tile-p.nx)*stride+top]*(1.f-a)+value*a;
    }
    int ox=tx?p.xs[tx-1]+p.w-p.xs[tx]:0;
    if(tx&&lx<ox){
        size_t left=((frame*p.h+ly)*p.w+p.w-ox+lx)*3+c;
        float a=(float)lx*__fdiv_rn(1.f,(float)ox);
        value=tiles[(tile-1)*stride+left]*(1.f-a)+value*a;
    }
    out[i]=value;
}
int h3_gpu_sglang_stitch_begin(h3_gpu *g,int h,int w,int ny,int nx,const int *ys,const int *xs,int first,int frames) {
    if(!g||!g->sglang_vae||getenv("H3_TEST_SGLANG_CPU_STITCH"))return 0;
    if(h<16||w<16||h>320||w>320||h%16||w%16||ny<1||nx<1||ny>16||nx>16||
       !ys||!xs||ys[0]||xs[0]||first<0||frames<1||first>22-frames)return -1;
    for(int j=1;j<ny;j++)if(ys[j]<=ys[j-1]||ys[j]>ys[j-1]+h)return -1;
    for(int j=1;j<nx;j++)if(xs[j]<=xs[j-1]||xs[j]>xs[j-1]+w)return -1;
    int height=ys[ny-1]+h,width=xs[nx-1]+w;
    size_t tiles=(size_t)frames*h*w*3*ny*nx,canvas=(size_t)frames*height*width*3;
    // Larger geometries retain the original bounded host stitching path.
    if(tiles>(256u<<20)/4||canvas>(128u<<20)/4)return 0;
    if(!vae_scratch(g,&g->sglang_vae_tiles,tiles,H3_GPU_F32)||
       !vae_scratch(g,&g->sglang_vae_canvas,canvas,H3_GPU_F32))return -1;
    auto &p=g->sglang_stitch;p.h=h;p.w=w;p.ny=ny;p.nx=nx;p.first=first;p.frames=frames;p.height=height;p.width=width;
    std::copy(ys,ys+ny,p.ys);std::copy(xs,xs+nx,p.xs);g->sglang_stitch_next=0;return 1;
}
int h3_gpu_sglang_stitch_tile(h3_gpu *g,const h3_gpu_tensor *projected,int tile) {
    if(!g||!g->sglang_vae||!projected||projected->dtype!=H3_GPU_F32)return 0;
    auto &p=g->sglang_stitch;
    if(tile!=g->sglang_stitch_next||tile>=p.ny*p.nx||!g->sglang_vae_tiles)return 0;
    size_t n=(size_t)p.frames*p.h*p.w*3;
    if(projected->elements<(size_t)7*(p.h/16)*(p.w/16)*3072||!h3_gpu_begin(g))return 0;
    sglang_vae_unpack<<<(n+255)/256,256,0,g->compute>>>((float*)tensor_pointer(g->sglang_vae_tiles)+n*tile,
        (const float*)tensor_pointer(projected),n,p.h,p.w,22,p.first,(unsigned*)tensor_pointer(g->vae_fault));
    if(!launch_status(g,"reference VAE device tile")||!h3_gpu_submit(g)){h3_gpu_cancel(g);return 0;}
    g->sglang_stitch_next++;return 1;
}
int h3_gpu_sglang_stitch_read(h3_gpu *g,float *rgb) {
    if(!g||!g->sglang_vae||!rgb)return 0;auto &p=g->sglang_stitch;
    if(g->sglang_stitch_next!=p.ny*p.nx||!g->sglang_vae_canvas)return 0;
    size_t n=(size_t)p.frames*p.height*p.width*3,bytes=n*4;
    if(bytes>(128u<<20))return 0;
    if(g->vae_pinned_bytes<bytes){
        if(!h3_memory_check(bytes,"reference VAE stitched output",g->error,sizeof(g->error)))return 0;
        if(g->vae_pinned_rgb){cudaFreeHost(g->vae_pinned_rgb);g->stats.pinned_bytes-=g->vae_pinned_bytes;g->vae_pinned_rgb=nullptr;g->vae_pinned_bytes=0;}
        if(!checked(g,cudaHostAlloc(&g->vae_pinned_rgb,bytes,cudaHostAllocDefault),"reference stitched output",bytes))return 0;
        g->vae_pinned_bytes=bytes;g->stats.pinned_bytes+=bytes;g->stats.peak_pinned_bytes=std::max(g->stats.peak_pinned_bytes,g->stats.pinned_bytes);
    }
    if(!h3_gpu_begin(g))return 0;
    sg_spatial_stitch<<<(n+255)/256,256,0,g->compute>>>((float*)tensor_pointer(g->sglang_vae_canvas),
        (const float*)tensor_pointer(g->sglang_vae_tiles),p,n);
    if(!launch_status(g,"reference VAE spatial blend")){h3_gpu_cancel(g);return 0;}
    double start=now();
    {profile_scope timing(g,5);
     if(!checked(g,cudaMemcpyAsync(g->vae_pinned_rgb,tensor_pointer(g->sglang_vae_canvas),bytes,cudaMemcpyDeviceToHost,g->compute),"reference stitched readback",bytes)){h3_gpu_cancel(g);return 0;}}
    if(!h3_gpu_submit(g)){h3_gpu_cancel(g);return 0;}
    memcpy(rgb,g->vae_pinned_rgb,bytes);g->stats.d2h_bytes+=bytes;g->stats.transfer_seconds+=now()-start;return 1;
}
int h3_gpu_video_unpack(h3_gpu *g,const h3_gpu_tensor *p,float *rgb,int h,int w,int total,int first,int frames) {
    if(!g || !g->sglang_vae)return 0;
    if(!p||!rgb||h<1||w<1||h>20||w>20||total<1||total>22||first<0||frames<1||first>total-frames)return -1;
    size_t n=(size_t)h*w*256*3*frames;
    if(n>(32u<<20)/sizeof(float)){h3_gpu_set_error(g,"VAE output staging exceeds 32 MiB budget");return -1;}
    int last=first+frames-1+3;if(total==22 && first+frames-1>=17)last+=3;
    if(p->dtype!=H3_GPU_F32 || p->elements<((size_t)last/4+1)*h*w*3072)return -1;
    if(!vae_scratch(g,&g->vae_rgb,n,H3_GPU_F32)||!h3_gpu_begin(g))return -1;
    sglang_vae_unpack<<<(n+255)/256,256,0,g->compute>>>((float*)tensor_pointer(g->vae_rgb),(const float*)tensor_pointer(p),n,h*16,w*16,total,first,(unsigned*)tensor_pointer(g->vae_fault));
    if(!launch_status(g,"VAE RGB unpack")){h3_gpu_cancel(g);return -1;}
    if(getenv("H3_TEST_FULL_VAE_PAGEABLE")) {
        if(!h3_gpu_submit(g)||!h3_gpu_tensor_read_f32(g->vae_rgb,rgb,n)){h3_gpu_cancel(g);return -1;}return 1;
    }
    size_t bytes=n*sizeof(float);
    if(bytes>(32u<<20)){h3_gpu_set_error(g,"VAE output staging exceeds 32 MiB budget");h3_gpu_cancel(g);return -1;}
    if(g->vae_pinned_bytes<bytes) {
        if(!h3_memory_check(bytes,"full VAE pinned output",g->error,sizeof(g->error))){h3_gpu_cancel(g);return -1;}
        if(g->vae_pinned_rgb){cudaFreeHost(g->vae_pinned_rgb);g->stats.pinned_bytes-=g->vae_pinned_bytes;g->vae_pinned_rgb=nullptr;g->vae_pinned_bytes=0;}
        if(!checked(g,cudaHostAlloc(&g->vae_pinned_rgb,bytes,cudaHostAllocDefault),"VAE output staging",bytes)){h3_gpu_cancel(g);return -1;}
        g->vae_pinned_bytes=bytes;g->stats.pinned_bytes+=bytes;
        g->stats.peak_pinned_bytes=std::max(g->stats.peak_pinned_bytes,g->stats.pinned_bytes);
    }
    /* Same-stream async D2H preserves order without a separate compute fence.
     * The existing bounded frame sink supplies backpressure before reuse. */
    double start=now();
    { profile_scope timing(g,5);
      if(!checked(g,cudaMemcpyAsync(g->vae_pinned_rgb,g->vae_rgb->data,bytes,cudaMemcpyDeviceToHost,g->compute),"VAE output copy",bytes)){h3_gpu_cancel(g);return -1;} }
    if(!h3_gpu_submit(g)){h3_gpu_cancel(g);return -1;}
    memcpy(rgb,g->vae_pinned_rgb,bytes);g->stats.d2h_bytes+=bytes;g->stats.transfer_seconds+=now()-start;
    return 1;
}

int h3_gpu_video_graph_begin(h3_gpu *g) {
    if(!launch_ready(g))return -1;
    const char *opt=getenv("H3_FULL_VAE_CUDA_GRAPH");
    const char *capture=getenv("H3_TEST_SGLANG_DIR");
    bool reference_graph=g->sglang_vae && (!capture||!*capture) && (!opt||!strcmp(opt,"1"));
    if(!reference_graph || g->profiling || g->vae_graph_disabled || !g->vae_warm)return 0;
    if(g->vae_graph){
        if(!checked(g,cudaGraphLaunch(g->vae_graph,g->compute),"VAE graph replay"))return -1;
        g->vae_replays++;return 2;
    }
    if(cudaStreamBeginCapture(g->compute,cudaStreamCaptureModeThreadLocal)!=cudaSuccess){
        cudaGetLastError();g->vae_graph_disabled=true;
        fprintf(stderr,"h3cli: full VAE graph capture unavailable; using ordinary submissions\n");return 0;
    }
    g->vae_capturing=true;return 1;
}
int h3_gpu_video_graph_end(h3_gpu *g) {
    if(!g)return -1;
    if(!g->vae_capturing){g->vae_warm=true;return 1;}
    cudaGraph_t graph=nullptr;auto status=cudaStreamEndCapture(g->compute,&graph);g->vae_capturing=false;
    if(status==cudaSuccess && graph)status=cudaGraphInstantiate(&g->vae_graph,graph,nullptr,nullptr,0);
    if(graph)cudaGraphDestroy(graph);
    if(status!=cudaSuccess){
        cudaGetLastError();g->vae_graph_disabled=true;g->active=false;g->deferred_error=false;
        fprintf(stderr,"h3cli: full VAE graph unavailable (%s); retrying ordinary tile\n",cudaGetErrorString(status));return 0;
    }
    if(!checked(g,cudaGraphLaunch(g->vae_graph,g->compute),"VAE captured tile"))return -1;
    H3_VERBOSE("h3cli: full VAE captured fixed tile graph; one graph per decoder\n");return 1;
}

int h3_gpu_video_qkv(h3_gpu *g,h3_gpu_tensor *q,h3_gpu_tensor *k,h3_gpu_tensor *v,
    const h3_gpu_tensor *qkv,const h3_gpu_tensor *c,const h3_gpu_tensor *s,
    uint32_t rows,uint32_t heads,uint32_t dim,uint32_t half,float epsilon) {
    if(g && g->sglang_vae)return sglang_vae_qkv(g,q,k,v,qkv,c,s,rows,heads,dim,half,epsilon);
    return h3_gpu_video_qkv_rope_f32(g,q,k,v,qkv,c,s,rows,heads,dim,half,epsilon);
}

/* Preserve the ordinary encoder's first heuristic and FP32 arithmetic while
 * reusing descriptors. Input/output allocations remain caller-owned. */
static int vae_conv_gemm(h3_gpu *g,void *out,const void *in,const void *w,void *bias,
    unsigned rows,unsigned k,unsigned n) {
    gemm_key key={rows,n,k,(int)H3_GPU_F32,bias?1:0,0};auto found=g->conv_plans.find(key);
    if(found==g->conv_plans.end()) {
        if(g->conv_plans.size()>=64)return 0;
        auto plan=std::make_unique<cuda_gemm_plan>();auto &p=*plan;
        cublasOperation_t transpose=CUBLAS_OP_T;cublasLtEpilogue_t epi=CUBLASLT_EPILOGUE_BIAS;
        bool ok=cublasLtMatmulDescCreate(&p.operation,CUBLAS_COMPUTE_32F,CUDA_R_32F)==CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_TRANSA,&transpose,sizeof(transpose))==CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&p.a,CUDA_R_32F,k,n,k)==CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&p.b,CUDA_R_32F,k,rows,k)==CUBLAS_STATUS_SUCCESS &&
            cublasLtMatrixLayoutCreate(&p.c,CUDA_R_32F,n,rows,n)==CUBLAS_STATUS_SUCCESS;
        if(bias)ok=ok && cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_EPILOGUE,&epi,sizeof(epi))==CUBLAS_STATUS_SUCCESS &&
            cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias))==CUBLAS_STATUS_SUCCESS;
        cublasLtMatmulPreference_t pref=nullptr;
        ok=ok && cublasLtMatmulPreferenceCreate(&pref)==CUBLAS_STATUS_SUCCESS;
        if(ok)ok=cublasLtMatmulPreferenceSetAttribute(pref,CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES,&g->workspace_bytes,sizeof(g->workspace_bytes))==CUBLAS_STATUS_SUCCESS;
        cublasLtMatmulHeuristicResult_t result[8]={};int count=0;
        if(ok)ok=cublasLtMatmulAlgoGetHeuristic(g->lt,p.operation,p.a,p.b,p.c,p.c,pref,8,result,&count)==CUBLAS_STATUS_SUCCESS && count>0 && result[0].state==CUBLAS_STATUS_SUCCESS;
        if(pref)cublasLtMatmulPreferenceDestroy(pref);if(!ok)return 0;
        p.algorithm=result[0].algo;found=g->conv_plans.emplace(key,std::move(plan)).first;
    }
    auto &p=*found->second;
    if(bias && cublasLtMatmulDescSetAttribute(p.operation,CUBLASLT_MATMUL_DESC_BIAS_POINTER,&bias,sizeof(bias))!=CUBLAS_STATUS_SUCCESS){h3_gpu_set_error(g,"encoder plan bias update failed");return -1;}
    float alpha=1,beta=0;
    if(cublasLtMatmul(g->lt,p.operation,&alpha,w,p.a,in,p.b,&beta,out,p.c,out,p.c,&p.algorithm,g->workspace,g->workspace_bytes,g->compute)!=CUBLAS_STATUS_SUCCESS){h3_gpu_set_error(g,"prepared encoder GEMM failed");return -1;}
    return 1;
}

int h3_gpu_video_graph_abort(h3_gpu *g) {
    if(!g||!g->vae_capturing)return 0;
    char reason[1024];snprintf(reason,sizeof(reason),"%s",g->error);
    h3_gpu_cancel(g);g->accessed.clear();cudaGetLastError();g->deferred_error=false;
    fprintf(stderr,"h3cli: full VAE capture rejected (%s); retrying ordinary tile\n",reason);return 1;
}
