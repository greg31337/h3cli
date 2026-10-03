/* Native adaptive recipe 1: deterministic FP32 reduction, no atomic sums.
 * Included once by gpu_cuda.cu after the tensor helpers. */
__global__ static void adaptive_probe_kernel(ushort *probe,const ushort *output,
    const ushort *anchor,float *partial,size_t elements,unsigned columns,
    unsigned video_start,unsigned video_rows,unsigned audio_start,unsigned audio_rows,bool ready) {
    __shared__ float sums[6][256];
    float local[6]={};
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<elements;i+=256*256) {
        float x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(probe+i));
        float y=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(output+i));
        __nv_bfloat16 p=__float2bfloat16_rn(y-x);
        probe[i]=*reinterpret_cast<ushort*>(&p);
        float a=ready?__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(anchor+i)):0;
        float v=__bfloat162float(p),diff=ready?fabsf(v-a):0;
        if(!isfinite(v)||!isfinite(a))diff=INFINITY;
        float denom=fabsf(a);local[0]+=diff;local[1]+=denom;
        unsigned row=(unsigned)(i/columns);
        if(row>=video_start&&row-video_start<video_rows){local[2]+=diff;local[3]+=denom;}
        if(row>=audio_start&&row-audio_start<audio_rows){local[4]+=diff;local[5]+=denom;}
    }
    for(unsigned j=0;j<6;j++)sums[j][threadIdx.x]=local[j];
    __syncthreads();
    for(unsigned stride=128;stride;stride>>=1) {
        if(threadIdx.x<stride)for(unsigned j=0;j<6;j++)sums[j][threadIdx.x]+=sums[j][threadIdx.x+stride];
        __syncthreads();
    }
    if(!threadIdx.x)for(unsigned j=0;j<6;j++)partial[blockIdx.x*6+j]=sums[j][0];
}
__global__ static void adaptive_finish_kernel(float *partial) {
    __shared__ float sums[6][256];
    for(unsigned j=0;j<6;j++)sums[j][threadIdx.x]=partial[threadIdx.x*6+j];
    __syncthreads();
    for(unsigned stride=128;stride;stride>>=1) {
        if(threadIdx.x<stride)for(unsigned j=0;j<6;j++)sums[j][threadIdx.x]+=sums[j][threadIdx.x+stride];
        __syncthreads();
    }
    if(!threadIdx.x)for(unsigned j=0;j<6;j++)partial[256*6+j]=sums[j][0];
}
int h3_gpu_adaptive_probe(h3_gpu *g,h3_gpu_tensor *probe,const h3_gpu_tensor *output,
    const h3_gpu_tensor *anchor,h3_gpu_tensor *scratch,unsigned rows,unsigned columns,
    unsigned video_start,unsigned video_rows,unsigned audio_start,unsigned audio_rows,
    int ready,float scores[3]) {
    size_t n=size_t(rows)*columns;
    if(!g||!launch_ready(g)||!rows||!columns||!scores||!probe||!output||!anchor||!scratch||
       probe==output||probe==anchor||output==anchor||
       probe->owner!=g||output->owner!=g||anchor->owner!=g||scratch->owner!=g||
       probe->dtype!=H3_GPU_BF16||output->dtype!=H3_GPU_BF16||anchor->dtype!=H3_GPU_BF16||scratch->dtype!=H3_GPU_F32||
       probe->elements<n||output->elements<n||anchor->elements<n||scratch->elements<256*6+6||
       video_start>rows||video_rows>rows-video_start||audio_start>rows||audio_rows>rows-audio_start)
        return h3_gpu_set_error(g,"invalid adaptive probe tensors/layout");
    auto *partials=(float*)tensor_pointer(scratch);
    adaptive_probe_kernel<<<256,256,0,g->compute>>>((ushort*)tensor_pointer(probe),
        (const ushort*)tensor_pointer(output),(const ushort*)tensor_pointer(anchor),partials,
        n,columns,video_start,video_rows,audio_start,audio_rows,ready!=0);
    adaptive_finish_kernel<<<1,256,0,g->compute>>>(partials);
    float sums[6];
    if(!launch_status(g,"adaptive probe")||!checked(g,cudaMemcpyAsync(sums,partials+256*6,sizeof(sums),cudaMemcpyDeviceToHost,g->compute),"adaptive scalar readback",sizeof(sums))||
       !checked(g,cudaStreamSynchronize(g->compute),"adaptive decision"))return 0;
    size_t counts[3]={n,size_t(video_rows)*columns,size_t(audio_rows)*columns};
    for(unsigned i=0;i<3;i++) {
        if(!std::isfinite(sums[2*i])||!std::isfinite(sums[2*i+1]))return h3_gpu_set_error(g,"nonfinite adaptive probe or reduction");
        float count=(float)counts[i];
        scores[i]=count?(sums[2*i]/count)/fmaxf(sums[2*i+1]/count,1e-6f):0;
        if(!std::isfinite(scores[i]))return h3_gpu_set_error(g,"nonfinite adaptive score");
    }
    return 1;
}

/* Recipe 4 has its own reduction, leaving recipes 1/2/3 bitwise unchanged.
 * The small temporal map is passed by value, with no per-token device map. */
__global__ static void adaptive_continuation_probe_kernel(ushort *probe,const ushort *output,
    const ushort *anchor,float *partial,size_t elements,unsigned columns,
    h3_adaptive_regions plan,bool ready) {
    __shared__ float sums[2*H3_ADAPTIVE_REGIONS][256];
    float local[2*H3_ADAPTIVE_REGIONS]={};
    for(size_t i=size_t(blockIdx.x)*256+threadIdx.x;i<elements;i+=256*256) {
        float x=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(probe+i));
        float y=__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(output+i));
        __nv_bfloat16 p=__float2bfloat16_rn(y-x);probe[i]=*reinterpret_cast<ushort*>(&p);
        float a=ready?__bfloat162float(*reinterpret_cast<const __nv_bfloat16*>(anchor+i)):0;
        float v=__bfloat162float(p),diff=ready?fabsf(v-a):0;
        if(!isfinite(v)||!isfinite(a))diff=INFINITY;
        float denom=fabsf(a);local[0]+=diff;local[1]+=denom;
        unsigned row=(unsigned)(i/columns),slot=0;
        if(row>=plan.video_start&&row-plan.video_start<plan.video_rows)
            slot=plan.video[(row-plan.video_start)/plan.spatial];
        else if(row>=plan.audio_start&&row-plan.audio_start<plan.audio_rows)
            slot=plan.audio[(row-plan.audio_start)%plan.audio_t];
        #pragma unroll
        for(unsigned j=1;j<H3_ADAPTIVE_REGIONS;j++)if(slot==j){local[2*j]+=diff;local[2*j+1]+=denom;}
    }
    for(unsigned j=0;j<2*H3_ADAPTIVE_REGIONS;j++)sums[j][threadIdx.x]=local[j];
    __syncthreads();
    for(unsigned stride=128;stride;stride>>=1) {
        if(threadIdx.x<stride)for(unsigned j=0;j<2*H3_ADAPTIVE_REGIONS;j++)sums[j][threadIdx.x]+=sums[j][threadIdx.x+stride];
        __syncthreads();
    }
    if(!threadIdx.x)for(unsigned j=0;j<2*H3_ADAPTIVE_REGIONS;j++)partial[blockIdx.x*2*H3_ADAPTIVE_REGIONS+j]=sums[j][0];
}
__global__ static void adaptive_continuation_finish_kernel(float *partial) {
    __shared__ float sums[2][256];unsigned region=blockIdx.x;
    for(unsigned j=0;j<2;j++)sums[j][threadIdx.x]=partial[threadIdx.x*2*H3_ADAPTIVE_REGIONS+2*region+j];
    __syncthreads();
    for(unsigned stride=128;stride;stride>>=1) {
        if(threadIdx.x<stride)for(unsigned j=0;j<2;j++)sums[j][threadIdx.x]+=sums[j][threadIdx.x+stride];
        __syncthreads();
    }
    if(!threadIdx.x)for(unsigned j=0;j<2;j++)partial[256*2*H3_ADAPTIVE_REGIONS+2*region+j]=sums[j][0];
}
int h3_gpu_adaptive_continuation_probe(h3_gpu *g,h3_gpu_tensor *probe,const h3_gpu_tensor *output,
    const h3_gpu_tensor *anchor,h3_gpu_tensor *scratch,unsigned columns,
    const h3_adaptive_regions *p,int ready,float scores[H3_ADAPTIVE_REGIONS]) {
    if(!g||!launch_ready(g)||!p||!columns||!scores||!p->rows||!p->spatial||!p->video_t||p->video_t>107||
       !p->audio_t||p->audio_t>604||p->video_rows!=(uint64_t)p->spatial*p->video_t||p->audio_rows!=2*p->audio_t||
       p->video_start>p->rows||p->video_rows>p->rows-p->video_start||
       p->audio_start>p->rows||p->audio_rows>p->rows-p->audio_start||
       !(p->video_start+p->video_rows<=p->audio_start||p->audio_start+p->audio_rows<=p->video_start)||
       (ready!=0&&ready!=1))return h3_gpu_set_error(g,"invalid adaptive continuation probe layout");
    uint32_t counts[H3_ADAPTIVE_REGIONS]={p->rows};
    for(unsigned i=0;i<p->video_t;i++) {
        unsigned s=p->video[i];if(s!=0&&s!=1&&(s<3||s>=13))return h3_gpu_set_error(g,"invalid adaptive video class");
        if(s)counts[s]+=p->spatial;
    }
    for(unsigned i=0;i<p->audio_t;i++) {
        unsigned s=p->audio[i];if(s!=0&&s!=2&&(s<13||s>=23))return h3_gpu_set_error(g,"invalid adaptive audio class");
        if(s)counts[s]+=2;
    }
    if(!counts[1]||!counts[2]||memcmp(counts,p->counts,sizeof(counts)))return h3_gpu_set_error(g,"invalid adaptive class counts");
    size_t n=size_t(p->rows)*columns;
    if(!probe||!output||!anchor||!scratch||probe==output||probe==anchor||output==anchor||
       probe->owner!=g||output->owner!=g||anchor->owner!=g||scratch->owner!=g||
       probe->dtype!=H3_GPU_BF16||output->dtype!=H3_GPU_BF16||anchor->dtype!=H3_GPU_BF16||scratch->dtype!=H3_GPU_F32||
       probe->elements<n||output->elements<n||anchor->elements<n||scratch->elements<257*2*H3_ADAPTIVE_REGIONS)
        return h3_gpu_set_error(g,"invalid adaptive continuation probe tensors");
    auto *partial=(float*)tensor_pointer(scratch);
    adaptive_continuation_probe_kernel<<<256,256,0,g->compute>>>((ushort*)tensor_pointer(probe),
        (const ushort*)tensor_pointer(output),(const ushort*)tensor_pointer(anchor),partial,n,columns,*p,ready!=0);
    adaptive_continuation_finish_kernel<<<H3_ADAPTIVE_REGIONS,256,0,g->compute>>>(partial);
    float sums[2*H3_ADAPTIVE_REGIONS];
    if(!launch_status(g,"adaptive continuation probe")||!checked(g,cudaMemcpyAsync(sums,partial+256*2*H3_ADAPTIVE_REGIONS,
        sizeof(sums),cudaMemcpyDeviceToHost,g->compute),"adaptive class readback",sizeof(sums))||
       !checked(g,cudaStreamSynchronize(g->compute),"adaptive continuation decision"))return 0;
    for(unsigned i=0;i<H3_ADAPTIVE_REGIONS;i++) {
        if(!std::isfinite(sums[2*i])||!std::isfinite(sums[2*i+1]))return h3_gpu_set_error(g,"nonfinite adaptive continuation probe");
        float count=(float)(size_t(p->counts[i])*columns);
        scores[i]=count?(sums[2*i]/count)/fmaxf(sums[2*i+1]/count,1e-6f):0;
        if(!std::isfinite(scores[i]))return h3_gpu_set_error(g,"nonfinite adaptive continuation score");
    }
    return 1;
}
