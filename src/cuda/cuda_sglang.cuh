/* Reference-only BF16 operation boundaries. No fast CUDA kernel is modified. */
__device__ inline float h3_sg_half(float x) {return __half2float(__float2half_rn(x));}
/* Pinned PyTorch vectorized LayerNorm Welford order (128 threads, vec4).
 * See licenses/PyTorch.txt. Explicit FMAs retain its arithmetic under this
 * translation unit's otherwise non-contracting reference build. */
struct h3_sg_welford { float mean,variance,count; };
/* PyTorch cf30153 GroupNorm / SharedReduceOps / block_reduce recipe.
 * Channels-first traversal, 512 threads, two descending warp reductions,
 * then separately rounded affine parameters. See licenses/PyTorch.txt. */
__device__ inline h3_sg_welford h3_sg_encoder_combine(h3_sg_welford a,h3_sg_welford b) {
    if(a.count==0)return b;if(b.count==0)return a;
    float count=a.count+b.count,delta=b.mean-a.mean,ratio=b.count/count;
    return {fmaf(delta,ratio,a.mean),fmaf((delta*delta)*a.count,ratio,a.variance+b.variance),count};
}
__global__ void h3_sg_encoder_norm(const float *input,const float *weight,const float *bias,float *out,vae_encoder_norm_args a) {
    unsigned group=blockIdx.x%a.groups,plane=blockIdx.x/a.groups,tid=threadIdx.x;
    unsigned spatial=a.height*a.width,cpg=a.channels/a.groups,count=spatial*cpg;
    __shared__ h3_sg_welford warps[16];__shared__ float mean,inverse;
    h3_sg_welford s={0,0,0};unsigned samples=0;
    for(unsigned j=tid;j<count;j+=blockDim.x){
        unsigned c=group*cpg+j/spatial,p=j%spatial;float x=input[((size_t)plane*spatial+p)*a.channels+c];
        float delta=x-s.mean;samples++;float next=s.mean+delta/(float)samples;
        s.variance=fmaf(delta,x-next,s.variance);s.mean=next;s.count=(float)samples;
    }
    for(int d=16;d;d>>=1){h3_sg_welford v={__shfl_down_sync(0xffffffff,s.mean,d),__shfl_down_sync(0xffffffff,s.variance,d),__shfl_down_sync(0xffffffff,s.count,d)};s=h3_sg_encoder_combine(s,v);}
    if(!(tid%32))warps[tid/32]=s;__syncthreads();
    if(tid<32){
        s=tid<blockDim.x/32?warps[tid]:h3_sg_welford{0,0,0};
        for(int d=16;d;d>>=1){h3_sg_welford v={__shfl_down_sync(0xffffffff,s.mean,d),__shfl_down_sync(0xffffffff,s.variance,d),__shfl_down_sync(0xffffffff,s.count,d)};s=h3_sg_encoder_combine(s,v);}
        if(!tid){mean=s.mean;inverse=rsqrtf(s.variance/s.count+a.epsilon);}
    }
    __syncthreads();
    for(unsigned j=tid;j<count;j+=blockDim.x){
        unsigned c=group*cpg+j%cpg,p=j/cpg;size_t at=((size_t)plane*spatial+p)*a.channels+c;
        float scale=inverse*weight[c],offset=fmaf(-scale,mean,bias[c]),value=fmaf(scale,input[at],offset);
        out[at]=value/(1.f+expf(-value));
    }
}
__device__ inline h3_sg_welford h3_sg_combine(h3_sg_welford b,h3_sg_welford a) {
    float count=a.count+b.count,delta=b.mean-a.mean;
    if(count==0)return {0,0,0};
    float reciprocal=1.f/count,na=a.count*reciprocal,nb=b.count*reciprocal;
    return {fmaf(na,a.mean,nb*b.mean),fmaf((delta*delta)*a.count,nb,a.variance+b.variance),count};
}
__global__ void h3_sg_vae_layer_norm(const float *input,const float *weight,const float *bias,float *out,norm_args a) {
    unsigned row=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    __shared__ h3_sg_welford sums[4];
    const float *x=input+(size_t)row*a.width;h3_sg_welford s={0,0,0};
    for(unsigned c=tid*4;c<a.width;c+=512)for(unsigned j=0;j<4&&c+j<a.width;j++) {
        float v=x[c+j],delta=v-s.mean;s.count+=1;
        s.mean=fmaf(delta,1.f/s.count,s.mean);s.variance=fmaf(delta,v-s.mean,s.variance);
    }
    for(int d=16;d;d>>=1) {
        h3_sg_welford other={__shfl_down_sync(0xffffffff,s.mean,d),__shfl_down_sync(0xffffffff,s.variance,d),__shfl_down_sync(0xffffffff,s.count,d)};
        s=h3_sg_combine(s,other);
    }
    if(!lane)sums[warp]=s;__syncthreads();
    if(tid<2)sums[tid]=h3_sg_combine(sums[tid],sums[tid+2]);__syncthreads();
    if(!tid)sums[0]=h3_sg_combine(sums[0],sums[1]);__syncthreads();
    float inverse=rsqrtf(sums[0].variance/float(a.width)+a.epsilon);
    for(unsigned c=tid;c<a.width;c+=128)out[(size_t)row*a.width+c]=fmaf(weight[c],inverse*(x[c]-sums[0].mean),bias[c]);
}
__global__ void h3_sg_vision_layer_norm(const ushort *input,const ushort *weight,const ushort *bias,ushort *out,norm_args a) {
    unsigned row=blockIdx.x,tid=threadIdx.x,lane=tid%32,warp=tid/32;
    __shared__ h3_sg_welford sums[4];
    const ushort *x=input+(size_t)row*a.width;h3_sg_welford s={0,0,0};
    for(unsigned c=tid*4;c<a.width;c+=512)for(unsigned j=0;j<4&&c+j<a.width;j++) {
        float v=h3_bf16_to_f32(x[c+j]),delta=v-s.mean;s.count+=1;
        s.mean=fmaf(delta,1.f/s.count,s.mean);s.variance=fmaf(delta,v-s.mean,s.variance);
    }
    for(int d=16;d;d>>=1) {
        h3_sg_welford other={__shfl_down_sync(0xffffffff,s.mean,d),__shfl_down_sync(0xffffffff,s.variance,d),__shfl_down_sync(0xffffffff,s.count,d)};
        s=h3_sg_combine(s,other);
    }
    if(!lane)sums[warp]=s;__syncthreads();
    if(tid<2)sums[tid]=h3_sg_combine(sums[tid],sums[tid+2]);__syncthreads();
    if(!tid)sums[0]=h3_sg_combine(sums[0],sums[1]);__syncthreads();
    float inverse=rsqrtf(sums[0].variance/float(a.width)+a.epsilon);
    for(unsigned c=tid;c<a.width;c+=128)out[(size_t)row*a.width+c]=h3_f32_to_bf16(fmaf(h3_bf16_to_f32(weight[c]),inverse*(h3_bf16_to_f32(x[c])-sums[0].mean),h3_bf16_to_f32(bias[c])));
}
__global__ void h3_sg_vae_norm(const float *input,const float *weight,float *out,norm_args a) {
    uint row=blockIdx.x,tid=threadIdx.x;__shared__ float warps[4];
    const float *x=input+(size_t)row*a.width;float sum=0;
    for(uint c=tid*4;c<a.width;c+=512)
        for(uint j=0;j<4&&c+j<a.width;j++)sum=fmaf(x[c+j],x[c+j],sum);
    for(int d=16;d;d>>=1)sum+=__shfl_down_sync(0xffffffff,sum,d);
    if(!(tid%32))warps[tid/32]=sum;__syncthreads();
    if(tid<2)warps[tid]+=warps[tid+2];__syncthreads();
    if(!tid)warps[0]+=warps[1];__syncthreads();
    float inverse=rsqrtf(warps[0]/float(a.width)+a.epsilon);
    for(uint c=tid;c<a.width;c+=128)out[(size_t)row*a.width+c]=(x[c]*inverse)*weight[c];
}
__global__ void h3_sg_vae_swiglu(const float *input,float *output,swiglu_args a) {
    uint c=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
    if(r>=a.rows||c>=a.width)return;
    size_t b=(size_t)r*a.width*2;float gate=input[b+c],up=input[b+a.width+c];
    output[(size_t)r*a.width+c]=h3_sg_half(h3_sg_half(gate/(1.0f+expf(-gate)))*up);
}
__device__ inline float h3_sg_round(float x) { return h3_bf16_to_f32(h3_f32_to_bf16(x)); }
__global__ void h3_sg_time_features(float *out,const float *times,uint rows) {
    uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=rows*256u)return;
    uint channel=i%256u;float frequency=expf((-logf(10000.0f)*(float)(channel%128u))/128.0f);
    float angle=times[i/256u]*frequency;
    out[i]=channel<128u?cosf(angle):sinf(angle);
}
__global__ void h3_sg_rope_trig(ushort *cosine,ushort *sine,const float *angles,uint count) {
    uint i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i<count){cosine[i]=h3_f32_to_bf16(cosf(angles[i]));sine[i]=h3_f32_to_bf16(sinf(angles[i]));}
}
__global__ void h3_sg_patch_store(const float *input,ushort *out,const uint *map,uint rows,uint width) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=(size_t)rows*width)return;
    size_t row=i/width,destination=map?map[row]:row;
    out[destination*width+i%width]=h3_f32_to_bf16(input[i]);
}
/* Qwen's HF-compatible RMSNorm rounds normalized activations before applying
 * the weight. DiT nn.RMSNorm has a different boundary, so this is text-only. */
__global__ void h3_sg_text_norm(const ushort *input,const ushort *weight,ushort *out,norm_args a) {
    uint row=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(row>=a.rows)return;
    float partial[4]={0,0,0,0};
    for(uint c=lane*4;c<a.width;c+=128)
        for(uint j=0;j<4&&c+j<a.width;j++){float x=h3_bf16_to_f32(input[(size_t)row*a.width+c+j]);partial[j]+=x*x;}
    float sum=((partial[0]+partial[1])+partial[2])+partial[3];
    for(int delta=16;delta;delta>>=1)sum+=__shfl_down_sync(0xffffffff,sum,delta);
    sum=__shfl_sync(0xffffffff,sum,0);
    float inverse=rsqrtf(sum*(1.0f/float(a.width))+a.epsilon);
    for(uint c=lane;c<a.width;c+=32)out[(size_t)row*a.width+c]=h3_f32_to_bf16(
        h3_sg_round(h3_bf16_to_f32(input[(size_t)row*a.width+c])*inverse)*h3_bf16_to_f32(weight[c]));
}
__global__ void h3_sg_text_head_norm(ushort *tensor,const ushort *weight,head_norm_args a) {
    uint item=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(item>=a.sequence*a.heads)return;
    size_t base=(size_t)item*a.head_dim;float partial[4]={0,0,0,0};
    /* PyTorch's contiguous FP32 mean uses four independent accumulators,
     * then descending shfl-down reduction. BF16 x*x is exact in FP32. */
    for(uint d=lane*4;d<a.head_dim;d+=128)
        for(uint j=0;j<4&&d+j<a.head_dim;j++){float x=h3_bf16_to_f32(tensor[base+d+j]);partial[j]+=x*x;}
    float sum=((partial[0]+partial[1])+partial[2])+partial[3];
    for(int delta=16;delta;delta>>=1)sum+=__shfl_down_sync(0xffffffff,sum,delta);
    sum=__shfl_sync(0xffffffff,sum,0);
    float inverse=rsqrtf(sum*(1.0f/float(a.head_dim))+a.epsilon);
    for(uint d=lane;d<a.head_dim;d+=32)tensor[base+d]=h3_f32_to_bf16(
        h3_sg_round(h3_bf16_to_f32(tensor[base+d])*inverse)*h3_bf16_to_f32(weight[d]));
}
__global__ void h3_sg_text_rope(ushort *query,ushort *key,const float *cosine,const float *sine,text_rope_inplace_args a) {
    uint row=blockIdx.x*blockDim.x+threadIdx.x,head=blockIdx.y*blockDim.y+threadIdx.y;
    if(row>=a.sequence)return;
    for(uint item=0;item<2;item++) {
        uint heads=item?a.kv_heads:a.query_heads;if(head>=heads)continue;
        ushort *x=(item?key:query)+((size_t)row*heads+head)*a.head_dim;
        for(uint d=0;d<a.head_dim/2;d++) {
            float first=h3_bf16_to_f32(x[d]),second=h3_bf16_to_f32(x[d+a.head_dim/2]);
            float c=h3_sg_round(cosine[(size_t)row*(a.head_dim/2)+d]);
            float s=h3_sg_round(sine[(size_t)row*(a.head_dim/2)+d]);
            x[d]=h3_f32_to_bf16(h3_sg_round(first*c)-h3_sg_round(second*s));
            x[d+a.head_dim/2]=h3_f32_to_bf16(h3_sg_round(second*c)+h3_sg_round(first*s));
        }
    }
}
__global__ void h3_sg_silu_mul(const ushort *gate,const ushort *up,ushort *out,uint count) {
    uint i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float x=h3_bf16_to_f32(gate[i]);
    out[i]=h3_f32_to_bf16(h3_sg_round(x/(1.0f+expf(-x)))*h3_bf16_to_f32(up[i]));
}
/* PyTorch fused RMSNorm uses 128 threads, four contiguous values per lane,
 * descending warp reductions and then a binary reduction of four warps.
 * Keep the reduction order separate from Qwen's explicit FP32 mean. */
__device__ float h3_sg_dit_inverse(const ushort *x,uint width,float epsilon,float *warps) {
    uint tid=threadIdx.x,lane=tid%32,warp=tid/32;float sum=0;
    for(uint c=tid*4;c<width;c+=512)
        for(uint j=0;j<4&&c+j<width;j++){float v=h3_bf16_to_f32(x[c+j]);sum=fmaf(v,v,sum);}
    for(int delta=16;delta;delta>>=1)sum+=__shfl_down_sync(0xffffffff,sum,delta);
    if(!lane)warps[warp]=sum;__syncthreads();
    if(tid<2)warps[tid]+=warps[tid+2];__syncthreads();
    if(!tid)warps[0]+=warps[1];__syncthreads();
    return rsqrtf(warps[0]/float(width)+epsilon);
}
__global__ void h3_sg_dit_norm(const ushort *input,const ushort *weight,ushort *out,norm_args a) {
    uint row=blockIdx.x;__shared__ float warps[4];
    const ushort*x=input+(size_t)row*a.width;
    float inverse=h3_sg_dit_inverse(x,a.width,a.epsilon,warps);
    for(uint c=threadIdx.x;c<a.width;c+=128)
        out[(size_t)row*a.width+c]=h3_f32_to_bf16((h3_bf16_to_f32(x[c])*inverse)*h3_bf16_to_f32(weight[c]));
}
__global__ void h3_sg_adaln_bf16(const ushort *input,const ushort *weight,
    const ushort *modulation,const uint *row_map,ushort *output,adaln_args args) {
    uint row=blockIdx.x,tid=threadIdx.x;
    __shared__ float warps[4];
    const ushort *x=input+(size_t)row*args.width;
    float inverse=h3_sg_dit_inverse(x,args.width,args.epsilon,warps);
    size_t base=(size_t)row_map[row]*args.slots*args.width;
    for(uint c=tid;c<args.width;c+=128) {
        float normalized=h3_sg_round(h3_bf16_to_f32(x[c])*inverse*h3_bf16_to_f32(weight[c]));
        float shift=h3_bf16_to_f32(modulation[base+args.shift_slot*args.width+c]);
        float scale=h3_sg_round(1.0f+h3_bf16_to_f32(modulation[base+args.scale_slot*args.width+c]));
        output[(size_t)row*args.width+c]=h3_f32_to_bf16(h3_sg_round(normalized*scale)+shift);
    }
}
__global__ void h3_sg_gate_bf16(const ushort *residual,const ushort *branch,
    const ushort *modulation,const uint *row_map,ushort *output,gate_args args) {
    uint c=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
    if(r>=args.rows||c>=args.width)return;
    size_t i=(size_t)r*args.width,base=(size_t)row_map[r]*args.slots*args.width;
    float gate=h3_bf16_to_f32(modulation[base+args.gate_slot*args.width+c]);
    output[i+c]=h3_f32_to_bf16(h3_bf16_to_f32(residual[i+c])+h3_sg_round(h3_bf16_to_f32(branch[i+c])*gate));
}
__global__ void h3_sg_swiglu_bf16(const ushort *input,ushort *output,swiglu_args args) {
    uint c=blockIdx.x*blockDim.x+threadIdx.x,r=blockIdx.y*blockDim.y+threadIdx.y;
    if(r>=args.rows||c>=args.width)return;
    size_t base=(size_t)r*args.width*2;
    float gate=h3_bf16_to_f32(input[base+c]),up=h3_bf16_to_f32(input[base+args.width+c]);
    output[(size_t)r*args.width+c]=h3_f32_to_bf16(h3_sg_round(gate/(1.0f+expf(-gate)))*up);
}
__global__ void h3_sg_qkv_rope(const ushort *input,const ushort *qw,const ushort *kw,
    const ushort *cosine,const ushort *sine,ushort *q,ushort *k,ushort *v,qkv_args a) {
    uint item=blockIdx.x*4+threadIdx.x/32,lane=threadIdx.x%32;
    if(item>=a.sequence*a.heads)return;
    uint row=item/a.heads,head=item%a.heads,dim=a.head_dim;
    size_t inner=(size_t)a.heads*dim,base=(size_t)row*3*inner;
    size_t qb=base+(a.grouped?head*3*dim:head*dim),kb=qb+(a.grouped?dim:inner),vb=kb+(a.grouped?dim:inner);
    float qs=0,ks=0;
    for(uint j=0;j<4;j++){uint d=lane*4+j;float x=h3_bf16_to_f32(input[qb+d]),y=h3_bf16_to_f32(input[kb+d]);qs=fmaf(x,x,qs);ks=fmaf(y,y,ks);}
    for(int delta=16;delta;delta>>=1){qs+=__shfl_xor_sync(0xffffffff,qs,delta);ks+=__shfl_xor_sync(0xffffffff,ks,delta);}
    /* The oracle rounds (x * inverse) before multiplying the norm weight.
     * Reassociating this into x * (inverse * weight) changes BF16 ties. */
    float qi=rsqrtf(qs/128.0f+a.epsilon),ki=rsqrtf(ks/128.0f+a.epsilon);
    for(uint j=0;j<4;j++) {
        uint d=lane*4+j;
        float x=h3_sg_round((h3_bf16_to_f32(input[qb+d])*qi)*h3_bf16_to_f32(qw[d]));
        float y=h3_sg_round((h3_bf16_to_f32(input[kb+d])*ki)*h3_bf16_to_f32(kw[d]));
        if(d<a.rope_half*2) {
            uint pair=d<a.rope_half?d+a.rope_half:d-a.rope_half,position=d%a.rope_half;
            float xp=h3_sg_round((h3_bf16_to_f32(input[qb+pair])*qi)*h3_bf16_to_f32(qw[pair]));
            float yp=h3_sg_round((h3_bf16_to_f32(input[kb+pair])*ki)*h3_bf16_to_f32(kw[pair]));
            float c=h3_bf16_to_f32(cosine[(size_t)row*a.rope_half+position]);
            float s=h3_bf16_to_f32(sine[(size_t)row*a.rope_half+position]);
            float sign=d<a.rope_half?-1.0f:1.0f;
            x=h3_sg_round(x*c)+sign*h3_sg_round(xp*s);
            y=h3_sg_round(y*c)+sign*h3_sg_round(yp*s);
        }
        size_t o=(size_t)item*dim+d;q[o]=h3_f32_to_bf16(x);k[o]=h3_f32_to_bf16(y);v[o]=input[vb+d];
    }
}

__global__ void h3_sg_vision_gelu(const ushort *in,ushort *out,gelu_bf16_args a) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=a.elements)return;
    float x=h3_bf16_to_f32(in[i]);
    float y=a.approximate?.5f*x*(1.f+tanhf(.7978845608028654f*(x+.044715f*x*x*x))):
        .5f*x*(1.f+erff(x*.7071067811865475f));
    out[i]=h3_f32_to_bf16(y);
}
/* Torch cf30153 / aten/native/cuda/WeightNorm.cu, dim=0 recipe. The
 * 256-lane reduction, sqrt/reciprocal and (g*v)*rnorm order are significant.
 * PyTorch attribution/license is retained in licenses/PyTorch.txt. */
__global__ void h3_sg_weight_norm(const float *v,const float *g,float *out,unsigned inner) {
    __shared__ float sums[256];unsigned lane=threadIdx.x;size_t base=(size_t)blockIdx.x*inner;
    float sum=0;for(unsigned j=lane;j<inner;j+=256){float x=v[base+j];sum=fmaf(x,x,sum);}
    sums[lane]=sum;__syncthreads();
    for(unsigned offset=128;offset>=64;offset/=2){if(lane<offset)sums[lane]+=sums[lane+offset];__syncthreads();}
    if(lane<32){float x=sums[lane]+sums[lane+32];for(int offset=16;offset;offset/=2)x+=__shfl_down_sync(0xffffffff,x,offset);if(!lane)sums[0]=x;}
    __syncthreads();float inverse=1.f/sqrtf(sums[0]),magnitude=g[blockIdx.x];
    for(unsigned j=lane;j<inner;j+=256)out[base+j]=(magnitude*v[base+j])*inverse;
}
__global__ void h3_sg_vision_rope(const ushort *qkv,const float *c,const float *s,ushort *q,ushort *k,ushort *v,qkv_args a) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,inner=(size_t)a.heads*a.head_dim;
    if(i>=(size_t)a.sequence*inner)return;
    size_t row=i/inner,component=i%inner,dim=component%a.head_dim,pair=dim<a.rope_half?component+a.rope_half:component-a.rope_half;
    size_t base=row*3*inner,p=row*a.rope_half+dim%a.rope_half;
    float sign=dim<a.rope_half?-1.f:1.f;
    q[i]=h3_f32_to_bf16(h3_bf16_to_f32(qkv[base+component])*c[p]+sign*h3_bf16_to_f32(qkv[base+pair])*s[p]);
    k[i]=h3_f32_to_bf16(h3_bf16_to_f32(qkv[base+inner+component])*c[p]+sign*h3_bf16_to_f32(qkv[base+inner+pair])*s[p]);
    v[i]=qkv[base+2*inner+component];
}
__global__ void h3_sg_vision_positions(ushort *out,const ushort *table,float *cosine,float *sine,unsigned height,unsigned width) {
    unsigned row=blockIdx.x,tid=threadIdx.x,block=row/4,iy=(block/(width/2))*2+(row/2)%2,ix=(block%(width/2))*2+row%2;
    float ys=47.f/(float)(height-1),xs=47.f/(float)(width-1);
    float y=iy<height/2?ys*iy:fmaf(-ys,(float)(height-iy-1),47.f),x=ix<width/2?xs*ix:fmaf(-xs,(float)(width-ix-1),47.f);
    unsigned y0=(unsigned)y,x0=(unsigned)x,y1=min(y0+1,47u),x1=min(x0+1,47u);
    float dy=y-y0,dx=x-x0,coeff[4]={(1.f-dy)*(1.f-dx),(1.f-dy)*dx,dy*(1.f-dx),dy*dx};
    unsigned idx[4]={y0*48+x0,y0*48+x1,y1*48+x0,y1*48+x1};
    for(unsigned c=tid;c<1152;c+=blockDim.x){float v[4];for(int j=0;j<4;j++)v[j]=h3_bf16_to_f32(table[idx[j]*1152+c])*coeff[j];out[(size_t)row*1152+c]=h3_f32_to_bf16(((v[0]+v[1])+v[2])+v[3]);}
    if(tid<36){
        // Qwen initializes this nonpersistent buffer on the CPU before moving
        // it to CUDA. These FP32 constants are the pinned CPU evaluation of
        // 1 / 10000**(arange(0,36,2,float32)/36), independent of input pixels,
        // seed and grid size. GPU pow produces different rounding at BF16 ties.
        constexpr unsigned frequency_bits[18]={0x3f800000u,0x3f1977ccu,0x3eb800d6u,0x3e5c9d35u,
            0x3e044133u,0x3d9e91b6u,0x3d3e1e95u,0x3ce3f280u,0x3c88a69bu,0x3c23d70au,
            0x3bc47060u,0x3b6b8631u,0x3b0d3169u,0x3aa94938u,0x3a4af7f3u,0x39f35a5cu,
            0x3991e2e1u,0x392ee9bfu};
        float inverse=__uint_as_float(frequency_bits[tid%18]),angle=(tid<18?iy:ix)*inverse;
        cosine[(size_t)row*36+tid]=cosf(angle);sine[(size_t)row*36+tid]=sinf(angle);}
}

/* Pinned soundtrack encoder Snake1d is unfused: reciprocal, square,
 * multiply, then add. Dividing the square by alpha changes FP32 rounding. */
__global__ void h3_sg_snake1d_f32(const float *input,const float *alpha,float *output,
                                audio_activation_args args) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=args.batch*args.length*args.channels)return;
    float a=alpha[i%args.channels],x=input[i],wave=sinf(a*x);
    float inverse=__fdiv_rn(1.f,__fadd_rn(a,1e-9f));
    output[i]=__fadd_rn(x,__fmul_rn(inverse,__fmul_rn(wave,wave)));
}

__global__ void h3_sg_audio_math_pack(float *packed,const float *q,const float *k,const float *v,
                                    unsigned seq,unsigned heads,unsigned dim,size_t count,float scale) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned d=i%dim,t=(i/dim)%seq,h=(i/dim/seq)%heads;size_t b=i/dim/seq/heads;
    size_t source=((b*seq+t)*heads+h)*dim+d;
    packed[i]=q[source]*scale;packed[count+i]=k[source]*scale;packed[count*2+i]=v[source];
}
__global__ void h3_sg_audio_math_softmax(float *scores,unsigned seq) {
    unsigned row=blockIdx.x,lane=threadIdx.x;float maximum=-INFINITY;
    float *x=scores+(size_t)row*seq;
    for(unsigned i=lane;i<seq;i+=32){if(i>row%seq)x[i]=-INFINITY;maximum=fmaxf(maximum,x[i]);}
    for(int d=16;d;d>>=1)maximum=fmaxf(maximum,__shfl_xor_sync(0xffffffff,maximum,d));
    float sum=0;for(unsigned i=lane;i<seq;i+=32){x[i]=expf(x[i]-maximum);sum+=x[i];}
    for(int d=16;d;d>>=1)sum+=__shfl_xor_sync(0xffffffff,sum,d);
    /* The pinned persistent softmax divides each exponential; multiplying by
     * a rounded reciprocal crosses FP32 ties before the audio patch projection. */
    for(unsigned i=lane;i<seq;i+=32)x[i]/=sum;
}
__global__ void h3_sg_audio_math_unpack(float *out,const float *packed,
                                      unsigned seq,unsigned heads,unsigned dim,size_t count) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    unsigned d=i%dim,h=(i/dim)%heads,t=(i/dim/heads)%seq;size_t b=i/dim/heads/seq;
    out[i]=packed[((b*heads+h)*seq+t)*dim+d];
}
__global__ void h3_sg_audio_pool(const float *in,float *out,audio_pool_args a) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;
    if(i>=a.batch*a.length*a.output_dim)return;
    unsigned row=i/a.output_dim,col=i%a.output_dim,pool=a.head_dim/a.output_dim;
    float total=0;
    for(unsigned d=0;d<pool;d++) {
        float h[8];for(unsigned j=0;j<8;j++)h[j]=in[((size_t)row*8+j)*a.head_dim+col*pool+d];
        /* Four independent accumulators, then sequential combination, as in
         * the pinned ReduceOp<float, MeanOps, ..., 4, 4> head reduction. */
        float mean=((((h[0]+h[4])+(h[1]+h[5]))+(h[2]+h[6]))+(h[3]+h[7]))*.125f;
        total+=mean;
    }
    out[i]=total/(float)pool;
}
__global__ void h3_sg_audio_geglu(const float *gate,const float *linear,float *out,unsigned count) {
    unsigned i=blockIdx.x*blockDim.x+threadIdx.x;if(i>=count)return;
    float x=gate[i],cube=(x*x)*x;
    float gelu=(.5f*x)*(1.f+tanhf(.7978845608028654f*fmaf(.044715f,cube,x)));
    out[i]=gelu*linear[i];
}

/* Keep the oracle reciprocal/multiply and resampling rounding boundaries. */
__global__ void h3_sg_alias_free_snake_f32(const float *input, const float *alpha_log, const float *beta_log, const float *upsample_filter, const float *downsample_filter, float *output, audio_activation_args args, uint time_base, bool fused) {
    uint3 gid = make_uint3((blockIdx.x*blockDim.x+threadIdx.x),(time_base+blockIdx.y*blockDim.y+threadIdx.y),(blockIdx.z*blockDim.z+threadIdx.z));

    uint channel = gid.x;
    uint time = gid.y;
    uint batch = gid.z;
    if (channel >= args.channels || time >= args.length ||
        batch >= args.batch) return;
    float alpha = expf(alpha_log[channel]);
    float beta = expf(beta_log[channel]);
    float result = 0.0f;
    for (int down_k = 0; down_k < 12; down_k++) {
        int up_time = int(time * 2) + down_k - 5;
        up_time = clamp(up_time, 0, int(args.length * 2) - 1);
        int raw_time = up_time + 15;
        float upsampled = 0.0f;
        for (int up_k = 0; up_k < 12; up_k++) {
            int numerator = raw_time - up_k;
            if (numerator < 0 || (numerator & 1)) continue;
            int padded_time = numerator / 2;
            int source_time = clamp(padded_time - 5, 0,
                                    int(args.length) - 1);
            uint source = (batch * args.length + uint(source_time)) *
                          args.channels + channel;
            upsampled = fmaf(input[source], upsample_filter[up_k],
                            upsampled);
        }
        upsampled *= 2.f;
        float sine = sinf(alpha * upsampled);
        float inverse=1.f/(beta+1e-9f),square=sine*sine;
        float activated = fused?fmaf(inverse,square,upsampled):upsampled+inverse*square;
        result = fmaf(activated, downsample_filter[down_k], result);
    }
    uint destination = (batch * args.length + time) * args.channels + channel;
    output[destination] = result;

}
