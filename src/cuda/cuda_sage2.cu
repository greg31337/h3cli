#include "src/cuda/cuda_sage_internal.cuh"
#include <cuda_fp8.h>
#include "third_party/sageattention/csrc/qattn/qk_int_sv_f8_cuda_sm89.cuh"

/* Match the pinned Triton quotient, including its approximate full-range
 * division. The constant scale division is fused explicitly below. */
__device__ __forceinline__ float sage2_div_full(float x,float y) {
    float result;asm("div.full.f32 %0, %1, %2;" : "=f"(result) : "f"(x),"f"(y));return result;
}
/* Native equivalent of upstream quant_per_thread.py: Q groups contain four
 * rows, spaced by eight within 32; K groups contain pairs within 64 rows.
 * K subtraction rounds to BF16 before INT8 quantization, like the oracle. */
template<bool Key>
static __global__ void pack_int8(const nv_bfloat16 *input,int8_t *output,float *scales,
    const nv_bfloat16 *mean,unsigned seq,unsigned heads,unsigned first,unsigned padded,unsigned *fault) {
    constexpr unsigned Groups=Key?4:8,Rows=Key?16:4,Block=Key?64:32;
    unsigned h=blockIdx.y,group=blockIdx.x,part=group%Groups,base=group/Groups*Block;
    float values[Rows/2];float maximum=0;
    unsigned d=threadIdx.x%128,half=threadIdx.x/128;
    for(unsigned i=0;i<Rows/2;i++){
        unsigned r=i*2+half;
        unsigned row=base+(Key?(r/2*8+part*2+r%2):(r*8+part));
        float x=row<seq?__bfloat162float(input[((size_t)row*heads+first+h)*128+d]):0;
        if(!isfinite(x)){atomicExch(fault,1u);x=0;}
        if(Key&&row<seq)x=__bfloat162float(__float2bfloat16_rn(x-__bfloat162float(mean[h*128+d])));
        values[i]=x;maximum=fmaxf(maximum,fabsf(x));
    }
    __shared__ float reduction[256];reduction[threadIdx.x]=maximum;__syncthreads();
    for(unsigned offset=128;offset;offset>>=1){if(threadIdx.x<offset)reduction[threadIdx.x]=fmaxf(reduction[threadIdx.x],reduction[threadIdx.x+offset]);__syncthreads();}
    // Pinned ptxas fuses max / 127 + epsilon into this FMA.
    float scale=fmaf(reduction[0],1.0f/127.0f,0.0000001f);
    if(threadIdx.x==0)scales[(size_t)h*gridDim.x+group]=scale;
    for(unsigned i=0;i<Rows/2;i++){
        unsigned r=i*2+half;
        unsigned row=base+(Key?(r/2*8+part*2+r%2):(r*8+part));
        if(row<padded){float x=sage2_div_full(values[i],scale);output[((size_t)h*padded+row)*128+d]=(int8_t)(x+(x>=0?0.5f:-0.5f));}
    }
}
static __global__ void pack_v_fp8(const nv_bfloat16 *v,int8_t *out,const float *maxima,
    unsigned seq,unsigned heads,unsigned first,unsigned padded,unsigned group,unsigned partials,float scale_max) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x,total=(size_t)group*padded*128;
    if(i>=total)return;
    unsigned d=i%128,row=(i/128)%padded,h=i/((size_t)padded*128);
    // Inverse of upstream transpose_pad_permute's 16-token permutation.
    unsigned local=row%16,permuted=(row/16)*16+(local%8/2)*4+(local/8)*2+local%2;
    float x=row<seq?__bfloat162float(v[((size_t)row*heads+first+h)*128+d]):0;
    // Upstream divides by the original maximum, not the rounded stored scale.
    float maximum=maxima[(size_t)h*partials*128+d];
    __nv_fp8_e4m3 packed(maximum>0?x*(scale_max/maximum):0.0f);
    out[((size_t)h*128+d)*padded+permuted]=(int8_t)packed.__x;
}
void sage2_run(sage_context &s,nv_bfloat16 *out,const nv_bfloat16 *q,const nv_bfloat16 *k,
    const nv_bfloat16 *v,unsigned seq,unsigned heads,float scale,bool head_major) {
    unsigned qp=(seq+127)/128*128,kp=(seq+63)/64*64,tiles=sage_reduce_partials(seq);
    size_t per_head=(size_t)(qp+2*kp)*128+((size_t)qp/4+kp/16+128)*4+128*2+(size_t)tiles*128*8+4096;
    unsigned group_max=(unsigned)std::min<size_t>(heads,(s.capacity-sage_reserved_bytes)/per_head);
    if(!group_max)throw std::runtime_error("Sage2 shape exceeds workspace admission");
    auto kernel=qk_int_sv_f8_attn_kernel<128,64,32,64,128,DataType::kInt8,
        QuantGranularity::kPerThread,QuantGranularity::kPerThread,float,true,nv_bfloat16,
        ComputeUnit::kCudaCore,MaskMode::kNone,false,true,false,true>;
    constexpr size_t shared=32768;
    sage_check(cudaFuncSetAttribute(kernel,cudaFuncAttributeMaxDynamicSharedMemorySize,shared));
    for(unsigned first=0;first<heads;first+=group_max){
        unsigned group=std::min(group_max,heads-first);s.offset=sage_reserved_bytes;
        auto qi=s.alloc<int8_t>((size_t)group*qp*128),ki=s.alloc<int8_t>((size_t)group*kp*128),vi=s.alloc<int8_t>((size_t)group*kp*128);
        auto qs=s.alloc<float>((size_t)group*(qp/4)),ks=s.alloc<float>((size_t)group*(kp/16)),vs=s.alloc<float>(group*128);
        auto means=s.alloc<nv_bfloat16>(group*128);
        auto sums=s.alloc<float>((size_t)group*tiles*128),maxima=s.alloc<float>((size_t)group*tiles*128);
        {sage_timer t(s,0);sage_reduce(s,k,v,sums,maxima,means,vs,seq,heads,first,group);}
        {sage_timer t(s,1);
            pack_int8<false><<<dim3(qp/4,group),256,0,s.stream>>>(q,qi,qs,means,seq,heads,first,qp,s.fault());
            pack_int8<true><<<dim3(kp/16,group),256,0,s.stream>>>(k,ki,ks,means,seq,heads,first,kp,s.fault());
            size_t n=(size_t)group*kp*128;
            pack_v_fp8<<<(n+255)/256,256,0,s.stream>>>(v,vi,maxima,seq,heads,first,kp,group,tiles,2.25f);
            sage_check(cudaGetLastError());
        }
        {sage_timer t(s,3);
            if(first==0) {
                sage_debug_dump(s,"k-mean.bf16",means,(size_t)group*128*2,seq);
                sage_debug_dump(s,"q.int8",qi,(size_t)group*qp*128,seq);
                sage_debug_dump(s,"k.int8",ki,(size_t)group*kp*128,seq);
                sage_debug_dump(s,"v.fp8",vi,(size_t)group*kp*128,seq);
                sage_debug_dump(s,"q-scale.f32",qs,(size_t)group*(qp/4)*4,seq);
                sage_debug_dump(s,"k-scale.f32",ks,(size_t)group*(kp/16)*4,seq);
                sage_debug_dump(s,"v-scale.f32",vs,(size_t)group*128*4,seq);
            }
            nv_bfloat16 *o=out+(head_major?(size_t)first*seq*128:first*128);
            kernel<<<dim3((seq+127)/128,group),dim3(32,4),shared,s.stream>>>(qi,ki,vi,o,nullptr,qs,ks,vs,nullptr,
                seq,seq,1,0,128,qp*128,0,128,kp*128,0,kp*128,kp,
                0,head_major?128:heads*128,head_major?seq*128:128,scale);
            sage_check(cudaGetLastError());
        }
        s.stats.head_groups++;
    }
}
