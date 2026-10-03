#include "src/cuda/cuda_sage_internal.cuh"
#include "third_party/sageattention/sageattention3_blackwell/sageattn3/blackwell/launch.h"
#include "third_party/sageattention/sageattention3_blackwell/sageattn3/quantization/fp4_quantization_4d.cu"

static __global__ void center_k(const nv_bfloat16 *k,nv_bfloat16 *out,const nv_bfloat16 *mean,
    unsigned seq,unsigned heads,unsigned first,unsigned padded,unsigned group) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)group*padded*128)return;
    unsigned d=i%128,row=i/128%padded,h=i/((size_t)padded*128);
    out[i]=__float2bfloat16_rn(row<seq?__bfloat162float(k[((size_t)row*heads+first+h)*128+d])-__bfloat162float(mean[h*128+d]):0);
}
static __global__ void center_q(const nv_bfloat16 *q,nv_bfloat16 *out,nv_bfloat16 *means,
    unsigned seq,unsigned heads,unsigned first,unsigned offset,unsigned count,unsigned *fault) {
    unsigned block=blockIdx.x,h=blockIdx.y,d=threadIdx.x;
    // Pinned Triton SM120 layout: eight row lanes, each accumulating sixteen
    // strided rows in BF16, then lane-pair and 4/2-row warp reductions.
    // A float mean changes FP4 bins and is not the upstream arithmetic recipe.
    nv_bfloat16 partial[8];
    #pragma unroll
    for(unsigned lane=0;lane<8;lane++)partial[lane]=__float2bfloat16_rn(0);
    for(unsigned i=0;i<16;i++){
        #pragma unroll
        for(unsigned lane=0;lane<8;lane++) {
            unsigned row=offset+block*128+i*8+lane;
            float x=row<seq?__bfloat162float(q[((size_t)row*heads+first+h)*128+d]):0;
            if(!isfinite(x)){atomicExch(fault,1u);x=0;}
            partial[lane]=__hadd(partial[lane],__float2bfloat16_rn(x));
        }
    }
    nv_bfloat16 sum=__hadd(__hadd(__hadd(partial[0],partial[1]),__hadd(partial[4],partial[5])),
                         __hadd(__hadd(partial[2],partial[3]),__hadd(partial[6],partial[7])));
    float mean=__bfloat162float(sum)/128.0f;means[((size_t)h*(count/128)+block)*128+d]=__float2bfloat16_rn(mean);
    for(unsigned i=0;i<128;i++) {
        unsigned row=offset+block*128+i;
        float x=row<seq?__bfloat162float(q[((size_t)row*heads+first+h)*128+d]):0;
        out[((size_t)h*count+block*128+i)*128+d]=__float2bfloat16_rn(x-mean);
    }
}
static __global__ void round_correction(float *delta,size_t n) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;
    // torch.matmul(BF16, BF16).to(float32) rounds its output to BF16 first.
    if(i<n)delta[i]=__bfloat162float(__float2bfloat16_rn(delta[i]));
}
static __global__ void scatter_output(const nv_bfloat16 *input,nv_bfloat16 *out,
    unsigned seq,unsigned heads,unsigned first,unsigned offset,unsigned count,unsigned group,bool head_major) {
    size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;if(i>=(size_t)group*count*128)return;
    unsigned d=i%128,row=i/128%count+offset,h=i/((size_t)count*128)+first;
    if(row<seq)out[head_major?((size_t)h*seq+row)*128+d:((size_t)row*heads+h)*128+d]=input[i];
}
static void pack_qk(sage_context &s,const nv_bfloat16 *input,uint8_t *packed,uint8_t *scales,unsigned rows,unsigned group,bool key) {
    dim3 grid(rows/128,1,group),block(1024);
    if(key)scaled_fp4_quant_kernel<128,128,true,nv_bfloat16><<<grid,block,0,s.stream>>>(input,packed,scales,1,group,rows,0,rows*128,128,0,rows*64,64,0,rows*8,8);
    else scaled_fp4_quant_kernel<128,128,false,nv_bfloat16><<<grid,block,0,s.stream>>>(input,packed,scales,1,group,rows,0,rows*128,128,0,rows*64,64,0,rows*8,8);
    sage_check(cudaGetLastError());
}
void sage3_run(sage_context &s,nv_bfloat16 *out,const nv_bfloat16 *q,const nv_bfloat16 *k,
    const nv_bfloat16 *v,unsigned seq,unsigned heads,float scale,bool head_major) {
    unsigned kp=(seq+127)/128*128,tiles=sage_reduce_partials(seq);
    size_t fixed=(size_t)kp*400+(size_t)tiles*128*8+8192;
    size_t available=s.capacity-sage_reserved_bytes;
    if(fixed>=available)throw std::runtime_error("Sage3 K/V exceeds workspace admission");
    // Favor complete query ranges per head. Larger shapes still use bounded
    // 128-aligned query chunks, determined solely by shape and fixed budget.
    size_t blocks=std::min<size_t>(kp/128,(available-fixed)/(128*600+(size_t)kp*4));
    if(!blocks)throw std::runtime_error("Sage3 correction exceeds workspace admission");
    unsigned chunk=(unsigned)blocks*128;
    // Includes K BF16, K/V FP4+scales, Q center/mean/pack, FP32 correction,
    // padded output/LSE and reductions. Alignment slack is reserved per head.
    size_t per_head=(size_t)kp*400+(size_t)chunk*600+(size_t)(chunk/128)*kp*4+(size_t)tiles*128*8+8192;
    unsigned group_max=(unsigned)std::min<size_t>(heads,(s.capacity-sage_reserved_bytes)/per_head);
    if(!group_max)throw std::runtime_error("Sage3 shape exceeds workspace admission");
    for(unsigned first=0;first<heads;first+=group_max){
        unsigned group=std::min(group_max,heads-first);s.offset=sage_reserved_bytes;
        auto km=s.alloc<nv_bfloat16>(group*128);auto vs=s.alloc<float>(group*128);
        auto sums=s.alloc<float>((size_t)group*tiles*128),maxima=s.alloc<float>((size_t)group*tiles*128);
        auto centered_k=s.alloc<nv_bfloat16>((size_t)group*kp*128);
        auto ki=s.alloc<uint8_t>((size_t)group*kp*64),vi=s.alloc<uint8_t>((size_t)group*kp*64);
        auto ks=s.alloc<uint8_t>((size_t)group*kp*8),vsc=s.alloc<uint8_t>((size_t)group*kp*8);
        {sage_timer t(s,0);sage_reduce(s,k,v,sums,maxima,km,vs,seq,heads,first,group);}
        {sage_timer t(s,1);
            size_t n=(size_t)group*kp*128;
            center_k<<<(n+255)/256,256,0,s.stream>>>(k,centered_k,km,seq,heads,first,kp,group);
            pack_qk(s,centered_k,ki,ks,kp,group,true);
            scaled_fp4_quant_trans_kernel<128,128,nv_bfloat16><<<dim3(kp/128,1,group),1024,0,s.stream>>>(
                v+first*128,vi,vsc,1,group,seq,0,128,heads*128,0,kp*64,kp/2,0,kp*8,kp/16);
            sage_check(cudaGetLastError());
        }
        size_t persistent=s.offset;
        for(unsigned offset=0;offset<kp;offset+=chunk){
            unsigned count=std::min(chunk,kp-offset),blocks=count/128;s.offset=persistent;
            auto centered_q=s.alloc<nv_bfloat16>((size_t)group*count*128),qm=s.alloc<nv_bfloat16>((size_t)group*blocks*128);
            auto qi=s.alloc<uint8_t>((size_t)group*count*64),qs=s.alloc<uint8_t>((size_t)group*count*8);
            auto delta=s.alloc<float>((size_t)group*blocks*kp);
            auto result=s.alloc<nv_bfloat16>((size_t)group*count*128);auto lse=s.alloc<float>((size_t)group*count);
            {sage_timer t(s,1);
                center_q<<<dim3(blocks,group),128,0,s.stream>>>(q,centered_q,qm,seq,heads,first,offset,count,s.fault());
                pack_qk(s,centered_q,qi,qs,count,group,false);
            }
            {sage_timer t(s,2);
                float one=1,zero=0;
                auto status=cublasGemmStridedBatchedEx(s.blas,CUBLAS_OP_T,CUBLAS_OP_N,kp,blocks,128,
                    &one,centered_k,CUDA_R_16BF,128,(long long)kp*128,qm,CUDA_R_16BF,128,(long long)blocks*128,
                    &zero,delta,CUDA_R_32F,kp,(long long)blocks*kp,group,CUBLAS_COMPUTE_32F,CUBLAS_GEMM_DEFAULT);
                if(status!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Sage3 bounded correction GEMM failed");
                size_t n=(size_t)group*blocks*kp;round_correction<<<(n+255)/256,256,0,s.stream>>>(delta,n);sage_check(cudaGetLastError());
            }
            {sage_timer t(s,3);
                if(first==0&&offset==0) {
                    sage_debug_dump(s,"k-center.bf16",centered_k,(size_t)group*kp*128*2,seq);
                    sage_debug_dump(s,"q-center.bf16",centered_q,(size_t)group*count*128*2,seq);
                    sage_debug_dump(s,"q-mean.bf16",qm,(size_t)group*blocks*128*2,seq);
                    sage_debug_dump(s,"delta.f32",delta,(size_t)group*blocks*kp*4,seq);
                    sage_debug_dump(s,"q.fp4",qi,(size_t)group*count*64,seq);
                    sage_debug_dump(s,"k.fp4",ki,(size_t)group*kp*64,seq);
                    sage_debug_dump(s,"v.fp4",vi,(size_t)group*kp*64,seq);
                    sage_debug_dump(s,"q.scales",qs,(size_t)group*count*8,seq);
                    sage_debug_dump(s,"k.scales",ks,(size_t)group*kp*8,seq);
                    sage_debug_dump(s,"v.scales",vsc,(size_t)group*kp*8,seq);
                }
                Flash_fwd_params p={};p.q_ptr=qi;p.k_ptr=ki;p.v_ptr=vi;p.sfq_ptr=qs;p.sfk_ptr=ks;p.sfv_ptr=vsc;p.delta_s_ptr=delta;p.o_ptr=result;p.softmax_lse_ptr=lse;
                p.b=1;p.h=p.h_k=group;p.h_h_k_ratio=1;p.d=p.d_rounded=128;
                p.seqlen_q=p.seqlen_q_rounded=p.seqlen_s=count;p.seqlen_k=p.seqlen_k_rounded=kp;p.unpadded_seqlen_k=seq;
                p.q_row_stride=p.k_row_stride=p.o_row_stride=128;p.v_row_stride=kp;
                p.q_head_stride=p.o_head_stride=(int64_t)count*128;p.k_head_stride=p.v_head_stride=(int64_t)kp*128;
                p.q_batch_stride=p.o_batch_stride=p.q_head_stride*group;p.k_batch_stride=p.v_batch_stride=p.k_head_stride*group;
                p.ds_row_stride=kp;p.ds_head_stride=(int64_t)blocks*kp;p.ds_batch_stride=p.ds_head_stride*group;
                p.scale_softmax=scale;p.scale_softmax_log2=scale*float(M_LOG2E);p.per_block_mean=true;p.is_bf16=true;
                p.head_divmod=cutlass::FastDivmod(group);
                using T=cutlass::nv_float4_t<cutlass::float_e2m1_t>;
                run_flash_fwd<Flash_fwd_kernel_traits<128,128,128,3,1,true,T,cutlass::bfloat16_t>,false>(p,s.stream);
            }
            {sage_timer t(s,4);size_t n=(size_t)group*count*128;
                scatter_output<<<(n+255)/256,256,0,s.stream>>>(result,out,seq,heads,first,offset,count,group,head_major);sage_check(cudaGetLastError());
            }
        }
        s.stats.head_groups++;
    }
}
