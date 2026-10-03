/* Native inference launch for the exact FlashAttention revision used by Torch. */
#include "src/cuda/cuda_sglang_flash.h"
#include <cstdio>
#include <cmath>
#define FLASH_NAMESPACE h3_reference_flash
/* PyTorch sets this deliberately: fused scale/subtract changes BF16 outputs. */
#define UNFUSE_FMA
#define FLASHATTENTION_DISABLE_DROPOUT
#define FLASHATTENTION_DISABLE_ALIBI
#define FLASHATTENTION_DISABLE_SOFTCAP
#define FLASHATTENTION_DISABLE_LOCAL
#define FLASHATTENTION_DISABLE_BACKWARD
#include "third_party/flash-attention/flash_fwd_launch_template.h"
static int reference_flash(void *out,const void *q,const void *k,const void *v,float *lse,
    unsigned batch,unsigned sequence,unsigned heads,unsigned kv_heads,unsigned dim,
    float scale,int causal,int head_major_output,cudaStream_t stream,char *error,size_t size,bool bf16) {
    if(!out||!q||!k||!v||!lse||!batch||!sequence||sequence>1000000||!heads||!kv_heads||heads%kv_heads||
       (bf16?(dim!=128&&dim!=72):dim!=64)||(causal&&dim!=128)||!std::isfinite(scale)||scale<=0)return 0;
    h3_reference_flash::Flash_fwd_params p={};
    p.q_ptr=const_cast<void*>(q);p.k_ptr=const_cast<void*>(k);p.v_ptr=const_cast<void*>(v);p.o_ptr=out;
    p.b=batch;p.h=heads;p.h_k=kv_heads;p.h_h_k_ratio=heads/kv_heads;
    p.seqlen_q=p.seqlen_k=sequence;p.seqlen_q_rounded=p.seqlen_k_rounded=(sequence+127u)/128u*128u;
    p.d=dim;p.d_rounded=(dim+31u)/32u*32u;p.is_bf16=bf16;p.softmax_lse_ptr=lse;
    p.q_row_stride=(int64_t)heads*dim;p.k_row_stride=p.v_row_stride=(int64_t)kv_heads*dim;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=dim;
    p.q_batch_stride=(int64_t)sequence*heads*dim;p.k_batch_stride=p.v_batch_stride=(int64_t)sequence*kv_heads*dim;
    p.o_batch_stride=p.q_batch_stride;p.o_row_stride=head_major_output?dim:p.q_row_stride;
    p.o_head_stride=head_major_output?(int64_t)sequence*dim:dim;
    p.scale_softmax=scale;p.scale_softmax_log2=scale*M_LOG2E;
    p.p_dropout=p.rp_dropout=1.f;p.p_dropout_in_uint8_t=255;p.scale_softmax_rp_dropout=scale;
    p.is_causal=causal!=0;p.window_size_left=causal?(int)sequence:-1;p.window_size_right=causal?0:-1;
    p.is_seqlens_k_cumulative=true;
    try {
        if(!bf16)h3_reference_flash::run_mha_fwd_hdim64<cutlass::half_t,false>(p,stream);
        else if(dim==72&&!causal)h3_reference_flash::run_mha_fwd_hdim96<cutlass::bfloat16_t,false>(p,stream);
        else if(causal)h3_reference_flash::run_mha_fwd_hdim128<cutlass::bfloat16_t,true>(p,stream);
        else h3_reference_flash::run_mha_fwd_hdim128<cutlass::bfloat16_t,false>(p,stream);
    } catch(const std::exception &e) {if(error&&size)snprintf(error,size,"native reference FlashAttention: %s",e.what());return -1;}
    return 1;
}
extern "C" int h3_sglang_flash(void *out,const void *q,const void *k,const void *v,float *lse,
    unsigned batch,unsigned sequence,unsigned heads,unsigned kv_heads,unsigned dim,
    float scale,int causal,int head_major_output,cudaStream_t stream,char *error,size_t size) {
    return reference_flash(out,q,k,v,lse,batch,sequence,heads,kv_heads,dim,scale,causal,head_major_output,stream,error,size,true);
}
extern "C" int h3_sglang_flash_vae(void *out,const void *q,const void *k,const void *v,float *lse,
    unsigned sequence,unsigned heads,float scale,cudaStream_t stream,char *error,size_t size) {
    return reference_flash(out,q,k,v,lse,1,sequence,heads,heads,64,scale,0,0,stream,error,size,false);
}
extern "C" int h3_sglang_flash_query_range(void *out,const void *q,const void *k,const void *v,float *lse,
    unsigned sequence,unsigned heads,unsigned first,unsigned count,float scale,
    int head_major_output,cudaStream_t stream,char *error,size_t size) {
    if(!out||!q||!k||!v||!lse||!sequence||sequence>1000000||!heads||!count||first>=sequence||count>sequence-first||
       !std::isfinite(scale)||scale<=0||(head_major_output!=0&&head_major_output!=1))return 0;
    h3_reference_flash::Flash_fwd_params p={};
    p.q_ptr=(char*)const_cast<void*>(q)+(size_t)first*heads*128*2;
    p.k_ptr=const_cast<void*>(k);p.v_ptr=const_cast<void*>(v);
    p.o_ptr=(char*)out+(size_t)first*(head_major_output?128:heads*128)*2;
    p.b=1;p.h=heads;p.h_k=heads;p.h_h_k_ratio=1;
    p.seqlen_q=count;p.seqlen_k=sequence;
    p.seqlen_q_rounded=(count+127u)/128u*128u;p.seqlen_k_rounded=(sequence+127u)/128u*128u;
    p.d=p.d_rounded=128;p.is_bf16=true;p.softmax_lse_ptr=lse;
    p.q_row_stride=p.k_row_stride=p.v_row_stride=(int64_t)heads*128;
    p.q_head_stride=p.k_head_stride=p.v_head_stride=128;
    p.q_batch_stride=p.k_batch_stride=p.v_batch_stride=(int64_t)sequence*heads*128;
    p.o_batch_stride=p.q_batch_stride;p.o_row_stride=head_major_output?128:p.q_row_stride;
    p.o_head_stride=head_major_output?(int64_t)sequence*128:128;
    p.scale_softmax=scale;p.scale_softmax_log2=scale*M_LOG2E;
    p.p_dropout=p.rp_dropout=1.f;p.p_dropout_in_uint8_t=255;p.scale_softmax_rp_dropout=scale;
    p.is_causal=false;p.window_size_left=p.window_size_right=-1;p.is_seqlens_k_cumulative=true;
    try {h3_reference_flash::run_mha_fwd_hdim128<cutlass::bfloat16_t,false>(p,stream);}
    catch(const std::exception &x){if(error&&size)snprintf(error,size,"native reference query slice: %s",x.what());return -1;}
    return 1;
}
