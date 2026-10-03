// Local M2 range adapter. No external implementation code.
struct H3FP16Range {
  float q_max,k_max,v_max,q_min,k_min,v_min;
  int q_exponent,k_exponent,v_exponent;
  uint flags;
  float output_max;
  uint output_invalid;
  float score_error_bound,value_error_bound;
  uint q_underflows,k_underflows,v_underflows;
};
struct H3FP16Shape { uint sequence,heads,input_major,output_major; float scale; };

inline H3FP16Range h3_fp16_range_policy(float3 mx,float3 mn,uint bad,float scale) {
    int3 ex=int3(0);
    // Put the largest magnitude in [4096,8192); ldexp avoids overflow in
    // an intermediate scaling factor for small source values.
    for(uint j=0;j<3;j++)if(mx[j]>0&&isfinite(mx[j]))ex[j]=ilogb(mx[j])-12;
    float bound=ldexp(scale,ex.x+ex.y)*128.0f*8192.0f*8192.0f;
    bad|=uint(!isfinite(bound));
    float3 delta=0;
    for(uint j=0;j<3;j++)if(mn[j]!=FLT_MAX&&ldexp(mn[j],-ex[j])<0x1p-14f)
      delta[j]=ldexp(0x1p-14f,ex[j]);
    // With BF16's 8 significant bits, normal half conversion after a power
    // of two is exact. The only operand loss is explicitly counted underflow.
    // Bound the score perturbation before admitting it; otherwise recover.
    float score_error=128.0f*scale*(delta.x*mx.y+delta.y*mx.x+delta.x*delta.y);
    bool recover=!isfinite(score_error)||score_error>1e-4f;
    // BF16 recovery uses FP32 matrix operands, so bound their unscaled QK.
    if(recover&&(!isfinite(mx.x*mx.y*128.0f)||!isfinite((mx.x*scale)*mx.y*128.0f)))bad=1;
    return {mx.x,mx.y,mx.z,mn.x,mn.y,mn.z,ex.x,ex.y,ex.z,
                  bad?1u:recover?2u:0u,0,0,score_error,delta.z,0,0,0};
}

// Each head owns one record. Reduction in registers and a small threadgroup
// buffer; no host synchronization or atomics shared between heads.
kernel void h3_fp16_ranges(const device bfloat *q [[buffer(0)]],
    const device bfloat *k [[buffer(1)]],const device bfloat *v [[buffer(2)]],
    device H3FP16Range *ranges [[buffer(3)]],constant H3FP16Shape &p [[buffer(4)]],
    uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
  float3 mx=0,mn=FLT_MAX;uint bad=0;
  for(uint i=lane;i<p.sequence*128;i+=256) {
    ulong idx=p.input_major?ulong(head)*p.sequence*128+i:ulong(i/128)*p.heads*128+head*128+i%128;
    float3 x=abs(float3(float(q[idx]),float(k[idx]),float(v[idx])));
    ushort3 bits=ushort3(reinterpret_cast<const device ushort*>(q)[idx],
        reinterpret_cast<const device ushort*>(k)[idx],reinterpret_cast<const device ushort*>(v)[idx]);
    // Metal cannot promise preservation of FP32-denormal BF16 source values.
    // Reject them explicitly instead of observing a silently flushed zero.
    bad|=uint(any(((bits&ushort3(0x7f80))==0)&((bits&ushort3(0x007f))!=0)));
    bad|=uint(!all(isfinite(x)));mx=max(mx,x);mn=min(mn,select(float3(FLT_MAX),x,x>0));
  }
  threadgroup float3 maxima[8],minima[8];threadgroup uint invalid[8];
  mx=float3(simd_max(mx.x),simd_max(mx.y),simd_max(mx.z));
  mn=float3(simd_min(mn.x),simd_min(mn.y),simd_min(mn.z));bad=simd_or(bad);
  if(lane%32==0){maxima[lane/32]=mx;minima[lane/32]=mn;invalid[lane/32]=bad;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0) {
    mx=0;mn=FLT_MAX;bad=0;
    for(uint j=0;j<8;j++){mx=max(mx,maxima[j]);mn=min(mn,minima[j]);bad|=invalid[j];}
    ranges[head]=h3_fp16_range_policy(mx,mn,bad,p.scale);
  }
}

kernel void h3_fp16_convert(const device bfloat *q [[buffer(0)]],
    const device bfloat *k [[buffer(1)]],const device bfloat *v [[buffer(2)]],
    device half *qh [[buffer(3)]],device half *kh [[buffer(4)]],device half *vh [[buffer(5)]],
    device H3FP16Range *ranges [[buffer(6)]],constant H3FP16Shape &p [[buffer(7)]],
    uint i [[thread_position_in_grid]]) {
  if(i>=p.sequence*p.heads*128)return;
  uint head=p.input_major?i/(p.sequence*128):(i/128)%p.heads;
  uint flags=ranges[head].flags;if(flags&1)return;
  uint dst=p.input_major?i:(head*p.sequence+i/(p.heads*128))*128+i%128;
  if(flags&2) {
    // Recovery shares the packed workspace but preserves every original BF16
    // bit. The conditional BF16 kernel reads only these heads; the half kernel
    // reads only admitted FP16 heads. No second full-size copy is necessary.
    reinterpret_cast<device bfloat*>(qh)[dst]=q[i];
    reinterpret_cast<device bfloat*>(kh)[dst]=k[i];
    reinterpret_cast<device bfloat*>(vh)[dst]=v[i];return;
  }
  float3 x=ldexp(float3(float(q[i]),float(k[i]),float(v[i])),
      int3(-ranges[head].q_exponent,-ranges[head].k_exponent,-ranges[head].v_exponent));
  bool3 lost=(abs(x)<0x1p-14f)&(x!=0);
  if(lost.x)atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&ranges[head].q_underflows),1u,memory_order_relaxed);
  if(lost.y)atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&ranges[head].k_underflows),1u,memory_order_relaxed);
  if(lost.z)atomic_fetch_add_explicit(reinterpret_cast<device atomic_uint*>(&ranges[head].v_underflows),1u,memory_order_relaxed);
  x=select(x,float3(0),lost);qh[dst]=half(x.x);kh[dst]=half(x.y);vh[dst]=half(x.z);
}

kernel void h3_fp16_output_check(const device float *in [[buffer(0)]],
    device H3FP16Range *ranges [[buffer(1)]],constant H3FP16Shape &p [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
  if(ranges[head].flags&1)return;
  float mx=0;uint bad=0;
  for(uint i=lane;i<p.sequence*128;i+=256) {
    ulong idx=p.output_major?ulong(head)*p.sequence*128+i:ulong(i/128)*p.heads*128+head*128+i%128;
    float x=in[idx];mx=max(mx,abs(x));bad|=uint(!isfinite(x)||!isfinite(float(bfloat(x))));
  }
  threadgroup float maxima[8];threadgroup uint invalid[8];
  mx=simd_max(mx);bad=simd_or(bad);
  if(lane%32==0){maxima[lane/32]=mx;invalid[lane/32]=bad;}
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0){mx=0;bad=0;for(uint j=0;j<8;j++){mx=max(mx,maxima[j]);bad|=invalid[j];}
    ranges[head].output_max=mx;ranges[head].output_invalid=bad;}
}

kernel void h3_fp16_ready(device H3FP16Range *ranges [[buffer(0)]],constant H3FP16Shape &p [[buffer(1)]]) {
  uint bad=0;for(uint h=0;h<p.heads;h++)bad|=(ranges[h].flags&1)|ranges[h].output_invalid;
  ranges[p.heads].flags=bad;
}

kernel void h3_fp16_commit(const device float *in [[buffer(0)]],device bfloat *out [[buffer(1)]],
    const device H3FP16Range *ranges [[buffer(2)]],constant H3FP16Shape &p [[buffer(3)]],
    uint i [[thread_position_in_grid]]) {
  if(i>=p.sequence*p.heads*128)return;
  if(ranges[p.heads].flags)return;
  out[i]=bfloat(in[i]);
}

// Reduce the tiny per-row records emitted by the unchanged norm/RoPE arithmetic.
kernel void h3_fp16_ranges_from_partials(device const uint *partials [[buffer(0)]],
    device H3FP16Range *ranges [[buffer(1)]],constant H3FP16Shape &p [[buffer(2)]],
    uint head [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
    float3 mx=0,mn=FLT_MAX;uint bad=0;
    for(uint row=lane;row<p.sequence;row+=256) {
        uint offset=(head*p.sequence+row)*7;
        for(uint d=0;d<3;d++){mx[d]=max(mx[d],as_type<float>(partials[offset+d]));mn[d]=min(mn[d],as_type<float>(partials[offset+3+d]));}
        bad|=partials[offset+6];
    }
    threadgroup float3 maxima[8],minima[8];threadgroup uint invalid[8];
    mx=float3(simd_max(mx.x),simd_max(mx.y),simd_max(mx.z));
    mn=float3(simd_min(mn.x),simd_min(mn.y),simd_min(mn.z));bad=simd_or(bad);
    if(lane%32==0){maxima[lane/32]=mx;minima[lane/32]=mn;invalid[lane/32]=bad;}
    threadgroup_barrier(mem_flags::mem_threadgroup);
    if(lane==0) {
        mx=0;mn=FLT_MAX;bad=0;
        for(uint j=0;j<8;j++){mx=max(mx,maxima[j]);mn=min(mn,minima[j]);bad|=invalid[j];}
        ranges[head]=h3_fp16_range_policy(mx,mn,bad,p.scale);
    }
}
