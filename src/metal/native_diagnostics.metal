// Diagnostic-only reductions. No sampled values feed model execution.
struct H3DiagnosticRange { float lo,hi,minimum_nonzero,pad;uint count,nonfinite,over_half,under_half; };
struct H3DiagnosticShape {uint count,width,stride,offset,record,groups;};
kernel void h3_diagnostic_range(const device bfloat *input [[buffer(0)]],
    device H3DiagnosticRange *output [[buffer(1)]],constant H3DiagnosticShape &p [[buffer(2)]],
    uint group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
  float lo=FLT_MAX,hi=-FLT_MAX,mn=FLT_MAX;uint count=0,bad=0,over=0,under=0;
  for(uint i=group*256+lane;i<p.count;i+=p.groups*256) {
    uint pos=(i/p.width)*p.stride+i%p.width+p.offset;
    uint bits=reinterpret_cast<const device ushort*>(input)[pos]&0x7fff;
    float x=float(input[pos]);count++;
    if(bits>=0x7f80){bad++;continue;}
    lo=min(lo,x);hi=max(hi,x);if(bits)mn=min(mn,abs(x));
    over+=uint(abs(x)>65504.f);under+=uint(bits!=0 && bits<0x3880);
  }
  lo=simd_min(lo);hi=simd_max(hi);mn=simd_min(mn);count=simd_sum(count);
  bad=simd_sum(bad);over=simd_sum(over);under=simd_sum(under);
  threadgroup H3DiagnosticRange values[8];
  if(lane%32==0)values[lane/32]={lo,hi,mn,0,count,bad,over,under};
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0){H3DiagnosticRange r={FLT_MAX,-FLT_MAX,FLT_MAX,0,0,0,0,0};
    for(uint j=0;j<8;j++){r.lo=min(r.lo,values[j].lo);r.hi=max(r.hi,values[j].hi);
      r.minimum_nonzero=min(r.minimum_nonzero,values[j].minimum_nonzero);r.count+=values[j].count;
      r.nonfinite+=values[j].nonfinite;r.over_half+=values[j].over_half;r.under_half+=values[j].under_half;}
    output[p.record*p.groups+group]=r;
  }
}

struct H3DiagnosticScore {float lo,hi,minimum_exp,sum_exp;uint nonfinite,zero_exp,half_under_exp,keys;};
struct H3DiagnosticScoreShape {uint sequence,heads,major,record;float scale;};
// Sixteen evenly spaced Q rows/head, every K row, original BF16 operands and
// independent scalar FP32 dot/softmax. O(16*H*N) scratch, never N*N.
kernel void h3_diagnostic_score(const device bfloat *q [[buffer(0)]],
    const device bfloat *k [[buffer(1)]],device float *scores [[buffer(2)]],
    device H3DiagnosticScore *out [[buffer(3)]],constant H3DiagnosticScoreShape &p [[buffer(4)]],
    uint sample [[threadgroup_position_in_grid]],uint lane [[thread_index_in_threadgroup]]) {
  uint head=sample/16,row=(sample%16)*(p.sequence-1)/15;
  threadgroup float query[128],reduction[8];
  if(lane<128)query[lane]=float(q[p.major?(head*p.sequence+row)*128+lane:(row*p.heads+head)*128+lane]);
  threadgroup_barrier(mem_flags::mem_threadgroup);
  float lo=FLT_MAX,hi=-FLT_MAX;uint bad=0;
  for(uint key=lane;key<p.sequence;key+=256){float dot=0;
    uint base=p.major?(head*p.sequence+key)*128:(key*p.heads+head)*128;
    for(uint d=0;d<128;d++)dot=fma(query[d],float(k[base+d]),dot);
    float score=dot*p.scale;scores[sample*p.sequence+key]=score;
    bad+=uint(!isfinite(score));lo=min(lo,score);hi=max(hi,score);
  }
  float local_hi=simd_max(hi);if(lane%32==0)reduction[lane/32]=local_hi;
  threadgroup_barrier(mem_flags::mem_threadgroup|mem_flags::mem_device);
  float maximum=-FLT_MAX;for(uint j=0;j<8;j++)maximum=max(maximum,reduction[j]);
  float sum=0,mn=FLT_MAX;uint zeros=0,under=0;
  for(uint key=lane;key<p.sequence;key+=256){float e=exp(scores[sample*p.sequence+key]-maximum);
    sum+=e;mn=min(mn,e);zeros+=uint(e==0);under+=uint(e>0 && e<0x1p-14f);bad+=uint(!isfinite(e));}
  H3DiagnosticScore r={simd_min(lo),simd_max(hi),simd_min(mn),simd_sum(sum),simd_sum(bad),simd_sum(zeros),simd_sum(under),p.sequence};
  threadgroup H3DiagnosticScore values[8];if(lane%32==0)values[lane/32]=r;
  threadgroup_barrier(mem_flags::mem_threadgroup);
  if(lane==0){r={FLT_MAX,-FLT_MAX,FLT_MAX,0,0,0,0,p.sequence};
    for(uint j=0;j<8;j++){r.lo=min(r.lo,values[j].lo);r.hi=max(r.hi,values[j].hi);
      r.minimum_exp=min(r.minimum_exp,values[j].minimum_exp);r.sum_exp+=values[j].sum_exp;
      r.nonfinite+=values[j].nonfinite;r.zero_exp+=values[j].zero_exp;r.half_under_exp+=values[j].half_under_exp;}
    out[p.record*p.heads*16+sample]=r;
  }
}
