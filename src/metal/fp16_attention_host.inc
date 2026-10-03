/* M2 mixed attention. The adapter and SOL retain original BF16 inputs.
 * Fused dense preparation owns/reuses Q/K/V storage: admitted heads become
 * scaled half operands, while recovery heads retain their original BF16 bits.
 * Every path validates the output before committing model-visible BF16 data. */
#include "src/metal/metal_fp16.h"
_Static_assert(sizeof(h3_metal_fp16_range)==68,"FP16 range ABI");
typedef struct { uint32_t sequence,heads,input_major,output_major;float scale; } h3_fp16_shape;

/* A preparation is single-use and bound to the exact owned buffers. */
static void mixed_clear_prepared(H3GPU *g) {
    @autoreleasepool {
    g.fp16Prepared=NO;g.fp16PreparedQ=nil;g.fp16PreparedK=nil;g.fp16PreparedV=nil;
    }
}
int h3_gpu_mixed_prepare(h3_gpu *gpu,h3_gpu_tensor *q,h3_gpu_tensor *k,h3_gpu_tensor *v,
    const h3_gpu_tensor *qkv,const h3_gpu_tensor *qn,const h3_gpu_tensor *kn,
    const h3_gpu_tensor *cos,const h3_gpu_tensor *sin,uint32_t seq,uint32_t heads,
    uint32_t dim,uint32_t rope_half,float epsilon,float scale) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g)return 0;
    if(g.fp16Prepared||!h3_metal_fp16_shape_valid(seq,heads)||heads%4||dim!=128||
       rope_half>64||!rope_half||(uint64_t)seq*heads*128>UINT32_MAX/3||
       !isfinite(epsilon)||epsilon<=0||!isfinite(scale)||scale<=0||
       getenv("H3_DISABLE_COOP_QKV")||getenv("H3_DISABLE_CACHED_QKV")) {
        h3_gpu_set_error(g,@"Invalid or unconsumed fused preparation; requires cached cooperative QKV");return 0;
    }
    size_t n=(size_t)seq*heads*128;
    const h3_gpu_tensor *inputs[]={qkv,qn,kn,cos,sin},*outputs[]={q,k,v};
    size_t counts[]={n*3,128,128,(size_t)seq*rope_half,(size_t)seq*rope_half};
    if(!h3_gpu_require_command(g))return 0;
    for(unsigned i=0;i<5;i++)if(!h3_gpu_require_bf16(g,inputs[i],counts[i],@"fused preparation input"))return 0;
    for(unsigned i=0;i<3;i++) {
        if(!h3_gpu_require_bf16(g,outputs[i],n,@"fused preparation output"))return 0;
        for(unsigned j=0;j<5;j++)if(TENSOR(outputs[i]).buffer==TENSOR(inputs[j]).buffer) {
            h3_gpu_set_error(g,@"Fused preparation must preserve raw inputs");return 0;
        }
        for(unsigned j=0;j<i;j++)if(TENSOR(outputs[i]).buffer==TENSOR(outputs[j]).buffer) {
            h3_gpu_set_error(g,@"Fused Q/K/V require distinct buffers");return 0;
        }
    }
    h3_gpu_tensor *partials=native_scratch(g,@"fp16-range-partials",(size_t)seq*heads*7,H3_GPU_U32);
    h3_gpu_tensor *ranges=native_scratch(g,@"fp16-ranges",(heads+1)*sizeof(h3_metal_fp16_range)/4,H3_GPU_U32);
    id<MTLComputePipelineState> pipeline=h3_gpu_pipeline(g,@"h3_qkv_rope_bf16_coop_ranges");
    if(!partials||!ranges||!pipeline||pipeline.maxTotalThreadsPerThreadgroup<128)return 0;
    qkv_args args={seq,heads,128,rope_half,3,epsilon};
    id<MTLComputeCommandEncoder> e=[g.command computeCommandEncoder];
    if(!e){h3_gpu_set_error(g,@"Cannot encode fused preparation");return 0;}
    e.label=@"h3_qkv_rope_bf16_coop_ranges";[e setComputePipelineState:pipeline];
    for(unsigned i=0;i<5;i++)[e setBuffer:TENSOR(inputs[i]).buffer offset:0 atIndex:i];
    for(unsigned i=0;i<3;i++)[e setBuffer:TENSOR(outputs[i]).buffer offset:0 atIndex:i+5];
    [e setBytes:&args length:sizeof(args) atIndex:8];[e setBuffer:TENSOR(partials).buffer offset:0 atIndex:9];
    [e dispatchThreadgroups:MTLSizeMake(heads/4,seq,1) threadsPerThreadgroup:MTLSizeMake(128,1,1)];[e endEncoding];
    h3_fp16_shape shape={seq,heads,1,0,scale};
    if(!native_aux(g,@"h3_fp16_ranges_from_partials",MTLSizeMake(heads,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> encoder){
        [encoder setBuffer:TENSOR(partials).buffer offset:0 atIndex:0];[encoder setBuffer:TENSOR(ranges).buffer offset:0 atIndex:1];
        [encoder setBytes:&shape length:sizeof(shape) atIndex:2];}))return 0;
    g.fp16Prepared=YES;g.fp16PreparedSequence=seq;g.fp16PreparedHeads=heads;g.fp16PreparedScale=scale;
    g.fp16PreparedQ=TENSOR(q).buffer;g.fp16PreparedK=TENSOR(k).buffer;g.fp16PreparedV=TENSOR(v).buffer;
    g.headMajorSDPAInputs=YES;
    h3_gpu_stats st=g.stats;st.direct_dispatches++;st.layout_prepares++;
    st.layout_scan_replaced_bytes+=n*6;st.layout_partial_bytes+=(uint64_t)seq*heads*28;g.stats=st;
    return 1;
    }
}

static int mixed_attention(H3GPU *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,h3_gpu_tensor *ranges,
    h3_fp16_shape shape,int tile,int recover,h3_gpu_tensor *flags,
    h3_route_params route,h3_gpu_tensor *maxima,h3_gpu_tensor *sums) {
    @autoreleasepool {
    int bq=tile?64:32,bk=tile?32:16;
    NSString *name=[NSString stringWithFormat:@"h3_fp16_%@_%d%@",recover?@"recover":@"half",tile,flags?@"_sol":@""];
    id<MTLComputePipelineState> pipeline=native_pipeline(g,name,(int)shape.sequence,(int)shape.sequence,bq,bk);
    if(!pipeline||pipeline.maxTotalThreadsPerThreadgroup<(NSUInteger)bq*4)return 0;
    int s=(int)shape.sequence;int64_t n=(int64_t)s*shape.heads*128;
    h3_native_attn_params p={.B=1,.H=(int)shape.heads,.D=128,.qL=s,.kL=s,.gqa_factor=1,.scale=shape.scale,
        .NQ=(s+bq-1)/bq,.NK=(s+bk-1)/bk,.NQ_aligned=s/bq,.NK_aligned=s/bk,.qL_rem=s%bq,.kL_rem=s%bk};
    p.Q_strides[0]=n;p.Q_strides[1]=shape.input_major?(int64_t)s*128:128;
    p.Q_strides[2]=shape.input_major?128:(int64_t)shape.heads*128;
    memcpy(p.K_strides,p.Q_strides,sizeof(p.K_strides));memcpy(p.V_strides,p.Q_strides,sizeof(p.V_strides));
    p.O_strides[0]=n;p.O_strides[1]=shape.output_major?(int64_t)s*128:128;
    p.O_strides[2]=shape.output_major?128:(int64_t)shape.heads*128;
    id<MTLComputeCommandEncoder> e=[g.command computeCommandEncoder];
    if(!e){h3_gpu_set_error(g,@"Cannot encode mixed attention");return 0;}
    e.label=name;[e setComputePipelineState:pipeline];
    const h3_gpu_tensor *buffers[]={q,k,v,out};
    for(unsigned i=0;i<4;i++)[e setBuffer:TENSOR(buffers[i]).buffer offset:0 atIndex:i];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e setBuffer:TENSOR(ranges).buffer offset:0 atIndex:12];
    if(flags) {
        [e setBuffer:TENSOR(flags).buffer offset:0 atIndex:8];[e setBytes:&route length:sizeof(route) atIndex:9];
        [e setBuffer:TENSOR(maxima).buffer offset:0 atIndex:10];[e setBuffer:TENSOR(sums).buffer offset:0 atIndex:11];
    }
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)p.NQ,shape.heads,1) threadsPerThreadgroup:MTLSizeMake(32,(NSUInteger)bq/8,1)];
    [e endEncoding];h3_gpu_stats st=g.stats;st.direct_dispatches++;g.stats=st;return 1;
    }
}

static int mixed_sol_exact(H3GPU *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,h3_gpu_tensor *ranges,uint32_t seq,
    uint32_t heads,float scale,int qb,h3_gpu_tensor *flags,h3_route_params route,
    h3_gpu_tensor *maxima,h3_gpu_tensor *sums) {
    @autoreleasepool {
    h3_fp16_shape shape={seq,heads,1,1,scale};
    return mixed_attention(g,out,q,k,v,ranges,shape,qb==64,0,flags,route,maxima,sums)&&
        mixed_attention(g,out,q,k,v,ranges,shape,qb==64,1,flags,route,maxima,sums);
    }
}

static int mixed_sdpa_impl(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t seq,uint32_t heads,
    float scale,int il,int ol,int tile,const h3_metal_attention_options *sol,unsigned block,int step,int prepared) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);
    if(!g)return 0;
    if(!h3_metal_fp16_shape_valid(seq,heads)||
       !isfinite(scale)||scale<=0||(il!=0&&il!=1)||(ol!=0&&ol!=1)||tile<0||tile>1) {
        h3_gpu_set_error(g,@"Invalid mixed attention shape, layout, scale or tile "
            "(sequence=%u heads=%u input_layout=%d output_layout=%d scale=%g tile=%d; "
            "requires 1..256 heads and sequence*heads*128 <= %u)",
            seq,heads,il,ol,(double)scale,tile,UINT32_MAX-255u);return 0;
    }
    size_t n=(size_t)seq*heads*128;
    if(!h3_gpu_require_command(g)||!h3_gpu_require_bf16(g,q,n,@"mixed Q")||
       !h3_gpu_require_bf16(g,k,n,@"mixed K")||!h3_gpu_require_bf16(g,v,n,@"mixed V")||
       !h3_gpu_require_bf16(g,out,n,@"mixed O"))return 0;
    if(TENSOR(out).buffer==TENSOR(q).buffer||TENSOR(out).buffer==TENSOR(k).buffer||TENSOR(out).buffer==TENSOR(v).buffer) {
        h3_gpu_set_error(g,@"Mixed attention requires distinct BF16 output");return 0;
    }
    if(prepared) {
        if(!g.fp16Prepared||il!=1||g.fp16PreparedSequence!=seq||g.fp16PreparedHeads!=heads||
           g.fp16PreparedScale!=scale||g.fp16PreparedQ!=TENSOR(q).buffer||
           g.fp16PreparedK!=TENSOR(k).buffer||g.fp16PreparedV!=TENSOR(v).buffer) {
            h3_gpu_set_error(g,@"Missing, reused or mismatched fused preparation");return 0;
        }
        mixed_clear_prepared(g);
    } else if(g.fp16Prepared) {
        h3_gpu_set_error(g,@"Consume fused preparation before invoking adapter attention");return 0;
    }
    BOOL inplace=prepared&&!sol;
    h3_gpu_tensor *qh=inplace?(h3_gpu_tensor*)q:native_scratch(g,@"fp16-q",n*2,H3_GPU_I8);
    h3_gpu_tensor *kh=inplace?(h3_gpu_tensor*)k:native_scratch(g,@"fp16-k",n*2,H3_GPU_I8);
    h3_gpu_tensor *vh=inplace?(h3_gpu_tensor*)v:native_scratch(g,@"fp16-v",n*2,H3_GPU_I8);
    h3_gpu_tensor *tmp=native_scratch(g,@"fp16-output",n,H3_GPU_F32);
    h3_gpu_tensor *ranges=native_scratch(g,@"fp16-ranges",(heads+1)*sizeof(h3_metal_fp16_range)/4,H3_GPU_U32);
    if(!qh||!kh||!vh||!tmp||!ranges)return 0;
    h3_fp16_shape p={seq,heads,(uint32_t)il,(uint32_t)ol,scale};
    if(!prepared&&!native_aux(g,@"h3_fp16_ranges",MTLSizeMake(heads,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> e){
        const h3_gpu_tensor *ts[]={q,k,v,ranges};for(unsigned i=0;i<4;i++)[e setBuffer:TENSOR(ts[i]).buffer offset:0 atIndex:i];
        [e setBytes:&p length:sizeof(p) atIndex:4];}))return 0;
    if(!native_aux(g,@"h3_fp16_convert",MTLSizeMake((n+255)/256,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> e){
        const h3_gpu_tensor *ts[]={q,k,v,qh,kh,vh,ranges};for(unsigned i=0;i<7;i++)[e setBuffer:TENSOR(ts[i]).buffer offset:0 atIndex:i];
        [e setBytes:&p length:sizeof(p) atIndex:7];}))return 0;
    h3_fp16_shape half_shape=p;half_shape.input_major=1;
    if(sol) {
        if(!native_sol_impl(gpu,tmp,q,k,v,seq,heads,scale,il,ol,*sol,block,step,qh,kh,vh,ranges))return 0;
    } else {
        if(!mixed_attention(g,tmp,qh,kh,vh,ranges,half_shape,tile,0,NULL,(h3_route_params){0},NULL,NULL)||
           !mixed_attention(g,tmp,qh,kh,vh,ranges,half_shape,tile,1,NULL,(h3_route_params){0},NULL,NULL))return 0;
    }
    if(!native_aux(g,@"h3_fp16_output_check",MTLSizeMake(heads,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:TENSOR(tmp).buffer offset:0 atIndex:0];[e setBuffer:TENSOR(ranges).buffer offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];}))return 0;
    if(!native_aux(g,@"h3_fp16_ready",MTLSizeMake(1,1,1),MTLSizeMake(1,1,1),^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:TENSOR(ranges).buffer offset:0 atIndex:0];[e setBytes:&p length:sizeof(p) atIndex:1];}))return 0;
    if(!native_aux(g,@"h3_fp16_commit",MTLSizeMake((n+255)/256,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:TENSOR(tmp).buffer offset:0 atIndex:0];[e setBuffer:TENSOR(out).buffer offset:0 atIndex:1];
        [e setBuffer:TENSOR(ranges).buffer offset:0 atIndex:2];[e setBytes:&p length:sizeof(p) atIndex:3];}))return 0;
    h3_gpu_stats st=g.stats;st.native_attention_dispatches++;if(inplace)st.layout_inplace_packs++;g.stats=st;return 1;
    }
}
int h3_gpu_mixed_sdpa(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t seq,uint32_t heads,
    float scale,int il,int ol,int tile) {
    @autoreleasepool {
    return mixed_sdpa_impl(gpu,out,q,k,v,seq,heads,scale,il,ol,tile,NULL,0,0,0);
    }
}
int h3_gpu_mixed_sol(h3_gpu *gpu,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t seq,uint32_t heads,
    float scale,int il,int ol,h3_metal_attention_options o,unsigned block,int step) {
    @autoreleasepool {
    return mixed_sdpa_impl(gpu,out,q,k,v,seq,heads,scale,il,ol,o.q_block==64,&o,block,step,0);
    }
}
int h3_gpu_mixed_prepared_attention(h3_gpu *gpu,h3_gpu_tensor *out,
    h3_gpu_tensor *q,h3_gpu_tensor *k,h3_gpu_tensor *v,uint32_t seq,uint32_t heads,
    float scale,int ol,h3_metal_attention_options o,int routed,unsigned block,int step) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g)return 0;
    if(o.layout_fusion!=1||o.precision!=1||o.candidate!=1||(routed!=0&&routed!=1)) {
        h3_gpu_set_error(g,@"Fused attention requires the admitted mixed Steel recipe");return 0;
    }
    return mixed_sdpa_impl(gpu,out,q,k,v,seq,heads,scale,1,ol,routed&&(o.q_block==64),routed?&o:NULL,block,step,1);
    }
}
int h3_gpu_mixed_range_report(h3_gpu *gpu,h3_metal_fp16_range *ranges,uint32_t heads) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g||!ranges||!heads)return 0;
    h3_gpu_tensor *t=g.nativeScratch.slots[@"fp16-ranges"].pointerValue;
    if(!t||TENSOR(t).elements!=(heads+1)*sizeof(*ranges)/4)return 0;
    const void *p=h3_gpu_tensor_contents(t);if(!p)return 0;
    memcpy(ranges,p,heads*sizeof(*ranges));return 1;
    }
}

int h3_gpu_mixed_record(h3_gpu *gpu,uint32_t heads,unsigned block,int step) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);
    if(!g||!heads||heads>256||block>=50||step<0||!h3_gpu_require_command(g))return 0;
    h3_gpu_tensor *source=g.nativeScratch.slots[@"fp16-ranges"].pointerValue;
    if(!source||TENSOR(source).elements!=(heads+1)*sizeof(h3_metal_fp16_range)/4)return 0;
    size_t bytes=heads*sizeof(h3_metal_fp16_range);
    h3_gpu_tensor *records=native_scratch(g,@"fp16-records",50*6*bytes/4,H3_GPU_U32);
    if(!records)return 0;
    NSUInteger slot=(NSUInteger)(step%6)*50+block;
    id<MTLBlitCommandEncoder> encoder=[g.command blitCommandEncoder];
    if(!encoder){h3_gpu_set_error(g,@"Cannot archive mixed attention validation");return 0;}
    [encoder copyFromBuffer:TENSOR(source).buffer sourceOffset:0 toBuffer:TENSOR(records).buffer
         destinationOffset:slot*bytes size:bytes];[encoder endEncoding];
    if(!g.fp16Used)g.fp16Used=[NSMutableIndexSet new];
    g.fp16Heads=heads;[g.fp16Used addIndex:slot];
    h3_gpu_stats st=g.stats;st.blit_copies++;st.d2d_bytes+=bytes;g.stats=st;return 1;
    }
}

int h3_gpu_mixed_step_finish(h3_gpu *gpu,int step) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g||step<0)return 0;
    h3_gpu_tensor *records=g.nativeScratch.slots[@"fp16-records"].pointerValue;
    const h3_metal_fp16_range *ranges=records?h3_gpu_tensor_contents(records):NULL;
    if(!ranges){h3_gpu_set_error(g,@"Missing mixed attention forward records");return 0;}
    unsigned invalid=0,blocks=0;
    for(unsigned b=0;b<50;b++) {
        NSUInteger slot=(NSUInteger)(step%6)*50+b;
        if(![g.fp16Used containsIndex:slot])continue;
        unsigned recovered=0,bad=0;uint64_t underflows=0;float q=0,k=0,v=0,o=0,bound=0;
        for(unsigned h=0;h<g.fp16Heads;h++) {
            const h3_metal_fp16_range *r=ranges+slot*g.fp16Heads+h;
            recovered+=(r->flags&2)!=0;bad+=((r->flags&1)!=0)||r->output_invalid;
            underflows+=(uint64_t)r->q_underflows+r->k_underflows+r->v_underflows;
            q=fmaxf(q,r->q_max);k=fmaxf(k,r->k_max);v=fmaxf(v,r->v_max);o=fmaxf(o,r->output_max);
            if(!r->flags)bound=fmaxf(bound,r->score_error_bound);
        }
        if(h3_profile_steps_enabled()||bad)
            fprintf(stderr,"h3_mixed {\"step\":%d,\"block\":%u,\"heads\":%u,\"recovered_heads\":%u,\"invalid_heads\":%u,\"underflows\":%llu,\"q_max\":%.9g,\"k_max\":%.9g,\"v_max\":%.9g,\"output_max\":%.9g,\"score_error_bound\":%.9g}\n",
                step+1,b,g.fp16Heads,recovered,bad,(unsigned long long)underflows,q,k,v,o,bound);
        invalid+=bad;blocks++;[g.fp16Used removeIndex:slot];
    }
    if(invalid||!blocks){h3_gpu_set_error(g,@"Mixed attention rejected forward: %u invalid heads in %u recorded blocks",invalid,blocks);return 0;}
    return 1;
    }
}
