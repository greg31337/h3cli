/* Native attention experiment support; included after the unchanged Steel A
 * dispatcher in src/metal/gpu.m. Scratch belongs to one DiT GPU context and is reused
 * in command order. No host readback or allocation occurs between warm layers. */
typedef struct { uint32_t query_blocks,key_blocks,key_block,unused; } h3_route_params;
typedef struct {
    uint32_t sequence,heads,query_block,key_block,query_blocks,key_blocks;
    uint32_t input_head_major,local_radius,count_offset,minimum_exact;
    float scale,tau;
} h3_sol_params;
_Static_assert(sizeof(h3_sol_params)==48,"SOL params ABI");
_Static_assert(sizeof(h3_sol_block)==16,"SOL protection ABI");

static h3_gpu_tensor *native_scratch(H3GPU *g,NSString *name,size_t elements,h3_gpu_dtype dtype) {
    @autoreleasepool {
    if(!g.nativeScratch)g.nativeScratch=[H3ScratchArena new];
    h3_gpu_tensor *t=g.nativeScratch.slots[name].pointerValue;
    if(t) {
        if(TENSOR(t).elements!=elements||TENSOR(t).dtype!=dtype) {
            h3_gpu_set_error(g,@"Native scratch geometry changed: %@; create a fresh DiT context",name);return NULL;
        }
        h3_gpu_stats stats=g.stats;stats.native_scratch_reuses++;g.stats=stats;
        return t;
    }
    t=h3_gpu_tensor_alloc((__bridge h3_gpu *)g,elements,dtype,H3_GPU_HOST_VISIBLE);
    if(t) {
        g.nativeScratch.slots[name]=[NSValue valueWithPointer:t];
        h3_gpu_stats stats=g.stats;stats.native_scratch_allocations++;
        stats.native_scratch_bytes+=TENSOR(t).bytes;g.stats=stats;
    }
    return t;
    }
}

static id<MTLComputePipelineState> native_pipeline(H3GPU *g,NSString *name,int ql,int kl,int bq,int bk) {
    @autoreleasepool {
    NSError *error=nil;
    if(!g.nativeAttentionLibrary) {
        MTLCompileOptions *options=[MTLCompileOptions new];options.languageVersion=MTLLanguageVersion3_1;
        options.mathMode=MTLMathModeSafe;
        NSString *source=[[NSString alloc] initWithBytes:h3_native_attention_source
            length:sizeof(h3_native_attention_source)-1 encoding:NSUTF8StringEncoding];
        g.nativeAttentionLibrary=[g.device newLibraryWithSource:source options:options error:&error];
        if(!g.nativeAttentionLibrary){h3_gpu_set_error(g,@"Native attention compilation: %@",error);return nil;}
        g.nativeAttentionPipelines=[NSMutableDictionary new];
    }
    BOOL aq=bq&&ql%bq==0,ak=bk&&kl%bk==0,no=NO;
    NSString *key=[NSString stringWithFormat:@"%@:%d:%d",name,aq,ak];
    id<MTLComputePipelineState> pipeline=g.nativeAttentionPipelines[key];
    if(!pipeline) {
        id<MTLFunction> f;
        if(bq) {
            MTLFunctionConstantValues *constants=[MTLFunctionConstantValues new];
            [constants setConstantValue:&aq type:MTLDataTypeBool atIndex:200];
            [constants setConstantValue:&ak type:MTLDataTypeBool atIndex:201];
            for(NSUInteger i=300;i<=302;i++)[constants setConstantValue:&no type:MTLDataTypeBool atIndex:i];
            f=[g.nativeAttentionLibrary newFunctionWithName:name constantValues:constants error:&error];
        } else f=[g.nativeAttentionLibrary newFunctionWithName:name];
        if(f)pipeline=[g.device newComputePipelineStateWithFunction:f error:&error];
        if(!pipeline){h3_gpu_set_error(g,@"Native pipeline %@: %@",name,error);return nil;}
        if(pipeline.threadExecutionWidth!=32||pipeline.staticThreadgroupMemoryLength>g.device.maxThreadgroupMemoryLength) {
            h3_gpu_set_error(g,@"Native pipeline %@ exceeds device limits",name);return nil;
        }
        g.nativeAttentionPipelines[key]=pipeline;
        h3_gpu_stats stats=g.stats;stats.native_pipeline_count=g.nativeAttentionPipelines.count;g.stats=stats;
    }
    return pipeline;
    }
}

static int native_aux(H3GPU *g,NSString *name,MTLSize groups,MTLSize threads,
                       void (^bind)(id<MTLComputeCommandEncoder>)) {
    @autoreleasepool {
    id<MTLComputePipelineState> p=native_pipeline(g,name,0,0,0,0);
    if(!p||!h3_gpu_require_command(g))return 0;
    if(threads.width*threads.height*threads.depth>p.maxTotalThreadsPerThreadgroup) {
        h3_gpu_set_error(g,@"Native pipeline %@ thread count exceeds limits",name);return 0;
    }
    id<MTLComputeCommandEncoder> e=[g.command computeCommandEncoder];
    if(!e){h3_gpu_set_error(g,@"Cannot encode %@",name);return 0;}
    e.label=name;[e setComputePipelineState:p];bind(e);
    [e dispatchThreadgroups:groups threadsPerThreadgroup:threads];[e endEncoding];
    h3_gpu_stats st=g.stats;st.direct_dispatches++;g.stats=st;return 1;
    }
}

static int native_routed(H3GPU *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,uint32_t ql,uint32_t kl,uint32_t heads,
    float scale,int qi,int ki,int oi,int bq,int bk,int mode,
    h3_gpu_tensor *flags,h3_route_params route,h3_gpu_tensor *mx,h3_gpu_tensor *sum) {
    @autoreleasepool {
    if(!g||!ql||!kl||ql>10000000||kl>10000000||!heads||heads>65535||
       !isfinite(scale)||scale<=0||(qi!=0&&qi!=1)||(ki!=0&&ki!=1)||(oi!=0&&oi!=1)) {
        if(g)h3_gpu_set_error(g,@"Invalid native routed attention dimensions/layout/scale");return 0;
    }
    size_t nq=(size_t)ql*heads*128,nk=(size_t)kl*heads*128;
    if(!h3_gpu_require_bf16(g,q,nq,@"routed Q")||!h3_gpu_require_bf16(g,k,nk,@"routed K")||
       !h3_gpu_require_bf16(g,v,nk,@"routed V")||
       !(mode?h3_gpu_require_f32(g,out,nq,@"partial O"):h3_gpu_require_bf16(g,out,nq,@"routed O"))||
       !h3_gpu_require_command(g))return 0;
    if(TENSOR(out).buffer==TENSOR(q).buffer||TENSOR(out).buffer==TENSOR(k).buffer||TENSOR(out).buffer==TENSOR(v).buffer) {
        h3_gpu_set_error(g,@"Native attention requires distinct output storage");return 0;
    }
    NSString *name=mode?[NSString stringWithFormat:@"h3_sol_%d_%d",mode,bq]:
        (bq==32?@"h3_candidate_b":@"h3_candidate_b64");
    id<MTLComputePipelineState> pipeline=native_pipeline(g,name,(int)ql,(int)kl,bq,bk);
    if(!pipeline)return 0;
    if(pipeline.maxTotalThreadsPerThreadgroup<(NSUInteger)(bq*4)) {
        h3_gpu_set_error(g,@"Routed attention tile exceeds thread limit");return 0;
    }
    h3_native_attn_params p={.B=1,.H=(int)heads,.D=128,.qL=(int)ql,.kL=(int)kl,.gqa_factor=1,.scale=scale,
        .NQ=((int)ql+bq-1)/bq,.NK=((int)kl+bk-1)/bk,.NQ_aligned=(int)ql/bq,.NK_aligned=(int)kl/bk,
        .qL_rem=(int)ql%bq,.kL_rem=(int)kl%bk};
    p.Q_strides[0]=(int64_t)nq;p.Q_strides[1]=qi?(int64_t)ql*128:128;p.Q_strides[2]=qi?128:(int64_t)heads*128;
    p.K_strides[0]=(int64_t)nk;p.K_strides[1]=ki?(int64_t)kl*128:128;p.K_strides[2]=ki?128:(int64_t)heads*128;
    memcpy(p.V_strides,p.K_strides,sizeof(p.K_strides));
    p.O_strides[0]=(int64_t)nq;p.O_strides[1]=oi?(int64_t)ql*128:128;p.O_strides[2]=oi?128:(int64_t)heads*128;
    id<MTLComputeCommandEncoder> e=[g.command computeCommandEncoder];
    if(!e){h3_gpu_set_error(g,@"Cannot create routed attention encoder");return 0;}
    e.label=name;[e setComputePipelineState:pipeline];
    [e setBuffer:TENSOR(q).buffer offset:0 atIndex:0];[e setBuffer:TENSOR(k).buffer offset:0 atIndex:1];
    [e setBuffer:TENSOR(v).buffer offset:0 atIndex:2];[e setBuffer:TENSOR(out).buffer offset:0 atIndex:3];
    [e setBytes:&p length:sizeof(p) atIndex:4];
    [e setBuffer:TENSOR(flags?flags:q).buffer offset:0 atIndex:8];[e setBytes:&route length:sizeof(route) atIndex:9];
    [e setBuffer:TENSOR(mx?mx:out).buffer offset:0 atIndex:10];[e setBuffer:TENSOR(sum?sum:out).buffer offset:0 atIndex:11];
    [e dispatchThreadgroups:MTLSizeMake((NSUInteger)p.NQ,heads,1) threadsPerThreadgroup:MTLSizeMake(32,(NSUInteger)bq/8,1)];
    [e endEncoding];h3_gpu_stats st=g.stats;st.direct_dispatches++;if(!mode)st.native_attention_dispatches++;g.stats=st;return 1;
    }
}

int h3_gpu_native_sdpa_candidate_b(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t seq,uint32_t heads,float scale,int il,int ol,int qb) {
    @autoreleasepool {
    if(qb!=32&&qb!=64){h3_gpu_set_error(GPU(gpu),@"Candidate B query block must be 32 or 64");return 0;}
    return native_routed(GPU(gpu),out,q,k,v,seq,seq,heads,scale,il,il,ol,qb,qb==32?16:32,0,NULL,(h3_route_params){0},NULL,NULL);
    }
}

int h3_gpu_native_attention_layout(h3_gpu *gpu,const h3_sol_layout *l) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);
    if(!g||!l||!l->sequence||!l->query||!l->key||g.command||g.solSequence) {
        if(g)h3_gpu_set_error(g,@"SOL protection must be configured once before encoding");return 0;
    }
    h3_gpu_tensor *q=native_scratch(g,@"qmeta",(size_t)l->query_blocks*4,H3_GPU_U32);
    h3_gpu_tensor *k=native_scratch(g,@"kmeta",(size_t)l->key_blocks*4,H3_GPU_U32);
    h3_gpu_tensor *counts=native_scratch(g,@"counts",50*6*4,H3_GPU_U32);
    if(!q||!k||!counts)return 0;
    memcpy(TENSOR(q).buffer.contents,l->query,l->query_blocks*sizeof(h3_sol_block));
    memcpy(TENSOR(k).buffer.contents,l->key,l->key_blocks*sizeof(h3_sol_block));
    memset(TENSOR(counts).buffer.contents,0,50*6*4*sizeof(uint32_t));
    g.solSequence=l->sequence;g.solQueryBlocks=l->query_blocks;g.solKeyBlocks=l->key_blocks;g.solProtectedRows=l->protected_rows;
    g.solUsed=[NSMutableIndexSet new];return 1;
    }
}

static int mixed_sol_exact(H3GPU *g,h3_gpu_tensor *out,const h3_gpu_tensor *q,
    const h3_gpu_tensor *k,const h3_gpu_tensor *v,h3_gpu_tensor *ranges,uint32_t seq,
    uint32_t heads,float scale,int qb,h3_gpu_tensor *flags,h3_route_params route,
    h3_gpu_tensor *maxima,h3_gpu_tensor *sums);

static int native_sol_impl(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t seq,uint32_t heads,float scale,int il,int ol,h3_metal_attention_options o,unsigned block,int step,
    const h3_gpu_tensor *qh,const h3_gpu_tensor *kh,const h3_gpu_tensor *vh,h3_gpu_tensor *ranges) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);char error[256];
    if(!g)return 0;
    if(!h3_metal_options_valid(o,error,sizeof(error))||seq!=g.solSequence||block>=50||step<0||
       !heads||heads>65535||!isfinite(scale)||scale<=0||(il!=0&&il!=1)||(ol!=0&&ol!=1)) {
        h3_gpu_set_error(g,@"Invalid SOL configuration, protection geometry or block/step");return 0;
    }
    uint32_t nq=(seq+(uint32_t)o.q_block-1)/(uint32_t)o.q_block,nk=(seq+(uint32_t)o.kv_block-1)/(uint32_t)o.kv_block;
    if(nq!=g.solQueryBlocks||nk!=g.solKeyBlocks||(uint64_t)nq*nk*heads>500000000) {
        h3_gpu_set_error(g,@"SOL routing geometry mismatch or workspace exceeds its limit");return 0;
    }
    size_t rows=(size_t)seq*heads,n=rows*128;
    if(!h3_gpu_require_bf16(g,q,n,@"SOL Q")||!h3_gpu_require_bf16(g,k,n,@"SOL K")||
       !h3_gpu_require_bf16(g,v,n,@"SOL V")||
       !(ranges?h3_gpu_require_f32(g,out,n,@"SOL FP32 output"):h3_gpu_require_bf16(g,out,n,@"SOL output"))||
       !h3_gpu_require_command(g))return 0;
    if(TENSOR(out).buffer==TENSOR(q).buffer||TENSOR(out).buffer==TENSOR(k).buffer||TENSOR(out).buffer==TENSOR(v).buffer) {
        h3_gpu_set_error(g,@"SOL requires distinct output storage");return 0;
    }
    h3_gpu_tensor *qc=native_scratch(g,@"qc",(size_t)heads*nq*128,H3_GPU_BF16);
    h3_gpu_tensor *kc=native_scratch(g,@"kc",(size_t)heads*nk*128,H3_GPU_BF16);
    h3_gpu_tensor *vc=native_scratch(g,@"vc",(size_t)heads*nk*128,H3_GPU_BF16);
    h3_gpu_tensor *mv=native_scratch(g,@"meanvar",(size_t)heads*256,H3_GPU_F32);
    h3_gpu_tensor *flags=native_scratch(g,@"routes",((size_t)heads*nq*nk+3)/4,H3_GPU_U32);
    /* Head-major merge reads/writes the same element, so its exact partial
     * becomes the merged FP32 output. Row-major output needs separate storage
     * because its transpose would otherwise race the input reads. */
    h3_gpu_tensor *ex=native_scratch(g,ranges&&ol?@"fp16-output":@"exact",n,H3_GPU_F32);
    h3_gpu_tensor *ap=native_scratch(g,@"approx",n,H3_GPU_F32);
    h3_gpu_tensor *em=native_scratch(g,@"em",rows,H3_GPU_F32),*el=native_scratch(g,@"el",rows,H3_GPU_F32);
    h3_gpu_tensor *am=native_scratch(g,@"am",rows,H3_GPU_F32),*al=native_scratch(g,@"al",rows,H3_GPU_F32);
    h3_gpu_tensor *qm=g.nativeScratch.slots[@"qmeta"].pointerValue,*km=g.nativeScratch.slots[@"kmeta"].pointerValue;
    h3_gpu_tensor *counts=g.nativeScratch.slots[@"counts"].pointerValue;
    if(!qc||!kc||!vc||!mv||!flags||!ex||!ap||!em||!el||!am||!al||!qm||!km||!counts)return 0;
    /* Six ring slots cover the benchmark ceiling, and still allow ordinary
     * user renders with any legal number of evaluations. Report at each step. */
    uint32_t slot=(uint32_t)(step%6)*50+block;
    h3_sol_params p={seq,heads,(uint32_t)o.q_block,(uint32_t)o.kv_block,nq,nk,(uint32_t)il,
        (uint32_t)o.local_radius,slot*4,(uint32_t)ceilf(o.min_exact*(float)nk),scale,o.tau};
#define SOL_OP(label, call) do { h3_profile_span span=h3_profile_component_begin(gpu,label,block,step); \
    int ok=span.started>=0&&(call);int finished=h3_profile_component_end(gpu,span,label,block,step); \
    if(!ok||!finished)return 0; } while(0)
    const h3_gpu_tensor *inputs[3]={q,k,v};h3_gpu_tensor *outputs[3]={qc,kc,vc};
    h3_profile_span summary=h3_profile_component_begin(gpu,"SOL summaries",block,step);
    if(summary.started<0)return 0;
    for(unsigned i=0;i<3;i++) {
        uint32_t bs=i?(uint32_t)o.kv_block:(uint32_t)o.q_block,nb=i?nk:nq;
        const h3_gpu_tensor *in=inputs[i];h3_gpu_tensor *dst=outputs[i];
        if(!native_aux(g,@"h3_sol_summary",MTLSizeMake(nb,heads,1),MTLSizeMake(128,1,1),^(id<MTLComputeCommandEncoder> e){
            [e setBuffer:TENSOR(in).buffer offset:0 atIndex:0];[e setBuffer:TENSOR(dst).buffer offset:0 atIndex:1];
            [e setBytes:&p length:sizeof(p) atIndex:2];[e setBytes:&bs length:sizeof(bs) atIndex:3];}))return 0;
    }
    if(!h3_profile_component_end(gpu,summary,"SOL summaries",block,step))return 0;
    SOL_OP("SOL key statistics",native_aux(g,@"h3_sol_key_stats",MTLSizeMake(heads,1,1),MTLSizeMake(128,1,1),^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:TENSOR(kc).buffer offset:0 atIndex:0];[e setBuffer:TENSOR(mv).buffer offset:0 atIndex:1];[e setBytes:&p length:sizeof(p) atIndex:2];}));
    if(!native_aux(g,@"h3_sol_reset",MTLSizeMake(1,1,1),MTLSizeMake(4,1,1),^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:TENSOR(counts).buffer offset:0 atIndex:0];[e setBytes:&p.count_offset length:4 atIndex:1];}))return 0;
    SOL_OP("SOL routing",native_aux(g,@"h3_sol_route",MTLSizeMake(nq,heads,1),MTLSizeMake(32,1,1),^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:TENSOR(qc).buffer offset:0 atIndex:0];[e setBuffer:TENSOR(kc).buffer offset:0 atIndex:1];
        [e setBuffer:TENSOR(mv).buffer offset:0 atIndex:2];[e setBuffer:TENSOR(qm).buffer offset:0 atIndex:3];
        [e setBuffer:TENSOR(km).buffer offset:0 atIndex:4];[e setBuffer:TENSOR(flags).buffer offset:0 atIndex:5];
        [e setBuffer:TENSOR(counts).buffer offset:0 atIndex:6];[e setBytes:&p length:sizeof(p) atIndex:7];}));
    h3_route_params route={nq,nk,(uint32_t)o.kv_block,0};
    if(ranges) {
        SOL_OP("SOL recovery protection",native_aux(g,@"h3_sol_recovery_routes",MTLSizeMake(((size_t)heads*nq*nk+255)/256,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> e){
            h3_gpu_tensor *ts[]={flags,ranges,counts};
            for(unsigned i=0;i<3;i++)[e setBuffer:TENSOR(ts[i]).buffer offset:0 atIndex:i];
            [e setBytes:&p length:sizeof(p) atIndex:3];}));
        SOL_OP("SOL exact attention",mixed_sol_exact(g,ex,qh,kh,vh,ranges,seq,heads,scale,o.q_block,flags,route,em,el));
    } else {
    SOL_OP("SOL exact attention",native_routed(g,ex,q,k,v,seq,seq,heads,scale,il,il,1,o.q_block,32,1,flags,route,em,el));
    }
    SOL_OP("SOL approximate attention",native_routed(g,ap,q,kc,vc,seq,nk,heads,scale,il,1,1,o.q_block,32,2,flags,route,am,al));
    uint32_t output_major=(uint32_t)ol;
    SOL_OP("SOL merge",native_aux(g,ranges?@"h3_sol_merge_f32":@"h3_sol_merge",MTLSizeMake(1,seq,heads),MTLSizeMake(128,1,1),^(id<MTLComputeCommandEncoder> e){
        h3_gpu_tensor *ts[7]={ex,ap,em,el,am,al,out};
        for(unsigned i=0;i<7;i++)[e setBuffer:TENSOR(ts[i]).buffer offset:0 atIndex:i];
        [e setBytes:&p length:sizeof(p) atIndex:7];[e setBytes:&output_major length:4 atIndex:8];}));
#undef SOL_OP
    [g.solUsed addIndex:slot];
    if(!ranges){h3_gpu_stats st=g.stats;st.native_attention_dispatches++;g.stats=st;}
    return 1;
    }
}

int h3_gpu_native_sol_bf16(h3_gpu *gpu,h3_gpu_tensor *out,
    const h3_gpu_tensor *q,const h3_gpu_tensor *k,const h3_gpu_tensor *v,
    uint32_t seq,uint32_t heads,float scale,int il,int ol,h3_metal_attention_options o,unsigned block,int step) {
    @autoreleasepool {
    return native_sol_impl(gpu,out,q,k,v,seq,heads,scale,il,ol,o,block,step,NULL,NULL,NULL,NULL);
    }
}

void h3_gpu_native_sol_report(h3_gpu *gpu,int step) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g||step<0||!g.solUsed.count)return;
    h3_gpu_tensor *tensor=g.nativeScratch.slots[@"counts"].pointerValue;
    const uint32_t *counts=h3_gpu_tensor_contents(tensor);if(!counts)return;
    for(unsigned b=0;b<50;b++) {
        NSUInteger slot=(NSUInteger)(step%6)*50+b;
        if(![g.solUsed containsIndex:slot])continue;
        const uint32_t *c=counts+slot*4;
        fprintf(stderr,"h3_sol {\"step\":%d,\"block\":%u,\"exact\":%u,\"approximate\":%u,\"protected\":%u,\"local\":%u,\"protected_rows\":%u}\n",
            step+1,b,c[0],c[1],c[2],c[3],g.solProtectedRows);
        [g.solUsed removeIndex:slot];
    }
    }
}
