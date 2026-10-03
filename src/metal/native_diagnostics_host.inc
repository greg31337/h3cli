/* Diagnostic range scans and independent sampled softmax. No host wait until
 * forward completion; no measured values feed model arithmetic. */
typedef struct {float lo,hi,minimum,pad;uint32_t count,nonfinite,over,under;} h3_diag_range;
typedef struct {float lo,hi,minimum,sum;uint32_t nonfinite,zeros,under,keys;} h3_diag_score;
_Static_assert(sizeof(h3_diag_range)==32 && sizeof(h3_diag_score)==32,"diagnostic ABI");
static int diagnostic_budget(H3GPU *g) {
    @autoreleasepool {
    const char *p=getenv("H3_TEST_MAX_EVALUATIONS");
    if(!p||strcmp(p,"6")){h3_gpu_set_error(g,@"Range diagnostics require six-evaluation test budget");return 0;}
    return 1;
    }
}
int h3_gpu_diagnostic_range(h3_gpu *gpu,const h3_gpu_tensor *input,uint32_t count,
    uint32_t width,uint32_t stride,uint32_t offset,unsigned block,unsigned stage) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g||!count||!width||stride<width||block>=50||stage>=12)return 0;
    if(!diagnostic_budget(g)||!h3_gpu_require_bf16(g,input,(size_t)((count-1)/width)*stride+
        (count-1)%width+offset+1,@"diagnostic tensor"))return 0;
    h3_gpu_tensor *out=native_scratch(g,@"diagnostic-ranges",50*12*256*sizeof(h3_diag_range)/4,H3_GPU_U32);
    if(!out)return 0;
    struct {uint32_t count,width,stride,offset,record,groups;} p={count,width,stride,offset,block*12+stage,256};
    if(!native_aux(g,@"h3_diagnostic_range",MTLSizeMake(256,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> e){
        [e setBuffer:TENSOR(input).buffer offset:0 atIndex:0];[e setBuffer:TENSOR(out).buffer offset:0 atIndex:1];
        [e setBytes:&p length:sizeof(p) atIndex:2];}))return 0;
    if(!g.diagnosticRanges)g.diagnosticRanges=[NSMutableIndexSet new];[g.diagnosticRanges addIndex:p.record];return 1;
    }
}
int h3_gpu_diagnostic_scores(h3_gpu *gpu,const h3_gpu_tensor *q,const h3_gpu_tensor *k,
    uint32_t sequence,uint32_t heads,float scale,int major,unsigned block) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g||!sequence||!heads||heads>256||block>=50||!diagnostic_budget(g))return 0;
    size_t n=(size_t)sequence*heads*128;
    if(!h3_gpu_require_bf16(g,q,n,@"score Q")||!h3_gpu_require_bf16(g,k,n,@"score K"))return 0;
    h3_gpu_tensor *scratch=native_scratch(g,@"diagnostic-score-scratch",(size_t)sequence*heads*16,H3_GPU_F32);
    h3_gpu_tensor *out=native_scratch(g,@"diagnostic-scores",50*heads*16*sizeof(h3_diag_score)/4,H3_GPU_U32);
    if(!scratch||!out)return 0;
    struct {uint32_t sequence,heads,major,record;float scale;} p={sequence,heads,(uint32_t)major,block,scale};
    if(!native_aux(g,@"h3_diagnostic_score",MTLSizeMake(heads*16,1,1),MTLSizeMake(256,1,1),^(id<MTLComputeCommandEncoder> e){
        const h3_gpu_tensor *ts[]={q,k,scratch,out};for(unsigned i=0;i<4;i++)[e setBuffer:TENSOR(ts[i]).buffer offset:0 atIndex:i];
        [e setBytes:&p length:sizeof(p) atIndex:4];}))return 0;
    if(!g.diagnosticScores)g.diagnosticScores=[NSMutableIndexSet new];
    [g.diagnosticScores addIndex:block];g.diagnosticHeads=heads;return 1;
    }
}
int h3_gpu_diagnostic_finish(h3_gpu *gpu,int step) {
    @autoreleasepool {
    H3GPU *g=GPU(gpu);if(!g||step<0||step>=6)return 0;
    const char *names[]={"residual-input","raw-q","raw-k","raw-v","rope-q","rope-k","value",
        "attention-output","projection-output","attention-residual","mlp-output","residual-output"};
    h3_gpu_tensor *buffer=g.nativeScratch.slots[@"diagnostic-ranges"].pointerValue;
    const h3_diag_range *ranges=buffer?h3_gpu_tensor_contents(buffer):NULL;
    if(!ranges||!g.diagnosticRanges.count)return 0;
    int ok=1;
    for(NSUInteger slot=g.diagnosticRanges.firstIndex;slot!=NSNotFound;slot=[g.diagnosticRanges indexGreaterThanIndex:slot]) {
        float lo=FLT_MAX,hi=-FLT_MAX,mn=FLT_MAX;uint64_t count=0,bad=0,over=0,under=0;
        for(unsigned i=0;i<256;i++){const h3_diag_range *r=ranges+slot*256+i;
            lo=fminf(lo,r->lo);hi=fmaxf(hi,r->hi);mn=fminf(mn,r->minimum);
            count+=r->count;bad+=r->nonfinite;over+=r->over;under+=r->under;}
        if(mn==FLT_MAX)mn=0;
        fprintf(stderr,"h3_range {\"step\":%d,\"block\":%lu,\"stage\":\"%s\",\"count\":%llu,\"minimum\":%.9g,\"maximum\":%.9g,\"minimum_nonzero\":%.9g,\"nonfinite\":%llu,\"over_fp16\":%llu,\"below_half_normal\":%llu}\n",
            step+1,(unsigned long)(slot/12),names[slot%12],(unsigned long long)count,lo,hi,mn,
            (unsigned long long)bad,(unsigned long long)over,(unsigned long long)under);
        if(bad||!count)ok=0;
    }
    [g.diagnosticRanges removeAllIndexes];
    buffer=g.nativeScratch.slots[@"diagnostic-scores"].pointerValue;
    const h3_diag_score *scores=buffer?h3_gpu_tensor_contents(buffer):NULL;
    if(!scores||!g.diagnosticScores.count)return 0;
    for(NSUInteger b=g.diagnosticScores.firstIndex;b!=NSNotFound;b=[g.diagnosticScores indexGreaterThanIndex:b]) {
        float lo=FLT_MAX,hi=-FLT_MAX,mn=FLT_MAX,sm=FLT_MAX,sx=0;uint64_t bad=0,zeros=0,under=0,keys=0;
        for(unsigned i=0;i<g.diagnosticHeads*16;i++){const h3_diag_score *r=scores+b*g.diagnosticHeads*16+i;
            lo=fminf(lo,r->lo);hi=fmaxf(hi,r->hi);mn=fminf(mn,r->minimum);sm=fminf(sm,r->sum);sx=fmaxf(sx,r->sum);
            bad+=r->nonfinite;zeros+=r->zeros;under+=r->under;keys+=r->keys;
            if(!isfinite(r->sum)||r->sum<1)ok=0;}
        fprintf(stderr,"h3_score {\"step\":%d,\"block\":%lu,\"query_rows_per_head\":16,\"heads\":%u,\"score_count\":%llu,\"minimum\":%.9g,\"maximum\":%.9g,\"minimum_exp\":%.9g,\"sum_minimum\":%.9g,\"sum_maximum\":%.9g,\"nonfinite\":%llu,\"zero_exp\":%llu,\"below_half_normal_exp\":%llu,\"scope\":\"independent BF16/FP32 probe; all K, sampled Q\"}\n",
            step+1,(unsigned long)b,g.diagnosticHeads,(unsigned long long)keys,lo,hi,mn,sm,sx,
            (unsigned long long)bad,(unsigned long long)zeros,(unsigned long long)under);if(bad)ok=0;
    }
    [g.diagnosticScores removeAllIndexes];
    if(!ok)h3_gpu_set_error(g,@"Diagnostic range/score check failed");return ok;
    }
}
