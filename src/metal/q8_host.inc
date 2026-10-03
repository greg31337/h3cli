/* Included after the Metal tensor/linear helpers. Only explicit DiT Q8 loads
 * create these descriptors; BF16 callers retain their existing dispatch. */
h3_gpu_tensor *h3_gpu_q8_load(h3_gpu *opaque,const char *path,uint64_t offset,
                              uint32_t rows,uint32_t columns) {
    @autoreleasepool {
    H3GPU *g=GPU(opaque); size_t scale_offset,bytes;
    uint64_t source_bytes=(uint64_t)rows*columns*2;
    if(!g||!path||columns>4u*1024*1024||!h3_q8_layout(rows,columns,&scale_offset,&bytes)||
       offset>INT64_MAX||source_bytes>INT64_MAX-offset||g.command) {
        if(g)h3_gpu_set_error(g,@"Invalid Q8 source/shape or active load command");return NULL;
    }
    double started=h3_gpu_now();
    int fd=open(path,O_RDONLY);struct stat st;
    if(fd<0||fstat(fd,&st)||st.st_size<0||(uint64_t)st.st_size<offset+source_bytes) {
        if(fd>=0)close(fd);h3_gpu_set_error(g,@"Cannot read complete Q8 source: %s",path);return NULL;
    }
    /* At most 8 MiB of BF16 staging, even for the large FC1 matrix. */
    uint32_t chunk_rows=(uint32_t)((8u*1024*1024)/((size_t)columns*2));
    if(!chunk_rows)chunk_rows=1;
    if(chunk_rows>rows)chunk_rows=rows;
    uint16_t *source=malloc((size_t)chunk_rows*columns*2);
    h3_gpu_tensor *out=source?h3_gpu_tensor_new_i8(opaque,bytes):NULL;
    if(!out) {
        free(source);close(fd);h3_gpu_set_error(g,@"Cannot allocate bounded Q8 load storage");return NULL;
    }
    int8_t *q=TENSOR(out).buffer.contents;
    float *scales=(float *)((char *)q+scale_offset);
    memset(q+(size_t)rows*columns,0,scale_offset-(size_t)rows*columns);
    size_t groups=((size_t)columns+H3_Q8_GROUP-1)/H3_Q8_GROUP;
    int ok=1;
    for(uint32_t first=0;ok&&first<rows;) {
        uint32_t count=MIN(chunk_rows,rows-first);
        size_t wanted=(size_t)count*columns*2,done=0;
        while(done<wanted) {
            ssize_t n=pread(fd,(char *)source+done,wanted-done,
                (off_t)(offset+(uint64_t)first*columns*2+done));
            if(n<0&&errno==EINTR)continue;
            if(n<=0){ok=0;break;}done+=(size_t)n;
        }
        if(ok)ok=h3_q8_pack_rows(source,q+(size_t)first*columns,
            scales+(size_t)first*groups,count,columns);
        first+=count;
    }
    free(source);close(fd);
    if(!ok){h3_gpu_tensor_free(out);h3_gpu_set_error(g,@"Q8 source read failed or contains nonfinite weights: %s",path);return NULL;}
    TENSOR(out).q8Rows=rows;TENSOR(out).q8Columns=columns;TENSOR(out).q8ScaleOffset=scale_offset;
    h3_gpu_stats stats=g.stats;stats.q8_weights++;stats.q8_source_bytes+=source_bytes;
    stats.q8_storage_bytes+=bytes;stats.q8_load_seconds+=h3_gpu_now()-started;g.stats=stats;
    return out;
    }
}

static h3_gpu_tensor *native_scratch(H3GPU *,NSString *,size_t,h3_gpu_dtype);
static int h3_gpu_linear_q8(H3GPU *g,h3_gpu_tensor *output,
    const h3_gpu_tensor *input,const h3_gpu_tensor *weight,const h3_gpu_tensor *bias,
    uint32_t rows,uint32_t columns,uint32_t outputs) {
    @autoreleasepool {
    size_t scale_offset,bytes;
    if(!g||!rows||!h3_q8_layout(outputs,columns,&scale_offset,&bytes)||
       (uint64_t)rows*columns>UINT32_MAX-31u||(uint64_t)rows*outputs>UINT32_MAX-31u||
       TENSOR(weight).q8Rows!=outputs||TENSOR(weight).q8Columns!=columns||
       TENSOR(weight).q8ScaleOffset!=scale_offset) {
        if(g)h3_gpu_set_error(g,@"Invalid Q8 linear shape/descriptor");return 0;
    }
    if(!h3_gpu_require_bf16(g,input,(size_t)rows*columns,@"Q8 activation")||
       !h3_gpu_require_bf16(g,output,(size_t)rows*outputs,@"Q8 output")||
       !h3_gpu_require_i8(g,weight,bytes,@"Q8 weight")||
       (bias&&!h3_gpu_require_bf16(g,bias,outputs,@"Q8 bias")))return 0;
    if(TENSOR(input).owner!=g||TENSOR(output).owner!=g||TENSOR(weight).owner!=g||
       (bias&&TENSOR(bias).owner!=g)||TENSOR(output).buffer==TENSOR(input).buffer||
       TENSOR(output).buffer==TENSOR(weight).buffer||
       (bias&&TENSOR(output).buffer==TENSOR(bias).buffer)) {
        h3_gpu_set_error(g,@"Q8 linear requires same-context tensors and distinct output storage");return 0;
    }
    if(g.nativeScope.metal.q8_kernel==0) {
        NSString *name=[NSString stringWithFormat:@"q8-dequant-%u-%u",outputs,columns];
        h3_gpu_tensor *expanded=native_scratch(g,name,(size_t)outputs*columns,H3_GPU_BF16);
        if(!expanded)return 0;
        struct {uint32_t rows,columns;} shape={outputs,columns};
        if(!h3_gpu_dispatch_1d(g,@"h3_q8_dequant",outputs*columns,^(id<MTLComputeCommandEncoder> e){
            [e setBuffer:TENSOR(weight).buffer offset:0 atIndex:0];
            [e setBuffer:TENSOR(weight).buffer offset:scale_offset atIndex:1];
            [e setBuffer:TENSOR(expanded).buffer offset:0 atIndex:2];
            [e setBytes:&shape length:sizeof(shape) atIndex:3];
        }))return 0;
        if(!h3_gpu_linear_mps(g,output,input,expanded,bias,rows,columns,outputs,MPSDataTypeBFloat16))return 0;
        h3_gpu_stats stats=g.stats;stats.q8_dispatches++;g.stats=stats;return 1;
    }
    id<MTLComputePipelineState> pipeline=h3_gpu_pipeline(g,@"h3_linear_q8_bf16");
    if(!pipeline||pipeline.threadExecutionWidth!=32||pipeline.maxTotalThreadsPerThreadgroup<128||
       pipeline.staticThreadgroupMemoryLength>g.device.maxThreadgroupMemoryLength) {
        h3_gpu_set_error(g,@"Device cannot dispatch Q8 simdgroup linear");return 0;
    }
    if(!h3_gpu_require_command(g))return 0;
    id<MTLComputeCommandEncoder> e=[g.command computeCommandEncoder];
    if(!e){h3_gpu_set_error(g,@"Cannot create Q8 linear encoder");return 0;}
    linear_args args={rows,columns,outputs,bias?1u:0u};
    e.label=@"h3 Q8/BF16 simdgroup linear";[e setComputePipelineState:pipeline];
    [e setBuffer:TENSOR(input).buffer offset:0 atIndex:0];
    [e setBuffer:TENSOR(weight).buffer offset:0 atIndex:1];
    [e setBuffer:TENSOR(bias?bias:output).buffer offset:0 atIndex:2];
    [e setBuffer:TENSOR(output).buffer offset:0 atIndex:3];
    [e setBytes:&args length:sizeof(args) atIndex:4];
    [e setBuffer:TENSOR(weight).buffer offset:scale_offset atIndex:5];
    [e dispatchThreadgroups:MTLSizeMake((outputs+31u)/32u,(rows+31u)/32u,1)
        threadsPerThreadgroup:MTLSizeMake(128,1,1)];
    [e endEncoding];h3_gpu_stats stats=g.stats;stats.direct_dispatches++;stats.q8_dispatches++;g.stats=stats;
    return 1;
    }
}
