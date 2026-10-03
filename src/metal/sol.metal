// H3 SOL proof of concept. BF16 centroids, FP32 routing/statistics/merge.
// Routing uses the diagonal key-centroid variance estimate described in
// docs/metal/m1-attention-sources.md. No external runtime is required.
struct H3SolParams {
    uint sequence, heads, query_block, key_block, query_blocks, key_blocks;
    uint input_head_major, local_radius, count_offset, minimum_exact;
    float scale, tau;
};
struct H3SolBlock { uint protect; int first_frame,last_frame; uint rows; };

kernel void h3_sol_summary(
    device const bfloat* input [[buffer(0)]],
    device bfloat* output [[buffer(1)]],
    constant H3SolParams& p [[buffer(2)]],
    constant uint& block_size [[buffer(3)]],
    uint3 gid [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    uint h=gid.y,b=gid.x,start=b*block_size,len=min(block_size,p.sequence-start);
    uint blocks=(p.sequence+block_size-1)/block_size;
    float sum=0;
    for(uint i=0;i<len;i++) {
        ulong index=p.input_head_major?(ulong(h)*p.sequence+start+i)*128+d:
            (ulong(start+i)*p.heads+h)*128+d;
        sum+=float(input[index]);
    }
    output[(ulong(h)*blocks+b)*128+d]=bfloat(sum/float(len));
}

kernel void h3_sol_key_stats(
    device const bfloat* kc [[buffer(0)]], device float* mean_var [[buffer(1)]],
    constant H3SolParams& p [[buffer(2)]],
    uint h [[threadgroup_position_in_grid]], uint d [[thread_index_in_threadgroup]]) {
    float sum=0,sq=0;
    for(uint b=0;b<p.key_blocks;b++) {
        float x=float(kc[(ulong(h)*p.key_blocks+b)*128+d]);sum+=x;sq+=x*x;
    }
    float mean=sum/float(p.key_blocks);
    mean_var[h*256+d]=mean;
    mean_var[h*256+128+d]=max(sq/float(p.key_blocks)-mean*mean,0.0f);
}

kernel void h3_sol_reset(device uint* counts [[buffer(0)]],
                          constant uint& offset [[buffer(1)]],
                          uint t [[thread_position_in_grid]]) {
    if(t<4)counts[offset+t]=0;
}

kernel void h3_sol_route(
    device const bfloat* qc [[buffer(0)]], device const bfloat* kc [[buffer(1)]],
    device const float* mean_var [[buffer(2)]],
    device const H3SolBlock* qm [[buffer(3)]], device const H3SolBlock* km [[buffer(4)]],
    device uchar* flags [[buffer(5)]], device atomic_uint* counts [[buffer(6)]],
    constant H3SolParams& p [[buffer(7)]],
    uint2 group [[threadgroup_position_in_grid]],uint lane [[thread_index_in_simdgroup]]) {
    uint qb=group.x,h=group.y;
    float q[4],mean=0,var=0;
    for(uint d=0;d<4;d++) {
        uint c=lane+32*d;q[d]=float(qc[(ulong(h)*p.query_blocks+qb)*128+c]);
        mean+=q[d]*mean_var[h*256+c];var+=q[d]*q[d]*mean_var[h*256+128+c];
    }
    float ls=p.scale*M_LOG2E_F;
    mean=simd_sum(mean)*ls;var=max(simd_sum(var),0.0f)*ls*ls;
    float threshold=mean+p.tau*sqrt(var+1.0e-6f);
    H3SolBlock query=qm[qb];
    uint kept=0,protected_count=0,local_count=0;
    for(uint k=0;k<p.key_blocks;k++) {
        float dot=0;
        for(uint d=0;d<4;d++)dot+=q[d]*float(kc[(ulong(h)*p.key_blocks+k)*128+lane+32*d]);
        dot=simd_sum(dot)*ls;
        H3SolBlock key=km[k];
        bool protected_pair=query.protect||key.protect;
        bool local=query.first_frame>=0&&key.first_frame>=0&&
            query.first_frame<=key.last_frame+int(p.local_radius)&&
            key.first_frame<=query.last_frame+int(p.local_radius);
        // Spread the guaranteed minimum across K/V rather than privileging a
        // packed prefix. Exactly minimum_exact positions satisfy this predicate.
        bool floor=(ulong(k+1)*p.minimum_exact/p.key_blocks)!=(ulong(k)*p.minimum_exact/p.key_blocks);
        bool exact=protected_pair||local||floor||key.rows<p.key_block||!isfinite(dot)||!isfinite(threshold)||dot>threshold;
        kept+=uint(exact);protected_count+=uint(protected_pair);local_count+=uint(local&&!protected_pair);
        if(lane==0)flags[(ulong(h)*p.query_blocks+qb)*p.key_blocks+k]=uchar(exact);
    }
    if(lane==0) {
        atomic_fetch_add_explicit(counts+p.count_offset,kept,memory_order_relaxed);
        atomic_fetch_add_explicit(counts+p.count_offset+1,p.key_blocks-kept,memory_order_relaxed);
        atomic_fetch_add_explicit(counts+p.count_offset+2,protected_count,memory_order_relaxed);
        atomic_fetch_add_explicit(counts+p.count_offset+3,local_count,memory_order_relaxed);
    }
}

template<typename Output>
kernel void h3_sol_merge_t(
    device const float* exact [[buffer(0)]], device const float* approximate [[buffer(1)]],
    device const float* em [[buffer(2)]], device const float* el [[buffer(3)]],
    device const float* am [[buffer(4)]], device const float* al [[buffer(5)]],
    device Output* output [[buffer(6)]], constant H3SolParams& p [[buffer(7)]],
    constant uint& output_head_major [[buffer(8)]], uint3 gid [[thread_position_in_grid]]) {
    uint d=gid.x,row=gid.y,h=gid.z;
    if(d>=128||row>=p.sequence||h>=p.heads)return;
    ulong stat=ulong(h)*p.sequence+row,idx=stat*128+d;
    float a=el[stat],b=al[stat];
    float ma=a>0?em[stat]:-FLT_MAX,mb=b>0?am[stat]+log2(float(p.key_block)):-FLT_MAX;
    float m=max(ma,mb),wa=a>0?fast::exp2(ma-m):0.0f,wb=b>0?fast::exp2(mb-m):0.0f;
    float denominator=a*wa+b*wb;
    float value=denominator>0?(exact[idx]*wa+approximate[idx]*wb)/denominator:0.0f;
    ulong out=output_head_major?idx:(ulong(row)*p.heads+h)*128+d;
    output[out]=Output(value);
}

template [[host_name("h3_sol_merge")]] [[kernel]]
decltype(h3_sol_merge_t<bfloat>) h3_sol_merge_t<bfloat>;
template [[host_name("h3_sol_merge_f32")]] [[kernel]]
decltype(h3_sol_merge_t<float>) h3_sol_merge_t<float>;

// Range-recovery heads never approximate. Each thread owns one route byte;
// update the original router counters to describe the work actually executed.
kernel void h3_sol_recovery_routes(device uchar* flags [[buffer(0)]],
    device const H3FP16Range* ranges [[buffer(1)]],
    device atomic_uint* counts [[buffer(2)]], constant H3SolParams& p [[buffer(3)]],
    uint index [[thread_position_in_grid]]) {
    uint per_head=p.query_blocks*p.key_blocks;
    if(index>=per_head*p.heads || !ranges[index/per_head].flags)return;
    if(!flags[index]) {
        flags[index]=1;
        atomic_fetch_add_explicit(counts+p.count_offset,1u,memory_order_relaxed);
        atomic_fetch_sub_explicit(counts+p.count_offset+1,1u,memory_order_relaxed);
    }
}
