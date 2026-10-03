/* Portable SM86+ online softmax. Scratch is 16 scores per query/head, never
 * sequence squared. Both softmax state and weighted values accumulate in F32. */
template<class T> __device__ float attention_float(T value){return float(value);}
template<> __device__ float attention_float(ushort value){return h3_bf16_to_f32(value);}
template<class T> __device__ T attention_store(float value){return T(value);}
template<> __device__ ushort attention_store(float value){return h3_f32_to_bf16(value);}
template<class T> __global__ void attention_online(T*out,const T*q,const T*k,const T*v,
    uint32_t sequence,uint32_t heads,uint32_t kv_heads,uint32_t dim,float scale,
    bool causal,bool head_major,int scale_mode) {
    uint row=blockIdx.x,head=blockIdx.y,batch=blockIdx.z,tid=threadIdx.x;
    uint lane=tid%32,warp=tid/32,kv_head=head/(heads/kv_heads);
    uint count=causal?row+1:sequence;
    size_t qbase=(((size_t)batch*sequence+row)*heads+head)*dim;
    __shared__ float weights[16],alpha,denominator,maximum;
    if(!tid){maximum=-INFINITY;denominator=0;}
    float sum[2]={0,0};
    __syncthreads();
    for(uint first=0;first<count;first+=16){
        for(uint j=warp;j<16;j+=4){
            uint keyrow=first+j;float dot=0;
            if(keyrow<count){
                size_t kbase=(((size_t)batch*sequence+keyrow)*kv_heads+kv_head)*dim;
                for(uint d=lane;d<dim;d+=32){float query=attention_float(q[qbase+d]);
                    if(scale_mode==1)query*=scale;
                    if(scale_mode==2)query=h3_bf16_to_f32(h3_f32_to_bf16(query*scale));
                    dot=fmaf(query,attention_float(k[kbase+d]),dot);
                }
                for(int shift=16;shift;shift/=2)dot+=__shfl_down_sync(0xffffffff,dot,shift);
                if(scale_mode==0)dot*=scale;
            }else dot=-INFINITY;
            if(!lane)weights[j]=dot;
        }
        __syncthreads();
        if(!tid){
            float next=maximum;for(uint j=0;j<16;j++)next=fmaxf(next,weights[j]);
            alpha=expf(maximum-next);float tile_sum=0;
            for(uint j=0;j<16;j++){weights[j]=expf(weights[j]-next);tile_sum+=weights[j];}
            denominator=denominator*alpha+tile_sum;maximum=next;
        }
        __syncthreads();
        for(uint d=tid;d<dim;d+=128){float value=sum[d/128]*alpha;
            for(uint j=0;j<16&&first+j<count;j++){
                size_t idx=(((size_t)batch*sequence+first+j)*kv_heads+kv_head)*dim+d;
                value=fmaf(weights[j],attention_float(v[idx]),value);
            }
            sum[d/128]=value;
        }
        __syncthreads();
    }
    for(uint d=tid;d<dim;d+=128){size_t at=head_major?(((size_t)batch*heads+head)*sequence+row)*dim+d:qbase+d;out[at]=attention_store<T>(sum[d/128]/denominator);}
}

// Each CTA reuses a 16-key/value tile across 16 queries. QK uses exact BF16
// products and F32 tensor-core accumulation; softmax and PV stay entirely F32.
__global__ void attention_bf16_tiled(ushort *out,const ushort *q,const ushort *k,const ushort *v,
    uint sequence,uint heads,uint dim,float scale,bool head_major) {
    const uint tid=threadIdx.x,first=blockIdx.x*16,head=blockIdx.y;
    const size_t batch_base=(size_t)blockIdx.z*sequence*heads*dim;
    __shared__ __align__(32) __nv_bfloat16 qs[16*256],ks[16*256],vs[16*256];
    __shared__ __align__(32) float scores[16*16],alpha[16],denom[16],maximum[16];
    for(uint x=tid;x<16*dim;x+=128){uint row=x/dim,d=x%dim;
        qs[x]=__ushort_as_bfloat16(first+row<sequence?q[batch_base+((size_t)(first+row)*heads+head)*dim+d]:0);}
    if(tid<16){maximum[tid]=-INFINITY;denom[tid]=0;}
    float result[32]={};
    __syncthreads();
    for(uint start=0;start<sequence;start+=16){
        for(uint x=tid;x<16*dim;x+=128){uint row=x/dim,d=x%dim;
            size_t at=batch_base+((size_t)(start+row)*heads+head)*dim+d;
            ks[x]=__ushort_as_bfloat16(start+row<sequence?k[at]:0);
            vs[x]=__ushort_as_bfloat16(start+row<sequence?v[at]:0);}
        __syncthreads();
        if(tid<32){
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> b;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> c;
            nvcuda::wmma::fill_fragment(c,0.0f);
            for(uint d=0;d<dim;d+=16){nvcuda::wmma::load_matrix_sync(a,qs+d,dim);nvcuda::wmma::load_matrix_sync(b,ks+d,dim);nvcuda::wmma::mma_sync(c,a,b,c);}
            nvcuda::wmma::store_matrix_sync(scores,c,16,nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        if(tid<16){float next=maximum[tid];
            for(uint j=0;j<16;j++){float s=start+j<sequence?scores[tid*16+j]*scale:-INFINITY;scores[tid*16+j]=s;next=fmaxf(next,s);}
            alpha[tid]=expf(maximum[tid]-next);float total=0;
            for(uint j=0;j<16;j++){float p=expf(scores[tid*16+j]-next);scores[tid*16+j]=p;total+=p;}
            denom[tid]=denom[tid]*alpha[tid]+total;maximum[tid]=next;}
        __syncthreads();
        for(uint x=tid;x<16*dim;x+=128){uint row=x/dim,d=x%dim;float sum=result[x/128]*alpha[row];
            for(uint j=0;j<16;j++)sum=fmaf(scores[row*16+j],__bfloat162float(vs[j*dim+d]),sum);
            result[x/128]=sum;}
        __syncthreads();
    }
    for(uint x=tid;x<16*dim;x+=128){uint row=x/dim,d=x%dim;if(first+row<sequence){size_t at=batch_base+(head_major?((size_t)head*sequence+first+row)*dim+d:((size_t)(first+row)*heads+head)*dim+d);out[at]=h3_f32_to_bf16(result[x/128]/denom[row]);}}
}

// SM90/SM120 tuning for the DiT's 128-wide heads. A fixed dimension halves shared
// storage, and statically indexed accumulators stay in registers. QK, softmax,
// and each F32 PV accumulation retain the portable kernel's operation order.
// Keep the portable implementation above as the independent numerical control.
// SM90 uses 512 threads to reduce per-thread F32 PV work; SM120 retains 128.
template<unsigned THREADS>
__global__ void attention_bf16_fixed128(ushort *out,const ushort *q,const ushort *k,const ushort *v,
    uint sequence,uint heads,float scale,bool head_major) {
    static_assert(THREADS==128 || THREADS==512, "qualified fixed-head launch size");
    constexpr uint dim=128;
    const uint tid=threadIdx.x,first=blockIdx.x*16,head=blockIdx.y;
    const size_t batch_base=(size_t)blockIdx.z*sequence*heads*dim;
    __shared__ __align__(32) __nv_bfloat16 qs[16*dim],ks[16*dim],vs[16*dim];
    __shared__ __align__(32) float scores[16*16],alpha[16],denom[16],maximum[16];
    #pragma unroll
    for(uint x=tid;x<16*dim;x+=THREADS){uint row=x/dim,d=x%dim;
        qs[x]=__ushort_as_bfloat16(first+row<sequence?q[batch_base+((size_t)(first+row)*heads+head)*dim+d]:0);}
    if(tid<16){maximum[tid]=-INFINITY;denom[tid]=0;}
    float result[16*dim/THREADS]={};
    __syncthreads();
    for(uint start=0;start<sequence;start+=16){
        #pragma unroll
        for(uint x=tid;x<16*dim;x+=THREADS){uint row=x/dim,d=x%dim;
            size_t at=batch_base+((size_t)(start+row)*heads+head)*dim+d;
            ks[x]=__ushort_as_bfloat16(start+row<sequence?k[at]:0);
            vs[x]=__ushort_as_bfloat16(start+row<sequence?v[at]:0);}
        __syncthreads();
        if(tid<32){
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_a,16,16,16,__nv_bfloat16,nvcuda::wmma::row_major> a;
            nvcuda::wmma::fragment<nvcuda::wmma::matrix_b,16,16,16,__nv_bfloat16,nvcuda::wmma::col_major> b;
            nvcuda::wmma::fragment<nvcuda::wmma::accumulator,16,16,16,float> c;
            nvcuda::wmma::fill_fragment(c,0.0f);
            for(uint d=0;d<dim;d+=16){nvcuda::wmma::load_matrix_sync(a,qs+d,dim);nvcuda::wmma::load_matrix_sync(b,ks+d,dim);nvcuda::wmma::mma_sync(c,a,b,c);}
            nvcuda::wmma::store_matrix_sync(scores,c,16,nvcuda::wmma::mem_row_major);
        }
        __syncthreads();
        if(tid<16){float next=maximum[tid];
            for(uint j=0;j<16;j++){float s=start+j<sequence?scores[tid*16+j]*scale:-INFINITY;scores[tid*16+j]=s;next=fmaxf(next,s);}
            alpha[tid]=expf(maximum[tid]-next);float total=0;
            for(uint j=0;j<16;j++){float p=expf(scores[tid*16+j]-next);scores[tid*16+j]=p;total+=p;}
            denom[tid]=denom[tid]*alpha[tid]+total;maximum[tid]=next;}
        __syncthreads();
        #pragma unroll
        for(uint i=0;i<16*dim/THREADS;i++){uint x=tid+i*THREADS;uint row=x/dim,d=x%dim;float sum=result[i]*alpha[row];
            for(uint j=0;j<16;j++)sum=fmaf(scores[row*16+j],__bfloat162float(vs[j*dim+d]),sum);
            result[i]=sum;}
        __syncthreads();
    }
    #pragma unroll
    for(uint i=0;i<16*dim/THREADS;i++){uint x=tid+i*THREADS;uint row=x/dim,d=x%dim;if(first+row<sequence){size_t at=batch_base+(head_major?((size_t)head*sequence+first+row)*dim+d:((size_t)(first+row)*heads+head)*dim+d);out[at]=h3_f32_to_bf16(result[i]/denom[row]);}}
}
