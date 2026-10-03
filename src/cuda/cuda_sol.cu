// CUDA SOL recipe 1. Adapted MMA/online-softmax primitives retain the
// existing fast-attention attribution below. Routing/centroid integration is
// local H3 code; no additional third-party runtime or source is imported.
/* Opt-in fast CUDA attention. Adapted from QuixiAI/h3.c commit
 * 69172740de9cf1bb12c8718479cefe47eeaf7a19, src/cuda/cuda_kernels.cuh.
 * Modifications: H3 types/dispatch integration; explicit asynchronous copy
 * group commits and final-stage drain. Original MIT license follows.
 *
 * MIT License
 * 
 * Copyright (c) 2026 Salvatore Sanfilippo
 * 
 * Permission is hereby granted, free of charge, to any person obtaining a copy
 * of this software and associated documentation files (the "Software"), to deal
 * in the Software without restriction, including without limitation the rights
 * to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
 * copies of the Software, and to permit persons to whom the Software is
 * furnished to do so, subject to the following conditions:
 * 
 * The above copyright notice and this permission notice shall be included in all
 * copies or substantial portions of the Software.
 * 
 * THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
 * IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
 * FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
 * AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
 * LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
 * OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
 * SOFTWARE.
 */


#include "src/cuda/cuda_sol.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstring>
#include <stdexcept>
#include <vector>
using ushort=unsigned short;
using counter=unsigned long long;
static_assert(sizeof(counter)==8 && sizeof(h3_sol_block)==16,"SOL ABI");
__device__ __forceinline__ float sol_float(ushort x){return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&x));}
__device__ __forceinline__ ushort sol_bf16(float x){auto b=__float2bfloat16_rn(x);return *reinterpret_cast<ushort*>(&b);}
__device__ __forceinline__ void sol_ldmatrix_a(uint32_t a[4],
                                               const ushort *row_pointer) {
    uint32_t address =
        (uint32_t)__cvta_generic_to_shared(row_pointer);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
}

__device__ __forceinline__ void sol_ldmatrix_b(uint32_t b[2],
                                               const ushort *row_pointer) {
    /* The [n][k] store is already the mma B register layout (n = lane/4,
     * pairs along k), so B loads WITHOUT .trans. */
    uint32_t address =
        (uint32_t)__cvta_generic_to_shared(row_pointer);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
}

__device__ __forceinline__ void sol_mma16816_bf16(float d[4],
                                                  const uint32_t a[4],
                                                  const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]));
}

__device__ __forceinline__ void sol_ldmatrix_b_trans(
        uint32_t b[2], const ushort *row_pointer) {
    /* Transposed load: a [k][n] fragment from an [n-rows-as-k... ] --
     * used where the store is k-major ([kv][d]) and the mma wants pairs
     * along the stored row axis. */
    uint32_t address =
        (uint32_t)__cvta_generic_to_shared(row_pointer);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
}


__device__ __forceinline__ uint32_t sol_pack_bf16x2(float a,float b){return uint32_t(sol_bf16(a))|(uint32_t(sol_bf16(b))<<16);}

__global__ void sol_finite(const ushort *q,const ushort *k,const ushort *v,size_t n,unsigned *fault) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
        if(!isfinite(sol_float(q[i]))||!isfinite(sol_float(k[i]))||!isfinite(sol_float(v[i])))atomicMax(fault,1u);
}
__global__ void sol_summary(const ushort *input,ushort *out,unsigned sequence,unsigned heads,
                             unsigned first_head,unsigned block_size,unsigned blocks,unsigned *fault) {
    unsigned h=blockIdx.y,b=blockIdx.x,d=threadIdx.x,start=b*block_size,len=min(block_size,sequence-start);
    float sum=0;
    for(unsigned i=0;i<len;i++)sum+=sol_float(input[(size_t(start+i)*heads+first_head+h)*128+d]);
    float mean=sum/float(len);
    if(!isfinite(mean)||!isfinite(sol_float(sol_bf16(mean))))atomicMax(fault,2u);
    out[(size_t(h)*blocks+b)*128+d]=sol_bf16(mean);
}
__global__ void sol_moments(const ushort *kc,float *out,unsigned blocks) {
    unsigned h=blockIdx.x,d=threadIdx.x;float sum=0,sq=0;
    for(unsigned b=0;b<blocks;b++){float x=sol_float(kc[(size_t(h)*blocks+b)*128+d]);sum+=x;sq+=x*x;}
    float mean=sum/float(blocks);out[h*256+d]=mean;out[h*256+128+d]=fmaxf(sq/float(blocks)-mean*mean,0);
}
__device__ __forceinline__ float sol_warp_sum(float x) {
    for(int d=16;d;d/=2)x+=__shfl_xor_sync(0xffffffffu,x,d);return x;
}
__global__ void sol_route(const ushort *qc,const ushort *kc,const float *moments,
    const h3_sol_block *qm,const h3_sol_block *km,unsigned char *routes,counter *counts,
    unsigned query_blocks,unsigned key_blocks,unsigned first_query,unsigned slab,unsigned radius,
    unsigned minimum,float scale,float tau,bool force_exact) {
    unsigned lane=threadIdx.x,h=blockIdx.y,qb=first_query+blockIdx.x;
    float q[4],mean=0,var=0;
    for(unsigned d=0;d<4;d++) {
        unsigned col=lane+32*d;q[d]=sol_float(qc[(size_t(h)*query_blocks+qb)*128+col]);
        mean+=q[d]*moments[h*256+col];var+=q[d]*q[d]*moments[h*256+128+col];
    }
    float a=scale*1.4426950408889634f;
    mean=sol_warp_sum(mean)*a;var=fmaxf(sol_warp_sum(var),0)*a*a;
    float threshold=mean+tau*sqrtf(var+1e-6f);h3_sol_block query=qm[qb];
    counter c[7]={};
    for(unsigned b=0;b<key_blocks;b++) {
        float dot=0;for(unsigned d=0;d<4;d++)dot+=q[d]*sol_float(kc[(size_t(h)*key_blocks+b)*128+lane+32*d]);
        dot=sol_warp_sum(dot)*a;h3_sol_block key=km[b];
        bool protect=query.protect||key.protect;
        bool local=query.first_frame>=0&&key.first_frame>=0&&
            query.first_frame<=key.last_frame+int(radius)&&key.first_frame<=query.last_frame+int(radius);
        bool floor=((uint64_t(b+1)*minimum/key_blocks)!=(uint64_t(b)*minimum/key_blocks));
        bool recovery=!isfinite(dot)||!isfinite(threshold),tail=key.rows<64;
        bool exact=force_exact||protect||local||floor||recovery||tail||dot>threshold;
        c[exact?0:1]++;c[2]+=protect;c[3]+=local&&!protect;c[4]+=floor;c[5]+=recovery;c[6]+=tail;
        if(!lane)routes[(size_t(h)*slab+blockIdx.x)*key_blocks+b]=exact;
    }
    if(!lane)for(unsigned j=0;j<7;j++)atomicAdd(counts+j,c[j]);
}

// One CTA owns exactly the requested routing block. The exact and centroid
// traversals share one FP32 online-softmax state. Both skip excluded tiles.
template<int QROWS>
__global__ void sol_attention(const ushort *query,const ushort *key,const ushort *value,
    const ushort *kc,const ushort *vc,const unsigned char *routes,ushort *output,unsigned *fault,
    unsigned sequence,unsigned heads,unsigned first_head,unsigned key_blocks,unsigned first_query,
    unsigned slab,float scale) {
    /* Another CTA can report overflow while this kernel is running. Snapshot
     * the fault once per CTA so every thread takes the same barrier path. */
    __shared__ unsigned abort_block;
    if(threadIdx.x==0)abort_block=*fault;
    __syncthreads();if(abort_block)return;
    constexpr unsigned THREADS=QROWS*2,DPAD=136;
    extern __shared__ ushort smem[];
    ushort (*ktile)[DPAD]=reinterpret_cast<ushort (*)[DPAD]>(smem);
    ushort (*vtile)[DPAD]=reinterpret_cast<ushort (*)[DPAD]>(smem+64*DPAD);
    unsigned warp=threadIdx.x/32,lane=threadIdx.x%32,h=blockIdx.y;
    unsigned q0=(first_query+blockIdx.x)*QROWS;
    size_t stride=size_t(heads)*128;
    const ushort *qbase=query+size_t(first_head+h)*128;
    for(unsigned x=threadIdx.x;x<QROWS*16;x+=THREADS) {
        unsigned row=x/16,d=(x%16)*8;
        if(q0+row<sequence)*reinterpret_cast<uint4*>(&ktile[row][d])=*reinterpret_cast<const uint4*>(qbase+size_t(q0+row)*stride+d);
        else *reinterpret_cast<uint4*>(&ktile[row][d])=make_uint4(0,0,0,0);
    }
    __syncthreads();
    uint32_t q_frag[8][4];
    for(int k=0;k<8;k++)sol_ldmatrix_a(q_frag[k],&ktile[warp*16+(lane&15)][k*16+((lane&16)?8:0)]);
    __syncthreads();
    const unsigned char *flags=routes+(size_t(h)*slab+blockIdx.x)*key_blocks;
    float o_acc[16][4]={},m_stat[2]={-INFINITY,-INFINITY},l_stat[2]={};
    unsigned my_rows[2]={q0+warp*16+lane/4,q0+warp*16+lane/4+8};
    for(unsigned phase=0;phase<2;phase++) {
      unsigned tiles=phase?(key_blocks+63)/64:key_blocks;
      for(unsigned tile=0;tile<tiles;tile++) {
        unsigned kv0=tile*64;
        if(!phase&&!flags[tile])continue;
        if(phase) {
            bool any=false;
            for(unsigned j=threadIdx.x;j<64;j+=THREADS)any|=kv0+j<key_blocks&&!flags[kv0+j];
            if(!__syncthreads_or(any))continue;
        }
        for(unsigned x=threadIdx.x;x<2048;x+=THREADS) {
            unsigned which=x/1024,j=(x%1024)/16,d=(x%16)*8,row=kv0+j;
            ushort *dest=which?&vtile[j][d]:&ktile[j][d];
            unsigned limit=phase?key_blocks:sequence;
            if(row<limit) {
                const ushort *src=phase?((which?vc:kc)+(size_t(h)*key_blocks+row)*128+d):
                    ((which?value:key)+(size_t(row)*heads+first_head+h)*128+d);
                unsigned shared=static_cast<unsigned>(__cvta_generic_to_shared(dest));
                asm volatile("cp.async.cg.shared.global [%0], [%1], 16;\n" :: "r"(shared),"l"(src));
            } else *reinterpret_cast<uint4*>(dest)=make_uint4(0,0,0,0);
        }
        asm volatile("cp.async.commit_group;\n" ::);
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
        /* S = Q x K^T for this warp's 16 rows x 64 keys. */
        float s_acc[8][4] = {};
        #pragma unroll
        for (int kc = 0; kc < 8; kc++) {
            uint32_t b_provider_kv = lane & 7u;
            uint32_t b_provider_d = kc * 16 + ((lane & 8u) ? 8u : 0u);
            #pragma unroll
            for (int group = 0; group < 8; group++) {
                uint32_t b[2];
                sol_ldmatrix_b(
                    b, &ktile[group * 8 + b_provider_kv]
                              [b_provider_d]);
                sol_mma16816_bf16(s_acc[group], q_frag[kc], b);
            }
        }
        /* Online softmax; probabilities overwrite s_acc in place. */
        #pragma unroll
        for (int half = 0; half < 2; half++) {
            float row_max = -INFINITY;
            #pragma unroll
            for (int group = 0; group < 8; group++) {
                uint32_t column = kv0 + (uint32_t)group * 8 +
                                  (lane & 3u) * 2;
                float s0 = s_acc[group][2 * half] * scale + (phase ? 4.1588830833596715f : 0.0f);
                float s1 = s_acc[group][2 * half + 1] * scale + (phase ? 4.1588830833596715f : 0.0f);
                int valid0 = phase ? (column < key_blocks && !flags[column]) : column < sequence;
                int valid1 = phase ? (column+1 < key_blocks && !flags[column+1]) : column+1 < sequence;
                s_acc[group][2 * half] = valid0 ? s0 : -INFINITY;
                s_acc[group][2 * half + 1] = valid1 ? s1 : -INFINITY;
                row_max = fmaxf(row_max,
                                fmaxf(s_acc[group][2 * half],
                                      s_acc[group][2 * half + 1]));
            }
            row_max = fmaxf(row_max,
                            __shfl_xor_sync(0xffffffffu, row_max, 1));
            row_max = fmaxf(row_max,
                            __shfl_xor_sync(0xffffffffu, row_max, 2));
            float m_new = fmaxf(m_stat[half], row_max);
            float correction = 1.0f, row_sum = 0.0f;
            if (m_new != -INFINITY) {
                correction = m_stat[half] == -INFINITY ?
                    0.0f : expf(m_stat[half] - m_new);
                #pragma unroll
                for (int group = 0; group < 8; group++) {
                    float p0 = s_acc[group][2 * half] == -INFINITY ? 0.0f :
                        expf(s_acc[group][2 * half] - m_new);
                    float p1 = s_acc[group][2 * half + 1] == -INFINITY ?
                        0.0f : expf(s_acc[group][2 * half + 1] - m_new);
                    s_acc[group][2 * half] = p0;
                    s_acc[group][2 * half + 1] = p1;
                    row_sum += p0 + p1;
                }
            } else {
                #pragma unroll
                for (int group = 0; group < 8; group++) {
                    s_acc[group][2 * half] = 0.0f;
                    s_acc[group][2 * half + 1] = 0.0f;
                }
            }
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 1);
            row_sum += __shfl_xor_sync(0xffffffffu, row_sum, 2);
            l_stat[half] = l_stat[half] * correction + row_sum;
            m_stat[half] = m_new;
            #pragma unroll
            for (int group = 0; group < 16; group++) {
                o_acc[group][2 * half] *= correction;
                o_acc[group][2 * half + 1] *= correction;
            }
        }
        /* O += P x V (P is a register repack of the S accumulators). */
        #pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            uint32_t a[4];
            a[0] = sol_pack_bf16x2(s_acc[2 * kc][0], s_acc[2 * kc][1]);
            a[1] = sol_pack_bf16x2(s_acc[2 * kc][2], s_acc[2 * kc][3]);
            a[2] = sol_pack_bf16x2(s_acc[2 * kc + 1][0],
                                   s_acc[2 * kc + 1][1]);
            a[3] = sol_pack_bf16x2(s_acc[2 * kc + 1][2],
                                   s_acc[2 * kc + 1][3]);
            uint32_t v_provider_kv = kc * 16 + (lane & 15u);
            #pragma unroll
            for (int group = 0; group < 16; group++) {
                uint32_t b[2];
                sol_ldmatrix_b_trans(
                    b, &vtile[v_provider_kv][group * 8]);
                sol_mma16816_bf16(o_acc[group], a, b);
            }
        }

        __syncthreads();
      }
    }
    for(unsigned half=0;half<2;half++) {
        unsigned row=my_rows[half];if(row>=sequence)continue;
        float inverse=l_stat[half]>0?1.0f/l_stat[half]:0;
        size_t base=(size_t(h)*slab*QROWS+blockIdx.x*QROWS+warp*16+lane/4+half*8)*128;
        unsigned col=(lane&3)*2;
        for(int group=0;group<16;group++) {
            float a=o_acc[group][2*half]*inverse,b=o_acc[group][2*half+1]*inverse;
            ushort ba=sol_bf16(a),bb=sol_bf16(b);
            if(!isfinite(a)||!isfinite(b)||!isfinite(sol_float(ba))||!isfinite(sol_float(bb)))atomicMax(fault,3u);
            output[base+group*8+col]=ba;output[base+group*8+col+1]=bb;
        }
    }
}
__global__ void sol_commit(const ushort *slab_out,ushort *out,const unsigned *fault,unsigned sequence,unsigned heads,
    unsigned first_head,unsigned group,unsigned first_query,unsigned slab,unsigned qb,bool head_major) {
    if(*fault)return;
    size_t n=size_t(group)*slab*qb*128;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        unsigned d=i%128,row=(i/128)%(slab*qb)+first_query*qb,h=i/(size_t(slab)*qb*128)+first_head;
        if(row<sequence)out[(head_major?(size_t(h)*sequence+row):(size_t(row)*heads+h))*128+d]=slab_out[i];
    }
}

namespace {
struct sol_event { cudaEvent_t start=nullptr,end=nullptr; unsigned kind=0; };
struct sol_context {
    unsigned char *memory;size_t capacity;cudaStream_t stream;bool profile,ready=false;
    h3_cuda_sol_options options=H3_CUDA_SOL_DEFAULT;h3_cuda_sol_plan plan={};
    h3_cuda_sol_stats stats={};std::vector<sol_event> events;size_t used=0;
    sol_context(void *p,size_t n,cudaStream_t s,bool f):memory((unsigned char*)p),capacity(n),stream(s),profile(f){}
    ~sol_context(){for(auto &e:events){if(e.start)cudaEventDestroy(e.start);if(e.end)cudaEventDestroy(e.end);}}
    template<class T>T *at(size_t offset){return reinterpret_cast<T*>(memory+offset);}
};
void sol_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
int sol_error(char *e,size_t n,const char *what){if(e&&n)snprintf(e,n,"CUDA SOL: %s",what);return 0;}
struct sol_timer {
    sol_context *c;sol_event *event=nullptr;
    sol_timer(sol_context *p,unsigned kind):c(p){
        if(!c->profile)return;
        if(c->used==c->events.size()){c->stats.unprofiled_regions++;return;}
        event=&c->events[c->used++];event->kind=kind;sol_check(cudaEventRecord(event->start,c->stream));
    }
    ~sol_timer(){if(event)cudaEventRecord(event->end,c->stream);}
};
bool sol_overlap(const void *a,const void *b,size_t n){uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<=y?y-x<n:x-y<n;}
}
void *h3_cuda_sol_create(void *workspace,size_t bytes,cudaStream_t stream,bool profile,char *error,size_t size) {
    if(!workspace||((uintptr_t)workspace&255)||!bytes||bytes>H3_CUDA_SOL_WORKSPACE_BYTES){sol_error(error,size,"invalid workspace");return nullptr;}
    sol_context *c=nullptr;
    try{c=new sol_context(workspace,bytes,stream,profile);return c;}
    catch(const std::exception &e){delete c;sol_error(error,size,e.what());return nullptr;}
}
void h3_cuda_sol_free(void *context){delete static_cast<sol_context*>(context);}
/* A failed submission keeps its fault sticky until explicit cancellation. */
int h3_cuda_sol_reset(void *context,char *error,size_t size) {
    auto *c=static_cast<sol_context*>(context);if(!c||!c->ready)return 1;
    try {
        sol_check(cudaStreamSynchronize(c->stream));
        sol_check(cudaMemsetAsync(c->at<unsigned>(c->plan.fault),0,sizeof(unsigned),c->stream));
        sol_check(cudaStreamSynchronize(c->stream));c->used=0;return 1;
    }catch(const std::exception &e){return sol_error(error,size,e.what());}
}
int h3_cuda_sol_configure(void *context,h3_cuda_sol_options options,const h3_sol_layout *layout,unsigned heads,char *error,size_t size) {
    auto *c=static_cast<sol_context*>(context);
    if(!c||c->ready||!layout||!layout->query||!layout->key)return sol_error(error,size,"missing or already configured layout");
    if(!h3_cuda_sol_plan_make(layout->sequence,heads,options,c->capacity,&c->plan,error,size))return 0;
    auto &p=c->plan;
    if(layout->query_blocks!=p.query_blocks||layout->key_blocks!=p.key_blocks)return sol_error(error,size,"layout block counts disagree with policy");
    for(unsigned which=0;which<2;which++) {
        unsigned count=which?p.key_blocks:p.query_blocks,block=which?64:options.q_block;
        const auto *m=which?layout->key:layout->query;
        for(unsigned i=0;i<count;i++)if(m[i].rows!=std::min(block,p.sequence-i*block)||
            m[i].first_frame < -1||m[i].last_frame<m[i].first_frame||m[i].last_frame>10000000)
            return sol_error(error,size,"invalid protection metadata");
    }
    try {
        c->options=options;c->stats.workspace_bytes=p.bytes;
        if(c->profile){c->events.resize(1024);for(auto &e:c->events){sol_check(cudaEventCreate(&e.start));sol_check(cudaEventCreate(&e.end));}}
        sol_check(cudaMemsetAsync(c->at<unsigned>(p.fault),0,sizeof(unsigned),c->stream));
        sol_check(cudaMemsetAsync(c->at<counter>(p.counts),0,7*sizeof(counter),c->stream));
        sol_check(cudaMemcpyAsync(c->at<h3_sol_block>(p.query_meta),layout->query,p.query_blocks*sizeof(h3_sol_block),cudaMemcpyHostToDevice,c->stream));
        sol_check(cudaMemcpyAsync(c->at<h3_sol_block>(p.key_meta),layout->key,p.key_blocks*sizeof(h3_sol_block),cudaMemcpyHostToDevice,c->stream));
        sol_check(cudaStreamSynchronize(c->stream));c->ready=true;return 1;
    }catch(const std::exception &e){return sol_error(error,size,e.what());}
}
int h3_cuda_sol_run(void *context,void *out,const void *q,const void *k,const void *v,unsigned seq,unsigned heads,float scale,
                    bool head_major,bool force_exact,char *error,size_t size) {
    auto *c=static_cast<sol_context*>(context);
    if(!c||!c->ready||seq!=c->plan.sequence||heads!=c->plan.heads||!std::isfinite(scale)||scale<=0||
       !out||!q||!k||!v||(((uintptr_t)out|(uintptr_t)q|(uintptr_t)k|(uintptr_t)v)&15))return sol_error(error,size,"invalid shape, scale, alignment or unconfigured layout");
    size_t elements=size_t(seq)*heads*128;
    if(sol_overlap(out,q,elements*2)||sol_overlap(out,k,elements*2)||sol_overlap(out,v,elements*2))return sol_error(error,size,"output aliases an input");
    auto &p=c->plan;auto s=c->stream;
    auto *fault=c->at<unsigned>(p.fault);auto *qc=c->at<ushort>(p.q_centroid),*kc=c->at<ushort>(p.k_centroid),*vc=c->at<ushort>(p.v_centroid);
    auto *route=c->at<unsigned char>(p.routes);auto *output=c->at<ushort>(p.output);
    unsigned minimum=unsigned(ceil(double(c->options.min_exact)*p.key_blocks));
    try {
        {sol_timer t(c,0);sol_finite<<<std::min(size_t(4096),(elements+255)/256),256,0,s>>>((const ushort*)q,(const ushort*)k,(const ushort*)v,elements,fault);sol_check(cudaGetLastError());}
        for(unsigned first=0;first<heads;first+=p.head_group) {
            unsigned group=std::min(p.head_group,heads-first);c->stats.head_groups++;
            {sol_timer t(c,1);
             sol_summary<<<dim3(p.query_blocks,group),128,0,s>>>((const ushort*)q,qc,seq,heads,first,c->options.q_block,p.query_blocks,fault);
             sol_summary<<<dim3(p.key_blocks,group),128,0,s>>>((const ushort*)k,kc,seq,heads,first,64,p.key_blocks,fault);
             sol_summary<<<dim3(p.key_blocks,group),128,0,s>>>((const ushort*)v,vc,seq,heads,first,64,p.key_blocks,fault);
             sol_moments<<<group,128,0,s>>>(kc,c->at<float>(p.moments),p.key_blocks);sol_check(cudaGetLastError());}
            for(unsigned first_query=0;first_query<p.query_blocks;first_query+=p.query_slab) {
                unsigned slab=std::min(p.query_slab,p.query_blocks-first_query);c->stats.slabs++;
                {sol_timer t(c,2);sol_route<<<dim3(slab,group),32,0,s>>>(qc,kc,c->at<float>(p.moments),c->at<h3_sol_block>(p.query_meta),c->at<h3_sol_block>(p.key_meta),route,c->at<counter>(p.counts),p.query_blocks,p.key_blocks,first_query,slab,c->options.local_radius,minimum,scale,c->options.tau,force_exact);sol_check(cudaGetLastError());}
                {sol_timer t(c,3);
                 if(c->options.q_block==32)sol_attention<32><<<dim3(slab,group),64,64*136*4,s>>>((const ushort*)q,(const ushort*)k,(const ushort*)v,kc,vc,route,output,fault,seq,heads,first,p.key_blocks,first_query,slab,scale);
                 else sol_attention<64><<<dim3(slab,group),128,64*136*4,s>>>((const ushort*)q,(const ushort*)k,(const ushort*)v,kc,vc,route,output,fault,seq,heads,first,p.key_blocks,first_query,slab,scale);
                 sol_check(cudaGetLastError());}
                {sol_timer t(c,4);size_t count=size_t(group)*slab*c->options.q_block*128;
                 sol_commit<<<std::min(size_t(4096),(count+255)/256),256,0,s>>>(output,(ushort*)out,fault,seq,heads,first,group,first_query,slab,c->options.q_block,head_major);sol_check(cudaGetLastError());}
            }
        }
        c->stats.calls++;return 1;
    }catch(const std::exception &e){return sol_error(error,size,e.what());}
}
int h3_cuda_sol_collect(void *context,h3_cuda_sol_stats *stats,char *error,size_t size) {
    auto *c=static_cast<sol_context*>(context);if(!c||!c->ready)return sol_error(error,size,"unconfigured context");
    try {
        unsigned fault=0;sol_check(cudaMemcpy(&fault,c->at<unsigned>(c->plan.fault),4,cudaMemcpyDeviceToHost));
        sol_check(cudaMemcpy(c->stats.pairs,c->at<counter>(c->plan.counts),7*sizeof(counter),cudaMemcpyDeviceToHost));
        double *durations[]={&c->stats.finite_seconds,&c->stats.summary_seconds,&c->stats.route_seconds,&c->stats.fused_seconds,&c->stats.output_seconds};
        for(size_t i=0;i<c->used;i++){float ms=0;auto &e=c->events[i];sol_check(cudaEventElapsedTime(&ms,e.start,e.end));*durations[e.kind]+=ms*.001;c->stats.profiled_regions++;}c->used=0;
        if(stats)*stats=c->stats;
        if(fault){char message[160];snprintf(message,sizeof(message),"nonfinite %s (device fault %u)",fault==1?"input":fault==2?"summary":"attention output",fault);return sol_error(error,size,message);}return 1;
    }catch(const std::exception &e){return sol_error(error,size,e.what());}
}
int h3_cuda_sol_routes(void *context,unsigned char *out,size_t bytes,char *error,size_t size) {
    auto *c=static_cast<sol_context*>(context);
    if(!c||!c->ready||c->plan.head_group<c->plan.heads||c->plan.query_slab<c->plan.query_blocks||
       bytes!=size_t(c->plan.heads)*c->plan.query_blocks*c->plan.key_blocks)return sol_error(error,size,"route export requires a single group/slab and exact byte count");
    return cudaMemcpy(out,c->at<unsigned char>(c->plan.routes),bytes,cudaMemcpyDeviceToHost)==cudaSuccess?1:sol_error(error,size,"route export failed");
}
