// Native SubBlock recipe 1. Selected tiles only; no centroid approximation.
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


#include "src/cuda/cuda_subblock.h"
#include "src/denoise/attention.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <stdexcept>
#include <vector>
using ushort=unsigned short;
using counter=unsigned long long;
static_assert(sizeof(counter)==8 && sizeof(h3_sol_block)==16,"SubBlock metadata ABI");
__device__ __forceinline__ float sb_float(ushort x){return __bfloat162float(*reinterpret_cast<__nv_bfloat16*>(&x));}
__device__ __forceinline__ ushort sb_bf16(float x){auto b=__float2bfloat16_rn(x);return *reinterpret_cast<ushort*>(&b);}
__device__ __forceinline__ void sb_ldmatrix_a(uint32_t a[4],
                                               const ushort *row_pointer) {
    uint32_t address =
        (uint32_t)__cvta_generic_to_shared(row_pointer);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
        : "=r"(a[0]), "=r"(a[1]), "=r"(a[2]), "=r"(a[3]) : "r"(address));
}

__device__ __forceinline__ void sb_ldmatrix_b(uint32_t b[2],
                                               const ushort *row_pointer) {
    /* The [n][k] store is already the mma B register layout (n = lane/4,
     * pairs along k), so B loads WITHOUT .trans. */
    uint32_t address =
        (uint32_t)__cvta_generic_to_shared(row_pointer);
    asm volatile(
        "ldmatrix.sync.aligned.m8n8.x2.shared.b16 {%0,%1}, [%2];\n"
        : "=r"(b[0]), "=r"(b[1]) : "r"(address));
}

__device__ __forceinline__ void sb_mma16816_bf16(float d[4],
                                                  const uint32_t a[4],
                                                  const uint32_t b[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]),
          "r"(b[1]));
}

__device__ __forceinline__ void sb_ldmatrix_b_trans(
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


__device__ __forceinline__ uint32_t sb_pack_bf16x2(float a,float b){return uint32_t(sb_bf16(a))|(uint32_t(sb_bf16(b))<<16);}

__global__ void sb_finite(const ushort *q,const ushort *k,const ushort *v,size_t n,unsigned *fault) {
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x)
        if(!isfinite(sb_float(q[i]))||!isfinite(sb_float(k[i]))||!isfinite(sb_float(v[i])))atomicMax(fault,1u);
}
template<int QROWS>
__global__ void sb_attention(const ushort *query,const ushort *key,const ushort *value,
    const unsigned char *routes,const h3_sol_block *meta,bool skip_protected,ushort *output,unsigned *fault,
    unsigned sequence,unsigned heads,unsigned first_head,unsigned key_blocks,unsigned first_query,
    unsigned slab,float scale) {
    /* Another CTA can report overflow while this kernel is running. Snapshot
     * the fault once per CTA so every thread takes the same barrier path. */
    __shared__ unsigned abort_block;
    if(threadIdx.x==0)abort_block=*fault || (skip_protected && meta[first_query+blockIdx.x].protect);
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
    for(int k=0;k<8;k++)sb_ldmatrix_a(q_frag[k],&ktile[warp*16+(lane&15)][k*16+((lane&16)?8:0)]);
    __syncthreads();
    const unsigned char *flags=routes+(size_t(h)*slab+blockIdx.x)*key_blocks;
    float o_acc[16][4]={},m_stat[2]={-INFINITY,-INFINITY},l_stat[2]={};
    unsigned my_rows[2]={q0+warp*16+lane/4,q0+warp*16+lane/4+8};
    for(unsigned tile=0;tile<key_blocks;tile++) {
        unsigned kv0=tile*64;
        if(!flags[tile])continue;
        for(unsigned x=threadIdx.x;x<2048;x+=THREADS) {
            unsigned which=x/1024,j=(x%1024)/16,d=(x%16)*8,row=kv0+j;
            ushort *dest=which?&vtile[j][d]:&ktile[j][d];
            unsigned limit=sequence;
            if(row<limit) {
                const ushort *src=(which?value:key)+(size_t(row)*heads+first_head+h)*128+d;
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
                sb_ldmatrix_b(
                    b, &ktile[group * 8 + b_provider_kv]
                              [b_provider_d]);
                sb_mma16816_bf16(s_acc[group], q_frag[kc], b);
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
                float s0 = s_acc[group][2 * half] * scale;
                float s1 = s_acc[group][2 * half + 1] * scale;
                int valid0 = column < sequence;
                int valid1 = column+1 < sequence;
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
            a[0] = sb_pack_bf16x2(s_acc[2 * kc][0], s_acc[2 * kc][1]);
            a[1] = sb_pack_bf16x2(s_acc[2 * kc][2], s_acc[2 * kc][3]);
            a[2] = sb_pack_bf16x2(s_acc[2 * kc + 1][0],
                                   s_acc[2 * kc + 1][1]);
            a[3] = sb_pack_bf16x2(s_acc[2 * kc + 1][2],
                                   s_acc[2 * kc + 1][3]);
            uint32_t v_provider_kv = kc * 16 + (lane & 15u);
            #pragma unroll
            for (int group = 0; group < 16; group++) {
                uint32_t b[2];
                sb_ldmatrix_b_trans(
                    b, &vtile[v_provider_kv][group * 8]);
                sb_mma16816_bf16(o_acc[group], a, b);
            }
        }

        __syncthreads();
    }
    for(unsigned half=0;half<2;half++) {
        unsigned row=my_rows[half];if(row>=sequence)continue;
        float inverse=l_stat[half]>0?1.0f/l_stat[half]:0;
        if(!(l_stat[half]>0)||!isfinite(l_stat[half]))atomicMax(fault,4u);
        size_t base=(size_t(h)*slab*QROWS+blockIdx.x*QROWS+warp*16+lane/4+half*8)*128;
        unsigned col=(lane&3)*2;
        for(int group=0;group<16;group++) {
            float a=o_acc[group][2*half]*inverse,b=o_acc[group][2*half+1]*inverse;
            ushort ba=sb_bf16(a),bb=sb_bf16(b);
            if(!isfinite(a)||!isfinite(b)||!isfinite(sb_float(ba))||!isfinite(sb_float(bb)))atomicMax(fault,3u);
            output[base+group*8+col]=ba;output[base+group*8+col+1]=bb;
        }
    }
}
__global__ void sb_commit(const ushort *slab_out,ushort *out,const unsigned *fault,unsigned sequence,unsigned heads,
    unsigned first_head,unsigned group,unsigned first_query,unsigned slab,unsigned qb,bool head_major,
    const h3_sol_block *meta,bool skip_protected) {
    if(*fault)return;
    size_t n=size_t(group)*slab*qb*128;
    for(size_t i=size_t(blockIdx.x)*blockDim.x+threadIdx.x;i<n;i+=size_t(gridDim.x)*blockDim.x) {
        unsigned d=i%128,row=(i/128)%(slab*qb)+first_query*qb,h=i/(size_t(slab)*qb*128)+first_head;
        if(row<sequence && !(skip_protected && meta[row/64].protect))out[(head_major?(size_t(h)*sequence+row):(size_t(row)*heads+h))*128+d]=slab_out[i];
    }
}

__global__ static void sb_pool(const ushort *input,ushort *output,unsigned sequence,
    unsigned heads,unsigned first,unsigned cells,float scale,unsigned *fault) {
    unsigned cell=blockIdx.x,h=blockIdx.y,d=threadIdx.x,start=cell*16;
    unsigned count=start<sequence?min(16u,sequence-start):0;
    float values[16];
    for(unsigned i=0;i<16;i++)values[i]=i<count?sb_float(input[(size_t(start+i)*heads+first+h)*128+d]):0;
    for(unsigned stride=8;stride;stride>>=1)for(unsigned i=0;i<stride;i++)values[i]+=values[i+stride];
    float mean=count?values[0]/float(count)*scale:0;
    ushort rounded=sb_bf16(mean);
    if(!isfinite(mean)||!isfinite(sb_float(rounded)))atomicMax(fault,2u);
    output[(size_t(h)*cells+cell)*128+d]=rounded;
}
__device__ static float sb_warp_sum(float value) {
    for(int delta=16;delta;delta>>=1)value+=__shfl_xor_sync(0xffffffffu,value,delta);
    return value;
}
__global__ static void sb_score(const ushort *q,const ushort *k,float *scores,
    unsigned sequence,unsigned cells,unsigned blocks,unsigned first_query,unsigned slab,unsigned *fault) {
    unsigned lane=threadIdx.x%32,warp=threadIdx.x/32,kb=blockIdx.x*4+warp;
    unsigned qi=blockIdx.y,qb=first_query+qi,h=blockIdx.z;
    if(kb>=blocks)return;
    float query_lse[4];unsigned nq=min(4u,(sequence-qb*64+15)/16);
    unsigned nk=min(4u,(sequence-kb*64+15)/16);
    for(unsigned a=0;a<nq;a++) {
        float dots[4];
        for(unsigned b=0;b<nk;b++) {
            float sum=0;
            for(unsigned d=lane;d<128;d+=32)
                sum+=sb_float(q[(size_t(h)*cells+qb*4+a)*128+d])*sb_float(k[(size_t(h)*cells+kb*4+b)*128+d]);
            dots[b]=sb_warp_sum(sum);
        }
        if(!lane) {
            float largest=dots[0];for(unsigned b=1;b<nk;b++)largest=fmaxf(largest,dots[b]);
            float sum=0;for(unsigned b=0;b<nk;b++)sum+=exp2f(dots[b]-largest);
            query_lse[a]=largest+log2f(sum);
        }
    }
    if(!lane) {
        float largest=query_lse[0];for(unsigned a=1;a<nq;a++)largest=fmaxf(largest,query_lse[a]);
        float sum=0;for(unsigned a=0;a<nq;a++)sum+=exp2f(query_lse[a]-largest);
        float score=(largest+log2f(sum))*0.6931471805599453f;
        if(!isfinite(score))atomicMax(fault,2u);
        scores[(size_t(h)*slab+qi)*blocks+kb]=score;
    }
}
/* Exact rank with deterministic ties. O(B^2) comparisons, bounded O(B) score
 * storage per query. Recipe 1 prioritizes auditable routing at the fixed shape. */
__global__ static void sb_select(const float *scores,unsigned char *routes,
    const h3_sol_block *qm,const h3_sol_block *km,unsigned blocks,unsigned first_query,
    unsigned slab,unsigned keep,bool force_all,counter *counts) {
    unsigned qi=blockIdx.x,h=blockIdx.y;size_t base=(size_t(h)*slab+qi)*blocks;
    bool protected_query=qm[first_query+qi].protect!=0;
    __shared__ unsigned selected[256],added[256];unsigned s=0,a=0;
    for(unsigned k=threadIdx.x;k<blocks;k+=256) {
        float score=scores[base+k];unsigned rank=0;
        for(unsigned j=0;j<blocks;j++)rank+=scores[base+j]>score||(scores[base+j]==score&&j<k);
        bool top=rank<keep,protect=km[k].protect!=0;
        bool retain=force_all||protected_query||top||protect;
        routes[base+k]=(unsigned char)retain;s+=retain;
        a+=protect&&!top&&!protected_query&&!force_all;
    }
    selected[threadIdx.x]=s;added[threadIdx.x]=a;__syncthreads();
    for(unsigned stride=128;stride;stride>>=1) {
        if(threadIdx.x<stride){selected[threadIdx.x]+=selected[threadIdx.x+stride];added[threadIdx.x]+=added[threadIdx.x+stride];}
        __syncthreads();
    }
    if(!threadIdx.x){atomicAdd(counts,(counter)selected[0]);atomicAdd(counts+1,(counter)blocks);
        atomicAdd(counts+2,(counter)protected_query);atomicAdd(counts+3,(counter)added[0]);}
}
namespace {
struct sb_event { cudaEvent_t start=nullptr,end=nullptr;unsigned kind=0; };
struct sb_context {
    unsigned char *memory;size_t capacity;cudaStream_t stream;bool profile,ready=false;
    h3_subblock_plan plan={};h3_cuda_subblock_stats stats={};std::vector<sb_event> events;size_t used=0;
    sb_context(void *p,size_t bytes,cudaStream_t s,bool timed):memory((unsigned char*)p),capacity(bytes),stream(s),profile(timed){}
    ~sb_context(){for(auto &e:events){if(e.start)cudaEventDestroy(e.start);if(e.end)cudaEventDestroy(e.end);}}
    template<class T>T *at(size_t offset){return reinterpret_cast<T*>(memory+offset);}
};
void sb_check(cudaError_t status){if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));}
int sb_error(char *error,size_t size,const char *message){if(error&&size)snprintf(error,size,"CUDA SubBlock: %s",message);return 0;}
struct sb_timer {
    sb_context *c;sb_event *event=nullptr;
    sb_timer(sb_context *context,unsigned kind):c(context) {
        if(!c->profile)return;
        if(c->used==c->events.size()){c->stats.unprofiled_regions++;return;}
        event=&c->events[c->used++];event->kind=kind;sb_check(cudaEventRecord(event->start,c->stream));
    }
    ~sb_timer(){if(event)cudaEventRecord(event->end,c->stream);}
};
bool sb_overlap(const void *a,const void *b,size_t bytes){uintptr_t x=(uintptr_t)a,y=(uintptr_t)b;return x<=y?y-x<bytes:x-y<bytes;}
}
void *h3_cuda_subblock_create(void *workspace,size_t bytes,cudaStream_t stream,bool profile,char *e,size_t n) {
    if(!workspace||((uintptr_t)workspace&255)||!bytes||bytes>H3_ATTENTION_WORKSPACE_BYTES){sb_error(e,n,"invalid workspace");return nullptr;}
    sb_context *c=nullptr;
    try{c=new sb_context(workspace,bytes,stream,profile);return c;}
    catch(const std::exception &x){delete c;sb_error(e,n,x.what());return nullptr;}
}
void h3_cuda_subblock_free(void *context){delete static_cast<sb_context*>(context);}
int h3_cuda_subblock_reset(void *context,char *e,size_t n) {
    auto *c=static_cast<sb_context*>(context);if(!c||!c->ready)return 1;
    try{sb_check(cudaStreamSynchronize(c->stream));sb_check(cudaMemsetAsync(c->at<unsigned>(c->plan.fault),0,4,c->stream));
        sb_check(cudaStreamSynchronize(c->stream));c->used=0;return 1;}
    catch(const std::exception &x){return sb_error(e,n,x.what());}
}
int h3_cuda_subblock_configure(void *context,const h3_sol_layout *layout,unsigned heads,float sparsity,char *e,size_t n) {
    auto *c=static_cast<sb_context*>(context);
    if(!c||c->ready||!layout||!layout->query||!layout->key)return sb_error(e,n,"missing/already configured layout");
    if(!h3_subblock_plan_make(layout->sequence,heads,sparsity,c->capacity,&c->plan,e,n))return 0;
    auto &p=c->plan;
    if(layout->query_blocks!=p.blocks||layout->key_blocks!=p.blocks)return sb_error(e,n,"layout geometry differs from Q64/K64");
    for(unsigned i=0;i<p.blocks;i++)if(layout->query[i].rows!=std::min(64u,p.sequence-i*64)||layout->key[i].rows!=layout->query[i].rows)
        return sb_error(e,n,"invalid true block lengths");
    try {
        if(c->profile){c->events.resize(1024);for(auto &event:c->events){sb_check(cudaEventCreate(&event.start));sb_check(cudaEventCreate(&event.end));}}
        sb_check(cudaMemsetAsync(c->at<unsigned>(p.fault),0,4,c->stream));
        sb_check(cudaMemsetAsync(c->at<counter>(p.counts),0,4*sizeof(counter),c->stream));
        sb_check(cudaMemcpyAsync(c->at<h3_sol_block>(p.q_meta),layout->query,p.blocks*sizeof(h3_sol_block),cudaMemcpyHostToDevice,c->stream));
        sb_check(cudaMemcpyAsync(c->at<h3_sol_block>(p.k_meta),layout->key,p.blocks*sizeof(h3_sol_block),cudaMemcpyHostToDevice,c->stream));
        sb_check(cudaStreamSynchronize(c->stream));c->stats.workspace_bytes=p.bytes;c->ready=true;return 1;
    }catch(const std::exception &x){return sb_error(e,n,x.what());}
}
int h3_cuda_subblock_run(void *context,void *out,const void *q,const void *k,const void *v,
    unsigned sequence,unsigned heads,float scale,bool head_major,bool force_all,bool dense_protected,char *e,size_t n) {
    auto *c=static_cast<sb_context*>(context);
    if(!c||!c->ready||sequence!=c->plan.sequence||heads!=c->plan.heads||!std::isfinite(scale)||scale<=0||
       !q||!k||!v||!out||(((uintptr_t)q|(uintptr_t)k|(uintptr_t)v|(uintptr_t)out)&15))return sb_error(e,n,"invalid shape, alignment, scale or layout");
    size_t elements=size_t(sequence)*heads*128;
    if(sb_overlap(out,q,elements*2)||sb_overlap(out,k,elements*2)||sb_overlap(out,v,elements*2))return sb_error(e,n,"output aliases an input");
    const void *buffers[]={out,q,k,v};
    for(const void *pointer:buffers) {
        uintptr_t x=(uintptr_t)pointer,y=(uintptr_t)c->memory;
        if(x<=y?y-x<elements*2:x-y<c->capacity)return sb_error(e,n,"input/output overlaps the workspace");
    }
    auto &p=c->plan;auto stream=c->stream;
    auto *fault=c->at<unsigned>(p.fault);auto *counts=c->at<counter>(p.counts);
    auto *qp=c->at<ushort>(p.q_pool),*kp=c->at<ushort>(p.k_pool),*output=c->at<ushort>(p.output);
    auto *qm=c->at<h3_sol_block>(p.q_meta),*km=c->at<h3_sol_block>(p.k_meta);
    auto *scores=c->at<float>(p.scores);auto *routes=c->at<unsigned char>(p.routes);
    try {
        {sb_timer t(c,0);sb_finite<<<std::min(size_t(4096),(elements+255)/256),256,0,stream>>>((const ushort*)q,(const ushort*)k,(const ushort*)v,elements,fault);sb_check(cudaGetLastError());}
        for(unsigned first=0;first<heads;first+=p.head_group) {
            unsigned group=std::min(p.head_group,heads-first);
            {sb_timer t(c,1);
             sb_pool<<<dim3(p.cells,group),128,0,stream>>>((const ushort*)q,qp,sequence,heads,first,p.cells,scale*1.4426950408889634f,fault);
             sb_pool<<<dim3(p.cells,group),128,0,stream>>>((const ushort*)k,kp,sequence,heads,first,p.cells,1,fault);sb_check(cudaGetLastError());}
            for(unsigned first_query=0;first_query<p.blocks;first_query+=p.query_slab) {
                unsigned slab=std::min(p.query_slab,p.blocks-first_query);
                c->stats.router_calls++;
                {sb_timer t(c,2);
                 sb_score<<<dim3((p.blocks+3)/4,slab,group),128,0,stream>>>(qp,kp,scores,sequence,p.cells,p.blocks,first_query,slab,fault);
                 sb_select<<<dim3(slab,group),256,0,stream>>>(scores,routes,qm,km,p.blocks,first_query,slab,p.keep,force_all,counts);sb_check(cudaGetLastError());}
                {sb_timer t(c,3);
                 sb_attention<64><<<dim3(slab,group),128,64*136*4,stream>>>((const ushort*)q,(const ushort*)k,(const ushort*)v,
                    routes,qm,dense_protected,output,fault,sequence,heads,first,p.blocks,first_query,slab,scale);sb_check(cudaGetLastError());}
                {sb_timer t(c,4);size_t count=size_t(group)*slab*64*128;
                 sb_commit<<<std::min(size_t(4096),(count+255)/256),256,0,stream>>>(output,(ushort*)out,fault,sequence,heads,first,group,
                    first_query,slab,64,head_major,qm,dense_protected);sb_check(cudaGetLastError());}
            }
        }
        c->stats.calls++;return 1;
    }catch(const std::exception &x){return sb_error(e,n,x.what());}
}
int h3_cuda_subblock_collect(void *context,h3_cuda_subblock_stats *stats,char *e,size_t n) {
    auto *c=static_cast<sb_context*>(context);if(!c||!c->ready)return sb_error(e,n,"unconfigured context");
    try {
        unsigned fault;sb_check(cudaMemcpy(&fault,c->at<unsigned>(c->plan.fault),4,cudaMemcpyDeviceToHost));
        sb_check(cudaMemcpy(c->stats.pairs,c->at<counter>(c->plan.counts),4*sizeof(counter),cudaMemcpyDeviceToHost));
        double *times[]={&c->stats.finite_seconds,&c->stats.pool_seconds,&c->stats.route_seconds,&c->stats.kernel_seconds,&c->stats.output_seconds};
        for(size_t i=0;i<c->used;i++){float ms;auto &event=c->events[i];sb_check(cudaEventElapsedTime(&ms,event.start,event.end));*times[event.kind]+=ms*.001;c->stats.profiled_regions++;}
        c->used=0;if(stats)*stats=c->stats;
        if(fault)return sb_error(e,n,"nonfinite input, pooled score or attention output");return 1;
    }catch(const std::exception &x){return sb_error(e,n,x.what());}
}
int h3_cuda_subblock_routes(void *context,unsigned char *routes,float *scores,size_t count,char *e,size_t n) {
    auto *c=static_cast<sb_context*>(context);
    if(!c||!c->ready||c->plan.head_group<c->plan.heads||c->plan.query_slab<c->plan.blocks||count!=size_t(c->plan.heads)*c->plan.blocks*c->plan.blocks)
        return sb_error(e,n,"diagnostics require one group/slab and an exact count");
    try {sb_check(cudaStreamSynchronize(c->stream));
        if(routes)sb_check(cudaMemcpy(routes,c->at<unsigned char>(c->plan.routes),count,cudaMemcpyDeviceToHost));
        if(scores)sb_check(cudaMemcpy(scores,c->at<float>(c->plan.scores),count*4,cudaMemcpyDeviceToHost));return 1;
    }catch(const std::exception &x){return sb_error(e,n,x.what());}
}
