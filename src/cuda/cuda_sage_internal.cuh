#pragma once
#include "src/cuda/cuda_sage.h"
#include "src/denoise/attention.h"
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <algorithm>
#include <cmath>
#include <cstring>
#include <stdexcept>
#include <vector>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <fcntl.h>
#include <unistd.h>
inline void sage_check(cudaError_t e){if(e!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(e));}
struct sage_timing {cudaEvent_t start=nullptr,end=nullptr;int category;};
constexpr size_t sage_reserved_bytes=256+(4u<<20);
struct sage_context {
    char *workspace;size_t capacity,offset=sage_reserved_bytes;
    cudaStream_t stream; cublasHandle_t blas=nullptr;
    bool profiling;
    cudaError_t timer_error=cudaSuccess;
    h3_sage_stats stats={};std::vector<sage_timing> timings;
    template<class T> T *alloc(size_t n) {
        offset=(offset+255)&~size_t(255);
        if(n>SIZE_MAX/sizeof(T)||offset>capacity||n*sizeof(T)>capacity-offset)
            throw std::runtime_error("Sage attention workspace exceeds 512 MiB plan");
        T *p=reinterpret_cast<T*>(workspace+offset);offset+=n*sizeof(T);
        stats.workspace_bytes=std::max<uint64_t>(stats.workspace_bytes,offset);return p;
    }
    unsigned *fault(){return reinterpret_cast<unsigned*>(workspace);}
};
struct sage_timer {
    sage_context &s;sage_timing t={};
    sage_timer(sage_context &context,int category):s(context){
        t.category=category;if(!s.profiling)return;
        sage_check(cudaEventCreate(&t.start));
        try {sage_check(cudaEventCreate(&t.end));sage_check(cudaEventRecord(t.start,s.stream));s.timings.push_back(t);}
        catch(...){cudaEventDestroy(t.start);if(t.end)cudaEventDestroy(t.end);throw;}
    }
    ~sage_timer(){if(t.start){auto e=cudaEventRecord(t.end,s.stream);if(e!=cudaSuccess)s.timer_error=e;}}
};
/* Explicit, bounded validation dump; never enabled by normal inference. */
inline void sage_debug_dump(sage_context &s,const char *name,const void *data,size_t bytes,unsigned seq) {
    const char *dir=getenv("H3_TEST_SAGE_PACK_DIR");
    if(!dir||!*dir||seq>4096||s.stats.calls[1]||s.stats.calls[2])return;
    std::string path=std::string(dir)+"/"+name;
    int fd=open(path.c_str(),O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW,0600);
    if(fd<0)throw std::runtime_error("cannot create Sage validation pack dump");
    try {
        std::vector<char> host(bytes);
        sage_check(cudaMemcpyAsync(host.data(),data,bytes,cudaMemcpyDeviceToHost,s.stream));
        sage_check(cudaStreamSynchronize(s.stream));
        size_t at=0;while(at<bytes){ssize_t n=write(fd,host.data()+at,bytes-at);if(n<=0)throw std::runtime_error("cannot write Sage validation pack dump");at+=n;}
        if(close(fd)){fd=-1;throw std::runtime_error("cannot close Sage validation pack dump");}fd=-1;
    }catch(...){if(fd>=0)close(fd);unlink(path.c_str());throw;}
}
void sage2_run(sage_context &,nv_bfloat16 *,const nv_bfloat16 *,const nv_bfloat16 *,const nv_bfloat16 *,unsigned,unsigned,float,bool);
void sage3_run(sage_context &,nv_bfloat16 *,const nv_bfloat16 *,const nv_bfloat16 *,const nv_bfloat16 *,unsigned,unsigned,float,bool);

/* Pinned Torch 2.8 BF16 HND mean on the qualified 5090: four output
 * channels/thread, four input warps, four independent accumulators. Its
 * global split targets 170 * (1536 / 128) CTAs across 56 heads. Preserve this
 * arithmetic plan even when the native workspace processes fewer heads.
 * The native thread mapping differs; each thread owns one channel and
 * reproduces the original per-warp sums explicitly. */
inline unsigned sage_reduce_partials(unsigned seq) {
    if(seq<64)return 1;
    unsigned values=(seq+3)/4;
    return values<256?1:std::max(std::min(37u,(values+15)/16),(values+255)/256);
}
static __global__ void sage_reduce_tiles(const nv_bfloat16 *k,const nv_bfloat16 *v,
    float *sums,float *maxima,unsigned seq,unsigned heads,unsigned first,unsigned partials,unsigned *fault) {
    unsigned h=blockIdx.y,d=threadIdx.x,part=blockIdx.x;
    unsigned lanes=seq>=64?4:1,stride=lanes*partials;
    float lane_sum[4]={0,0,0,0},maximum=0;
    for(unsigned lane=0;lane<lanes;lane++) {
        float values[4]={0,0,0,0};
        for(unsigned base=part*lanes+lane;base<seq;base+=stride*4) {
            #pragma unroll
            for(unsigned i=0;i<4;i++) {
                unsigned row=base+i*stride;
                if(row<seq) {
                    size_t at=((size_t)row*heads+first+h)*128+d;
                    float x=__bfloat162float(k[at]),y=__bfloat162float(v[at]);
                    if(!isfinite(x)||!isfinite(y)){atomicExch(fault,1u);x=0;y=0;}
                    values[i]=__fadd_rn(values[i],x);maximum=fmaxf(maximum,fabsf(y));
                }
            }
        }
        lane_sum[lane]=__fadd_rn(__fadd_rn(__fadd_rn(values[0],values[1]),values[2]),values[3]);
    }
    float sum=lanes==1?lane_sum[0]:__fadd_rn(__fadd_rn(lane_sum[0],lane_sum[2]),__fadd_rn(lane_sum[1],lane_sum[3]));
    size_t at=((size_t)h*partials+part)*128+d;sums[at]=sum;maxima[at]=maximum;
}
static __global__ void sage_reduce_finish(const float *sums,float *maxima,
    nv_bfloat16 *means,float *vscale,unsigned partials,float mean_factor,float scale_max) {
    unsigned h=blockIdx.x,d=threadIdx.x;float lanes[4]={0,0,0,0},maximum=0;
    #pragma unroll
    for(unsigned lane=0;lane<4;lane++)for(unsigned part=lane;part<partials;part+=4) {
        size_t at=((size_t)h*partials+part)*128+d;
        lanes[lane]=__fadd_rn(lanes[lane],sums[at]);maximum=fmaxf(maximum,maxima[at]);
    }
    float sum=__fadd_rn(__fadd_rn(lanes[0],lanes[2]),__fadd_rn(lanes[1],lanes[3]));
    means[h*128+d]=__float2bfloat16_rn(__fmul_rn(sum,mean_factor));
    vscale[h*128+d]=maximum/scale_max;
    // Reuse the first partial-max slot; each thread owns one channel.
    maxima[(size_t)h*partials*128+d]=maximum;
}
inline void sage_reduce(sage_context &s,const nv_bfloat16 *k,const nv_bfloat16 *v,
    float *sums,float *maxima,nv_bfloat16 *means,float *vscale,unsigned seq,unsigned heads,unsigned first,unsigned group) {
    unsigned partials=sage_reduce_partials(seq);
    sage_reduce_tiles<<<dim3(partials,group),128,0,s.stream>>>(k,v,sums,maxima,seq,heads,first,partials,s.fault());
    sage_reduce_finish<<<group,128,0,s.stream>>>(sums,maxima,means,vscale,partials,(float)(1.0/(double)seq),2.25f);
    sage_check(cudaGetLastError());
}
