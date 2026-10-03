/* Compare SM90/SM120 tuning byte-for-byte with the unchanged portable kernel.
 * This test includes both kernels directly, independent of runtime dispatch. */
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <mma.h>
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include "src/cuda/cuda_dispatch.h"

#ifndef H3_TEST_TARGET_SM
#error "Set H3_TEST_TARGET_SM to 90 or 120"
#endif
static_assert(H3_TEST_TARGET_SM==90 || H3_TEST_TARGET_SM==120, "qualified architecture");
static constexpr unsigned THREADS=H3_TEST_TARGET_SM==90?512:128;

using ushort = unsigned short;
__device__ float h3_bf16_to_f32(ushort x) {
    return __bfloat162float(__ushort_as_bfloat16(x));
}
__device__ ushort h3_f32_to_bf16(float x) {
    return __bfloat16_as_ushort(__float2bfloat16_rn(x));
}
#include "src/cuda/cuda_attention.cuh"

#define CHECK(x) do { if (!(x)) { std::fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); std::exit(1); } } while (0)
#define CUDA(x) do { cudaError_t e=(x); if(e!=cudaSuccess) { std::fprintf(stderr,"%s: %s\n",#x,cudaGetErrorString(e)); std::exit(1); } } while (0)

static void compare(unsigned sequence, unsigned heads, unsigned batch) {
    const unsigned dim=128;
    size_t count=(size_t)sequence*heads*dim*batch, bytes=count*sizeof(ushort);
    std::vector<ushort> input(count*3), expected(count), actual(count);
    unsigned seed=123;
    for(auto &value:input) {
        seed^=seed<<13; seed^=seed>>17; seed^=seed<<5;
        float f=(float)((int)(seed%4097)-2048)/1024;
        uint32_t u; std::memcpy(&u,&f,4);
        value=(ushort)((u+0x7fff+((u>>16)&1))>>16);
    }
    float scale=1/std::sqrt((float)dim);
    uint32_t u; std::memcpy(&u,&scale,4);
    u=(u+0x7fff+((u>>16)&1))&0xffff0000u;
    std::memcpy(&scale,&u,4);
    ushort *q,*k,*v,*out;
    CUDA(cudaMalloc(&q,bytes)); CUDA(cudaMalloc(&k,bytes));
    CUDA(cudaMalloc(&v,bytes)); CUDA(cudaMalloc(&out,bytes));
    CUDA(cudaMemcpy(q,input.data(),bytes,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(k,input.data()+count,bytes,cudaMemcpyHostToDevice));
    CUDA(cudaMemcpy(v,input.data()+2*count,bytes,cudaMemcpyHostToDevice));
    cudaEvent_t start,end;
    CUDA(cudaEventCreate(&start)); CUDA(cudaEventCreate(&end));
    dim3 grid((sequence+15)/16,heads,batch);
    for(int layout=0;layout<2;layout++) {
        attention_bf16_tiled<<<grid,128>>>(out,q,k,v,sequence,heads,dim,scale,layout!=0);
        CUDA(cudaGetLastError());
        CUDA(cudaMemcpy(expected.data(),out,bytes,cudaMemcpyDeviceToHost));
        std::vector<float> timings[2];
        for(int round=0;round<4;round++) for(int order=0;order<2;order++) {
            int tuned=(round+order)%2;
            CUDA(cudaEventRecord(start));
            if(tuned) attention_bf16_fixed128<THREADS><<<grid,THREADS>>>(out,q,k,v,sequence,heads,scale,layout!=0);
            else attention_bf16_tiled<<<grid,128>>>(out,q,k,v,sequence,heads,dim,scale,layout!=0);
            CUDA(cudaGetLastError()); CUDA(cudaEventRecord(end));
            CUDA(cudaEventSynchronize(end));
            float ms; CUDA(cudaEventElapsedTime(&ms,start,end));
            if(round) timings[tuned].push_back(ms);
            CUDA(cudaMemcpy(actual.data(),out,bytes,cudaMemcpyDeviceToHost));
            CHECK(!std::memcmp(expected.data(),actual.data(),bytes));
        }
        float ms[2];
        for(int i=0;i<2;i++) {
            std::sort(timings[i].begin(),timings[i].end());
            ms[i]=timings[i][1];
        }
        std::printf("{\"architecture\":%d,\"threads\":%u,\"sequence\":%u,\"heads\":%u,\"batch\":%u,\"dim\":128,\"head_major\":%d,\"portable_ms\":%.6f,\"tuned_ms\":%.6f,\"speedup\":%.3f,\"exact\":true}\n",
                    H3_TEST_TARGET_SM,THREADS,sequence,heads,batch,layout,ms[0],ms[1],ms[0]/ms[1]);
        std::fflush(stdout);
    }
    CUDA(cudaEventDestroy(start)); CUDA(cudaEventDestroy(end));
    CUDA(cudaFree(q)); CUDA(cudaFree(k)); CUDA(cudaFree(v)); CUDA(cudaFree(out));
}

int main() {
    for(int sm:{86,89,100,121}) CHECK(h3_cuda_dispatch_select(sm,0).attention==H3_CUDA_PORTABLE);
    CHECK(h3_cuda_dispatch_select(90,0).attention==H3_CUDA_SM90_ATTENTION);
    CHECK(h3_cuda_dispatch_select(90,1).attention==H3_CUDA_PORTABLE);
    CHECK(h3_cuda_dispatch_select(120,0).attention==H3_CUDA_SM120_ATTENTION);
    CHECK(h3_cuda_dispatch_select(120,1).attention==H3_CUDA_PORTABLE);
    int device=0;
    if(const char *s=std::getenv("H3_CUDA_DEVICE")) {
        char *end=nullptr; long value=std::strtol(s,&end,10);
        CHECK(end!=s&&!*end&&value>=0&&value<=INT32_MAX); device=(int)value;
    }
    CUDA(cudaSetDevice(device));
    cudaDeviceProp props; CUDA(cudaGetDeviceProperties(&props,device));
    if(props.major*10+props.minor!=H3_TEST_TARGET_SM) {
        std::printf("skip: SM%d tuning requires an SM%d device\n",H3_TEST_TARGET_SM,H3_TEST_TARGET_SM); return 0;
    }
    compare(19,4,1); compare(512,3,1); compare(513,3,2);
    // Keep sanitizer runs bounded; ordinary qualification always includes the large case.
    if(!std::getenv("H3_TEST_FIXED128_QUICK")) {
        compare(2048,8,1); compare(18225,4,1);
        if(std::getenv("H3_TEST_FIXED128_BENCH")) { compare(2281,56,1); compare(18225,56,1); }
        else if(std::getenv("H3_TEST_SM120_BENCH") || std::getenv("H3_TEST_SM90_BENCH")) compare(18225,56,1);
    }
    std::printf("ok: SM%d attention is byte-identical to portable tiled attention\n",H3_TEST_TARGET_SM);
}
