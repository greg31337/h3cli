#include "src/cuda/cuda_sage_internal.cuh"
#include <cstdio>
#include <climits>
static __global__ void validate_output(const nv_bfloat16 *out,size_t n,unsigned *fault) {
    for(size_t i=(size_t)blockIdx.x*blockDim.x+threadIdx.x;i<n;i+=(size_t)gridDim.x*blockDim.x)
        if(!isfinite(__bfloat162float(out[i])))atomicMax(fault,2u);
}
void *h3_sage_create(void *workspace,size_t bytes,cudaStream_t stream,bool profile,char *error,size_t size) {
    sage_context *s=nullptr;
    try {
        if(!workspace||bytes<H3_ATTENTION_WORKSPACE_BYTES)throw std::runtime_error("Sage requires 512 MiB admitted workspace");
        s=new sage_context;s->workspace=(char*)workspace;s->capacity=bytes;s->stream=stream;s->profiling=profile;
        sage_check(cudaMemsetAsync(s->fault(),0,256,stream));
        if(cublasCreate(&s->blas)!=CUBLAS_STATUS_SUCCESS)throw std::runtime_error("Sage correction cuBLAS creation failed");
        if(cublasSetStream(s->blas,stream)!=CUBLAS_STATUS_SUCCESS||cublasSetMathMode(s->blas,CUBLAS_DEFAULT_MATH)!=CUBLAS_STATUS_SUCCESS||
           cublasSetWorkspace(s->blas,s->workspace+256,4u<<20)!=CUBLAS_STATUS_SUCCESS)
            throw std::runtime_error("Sage correction cuBLAS configuration failed");
        return s;
    }catch(const std::exception &e){if(error&&size)snprintf(error,size,"Sage initialization: %s",e.what());h3_sage_free(s);return nullptr;}
}
void h3_sage_free(void *context){
    auto *s=(sage_context*)context;if(!s)return;
    cudaStreamSynchronize(s->stream);
    for(auto &t:s->timings){cudaEventDestroy(t.start);cudaEventDestroy(t.end);}
    if(s->blas)cublasDestroy(s->blas);delete s;
}
int h3_sage_run(void *context,int mode,void *out,const void *q,const void *k,const void *v,
    unsigned seq,unsigned heads,float scale,bool head_major,char *error,size_t size) {
    try {
        auto *s=(sage_context*)context;
        if(!s||!out||!q||!k||!v||!seq||heads!=56||(size_t)seq*heads*128>INT_MAX||
           mode<1||mode>2||!std::isfinite(scale)||((uintptr_t)out|(uintptr_t)q|(uintptr_t)k|(uintptr_t)v)%16)
            throw std::runtime_error("invalid Sage shape, pointers, alignment or precision");
        if(mode==1)sage2_run(*s,(nv_bfloat16*)out,(const nv_bfloat16*)q,(const nv_bfloat16*)k,(const nv_bfloat16*)v,seq,heads,scale,head_major);
        else sage3_run(*s,(nv_bfloat16*)out,(const nv_bfloat16*)q,(const nv_bfloat16*)k,(const nv_bfloat16*)v,seq,heads,scale,head_major);
        {sage_timer t(*s,4);size_t n=(size_t)seq*heads*128;
            validate_output<<<std::min<size_t>((n+255)/256,65535),256,0,s->stream>>>((const nv_bfloat16*)out,n,s->fault());}
        sage_check(cudaGetLastError());s->stats.calls[mode]++;return 1;
    }catch(const std::exception &e){if(error&&size)snprintf(error,size,"Sage attention: %s",e.what());return 0;}
}
int h3_sage_collect(void *context,h3_sage_stats *stats,char *error,size_t size){
    auto *s=(sage_context*)context;if(!s)return 1;
    try {
        /* Called after the owning compute stream has synchronized. */
        sage_check(s->timer_error);
        unsigned fault=0;sage_check(cudaMemcpy(&fault,s->fault(),sizeof(fault),cudaMemcpyDeviceToHost));
        if(fault)throw std::runtime_error(fault==1?"nonfinite input encountered in Sage attention":"nonfinite output encountered in Sage attention");
        double *values[]={&s->stats.reduction_seconds,&s->stats.pack_seconds,&s->stats.correction_seconds,&s->stats.kernel_seconds,&s->stats.output_seconds};
        for(auto &t:s->timings){float ms=0;sage_check(cudaEventElapsedTime(&ms,t.start,t.end));*values[t.category]+=ms/1000.;}
        for(auto &t:s->timings){cudaEventDestroy(t.start);cudaEventDestroy(t.end);}s->timings.clear();
        if(stats)*stats=s->stats;return 1;
    }catch(const std::exception &e){if(error&&size)snprintf(error,size,"Sage synchronization: %s",e.what());return 0;}
}

/* Cancellation is the only boundary that clears a sticky asynchronous fault.
 * Clearing it per layer would hide an earlier failed layer in a submission. */
int h3_sage_reset(void *context,char *error,size_t size) {
    auto *s=(sage_context*)context;if(!s)return 1;
    try {
        sage_check(cudaStreamSynchronize(s->stream));
        sage_check(cudaMemsetAsync(s->fault(),0,256,s->stream));
        sage_check(cudaStreamSynchronize(s->stream));
        for(auto &t:s->timings){cudaEventDestroy(t.start);cudaEventDestroy(t.end);}s->timings.clear();
        s->timer_error=cudaSuccess;return 1;
    }catch(const std::exception &e){if(error&&size)snprintf(error,size,"Sage cancellation: %s",e.what());return 0;}
}
