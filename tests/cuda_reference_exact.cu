/* Mutable M3A tests; never included in the frozen parity suite. */
#define H3_CUDA_EXACT_TEST 1
#include "../src/cuda/gpu_cuda.cu"
#include <cstdio>
#define REQUIRE(x) do { if(!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while(0)
#define CU(x) REQUIRE((x)==cudaSuccess)
static h3_gpu *context(bool fast=false) {
    h3_params p=H3_PARAMS_DEFAULT;p.cuda_attention=fast?1:0;
    char err[1024];h3_cuda_policy policy;
    REQUIRE(h3_cuda_policy_resolve(&p,"cuda",0,&policy,err,sizeof(err)));
    auto old=h3_cuda_policy_exchange(policy);auto *g=h3_gpu_create(nullptr,err,sizeof(err));
    h3_cuda_policy_exchange(old);REQUIRE(g);return g;
}
static void staging() {
    char path[]="/tmp/h3-m3a-XXXXXX";int fd=mkstemp(path);REQUIRE(fd>=0);
    std::vector<uint16_t> data(16384),got(2051);
    for(size_t i=0;i<data.size();i++)data[i]=0x3c00+i%256;
    REQUIRE(write(fd,data.data(),data.size()*2)==(ssize_t)data.size()*2);REQUIRE(!fsync(fd));
    const uint64_t base=sglang_host_bytes;
    auto *g=context();REQUIRE(g->exact_mapped_weights&&g->sglang_reference&&!g->exact_fused_vae_casts);
    REQUIRE(!setenv("H3_TEST_EXACT_NO_MMAP","1",1));
    auto *copied=context();REQUIRE(!copied->exact_mapped_weights&&g->exact_mapped_weights);
    REQUIRE(!unsetenv("H3_TEST_EXACT_NO_MMAP"));
    auto *fast=context(true);REQUIRE(fast->exact_mapped_weights&&!fast->exact_fused_vae_casts);
    uint64_t pinned=g->stats.pinned_bytes;
    auto *t=h3_gpu_tensor_new_bf16(g,got.size());REQUIRE(t);
    for(int i=0;i<4;i++) {
        REQUIRE(h3_gpu_tensor_stream_file_bf16(t,path,6,got.size(),g->error,sizeof(g->error)));
        REQUIRE(h3_gpu_tensor_read_bf16(t,got.data(),got.size()));
        REQUIRE(!memcmp(got.data(),data.data()+3,got.size()*2));
        REQUIRE(g->stats.pinned_bytes-pinned==8192);
    }
    REQUIRE(g->exact_mapped_weight_entries==1&&g->sglang_host_hits==3);
    REQUIRE(h3_gpu_release_weight_cache(g));REQUIRE(sglang_host_bytes==base);
    /* Actual registration failure selects copied storage and preserves bytes. */
    exact_test_register_failure=true;
    REQUIRE(h3_gpu_sglang_preload_weight(g,path,0,2048));
    REQUIRE(g->exact_copied_weights==1);
    exact_test_register_failure=false;
    REQUIRE(h3_gpu_sglang_preload_weight(g,path,8192,2048));
    REQUIRE(g->sglang_host_weights.size()==2);
    exact_test_unregister_failure=true;
    REQUIRE(!h3_gpu_release_weight_cache(g)); // copied entry has already gone
    REQUIRE(g->sglang_host_weights.size()==1&&sglang_host_bytes-base==4096);
    h3_gpu_cancel(g);REQUIRE(h3_gpu_release_weight_cache(g));
    REQUIRE(h3_gpu_release_weight_cache(g));REQUIRE(sglang_host_bytes==base);
    /* In-place edits between completed reads invalidate metadata keys. */
    REQUIRE(h3_gpu_tensor_stream_file_bf16(t,path,6,got.size(),g->error,sizeof(g->error)));
    REQUIRE(h3_gpu_tensor_read_bf16(t,got.data(),got.size()));
    for(auto &x:data)x=0x3e80;
    REQUIRE(pwrite(fd,data.data(),data.size()*2,0)==(ssize_t)data.size()*2);REQUIRE(!fsync(fd));
    REQUIRE(h3_gpu_tensor_stream_file_bf16(t,path,6,got.size(),g->error,sizeof(g->error)));
    REQUIRE(h3_gpu_tensor_read_bf16(t,got.data(),got.size()));REQUIRE(!memcmp(got.data(),data.data()+3,got.size()*2));
    REQUIRE(g->sglang_host_weights.size()==2);REQUIRE(h3_gpu_release_weight_cache(g));
    /* Replacement inode and short-file/tail validation. */
    REQUIRE(!unlink(path));int fresh=open(path,O_CREAT|O_EXCL|O_RDWR,0600);REQUIRE(fresh>=0);
    data[3]=0x3f80;REQUIRE(write(fresh,data.data(),data.size()*2)==(ssize_t)data.size()*2);
    REQUIRE(h3_gpu_tensor_stream_file_bf16(t,path,6,got.size(),g->error,sizeof(g->error)));
    REQUIRE(h3_gpu_tensor_read_bf16(t,got.data(),got.size()));REQUIRE(got[0]==0x3f80);
    REQUIRE(!h3_gpu_sglang_preload_weight(g,path,data.size()*2-1,2));h3_gpu_cancel(g);
    REQUIRE(h3_gpu_release_weight_cache(g));
    REQUIRE(!setenv("H3_MEMORY_LIMIT_BYTES","1",1));
    REQUIRE(h3_gpu_tensor_stream_file_bf16(t,path,6,got.size(),g->error,sizeof(g->error)));
    REQUIRE(h3_gpu_tensor_read_bf16(t,got.data(),got.size()));REQUIRE(got[0]==0x3f80);
    REQUIRE(g->sglang_host_weights.empty());REQUIRE(!unsetenv("H3_MEMORY_LIMIT_BYTES"));
    for(int i=0;i<8;i++) {
        REQUIRE(h3_gpu_sglang_preload_weight(g,path,6,2051));
        REQUIRE(h3_gpu_release_weight_cache(g));REQUIRE(sglang_host_bytes==base);
        auto *transient=context(i%2);REQUIRE(h3_gpu_sglang_preload_weight(transient,path,0,2048));
        h3_gpu_free(transient);REQUIRE(sglang_host_bytes==base);
    }
    void *raw=nullptr;
    REQUIRE(sglang_host_weight(copied,fresh,6,4102,H3_GPU_BF16,&raw)&&raw);
    REQUIRE(copied->exact_copied_weights==1&&!copied->exact_mapped_weight_entries);
    REQUIRE(sglang_host_weight(fast,fresh,0,4096,H3_GPU_BF16,&raw)&&raw);
    REQUIRE(fast->exact_mapped_weight_entries==1);
    h3_gpu_tensor_free(t);h3_gpu_free(g);h3_gpu_free(copied);h3_gpu_free(fast);REQUIRE(sglang_host_bytes==base);
    std::vector<uint16_t> disk(data.size());REQUIRE(pread(fresh,disk.data(),disk.size()*2,0)==(ssize_t)disk.size()*2);
    REQUIRE(disk==data);REQUIRE(!close(fresh)&&!close(fd)&&!unlink(path));
    puts("PASS staging: policy capture, mapped/copied, tails, invalidation, replacement, pressure, partial cleanup, cancellation, lifetime, file bytes");
}
static void casts() {
    auto *g=context();const size_t n=1025;
    float samples[]={0.f,-0.f,1.f,-1.f,65504.f,65519.f,65520.f,-65520.f,
        0x1p-24f,0x1p-25f,-0x1p-25f,1.00048828125f,1.00146484375f,INFINITY,-INFINITY,NAN};
    float *in;__half *a,*b;unsigned *fa,*fb;
    CU(cudaMalloc(&in,3*n*4));CU(cudaMalloc(&a,3*n*2));CU(cudaMalloc(&b,3*n*2));
    CU(cudaMalloc(&fa,4));CU(cudaMalloc(&fb,4));
    for(int finite=0;finite<2;finite++) {
        std::vector<float> input(3*n);for(size_t i=0;i<input.size();i++)input[i]=finite?(i%127)*.001f:samples[i%16];
        CU(cudaMemcpy(in,input.data(),input.size()*4,cudaMemcpyHostToDevice));
        CU(cudaMemset(fa,0,4));CU(cudaMemset(fb,0,4));
        CU(cudaStreamBeginCapture(g->compute,cudaStreamCaptureModeThreadLocal));
        for(int j=0;j<3;j++)sg_vae_cast_half<<<5,256,0,g->compute>>>(a+j*n,in+j*n,n,fa);
        exact_vae_cast_qkv<<<5,256,0,g->compute>>>(b,b+n,b+2*n,in,in+n,in+2*n,n,fb);
        cudaGraph_t graph;cudaGraphExec_t exec;CU(cudaStreamEndCapture(g->compute,&graph));
        CU(cudaGraphInstantiate(&exec,graph,nullptr,nullptr,0));
        for(int repeat=0;repeat<3;repeat++) {
            CU(cudaGraphLaunch(exec,g->compute));CU(cudaStreamSynchronize(g->compute));
            std::vector<uint16_t> x(3*n),y(3*n);unsigned f1,f2;
            CU(cudaMemcpy(x.data(),a,x.size()*2,cudaMemcpyDeviceToHost));CU(cudaMemcpy(y.data(),b,y.size()*2,cudaMemcpyDeviceToHost));
            CU(cudaMemcpy(&f1,fa,4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(&f2,fb,4,cudaMemcpyDeviceToHost));
            REQUIRE(x==y&&f1==f2&&f1==(unsigned)!finite);
            if(finite) { // sticky fault cannot be cleared by a finite conversion
                unsigned one=1;CU(cudaMemcpy(fa,&one,4,cudaMemcpyHostToDevice));CU(cudaMemcpy(fb,&one,4,cudaMemcpyHostToDevice));
                CU(cudaGraphLaunch(exec,g->compute));CU(cudaStreamSynchronize(g->compute));
                CU(cudaMemcpy(&f1,fa,4,cudaMemcpyDeviceToHost));CU(cudaMemcpy(&f2,fb,4,cudaMemcpyDeviceToHost));REQUIRE(f1==1&&f2==1);
                CU(cudaMemset(fa,0,4));CU(cudaMemset(fb,0,4));
            }
        }
        CU(cudaGraphExecDestroy(exec));CU(cudaGraphDestroy(graph));
    }
    CU(cudaFree(in));CU(cudaFree(a));CU(cudaFree(b));CU(cudaFree(fa));CU(cudaFree(fb));h3_gpu_free(g);
    puts("PASS casts: all conversion bits, signed zeros, rounding ties, overflow/nonfinite, sticky faults, tails and graph replay");
}
__global__ static void delay_slot(unsigned long long cycles) {
    unsigned long long begin=clock64();while(clock64()-begin<cycles){}
}
static void slot_generations() {
    REQUIRE(!setenv("H3_CUDA_WEIGHT_MODE","auto",1));
    auto *g=context();const uint64_t base=sglang_host_bytes;
    REQUIRE(!setenv("H3_CUDA_WEIGHT_MODE","resident",1));
    h3_weight_plan plan;
    REQUIRE(h3_gpu_plan_bf16_weights(g,3,1<<20,0,0,-1,&plan));
    REQUIRE(plan.options.mode==H3_WEIGHTS_AUTO&&plan.resident_count==2);
    REQUIRE(!unsetenv("H3_CUDA_WEIGHT_MODE"));
    constexpr unsigned n=4096,generations=32;
    std::vector<uint16_t> expected(n*generations),got(expected.size());
    for(unsigned i=0;i<expected.size();i++)expected[i]=(uint16_t)(0x3f00+i/n);
    char path[]="/tmp/h3-slot-XXXXXX";int fd=mkstemp(path);REQUIRE(fd>=0);
    REQUIRE(write(fd,expected.data(),expected.size()*2)==(ssize_t)expected.size()*2);REQUIRE(!close(fd));
    auto *a=h3_gpu_tensor_new_bf16(g,n),*b=h3_gpu_tensor_new_bf16(g,n);
    auto *out=h3_gpu_tensor_new_bf16(g,expected.size());REQUIRE(a&&b&&out);
    REQUIRE(h3_gpu_begin(g));
    for(unsigned i=0;i<generations;i++) {
        auto *slot=i%2?b:a;
        delay_slot<<<1,1,0,g->copy>>>(300000);
        REQUIRE(h3_gpu_tensor_stream_file_bf16(slot,path,(uint64_t)i*n*2,n,g->error,sizeof(g->error)));
        delay_slot<<<1,1,0,g->compute>>>(500000);
        REQUIRE(h3_gpu_copy_bf16(g,out,(size_t)i*n,slot,0,n));
        REQUIRE(h3_gpu_continue(g)); /* No host/device fence between generations. */
    }
    REQUIRE(h3_gpu_submit(g)&&h3_gpu_tensor_read_bf16(out,got.data(),got.size()));
    REQUIRE(got==expected&&g->stats.streamed_bytes==expected.size()*2);
    /* Read failure, cancellation and another successful submission on one context. */
    REQUIRE(!h3_gpu_tensor_stream_file_bf16(a,"/nonexistent/h3-slot",0,n,g->error,sizeof(g->error)));
    h3_gpu_cancel(g);
    REQUIRE(h3_gpu_tensor_stream_file_bf16(a,path,0,n,g->error,sizeof(g->error)));
    REQUIRE(h3_gpu_begin(g)&&h3_gpu_copy_bf16(g,out,0,a,0,n)&&h3_gpu_submit(g));
    /* Teardown with a pending upload must drain before unregistering its source. */
    delay_slot<<<1,1,0,g->copy>>>(500000);
    REQUIRE(h3_gpu_tensor_stream_file_bf16(b,path,0,n,g->error,sizeof(g->error)));
    h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);h3_gpu_tensor_free(out);h3_gpu_free(g);
    REQUIRE(sglang_host_bytes==base);REQUIRE(!unlink(path));
    REQUIRE(!setenv("H3_CUDA_TEST_MEMORY_BUDGET","1024",1));g=context();
    REQUIRE(!unsetenv("H3_CUDA_TEST_MEMORY_BUDGET"));
    REQUIRE(!h3_gpu_tensor_new_bf16(g,1024)&&h3_gpu_should_retry_streaming(g));h3_gpu_free(g);
    g=context();a=h3_gpu_tensor_new_bf16(g,1024);REQUIRE(a);h3_gpu_tensor_free(a);h3_gpu_free(g);
    puts("PASS slots: delayed copies/consumers, 32 generations without host fences, read/cancel recovery, pending-copy teardown and captured allocation failure");
}
int main(){staging();slot_generations();casts();puts("PASS M3A exact substitutions and residency slots");}
