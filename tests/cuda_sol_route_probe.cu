/* Export only the last bounded routing slab. No production mask cache needed. */
#include "src/cuda/cuda_sol.h"
#include <cstdio>
#include <cstdlib>
#include <cmath>
#include <cstring>
#include <vector>
#include <stdexcept>
#include <string>
static void check(cudaError_t x){if(x!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(x));}
int main(int argc,char**argv){
 void *workspace=nullptr,*buffers[4]={};void *context=nullptr;cudaStream_t stream=nullptr;int result=1;
 try{
    if(argc!=6)throw std::runtime_error("usage: QKV LAYOUT MINIMUM MASK STATS");
    FILE*f=fopen(argv[2],"rb");if(!f)throw std::runtime_error("layout open");unsigned header[3];if(fread(header,4,3,f)!=3)throw std::runtime_error("layout header");
    unsigned seq=header[0],qb=header[1],kb=header[2],heads=56;
    if(!seq||seq>131072||qb!=(seq+31)/32||kb!=(seq+63)/64)throw std::runtime_error("bounded layout shape");
    std::vector<h3_sol_block> qm(qb),km(kb);if(fread(qm.data(),16,qb,f)!=qb||fread(km.data(),16,kb,f)!=kb||fgetc(f)!=EOF)throw std::runtime_error("layout size");fclose(f);
    h3_sol_layout layout={seq,qb,kb,0,qm.data(),km.data()};h3_cuda_sol_options o=H3_CUDA_SOL_DEFAULT;o.min_exact=std::stof(argv[3]);
    h3_cuda_sol_plan p;char error[512]={};if(!h3_cuda_sol_plan_make(seq,heads,o,H3_CUDA_SOL_WORKSPACE_BYTES,&p,error,sizeof(error)))throw std::runtime_error(error);
    check(cudaStreamCreate(&stream));check(cudaMalloc(&workspace,p.bytes));size_t n=size_t(seq)*heads*128*2;std::vector<unsigned char> host(n);
    f=fopen(argv[1],"rb");if(!f)throw std::runtime_error("QKV open");
    for(int i=0;i<4;i++){check(cudaMalloc(&buffers[i],n));if(i<3){if(fread(host.data(),1,n,f)!=n)throw std::runtime_error("QKV size");check(cudaMemcpy(buffers[i],host.data(),n,cudaMemcpyHostToDevice));}}
    if(fgetc(f)!=EOF)throw std::runtime_error("QKV tail");fclose(f);
    context=h3_cuda_sol_create(workspace,p.bytes,stream,false,error,sizeof(error));if(!context||!h3_cuda_sol_configure(context,o,&layout,heads,error,sizeof(error)))throw std::runtime_error(error);
    float scale=1.f/sqrtf(128.f);unsigned bits;memcpy(&bits,&scale,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000;memcpy(&scale,&bits,4);
    if(!h3_cuda_sol_run(context,buffers[3],buffers[0],buffers[1],buffers[2],seq,heads,scale,false,false,error,sizeof(error)))throw std::runtime_error(error);
    check(cudaStreamSynchronize(stream));h3_cuda_sol_stats stats;if(!h3_cuda_sol_collect(context,&stats,error,sizeof(error)))throw std::runtime_error(error);
    unsigned first_head=(heads-1)/p.head_group*p.head_group,group=heads-first_head,first_query=(qb-1)/p.query_slab*p.query_slab,slab=qb-first_query;
    std::vector<unsigned char> mask(size_t(group)*slab*kb);check(cudaMemcpy(mask.data(),(unsigned char*)workspace+p.routes,mask.size(),cudaMemcpyDeviceToHost));
    f=fopen(argv[4],"wb");if(!f||fwrite(mask.data(),1,mask.size(),f)!=mask.size())throw std::runtime_error("mask write");fclose(f);
    f=fopen(argv[5],"w");if(!f)throw std::runtime_error("stats write");fprintf(f,"{\"sequence\":%u,\"heads\":%u,\"first_head\":%u,\"group\":%u,\"first_query\":%u,\"slab\":%u,\"keys\":%u,\"workspace\":%zu,\"approximate_pairs\":%llu}\n",seq,heads,first_head,group,first_query,slab,kb,p.bytes,(unsigned long long)stats.pairs[1]);fclose(f);result=0;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());}
 if(stream)cudaStreamSynchronize(stream);h3_cuda_sol_free(context);for(auto*p:buffers)if(p)cudaFree(p);if(workspace)cudaFree(workspace);if(stream)cudaStreamDestroy(stream);return result;
}
