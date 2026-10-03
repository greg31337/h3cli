/* File-based native operator harness; production code owns all GPU arithmetic. */
#include "src/cuda/cuda_sol.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <cmath>
#include <vector>
#include <string>
#include <map>
#include <stdexcept>
#include <chrono>
static void check(cudaError_t c){if(c!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(c));}
static void read(const std::string &p,void *v,size_t n){FILE*f=fopen(p.c_str(),"rb");if(!f)throw std::runtime_error("open input");bool ok=fread(v,1,n,f)==n&&fgetc(f)==EOF;fclose(f);if(!ok)throw std::runtime_error("input size mismatch");}
static void write(const std::string &p,const void*v,size_t n){FILE*f=fopen(p.c_str(),"wb");if(!f)throw std::runtime_error("open output");bool ok=fwrite(v,1,n,f)==n;fclose(f);if(!ok)throw std::runtime_error("write output");}
struct allocation{void*p=nullptr;explicit allocation(size_t n){check(cudaMalloc(&p,n));}~allocation(){if(p)cudaFree(p);}unsigned char*data(){return (unsigned char*)p+256;}};
int main(int argc,char **argv){
 void *context=nullptr;cudaStream_t stream=nullptr;
 try{
    std::map<std::string,std::string>a;for(int i=1;i<argc;i+=2){if(i+1==argc)throw std::runtime_error("argument needs value");a[argv[i]]=argv[i+1];}
    auto integer=[&](const char*n,int d){return a.count(n)?std::stoi(a[n]):d;};
    unsigned seq=integer("--sequence",65),heads=integer("--heads",2);int repeat=integer("--repeat",1);
    if(!seq||seq>131072||!heads||heads>56||repeat<1||repeat>1000)throw std::runtime_error("test budget/shape");
    h3_cuda_sol_options o=H3_CUDA_SOL_DEFAULT;o.q_block=integer("--q",32);o.dense_layers=0;o.dense_steps=0;
    o.local_radius=integer("--radius",1);o.min_exact=a.count("--minimum")?std::stof(a["--minimum"]):.5f;
    o.tau=a.count("--tau")?std::stof(a["--tau"]):1.f;
    bool exact=integer("--exact",0),head_major=integer("--head-major",0);
    float scale=1.f/sqrtf(128.f);uint32_t bits;memcpy(&bits,&scale,4);bits=(bits+0x7fff+((bits>>16)&1))&0xffff0000u;memcpy(&scale,&bits,4);
    size_t n=size_t(seq)*heads*128*2;std::vector<unsigned char> input(3*n);read(a.at("--input"),input.data(),input.size());
    unsigned qb=(seq+o.q_block-1)/o.q_block,kb=(seq+63)/64;
    std::vector<unsigned char> metadata(12+(qb+kb)*sizeof(h3_sol_block));read(a.at("--layout"),metadata.data(),metadata.size());
    auto*header=(uint32_t*)metadata.data();if(header[0]!=seq||header[1]!=qb||header[2]!=kb)throw std::runtime_error("layout header mismatch");
    h3_sol_layout layout={seq,qb,kb,0,(h3_sol_block*)(metadata.data()+12),(h3_sol_block*)(metadata.data()+12+qb*16)};
    char error[512]={};h3_cuda_sol_plan plan;
    if(!h3_cuda_sol_plan_make(seq,heads,o,H3_CUDA_SOL_WORKSPACE_BYTES,&plan,error,sizeof(error)))throw std::runtime_error(error);
    check(cudaStreamCreate(&stream));allocation workspace(plan.bytes),q(n+512),k(n+512),v(n+512),out(n+512);
    allocation *tensors[]={&q,&k,&v,&out};
    for(auto*t:tensors)check(cudaMemset(t->p,0xa5,n+512));
    for(int i=0;i<3;i++)check(cudaMemcpy(tensors[i]->data(),input.data()+i*n,n,cudaMemcpyHostToDevice));
    context=h3_cuda_sol_create(workspace.p,plan.bytes,stream,true,error,sizeof(error));if(!context)throw std::runtime_error(error);
    if(!h3_cuda_sol_configure(context,o,&layout,heads,error,sizeof(error)))throw std::runtime_error(error);
    size_t free_before,total,free_after;check(cudaMemGetInfo(&free_before,&total));
    std::vector<double> samples;auto start=std::chrono::steady_clock::now();int ok=1;h3_cuda_sol_stats stats={};
    bool rejected=true;
    if(integer("--failure-tests",0)){
        rejected &= !h3_cuda_sol_run(context,q.data(),q.data(),k.data(),v.data(),seq,heads,scale,false,false,error,sizeof(error));
        rejected &= !h3_cuda_sol_run(context,out.data(),q.data(),k.data(),v.data(),seq+1,heads,scale,false,false,error,sizeof(error));
        rejected &= !h3_cuda_sol_create(workspace.data()+1,plan.bytes,stream,false,error,sizeof(error));
    }
    for(int i=0;i<repeat&&ok;i++){
        auto sample_start=std::chrono::steady_clock::now();
        ok=h3_cuda_sol_run(context,out.data(),q.data(),k.data(),v.data(),seq,heads,scale,head_major,exact,error,sizeof(error));
        check(cudaStreamSynchronize(stream));if(ok)ok=h3_cuda_sol_collect(context,&stats,error,sizeof(error));
        samples.push_back(std::chrono::duration<double>(std::chrono::steady_clock::now()-sample_start).count());
    }
    double seconds=std::chrono::duration<double>(std::chrono::steady_clock::now()-start).count();check(cudaMemGetInfo(&free_after,&total));
    bool guards=rejected,inputs=true,unchanged=true;std::vector<unsigned char> data(n+512);
    for(int i=0;i<4;i++){
        check(cudaMemcpy(data.data(),tensors[i]->p,n+512,cudaMemcpyDeviceToHost));
        for(unsigned j=0;j<256;j++)guards&=data[j]==0xa5&&data[n+256+j]==0xa5;
        if(i<3)inputs&=!memcmp(data.data()+256,input.data()+i*n,n);
        else {for(size_t j=0;j<n;j++)unchanged&=data[j+256]==0xa5;write(a.at("--output"),data.data()+256,n);}
    }
    bool exported=false;
    if(heads<=plan.head_group&&qb<=plan.query_slab){std::vector<unsigned char> routes(size_t(heads)*qb*kb);if(!h3_cuda_sol_routes(context,routes.data(),routes.size(),error,sizeof(error)))throw std::runtime_error(error);write(a.at("--routes"),routes.data(),routes.size());exported=true;}
    FILE*f=fopen(a.at("--stats").c_str(),"w");if(!f)throw std::runtime_error("stats open");
    fprintf(f,"{\"ok\":%s,\"guards\":%s,\"inputs_unchanged\":%s,\"output_unchanged\":%s,\"routes_exported\":%s,\"repeat\":%d,\"wall_seconds\":%.9f,\"free_before\":%zu,\"free_after\":%zu,\"workspace\":%zu,\"calls\":%llu,\"pairs\":[",
        ok?"true":"false",guards?"true":"false",inputs?"true":"false",unchanged?"true":"false",exported?"true":"false",repeat,seconds,free_before,free_after,plan.bytes,(unsigned long long)stats.calls);
    for(int j=0;j<7;j++)fprintf(f,"%s%llu",j?",":"",(unsigned long long)stats.pairs[j]);
    fprintf(f,"],\"seconds\":{\"finite\":%.9f,\"summary\":%.9f,\"router\":%.9f,\"fused\":%.9f,\"output\":%.9f},\"profiled_regions\":%llu,\"unprofiled_regions\":%llu}\n",stats.finite_seconds,stats.summary_seconds,stats.route_seconds,stats.fused_seconds,stats.output_seconds,(unsigned long long)stats.profiled_regions,(unsigned long long)stats.unprofiled_regions);fclose(f);
    FILE *timings=fopen((a.at("--stats")+".timings.json").c_str(),"w");if(!timings)throw std::runtime_error("timings open");
    fputs("[",timings);for(size_t i=0;i<samples.size();i++)fprintf(timings,"%s%.9f",i?",":"",samples[i]);fputs("]\n",timings);fclose(timings);
    h3_cuda_sol_free(context);context=nullptr;cudaStreamDestroy(stream);stream=nullptr;
    if(!ok)fprintf(stderr,"%s\n",error);return !guards||!inputs?4:ok?0:3;
 }catch(const std::exception&e){fprintf(stderr,"%s\n",e.what());h3_cuda_sol_free(context);if(stream)cudaStreamDestroy(stream);return 2;}
}
