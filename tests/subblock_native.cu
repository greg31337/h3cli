/* Standalone native operator: bounded deterministic fixtures or recorded QKV.
 * Python validation reads the routes and compares an independent FP64 oracle. */
#include "src/cuda/cuda_subblock.h"
#include "src/denoise/attention.h"
#include <cuda_bf16.h>
#include <algorithm>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);return 1;}}while(0)
static unsigned short bf(float x){auto b=__float2bfloat16_rn(x);return *reinterpret_cast<unsigned short*>(&b);}
static bool save(const std::string &path,const void *data,size_t bytes){FILE *f=fopen(path.c_str(),"wb");if(!f)return false;bool ok=fwrite(data,1,bytes,f)==bytes;return fclose(f)==0&&ok;}
int main(int argc,char **argv) {
    if(argc<7||argc>8){fprintf(stderr,"usage: subblock_native OUT SEQUENCE HEADS SPARSITY FORCE_ALL KIND [QKV]\n");return 2;}
    char error[1024]={};std::string root=argv[1];unsigned seq=(unsigned)atoi(argv[2]),heads=(unsigned)atoi(argv[3]);
    float sparsity=strtof(argv[4],nullptr);bool force=atoi(argv[5])!=0;int kind=atoi(argv[6]);
    CHECK(seq&&seq<=8192&&heads&&heads<=56);
    size_t n=size_t(seq)*heads*128;std::vector<unsigned short> q(n),k(n),v(n),output(n);
    if(argc==8){FILE *f=fopen(argv[7],"rb");CHECK(f);CHECK(fread(q.data(),2,n,f)==n&&fread(k.data(),2,n,f)==n&&fread(v.data(),2,n,f)==n);CHECK(fgetc(f)==EOF);fclose(f);}
    else for(size_t i=0;i<n;i++) {
        q[i]=bf(kind==1?0:float(int((i*7+3)%71)-35)*.03125f);
        k[i]=bf(kind==1?0:float(int((i*11+1)%67)-33)*.03125f);
        v[i]=bf(kind==1?0.5f:float(int((i*13+5)%79)-39)*.025f);
    }
    if(kind==2)q[n-1]=0x7fc0;
    if(kind==3)for(size_t i=0;i<n;i++){q[i]=bf(100);k[i]=bf(i%2?100:-100);}
    h3_sol_layout layout={};layout.sequence=seq;layout.query_blocks=layout.key_blocks=(seq+63)/64;
    std::vector<h3_sol_block> meta(layout.query_blocks);
    for(unsigned b=0;b<layout.query_blocks;b++){meta[b].rows=std::min(64u,seq-b*64);meta[b].protect=kind==4&&(b==0||b==layout.query_blocks-1);}
    layout.query=layout.key=meta.data();
    size_t capacity=H3_ATTENTION_WORKSPACE_BYTES;
    if(const char *limit=getenv("H3_TEST_SUBBLOCK_WORKSPACE"))capacity=strtoull(limit,nullptr,10);
    h3_subblock_plan plan;CHECK(h3_subblock_plan_make(seq,heads,sparsity,capacity,&plan,error,sizeof(error)));
    void *workspace=nullptr,*dq=nullptr,*dk=nullptr,*dv=nullptr,*out=nullptr;cudaStream_t stream;CHECK(cudaStreamCreate(&stream)==cudaSuccess);
    CHECK(cudaMalloc(&workspace,capacity)==cudaSuccess);
    CHECK(cudaMalloc(&dq,n*2)==cudaSuccess&&cudaMalloc(&dk,n*2)==cudaSuccess&&cudaMalloc(&dv,n*2)==cudaSuccess&&cudaMalloc(&out,n*2)==cudaSuccess);
    CHECK(cudaMemcpy(dq,q.data(),n*2,cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(dk,k.data(),n*2,cudaMemcpyHostToDevice)==cudaSuccess&&cudaMemcpy(dv,v.data(),n*2,cudaMemcpyHostToDevice)==cudaSuccess);
    void *context=h3_cuda_subblock_create(workspace,capacity,stream,true,error,sizeof(error));CHECK(context);
    CHECK(!h3_cuda_subblock_configure(context,nullptr,heads,sparsity,error,sizeof(error)));
    unsigned saved_rows=meta[0].rows;meta[0].rows=0;
    CHECK(!h3_cuda_subblock_configure(context,&layout,heads,sparsity,error,sizeof(error)));meta[0].rows=saved_rows;
    CHECK(h3_cuda_subblock_configure(context,&layout,heads,sparsity,error,sizeof(error)));
    CHECK(!h3_cuda_subblock_run(context,dq,dq,dk,dv,seq,heads,1/sqrtf(128.f),false,force,false,error,sizeof(error)));
    h3_cuda_subblock_stats stats={};float milliseconds[2]={};
    for(int hm=0;hm<2;hm++) {
        cudaEvent_t start,end;CHECK(cudaEventCreate(&start)==cudaSuccess&&cudaEventCreate(&end)==cudaSuccess);
        CHECK(cudaEventRecord(start,stream)==cudaSuccess);
        CHECK(h3_cuda_subblock_run(context,out,dq,dk,dv,seq,heads,1/sqrtf(128.f),hm!=0,force,false,error,sizeof(error)));
        CHECK(cudaEventRecord(end,stream)==cudaSuccess&&cudaStreamSynchronize(stream)==cudaSuccess);
        int finite=h3_cuda_subblock_collect(context,&stats,error,sizeof(error));
        if(kind==2){
            CHECK(!finite);CHECK(h3_cuda_subblock_reset(context,error,sizeof(error)));
            q[n-1]=0;CHECK(cudaMemcpy(dq,q.data(),n*2,cudaMemcpyHostToDevice)==cudaSuccess);
            CHECK(h3_cuda_subblock_run(context,out,dq,dk,dv,seq,heads,1/sqrtf(128.f),false,force,false,error,sizeof(error)));
            CHECK(cudaStreamSynchronize(stream)==cudaSuccess&&h3_cuda_subblock_collect(context,&stats,error,sizeof(error)));
            cudaEventDestroy(start);cudaEventDestroy(end);break;
        }
        CHECK(finite);CHECK(cudaEventElapsedTime(&milliseconds[hm],start,end)==cudaSuccess);
        CHECK(cudaMemcpy(output.data(),out,n*2,cudaMemcpyDeviceToHost)==cudaSuccess);
        CHECK(save(root+(hm?".head.bf16":".row.bf16"),output.data(),n*2));cudaEventDestroy(start);cudaEventDestroy(end);
    }
    if(kind!=2) {
        if(plan.head_group==heads&&plan.query_slab==plan.blocks) {
            size_t routes=size_t(heads)*plan.blocks*plan.blocks;std::vector<unsigned char> selected(routes);std::vector<float> scores(routes);
            CHECK(h3_cuda_subblock_routes(context,selected.data(),scores.data(),routes,error,sizeof(error)));
            CHECK(save(root+".routes",selected.data(),routes)&&save(root+".scores",scores.data(),routes*4));
        } else CHECK(force); /* all-selected oracle tests tiling without route capture */
        if(argc!=8)CHECK(save(root+".q.bf16",q.data(),n*2)&&save(root+".k.bf16",k.data(),n*2)&&save(root+".v.bf16",v.data(),n*2));
    }
    printf("{\"sequence\":%u,\"heads\":%u,\"keep\":%u,\"selected\":%llu,\"possible\":%llu,\"milliseconds\":[%.6f,%.6f],\"workspace\":%zu,\"nonfinite_rejected\":%s}\n",
        seq,heads,plan.keep,(unsigned long long)stats.pairs[0],(unsigned long long)stats.pairs[1],milliseconds[0],milliseconds[1],plan.bytes,kind==2?"true":"false");
    h3_cuda_subblock_free(context);cudaFree(out);cudaFree(dv);cudaFree(dk);cudaFree(dq);cudaFree(workspace);cudaStreamDestroy(stream);return 0;
}
