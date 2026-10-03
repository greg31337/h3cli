/* Standalone real-QKV probe: no model/runtime or Python inference dependency. */
#include "src/cuda/cuda_sglang_flash.h"
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <vector>
#include <cmath>
int main(int argc,char**argv) {
 if(argc!=10)return 2;
 unsigned rows=atoi(argv[6]),heads=atoi(argv[7]),kv=atoi(argv[8]),dim=atoi(argv[9]);
 if(!rows||rows>1000000||!heads||heads>256||!kv||kv>heads||dim!=128)return 2;
 size_t nq=(size_t)rows*heads*dim,nk=(size_t)rows*kv*dim;
 void *q=nullptr,*k=nullptr,*v=nullptr,*o=nullptr;float*lse=nullptr;
 if(cudaMalloc(&q,nq*2)||cudaMalloc(&k,nk*2)||cudaMalloc(&v,nk*2)||cudaMalloc(&o,nq*2)||cudaMalloc(&lse,(size_t)rows*heads*4))return 1;
 void *ptrs[]={q,k,v};size_t counts[]={nq,nk,nk};
 for(int i=0;i<3;i++){std::vector<unsigned short>x(counts[i]);FILE*f=fopen(argv[i+2],"rb");if(!f)return 1;bool ok=fread(x.data(),2,x.size(),f)==x.size()&&fgetc(f)==EOF;fclose(f);if(!ok||cudaMemcpy(ptrs[i],x.data(),x.size()*2,cudaMemcpyHostToDevice))return 1;}
 char error[1024]={0};int ok=h3_sglang_flash(o,q,k,v,lse,1,rows,heads,kv,dim,1/sqrtf(dim),!strcmp(argv[1],"gqa"),0,nullptr,error,sizeof(error));
 if(ok!=1||cudaDeviceSynchronize()){fprintf(stderr,"%s\n",error);return 1;}
 std::vector<unsigned short>y(nq);if(cudaMemcpy(y.data(),o,nq*2,cudaMemcpyDeviceToHost))return 1;
 FILE*f=fopen(argv[5],"wbx");if(!f)return 1;bool wrote=fwrite(y.data(),2,y.size(),f)==y.size();if(fclose(f))wrote=false;
 cudaFree(q);cudaFree(k);cudaFree(v);cudaFree(o);cudaFree(lse);return wrote?0:1;
}
