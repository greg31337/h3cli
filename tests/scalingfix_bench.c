/* Rotate warmed variants, amortizing CPU submission and GPU frequency changes. */
#include "src/gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s\n",__LINE__,#x);exit(1);}}while(0)
int main(void){
    char error[512];h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(gpu);
    const char *mode[]={"legacy","scaled-q","reference"};unsigned sequences[]={13,128,829,2048};
    for(size_t c=0;c<4;c++){
        unsigned n=sequences[c],batch=n<200?64:2;size_t nq=(size_t)n*64*128,nk=(size_t)n*8*128;
        uint16_t *data=malloc(nq*2);CHECK(data);for(size_t i=0;i<nq;i++)data[i]=(uint16_t)(0x3e80+(i%256));
        h3_gpu_tensor *q=h3_gpu_tensor_from_bf16(gpu,data,nq),*k=h3_gpu_tensor_from_bf16(gpu,data,nk),*v=h3_gpu_tensor_from_bf16(gpu,data,nk),*out=h3_gpu_tensor_new_bf16(gpu,nq);CHECK(q&&k&&v&&out);
        for(int round=0;round<12;round++)for(int order=0;order<3;order++){
            int m=(round+order)%3;setenv("H3_QWEN_GQA_SCALE_MODE",mode[m],1);h3_gpu_stats a,b;CHECK(h3_gpu_get_stats(gpu,&a));
            CHECK(h3_gpu_begin(gpu));for(unsigned r=0;r<batch;r++)CHECK(h3_gpu_gqa_causal_bf16(gpu,out,q,k,v,n,64,8,128,1/sqrtf(128)));
            CHECK(h3_gpu_submit(gpu));CHECK(h3_gpu_get_stats(gpu,&b));
            if(round>1)printf("{\"sequence\":%u,\"mode\":\"%s\",\"round\":%d,\"gpu_seconds\":%.9g,\"wait_seconds\":%.9g,\"batch\":%u}\n",n,mode[m],round,(b.gpu_seconds-a.gpu_seconds)/batch,(b.command_wait_seconds-a.command_wait_seconds)/batch,batch);
            CHECK(a.allocated_bytes==b.allocated_bytes);
        }
        fflush(stdout);free(data);h3_gpu_tensor_free(q);h3_gpu_tensor_free(k);h3_gpu_tensor_free(v);h3_gpu_tensor_free(out);
    }
    h3_gpu_free(gpu);return 0;
}
