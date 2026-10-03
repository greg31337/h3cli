/* Production-sized BF16 attention versus the permanent scalar reference. */
#include "src/gpu.h"
#include "src/device.h"
#include "cuda_operator_reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static char error[1024];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
static float fp(uint16_t x){uint32_t u=(uint32_t)x<<16;float f;memcpy(&f,&u,4);return f;}
static uint16_t *values(size_t count,unsigned *seed){
    uint16_t *v=malloc(count*2);CHECK(v);
    for(size_t i=0;i<count;i++){*seed^=*seed<<13;*seed^=*seed>>17;*seed^=*seed<<5;
        float x=(float)((int)(*seed%4097)-2048)/1024.0f;uint32_t bits;memcpy(&bits,&x,4);v[i]=(uint16_t)((bits+0x7fff+((bits>>16)&1))>>16);}
    return v;
}
static void compare(unsigned sequence,unsigned heads,unsigned dim,int head_major){
    size_t count=(size_t)sequence*heads*dim;unsigned seed=123;
    uint16_t *q=values(count,&seed),*k=values(count,&seed),*v=values(count,&seed),*result[2]={malloc(count*2),malloc(count*2)};
    CHECK(result[0]&&result[1]);double seconds[2];
    for(int pass=0;pass<2;pass++){
        h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
        h3_gpu_tensor *qt=h3_gpu_tensor_from_bf16(g,q,count),*kt=h3_gpu_tensor_from_bf16(g,k,count),*vt=h3_gpu_tensor_from_bf16(g,v,count),*out=h3_gpu_tensor_new_bf16(g,count);CHECK(qt&&kt&&vt&&out);
        seconds[pass]=0;
        for(int iteration=0;iteration<3;iteration++){
            double start=now();CHECK(h3_gpu_begin(g));
            float scale=1/sqrtf((float)dim);
            CHECK(!pass?test_cuda_attention_reference(g,out,qt,kt,vt,sequence,heads,dim,scale,head_major):
                  head_major?h3_gpu_sdpa_bf16_head_major_output(g,out,qt,kt,vt,sequence,heads,dim,scale):
                             h3_gpu_sdpa_bf16(g,out,qt,kt,vt,sequence,heads,dim,scale));
            CHECK(h3_gpu_submit(g));if(iteration)seconds[pass]+=(now()-start)/2;
        }
        CHECK(h3_gpu_tensor_read_bf16(out,result[pass],count));
        h3_gpu_tensor_free(qt);h3_gpu_tensor_free(kt);h3_gpu_tensor_free(vt);h3_gpu_tensor_free(out);h3_gpu_free(g);
    }
    double max=0,absolute=0,square=0,norm=0,actual_norm=0,dot=0;
    for(size_t i=0;i<count;i++){double a=fp(result[0][i]),b=fp(result[1][i]),delta=b-a;CHECK(isfinite(a)&&isfinite(b));max=fmax(max,fabs(delta));absolute+=fabs(delta);square+=delta*delta;norm+=a*a;actual_norm+=b*b;dot+=a*b;}
    double relative=sqrt(square/fmax(norm,1e-30));
    printf("{\"shape\":[%u,%u,%u],\"head_major\":%d,\"max_abs\":%.9g,\"mean_abs\":%.9g,\"relative_l2\":%.9g,\"cosine\":%.9g,\"reference_seconds\":%.6f,\"tiled_seconds\":%.6f,\"speedup\":%.3f}\n",sequence,heads,dim,head_major,max,absolute/(double)count,relative,dot/sqrt(fmax(norm*actual_norm,1e-30)),seconds[0],seconds[1],seconds[0]/seconds[1]);fflush(stdout);
    CHECK(max<0.016&&relative<0.0005);
    free(q);free(k);free(v);free(result[0]);free(result[1]);
}
int main(void){
    h3_device_info info;CHECK(h3_device_query(&info,error,sizeof(error))&&!strcmp(info.backend,"cuda"));
    compare(19,4,16,0);compare(513,3,64,0);compare(1025,4,256,1);compare(2048,8,128,0);compare(18225,4,128,1);
    puts("ok: tiled BF16 attention matches the permanent F32-accumulating reference");return 0;
}
