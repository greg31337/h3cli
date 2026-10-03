/* Compare optimized convolution with the permanent direct CUDA reference. */
#include "src/gpu.h"
#include "src/device.h"
#include "cuda_operator_reference.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char error[1024];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec/1e9;}
static float *values(size_t n,unsigned seed){float *v=malloc(n*4);CHECK(v);for(size_t i=0;i<n;i++){seed=1664525*seed+1013904223;v[i]=(float)(seed>>8)/16777216.0f-0.5f;}return v;}

static void compare(unsigned batch,unsigned d,unsigned h,unsigned width,unsigned ci,unsigned co,unsigned kernel,unsigned stride) {
    size_t ni=(size_t)batch*d*h*width*ci,nw=(size_t)co*ci*kernel*kernel*kernel;
    size_t no=(size_t)batch*((d-kernel)/stride+1)*((h-kernel)/stride+1)*((width-kernel)/stride+1)*co;
    float *input=values(ni,72),*weight=values(nw,73),*bias=values(co,74),*outputs[2]={malloc(no*4),malloc(no*4)};
    CHECK(outputs[0]&&outputs[1]);double seconds[2];
    for(int pass=0;pass<2;pass++) {
        h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
        h3_gpu_tensor *x=h3_gpu_tensor_from_f32(g,input,ni),*w=h3_gpu_tensor_from_f32(g,weight,nw),*b=h3_gpu_tensor_from_f32(g,bias,co),*o=h3_gpu_tensor_new_f32(g,no);
        CHECK(x&&w&&b&&o);seconds[pass]=0;
        for(int iteration=0;iteration<3;iteration++) {
            double start=now();
            CHECK(h3_gpu_begin(g));
            CHECK(!pass?test_cuda_conv_reference(g,o,x,w,b,batch,d,h,width,ci,co,kernel,stride):
                  h3_gpu_conv3d_f32(g,o,x,w,b,batch,d,h,width,ci,co,kernel,kernel,kernel,stride,stride,stride));
            CHECK(h3_gpu_submit(g));
            if(iteration)seconds[pass]+=(now()-start)/2;
        }
        CHECK(h3_gpu_tensor_read_f32(o,outputs[pass],no));
        h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);h3_gpu_tensor_free(b);h3_gpu_tensor_free(o);h3_gpu_free(g);
    }
    double max=0,sum=0,norm=0,mean=0;
    for(size_t i=0;i<no;i++){CHECK(isfinite(outputs[0][i])&&isfinite(outputs[1][i]));double delta=(double)outputs[0][i]-outputs[1][i];max=fmax(max,fabs(delta));mean+=fabs(delta);sum+=delta*delta;norm+=(double)outputs[0][i]*outputs[0][i];}
    double relative=sqrt(sum/fmax(norm,1e-30));
    printf("{\"shape\":[%u,%u,%u,%u,%u,%u,%u,%u],\"max_abs\":%.9g,\"mean_abs\":%.9g,\"relative_l2\":%.9g,\"direct_seconds\":%.6f,\"gemm_seconds\":%.6f,\"speedup\":%.3f}\n",batch,d,h,width,ci,co,kernel,stride,max,mean/(double)no,relative,seconds[0],seconds[1],seconds[0]/seconds[1]);fflush(stdout);
    CHECK(max<0.001&&relative<0.00005);
    free(input);free(weight);free(bias);free(outputs[0]);free(outputs[1]);
}
int main(void) {
    h3_device_info info;CHECK(h3_device_query(&info,error,sizeof(error))&&!strcmp(info.backend,"cuda"));
    compare(2,5,48,43,32,64,3,1);
    compare(2,7,49,44,16,48,3,2);
    compare(1,3,17,15,31,33,1,2);
    compare(1,3,128,128,128,128,3,1);
    /* A large odd filter forces 605-row packing chunks, so subsequent output
     * pointers lose their original allocation alignment and reuse a GEMM key. */
    compare(1,3,42,42,513,33,3,1);
    puts("ok: bounded convolution matches the direct CUDA reference");return 0;
}
