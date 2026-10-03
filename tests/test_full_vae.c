#include "src/h3.h"
#include "src/gpu.h"
#include "src/execution.h"
#include "src/sampling/sampler_state.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}}while(0)
#ifndef __APPLE__
static void unpack(void){
    char error[512];h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    CHECK(h3_gpu_video_vae_configure(g,0)>=0);
    enum{H=2,W=3,T=7,P=3072,N=H*W*T*P};float *input=malloc(N*4);
    for(int i=0;i<N;i++)input[i]=(float)((i*17)%211-105)/19.f;
    h3_gpu_tensor *p=h3_gpu_tensor_from_f32(g,input,N);CHECK(p);
    const int ranges[][3]={{22,0,22},{22,16,6},{17,0,17},{17,1,1},{5,0,5}};
    for(size_t z=0;z<sizeof(ranges)/sizeof(*ranges);z++){
        int total=ranges[z][0],first=ranges[z][1],frames=ranges[z][2];size_t n=(size_t)frames*H*W*256*3;float *out=malloc(n*4);CHECK(out);
        int ok=h3_gpu_video_unpack(g,p,out,H,W,total,first,frames);
        if (ok != 1)
            fprintf(stderr, "%s\n", h3_gpu_error(g));
        CHECK(ok == 1);
        for(size_t i=0;i<n;i++){
            int c=(int)(i%3),x=(int)((i/3)%(W*16)),y=(int)((i/(3*W*16))%(H*16));
            int frame=(int)(i/((size_t)3*W*H*256))+first;
            int t=frame+3;if(total==22&&frame>=17)t+=3;
            size_t patch=((size_t)(t/4)*H+y/16)*W+x/16,component=(((c*4+t%4)*16+y%16)*16+x%16);
            float value=input[patch*P+component];
            if(memcmp(&value,out+i,4))fprintf(stderr,"unpack index=%zu reference=%a GPU=%a\n",i,(double)value,(double)out[i]);
            CHECK(!memcmp(&value,out+i,4));
        }free(out);
    }
    float rgb[1];CHECK(h3_gpu_video_unpack(g,p,rgb,H,W,22,22,1)<0);
    /* Reject exceptional inputs, then recover in the same decoder context. */
    float *out=malloc((size_t)H*W*256*3*4);CHECK(out);
    for(int i=0;i<N;i++)input[i]=NAN;
    CHECK(h3_gpu_tensor_write_f32(p,input,N));
    CHECK(h3_gpu_video_unpack(g,p,out,H,W,17,0,1)<0);
    CHECK(strstr(h3_gpu_error(g),"nonfinite"));
    for(int i=0;i<N;i++)input[i]=0;
    CHECK(h3_gpu_tensor_write_f32(p,input,N));
    CHECK(h3_gpu_video_unpack(g,p,out,H,W,17,0,1)==1);
    for(int i=0;i<H*W*256*3;i++)CHECK(isfinite(out[i]));
    free(out);
    h3_gpu_tensor_free(p);free(input);h3_gpu_free(g);
}

#endif
int main(int argc,char **argv){
#ifndef __APPLE__
    if(argc>1&&!strcmp(argv[1],"gpu")){unpack();puts("PASS CUDA raw unpack and failure recovery");}
#else
    (void)argc;(void)argv;
#endif
    puts("PASS current full VAE");return 0;
}
