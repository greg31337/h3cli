#include "src/vae/video_encoder.h"
#include "src/vae/video_posterior.h"
#include "src/sampling/av_state.h"
#include "src/execution.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void die(const char *s){fprintf(stderr,"%s\n",s);exit(1);}
static void dump(const char *prefix,const char *name,const float *v,size_t n){
    char p[4096];snprintf(p,sizeof(p),"%s.%s",prefix,name);FILE *f=fopen(p,"wb");if(!f||fwrite(v,4,n,f)!=n||fclose(f))die("output failed");
}
int main(int argc,char **argv){
    if(argc!=7)die("usage: fast_vae_encoder WEIGHTS PIXELS.f32 FRAMES HEIGHT WIDTH OUTPUT_PREFIX");
    int frames=atoi(argv[3]),h=atoi(argv[4]),w=atoi(argv[5]);if(frames<1||frames>124||h<16||w<16||h%16||w%16)die("invalid dimensions");
    size_t n=(size_t)3*frames*h*w;float *pixels=malloc(n*4);FILE *f=fopen(argv[2],"rb");if(!pixels||!f||fread(pixels,4,n,f)!=n||fgetc(f)!=EOF)die("pixel input failed");fclose(f);
    char error[512];h3_video_moments m={0};double start=h3_av_now();
    if(!h3_video_vae_encode_moments(argv[1],"src/metal/shaders.metal",pixels,frames,h,w,NULL,NULL,&m,error,sizeof(error)))die(error);
    double seconds=h3_av_now()-start;size_t count=(size_t)24*m.time*m.height*m.width;
    float *epsilon=malloc(count*4),*sample=malloc(count*4);if(!epsilon||!sample||!h3_video_posterior_epsilon(epsilon,count)||!h3_video_posterior_sample(m.values,epsilon,count,sample))die("posterior failed");
    dump(argv[6],"moments",m.values,count*2);dump(argv[6],"epsilon",epsilon,count);dump(argv[6],"sample",sample,count);
    printf("{\"frames\":%d,\"height\":%d,\"width\":%d,\"latent_time\":%d,\"seconds\":%.9f,\"convolution_seconds\":%.9f,\"copy_seconds\":%.9f,\"allocations\":%llu,\"peak_tensor_bytes\":%llu}\n",frames,h,w,m.time,seconds,m.gpu_stats.conv_seconds,m.gpu_stats.transfer_seconds,(unsigned long long)m.gpu_stats.tensor_allocations,(unsigned long long)m.gpu_stats.peak_live_bytes);
    h3_video_moments_free(&m);free(pixels);free(epsilon);free(sample);return 0;
}
