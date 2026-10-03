#include "src/media/refvideo.h"
#include "src/internal.h"
#include "src/media/ffmpeg.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
static char error[512];
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s (%s)\n",__FILE__,__LINE__,#x,error); exit(1); } } while (0)

static void geometry(void) {
    const int frames[]={39,56,73,107,124}, released[]={12,17,22,32,37}, raw[]={10,14,19,27,31};
    for (size_t i=0;i<5;i++) {
        CHECK(h3_ref2va_video_latent_t(frames[i])==released[i]);
        CHECK(h3_video_encoder_latent_t(frames[i])==raw[i]);
    }
    CHECK(h3_ref2va_video_vae_frames(48)==39);
    CHECK(h3_ref2va_video_vae_frames(60)==56);
    CHECK(h3_ref2va_video_vae_frames(INT_MAX)<=INT_MAX);
    CHECK(h3_ref2va_video_vae_frames(-1)==0);
    CHECK(h3_ref2va_video_latent_t(60)==0);
    CHECK(h3_ref2va_video_latent_t(1)==0);
    h3_refvideo_plan p;
    CHECK(h3_refvideo_plan_build(60,&p));
    CHECK(p.frames==60 && p.vae_frames==56 && p.latent_t==17 && p.soundtrack_samples==80000);
    CHECK(!h3_refvideo_plan_build(INT_MAX,&p));
    for (int n=5;n<=362;n++) {
        CHECK(h3_refvideo_plan_build(n,&p));
        int selected=n; while ((selected-5)%17) selected--;
        CHECK(p.frames==n && p.vae_frames==selected);
        CHECK(p.latent_t==(selected-5)/17*5+2);
    }
    int total=0;
    CHECK(!h3_refvideo_validate_duration(47,&total,error,sizeof(error)) && total==0);
    CHECK(h3_refvideo_validate_duration(48,&total,error,sizeof(error)) && total==48);
    CHECK(h3_refvideo_validate_duration(312,&total,error,sizeof(error)) && total==360);
    CHECK(!h3_refvideo_validate_duration(48,&total,error,sizeof(error)) && total==360);
    total=0; CHECK(!h3_refvideo_validate_duration(361,&total,error,sizeof(error)) && total==0);
    CHECK(h3_refvideo_validate_duration(360,&total,error,sizeof(error)) && total==360);
    total=0; for (int i=0;i<3;i++) CHECK(h3_refvideo_validate_duration(120,&total,error,sizeof(error)));
    total=0; CHECK(h3_refvideo_validate_duration(200,&total,error,sizeof(error)));
    CHECK(!h3_refvideo_validate_duration(200,&total,error,sizeof(error)) && total==200);
}

static void qwen(void) {
    for (int n=1;n<=362;n++) {
        int indices[31],count=0;
        for (int frame=0;frame<n;frame+=12) indices[count++]=frame;
        CHECK(h3_refvideo_qwen_blocks(n)==(size_t)((count+1)/2));
        for (int k=0;k<count;k+=2) {
            int first,second; double timestamp;
            CHECK(h3_refvideo_qwen_pair(n,(size_t)k/2,&first,&second,&timestamp));
            CHECK(first==indices[k] && second==indices[k+1<count?k+1:k]);
            CHECK(timestamp==((double)first+second)/48.0);
        }
    }
    int first,second; double time;
    CHECK(h3_refvideo_qwen_pair(72,2,&first,&second,&time));
    CHECK(first==48 && second==60 && time==2.25);
    CHECK(h3_refvideo_qwen_pair(56,2,&first,&second,&time));
    CHECK(first==48 && second==48 && time==2.0);
    CHECK(!h3_refvideo_qwen_pair(56,SIZE_MAX,&first,&second,&time));
    CHECK(!h3_refvideo_qwen_pair(0,0,&first,&second,&time));
    enum {T=72,H=2,W=3,N=3*T*H*W};
    float pixels[N],before[N];
    for (int c=0;c<3;c++) for (int t=0;t<T;t++) for (int p=0;p<H*W;p++)
        pixels[(c*T+t)*H*W+p]=(float)(c*100000+t*100+p);
    memcpy(before,pixels,sizeof(pixels));
    h3_refvideo_plan plan; CHECK(h3_refvideo_plan_build(T,&plan));
    CHECK(plan.frames==T && plan.vae_frames==56);
    for (size_t block=0;block<plan.qwen_blocks;block++) {
        CHECK(h3_refvideo_qwen_pair(T,block,&first,&second,&time));
        float *pair=h3_refvideo_extract_pair(pixels,T,H,W,first,second); CHECK(pair);
        for (int t=0;t<2;t++) for (int c=0;c<3;c++) for (int p=0;p<H*W;p++)
            CHECK(pair[(t*3+c)*H*W+p]==(float)(c*100000+(t?second:first)*100+p));
        free(pair);
    }
    CHECK(!memcmp(before,pixels,sizeof(pixels)));
    CHECK(!h3_refvideo_extract_pair(pixels,T,H,W,0,T));
}

static void keys(void) {
    h3_params p=H3_PARAMS_DEFAULT;
    h3_reference refs[]={{H3_REFERENCE_IMAGE,"inputs/1.jpg",NULL,0},
                         {H3_REFERENCE_VIDEO,"video.mkv",NULL,1}};
    p.references=refs;p.reference_count=2;
    char *key=h3_conditioning_key("prompt",&p,256,256,1);CHECK(key);
    CHECK(strstr(key,"ref2va-video-pipeline=released-v1"));
    char *other=h3_conditioning_key("different prompt",&p,256,256,1);CHECK(other);
    CHECK(strcmp(key,other));free(key);free(other);
}

static void dump(const char *base,const char *suffix,const void *data,size_t bytes) {
    char name[4096]; CHECK(snprintf(name,sizeof(name),"%s.%s",base,suffix)<(int)sizeof(name));
    FILE *f=fopen(name,"wb"); CHECK(f); CHECK(fwrite(data,1,bytes,f)==bytes); CHECK(!fclose(f));
}

int main(int argc,char **argv) {
    if (argc>=5 && !strcmp(argv[1],"--decode")) {
        float *pixels=NULL; h3_refvideo_plan p; int total=0;
        if (!h3_refvideo_read(argv[2],32,32,atoi(argv[3]),&total,&pixels,&p,error,sizeof(error))) {
            fprintf(stderr,"%s\n",error); return 1;
        }
        dump(argv[4],"rgb",pixels,(size_t)3*p.frames*32*32*4);
        float *pairs=malloc(p.qwen_blocks*6*32*32*4); CHECK(pairs);
        printf("{\"frames\":%d,\"vae_frames\":%d,\"latent_t\":%d,\"soundtrack_samples\":%d,\"pairs\":[",
            p.frames,p.vae_frames,p.latent_t,p.soundtrack_samples);
        for (size_t b=0;b<p.qwen_blocks;b++) {
            int first,second; double timestamp;
            CHECK(h3_refvideo_qwen_pair(p.frames,b,&first,&second,&timestamp));
            float *pair=h3_refvideo_extract_pair(pixels,p.frames,32,32,first,second); CHECK(pair);
            memcpy(pairs+b*6*32*32,pair,6*32*32*4); free(pair);
            printf("%s[%d,%d,%.9g]",b?",":"",first,second,timestamp);
        }
        puts("]}"); dump(argv[4],"pairs",pairs,p.qwen_blocks*6*32*32*4);
        if (argc>5) {
            float *pcm=NULL; int samples=0;
            CHECK(h3_ffmpeg_read_audio_f32(argv[5],p.soundtrack_samples,1,&pcm,&samples,error,sizeof(error)));
            dump(argv[4],"audio",pcm,(size_t)2*samples*4); free(pcm);
        }
        free(pairs); free(pixels); return 0;
    }
    geometry(); qwen(); keys();
    printf("ok: %d Ref2VA video geometry, duration, Qwen and cache checks\n",checks);
    return 0;
}
