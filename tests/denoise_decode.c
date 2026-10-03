/* Decode matched raw and x0 previews from the same released-model trajectory. */
#include "src/sampling/sampler_state.h"
#include "src/vae/video_vae.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static char error[512];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s (%s)\n",__LINE__,#x,error);exit(1);}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}
static void ppm(const char *path,const h3_video_frames *f,int index){
    FILE *out=fopen(path,"wb");CHECK(out);fprintf(out,"P6\n%d %d\n255\n",f->width,f->height);
    size_t n=(size_t)f->width*(size_t)f->height*3;const float *rgb=f->rgb+(size_t)index*n;
    for(size_t i=0;i<n;i++){CHECK(isfinite(rgb[i]));float x=fminf(255,fmaxf(0,rgb[i]*255));unsigned char v=(unsigned char)lrintf(x);CHECK(fwrite(&v,1,1,out)==1);}
    CHECK(!fclose(out));
}
int main(int argc,char **argv){
    CHECK(argc==3);char file[4096];snprintf(file,sizeof(file),"%s.denoised.h3sample",argv[2]);
    h3_sampler_state *s=h3_sampler_state_load(file,error,sizeof(error));CHECK(s);
    snprintf(file,sizeof(file),"%s/%s/video_vae/source",argv[1],s->ref2va?"Ref2VA":"FL2VA");
    h3_video_vae_decoder *decoder=h3_video_vae_decoder_load(file,"src/metal/shaders.metal",s->latent_h,s->latent_w,NULL,NULL,error,sizeof(error));CHECK(decoder);
    snprintf(file,sizeof(file),"%s.trace",argv[2]);FILE *trace=fopen(file,"rb");CHECK(trace);
    float *v=malloc(s->video_elements*4);CHECK(v);double seconds=0;int calls=0;
    const int steps[]={1,2,3,5,10,15,20};
    for(size_t i=0;i<sizeof(steps)/sizeof(*steps);i++) {
        int step=steps[i];if(step>s->total_steps)continue;
        for(int clean=0;clean<2;clean++) {
            if(clean){snprintf(file,sizeof(file),"%s.x0-step-%02d.f32",argv[2],step);FILE *f=fopen(file,"rb");CHECK(f);CHECK(fread(v,4,s->video_elements,f)==s->video_elements);fclose(f);}
            else{CHECK(!fseeko(trace,(off_t)((s->video_elements+s->audio_elements)*4*(size_t)step),SEEK_SET));CHECK(fread(v,4,s->video_elements,trace)==s->video_elements);}
            h3_video_frames frame={0};int index;double start=now();
            CHECK(h3_video_vae_decoder_preview(decoder,v,s->latent_t,&frame,&index,error,sizeof(error)));seconds+=now()-start;calls++;
            snprintf(file,sizeof(file),"%s.%s-step-%02d.ppm",argv[2],clean?"decoded-x0":"noisy",step);ppm(file,&frame,0);h3_video_frames_free(&frame);
            if(s->continuation && (step==1 || step==10 || step==20)) {
                /* Decode the whole latent for visual inspection of both the
                 * protected/bridge prefix and generated suffix. */
                CHECK(h3_video_vae_decoder_decode(decoder,v,s->latent_t,&frame,error,sizeof(error)));
                snprintf(file,sizeof(file),"%s.%s-prefix-step-%02d.ppm",argv[2],clean?"x0":"noisy",step);ppm(file,&frame,0);
                int transition=s->context_frames;if(transition>=frame.frames)transition=frame.frames-1;
                snprintf(file,sizeof(file),"%s.%s-transition-step-%02d.ppm",argv[2],clean?"x0":"noisy",step);ppm(file,&frame,transition);
                snprintf(file,sizeof(file),"%s.%s-suffix-step-%02d.ppm",argv[2],clean?"x0":"noisy",step);ppm(file,&frame,frame.frames-1);h3_video_frames_free(&frame);
            }
        }
    }
    printf("{\"preview_decode_calls\":%d,\"preview_decode_seconds\":%.9g,\"mean_preview_decode_seconds\":%.9g}\n",calls,seconds,seconds/calls);
    free(v);fclose(trace);h3_video_vae_decoder_free(decoder);h3_sampler_state_free(s);return 0;
}
