#include "src/vae/image_vae.h"
#include "src/weights/weights.h"
#include "src/host.h"
#include "src/media/ffmpeg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int cancel(int a,int b,void *p){(void)b;(void)p;return a>=1;}
static int write_raw(const char *p,const float *v,size_t n){FILE *f=fopen(p,"wb");if(!f)return 0;int ok=fwrite(v,4,n,f)==n;return !fclose(f)&&ok;}
int main(int argc,char **argv){
    if(argc<3)return 2;
    char error[512]={0},directory[4096],digest[65];
    if(!strcmp(argv[1],"png") && argc==5){
        int w=atoi(argv[2]),h=atoi(argv[3]);
        if(w<1||h<1||w>H3_STILL_MAX_DIMENSION||h>H3_STILL_MAX_DIMENSION||(uint64_t)w*h>H3_STILL_MAX_PIXELS)return 2;
        size_t n=(size_t)w*h*3;uint8_t *rgb=calloc(n,1);if(!rgb)return 1;
        int ok=h3_ffmpeg_write_png(argv[4],rgb,w,h,error,sizeof(error));free(rgb);if(!ok)goto fail;return 0;
    }
    if(!strcmp(argv[1],"validate")){h3_still_latent z={0};int ok=h3_still_latent_load(argv[2],&z,error,sizeof(error));h3_still_latent_free(&z);if(!ok)goto fail;return 0;}
    if(!strcmp(argv[1],"store")){h3_weight_store *s=h3_weight_store_open(argv[2],error,sizeof(error));if(!s)goto fail;h3_weight_store_free(s);return 0;}
    snprintf(directory,sizeof(directory),"%s/FL2VA/video_vae/source",argv[2]);
    if(!strcmp(argv[1],"fixture") && argc>=6){
        int w=atoi(argv[3]),h=atoi(argv[4]),ref=argc>6?atoi(argv[6]):0;
        if(w%32 || h%32 || !h3_still_geometry(h/16,w/16,error,sizeof(error)))goto fail;
        snprintf(directory,sizeof(directory),"%s/%s/video_vae/source",argv[2],ref?"Ref2VA":"FL2VA");
        if(!h3_still_vae_compatibility(directory,digest,error,sizeof(error)))goto fail;
        size_t n=(size_t)24*(size_t)(h/16)*(size_t)(w/16);float *v=malloc(n*sizeof(float));if(!v)return 1;
        h3_rng rng;h3_rng_seed(&rng,123);h3_rng_fill_normal(&rng,v,n);
        h3_still_latent z={.height=h/16,.width=w/16,.values=v,.reference_model=ref};strcpy(z.compatibility_sha256,digest);
        int ok=h3_still_latent_save(argv[5],&z,error,sizeof(error));free(v);if(!ok)goto fail;return 0;
    }
    if(!strcmp(argv[1],"inspect")){if(!h3_still_vae_compatibility(directory,digest,error,sizeof(error)))goto fail;puts(digest);return 0;}
    if(!strcmp(argv[1],"decode") && argc>=5){
        h3_still_latent z={0};h3_video_frames out={0};
        if(!h3_still_latent_load(argv[3],&z,error,sizeof(error)))goto fail;
        snprintf(directory,sizeof(directory),"%s/%s/video_vae/source",argv[2],z.reference_model?"Ref2VA":"FL2VA");
        h3_video_vae_decoder *d=h3_video_vae_still_load(directory,"src/metal/shaders.metal",z.height,z.width,NULL,NULL,error,sizeof(error));
        if(!d){h3_still_latent_free(&z);goto fail;}
        if(h3_video_vae_still_decode(d,z.values,cancel,NULL,&out,error,sizeof(error)) || out.rgb || !strstr(error,"cancel")){fputs("cancellation failed\n",stderr);return 1;}
        int ok=1,repeat=argc>5?atoi(argv[5]):1;float *first=NULL;size_t n=(size_t)z.height*z.width*256*3;
        for(int i=0;ok && i<repeat;i++){
            h3_video_frames_free(&out);ok=h3_video_vae_still_decode(d,z.values,NULL,NULL,&out,error,sizeof(error));
            if(ok && i==0){first=malloc(n*4);if(first)memcpy(first,out.rgb,n*4);else ok=0;}
            else if(ok && memcmp(first,out.rgb,n*4)){snprintf(error,sizeof(error),"resident repeat changed pixels");ok=0;}
        }
        if(ok)ok=write_raw(argv[4],out.rgb,n);
        free(first);h3_video_frames_free(&out);h3_still_latent_free(&z);h3_video_vae_decoder_free(d);if(!ok)goto fail;return 0;
    }
    return 2;
fail:fprintf(stderr,"still_probe: %s\n",error);return 1;
}
