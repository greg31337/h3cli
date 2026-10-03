#include "src/vae/tiny_vae.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>

typedef struct { FILE *file; int next; } output;
static int frames(const float *rgb,int first,int count,int width,int height,void *opaque){
    output *o=opaque;if(first!=o->next)return 1;o->next+=count;
    size_t n=(size_t)count*(size_t)width*(size_t)height*3;
    return fwrite(rgb,sizeof(float),n,o->file)!=n;
}
int main(int argc,char **argv){
    if(argc<4){fprintf(stderr,"usage: %s WEIGHTS STATE OUTPUT.f32 [BATCH]\n",argv[0]);return 2;}
    char error[512];double start=h3_av_now();h3_av_state *s=h3_av_state_load(argv[2],error,sizeof(error));
    if(!s){fprintf(stderr,"%s\n",error);return 1;}
    h3_tiny_vae *d=h3_tiny_vae_load(argv[1],error,sizeof(error));
    if(!d){fprintf(stderr,"%s\n",error);h3_av_state_free(s);return 1;}
    double loaded=h3_av_now();output o={fopen(argv[3],"wb"),0};if(!o.file)return 1;
    int ok=h3_tiny_vae_stream(d,s->video,s->info.video_t,s->info.latent_h,s->info.latent_w,
        argc>4?atoi(argv[4]):5,-1,frames,&o,NULL,NULL,error,sizeof(error));
    if(fclose(o.file))ok=0;
    double end=h3_av_now();
    if(!ok)fprintf(stderr,"%s\n",error);
    printf("{\"load_seconds\":%.9f,\"decode_write_seconds\":%.9f,\"frames\":%d,\"model_sha256\":\"%s\"}\n",loaded-start,end-loaded,o.next,h3_tiny_vae_digest(d));
    h3_tiny_vae_free(d);h3_av_state_free(s);return ok?0:1;
}
