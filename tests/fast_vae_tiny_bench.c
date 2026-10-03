/* Timing comparator only; the existing TAEH3 decoder is unchanged. */
#include "src/vae/tiny_vae.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
static int receive(const float *rgb,int first,int n,int w,int h,void *opaque){
    (void)rgb;(void)w;(void)h;int *next=opaque;
    if(first!=*next)return 1;*next+=n;return 0;
}
int main(int argc,char **argv){
    if(argc!=3)return 2;char error[512];
    h3_av_state *state=h3_av_state_load(argv[2],error,sizeof(error));
    if(!state){fprintf(stderr,"%s\n",error);return 1;}
    double start=h3_av_now();h3_tiny_vae *decoder=h3_tiny_vae_load(argv[1],error,sizeof(error));
    double load=h3_av_now()-start;
    if(!decoder){fprintf(stderr,"%s\n",error);h3_av_state_free(state);return 1;}
    int ok=1;
    for(int repeat=0;ok&&repeat<4;repeat++){
        int frames=0;start=h3_av_now();
        ok=h3_tiny_vae_stream(decoder,state->video,state->info.video_t,
            state->info.latent_h,state->info.latent_w,5,-1,receive,&frames,NULL,NULL,error,sizeof(error));
        if(ok&&frames!=state->info.frames){snprintf(error,sizeof(error),"wrong frame count: %d",frames);ok=0;}
        if(ok)printf("{\"mode\":\"tiny\",\"repeat\":%d,\"load_seconds\":%.9f,\"decode_seconds\":%.9f,\"frames\":%d,\"model_sha256\":\"%s\"}\n",repeat,load,h3_av_now()-start,frames,h3_tiny_vae_digest(decoder));
    }
    if(!ok)fprintf(stderr,"%s\n",error);
    h3_tiny_vae_free(decoder);h3_av_state_free(state);return ok?0:1;
}
