/* Denoser-free fixtures: spatial/time crops, repeat-tail memory stress, or
 * original reference-video encoder outputs. Provenance lives beside each run. */
#include "src/sampling/av_state.h"
#include "src/vae/video_encoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void die(const char *s){fprintf(stderr,"%s\n",s);exit(1);}
int main(int argc,char **argv){
    if(argc!=8)die("usage: fast_vae_fixture crop|encode SOURCE FRAMES HEIGHT WIDTH OUTPUT WEIGHTS");
    int frames=atoi(argv[3]),h=atoi(argv[4]),w=atoi(argv[5]);char error[512];uint8_t sig[32]={0};
    h3_av_state *out=NULL;
    if(!strcmp(argv[1],"crop")){
        h3_av_state *s=h3_av_state_load(argv[2],error,sizeof(error));if(!s)die(error);
        out=h3_av_state_new(w,h,frames,s->info.seed,s->info.compatibility);if(!out||out->info.latent_h>s->info.latent_h||out->info.latent_w>s->info.latent_w)die("invalid crop");
        for(int c=0;c<24;c++)for(int t=0;t<out->info.video_t;t++)for(int y=0;y<out->info.latent_h;y++){
            int from=t<s->info.video_t?t:s->info.video_t-1;
            memcpy(out->video+(((size_t)c*out->info.video_t+t)*out->info.latent_h+y)*out->info.latent_w,
                s->video+(((size_t)c*s->info.video_t+from)*s->info.latent_h+y)*s->info.latent_w,(size_t)out->info.latent_w*4);
        }
        memset(out->audio,0,out->info.audio_elements*4);h3_av_state_free(s);
    }else if(!strcmp(argv[1],"encode")){
        size_t n=(size_t)3*frames*h*w;float *pixels=malloc(n*4);FILE *f=fopen(argv[2],"rb");if(!f||!pixels||fread(pixels,4,n,f)!=n||fgetc(f)!=EOF)die("bad RGB input");fclose(f);
        h3_video_latent z={0};if(!h3_ref2va_video_vae_encode(argv[7],"src/metal/shaders.metal",pixels,frames,frames,h,w,NULL,NULL,&z,error,sizeof(error)))die(error);
        out=h3_av_state_new(w,h,frames,42,sig);if(!out||out->info.video_t!=z.time||out->info.latent_h!=z.height||out->info.latent_w!=z.width)die("encoded shape mismatch");
        memcpy(out->video,z.values,out->info.video_elements*4);memset(out->audio,0,out->info.audio_elements*4);h3_video_latent_free(&z);free(pixels);
    }else die("invalid action");
    if(!h3_av_state_save(out,argv[6],error,sizeof(error)))die(error);
    printf("{\"kind\":\"%s\",\"source\":\"%s\",\"frames\":%d,\"height\":%d,\"width\":%d,\"denoising_calls\":0}\n",argv[1],argv[2],frames,h,w);
    h3_av_state_free(out);return 0;
}
