/* Load real downloaded server artifacts with the ordinary native validators.
 * CPU only: no model loading and no tensor allocation on the GPU. */
#include "src/h3.h"
#include "src/media/delivery.h"
#include "src/conditioning/conditioning.h"
#include "src/vae/image_vae.h"
#include <stdio.h>
#include <string.h>

static int suffix(const char *path,const char *end) {
    size_t n=strlen(path),m=strlen(end);
    return n>=m&&!strcmp(path+n-m,end);
}
int main(int argc,char **argv) {
    if(argc<2){fprintf(stderr,"usage: %s STATE...\n",argv[0]);return 2;}
    for(int i=1;i<argc;i++) {
        char error[1024]={0};int ok=0;const char *path=argv[i];
        if(suffix(path,".h3av")) {
            h3_av_state *state=h3_av_state_load(path,error,sizeof(error));
            if(state){h3_presentation p={0};ok=h3_presentation_load(path,state,&p,error,sizeof(error))==1;}
            h3_av_state_free(state);
        }else if(suffix(path,".h3sample")) {
            h3_sampler_state *state=h3_sampler_state_load(path,error,sizeof(error));
            ok=state!=NULL;h3_sampler_state_free(state);
        }else if(suffix(path,".h3up")) {
            h3_upscale_source *state=h3_upscale_source_load(path,error,sizeof(error));
            ok=state!=NULL;h3_upscale_source_free(state);
        }else if(suffix(path,".h3cond")) {
            h3_conditioning *state=h3_conditioning_load(path,error,sizeof(error));
            ok=state!=NULL;h3_conditioning_free(state);
        }else if(suffix(path,".safetensors")) {
            h3_still_latent state={0};ok=h3_still_latent_load(path,&state,error,sizeof(error));
            h3_still_latent_free(&state);
        }
        if(!ok){fprintf(stderr,"%s: %s\n",path,*error?error:"missing sidecar or unsupported artifact");return 1;}
        printf("PASS %s\n",path);
    }
    return 0;
}
