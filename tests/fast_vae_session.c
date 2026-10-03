/* Every video generation here completes exactly six denoising evaluations. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
static void die(const char *s){fprintf(stderr,"%s\n",s);exit(1);}
typedef struct {int steps;} trace;
static int step(int done,int total,const float *v,size_t nv,const float *a,size_t na,void *p){(void)v;(void)nv;(void)a;(void)na;if(total!=6)die("six-step contract failed");((trace*)p)->steps=done;return 0;}
int main(int argc,char **argv){
    if(argc!=3)return 2;mkdir(argv[2],0755);char error[512];h3_ctx *ctx=h3_load_dir(argv[1]);if(!ctx)die(h3_last_error(NULL));h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=64;p.frames=22;p.steps=6;
    p.seed=723;p.on_latent_step=step;
    h3_av_state *oracle=NULL,*shape=NULL;const char *names[]={"reference","repeat","preview","shape-change","shape-repeat"};
    for(int i=0;i<5;i++){
        trace t={0};p.callback_opaque=&t;p.preview_vae=i==2;p.preview_vae_model=i==2?(getenv("H3_TEST_PREVIEW_VAE_MODEL")?getenv("H3_TEST_PREVIEW_VAE_MODEL"):"taeh3.safetensors"):NULL;
        if(i==3){p.width=96;p.frames=39;}
        char output[4096],state[4096];snprintf(output,sizeof(output),"%s/%s.mp4",argv[2],names[i]);snprintf(state,sizeof(state),"%s/%s.h3av",argv[2],names[i]);p.output_path=output;
        double start=h3_av_now();h3_result *r=h3_generate(ctx,"A grand piano in a sunlit room, slow camera movement.",&p);if(!r)die(h3_last_error(ctx));
        if(t.steps!=6||r->status!=H3_RESULT_COMPLETE)die("generation did not complete six steps");
        if(!i)oracle=h3_av_state_clone(r->av_state);
        else if(i<3 && (memcmp(oracle->video,r->av_state->video,oracle->info.video_elements*4)||memcmp(oracle->audio,r->av_state->audio,oracle->info.audio_elements*4)))die("decoder selection changed denoised latents");
        if(i==3)shape=h3_av_state_clone(r->av_state);
        if(i==4 && (!shape||memcmp(shape->video,r->av_state->video,shape->info.video_elements*4)||memcmp(shape->audio,r->av_state->audio,shape->info.audio_elements*4)))die("cached same-shape repeat changed denoised latents");
        if(!h3_result_save_av_state(r,state,error,sizeof(error)))die(error);
        printf("{\"mode\":\"%s\",\"completed_steps\":%d,\"frames\":%d,\"seconds\":%.9f,\"latent_invariant\":%s}\n",names[i],t.steps,r->frames,h3_av_now()-start,i==3?"null":"true");fflush(stdout);h3_result_free(r);
    }
    h3_av_state_free(shape);h3_av_state_free(oracle);h3_free(ctx);return 0;
}
