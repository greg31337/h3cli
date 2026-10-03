/* Full M5 SOL B5 with prepared caching, zero-step pause/resume, and production
 * VAE teardown assertions. Fixed user reproduction: 640x480, 362 frames.
 * Usage: MODEL CONDITIONING OUTPUT_DIR --steps 5 --frames 362 --width 640 --height 480 */
#include "src/internal.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static h3_ctx *context;
static int vae_seen,violation;
static int progress(const char *phase,int completed,int total,void *opaque) {
    (void)opaque;
    if(!strcmp(phase,"video VAE load")) {
        vae_seen=1;
        if(context->dit){fprintf(stderr,"FAIL: transformer overlaps production VAE\n");violation=1;return 1;}
    }
    if(completed==0||completed==total)fprintf(stderr,"cache regression: %s %d/%d\n",phase,completed,total);
    return 0;
}
int main(int argc,char **argv) {
    const char *required[]={"--steps","5","--frames","362","--width","640","--height","480"};
    if(argc!=12)return 2;
    for(int i=0;i<8;i++)if(strcmp(argv[i+4],required[i]))return 2;
    char pause[4096],output[4096],state[4096],error[512]={0};
    if(snprintf(pause,sizeof(pause),"%s/pause.h3sample",argv[3])>=(int)sizeof(pause)||
       snprintf(output,sizeof(output),"%s/result.mp4",argv[3])>=(int)sizeof(output)||
       snprintf(state,sizeof(state),"%s/result.h3av",argv[3])>=(int)sizeof(state))return 2;
    const char *prompt="The person in <Picture 1> walks through a sunlit park wearing a blue jacket. A steady camera follows the walk. Natural birdsong and soft footsteps.";
    h3_reference ref={H3_REFERENCE_IMAGE,"inputs/2.jpg",NULL,0};
    h3_params p=H3_PARAMS_DEFAULT;
    p.width=640;p.height=480;p.frames=362;p.steps=5;p.seed=12001;
    p.references=&ref;p.reference_count=1;p.reference_image_size=H3_REFERENCE_IMAGE_MAX;
    p.backend=H3_BACKEND_METAL;p.attention_mode=H3_ATTN_SOL;p.backend_set=15;
    p.metal_attention.candidate=1;p.metal_attention.precision=1;
    p.metal_attention.q_block=64;p.metal_attention.kv_block=64;p.metal_attention.min_exact=.75f;
    p.metal_attention.dense_steps=1;p.use_slower_bf16_qkv=1;
    p.output_path=output;p.load_conditioning=argv[2];p.on_progress=progress;
    p.stop_after_step=0;p.save_sampler_state=pause;
    context=h3_load_dir(argv[1]);h3_result *r=NULL;int result=1;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s: %s %s\n",__LINE__,#x,error,h3_last_error(context));goto cleanup;}}while(0)
    CHECK(context);h3_cache_set_enabled(context,1);
    r=h3_generate(context,prompt,&p);CHECK(r&&r->status==H3_RESULT_PAUSED&&context->dit&&!vae_seen);
    struct h3_dit *prepared=context->dit;
    h3_result_free(r);r=NULL;
    p.stop_after_step=-1;p.save_sampler_state=NULL;p.load_conditioning=NULL;p.resume_sampler_state=pause;
    CHECK(context->dit==prepared);
    r=h3_generate(context,NULL,&p);CHECK(r&&r->status==H3_RESULT_COMPLETE&&vae_seen&&!violation&&!context->dit);
    CHECK(h3_result_save_av_state(r,state,error,sizeof(error)));
    puts("PASS: 362-frame SOL B5, prepared cache retained at pause, resumed to completion, DiT absent at every production VAE load callback; AV state saved");result=0;
cleanup:
    h3_result_free(r);h3_free(context);return result;
}
