/* Two complete six-step videos; a three-step checkpoint is diagnostic only. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,h3_last_error(ctx));return 1;}}while(0)
int main(int argc,char **argv){
    if(argc!=3)return 2;h3_ctx *ctx=h3_load_dir(argv[1]);CHECK(ctx);h3_cache_set_enabled(ctx,1);mkdir(argv[2],0755);
    char reference[4096],resumed[4096],checkpoint[4096],state[4096],error[512];
    snprintf(reference,sizeof(reference),"%s/reference.mp4",argv[2]);snprintf(resumed,sizeof(resumed),"%s/resumed.mp4",argv[2]);snprintf(checkpoint,sizeof(checkpoint),"%s/diagnostic-step3.h3sample",argv[2]);
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=64;p.frames=22;p.steps=6;p.seed=771;
    p.output_path=reference;
    const char *prompt="A grand piano in a sunlit room, slow camera movement.";
    h3_result *baseline=h3_generate(ctx,prompt,&p);CHECK(baseline&&baseline->status==H3_RESULT_COMPLETE&&baseline->completed_steps==6);
    snprintf(state,sizeof(state),"%s/reference.h3av",argv[2]);CHECK(h3_result_save_av_state(baseline,state,error,sizeof(error)));
    p.output_path=NULL;p.stop_after_step=3;p.save_sampler_state=checkpoint;
    CHECK(!setenv("H3_FULL_VAE_CUDA_GRAPH","0",1));
    h3_result *paused=h3_generate(ctx,prompt,&p);CHECK(paused&&paused->status==H3_RESULT_PAUSED&&paused->completed_steps==3);h3_result_free(paused);
    /* Resume restores the saved shared CUDA recipe. */
    p.stop_after_step=-1;p.save_sampler_state=NULL;p.resume_sampler_state=checkpoint;p.output_path=resumed;
    CHECK(!setenv("H3_FULL_VAE_CUDA_GRAPH","1",1));
    h3_result *result=h3_generate(ctx,prompt,&p);CHECK(result&&result->status==H3_RESULT_COMPLETE&&result->completed_steps==6);
    CHECK(baseline->av_state->info.video_elements==result->av_state->info.video_elements&&baseline->av_state->info.audio_elements==result->av_state->info.audio_elements);
    CHECK(!memcmp(baseline->av_state->video,result->av_state->video,result->av_state->info.video_elements*4));
    CHECK(!memcmp(baseline->av_state->audio,result->av_state->audio,result->av_state->info.audio_elements*4));
    snprintf(state,sizeof(state),"%s/resumed.h3av",argv[2]);CHECK(h3_result_save_av_state(result,state,error,sizeof(error)));
    printf("{\"completed_steps\":6,\"reference_steps\":6,\"diagnostic_checkpoint_steps\":3,\"resumed_total_steps\":6,\"latent_invariant\":true,\"frames\":%d}\n",result->frames);
    h3_result_free(result);h3_result_free(baseline);h3_free(ctx);unsetenv("H3_FULL_VAE_CUDA_GRAPH");return 0;
}
