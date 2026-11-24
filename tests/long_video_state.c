#include "src/sampling/sampler_state.h"
#include "src/sampling/bridge.h"
#include "src/sampling/av_state.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>
int main(int argc,char **argv) {
    char error[1024];
    if(argc==2) {
        h3_upscale_source *source=h3_upscale_source_load(argv[1],error,sizeof(error));assert(source);
        h3_upscale_plan *plan=h3_upscale_plan_create(source,error,sizeof(error));assert(plan);
        const h3_upscale_plan_info *info=h3_upscale_plan_get_info(plan);
        assert(info->source_width==256&&info->source_height==256&&info->width==512&&info->height==512&&info->frames==464);
        assert(info->video_t==137&&info->audio_t==773&&info->sequence_rows<=H3_CUDA_SEQUENCE_ROWS);
        h3_sigma_schedule sigmas;
        for(int k=2;k<=4;k++)assert(h3_upscale_schedule(k,.3f,&sigmas)&&sigmas.steps==k);
        h3_upscale_plan_free(plan);h3_upscale_source_free(source);
        puts("PASS extended source load and 2x temporal plan; no upscale execution");return 0;
    }
    assert(argc==3);
    h3_sampler_state *s=h3_sampler_state_load(argv[1],error,sizeof(error));
    if(!s){fprintf(stderr,"%s\n",error);return 1;}
    assert(s->render_width==256&&s->render_height==256&&s->aligned_frames==464&&s->context_frames==447);
    assert(s->next_step==s->total_steps&&s->continuation);
    h3_av_state *source=h3_av_state_load(argv[2],error,sizeof(error));assert(source);
    assert(source->info.render_width==256&&source->info.render_height==256&&source->info.frames==447);
    h3_av_state_info target;assert(h3_av_state_shape(256,256,464,&target));
    assert(s->original_video_noise&&s->original_audio_noise);
    assert(s->layout.bridge?h3_av_state_insert_bridge(source,&target,s->layout.bridge,s->sigmas.video[0],s->sigmas.audio[0],s->original_video_noise,s->original_audio_noise):
        h3_av_state_insert_prefix(source,&target,s->layout.prefix,s->original_video_noise,s->original_audio_noise,1));
    if(s->layout.bridge) {
        assert(h3_bridge_check_exact(s->layout.bridge,s->latent_t,s->latent_h,s->latent_w,s->audio_t,
            s->video,s->audio,s->original_video_noise,s->original_audio_noise));
    } else {
        size_t hw=(size_t)s->latent_h*s->latent_w;
        for(int c=0;c<24;c++)assert(!memcmp(s->video+c*(size_t)s->latent_t*hw,s->original_video_noise+c*(size_t)s->latent_t*hw,(size_t)s->layout.prefix.video_prefix_t*hw*4));
        for(int c=0;c<64;c++)assert(!memcmp(s->audio+c*(size_t)s->audio_t,s->original_audio_noise+c*(size_t)s->audio_t,(size_t)s->layout.prefix.audio_prefix_t*4));
    }
    h3_av_state_free(source);h3_sampler_state_free(s);puts("PASS exact protected continuation prefix after extended denoising");return 0;
}
