/* Audit saved continuation prefixes against the existing seeded augmentation
 * contract. This does not run a model or repeat a rendered test case. */
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

int main(int argc,char **argv) {
    if(argc!=4)return 2;
    char error[512]={0};
    h3_av_state *parent=h3_av_state_load(argv[1],error,sizeof(error));
    h3_av_state *state[2]={h3_av_state_load(argv[2],error,sizeof(error)),
                           h3_av_state_load(argv[3],error,sizeof(error))};
    if(!parent||!state[0]||!state[1]){fprintf(stderr,"%s\n",error);return 2;}
    h3_denoise_prefix prefix;
    if(!h3_continuation_context(39,&prefix))return 2;
    int passed=1;
    puts("{");
    for(int mode=0;mode<2;mode++) {
        const h3_av_state *s=state[mode];
        if(s->info.latent_h!=parent->info.latent_h||s->info.latent_w!=parent->info.latent_w||
           s->info.seed!=4343||s->info.video_t<prefix.video_prefix_t||
           parent->info.video_t<prefix.video_prefix_t||s->info.audio_t<prefix.audio_prefix_t||
           parent->info.audio_t<prefix.audio_prefix_t)return 2;
        float *noise=malloc(s->info.video_elements*sizeof(float));
        if(!noise)return 2;
        h3_rng rng;h3_rng_seed(&rng,s->info.seed);
        h3_rng_fill_normal(&rng,noise,s->info.video_elements);
        size_t hw=(size_t)s->info.latent_h*(size_t)s->info.latent_w;
        size_t video_mismatch=0,audio_mismatch=0;
        for(size_t c=0;c<24;c++)for(size_t i=0;i<(size_t)prefix.video_prefix_t*hw;i++) {
            size_t dst=c*(size_t)s->info.video_t*hw+i;
            size_t src=(c*(size_t)parent->info.video_t+parent->info.video_t-prefix.video_prefix_t)*hw+i;
            volatile float clean=0.999f*parent->video[src],random=0.001f*noise[dst];
            float expected=clean+random;
            video_mismatch+=(memcmp(&expected,s->video+dst,sizeof(float))!=0);
        }
        for(size_t c=0;c<64;c++)for(size_t i=0;i<(size_t)prefix.audio_prefix_t;i++) {
            size_t dst=c*(size_t)s->info.audio_t+i;
            size_t src=c*(size_t)parent->info.audio_t+parent->info.audio_t-prefix.audio_prefix_t+i;
            audio_mismatch+=(memcmp(parent->audio+src,s->audio+dst,sizeof(float))!=0);
        }
        printf("  \"%s\": {\"video_prefix_exact\": %s, \"audio_prefix_exact\": %s, "
               "\"video_mismatches\": %zu, \"audio_mismatches\": %zu, \"seed\": 4343, "
               "\"contract\": \"F32(0.999 * parent + 0.001 * seeded_noise), separate products\"}%s\n",
               mode?"sol":"default",video_mismatch?"false":"true",audio_mismatch?"false":"true",
               video_mismatch,audio_mismatch,mode?"":",");
        passed&=!video_mismatch&&!audio_mismatch;free(noise);
    }
    puts("}");
    for(int i=0;i<2;i++)h3_av_state_free(state[i]);h3_av_state_free(parent);
    return passed?0:1;
}
