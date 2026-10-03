/* Mutable M1 integration tests; deliberately outside the frozen parity suite. */
#define main old_isolation_main
#include "cuda_sglang_isolation.c"
#undef main
#include "src/sampling/av_state.h"
static void crop_state(const char *input,const char *output) {
    h3_av_state *source=h3_av_state_load(input,error,sizeof(error));CHECK(source);
    h3_av_state *small=h3_av_state_new(64,64,22,source->info.seed,source->info.compatibility);CHECK(small);
    CHECK(source->info.video_t>=small->info.video_t&&source->info.latent_h>=4&&source->info.latent_w>=4&&source->info.audio_t>=small->info.audio_t);
    for(int c=0;c<24;c++)for(int t=0;t<small->info.video_t;t++)for(int y=0;y<4;y++)for(int x=0;x<4;x++)
        small->video[((c*small->info.video_t+t)*4+y)*4+x]=source->video[((c*source->info.video_t+t)*source->info.latent_h+y)*source->info.latent_w+x];
    for(int c=0;c<64;c++)for(int t=0;t<small->info.audio_t;t++)small->audio[c*small->info.audio_t+t]=source->audio[c*source->info.audio_t+t];
    h3_result r={.av_state=small,.presentation={.version=9,.av_metadata_identity=1,.ref2va=0,.render_width=64,.render_height=64,.width=64,.height=64,.fps=24,.sample_rate=32000,.codec_version=2}};
    CHECK(h3_result_save_av_state(&r,output,error,sizeof(error)));h3_av_state_free(small);h3_av_state_free(source);
}
int main(int argc,char **argv) {
    CHECK(argc==1||argc==3);
    if(argc==3){crop_state(argv[1],argv[2]);return 0;}

    h3_params params=H3_PARAMS_DEFAULT;h3_cuda_policy reference,fast;
    CHECK(h3_cuda_policy_resolve(&params,"cuda",0,&reference,error,sizeof(error)));
    params.cuda_attention=1;CHECK(h3_cuda_policy_resolve(&params,"cuda",0,&fast,error,sizeof(error)));
    CHECK(h3_cuda_policy_preflight(&reference,error,sizeof(error)));
    h3_cuda_policy old=h3_cuda_policy_exchange(reference);uint16_t *a=evaluate(1);
    h3_cuda_policy_exchange(fast);uint16_t *b=evaluate(1);CHECK(!memcmp(a,b,E*2));free(b);
    h3_cuda_policy_exchange(reference);b=evaluate(1);CHECK(!memcmp(a,b,E*2));free(b);free(a);
    h3_cuda_policy_exchange(old);CHECK(!h3_cuda_policy_current().active);
    puts("PASS: captured dense/optional/dense scopes, exact shared operators, allocation plateau and cancellation");return 0;
}
