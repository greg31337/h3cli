/* Decode completed approximate latents with a build lacking SubBlock support.
 * Delivers RGB to a checksum callback; no additional video is generated. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include "src/digest.h"
#include <stdio.h>
#include <string.h>
typedef struct {int frames;h3_sha256_ctx digest;} capture;
static int frame(const h3_frame *f,void *opaque) {
    capture *c=opaque;
    if(f->frame_index!=c->frames||f->denoise_step!=-1||f->width!=640||f->height!=480)return 1;
    h3_sha256_update(&c->digest,f->rgb,(h3_sha256_size)((size_t)f->stride*f->height));c->frames++;return 0;
}
int main(int argc,char **argv) {
    if(argc!=4)return 2;
    char error[512];h3_sampler_state *s=h3_sampler_state_load(argv[2],error,sizeof(error));
    if(!s||s->next_step!=50){fprintf(stderr,"%s\n",error);h3_sampler_state_free(s);return 1;}
    h3_av_state *av=h3_av_state_new(s->render_width,s->render_height,s->aligned_frames,s->params.seed,s->av_signature);
    if(!av){h3_sampler_state_free(s);return 1;}
    memcpy(av->video,s->video,s->video_elements*4);memcpy(av->audio,s->audio,s->audio_elements*4);
    h3_result saved={.status=H3_RESULT_COMPLETE,.av_state=av};
    saved.presentation=(h3_presentation){.version=5,.ref2va=s->ref2va,.render_width=s->render_width,.render_height=s->render_height,
        .width=s->params.width,.height=s->params.height,.fps=24,.sample_rate=32000,.codec_version=2,
        .cuda_attention=s->params.cuda_attention,.adaptive_cache=s->params.adaptive_cache,.adaptive_version=s->adaptive_version,
        .attention_version=s->attention_version,.attention_plan=s->attention_plan,
        .subblock_sparsity=s->params.subblock_sparsity};
    int ok=h3_result_save_av_state(&saved,argv[3],error,sizeof(error));h3_av_state_free(av);h3_sampler_state_free(s);
    if(!ok){fprintf(stderr,"%s\n",error);return 1;}
    capture c={0};h3_sha256_init(&c.digest);
    h3_decode_options options={.on_frame=frame,.callback_opaque=&c};
    h3_result *r=h3_decode_av_state(argv[1],argv[3],&options,error,sizeof(error));
    ok=r&&r->frames==c.frames&&c.frames==56&&r->sample_rate==32000&&r->audio_samples>0;
    if(!ok)fprintf(stderr,"%s\n",error);
    uint8_t digest[32];h3_sha256_final(digest,&c.digest);
    if(ok){printf("PASS clean AV decode without optional attention: frames=%d rgb_sha256=",c.frames);for(int i=0;i<32;i++)printf("%02x",digest[i]);puts("");}
    h3_result_free(r);return !ok;
}
