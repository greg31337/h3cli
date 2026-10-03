/* Fixed-manifest runner's native adapter. Python orchestrates files/telemetry;
 * every network, reference transform, sampler and decoder remains native. */
#include "src/h3.h"
#include "src/upscale/upscale.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/vae/audio_vae.h"
#include "src/media/delivery.h"
#include "src/sglang/sglang.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[1024];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
static char *path(const char *d,const char *n){char *p=NULL;CHECK(asprintf(&p,"%s/%s",d,n)>0);return p;}
typedef struct {char phase[80];double began;const h3_sampler_state *state;int callbacks;} audit;
static void phase_end(audit *a) {
    if(*a->phase)fprintf(stderr,"h3_upscale_phase {\"phase\":\"%s\",\"seconds\":%.9g}\n",a->phase,h3_av_now()-a->began);
    a->phase[0]=0;
}
static int progress(const char *phase,int done,int total,void *opaque) {
    (void)done;(void)total;audit *a=opaque;
    if(strcmp(phase,a->phase)){phase_end(a);snprintf(a->phase,sizeof(a->phase),"%s",phase);a->began=h3_av_now();}return 0;
}
static int latent(int step,int total,const float *v,size_t nv,const float *audio,size_t na,void *opaque) {
    (void)step;(void)total;(void)v;(void)nv;audit *a=opaque;a->callbacks++;
    if(a->state)CHECK(na==a->state->audio_elements&&!memcmp(audio,a->state->audio,na*4));return 0;
}
static void save(h3_result *r,const char *d) {
    char *p=path(d,"final.h3av");CHECK(h3_result_save_av_state(r,p,error,sizeof(error)));free(p);
}
static void fromhex(const char *hex,uint8_t hash[32]) {
    CHECK(strlen(hex)==64);for(int i=0;i<32;i++){unsigned byte;CHECK(sscanf(hex+2*i,"%2x",&byte)==1);hash[i]=(uint8_t)byte;}
}
int main(int argc,char **argv) {
    CHECK(argc>=2);const char *mode=argv[1];audit a={0};
    if(!strcmp(mode,"generate")) {
        CHECK(argc==8); /* model out width height source|direct prompt */
        int source=!strcmp(argv[6],"source");CHECK(source||!strcmp(argv[6],"direct"));
        CHECK(getenv("H3_TEST_MAX_EVALUATIONS")&&!strcmp(getenv("H3_TEST_MAX_EVALUATIONS"),"50"));
        h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);const char *d=argv[3];
        h3_params p=H3_PARAMS_DEFAULT;p.width=atoi(argv[4]);p.height=atoi(argv[5]);p.frames=90;p.steps=50;p.seed=42;
        CHECK(source?((p.width==672&&p.height==384)||(p.width==960&&p.height==544)):
            ((p.width==1344&&p.height==768)||(p.width==1920&&p.height==1088)));
        char *bundle=path(d,"source.h3up"),*video=path(d,"video.mp4");
        p.save_upscale_state=source?bundle:NULL;p.geometry_profile=source?0:1;p.output_path=video;
        p.on_progress=progress;p.on_latent_step=latent;p.callback_opaque=&a;
        h3_result *r=h3_generate(ctx,argv[7],&p);phase_end(&a);
        if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r&&a.callbacks==51);
        save(r,d);h3_result_free(r);h3_free(ctx);free(bundle);free(video);return 0;
    }
    if(!strcmp(mode,"transfer")) {
        CHECK(argc==7); /* model source weights output recipe */
        h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);
        h3_upscale_source *s=h3_upscale_source_load(argv[3],error,sizeof(error));CHECK(s&&!s->state->conditioned);
        h3_upscale_plan *p=h3_upscale_plan_create(s,error,sizeof(error));CHECK(p);int recipe=atoi(argv[6]);
        h3_upscale_transfer *t=h3_upscale_transfer_create(s,p,argv[4],recipe,progress,&a,error,sizeof(error));phase_end(&a);CHECK(t);
        h3_upscale_options o=H3_UPSCALE_OPTIONS_DEFAULT;o.refine_steps=0;o.state_only=1;
        h3_result *r=h3_upscale_execute(ctx,s,p,t,&o);if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r);save(r,argv[5]);
        h3_result_free(r);h3_upscale_transfer_free(t);h3_upscale_plan_free(p);h3_upscale_source_free(s);h3_free(ctx);return 0;
    }
    if(!strcmp(mode,"initialize")) {
        CHECK(argc==8); /* model source transfer AV shared-noise-sampler|- output K */
        h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);h3_upscale_source *s=h3_upscale_source_load(argv[3],error,sizeof(error));CHECK(s&&!s->state->conditioned);
        h3_upscale_plan *p=h3_upscale_plan_create(s,error,sizeof(error));CHECK(p);
        h3_av_state *av=h3_av_state_load(argv[4],error,sizeof(error));CHECK(av);h3_presentation pr;
        CHECK(h3_presentation_load(argv[4],av,&pr,error,sizeof(error))==1&&pr.version==7&&pr.upscale_steps==0);
        CHECK(av->info.render_width==p->info.width&&av->info.render_height==p->info.height&&av->info.frames==90);
        CHECK(!memcmp(av->audio,s->state->audio,s->state->audio_elements*4));
        h3_upscale_transfer *t=calloc(1,sizeof(*t));CHECK(t);t->recipe=pr.upscale_recipe;
        t->video=av->video;av->video=NULL;fromhex(pr.upscale_parent_sha256,t->parent_hash);fromhex(pr.upscale_artifact_sha256,t->artifact_hash);
        CHECK(!memcmp(t->parent_hash,s->state->loaded_hash,32));
        CHECK(h3_upscale_retarget(s,p,NULL,&t->conditions,NULL,NULL,error,sizeof(error)));
        h3_sampler_state *noise=strcmp(argv[5],"-")?h3_sampler_state_load(argv[5],error,sizeof(error)):NULL;
        CHECK(!strcmp(argv[5],"-")||noise);t->noise_source=noise;
        h3_upscale_options o=H3_UPSCALE_OPTIONS_DEFAULT;o.refine_steps=atoi(argv[7]);o.state_only=1;o.stop_after_step=0;
        char *checkpoint=path(argv[6],"initial.h3sample");o.save_sampler_state=checkpoint;
        h3_result *r=h3_upscale_execute(ctx,s,p,t,&o);if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r&&r->sampler_state);
        if(!noise){double begin=h3_av_now();char *name=path(argv[6],"shared-noise.f32");FILE *f=fopen(name,"wbx");CHECK(f);
            CHECK(fwrite(r->sampler_state->original_video_noise,4,r->sampler_state->video_elements,f)==r->sampler_state->video_elements&&!fclose(f));free(name);
            fprintf(stderr,"h3_upscale_noise_persist {\"seconds\":%.9g}\n",h3_av_now()-begin);}
        else CHECK(!memcmp(r->sampler_state->original_video_noise,noise->original_video_noise,noise->video_elements*4));
        fprintf(stderr,"h3_upscale_noise {\"new_draws\":%zu,\"audio_draws\":0}\n",noise?0:r->sampler_state->video_elements);
        free(checkpoint);h3_result_free(r);h3_sampler_state_free(noise);h3_upscale_transfer_free(t);h3_av_state_free(av);h3_upscale_plan_free(p);h3_upscale_source_free(s);h3_free(ctx);return 0;
    }
    if(!strcmp(mode,"render")) {
        CHECK(argc==6); /* model input output av|sampler */
        const char *d=argv[4];char *video=path(d,"video.mp4");h3_result *r=NULL;h3_ctx *ctx=NULL;
        if(!strcmp(argv[5],"av")) {
            h3_decode_options o={.output_path=video,.on_progress=progress,.callback_opaque=&a};
            r=h3_decode_av_state(argv[2],argv[3],&o,error,sizeof(error));
        }else {CHECK(!strcmp(argv[5],"sampler"));a.state=h3_sampler_state_load(argv[3],error,sizeof(error));CHECK(a.state&&a.state->upscale.stage==2&&!a.state->next_step);
            ctx=h3_load_dir(argv[2]);CHECK(ctx);h3_params p=H3_PARAMS_DEFAULT;p.resume_sampler_state=argv[3];p.output_path=video;
            p.on_progress=progress;p.on_latent_step=latent;p.callback_opaque=&a;r=h3_generate(ctx,NULL,&p);
            if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r&&a.callbacks==a.state->total_steps+1);
        }
        phase_end(&a);CHECK(r);save(r,d);h3_result_free(r);h3_sampler_state_free((h3_sampler_state *)a.state);h3_free(ctx);free(video);return 0;
    }
    CHECK(!strcmp(mode,"pcm")&&argc==5); /* model av output */
    h3_av_state *s=h3_av_state_load(argv[3],error,sizeof(error));CHECK(s);h3_presentation p;
    CHECK(h3_presentation_load(argv[3],s,&p,error,sizeof(error))==1);h3_sglang_exchange(p.codec_version==2?H3_SGLANG_VERSION:0);
    char *weights=NULL;CHECK(asprintf(&weights,"%s/%s/audio_vae",argv[2],p.ref2va?"Ref2VA":"FL2VA")>0);h3_audio_waveform out={0};
    CHECK(h3_audio_vae_decode(weights,"src/metal/shaders.metal",s->audio,s->info.audio_t,NULL,NULL,&out,error,sizeof(error)));
    CHECK(out.channels==2&&out.sample_rate==32000&&out.samples==120000);size_t n=(size_t)out.channels*out.samples;
    for(size_t i=0;i<n;i++)CHECK(isfinite(out.pcm[i]));FILE *f=fopen(argv[4],"wbx");CHECK(f&&fwrite(out.pcm,4,n,f)==n&&!fclose(f));
    h3_audio_waveform_free(&out);h3_av_state_free(s);free(weights);return 0;
}
