/* Bounded public-API lifecycle probes. Placement never enters sampler payloads. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include "src/upscale/upscale.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[1024];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
typedef struct {float *values[3];size_t nv,na;int compare,cancel_load,cancel_step,cancelled,calls;} audit;
static int latent(int step,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque) {
    audit *t=opaque;CHECK(total==2&&step>=0&&step<=2);t->calls++;
    if(!t->compare){CHECK(!t->values[step]);t->nv=nv;t->na=na;t->values[step]=malloc((nv+na)*4);CHECK(t->values[step]);memcpy(t->values[step],v,nv*4);memcpy(t->values[step]+nv,a,na*4);}
    else CHECK(nv==t->nv&&na==t->na&&t->values[step]&&!memcmp(t->values[step],v,nv*4)&&!memcmp(t->values[step]+nv,a,na*4));
    if(t->cancel_step==step){t->cancelled=1;return 1;}return 0;
}
static int progress(const char *stage,int current,int total,void *opaque) {
    (void)total;audit *t=opaque;
    if(t->cancel_load&&!strcmp(stage,"load transformer core")&&current==1){t->cancelled=1;return 1;}return 0;
}
static void placement(const char *mode,int cap) {
    CHECK(!setenv("H3_CUDA_WEIGHT_MODE",mode,1));
    if(cap<0)unsetenv("H3_TEST_CUDA_RESIDENT_BLOCKS");
    else {char s[32];snprintf(s,sizeof(s),"%d",cap);CHECK(!setenv("H3_TEST_CUDA_RESIDENT_BLOCKS",s,1));}
    fprintf(stderr,"lifecycle placement request=%s cap=%d\n",mode,cap);
}
static h3_result *generate(h3_ctx *ctx,h3_params *p) {
    h3_result *r=h3_generate(ctx,p->resume_sampler_state?NULL:"A woman plays piano in warm sunlight with flowing piano music.",p);
    if (!r)
        snprintf(error, sizeof(error), "%s", h3_last_error(ctx));
    return r;
}
static void release_trace(audit *t){for(int i=0;i<3;i++)free(t->values[i]);memset(t,0,sizeof(*t));t->cancel_step=-1;}
static int upscale(int argc,char **argv) {
    CHECK(argc==7); /* upscale MODEL OUTPUT UPSCALER auto|stream CAP */
    placement(argv[5],atoi(argv[6]));h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    char bundle[4096],av[4096],movie[4096];CHECK(snprintf(bundle,sizeof(bundle),"%s/source.h3up",argv[3])<(int)sizeof(bundle));
    CHECK(snprintf(av,sizeof(av),"%s/final.h3av",argv[3])<(int)sizeof(av));CHECK(snprintf(movie,sizeof(movie),"%s/video.mp4",argv[3])<(int)sizeof(movie));
    h3_params p=H3_PARAMS_DEFAULT;p.width=672;p.height=384;p.frames=90;p.steps=2;p.seed=42;p.state_only=1;p.save_upscale_state=bundle;
    h3_result *r=generate(ctx,&p);CHECK(r&&r->av_state);h3_result_free(r);
    h3_upscale_source *source=h3_upscale_source_load(bundle,error,sizeof(error));CHECK(source);
    h3_upscale_plan *plan=h3_upscale_plan_create(source,error,sizeof(error));CHECK(plan);
    h3_upscale_transfer *transfer=h3_upscale_transfer_create(source,plan,argv[4],1,NULL,NULL,error,sizeof(error));CHECK(transfer);
    h3_upscale_options o=H3_UPSCALE_OPTIONS_DEFAULT;o.refine_steps=4;o.delivery.output_path=movie;
    r=h3_upscale_execute(ctx,source,plan,transfer,&o);if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));
    CHECK(r&&r->status==H3_RESULT_COMPLETE&&r->av_state&&r->width==1344&&r->height==768&&r->frames==90);
    CHECK(r->av_state->info.audio_elements==source->state->audio_elements&&!memcmp(r->av_state->audio,source->state->audio,source->state->audio_elements*4));
    CHECK(h3_result_save_av_state(r,av,error,sizeof(error)));h3_result_free(r);
    h3_upscale_transfer_free(transfer);h3_upscale_plan_free(plan);h3_upscale_source_free(source);h3_free(ctx);
    puts("ok: retained source context, target geometry replan, frozen audio, four-step refinement and full AV handoff (2+4 evaluations)");return 0;
}
int main(int argc,char **argv) {
    CHECK(argc>=2);if(!strcmp(argv[1],"upscale"))return upscale(argc,argv);
    CHECK(argc==4&&!strcmp(argv[1],"context")); /* context MODEL OUTPUT */
    h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    audit t={.cancel_step=-1};h3_params p=H3_PARAMS_DEFAULT;
    p.width=p.height=128;p.frames=22;p.steps=2;p.seed=42;
    p.on_latent_step=latent;p.on_progress=progress;p.callback_opaque=&t;
    for(int pass=0;pass<4;pass++) {
        placement(pass==0?"stream":"auto",pass==1||pass==3?2:-1);t.calls=0;
        h3_result *r=generate(ctx,&p);CHECK(r&&r->status==H3_RESULT_COMPLETE&&t.calls==3);h3_result_free(r);t.compare=1;
    }
    placement("resident",-1);p.ssd_streaming=1;
    CHECK(!generate(ctx,&p)&&strstr(error,"conflicts"));p.ssd_streaming=0;
    placement("auto",3);t.cancel_load=1;t.cancelled=0;
    CHECK(!generate(ctx,&p)&&t.cancelled);t.cancel_load=0;t.cancelled=0;
    h3_result *r=generate(ctx,&p);CHECK(r);h3_result_free(r);
    t.cancel_step=1;CHECK(!generate(ctx,&p)&&t.cancelled);t.cancel_step=-1;
    r=generate(ctx,&p);CHECK(r);h3_result_free(r);
    char checkpoint[4096];CHECK(snprintf(checkpoint,sizeof(checkpoint),"%s/paused.h3sample",argv[3])<(int)sizeof(checkpoint));
    placement("auto",2);p.stop_after_step=1;p.save_sampler_state=checkpoint;
    r=generate(ctx,&p);CHECK(r&&r->status==H3_RESULT_PAUSED);h3_result_free(r);
    h3_params resumed=H3_PARAMS_DEFAULT;resumed.resume_sampler_state=checkpoint;resumed.on_latent_step=latent;resumed.callback_opaque=&t;
    placement("stream",-1);r=generate(ctx,&resumed);CHECK(r&&r->status==H3_RESULT_COMPLETE);h3_result_free(r);
    placement("auto",2);r=generate(ctx,&resumed);CHECK(r&&r->status==H3_RESULT_COMPLETE);h3_result_free(r);
    /* Paused preview followed by another request exercises decoder headroom. */
    p.state_only=0;p.preview_on_stop=1;r=generate(ctx,&p);CHECK(r&&r->status==H3_RESULT_PAUSED&&r->frames>0);h3_result_free(r);
    /* New packed geometry in the same public context, matched to forced stream. */
    release_trace(&t);p.stop_after_step=-1;p.save_sampler_state=NULL;p.preview_on_stop=0;p.state_only=0;p.width=256;
    placement("stream",-1);r=generate(ctx,&p);CHECK(r);h3_result_free(r);t.compare=1;
    placement("auto",2);r=generate(ctx,&p);CHECK(r);h3_result_free(r);
    h3_cache_clear(ctx);h3_cache_info info;h3_cache_get_info(ctx,&info);CHECK(!info.prepared_dit&&!info.video_decoder);
    r=generate(ctx,&p);CHECK(r);h3_result_free(r);h3_free(ctx);release_trace(&t);
    puts("ok: retained placement changes, explicit conflicts, load/step cancellation recovery, resume across placements, paused preview, changed geometry and cache clear");return 0;
}
