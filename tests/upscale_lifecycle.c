/* Actual-model refinement boundaries, cancellation recovery and caller ownership.
 * Each invocation executes at most four full-depth evaluations. */
#include "src/h3.h"
#include "src/upscale/upscale.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[1024];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
typedef struct {FILE *trace;int compare,cancel,calls,first,last;const h3_sampler_state *initial;} audit;
static int latent(int step,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque) {
    audit *t=opaque;CHECK(total==4&&nv==t->initial->video_elements&&na==t->initial->audio_elements);
    CHECK(!memcmp(a,t->initial->audio,na*4));
    CHECK(!fseeko(t->trace,(off_t)((nv+na)*4*(size_t)step),SEEK_SET));
    if(t->compare){float *ref=malloc((nv+na)*4);CHECK(ref&&fread(ref,4,nv+na,t->trace)==nv+na);
        CHECK(!memcmp(ref,v,nv*4)&&!memcmp(ref+nv,a,na*4));free(ref);
    }else {CHECK(fwrite(v,4,nv,t->trace)==nv&&fwrite(a,4,na,t->trace)==na);CHECK(!fflush(t->trace));}
    if(!t->calls)t->first=step;t->calls++;t->last=step;
    fprintf(stderr,"upscale boundary %d/%d: exact audio and %s trajectory\n",step,total,t->compare?"matching":"recorded");
    return step==t->cancel;
}
int main(int argc,char **argv) {
    CHECK(argc>=2);
    if(!strcmp(argv[1],"prepare")) {
        CHECK(argc==7); /* model, source, weights, zero checkpoint, K0 AV */
        h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);
        h3_upscale_source *source=h3_upscale_source_load(argv[3],error,sizeof(error));CHECK(source);
        h3_upscale_plan *p=h3_upscale_plan_create(source,error,sizeof(error));CHECK(p);
        h3_upscale_transfer *t=h3_upscale_transfer_create(source,p,argv[4],1,NULL,NULL,error,sizeof(error));CHECK(t);
        h3_upscale_options o=H3_UPSCALE_OPTIONS_DEFAULT;o.state_only=1;o.refine_steps=0;
        h3_result *r=h3_upscale_execute(ctx,source,p,t,&o);
        if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r&&r->av_state);
        CHECK(!memcmp(r->av_state->audio,source->state->audio,source->state->audio_elements*4));
        CHECK(!memcmp(r->av_state->video,t->video,p->info.video_elements*4));
        CHECK(h3_result_save_av_state(r,argv[6],error,sizeof(error)));h3_result_free(r);
        o.refine_steps=4;o.stop_after_step=0;o.save_sampler_state=argv[5];
        r=h3_upscale_execute(ctx,source,p,t,&o);
        if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r&&r->status==H3_RESULT_PAUSED);
        CHECK(r->sampler_state->next_step==0&&h3_upscale_audio_intact(r->sampler_state));h3_result_free(r);
        /* Transferred arrays and source are still owned and readable. */
        CHECK(h3_upscale_audio_intact(source->state));
        h3_upscale_transfer_free(t);h3_upscale_plan_free(p);h3_upscale_source_free(source);h3_free(ctx);
        puts("ok: K0 clean transfer, audio identity, initialized boundary zero, caller ownership");return 0;
    }
    CHECK(!strcmp(argv[1],"run")&&argc==10);
    /* model, input checkpoint, output prefix, trace, stop, cancel, compare, expected-start */
    const char *base=argv[4];int stop=atoi(argv[6]),cancel=atoi(argv[7]),compare=atoi(argv[8]);
    h3_sampler_state *initial=h3_sampler_state_load(argv[3],error,sizeof(error));CHECK(initial&&initial->upscale.stage==2);
    audit t={.trace=fopen(argv[5],compare?"rb":"w+b"),.compare=compare,.cancel=cancel,.initial=initial};CHECK(t.trace);
    char *checkpoint=NULL,*av=NULL;CHECK(asprintf(&checkpoint,"%s.h3sample",base)>0&&asprintf(&av,"%s.h3av",base)>0);
    h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.resume_sampler_state=argv[3];p.state_only=1;p.stop_after_step=stop;
    p.save_sampler_state=checkpoint;p.on_latent_step=latent;p.callback_opaque=&t;
    h3_result *r=h3_generate(ctx,NULL,&p);
    if(cancel>=0){CHECK(!r);CHECK(strstr(h3_last_error(ctx),"latent callback stopped"));}
    else {if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r);
        if(r->status==H3_RESULT_COMPLETE){CHECK(r->av_state);CHECK(h3_result_save_av_state(r,av,error,sizeof(error)));}
        else CHECK(r->status==H3_RESULT_PAUSED);
    }
    h3_sampler_state *saved=h3_sampler_state_load(checkpoint,error,sizeof(error));CHECK(saved);
    CHECK(saved->next_step==(cancel>=0?cancel:stop>=0?stop:4)&&h3_upscale_audio_intact(saved));
    CHECK(!memcmp(saved->original_video_noise,initial->original_video_noise,initial->video_elements*4));
    CHECK(t.calls&&t.first==atoi(argv[9])&&t.last==saved->next_step);
    printf("ok: boundary %d to %d, %d callbacks, cancellation=%d, exact AV trajectory\n",t.first,t.last,t.calls,cancel);
    h3_sampler_state_free(saved);h3_result_free(r);h3_free(ctx);h3_sampler_state_free(initial);
    fclose(t.trace);free(checkpoint);free(av);return 0;
}
