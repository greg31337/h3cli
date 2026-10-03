/* Public API/state/callback coverage; short functional clips are not quality evidence. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include "src/weights/quant.h"
#include "src/gpu.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
static char error[512];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
typedef struct {
    int frames,previews,latent_calls,cancel,cancel_after,cancel_step,reclaim,mode,reclaimed;
    const char *cancel_stage;double first_latent,last_latent;
    h3_ctx *ctx;h3_gpu *ballast_gpu;h3_gpu_tensor *ballast;
} audit;
static int latent(int step,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque){
    audit *s=opaque;CHECK(step<=total);if(!s->latent_calls)s->first_latent=h3_av_now();s->last_latent=h3_av_now();s->latent_calls++;
    for(size_t i=0;i<nv;i++)CHECK(isfinite(v[i]));for(size_t i=0;i<na;i++)CHECK(isfinite(a[i]));
    if(s->cancel_step&&step==s->cancel_step){s->cancel++;return 1;}
    if(s->reclaim&&step==total){
        s->ballast_gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(s->ballast_gpu);
        size_t bytes=(size_t)(s->mode==H3_QUANT_NVFP4?12:4)<<30;
        s->ballast=h3_gpu_tensor_alloc(s->ballast_gpu,bytes,H3_GPU_I8,H3_GPU_DEVICE_ONLY);CHECK(s->ballast);
    }
    return 0;
}
static int frame(const h3_frame *f,void *opaque){audit *s=opaque;CHECK(f->rgb);if(f->denoise_step>=0)s->previews++;else s->frames++;return 0;}
static int progress(const char *stage,int completed,int total,void *opaque){
    (void)total;audit *s=opaque;
    if(s->ballast&&!strcmp(stage,"audio VAE")&&!completed){
        h3_cache_info info;h3_cache_get_info(s->ctx,&info);CHECK(!info.prepared_dit);s->reclaimed++;
        h3_gpu_tensor_free(s->ballast);s->ballast=NULL;h3_gpu_free(s->ballast_gpu);s->ballast_gpu=NULL;
    }
    if(s->cancel_stage&&!strcmp(stage,s->cancel_stage)&&completed>=s->cancel_after){s->cancel++;return 1;}return 0;
}
static h3_result *run(h3_ctx *ctx,h3_params *p,audit *s,const char *dir,const char *name){
    char output[1024],state[1024];snprintf(output,sizeof(output),"%s/%s.mp4",dir,name);snprintf(state,sizeof(state),"%s/%s.h3av",dir,name);
    p->output_path=output;s->frames=s->previews=s->latent_calls=0;
    double start=h3_av_now();h3_result *r=h3_generate(ctx,"A person smiles gently in a quiet studio. Soft daylight, fixed camera.",p);
    if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r);
    if(r->status==H3_RESULT_COMPLETE){CHECK(r->av_state&&s->frames==r->frames);CHECK(h3_result_save_av_state(r,state,error,sizeof(error)));}
    CHECK(r->presentation.cuda_denoise_quant==p->cuda_denoise_quant);
    printf("PASS %s: %.6fs denoise=%.6fs frames=%d previews=%d latent_callbacks=%d mode=%d\n",name,h3_av_now()-start,s->last_latent-s->first_latent,r->frames,s->previews,s->latent_calls,p->cuda_denoise_quant);fflush(stdout);p->output_path=NULL;return r;
}
static void same(const h3_av_state *a,const h3_av_state *b){
    CHECK(a&&b&&a->info.video_elements==b->info.video_elements&&a->info.audio_elements==b->info.audio_elements);
    CHECK(!memcmp(a->video,b->video,a->info.video_elements*4)&&!memcmp(a->audio,b->audio,a->info.audio_elements*4));
}
int main(int argc,char **argv){
    if(argc!=6&&argc!=7){fprintf(stderr,"usage: %s MODEL CACHE OUT off|fp8|nvfp4 switches|cancel|lifecycle|variants|adapters|reclaim|pressure|state|continuation|warm288|warm480 [ADAPTER_MODEL]\n",argv[0]);return 2;}
    /* These identity checks rebuild conditioning. Timed BF16 GEMM selection
     * can change rounding after a cache clear, so fix plans for exact repeats.
     * Performance/quality cases retain normal fast-CUDA tuning. */
    int mode;CHECK(h3_quant_parse(argv[4],&mode));mkdir(argv[3],0755);
    h3_ctx *ctx=h3_load_dir(argv[1]);if(!ctx)snprintf(error,sizeof(error),"%s",h3_last_error(NULL));CHECK(ctx);h3_cache_set_enabled(ctx,1);
    audit s={.mode=mode,.ctx=ctx};h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=128;p.frames=22;p.steps=2;p.seed=72;p.preview_vae=1;
    p.cuda_denoise_quant=mode;p.cuda_denoise_quant_cache=mode?argv[2]:NULL;p.on_frame=frame;p.on_progress=progress;p.on_latent_step=latent;p.callback_opaque=&s;
    h3_reference ref={H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0};p.references=&ref;p.reference_count=1;
    h3_result *r;
    if(!strcmp(argv[5],"switches")){
        int modes[]={0,1,1,2,2,0};h3_av_state *original=NULL,*previous=NULL;
        for(unsigned i=0;i<6;i++){
            p.cuda_denoise_quant=modes[i];p.cuda_denoise_quant_cache=modes[i]?argv[2]:NULL;
            char name[32];snprintf(name,sizeof(name),"switch-%u-%s",i,h3_quant_name(modes[i]));r=run(ctx,&p,&s,argv[3],name);
            if(i==0)original=h3_av_state_clone(r->av_state);if(i==5)same(r->av_state,original);
            if(i==2||i==4)same(r->av_state,previous);
            h3_av_state_free(previous);previous=h3_av_state_clone(r->av_state);CHECK(previous);h3_result_free(r);
        }
        h3_av_state_free(original);h3_av_state_free(previous);
        p.cuda_denoise_quant=mode;p.cuda_denoise_quant_cache=argv[2];p.width=160;ref.path="inputs/face2.jpg";
        r=run(ctx,&p,&s,argv[3],"geometry-reference-switch");h3_result_free(r);
    }else if(!strcmp(argv[5],"adapters")){
        CHECK(argc==7&&mode);p.steps=8;
        r=run(ctx,&p,&s,argv[3],"base-before");h3_av_state *before=h3_av_state_clone(r->av_state);CHECK(before);
        char identity[65];memcpy(identity,r->presentation.quant_model_sha256,65);h3_result_free(r);h3_cache_clear(ctx);
        /* A model directory is immutable for a context. Adapter changes use
         * another context, retaining the first context's model identity. */
        h3_ctx *adapted=h3_load_dir(argv[6]);CHECK(adapted);h3_cache_set_enabled(adapted,1);s.ctx=adapted;
        r=run(adapted,&p,&s,argv[3],"turbo-middle");CHECK(strcmp(identity,r->presentation.quant_model_sha256));h3_result_free(r);h3_free(adapted);
        s.ctx=ctx;r=run(ctx,&p,&s,argv[3],"base-after");same(before,r->av_state);CHECK(!strcmp(identity,r->presentation.quant_model_sha256));h3_result_free(r);h3_av_state_free(before);
    }else if(!strcmp(argv[5],"variants")){
        r=run(ctx,&p,&s,argv[3],"ref-before");CHECK(r->presentation.ref2va);h3_av_state *before=h3_av_state_clone(r->av_state);CHECK(before);h3_result_free(r);
        p.references=NULL;p.reference_count=0;p.first_frame=ref.path;
        r=run(ctx,&p,&s,argv[3],"fl-middle");CHECK(!r->presentation.ref2va);h3_result_free(r);
        p.references=&ref;p.reference_count=1;p.first_frame=NULL;
        r=run(ctx,&p,&s,argv[3],"ref-after");CHECK(r->presentation.ref2va);same(before,r->av_state);h3_result_free(r);h3_av_state_free(before);
    }else if(!strcmp(argv[5],"reclaim")){
        s.reclaim=1;p.preview_vae=0;
        r=run(ctx,&p,&s,argv[3],"reclaimed-full-vae");CHECK(s.reclaimed==1&&!s.ballast);h3_result_free(r);
        s.reclaim=0;
        r=run(ctx,&p,&s,argv[3],"after-reclamation");h3_result_free(r);
    }else if(!strcmp(argv[5],"lifecycle")){
        char aborted[1024];snprintf(aborted,sizeof(aborted),"%s/aborted.mp4",argv[3]);p.output_path=aborted;
        for(int i=1;i<=2;i++){
            s.cancel_stage="load transformer core";s.cancel_after=i;s.cancel=0;
            CHECK(!h3_generate(ctx,"A quiet studio.",&p)&&s.cancel);CHECK(access(aborted,F_OK)!=0);
            h3_cache_info info;h3_cache_get_info(ctx,&info);CHECK(!info.prepared_dit);
        }
        s.cancel_stage=NULL;s.cancel_step=1;s.cancel=0;
        CHECK(!h3_generate(ctx,"A quiet studio.",&p)&&s.cancel);CHECK(access(aborted,F_OK)!=0);
        s.cancel_step=0;
        r=run(ctx,&p,&s,argv[3],"after-repeated-cancel");h3_result_free(r);
    }else if(!strcmp(argv[5],"cancel")){
        s.cancel_stage="load transformer core";
        CHECK(!h3_generate(ctx,"A quiet studio.",&p)&&s.cancel);s.cancel_stage=NULL;
        r=run(ctx,&p,&s,argv[3],"after-cancel");h3_result_free(r);
    }else if(!strcmp(argv[5],"warm288")||!strcmp(argv[5],"warm480")){
        int large=!strcmp(argv[5],"warm480");p.width=large?480:288;p.height=large?640:384;p.frames=large?22:56;p.steps=20;
        setenv("H3_PROFILE","1",1);
        r=run(ctx,&p,&s,argv[3],"first");h3_result_free(r);
        r=run(ctx,&p,&s,argv[3],"warm");h3_result_free(r);
    }else if(!strcmp(argv[5],"pressure")){
        /* Fault budgets, like placement, are captured before request entry.
         * The full decoder also fits the narrow NVFP4 fault budget. */
        p.preview_vae=0;
        CHECK(!setenv("H3_CUDA_TEST_MEMORY_BUDGET",mode==H3_QUANT_NVFP4?"6000000000":"12000000000",1));
        r=run(ctx,&p,&s,argv[3],"pressure-stream-recovery");h3_result_free(r);
        CHECK(!unsetenv("H3_CUDA_TEST_MEMORY_BUDGET"));
    }else if(!strcmp(argv[5],"state")){
        r=run(ctx,&p,&s,argv[3],"complete");h3_av_state *expected=h3_av_state_clone(r->av_state);CHECK(expected);h3_result_free(r);
        char checkpoint[1024];snprintf(checkpoint,sizeof(checkpoint),"%s/paused.h3sample",argv[3]);
        p.save_sampler_state=checkpoint;p.stop_after_step=1;p.preview_on_stop=1;p.preview_denoise=1;
        r=run(ctx,&p,&s,argv[3],"paused");CHECK(r->status==H3_RESULT_PAUSED&&!r->av_state&&!r->sample_rate&&s.previews);h3_result_free(r);
        p.resume_sampler_state=checkpoint;p.save_sampler_state=NULL;p.stop_after_step=-1;p.preview_on_stop=0;p.preview_denoise=0;
        p.cuda_denoise_quant_set=1;p.cuda_denoise_quant=mode==1?2:1;
        CHECK(!h3_generate(ctx,NULL,&p)&&strstr(h3_last_error(ctx),"quantization"));p.cuda_denoise_quant=mode;
        r=run(ctx,&p,&s,argv[3],"resumed");same(r->av_state,expected);h3_result_free(r);h3_av_state_free(expected);
        p.resume_sampler_state=NULL;p.cuda_denoise_quant_set=0;p.preview_denoise=1;p.token_reduction=1;
        CHECK(!h3_generate(ctx,NULL,&p)); /* Shared reference recipe rejects token reduction. */
        p.token_reduction=0;p.denoise_reuse=2;
        CHECK(!h3_generate(ctx,NULL,&p)); /* Reuse would change the shared sampler contract. */
    }else if(!strcmp(argv[5],"continuation")){
        setenv("H3_CPU_SAMPLER","1",1);p.width=p.height=64;p.frames=90;p.references=NULL;p.reference_count=0;
        r=run(ctx,&p,&s,argv[3],"source");h3_av_state *source=h3_av_state_clone(r->av_state);CHECK(source);h3_result_free(r);p.continuation=source;
        for(int bridge=0;bridge<2;bridge++){
            p.continuation_mode=bridge?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;
            r=run(ctx,&p,&s,argv[3],bridge?"bridge":"hard");CHECK(r->frames==51&&r->presentation.trim_frames==39&&r->presentation.trim_samples==52000);h3_result_free(r);
        }h3_av_state_free(source);
    }else return 2;
    h3_cache_clear(ctx);h3_cache_info info;h3_cache_get_info(ctx,&info);CHECK(!info.prepared_dit&&!info.video_decoder&&!info.embedding_entries);h3_free(ctx);return 0;
}
