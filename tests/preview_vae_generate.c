/* Bounded public-API integration; rendered smoke media is not a quality gate. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <sys/stat.h>

static char error[512];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
typedef struct {float *trace[3];size_t elements;int compare,frames,previews,calls;double last,preview_seconds;} audit;
static int latent(int step,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque){
    audit *s=opaque;CHECK(total==2&&step<=2);size_t n=nv+na;s->last=h3_av_now();s->calls++;
    if(s->compare){CHECK(s->elements==n&&s->trace[step]);CHECK(!memcmp(s->trace[step],v,nv*4)&&!memcmp(s->trace[step]+nv,a,na*4));}
    else{if(!s->trace[step])s->trace[step]=malloc(n*4);CHECK(s->trace[step]);memcpy(s->trace[step],v,nv*4);memcpy(s->trace[step]+nv,a,na*4);s->elements=n;}return 0;
}
static int frame(const h3_frame *f,void *opaque){audit *s=opaque;CHECK(f->rgb&&f->frame_index>=0&&f->frame_index<f->frame_count);
    if(f->denoise_step>=0){s->previews++;if(s->last>0)s->preview_seconds+=h3_av_now()-s->last;}
    else{CHECK(f->frame_index==s->frames);s->frames++;}return 0;}
static h3_result *run(h3_ctx *ctx,h3_params *p,audit *s,const char *dir,const char *name){
    char output[1024],state[1024],stats[1024];snprintf(output,sizeof(output),"%s/%s.mp4",dir,name);
    snprintf(state,sizeof(state),"%s/%s.h3av",dir,name);snprintf(stats,sizeof(stats),"%s/%s.json",dir,name);
    p->output_path=output;p->on_frame=frame;p->callback_opaque=s;
    s->frames=s->previews=s->calls=0;s->preview_seconds=0;s->last=0;double begin=h3_av_now();
    h3_result *r=h3_generate(ctx,"A person smiles gently in a quiet studio. Soft daylight, fixed camera.",p);
    if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r);
    if(r->status==H3_RESULT_COMPLETE){CHECK(s->frames==r->frames);CHECK(h3_result_save_av_state(r,state,error,sizeof(error)));}
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);FILE *f=fopen(stats,"w");CHECK(f);
    fprintf(f,"{\"seconds\":%.9f,\"preview_seconds\":%.9f,\"previews\":%d,\"frames\":%d,\"max_rss\":%ld,\"status\":%d}\n",
        h3_av_now()-begin,s->preview_seconds,s->previews,r->frames,usage.ru_maxrss,r->status);fclose(f);
    fprintf(stderr,"PASS %s: %.3f seconds, frames=%d previews=%d\n",name,h3_av_now()-begin,r->frames,s->previews);return r;
}
static void same_state(const h3_result *r,const h3_av_state *expected){
    const h3_av_state *a=r->av_state;CHECK(a&&a->info.video_elements==expected->info.video_elements&&a->info.audio_elements==expected->info.audio_elements);
    CHECK(!memcmp(a->video,expected->video,a->info.video_elements*4));CHECK(!memcmp(a->audio,expected->audio,a->info.audio_elements*4));
}
int main(int argc,char **argv){
    if(argc!=4||(strcmp(argv[3],"invariance")&&strcmp(argv[3],"matrix")&&strcmp(argv[3],"continuation")&&strcmp(argv[3],"adapter")&&strcmp(argv[3],"cache"))){fprintf(stderr,"usage: %s MODEL_DIR OUTPUT_DIR invariance|matrix|continuation|adapter|cache\n",argv[0]);return 2;}
    setenv("H3_CPU_SAMPLER","1",1);unsetenv("H3_GPU_SAMPLER");mkdir(argv[2],0755);
    h3_ctx *ctx=h3_load_dir(argv[1]);if(!ctx)snprintf(error,sizeof(error),"%s",h3_last_error(NULL));CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=128;p.render_width=p.render_height=64;p.frames=22;p.steps=2;p.seed=72;
    audit s={0};h3_result *r;
    if(!strcmp(argv[3],"invariance")){
        p.on_latent_step=latent;r=run(ctx,&p,&s,argv[2],"default");h3_av_state *baseline=h3_av_state_clone(r->av_state);CHECK(baseline);h3_result_free(r);s.compare=1;
        p.preview_vae=1;r=run(ctx,&p,&s,argv[2],"tiny");same_state(r,baseline);h3_result_free(r);
        p.preview_denoise=1;setenv("H3_PREVIEW_MODE","noisy",1);r=run(ctx,&p,&s,argv[2],"tiny-show-noisy");CHECK(s.previews==2);same_state(r,baseline);h3_result_free(r);
        setenv("H3_PREVIEW_MODE","denoised",1);r=run(ctx,&p,&s,argv[2],"tiny-show-denoised");CHECK(s.previews==2);same_state(r,baseline);h3_result_free(r);
        char checkpoint[1024];snprintf(checkpoint,sizeof(checkpoint),"%s/paused.h3sample",argv[2]);p.stop_after_step=1;p.save_sampler_state=checkpoint;p.preview_on_stop=1;
        r=run(ctx,&p,&s,argv[2],"paused-tiny");CHECK(r->status==H3_RESULT_PAUSED&&!r->av_state&&!r->sample_rate);h3_result_free(r);
        p.resume_sampler_state=checkpoint;p.save_sampler_state=NULL;p.stop_after_step=-1;p.preview_on_stop=0;p.preview_denoise=0;p.preview_vae=0;
        r=run(ctx,&p,&s,argv[2],"resumed-full");same_state(r,baseline);h3_result_free(r);p.resume_sampler_state=NULL;
        r=run(ctx,&p,&s,argv[2],"default-after");same_state(r,baseline);h3_result_free(r);
        h3_cache_clear(ctx);h3_cache_info info;h3_cache_get_info(ctx,&info);CHECK(!info.prepared_dit&&!info.video_decoder&&!info.embedding_entries);
        h3_av_state_free(baseline);puts("PASS exact authoritative trajectory: default/tiny/show noisy/show denoised/pause tiny/resume full/full-after; cache cleared");
    }else if(!strcmp(argv[3],"cache")){
        p.preview_vae=1;p.on_latent_step=latent;
        r=run(ctx,&p,&s,argv[2],"real");h3_av_state *baseline=h3_av_state_clone(r->av_state);CHECK(baseline);
        char original[65];memcpy(original,r->presentation.tiny_sha256,65);h3_result_free(r);s.compare=1;
        p.preview_vae_model=getenv("H3_PREVIEW_TEST_PATTERN");if(!p.preview_vae_model)p.preview_vae_model="outputs/preview-vae/metal/pattern.safetensors";
        r=run(ctx,&p,&s,argv[2],"pattern");same_state(r,baseline);CHECK(strcmp(original,r->presentation.tiny_sha256));h3_result_free(r);
        p.preview_vae_model="missing-preview-model";
        CHECK(!h3_generate(ctx,"A person smiles gently in a quiet studio. Soft daylight, fixed camera.",&p));
        h3_cache_info info;h3_cache_get_info(ctx,&info);CHECK(info.prepared_dit&&info.video_decoder);
        p.preview_vae_model=NULL;r=run(ctx,&p,&s,argv[2],"real-again");same_state(r,baseline);CHECK(!strcmp(original,r->presentation.tiny_sha256));h3_result_free(r);
        h3_cache_clear(ctx);h3_cache_get_info(ctx,&info);CHECK(!info.video_decoder&&!info.prepared_dit);h3_av_state_free(baseline);
        puts("PASS model identity cache invalidation, missing-model early failure, real/pattern/real sampling invariance, explicit clear");
    }else if(!strcmp(argv[3],"continuation")){
        p.width=p.height=64;p.render_width=p.render_height=0;p.frames=90;p.preview_vae=1;p.preview_denoise=1;
        r=run(ctx,&p,&s,argv[2],"source");h3_av_state *source=h3_av_state_clone(r->av_state);CHECK(source);h3_result_free(r);
        p.continuation=source;p.on_latent_step=latent;
        for(int bridge=0;bridge<2;bridge++){
            for(int i=0;i<3;i++){free(s.trace[i]);s.trace[i]=NULL;}s.compare=0;
            p.continuation_mode=bridge?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;p.preview_vae=0;p.preview_denoise=0;
            r=run(ctx,&p,&s,argv[2],bridge?"bridge-full":"hard-full");CHECK(r->frames==51&&r->presentation.trim_frames==39&&r->presentation.trim_samples==52000);
            h3_av_state *expected=h3_av_state_clone(r->av_state);CHECK(expected);h3_result_free(r);
            s.compare=1;p.preview_vae=1;p.preview_denoise=1;r=run(ctx,&p,&s,argv[2],bridge?"bridge-tiny":"hard-tiny");same_state(r,expected);CHECK(r->frames==51&&s.previews==2);h3_result_free(r);h3_av_state_free(expected);
        }h3_av_state_free(source);puts("PASS hard/bridge continuation invariance and 39-frame/52000-sample trim; tiny live previews");
    }else{
        p.preview_vae=1;h3_reference refs[2]={{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0},{H3_REFERENCE_IMAGE,"inputs/body1.jpg",NULL,0}};
        const char *names[]={"first","last","first-last","image","images","video","audio","video-audio","reduction","reuse"};
        int count=!strcmp(argv[3],"adapter")?2:10;
        int start=getenv("H3_PREVIEW_START_CASE")?atoi(getenv("H3_PREVIEW_START_CASE")):0;
        CHECK(start>=0&&start<count);
        for(int i=start;i<count;i++){
            p.frames=(i==5||i==7)?56:22;
            p.first_frame=p.last_frame=NULL;p.references=NULL;p.reference_count=0;p.token_reduction=0;p.denoise_reuse=1;
            refs[0]=(h3_reference){H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0};
            if(i==0||i==2)p.first_frame="inputs/face1.jpg";if(i==1||i==2)p.last_frame="inputs/face2.jpg";
            if(i>=3){p.references=refs;p.reference_count=i==4?2:1;}
            if(i==5||i==7)refs[0]=(h3_reference){i==5?H3_REFERENCE_VIDEO:H3_REFERENCE_VIDEO_AUDIO,"outputs/refvideo-integration-validation/legacy-renders/motion-reference.mp4",i==7?"outputs/continuation-validation/reference.wav":NULL,0};
            if(i==6){refs[1]=(h3_reference){H3_REFERENCE_AUDIO,"outputs/continuation-validation/reference.wav",NULL,0};p.reference_count=2;}
            if(i==8)p.token_reduction=1;if(i==9)p.denoise_reuse=2;
            const char *name=names[i];
            if(!strcmp(argv[3],"adapter")&&i==1){p.first_frame=p.last_frame=NULL;p.references=refs;p.reference_count=1;name="image";}
            r=run(ctx,&p,&s,argv[2],name);h3_result_free(r);
        }
    }
    for(int i=0;i<3;i++)free(s.trace[i]);h3_free(ctx);return 0;
}
