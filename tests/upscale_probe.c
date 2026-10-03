/* Bounded full-canvas/conditioning qualification. No comparison renders here. */
#include "src/h3.h"
#include "src/upscale/upscale.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/vae/audio_vae.h"
#include "src/sglang/sglang.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static char error[1024];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
static const char *prompt="A red ball rolls across a wooden table. Steady camera.";
static char *path(const char *dir,const char *name){char *p=NULL;CHECK(asprintf(&p,"%s/%s",dir,name)>0);return p;}
static void digest(const char *label,const void *bytes,size_t n) {
    uint8_t d[32];h3_sampler_hash(bytes,n,d);printf("%s ",label);for(int i=0;i<32;i++)printf("%02x",d[i]);puts("");
}
static h3_result *generated(h3_ctx *ctx,h3_params *p) {
    h3_result *r=h3_generate(ctx,prompt,p);if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r&&r->av_state);return r;
}
static void saved(h3_result *r,const char *dir,const char *name) {
    char *p=path(dir,name);CHECK(h3_result_save_av_state(r,p,error,sizeof(error)));free(p);
}
typedef struct {int width,height,frames,next;uint64_t edges;} frame_audit;
static int frame(const h3_frame *f,void *opaque) {
    frame_audit *a=opaque;CHECK(f->width==a->width&&f->height==a->height&&f->frame_count==a->frames&&f->frame_index==a->next++);
    CHECK(f->stride>=f->width*3&&f->rgb&&f->denoise_step<0);
    for(int x=0;x<f->width*3;x++)a->edges+=f->rgb[x]+f->rgb[(size_t)(f->height-1)*f->stride+x];
    return 0;
}
static void decoded(const char *model,const char *dir,const char *name,int w,int h,int frames) {
    char *p=path(dir,name);frame_audit a={.width=w,.height=h,.frames=frames};
    h3_decode_options o={.on_frame=frame,.callback_opaque=&a};
    double begin=h3_av_now();h3_result *r=h3_decode_av_state(model,p,&o,error,sizeof(error));CHECK(r);
    CHECK(a.next==frames&&a.edges&&r->fps==24&&r->sample_rate==32000&&r->audio_samples>=frames*32000/24);
    printf("decode %s: %dx%d frames=%d fps=%d audio_samples=%d edge_sum=%llu seconds=%.9g\n",name,w,h,a.next,r->fps,r->audio_samples,(unsigned long long)a.edges,h3_av_now()-begin);
    h3_result_free(r);free(p);
}
static void pcm_equal(const char *model,const h3_result *source,const h3_result *target,const char *dir,int cuda) {
    CHECK(source->av_state->info.audio_elements==target->av_state->info.audio_elements);
    CHECK(!memcmp(source->av_state->audio,target->av_state->audio,source->av_state->info.audio_elements*4));
    char *weights=NULL;CHECK(asprintf(&weights,"%s/%s/audio_vae",model,source->presentation.ref2va?"Ref2VA":"FL2VA")>0);
    int old=h3_sglang_exchange(cuda);h3_audio_waveform a={0},b={0};
    CHECK(h3_audio_vae_decode(weights,"src/metal/shaders.metal",source->av_state->audio,source->av_state->info.audio_t,NULL,NULL,&a,error,sizeof(error)));
    CHECK(h3_audio_vae_decode(weights,"src/metal/shaders.metal",target->av_state->audio,target->av_state->info.audio_t,NULL,NULL,&b,error,sizeof(error)));
    CHECK(a.channels==2&&b.channels==2&&a.samples==b.samples&&a.sample_rate==32000&&b.sample_rate==32000);
    size_t n=(size_t)a.samples*2;for(size_t i=0;i<n;i++)CHECK(isfinite(a.pcm[i]));
    CHECK(!memcmp(a.pcm,b.pcm,n*4));digest("identical_source_target_PCM",a.pcm,n*4);
    char *p=path(dir,"source.pcm.f32");FILE *f=fopen(p,"wbx");CHECK(f&&fwrite(a.pcm,4,n,f)==n&&!fclose(f));free(p);
    h3_audio_waveform_free(&a);h3_audio_waveform_free(&b);h3_sglang_exchange(old);free(weights);
}
static int cancel_decode(const char *phase,int done,int total,void *opaque) {
    (void)done;(void)total;(void)opaque;return !strcmp(phase,"audio VAE");
}
static int cancel_denoise(int done,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque) {
    (void)total;(void)v;(void)nv;(void)a;(void)na;(void)opaque;return done==1;
}
int main(int argc,char **argv) {
    CHECK(argc>=5);const char *mode=argv[1],*model=argv[2],*dir=argv[4];
    if(!strcmp(mode,"retarget")) {
        CHECK(argc==5);h3_ctx *ctx=h3_load_dir(model);CHECK(ctx);char *p=path(dir,"moved.h3up");
        h3_upscale_source *s=h3_upscale_source_load(p,error,sizeof(error));CHECK(s);free(p);
        h3_upscale_options o=H3_UPSCALE_OPTIONS_DEFAULT;o.model_path=argv[3];o.refine_steps=2;o.state_only=1;
        h3_result *r=h3_upscale(ctx,s,&o);if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r&&r->av_state);
        CHECK(!memcmp(r->av_state->audio,s->state->audio,s->state->audio_elements*4));saved(r,dir,"target.h3av");
        digest("target_audio",r->av_state->audio,r->av_state->info.audio_elements*4);
        h3_result_free(r);h3_upscale_source_free(s);h3_free(ctx);puts("ok: moved source, removed original media, learned retarget and two refinements");return 0;
    }
    h3_ctx *ctx=h3_load_dir(model);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=128;p.height=64;p.frames=22;p.steps=2;p.seed=17;p.state_only=1;
    char *source_path=path(dir,"source.h3up");p.save_upscale_state=source_path;
    if(!strcmp(mode,"capture")) {
        CHECK(argc==6);const char *kind=argv[5];char *first=path(dir,"first.png"),*last=path(dir,"portrait.jpg"),*video=path(dir,"reference.mp4"),*audio=path(dir,"audio.wav");
        h3_reference refs[]={{H3_REFERENCE_IMAGE,last,NULL,0},{H3_REFERENCE_VIDEO_AUDIO,video,audio,0},{H3_REFERENCE_AUDIO,audio,NULL,0},{H3_REFERENCE_IMAGE,first,NULL,0}};
        if(!strcmp(kind,"first")||!strcmp(kind,"both"))p.first_frame=first;
        if(!strcmp(kind,"last")||!strcmp(kind,"both"))p.last_frame=last;
        if(!strcmp(kind,"mixed")||!strcmp(kind,"max")){p.frames=56;p.references=refs;p.reference_count=!strcmp(kind,"max")?1:4;if(!strcmp(kind,"max"))p.reference_image_size=H3_REFERENCE_IMAGE_MAX;}
        int cancelled=!strcmp(kind,"cancel");
        if(cancelled)p.on_latent_step=cancel_denoise;
        else {p.state_only=0;p.on_progress=cancel_decode;}
        h3_result *r=h3_generate(ctx,prompt,&p);snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(!r);
        if(cancelled){CHECK(strstr(h3_last_error(ctx),"latent callback stopped"));CHECK(access(source_path,F_OK)!=0);puts("ok: denoise cancellation publishes no clean source");}
        else {CHECK(strstr(h3_last_error(ctx),"cancelled during audio VAE"));h3_upscale_source *s=h3_upscale_source_load(source_path,error,sizeof(error));CHECK(s);
            CHECK(s->state->conditioned==(p.first_frame||p.last_frame||p.reference_count?1:0));
            printf("ok: %s clean source survives decoder cancellation; references=%zu anchors=%u\n",kind,s->state->reference_count,s->state->upscale.keyframe_count);
            h3_upscale_source_free(s);
        }
        free(first);free(last);free(video);free(audio);free(source_path);h3_free(ctx);return 0;
    }
    CHECK(argc==7);p.width=atoi(argv[5]);p.height=atoi(argv[6]);p.frames=90;
    if(!strcmp(mode,"direct")) {
        p.geometry_profile=1;p.save_upscale_state=NULL;p.steps=1;
        h3_result *r=generated(ctx,&p);saved(r,dir,"direct.h3av");h3_result_free(r);h3_free(ctx);free(source_path);
        puts("ok: direct explicit-profile full-depth 90-frame probe, one evaluation");return 0;
    }
    CHECK(!strcmp(mode,"canvas"));h3_result *source=generated(ctx,&p);saved(source,dir,"source.h3av");
    h3_upscale_source *s=h3_upscale_source_load(source_path,error,sizeof(error));CHECK(s);
    h3_upscale_options o=H3_UPSCALE_OPTIONS_DEFAULT;o.model_path=argv[3];o.refine_steps=2;o.state_only=1;
    h3_result *target=h3_upscale(ctx,s,&o);if(!target)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(target&&target->av_state);
    CHECK(target->width==2*p.width&&target->height==2*p.height&&target->frames==90);saved(target,dir,"target.h3av");
    char *repeat=path(dir,"isolation.h3up");p.save_upscale_state=repeat;
    h3_result *again=generated(ctx,&p);CHECK(again->av_state->info.video_elements==source->av_state->info.video_elements);
    CHECK(!memcmp(again->av_state->video,source->av_state->video,source->av_state->info.video_elements*4));
    CHECK(!memcmp(again->av_state->audio,source->av_state->audio,source->av_state->info.audio_elements*4));
    h3_result_free(again);free(repeat);h3_upscale_source_free(s);
    int cuda=!strcmp(h3_device(ctx)->backend,"cuda")?H3_SGLANG_VERSION:0;h3_free(ctx);
    pcm_equal(model,source,target,dir,cuda);
    h3_result_free(source);h3_result_free(target);
    decoded(model,dir,"source.h3av",p.width,p.height,90);decoded(model,dir,"target.h3av",2*p.width,2*p.height,90);
    free(source_path);puts("ok: full canvas, six evaluations, dense-upscale-dense isolation, PCM identity, full tiled decode");return 0;
}
