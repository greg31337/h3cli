#include "src/denoise/attention.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/denoise/dit.h"
#include "src/internal.h"
#include "src/platform.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include "src/weights/quant.h"
#include "src/media/delivery.h"
#include "src/denoise/approximate.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>
#include <signal.h>
#include <sys/resource.h>
#include <sys/wait.h>

static int checks;
static char error[512];
static int fixture_width=64, fixture_height=32, fixture_frames=56, fixture_profile, fixture_keyframes, fixture_images;
static const char *adaptive_fixture_path,*adaptive_reference_fixture_path;
#define CHECK(x) do { checks++; if(!(x)) { fprintf(stderr,"FAIL %s:%d: %s (%s)\n",__FILE__,__LINE__,#x,error); exit(1); } } while(0)
static void *clone(const void *p,size_t n) { void *q=malloc(n); CHECK(q); memcpy(q,p,n); return q; }
static void write_file(const char *p,const char *value) { FILE *f=fopen(p,"wb"); CHECK(f); CHECK(fwrite(value,1,strlen(value),f)==strlen(value)); CHECK(!fclose(f));
    /* Linux filesystems can stamp rapid same-size fixture rewrites in the same
     * clock tick. Separate the test mutations before exercising cache identity. */
    struct timespec tick={0,10000000}; nanosleep(&tick,NULL);
}
static uint8_t *read_file(const char *p,size_t *n) { FILE *f=fopen(p,"rb"); CHECK(f); CHECK(!fseek(f,0,SEEK_END)); *n=(size_t)ftell(f); rewind(f); uint8_t *data=malloc(*n); CHECK(data); CHECK(fread(data,1,*n,f)==*n); fclose(f); return data; }
static void reference_av_metadata(void) {
    char root[]="/tmp/h3-reference-av-XXXXXX",path[1024];CHECK(mkdtemp(root));
    const char *dirs[]={"FL2VA","FL2VA/transformer","FL2VA/video_vae","FL2VA/video_vae/source","FL2VA/audio_vae"};
    const char *files[]={"transformer/config.json","video_vae/config.json","video_vae/source/config.json",
        "audio_vae/config.json","audio_vae/config.yaml","audio_vae/metadata.json",
        "video_vae/source/model.safetensors","audio_vae/model.safetensors"};
    for(size_t i=0;i<5;i++){snprintf(path,sizeof(path),"%s/%s",root,dirs[i]);CHECK(!mkdir(path,0700));}
    for(size_t i=0;i<8;i++){snprintf(path,sizeof(path),"%s/FL2VA/%s",root,files[i]);write_file(path,"fixture");}
    uint8_t a[32],b[32];
    CHECK(h3_av_state_metadata_signature(root,0,a,error,sizeof(error)));
    CHECK(h3_av_state_metadata_signature(root,0,b,error,sizeof(error)));CHECK(!memcmp(a,b,32));
    /* Unreadable weights still allow metadata identity; ctime invalidates it. */
    CHECK(!chmod(path,0000));CHECK(h3_av_state_metadata_signature(root,0,b,error,sizeof(error)));CHECK(memcmp(a,b,32));
    CHECK(h3_av_state_metadata_signature(root,0,a,error,sizeof(error)));CHECK(!memcmp(a,b,32));
    CHECK(!chmod(path,0600));write_file(path,"changed");
    CHECK(h3_av_state_metadata_signature(root,0,b,error,sizeof(error)));CHECK(memcmp(a,b,32));
    CHECK(!h3_av_state_metadata_signature(root,2,b,error,sizeof(error)));
    for(size_t i=0;i<8;i++){snprintf(path,sizeof(path),"%s/FL2VA/%s",root,files[i]);CHECK(!unlink(path));}
    CHECK(!h3_av_state_metadata_signature(root,0,b,error,sizeof(error)));
    for(int i=4;i>=0;i--){snprintf(path,sizeof(path),"%s/%s",root,dirs[i]);CHECK(!rmdir(path));}CHECK(!rmdir(root));
}
static h3_sampler_state *fixture(const char *directory,int ref2va,int continuation,int reuse) {
    h3_params p=H3_PARAMS_DEFAULT; p.width=fixture_width; p.height=fixture_height; p.frames=fixture_frames; p.steps=20; p.seed=UINT64_MAX; p.denoise_reuse=reuse;p.geometry_profile=fixture_profile;
    p.continuation_mode=continuation==2?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;
    uint8_t signature[32]={1,2,3};
    h3_av_state *source=h3_av_state_new_profile(p.width,p.height,p.frames,p.geometry_profile,42,signature); CHECK(source);
    h3_rng random; h3_rng_seed(&random,42); h3_rng_fill_normal(&random,source->video,source->info.video_elements); h3_rng_fill_normal(&random,source->audio,source->info.audio_elements);
    if(continuation) p.continuation=source;
    h3_text_embedding text={0}; text.tokens=3; text.width=H3_TEXT_HIDDEN_SIZE;
    text.values=malloc(text.tokens*text.width*2); text.tags=malloc(text.tokens); CHECK(text.values&&text.tags);
    for(size_t i=0;i<text.tokens*text.width;i++) text.values[i]=(uint16_t)(i*71); /* includes raw BF16 NaN payload bits */
    text.tags[0]=1; text.tags[1]=0; text.tags[2]=1;
    char path[1024]; snprintf(path,sizeof(path),"%s/reference",directory);
    h3_reference refs[]={{H3_REFERENCE_IMAGE,path,NULL,0},{H3_REFERENCE_AUDIO,path,NULL,0},{H3_REFERENCE_VIDEO_AUDIO,path,path,1}};
    h3_layout_ref layout_refs[]={{H3_LAYOUT_REF_IMAGE,1,2,4,0},{H3_LAYOUT_REF_AUDIO,0,0,0,4},{H3_LAYOUT_REF_VIDEO,2,2,4,3}};
    if(fixture_images)for(int i=0;i<3;i++){refs[i]=refs[0];layout_refs[i]=layout_refs[0];}
    if(ref2va) { p.references=refs; p.reference_count=3; }
    int keys[2],nk=0;
    if(fixture_keyframes&1){p.first_frame=path;keys[nk++]=0;}
    if(fixture_keyframes&2){p.last_frame=path;keys[nk++]=p.frames-1;}
    h3_layout_spec spec={3,source->info.video_t,source->info.latent_h,source->info.latent_w,source->info.audio_t,p.frames,keys,(size_t)nk,layout_refs,p.reference_count};
    h3_layout layout; CHECK(h3_layout_build(&spec,&layout,error,sizeof(error)));
    h3_bridge_profile bridge;
    if(continuation) CHECK(h3_continuation_context(39,&layout.prefix));
    if(continuation==2) { CHECK(h3_bridge_profile_build(39,8,0.5f,H3_BRIDGE_STEPPED,&bridge,NULL,0,error,sizeof(error))); layout.bridge=&bridge; }
    size_t nv=layout.img_cond_rows*96,na=layout.audio_cond_rows*32;
    float *cv=calloc(nv?nv:1,4),*ca=calloc(na?na:1,4); CHECK(cv&&ca);
    for(size_t i=0;i<nv;i++) cv[i]=(float)i/128;
    for(size_t i=0;i<na;i++) ca[i]=-(float)i/128;
    h3_sigma_schedule sigmas; CHECK(h3_serving_schedule_build(20,&sigmas));
    h3_sampler_state *s=h3_sampler_state_create(&sigmas,source->info.video_elements,source->info.audio_elements,reuse); CHECK(s);
    h3_device_info device={0}; device.apple_gpu_family=9; device.metal4=1; device.unified_memory=1;
    CHECK(h3_sampler_state_capture_effective(s,directory,NULL,&device,&p,"  exact prompt\n<Picture 1> café 🙂  ",&text,&layout,layout_refs,p.reference_count,cv,nv,ca,na,nv||na,signature,error,sizeof(error)));
    h3_rng_seed(&s->video_rng,p.seed); h3_rng_fill_normal(&s->video_rng,s->video,s->video_elements);
    h3_rng_seed(&s->audio_rng,p.seed); h3_rng_fill_normal(&s->audio_rng,s->audio,s->audio_elements);
    s->video_random_count=s->video_elements; s->audio_random_count=s->audio_elements;
    s->original_video_noise=clone(s->video,s->video_elements*4); s->original_audio_noise=clone(s->audio,s->audio_elements*4);
    if(continuation) {
        if(continuation==2) CHECK(h3_av_state_insert_bridge(source,&source->info,&bridge,1,1,s->video,s->audio));
        else CHECK(h3_av_state_insert_prefix(source,&source->info,layout.prefix,s->video,s->audio,1));
    }
    s->video[0]=-0.0f; s->audio[0]=-0.0f;
    s->next_step=4;
    if(reuse>1) {
        for(int i=0;i<4;i++) if(s->selected[i]) { s->previous_evaluated=s->last_evaluated; s->last_evaluated=i; }
        for(size_t i=0;i<s->video_elements;i++) { s->last_video_velocity[i]=(float)i/16; s->previous_video_velocity[i]=-(float)i/32; }
        for(size_t i=0;i<s->audio_elements;i++) { s->last_audio_velocity[i]=-(float)i/16; s->previous_audio_velocity[i]=(float)i/32; }
    }
    s->token_count=3; s->token_ids=malloc(12); CHECK(s->token_ids);
    s->token_ids[0]=1; s->token_ids[1]=UINT32_MAX; s->token_ids[2]=151644;
    s->presentation_positions=calloc(9,4); s->presentation_spans=calloc(2,8); CHECK(s->presentation_positions&&s->presentation_spans);
    s->presentation_span_count=1; s->presentation_spans[0]=1; s->presentation_spans[1]=1;
    CHECK(h3_sampler_state_validate(s,error,sizeof(error)));
    free(cv); free(ca); h3_layout_free(&layout); h3_text_embedding_free(&text); h3_av_state_free(source);
    return s;
}
static void adaptive_continuation_extensions(const char *directory,const char *path) {
    for(int mode=1;mode<=2;mode++)for(int ref=0;ref<=1;ref++)for(int kind=1;kind<=3;kind++) {
        h3_sampler_state *s=fixture(directory,ref,mode,1);
        s->params.adaptive_cache=kind!=2;s->params.cuda_attention=kind!=1?H3_ATTENTION_SUBBLOCK:0;
        s->params.subblock_warmup=s->params.cuda_attention?3:0;
        s->params.adaptive_cache_warmup=s->params.adaptive_cache?2:0;
        s->attention_version=h3_attention_execution_recipe(s->params.cuda_attention,0);
        s->attention_plan=h3_attention_plan(s->params.cuda_attention);
        s->full_sequence=s->layout.seq_len;
        if(s->params.adaptive_cache) {
            h3_adaptive_plan plan;CHECK(h3_adaptive_plan_recipe(1,4,s->full_sequence,5376,0,&plan,error,sizeof(error)));
            s->adaptive_version=4;s->adaptive_elements=plan.elements;
            s->adaptive_anchor=calloc(plan.elements,2);s->adaptive_delta=calloc(plan.elements,2);CHECK(s->adaptive_anchor&&s->adaptive_delta);
            s->adaptive_history=(h3_adaptive_history){.ready=1,.last_step=3,.last_refresh=3,.phase=(unsigned)(kind==3)};
        }
        CHECK(h3_sampler_state_validate(s,error,sizeof(error)));
        CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error));CHECK(r);
        CHECK(r->continuation&&r->context_frames==39&&r->adaptive_version==s->adaptive_version);
        CHECK(!memcmp(&r->adaptive_history,&s->adaptive_history,sizeof(s->adaptive_history)));
        uint8_t a[32],b[32];CHECK(h3_sampler_prepared_key(r,a));
        if(mode==1) {
            r->params.bridge_max_strength=.25f;r->params.bridge_video_steps=1;r->params.bridge_profile=H3_BRIDGE_LINEAR;
            CHECK(h3_sampler_prepared_key(r,b)&&!memcmp(a,b,32));
        }
        r->context_frames=90;CHECK(!h3_sampler_layout_validate(r,error,sizeof(error)));r->context_frames=39;
        r->continuation=0;CHECK(!h3_sampler_layout_validate(r,error,sizeof(error)));r->continuation=1;
        r->layout.prefix.video_prefix_t++;CHECK(!h3_sampler_layout_validate(r,error,sizeof(error)));r->layout.prefix.video_prefix_t--;
        r->params.cuda_denoise_quant=1;CHECK(!h3_sampler_layout_validate(r,error,sizeof(error)));r->params.cuda_denoise_quant=0;
        if(mode==2) {r->bridge.class_mask[4]=NAN;CHECK(!h3_sampler_layout_validate(r,error,sizeof(error)));r->bridge=s->bridge;}
        if(s->params.adaptive_cache) {
            r->adaptive_version=3;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->adaptive_version=4;
            r->adaptive_history.streak=17;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->adaptive_history=s->adaptive_history;
            CHECK(!h3_sampler_state_load_with_budget(path,1,1,error,sizeof(error)));
        }
        CHECK(h3_sampler_state_validate(r,error,sizeof(error)));
        h3_sampler_state_free(r);h3_sampler_state_free(s);
    }
}
static void warmup_extensions(const char *directory,const char *path) {
    for(int kind=1;kind<=3;kind++)for(int quant=0;quant<=2;quant++)for(int n=2;n<=16;n+=14) {
        if(kind==3&&quant)continue;
        h3_sampler_state *s=fixture(directory,0,0,1);
        s->full_sequence=s->layout.seq_len;
        s->params.adaptive_cache=kind!=2;
        s->params.cuda_attention=kind!=1?H3_ATTENTION_SUBBLOCK:0;
        s->params.cuda_denoise_quant=quant;
        s->quant_version=h3_quant_execution_recipe(quant,s->params.adaptive_cache,s->params.cuda_attention);
        s->attention_version=h3_attention_execution_recipe(s->params.cuda_attention,quant);
        s->attention_plan=h3_attention_plan(s->params.cuda_attention);
        s->params.adaptive_cache_warmup=s->params.adaptive_cache?n:0;
        s->params.subblock_warmup=s->params.cuda_attention?18-n:0;
        if(s->params.adaptive_cache) {
            h3_adaptive_plan p;CHECK(h3_adaptive_plan_make(1,s->full_sequence,5376,&p,error,sizeof(error)));
            s->adaptive_version=h3_adaptive_recipe(1,quant,0);s->adaptive_elements=p.elements;
            s->adaptive_anchor=calloc(p.elements,2);s->adaptive_delta=calloc(p.elements,2);CHECK(s->adaptive_anchor&&s->adaptive_delta);
            s->adaptive_history=(h3_adaptive_history){.ready=1,.last_step=3,.last_refresh=3,
                .phase=(unsigned)(s->params.cuda_attention&&3>=s->params.subblock_warmup)};
        }
        CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        if(kind==1&&!quant&&n==2&&adaptive_fixture_path)CHECK(h3_sampler_state_save(s,adaptive_fixture_path,error,sizeof(error)));
        h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error));CHECK(r);
        if(s->params.adaptive_cache) {
            CHECK(r->params.adaptive_cache_max_bytes==H3_ADAPTIVE_DEFAULT_BYTES);
            uint8_t a[32],b[32];CHECK(h3_sampler_prepared_key(r,a));
            r->params.adaptive_cache_max_bytes=(UINT64_C(512)*1024*1024);
            CHECK(h3_sampler_prepared_key(r,b)&&!memcmp(a,b,32));
            h3_sampler_state *override=h3_sampler_state_load_with_budget(path,(UINT64_C(512)*1024*1024),1,error,sizeof(error));CHECK(override);
            CHECK(override->params.adaptive_cache_max_bytes==(UINT64_C(512)*1024*1024));
            CHECK(!memcmp(&override->adaptive_history,&r->adaptive_history,sizeof(r->adaptive_history)));
            h3_sampler_state_free(override);
            CHECK(!h3_sampler_state_load_with_budget(path,1,1,error,sizeof(error))&&strstr(error,"minimum"));
            CHECK(!h3_sampler_state_load_with_budget(path,0,1,error,sizeof(error)));
        }
        CHECK(r->params.adaptive_cache_warmup==s->params.adaptive_cache_warmup&&r->params.subblock_warmup==s->params.subblock_warmup);
        uint8_t before[32],after[32];CHECK(h3_sampler_prepared_key(r,before));
        if(kind==2)r->params.subblock_warmup=n;else r->params.adaptive_cache_warmup=18-n;
        CHECK(h3_sampler_prepared_key(r,after));CHECK(memcmp(before,after,32));
        if(s->params.adaptive_cache){
            r->params.adaptive_cache_warmup=s->params.adaptive_cache_warmup;
            r->params.subblock_warmup=s->params.subblock_warmup;
            r->adaptive_history.streak=17;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));
            r->adaptive_history=s->adaptive_history;
            r->adaptive_history.last_refresh=-1;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));
            r->adaptive_history=s->adaptive_history;
        }
        r->params.subblock_warmup=17;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));
        h3_sampler_state_free(r);h3_sampler_state_free(s);
    }
}
static void subblock_reference_state(const char *directory,const char *path) {
    for(int images_only=0;images_only<=1;images_only++)for(int mode=0;mode<=2;mode++)for(int cache=0;cache<=2;cache++)for(int sparse=0;sparse<=1;sparse++) {
        if(!cache&&!sparse)continue;
        fixture_images=images_only;
        h3_sampler_state *s=fixture(directory,1,0,1);
        s->params.reference_image_size=mode;s->params.cuda_attention=sparse?H3_ATTENTION_SUBBLOCK:0;
        s->attention_version=sparse?H3_ATTENTION_VERSION:0;s->attention_plan=sparse?H3_ATTENTION_PLAN_VERSION:0;
        s->params.subblock_warmup=sparse?2:0;s->params.subblock_sparsity=sparse?.75f:0;
        s->params.adaptive_cache=cache;s->full_sequence=s->layout.seq_len;
        if(cache){
            s->params.adaptive_cache_threshold=mode==0?0:mode==1?.06f:1;
            s->params.adaptive_cache_threshold_set=1;s->params.adaptive_cache_max_hits=mode==2?16:2;
            h3_adaptive_plan plan;CHECK(h3_adaptive_plan_make(cache,s->full_sequence,5376,&plan,error,sizeof(error)));
            s->adaptive_version=3;s->adaptive_elements=plan.elements;
            s->adaptive_anchor=calloc(plan.elements,2);s->adaptive_delta=calloc(plan.elements,2);CHECK(s->adaptive_anchor&&s->adaptive_delta);
            s->adaptive_history=(h3_adaptive_history){.ready=1,.last_step=3,.last_refresh=3,.phase=(unsigned)sparse};
        }
        CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        if(adaptive_reference_fixture_path&&!images_only&&mode==1&&cache==1&&sparse)
            CHECK(h3_sampler_state_save(s,adaptive_reference_fixture_path,error,sizeof(error)));
        h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error));CHECK(r);
        CHECK(r->reference_count==3&&r->params.subblock_warmup==(sparse?2:0)&&r->params.reference_image_size==(h3_reference_image_size)mode);
        CHECK(!memcmp(r->references,s->references,3*sizeof(*s->references)));
        if(cache){
            CHECK(r->adaptive_version==3&&r->params.adaptive_cache_threshold_set);
            CHECK(r->params.adaptive_cache_threshold==s->params.adaptive_cache_threshold&&r->params.adaptive_cache_max_hits==s->params.adaptive_cache_max_hits);
            CHECK(h3_adaptive_threshold(&r->params)==s->params.adaptive_cache_threshold);
            uint8_t before[32],after[32];CHECK(h3_sampler_prepared_key(r,before));
            r->params.adaptive_cache_max_hits--;CHECK(h3_sampler_prepared_key(r,after)&&memcmp(before,after,32));r->params.adaptive_cache_max_hits++;
            r->adaptive_version=1;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->adaptive_version=3;
            r->params.cuda_denoise_quant=1;r->quant_version=3;CHECK(!h3_sampler_layout_validate(r,error,sizeof(error)));r->params.cuda_denoise_quant=0;r->quant_version=0;
        }
        r->references[2].latent_w+=2;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->references[2].latent_w-=2;
        int old_h=r->references[2].latent_h,old_w=r->references[2].latent_w;
        r->references[2].latent_h=r->references[2].latent_w=4096;
        CHECK(!h3_sampler_layout_validate(r,error,sizeof(error)));
        r->references[2].latent_h=old_h;r->references[2].latent_w=old_w;
        int saved_t=r->references[2].latent_t;r->references[2].latent_t=0;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->references[2].latent_t=saved_t;
        r->layout.positions[0].w+=1;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->layout.positions[0].w-=1;
        r->provenance[8]=H3_REFERENCE_AUDIO;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->provenance[8]=H3_REFERENCE_IMAGE;
        r->references[2].kind=H3_LAYOUT_REF_AUDIO;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));
        h3_sampler_state_free(r);h3_sampler_state_free(s);
    }
    fixture_images=0;
}
static void reference_image_modes(const char *directory,const char *path) {
    char original[1024],hidden[1024];
    snprintf(original,sizeof(original),"%s/reference",directory);
    snprintf(hidden,sizeof(hidden),"%s/reference-hidden",directory);
    for(int mode=0;mode<=2;mode++) {
        h3_sampler_state *s=fixture(directory,1,0,1);
        s->params.reference_image_size=mode;
        CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        CHECK(!rename(original,hidden));
        h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error));CHECK(r);
        CHECK(r->params.reference_image_size==(h3_reference_image_size)mode);
        CHECK(r->references[0].latent_w==4&&r->references[0].latent_h==2);
        CHECK(r->condition_video_elements==s->condition_video_elements);
        CHECK(!memcmp(r->condition_video,s->condition_video,s->condition_video_elements*4));
        CHECK(!rename(hidden,original));h3_sampler_state_free(r);
        s->params.reference_image_size=3;CHECK(!h3_sampler_state_validate(s,error,sizeof(error)));
        s->params.reference_image_size=mode;
        s->next_step=s->total_steps;memset(s->text.values,0,s->text.tokens*s->text.width*2);
        h3_upscale_record u={0};
        for(size_t i=0;i<s->reference_count;i++)if(s->references[i].kind!=H3_LAYOUT_REF_AUDIO) {
            u.original_width[i]=128;u.original_height[i]=64;
            u.semantic_width[i]=s->references[i].latent_w*16;
            u.semantic_height[i]=s->references[i].latent_h*16;
        }
        CHECK(h3_upscale_capture(s,s->condition_video,s->condition_audio,&u,path,error,sizeof(error)));
        CHECK(!rename(original,hidden));
        h3_upscale_source *source=h3_upscale_source_load(path,error,sizeof(error));CHECK(source);
        h3_upscale_plan *plan=h3_upscale_plan_create(source,error,sizeof(error));CHECK(plan);
        CHECK(plan->references[0].latent_w==(mode?4:8));
        CHECK(plan->references[0].latent_h==(mode?2:4));
        CHECK(!rename(hidden,original));
        h3_upscale_plan_free(plan);h3_upscale_source_free(source);h3_sampler_state_free(s);
    }
}

static void upscale_sources(const char *directory,const char *path,const char *other) {
    h3_params options=H3_PARAMS_DEFAULT;
    CHECK(h3_upscale_options_valid(&options,error,sizeof(error)));
    options.state_only=1;CHECK(!h3_upscale_options_valid(&options,error,sizeof(error)));
    options.save_upscale_state=path;CHECK(h3_upscale_options_valid(&options,error,sizeof(error)));
    options.output_path=path;CHECK(!h3_upscale_options_valid(&options,error,sizeof(error)));
    options.output_path=NULL;options.cuda_denoise_quant=1;CHECK(!h3_upscale_options_valid(&options,error,sizeof(error)));
    for(int mode=0;mode<5;mode++) {
        int ref=mode==1;fixture_keyframes=mode>=2?mode-1:0;
        h3_sampler_state *s=fixture(directory,ref,0,1);
        memset(s->text.values,0,s->text.tokens*s->text.width*2);
        h3_upscale_record u={0};
        if(fixture_keyframes&1)u.keyframes[u.keyframe_count++]=0;
        if(fixture_keyframes&2)u.keyframes[u.keyframe_count++]=s->aligned_frames-1;
        for(size_t i=0;i<s->reference_count;i++)if(s->references[i].kind!=H3_LAYOUT_REF_AUDIO) {
            u.original_width[i]=128;u.original_height[i]=64;
            u.semantic_width[i]=s->references[i].latent_w*16;
            u.semantic_height[i]=s->references[i].latent_h*16;
        }
        CHECK(!h3_upscale_capture(s,s->condition_video,s->condition_audio,&u,path,error,sizeof(error)));
        s->next_step=s->total_steps;
        CHECK(h3_upscale_capture(s,s->condition_video,s->condition_audio,&u,path,error,sizeof(error)));
        CHECK(!h3_sampler_state_load(path,error,sizeof(error)));
        h3_upscale_source *a=h3_upscale_source_load(path,error,sizeof(error));CHECK(a);
        CHECK(a->state->upscale.stage==1&&!a->state->prepared.count&&!a->state->original_video_noise);
        CHECK(!memcmp(a->state->video,s->video,s->video_elements*4));
        CHECK(!memcmp(a->state->audio,s->audio,s->audio_elements*4));
        CHECK(!a->state->condition_video_elements||!memcmp(a->state->condition_video,s->condition_video,s->condition_video_elements*4));
        h3_upscale_plan *plan=h3_upscale_plan_create(a,error,sizeof(error));CHECK(plan);
        CHECK(plan->info.width==128&&plan->info.height==64&&plan->info.frames==s->aligned_frames);
        CHECK(plan->info.video_t==s->latent_t&&plan->info.audio_t==s->audio_t);
        CHECK(plan->info.video_elements==s->video_elements*4&&plan->info.audio_elements==s->audio_elements);
        h3_upscale_conditions conditions={0};
        CHECK(h3_upscale_retarget(a,plan,NULL,&conditions,NULL,NULL,error,sizeof(error)));
        CHECK(conditions.layout.prefix.video_prefix_t==0&&conditions.layout.prefix.audio_prefix_t==s->audio_t);
        CHECK(conditions.layout.img_target_rows==s->layout.img_target_rows*4);
        CHECK(conditions.layout.audio_target_rows==s->layout.audio_target_rows);
        CHECK(conditions.audio_elements==s->condition_audio_elements);
        if(conditions.audio_elements)CHECK(!memcmp(conditions.audio,s->condition_audio,conditions.audio_elements*4));
        if(ref) {
            CHECK(plan->references[0].latent_w==8&&plan->references[0].latent_h==4);
            CHECK(plan->references[2].latent_w==4&&plan->references[2].latent_h==2);
            CHECK(!memcmp(conditions.video+8*4*24,s->condition_video+4*2*24,2*4*2*24*4));
        }
        CHECK(!s->condition_video_elements||!memcmp(a->state->condition_video,s->condition_video,s->condition_video_elements*4));
        h3_upscale_conditions_free(&conditions);
        plan->info.width+=32;CHECK(!h3_upscale_retarget(a,plan,NULL,&conditions,NULL,NULL,error,sizeof(error)));
        h3_upscale_plan_free(plan);
        CHECK(h3_upscale_source_save(a,other,error,sizeof(error)));
        pid_t child=fork();CHECK(child>=0);
        if(!child) {
            struct rlimit limit={1024,1024};signal(SIGXFSZ,SIG_IGN);
            if(setrlimit(RLIMIT_FSIZE,&limit))_exit(2);
            _exit(h3_upscale_source_save(a,other,error,sizeof(error))?1:0);
        }
        int status=0;CHECK(waitpid(child,&status,0)==child);
        CHECK(WIFEXITED(status)&&WEXITSTATUS(status)==0);

        h3_upscale_source_free(a);
        CHECK(!unlink(path)); /* Bundle survives moving and deleting its parent/source. */
        a=h3_upscale_source_load(other,error,sizeof(error));CHECK(a);
        a->state->audio[0]=1;CHECK(!h3_upscale_source_save(a,path,error,sizeof(error)));
        CHECK(!access(other,F_OK));
        h3_upscale_source_free(a);
        size_t bytes;uint8_t *data=read_file(other,&bytes);
        FILE *file=fopen(path,"wb");CHECK(file);CHECK(fwrite(data,1,bytes-1,file)==bytes-1);CHECK(!fclose(file));
        CHECK(!h3_upscale_source_load(path,error,sizeof(error)));
        data[bytes-1]^=1;file=fopen(path,"wb");CHECK(file);CHECK(fwrite(data,1,bytes,file)==bytes);CHECK(!fclose(file));
        CHECK(!h3_upscale_source_load(path,error,sizeof(error)));free(data);
        if(ref){u.original_width[0]=0;CHECK(!h3_upscale_capture(s,s->condition_video,s->condition_audio,&u,path,error,sizeof(error)));}
        h3_sampler_state_free(s);
    }
    fixture_keyframes=0;
    /* Explicit current metadata-recipe import, never inferred from file suffix. */
    for(int conditioned=0;conditioned<2;conditioned++) {
        h3_sampler_state *s=fixture(directory,conditioned,0,1);
        memset(s->text.values,0,s->text.tokens*s->text.width*2);
        s->params._arithmetic_recipe=H3_SGLANG_VERSION;s->rng_version=2;
        CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        CHECK(!h3_upscale_source_import_sampler(path,error,sizeof(error)));
        s->next_step=s->total_steps;CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        h3_upscale_source *u=h3_upscale_source_import_sampler(path,error,sizeof(error));
        if(conditioned)CHECK(!u);
        else {CHECK(u&&u->state->upscale.stage==1);CHECK(!memcmp(u->state->audio,s->audio,s->audio_elements*4));h3_upscale_source_free(u);}
        s->params._arithmetic_recipe=0;s->rng_version=1;CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        CHECK(!h3_upscale_source_import_sampler(path,error,sizeof(error)));h3_sampler_state_free(s);
    }
}
static void upscale_geometry(const char *directory,const char *path) {
    h3_av_state_info info;
    CHECK(h3_av_state_shape(1344,768,90,&info));CHECK(info.video_t==27&&info.audio_t==150);
    CHECK(!h3_av_state_shape(1920,1088,90,&info));
    CHECK(h3_av_state_shape_profile(1920,1088,90,1,&info));CHECK(info.geometry_profile==1);
    CHECK(!h3_av_state_shape_profile(1952,1056,90,1,&info));
    CHECK(!h3_av_state_shape_profile(1920,1120,90,1,&info));
    CHECK(!h3_av_state_shape_profile(1920,1080,90,1,&info));
    CHECK(!h3_av_state_shape_profile(1920,1088,90,2,&info));
    uint8_t signature[32]={1};h3_av_state *av=h3_av_state_new_profile(1920,1088,22,1,42,signature);CHECK(av);
    memset(av->video,0,av->info.video_elements*4);memset(av->audio,0,av->info.audio_elements*4);
    CHECK(h3_av_state_save(av,path,error,sizeof(error)));
    h3_av_state *loaded=h3_av_state_load(path,error,sizeof(error));CHECK(loaded&&loaded->info.geometry_profile==1);
    h3_av_state *copy=h3_av_state_clone(loaded);CHECK(copy&&copy->info.geometry_profile==1);
    h3_denoise_prefix prefix;CHECK(!h3_av_state_validate_continuation(av,1920,1088,90,39,signature,&prefix,error,sizeof(error)));
    h3_result r={.av_state=av,.presentation={.version=9,.render_width=1920,.render_height=1088,.width=1920,.height=1088,
        .fps=24,.sample_rate=32000,.codec_version=1,.geometry_profile=1,.av_metadata_identity=1}};
    CHECK(h3_result_save_av_state(&r,path,error,sizeof(error)));
    h3_presentation pres;CHECK(h3_presentation_load(path,av,&pres,error,sizeof(error))==1);
    CHECK(pres.version==9&&pres.geometry_profile==1&&pres.av_metadata_identity==1);
    pres.geometry_profile=0;CHECK(!h3_presentation_validate(&pres,&av->info,error,sizeof(error)));
    char sidecar[1100];snprintf(sidecar,sizeof(sidecar),"%s.presentation",path);CHECK(!unlink(sidecar));
    h3_av_state_free(av);h3_av_state_free(loaded);h3_av_state_free(copy);
    fixture_width=1920;fixture_height=1088;fixture_frames=22;fixture_profile=1;
    h3_sampler_state *s=fixture(directory,0,0,1);CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
    h3_sampler_state *q=h3_sampler_state_load(path,error,sizeof(error));CHECK(q&&q->params.geometry_profile==1);
    q->params.geometry_profile=0;CHECK(!h3_sampler_state_validate(q,error,sizeof(error)));
    q->params.geometry_profile=1;q->params.denoise_reuse=2;CHECK(!h3_sampler_state_validate(q,error,sizeof(error)));
    h3_sampler_state_free(q);h3_sampler_state_free(s);
    fixture_width=960;fixture_height=544;fixture_frames=90;fixture_profile=0;
    s=fixture(directory,0,0,1);memset(s->text.values,0,s->text.tokens*s->text.width*2);s->next_step=s->total_steps;
    CHECK(h3_upscale_capture(s,NULL,NULL,&(h3_upscale_record){0},path,error,sizeof(error)));
    h3_upscale_source *source=h3_upscale_source_load(path,error,sizeof(error));CHECK(source);
    h3_upscale_plan *plan=h3_upscale_plan_create(source,error,sizeof(error));CHECK(plan);
    CHECK(plan->info.width==1920&&plan->info.height==1088&&plan->info.geometry_profile==1);
    CHECK(plan->info.sequence_rows==55080+300+3&&plan->info.video_t==27&&plan->info.audio_t==150);
    h3_upscale_plan_free(plan);h3_upscale_source_free(source);h3_sampler_state_free(s);
    fixture_width=64;fixture_height=32;fixture_frames=56;fixture_profile=0;
    float input[24*4];for(int c=0;c<24;c++)for(int i=0;i<4;i++)input[c*4+i]=(float)c;
    float *v=h3_upscale_bilinear(input,1,2,2,4,4,error,sizeof(error));CHECK(v);
    for (int c = 0; c < 24; c++)
        for (int i = 0; i < 16; i++)
            CHECK(v[c * 16 + i] == (float)c);
    free(v);
}
static const char *refinement_fixture_path;
static void upscale_refinement(const char *directory,const char *path,const char *other) {
    h3_sigma_schedule sigmas;
    CHECK(h3_upscale_schedule(0,0,&sigmas));CHECK(!h3_upscale_schedule(0,.25f,&sigmas));
    CHECK(!h3_upscale_schedule(1,.25f,&sigmas));CHECK(!h3_upscale_schedule(5,.25f,&sigmas));
    CHECK(!h3_upscale_schedule(4,0,&sigmas));CHECK(!h3_upscale_schedule(4,NAN,&sigmas));
    CHECK(!h3_upscale_schedule(4,INFINITY,&sigmas));CHECK(!h3_upscale_schedule(4,.51f,&sigmas));
    CHECK(!h3_upscale_schedule(4,0x1p-149f,&sigmas));
    CHECK(h3_upscale_schedule(4,.25f,&sigmas));
    const uint32_t expected[]={0x3e800000,0x3e4bab23,0x3e109091,0x3d9a90e8,0};
    /* Fixed recipe bytes independently computed from the rational flow curve. */
    CHECK(!memcmp(sigmas.video,expected,sizeof(expected)));
    h3_upscale_options o=H3_UPSCALE_OPTIONS_DEFAULT;CHECK(h3_upscale_request_valid(&o,error,sizeof(error)));
    o.refine_steps=0;o.sigma_set=1;CHECK(!h3_upscale_request_valid(&o,error,sizeof(error)));
    o.sigma_set=0;CHECK(h3_upscale_request_valid(&o,error,sizeof(error)));
    const char *dirs[]={"FL2VA/transformer","FL2VA/video_vae","FL2VA/video_vae/source","FL2VA/audio_vae"};
    const char *files[]={"transformer/config.json","video_vae/config.json","video_vae/source/config.json",
        "audio_vae/config.json","audio_vae/config.yaml","audio_vae/metadata.json","video_vae/source/model.safetensors","audio_vae/model.safetensors"};
    char filename[1200];
    for(size_t i=0;i<4;i++){snprintf(filename,sizeof(filename),"%s/%s",directory,dirs[i]);CHECK(!mkdir(filename,0700));}
    for(size_t i=0;i<8;i++){snprintf(filename,sizeof(filename),"%s/FL2VA/%s",directory,files[i]);write_file(filename,"test metadata only");}
    h3_sampler_state *base=fixture(directory,0,0,1);base->next_step=base->total_steps;
    memset(base->text.values,0,base->text.tokens*base->text.width*2);
    CHECK(h3_upscale_capture(base,NULL,NULL,&(h3_upscale_record){0},path,error,sizeof(error)));
    h3_upscale_source *source=h3_upscale_source_load(path,error,sizeof(error));CHECK(source);
    h3_upscale_plan *plan=h3_upscale_plan_create(source,error,sizeof(error));CHECK(plan);
    h3_upscale_transfer *transfer=h3_upscale_transfer_create(source,plan,NULL,2,NULL,NULL,error,sizeof(error));CHECK(transfer);
    h3_device_info device={.apple_gpu_family=9,.metal4=1,.unified_memory=1};snprintf(device.backend,sizeof(device.backend),"metal");
    uint8_t noise[32]={0};h3_sampler_state *shared=NULL;
    for(int k=2;k<=4;k++) {
        o=(h3_upscale_options)H3_UPSCALE_OPTIONS_DEFAULT;o.refine_steps=k;
        transfer->noise_source=shared;
        h3_sampler_state *s=h3_upscale_initialize(source,plan,transfer,&o,directory,&device,error,sizeof(error));CHECK(s);
        CHECK(s->upscale.stage==2&&!s->next_step&&s->audio_random_count==0&&s->video_random_count==s->video_elements);
        CHECK(!memcmp(s->audio,source->state->audio,s->audio_elements*4));
        CHECK(s->layout.frozen_audio&&s->layout.prefix.audio_prefix_t==s->audio_t&&s->sigmas.audio[0]==0);
        if(k==2)memcpy(noise,s->upscale.noise_hash,32);else CHECK(!memcmp(noise,s->upscale.noise_hash,32));
        CHECK(h3_sampler_state_save(s,other,error,sizeof(error)));
        if(k==4&&refinement_fixture_path)CHECK(h3_sampler_state_save(s,refinement_fixture_path,error,sizeof(error)));
        h3_sampler_state *r=h3_sampler_state_load(other,error,sizeof(error));CHECK(r);
        if(!shared){shared=h3_sampler_state_load(other,error,sizeof(error));CHECK(shared);}
        CHECK(!memcmp(&s->video_rng,&shared->video_rng,sizeof(s->video_rng)));
        CHECK(r->upscale.stage==2&&r->layout.frozen_audio&&r->total_steps==k);
        CHECK(!memcmp(r->video,s->video,s->video_elements*4)&&!memcmp(r->audio,s->audio,s->audio_elements*4));
        uint8_t key[32],changed[32];CHECK(h3_sampler_prepared_key(r,key));r->upscale.parent_hash[0]^=1;
        CHECK(h3_sampler_prepared_key(r,changed));CHECK(memcmp(key,changed,32));r->upscale.parent_hash[0]^=1;
        r->audio[0]=1;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->audio[0]=s->audio[0];
        r->original_video_noise[0]+=1;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->original_video_noise[0]=s->original_video_noise[0];
        r->text.values[0]^=1;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->text.values[0]^=1;
        r->sigmas.audio[0]=.1f;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));r->sigmas.audio[0]=0;
        r->sigmas.video[1]*=.5f;CHECK(!h3_sampler_state_validate(r,error,sizeof(error)));
        h3_sampler_state_free(r);h3_sampler_state_free(s);
    }
    transfer->noise_source=shared;shared->params.seed++;
    CHECK(!h3_upscale_initialize(source,plan,transfer,&o,directory,&device,error,sizeof(error)));
    transfer->noise_source=NULL;h3_sampler_state_free(shared);
    size_t admitted=plan->info.video_elements;plan->info.video_elements=SIZE_MAX/4;
    CHECK(!h3_upscale_initialize(source,plan,transfer,&o,directory,&device,error,sizeof(error)));
    plan->info.video_elements=admitted;
    h3_sampler_state *recovered=h3_upscale_initialize(source,plan,transfer,&o,directory,&device,error,sizeof(error));
    CHECK(recovered&&h3_upscale_audio_intact(recovered));h3_sampler_state_free(recovered);
    h3_upscale_transfer_free(transfer);h3_upscale_plan_free(plan);h3_upscale_source_free(source);h3_sampler_state_free(base);
    for(size_t i=0;i<8;i++){snprintf(filename,sizeof(filename),"%s/FL2VA/%s",directory,files[i]);CHECK(!unlink(filename));}
    for(int i=3;i>=0;i--){snprintf(filename,sizeof(filename),"%s/%s",directory,dirs[i]);CHECK(!rmdir(filename));}
}
static void extensions(const char *directory,const char *path,const char *other) {
    upscale_refinement(directory,path,other);
    upscale_geometry(directory,path);
    upscale_sources(directory,path,other);
    warmup_extensions(directory,path);
    adaptive_continuation_extensions(directory,path);
    subblock_reference_state(directory,path);
    h3_params defaults=H3_PARAMS_DEFAULT;
    CHECK(defaults.cuda_denoise_quant==H3_QUANT_OFF&&!defaults.cuda_denoise_quant_cache&&!defaults.cuda_denoise_quant_set);
    CHECK(h3_quant_projection("blocks.0.attn.qkv_proj.weight",H3_QUANT_FP8)==H3_QUANT_FP8);
    CHECK(h3_quant_projection("blocks.49.attn.out_proj.weight",H3_QUANT_NVFP4)==H3_QUANT_NVFP4);
    CHECK(h3_quant_projection("blocks.12.mlp.fc1.weight",H3_QUANT_FP8)==H3_QUANT_FP8);
    CHECK(h3_quant_projection("blocks.12.mlp.fc2.weight",H3_QUANT_NVFP4)==H3_QUANT_NVFP4);
    CHECK(h3_quant_projection("blocks.0.attn.qkv_proj.bias",H3_QUANT_FP8)==H3_QUANT_OFF);
    CHECK(h3_quant_projection("text_refiner.blocks.0.mlp.fc1.weight",H3_QUANT_FP8)==H3_QUANT_OFF);
    CHECK(h3_quant_projection("blocks.0.adaln.weight",H3_QUANT_NVFP4)==H3_QUANT_OFF);
    CHECK(h3_quant_projection("blocks..mlp.fc2.weight",H3_QUANT_FP8)==H3_QUANT_OFF);
    h3_quant_scope initial=h3_quant_exchange((h3_quant_scope){H3_QUANT_FP8,"cache-a"});
    CHECK(initial.mode==H3_QUANT_OFF);
    CHECK(h3_quant_current().mode==H3_QUANT_FP8&&!strcmp(h3_quant_current().cache,"cache-a"));
    CHECK(h3_quant_exchange(initial).mode==H3_QUANT_FP8&&h3_quant_current().mode==H3_QUANT_OFF);
    /* Default files omit the new required extension. Fast files require a
     * supported implementation and change the prepared-object key. */
    h3_sampler_state *fast=fixture(directory,1,0,1);
    uint8_t default_key[32],fast_key[32];
    CHECK(h3_sampler_prepared_key(fast,default_key));
    CHECK(h3_sampler_state_save(fast,path,error,sizeof(error)));
    h3_sampler_state *roundtrip=h3_sampler_state_load(path,error,sizeof(error)); CHECK(roundtrip);
    h3_sampler_state_free(roundtrip);
    /* The new non-fast reference recipe owns a required extension and a
     * distinct prepared-object identity. Returning to legacy restores its key. */
    fast->params._arithmetic_recipe=H3_SGLANG_VERSION;fast->rng_version=2;
    CHECK(h3_sampler_state_validate(fast,error,sizeof(error)));
    CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(memcmp(default_key,fast_key,32));
    CHECK(h3_sampler_state_save(fast,other,error,sizeof(error)));
    roundtrip=h3_sampler_state_load(other,error,sizeof(error));CHECK(roundtrip);
    CHECK(roundtrip->params._arithmetic_recipe==H3_SGLANG_VERSION && roundtrip->rng_version==2);
    h3_sampler_state_free(roundtrip);
    fast->params.preview_vae=1;CHECK(h3_sampler_state_validate(fast,error,sizeof(error)));fast->params.preview_vae=0;
    fast->params._arithmetic_recipe=1;CHECK(!h3_sampler_state_validate(fast,error,sizeof(error)));
    fast->params._arithmetic_recipe=2;CHECK(!h3_sampler_state_validate(fast,error,sizeof(error)));
    fast->params._arithmetic_recipe=3;CHECK(!h3_sampler_state_validate(fast,error,sizeof(error)));
    fast->params._arithmetic_recipe=H3_SGLANG_VERSION+1;CHECK(!h3_sampler_state_validate(fast,error,sizeof(error)));
    fast->params._arithmetic_recipe=H3_SGLANG_VERSION;fast->sampler_mode=1;CHECK(!h3_sampler_state_validate(fast,error,sizeof(error)));
    fast->sampler_mode=0;fast->params._arithmetic_recipe=0;fast->rng_version=1;
    CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(!memcmp(default_key,fast_key,32));
    fast->params.backend=H3_BACKEND_METAL;fast->metal_attention_version=H3_METAL_ATTENTION_VERSION;
    fast->params.metal_attention=(h3_metal_attention_options)H3_METAL_ATTENTION_DEFAULT;
    fast->params.metal_attention.tau=1.25f;fast->params.metal_attention.q_block=64;
    fast->params.metal_attention.candidate=1;fast->params.metal_attention.kv_block=128;
    fast->params.metal_attention.dense_layers=2;fast->params.metal_attention.local_radius=3;
    fast->params.metal_attention.min_exact=.25f;
    fast->params.metal_attention.dense_steps=2;fast->params.metal_attention.dense_sigma=.9f;
    fast->params.attention_mode=H3_ATTN_SOL;
    CHECK(h3_sampler_state_validate(fast,error,sizeof(error)));
    CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(memcmp(default_key,fast_key,32));
    h3_metal_attention_options saved_metal=fast->params.metal_attention;
    uint8_t changed_key[32];
    for(int field=0;field<9;field++) {
        fast->params.metal_attention=saved_metal;
        switch(field) {
            case 0:fast->params.metal_attention.candidate=0;break;
            case 1:fast->params.metal_attention.q_block=32;break;
            case 2:fast->params.metal_attention.kv_block=64;break;
            case 3:fast->params.metal_attention.dense_layers=1;break;
            case 4:fast->params.metal_attention.local_radius=1;break;
            case 5:fast->params.metal_attention.tau=1;break;
            case 6:fast->params.metal_attention.min_exact=.5f;break;
            case 7:fast->params.metal_attention.dense_steps=1;break;
            case 8:fast->params.metal_attention.dense_sigma=.8f;break;
        }
        CHECK(h3_sampler_state_validate(fast,error,sizeof(error)));
        CHECK(h3_sampler_prepared_key(fast,changed_key));CHECK(memcmp(fast_key,changed_key,32));
    }
    fast->params.metal_attention=saved_metal;
    CHECK(h3_sampler_state_save(fast,path,error,sizeof(error)));
    roundtrip=h3_sampler_state_load(path,error,sizeof(error));CHECK(roundtrip);
    CHECK(roundtrip->params.backend==H3_BACKEND_METAL&&roundtrip->metal_attention_version==H3_METAL_ATTENTION_VERSION);
    CHECK(h3_metal_options_equal(roundtrip->params.metal_attention,fast->params.metal_attention));
    CHECK(roundtrip->params.attention_mode==H3_ATTN_SOL);
    uint8_t weight_bf16_key[32],weight_q8_key[32];
    CHECK(h3_sampler_prepared_key(roundtrip,weight_bf16_key));
    roundtrip->params.metal_attention.weight_format=H3_WEIGHT_Q8;
    CHECK(h3_sampler_prepared_key(roundtrip,weight_q8_key));
    CHECK(memcmp(weight_bf16_key,weight_q8_key,32));
    CHECK(!h3_metal_options_equal(roundtrip->params.metal_attention,fast->params.metal_attention));
    CHECK(h3_sampler_state_save(roundtrip,path,error,sizeof(error)));
    h3_sampler_state *q8_roundtrip=h3_sampler_state_load(path,error,sizeof(error));CHECK(q8_roundtrip);
    CHECK(q8_roundtrip->params.metal_attention.weight_format==H3_WEIGHT_Q8);
    CHECK(h3_metal_options_equal(q8_roundtrip->params.metal_attention,roundtrip->params.metal_attention));
    q8_roundtrip->params.metal_attention.q8_kernel=1;
    CHECK(h3_sampler_prepared_key(q8_roundtrip,weight_bf16_key));CHECK(memcmp(weight_bf16_key,weight_q8_key,32));
    CHECK(h3_sampler_state_save(q8_roundtrip,path,error,sizeof(error)));
    h3_sampler_state *q8_native=h3_sampler_state_load(path,error,sizeof(error));CHECK(q8_native);
    CHECK(q8_native->params.metal_attention.q8_kernel==1);h3_sampler_state_free(q8_native);
    q8_roundtrip->params.metal_attention.weight_format=(h3_weight_format)2;
    CHECK(!h3_sampler_state_validate(q8_roundtrip,error,sizeof(error)));
    q8_roundtrip->params.metal_attention.weight_format=H3_WEIGHT_Q8;
    q8_roundtrip->params.ssd_streaming=1;CHECK(!h3_sampler_state_validate(q8_roundtrip,error,sizeof(error)));
    q8_roundtrip->params.ssd_streaming=0;q8_roundtrip->params.backend=H3_BACKEND_MPSGRAPH_REFERENCE;
    CHECK(!h3_sampler_state_validate(q8_roundtrip,error,sizeof(error)));
    h3_sampler_state_free(q8_roundtrip);
    roundtrip->params.metal_attention.weight_format=H3_WEIGHT_BF16;
    roundtrip->params.metal_attention.tau=NAN;CHECK(!h3_sampler_state_validate(roundtrip,error,sizeof(error)));
    roundtrip->params.metal_attention=fast->params.metal_attention;
    roundtrip->metal_attention_version++;CHECK(!h3_sampler_state_validate(roundtrip,error,sizeof(error)));
    h3_sampler_state_free(roundtrip);
    /* New precision/tier fields round-trip and change both prepared identities.
     * A mixed/SOL recipe is not silently accepted as dense FP16. */
    fast->params.attention_mode=H3_ATTN_DENSE;
    uint8_t bf16_key[32],fp16_key[32];
    CHECK(h3_sampler_prepared_key(fast,bf16_key));
    fast->params.metal_attention.precision=1;
    CHECK(h3_sampler_prepared_key(fast,fp16_key));CHECK(memcmp(bf16_key,fp16_key,32));
    fast->params.metal_attention.layout_fusion=1;
    CHECK(h3_sampler_prepared_key(fast,changed_key));CHECK(memcmp(fp16_key,changed_key,32));
    h3_metal_attention_options adapter=fast->params.metal_attention;adapter.layout_fusion=0;
    CHECK(!h3_metal_options_equal(adapter,fast->params.metal_attention));
    fast->params.metal_attention.tier=1;
    CHECK(h3_sampler_state_validate(fast,error,sizeof(error)));
    CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(memcmp(fp16_key,fast_key,32));
    CHECK(h3_sampler_state_save(fast,path,error,sizeof(error)));
    roundtrip=h3_sampler_state_load(path,error,sizeof(error));CHECK(roundtrip);
    CHECK(roundtrip->params.metal_attention.precision==1&&roundtrip->params.metal_attention.tier==1&&roundtrip->params.metal_attention.layout_fusion==1);
    /* Runtime ANE decisions survive checkpoints, but immutable conditioning
     * keys depend only on the execution recipe, not adaptation history. */
    int saved_width=fixture_width,saved_height=fixture_height,saved_frames=fixture_frames;
    fixture_width=640;fixture_height=480;fixture_frames=243;
    h3_sampler_state *ane_state=fixture(directory,1,0,1);
    fixture_width=saved_width;fixture_height=saved_height;fixture_frames=saved_frames;
    ane_state->params.backend=H3_BACKEND_METAL;ane_state->metal_attention_version=H3_METAL_ATTENTION_VERSION;
    ane_state->params.attention_mode=roundtrip->params.attention_mode;
    ane_state->params.metal_attention=roundtrip->params.metal_attention;
    ane_state->params.metal_attention.ane_mode=3;
    ane_state->params.metal_attention.ane_rows=4096;
    ane_state->params.metal_attention.ane_chunk=512;
    uint8_t ane_key[32];
    CHECK(h3_sampler_prepared_key(ane_state,ane_key));
    CHECK(!h3_sampler_state_validate(ane_state,error,sizeof(error)));
    ane_state->ane_decided=(1ull<<50)-1;ane_state->ane_rows[1]=4096;ane_state->ane_rows[49]=512;
    CHECK(h3_sampler_prepared_key(ane_state,changed_key));CHECK(!memcmp(ane_key,changed_key,32));
    CHECK(h3_sampler_state_save(ane_state,path,error,sizeof(error)));
    h3_sampler_state *ane_roundtrip=h3_sampler_state_load(path,error,sizeof(error));CHECK(ane_roundtrip);
    CHECK(ane_roundtrip->ane_decided==ane_state->ane_decided&&!memcmp(ane_roundtrip->ane_rows,ane_state->ane_rows,sizeof(ane_state->ane_rows)));
    CHECK(h3_metal_options_equal(ane_roundtrip->params.metal_attention,ane_state->params.metal_attention));
    ane_roundtrip->ane_decided|=1ull<<51;CHECK(!h3_sampler_state_validate(ane_roundtrip,error,sizeof(error)));
    ane_roundtrip->ane_decided=ane_state->ane_decided;ane_roundtrip->ane_rows[1]=513;CHECK(!h3_sampler_state_validate(ane_roundtrip,error,sizeof(error)));
    ane_roundtrip->params.metal_attention.ane_rows=16384;ane_roundtrip->ane_rows[1]=16384;
    CHECK(!h3_sampler_state_validate(ane_roundtrip,error,sizeof(error)));
    h3_sampler_state_free(ane_roundtrip);h3_sampler_state_free(ane_state);
    roundtrip->params.metal_attention.ane_mode=0;roundtrip->ane_decided=0;memset(roundtrip->ane_rows,0,sizeof(roundtrip->ane_rows));
    roundtrip->params.metal_attention.tier=2;
    CHECK(h3_sampler_prepared_key(roundtrip,changed_key));CHECK(memcmp(fast_key,changed_key,32));
    roundtrip->params.attention_mode=H3_ATTN_SOL;CHECK(h3_sampler_state_validate(roundtrip,error,sizeof(error)));
    CHECK(h3_sampler_prepared_key(roundtrip,changed_key));CHECK(memcmp(fast_key,changed_key,32));
    CHECK(h3_sampler_state_save(roundtrip,path,error,sizeof(error)));
    h3_sampler_state *sol_roundtrip=h3_sampler_state_load(path,error,sizeof(error));CHECK(sol_roundtrip);
    CHECK(h3_metal_options_equal(sol_roundtrip->params.metal_attention,roundtrip->params.metal_attention));
    CHECK(!strcmp(h3_metal_sol_dense_reason(sol_roundtrip->params.metal_attention,2,1,.5f,.5f),"early-evaluation"));
    CHECK(!h3_metal_sol_dense_reason(sol_roundtrip->params.metal_attention,2,2,.5f,.5f));
    int resumed=sol_roundtrip->next_step;
    CHECK(resumed==roundtrip->next_step&&resumed>=sol_roundtrip->params.metal_attention.dense_steps);
    CHECK(!strcmp(h3_metal_sol_dense_reason(sol_roundtrip->params.metal_attention,2,resumed,
        sol_roundtrip->sigmas.video[resumed],sol_roundtrip->sigmas.audio[resumed]),"high-noise"));
    sol_roundtrip->params.metal_attention.dense_sigma=-1;
    CHECK(h3_sampler_state_save(sol_roundtrip,path,error,sizeof(error)));
    h3_sampler_state *resumed_sol=h3_sampler_state_load(path,error,sizeof(error));CHECK(resumed_sol);
    CHECK(!h3_metal_sol_dense_reason(resumed_sol->params.metal_attention,2,resumed_sol->next_step,
        resumed_sol->sigmas.video[resumed_sol->next_step],resumed_sol->sigmas.audio[resumed_sol->next_step]));
    CHECK(!strcmp(h3_metal_sol_dense_reason(resumed_sol->params.metal_attention,2,0,
        resumed_sol->sigmas.video[0],resumed_sol->sigmas.audio[0]),"early-evaluation"));
    h3_sampler_state_free(resumed_sol);
    h3_sampler_state_free(sol_roundtrip);
    h3_sampler_state_free(roundtrip);
    fast->params.backend=H3_BACKEND_MPSGRAPH_REFERENCE;fast->metal_attention_version=0;
    fast->params.attention_mode=H3_ATTN_DENSE;
    fast->params.metal_attention=(h3_metal_attention_options)H3_METAL_ATTENTION_DEFAULT;
    CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(!memcmp(default_key,fast_key,32));
    for(int mode=H3_QUANT_FP8;mode<=H3_QUANT_NVFP4;mode++) {
        fast->params.cuda_denoise_quant=mode;fast->quant_version=H3_QUANT_VERSION;
        CHECK(h3_sampler_state_validate(fast,error,sizeof(error)));
        CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(memcmp(default_key,fast_key,32));
        CHECK(h3_sampler_state_save(fast,path,error,sizeof(error)));
        roundtrip=h3_sampler_state_load(path,error,sizeof(error));CHECK(roundtrip);
        CHECK(roundtrip->params.cuda_denoise_quant==mode&&roundtrip->quant_version==H3_QUANT_VERSION);
        CHECK(!roundtrip->params.cuda_denoise_quant_cache);
        roundtrip->quant_version++;CHECK(!h3_sampler_state_validate(roundtrip,error,sizeof(error)));
        h3_sampler_state_free(roundtrip);
    }
    fast->params.cuda_denoise_quant=0;fast->quant_version=0;
    CHECK(defaults.cuda_attention==0&&!defaults.cuda_attention_set);
    int parsed=-1;CHECK(!h3_attention_parse("fp8",&parsed));CHECK(!h3_attention_options(5,error,sizeof(error)));
    CHECK(h3_attention_parse("subblock",&parsed)&&parsed==H3_ATTENTION_SUBBLOCK);
    CHECK(h3_attention_exchange(1)==0);CHECK(h3_attention_current()==1);
    CHECK(h3_attention_exchange(2)==1);CHECK(h3_attention_exchange(0)==2);
    for(int mode=1;mode<=4;mode++) {
        CHECK(h3_attention_parse(h3_attention_name(mode),&parsed)&&parsed==mode);
        fast->params.cuda_attention=mode;fast->attention_version=H3_ATTENTION_VERSION;fast->attention_plan=H3_ATTENTION_PLAN_VERSION;
        CHECK(h3_sampler_state_validate(fast,error,sizeof(error)));
        CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(memcmp(default_key,fast_key,32));
        CHECK(h3_sampler_state_save(fast,path,error,sizeof(error)));
        roundtrip=h3_sampler_state_load(path,error,sizeof(error));CHECK(roundtrip);
        CHECK(roundtrip->params.cuda_attention==mode&&roundtrip->attention_version==H3_ATTENTION_VERSION&&roundtrip->attention_plan==H3_ATTENTION_PLAN_VERSION);
        if(mode==H3_ATTENTION_SOL) {
            h3_cuda_sol_options original=roundtrip->params.cuda_sol;uint8_t sol_key[32],changed[32];
            CHECK(h3_sampler_prepared_key(roundtrip,sol_key));
            for(int field=0;field<8;field++) {
                roundtrip->params.cuda_sol=original;
                switch(field){case 0:roundtrip->params.cuda_sol.q_block=64;break;
                case 1:roundtrip->params.cuda_sol.dense_layers++;break;case 2:roundtrip->params.cuda_sol.dense_steps++;break;
                case 3:roundtrip->params.cuda_sol.local_radius++;break;case 4:roundtrip->params.cuda_sol.tau+=.25f;break;
                case 5:roundtrip->params.cuda_sol.min_exact=.5f;break;case 6:roundtrip->params.cuda_sol.dense_sigma=.8f;break;
                case 7:roundtrip->params.cuda_sol.q_block=64;roundtrip->params.cuda_sol.tau=2;break;}
                CHECK(h3_sampler_prepared_key(roundtrip,changed));CHECK(memcmp(sol_key,changed,32));
            }
            roundtrip->params.cuda_sol=original;
        }
        roundtrip->attention_plan++;CHECK(!h3_sampler_state_validate(roundtrip,error,sizeof(error)));roundtrip->attention_plan--;
        roundtrip->attention_version++;CHECK(!h3_sampler_state_validate(roundtrip,error,sizeof(error)));
        h3_sampler_state_free(roundtrip);
    }
    fast->params.cuda_attention=0;fast->attention_version=0;fast->attention_plan=0;
    CHECK(h3_sampler_prepared_key(fast,fast_key));CHECK(!memcmp(default_key,fast_key,32));

    h3_sampler_state_free(fast);
    h3_sampler_state *pipeline=fixture(directory,1,0,1);
    CHECK(pipeline->refvideo_pipeline==H3_REFVIDEO_RELEASED_V1);
    CHECK(h3_sampler_state_save(pipeline,path,error,sizeof(error)));
    h3_sampler_state *loaded=h3_sampler_state_load(path,error,sizeof(error));CHECK(loaded);
    CHECK(loaded->refvideo_pipeline==H3_REFVIDEO_RELEASED_V1);
    loaded->refvideo_pipeline=99;CHECK(!h3_sampler_state_validate(loaded,error,sizeof(error)));
    h3_sampler_state_free(loaded);h3_sampler_state_free(pipeline);
    for(int gpu=0;gpu<2;gpu++) for(int core=0;core<2;core++) for(int reduced=0;reduced<2;reduced++) {
        h3_sampler_state *s=fixture(directory,0,0,core?1:3);
        s->sampler_mode=(uint32_t)gpu; s->full_sequence=s->layout.seq_len;
        s->params.core_reuse=core?4:1; s->core_forward_count=core?4:2;
        if(gpu && core) { s->last_evaluated=3; s->previous_evaluated=2; }
        if(gpu && !core) {
            size_t counts[]={s->video_elements,s->video_elements,s->audio_elements,s->audio_elements};
            uint16_t **arrays[]={&s->gpu_last_video,&s->gpu_previous_video,&s->gpu_last_audio,&s->gpu_previous_audio};
            for(int j=0;j<4;j++) { *arrays[j]=malloc(counts[j]*2); CHECK(*arrays[j]);
                for(size_t i=0;i<counts[j];i++) (*arrays[j])[i]=(uint16_t)(i*131+(size_t)j); }
        }
        if(core) {
            s->core_residual_ready=1; s->core_rows=s->layout.seq_len; s->core_columns=5376;
            s->core_elements=s->core_rows*s->core_columns; s->core_residual=malloc(s->core_elements*2); CHECK(s->core_residual);
            for(size_t i=0;i<s->core_elements;i++) s->core_residual[i]=(uint16_t)(i*7);
        }
        if(reduced) {
            s->params.token_reduction=1; s->reduction_enabled=1; s->reduction_begin=4; s->reduction_end=30;
            s->reduction_early_steps=10; s->reduction_early_end=40; s->reduction_scale=1;
            s->reduced_sequence=s->layout.seq_len-s->layout.img_target_rows+(size_t)s->latent_t*(s->latent_h/2)*((s->latent_w/2+1)/2);
        }
        s->prepared.version=1; s->prepared.count=2;
        CHECK(h3_sampler_prepared_key(s,s->prepared.key));
        for(int i=0;i<2;i++) {
            h3_prepared_tensor *t=&s->prepared.tensors[i]; t->id=(uint32_t)i+1; t->elements=513; t->values=malloc(1026); CHECK(t->values);
            for(size_t j=0;j<t->elements;j++) t->values[j]=(uint16_t)(j*3277);
        }
        s->resume_count=2; s->resume_step=2; s->resume_format=H3_SAMPLE_VERSION; memset(s->resume_hash,31,32);
        CHECK(h3_sampler_state_validate(s,error,sizeof(error))); CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error)); CHECK(r);
        CHECK(h3_sampler_state_save(r,other,error,sizeof(error)));
        size_t na,nb; uint8_t *a=read_file(path,&na),*b=read_file(other,&nb); CHECK(na==nb && !memcmp(a,b,na)); free(a); free(b);
        CHECK(r->sampler_mode==s->sampler_mode && r->resume_count==2 && !memcmp(r->resume_hash,s->resume_hash,32));
        if(core) CHECK(!memcmp(r->core_residual,s->core_residual,s->core_elements*2));
        if(gpu && !core) { CHECK(!memcmp(r->gpu_last_video,s->gpu_last_video,s->video_elements*2)); CHECK(!memcmp(r->gpu_previous_audio,s->gpu_previous_audio,s->audio_elements*2)); }
        CHECK(!memcmp(r->prepared.tensors[0].values,s->prepared.tensors[0].values,1026));
        uint8_t key[32]; CHECK(h3_sampler_prepared_key(r,key)); CHECK(!memcmp(key,s->prepared.key,32));
        r->next_step=10; CHECK(h3_sampler_prepared_key(r,key)); CHECK(!memcmp(key,s->prepared.key,32)); r->next_step=4;
        r->text.values[0]^=1; CHECK(h3_sampler_prepared_key(r,key)); CHECK(memcmp(key,s->prepared.key,32)); r->text.values[0]^=1;
        r->reduction_active=1; CHECK(!h3_sampler_state_validate(r,error,sizeof(error))); r->reduction_active=0;
        if(core) { r->core_forward_count++; CHECK(!h3_sampler_state_validate(r,error,sizeof(error))); r->core_forward_count--; }
        h3_prepared_cache_free(&r->prepared); CHECK(h3_sampler_state_save(r,other,error,sizeof(error)));
        h3_sampler_state_free(r); h3_sampler_state_free(s);
    }
    setenv("H3_REUSE_STEPS","0,2,5,9,19",1);
    h3_sampler_state *s=fixture(directory,0,0,3); CHECK(s->selected[2] && !s->selected[3] && s->last_evaluated==2);
    CHECK(h3_sampler_state_save(s,path,error,sizeof(error))); h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error)); CHECK(r);
    CHECK(!memcmp(s->selected,r->selected,20)); h3_sampler_state_free(r); h3_sampler_state_free(s);
    uint8_t mask[H3_MAX_STEPS]; setenv("H3_REUSE_STEPS","0,4,3,19",1); CHECK(h3_dit_resolve_reuse(20,3,mask)<0);
    unsetenv("H3_REUSE_STEPS");
}

int main(int argc,char **argv) {
    if(argc==3&&!strcmp(argv[1],"--adaptive-fixture"))adaptive_fixture_path=argv[2];
    if(argc==3&&!strcmp(argv[1],"--adaptive-reference-fixture"))adaptive_reference_fixture_path=argv[2];
    if(argc==3&&!strcmp(argv[1],"--refinement-fixture"))refinement_fixture_path=argv[2];
    reference_av_metadata();
    if(argc==3 && !strcmp(argv[1],"--load-upscale")) {
        h3_upscale_source *s=h3_upscale_source_load(argv[2],error,sizeof(error));
        if(!s){fprintf(stderr,"%s\n",error);return 1;}
        h3_upscale_source_free(s);return 0;
    }
    if(argc==3 && !strcmp(argv[1],"--load")) {
        h3_sampler_state *s=h3_sampler_state_load(argv[2],error,sizeof(error));
        if(!s) { fprintf(stderr,"%s\n",error); return 1; }
        h3_sampler_state_free(s); return 0;
    }
    char directory[512];
    CHECK(snprintf(directory,sizeof(directory),"%s/h3-sampler-tests-XXXXXX",getenv("TMPDIR")?getenv("TMPDIR"):"/tmp")<(int)sizeof(directory));
    CHECK(mkdtemp(directory));
    char path[1024],other[1024];
    snprintf(path,sizeof(path),"%s/FL2VA",directory); CHECK(!mkdir(path,0700));
    snprintf(path,sizeof(path),"%s/Ref2VA",directory); CHECK(!mkdir(path,0700));
    snprintf(path,sizeof(path),"%s/FL2VA/model",directory); write_file(path,"all immutable FL2VA components");
    snprintf(path,sizeof(path),"%s/Ref2VA/model",directory); write_file(path,"all immutable Ref2VA components");
    snprintf(path,sizeof(path),"%s/reference",directory); write_file(path,"ordered input reference contents");
    snprintf(path,sizeof(path),"%s/state",directory); snprintf(other,sizeof(other),"%s/roundtrip",directory);
    reference_image_modes(directory,path);
    /* Loading stored conditioning does not run Qwen. Exact sampler resume
     * still enforces the numerical environment recorded at creation. */
    const char *scale_env=getenv("H3_QWEN_GQA_SCALE_MODE");
    char *saved_scale_env=scale_env?strdup(scale_env):NULL;
    const char *scale_modes[]={"legacy","scaled-q","reference"};
    for(int mode=0;mode<3;mode++) {
        setenv("H3_QWEN_GQA_SCALE_MODE",scale_modes[mode],1);
        h3_sampler_state *s=fixture(directory,0,0,1);
        CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        setenv("H3_QWEN_GQA_SCALE_MODE",scale_modes[(mode+1)%3],1);
        h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error)); CHECK(r);
        CHECK(!memcmp(s->text.values,r->text.values,s->text.tokens*s->text.width*2));
        CHECK(!h3_sampler_state_compatible_effective(r,directory,NULL,&r->device,error,sizeof(error)));
        CHECK(strstr(error,"environment differs"));
        setenv("H3_QWEN_GQA_SCALE_MODE",scale_modes[mode],1);
        CHECK(h3_sampler_state_compatible_effective(r,directory,NULL,&r->device,error,sizeof(error)));
        h3_sampler_state_free(r); h3_sampler_state_free(s);
    }
    if(saved_scale_env) { setenv("H3_QWEN_GQA_SCALE_MODE",saved_scale_env,1); free(saved_scale_env); }
    else unsetenv("H3_QWEN_GQA_SCALE_MODE");
    /* Display mode is absent from mathematical checkpoint identity. Changing
     * it preserves file bytes and must never prevent stop/resume. */
    const char *preview_env=getenv("H3_PREVIEW_MODE");
    char *saved_preview_env=preview_env?strdup(preview_env):NULL;
    unsetenv("H3_PREVIEW_MODE");
    h3_sampler_state *preview_baseline=fixture(directory,1,2,3);
    CHECK(h3_sampler_state_save(preview_baseline,path,error,sizeof(error)));
    size_t preview_bytes;uint8_t *preview_data=read_file(path,&preview_bytes);
    for(int mode=0;mode<3;mode++) {
        const char *names[]={"noisy","denoised",""};setenv("H3_PREVIEW_MODE",names[mode],1);
        CHECK(h3_sampler_state_compatible_effective(preview_baseline,directory,NULL,&preview_baseline->device,error,sizeof(error)));
        h3_sampler_state *s=fixture(directory,1,2,3);
        CHECK(h3_sampler_state_save(s,other,error,sizeof(error)));
        size_t bytes;uint8_t *data=read_file(other,&bytes);
        CHECK(bytes==preview_bytes && !memcmp(data,preview_data,bytes));
        free(data);h3_sampler_state_free(s);
    }
    free(preview_data);h3_sampler_state_free(preview_baseline);
    if(saved_preview_env) {setenv("H3_PREVIEW_MODE",saved_preview_env,1);free(saved_preview_env);}
    else unsetenv("H3_PREVIEW_MODE");
    for(int ref=0;ref<2;ref++) for(int cont=0;cont<3;cont++) for(int reuse=1;reuse<=3;reuse++) {
        h3_sampler_state *s=fixture(directory,ref,cont,reuse);
        CHECK(h3_sampler_state_compatible_effective(s,directory,NULL,&s->device,error,sizeof(error)));
        h3_device_info cuda_device={0};
        snprintf(cuda_device.backend,sizeof(cuda_device.backend),"cuda");
        snprintf(cuda_device.architecture,sizeof(cuda_device.architecture),"SM89");
        const char *portable=strstr(s->build_id,";portable=");
        CHECK(h3_sampler_state_compatible_effective(s,directory,NULL,&cuda_device,error,sizeof(error)) ==
              (portable && strlen(portable+10)==64));
        s->model_fingerprint[0]^=1; CHECK(!h3_sampler_state_compatible_effective(s,directory,NULL,&s->device,error,sizeof(error))); s->model_fingerprint[0]^=1;
        h3_device_info incompatible=s->device; incompatible.apple_gpu_family++;
        CHECK(!h3_sampler_state_compatible_effective(s,directory,NULL,&incompatible,error,sizeof(error)));
        s->build_id[0]^=1; CHECK(!h3_sampler_state_compatible_effective(s,directory,NULL,&s->device,error,sizeof(error))); s->build_id[0]^=1;
        setenv("H3_DISABLE_FUSED_MLP","1",1); CHECK(!h3_sampler_state_compatible_effective(s,directory,NULL,&s->device,error,sizeof(error))); unsetenv("H3_DISABLE_FUSED_MLP");
        CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
        h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error)); CHECK(r);
        CHECK(h3_sampler_state_save(r,other,error,sizeof(error)));
        size_t a,b; uint8_t *av=read_file(path,&a),*bv=read_file(other,&b); CHECK(a==b && !memcmp(av,bv,a)); free(bv);
        CHECK(!memcmp(s->video,r->video,s->video_elements*4)); CHECK(!memcmp(s->audio,r->audio,s->audio_elements*4));
        CHECK(!memcmp(s->text.values,r->text.values,s->text.tokens*s->text.width*2));
        CHECK(!memcmp(s->layout.positions,r->layout.positions,s->layout.seq_len*sizeof(*s->layout.positions)));
        CHECK(!strcmp(s->prompt,r->prompt)); CHECK(s->next_step==r->next_step && s->total_steps==r->total_steps);
        CHECK(!memcmp(s->source_fingerprint,r->source_fingerprint,32));
        if(reuse>1) { CHECK(s->last_evaluated==r->last_evaluated); CHECK(s->previous_evaluated==r->previous_evaluated);
            CHECK(!memcmp(s->last_video_velocity,r->last_video_velocity,s->video_elements*4));
            CHECK(!memcmp(s->previous_audio_velocity,r->previous_audio_velocity,s->audio_elements*4)); }
        if(!ref && !cont && reuse==1) {
            pid_t child=fork(); CHECK(child>=0);
            if(!child) {
                struct rlimit limit={1024,1024}; signal(SIGXFSZ,SIG_IGN);
                if(setrlimit(RLIMIT_FSIZE,&limit)) _exit(2);
                _exit(h3_sampler_state_save(s,path,error,sizeof(error))?1:0);
            }
            int status=0; CHECK(waitpid(child,&status,0)==child); CHECK(WIFEXITED(status) && WEXITSTATUS(status)==0);
            bv=read_file(path,&b); CHECK(a==b && !memcmp(av,bv,a)); free(bv);
        }
        s->next_step=21; CHECK(!h3_sampler_state_save(s,path,error,sizeof(error)));
        bv=read_file(path,&b); CHECK(a==b && !memcmp(av,bv,a)); free(av); free(bv); s->next_step=4;
        CHECK(!h3_sampler_state_save(s,"/no-such-sampler-directory/state",error,sizeof(error)));
        if(argc==3 && !strcmp(argv[1],"--fixture") && ref==1 && cont==2 && reuse==3) CHECK(h3_sampler_state_save(s,argv[2],error,sizeof(error)));
        if(argc==3 && !strcmp(argv[1],"--reference-fixture") && ref==1 && cont==0 && reuse==1) {
            s->params._arithmetic_recipe=H3_SGLANG_VERSION;s->rng_version=2;
            CHECK(h3_sampler_state_save(s,argv[2],error,sizeof(error)));
        }
        if(argc==3 && !strcmp(argv[1],"--upscale-fixture") && !ref && !cont && reuse==1) {
            memset(s->text.values,0,s->text.tokens*s->text.width*2);s->next_step=s->total_steps;
            h3_upscale_record record={0};
            CHECK(h3_upscale_capture(s,NULL,NULL,&record,argv[2],error,sizeof(error)));
        }
        h3_sampler_state_free(r); h3_sampler_state_free(s);
    }
    if(argc==3 && !strcmp(argv[1],"--encoded")) {
        const char *names[]={"face1","body1","face2","body2","12"};
        for(size_t i=0;i<5;i++) {
            char encoded_path[2048]; snprintf(encoded_path,sizeof(encoded_path),"%s/%s.h3av",argv[2],names[i]);
            h3_av_state *encoded=h3_av_state_load(encoded_path,error,sizeof(error)); CHECK(encoded);
            fixture_width=encoded->info.render_width; fixture_height=encoded->info.render_height; fixture_frames=encoded->info.frames;
            h3_sampler_state *s=fixture(directory,0,0,1);
            memcpy(s->video,encoded->video,s->video_elements*4); memcpy(s->audio,encoded->audio,s->audio_elements*4);
            CHECK(h3_sampler_state_save(s,path,error,sizeof(error)));
            h3_sampler_state *r=h3_sampler_state_load(path,error,sizeof(error)); CHECK(r);
            CHECK(!memcmp(r->video,encoded->video,s->video_elements*4)); CHECK(!memcmp(r->audio,encoded->audio,s->audio_elements*4));
            fprintf(stderr,"ok: encoded %s F32 roundtrip, %zu values\n",names[i],s->video_elements+s->audio_elements);
            h3_sampler_state_free(r); h3_sampler_state_free(s); h3_av_state_free(encoded);
        }
        fixture_width=64; fixture_height=32; fixture_frames=56;
    }
    const char *offline=getenv("H3_OFFLINE");char *saved_offline=offline?strdup(offline):NULL;
    unsetenv("H3_OFFLINE");char *online_identity=h3_sampler_environment();
    setenv("H3_OFFLINE","0",1);char *enabled_identity=h3_sampler_environment();
    setenv("H3_OFFLINE","1",1);char *offline_identity=h3_sampler_environment();
    CHECK(online_identity&&enabled_identity&&offline_identity);
    CHECK(!strcmp(online_identity,enabled_identity)&&!strcmp(online_identity,offline_identity));
    free(online_identity);free(enabled_identity);free(offline_identity);
    if(saved_offline){setenv("H3_OFFLINE",saved_offline,1);free(saved_offline);}else unsetenv("H3_OFFLINE");
    const char *verbose=getenv("H3_VERBOSE");char *saved_verbose=verbose?strdup(verbose):NULL;
    unsetenv("H3_VERBOSE");char *quiet_identity=h3_sampler_environment();
    setenv("H3_VERBOSE","1",1);char *verbose_identity=h3_sampler_environment();
    CHECK(quiet_identity&&verbose_identity&&!strcmp(quiet_identity,verbose_identity));
    free(quiet_identity);free(verbose_identity);
    if(saved_verbose){setenv("H3_VERBOSE",saved_verbose,1);free(saved_verbose);}else unsetenv("H3_VERBOSE");
    setenv("H3_DISABLE_FUSED_MLP","1\nH3_DISABLE_FUSED_PATCH_CAST=1",1);
    h3_sampler_state *nested_environment=fixture(directory,0,0,1);
    setenv("H3_DISABLE_FUSED_MLP","1",1); setenv("H3_DISABLE_FUSED_PATCH_CAST","1",1);
    CHECK(!h3_sampler_state_compatible_effective(nested_environment,directory,NULL,&nested_environment->device,error,sizeof(error)));
    h3_sampler_state_free(nested_environment); unsetenv("H3_DISABLE_FUSED_MLP"); unsetenv("H3_DISABLE_FUSED_PATCH_CAST");
    extensions(directory,path,other);
    uint8_t first[32],second[32],third[32];
    CHECK(h3_sampler_model_fingerprint_effective(directory,NULL,0,first,error,sizeof(error)));
    CHECK(h3_sampler_model_fingerprint_effective(directory,NULL,0,second,error,sizeof(error))); CHECK(!memcmp(first,second,32));
    snprintf(path,sizeof(path),"%s/FL2VA/model",directory); struct stat st; CHECK(!stat(path,&st));
    write_file(path,"all immutable FL2VA componentX");
    struct timespec times[2]={h3_stat_atime(&st),h3_stat_mtime(&st)}; CHECK(!utimensat(AT_FDCWD,path,times,0));
    CHECK(h3_sampler_model_fingerprint_effective(directory,NULL,0,third,error,sizeof(error))); CHECK(memcmp(first,third,32));
    /* Effective transformer substitution retains logical relative names.
     * A physical relocation and byte-identical offline tree produce the same
     * fingerprint; changed shared assets remain authoritative. */
    char transform[1024],effective[1024],tensor_path[1100],effective_tensor[1100];
    snprintf(transform,sizeof(transform),"%s/FL2VA/transformer",directory);CHECK(!mkdir(transform,0700));
    snprintf(effective,sizeof(effective),"%s/effective-transformer",directory);CHECK(!mkdir(effective,0700));
    snprintf(tensor_path,sizeof(tensor_path),"%s/weights",transform);write_file(tensor_path,"folded bytes");
    snprintf(effective_tensor,sizeof(effective_tensor),"%s/weights",effective);write_file(effective_tensor,"folded bytes");
    CHECK(h3_sampler_model_fingerprint_effective(directory,NULL,0,first,error,sizeof(error)));
    CHECK(h3_sampler_model_fingerprint_effective(directory,effective,0,second,error,sizeof(error)));
    CHECK(!memcmp(first,second,32));
    CHECK(!stat(effective_tensor,&st));
    write_file(effective_tensor,"changed fold");
    times[0]=h3_stat_atime(&st);times[1]=h3_stat_mtime(&st);
    CHECK(!utimensat(AT_FDCWD,effective_tensor,times,0));
    CHECK(h3_sampler_model_fingerprint_effective(directory,effective,0,third,error,sizeof(error)));
    CHECK(memcmp(first,third,32));
    write_file(tensor_path,"changed fold");
    CHECK(h3_sampler_model_fingerprint_effective(directory,NULL,0,second,error,sizeof(error)));
    CHECK(!memcmp(second,third,32));
    snprintf(path,sizeof(path),"%s/FL2VA/model",directory);write_file(path,"changed shared assets");
    CHECK(h3_sampler_model_fingerprint_effective(directory,effective,0,first,error,sizeof(error)));
    CHECK(memcmp(first,third,32));
    CHECK(h3_sampler_model_metadata_effective(directory,effective,0,first,error,sizeof(error)));
    CHECK(h3_sampler_model_metadata_effective(directory,effective,0,second,error,sizeof(error)));
    CHECK(!memcmp(first,second,32));
    CHECK(h3_sampler_model_fingerprint_effective(directory,effective,0,third,error,sizeof(error)));
    CHECK(memcmp(first,third,32)); /* Metadata identity is explicitly a different domain. */
    CHECK(!chmod(effective_tensor,0000));
    CHECK(h3_sampler_model_metadata_effective(directory,effective,0,second,error,sizeof(error)));
    CHECK(memcmp(first,second,32)); /* No file content access, and changed ctime invalidates. */
    CHECK(!chmod(effective_tensor,0600));
    char *before_verify=h3_sampler_environment();CHECK(before_verify);
    CHECK(!setenv("H3_QUANT_VERIFY","1",1));char *after_verify=h3_sampler_environment();CHECK(after_verify);
    CHECK(!strcmp(before_verify,after_verify));free(before_verify);free(after_verify);unsetenv("H3_QUANT_VERIFY");
    CHECK(!unlink(tensor_path));CHECK(!rmdir(transform));CHECK(!unlink(effective_tensor));CHECK(!rmdir(effective));
    h3_sigma_schedule full_schedule,short_schedule;
    CHECK(h3_serving_schedule_build(20,&full_schedule)); CHECK(h3_serving_schedule_build(4,&short_schedule));
    CHECK(memcmp(full_schedule.video,short_schedule.video,5*sizeof(float)));
    CHECK(memcmp(full_schedule.audio,short_schedule.audio,5*sizeof(float)));
    h3_params p=H3_PARAMS_DEFAULT; h3_device_info device={0}; device.apple_gpu_family=9;
    CHECK(h3_sampler_checkpoint_options(&p,&device,error,sizeof(error))); p.core_reuse=4; CHECK(h3_sampler_checkpoint_options(&p,&device,error,sizeof(error))); p.core_reuse=1;
    p.token_reduction=1; CHECK(h3_sampler_checkpoint_options(&p,&device,error,sizeof(error))); p.token_reduction=0;
    setenv("H3_GPU_SAMPLER","1",1); unsetenv("H3_CPU_SAMPLER"); CHECK(h3_sampler_checkpoint_options(&p,&device,error,sizeof(error)));
    setenv("H3_CPU_SAMPLER","1",1); CHECK(h3_sampler_checkpoint_options(&p,&device,error,sizeof(error))); unsetenv("H3_GPU_SAMPLER"); unsetenv("H3_CPU_SAMPLER");
    char cleanup[1024]; const char *files[]={"FL2VA/model","Ref2VA/model","reference","state","roundtrip"};
    for(size_t i=0;i<5;i++) { snprintf(cleanup,sizeof(cleanup),"%s/%s",directory,files[i]); unlink(cleanup); }
    snprintf(cleanup,sizeof(cleanup),"%s/FL2VA",directory); rmdir(cleanup);
    snprintf(cleanup,sizeof(cleanup),"%s/Ref2VA",directory); rmdir(cleanup); rmdir(directory);
    printf("ok: %d sampler state checks\n",checks); return 0;
}
