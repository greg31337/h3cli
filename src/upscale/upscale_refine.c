#include "src/log.h"
#include "src/upscale/upscale.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/sglang/sglang.h"
#include "src/backend.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif

static int fail(char *e,size_t n,const char *m){if(e&&n)snprintf(e,n,"upscale: %s",m);return 0;}
int h3_upscale_schedule(int k,float sigma,h3_sigma_schedule *s) {
    if(!s||(k!=0&&k!=2&&k!=3&&k!=4)||!isfinite(sigma)||(k?(sigma<=0||sigma>.5f):sigma!=0))return 0;
    memset(s,0,sizeof(*s));s->steps=k;if(!k)return 1;
    double q0=(double)sigma/(12.-11.*(double)sigma);
    s->video[0]=sigma;
    for(int i=1;i<k;i++){double q=q0*(1.-(double)i/(double)k);s->video[i]=(float)(12.*q/(1.+11.*q));}
    for(int i=0;i<k;i++)if(!isfinite(s->video[i])||s->video[i]<=s->video[i+1])return 0;
    return 1;
}
int h3_upscale_request_valid(const h3_upscale_options *o,char *e,size_t n) {
    if(!o)return fail(e,n,"missing options");
    if(!h3_output_encoding_valid(&o->delivery.output_encoding,e,n))return 0;
    h3_sigma_schedule s;float sigma=o->refine_steps?o->sigma:0;
    if(!h3_upscale_schedule(o->refine_steps,sigma,&s)||(!o->refine_steps&&o->sigma_set))
        return fail(e,n,"refinement steps must be 0/2/3/4; 0 < sigma <= 0.5 for refinement; K=0 rejects explicit sigma");
    if((o->seed_set!=0&&o->seed_set!=1)||(o->sigma_set!=0&&o->sigma_set!=1)||(o->state_only!=0&&o->state_only!=1)||
       o->stop_after_step< -1||o->stop_after_step>o->refine_steps||
       (!o->refine_steps&&(o->save_sampler_state||o->stop_after_step>=0))||
       (o->stop_after_step>=0&&!o->save_sampler_state))return fail(e,n,"invalid refinement seed/stop/checkpoint controls");
    if(o->delivery.preview_vae||o->delivery.preview_vae_model||(o->state_only&&o->delivery.on_frame))
        return fail(e,n,"upscale delivery requires the full native VAEs; state-only cannot deliver frames");
    if((o->model_path&&!*o->model_path)||(o->save_sampler_state&&!*o->save_sampler_state)||
       h3_upscale_paths_alias(o->delivery.output_path,o->model_path)||h3_upscale_paths_alias(o->save_sampler_state,o->model_path)||
       h3_upscale_paths_alias(o->delivery.output_path,o->save_sampler_state))return fail(e,n,"empty or aliased upscale input/output path");
    const char *reduction=getenv("H3_TOKEN_REDUCTION"),*blocks=getenv("H3_TEST_DIT_BLOCKS");
    if((reduction&&*reduction&&strcmp(reduction,"0"))||(blocks&&strcmp(blocks,"50"))||
       getenv("H3_TEST_NATIVE_TEACHER_DIR")||getenv("H3_TEST_SGLANG_INPUT_DIR")||getenv("H3_TEST_SGLANG_CONDITION_DIR"))
        return fail(e,n,"upscale requires dense full-depth native conditions without test input overrides");
    return h3_test_evaluation_budget(o->stop_after_step>=0?o->stop_after_step:o->refine_steps,e,n);
}
int h3_upscale_audio_intact(const h3_sampler_state *s) {
    if (!s || !s->audio)
        return 0;
    uint8_t hash[32];
    h3_sampler_hash(s->audio, s->audio_elements * 4, hash);
    return !memcmp(hash,s->upscale.audio_hash,32);
}
void h3_upscale_condition_hash(const h3_sampler_state *s,uint8_t hash[32]) {
    uint8_t parts[8][32];
    h3_sampler_hash(s->condition_video,s->condition_video_elements*4,parts[0]);
    h3_sampler_hash(s->condition_audio,s->condition_audio_elements*4,parts[1]);
    h3_sampler_hash(s->text.values,s->text.tokens*s->text.width*2,parts[2]);
    h3_sampler_hash(s->text.tags,s->text.tags?s->text.tokens:0,parts[3]);
    h3_sampler_hash(s->layout.positions,s->layout.seq_len*sizeof(*s->layout.positions),parts[4]);
    /* Avoid native padding in segment/reference structs. */
    uint64_t fields[1000*3];
    for(size_t i=0;i<s->layout.segment_count;i++) {
        fields[i*3]=s->layout.segments[i].start;fields[i*3+1]=s->layout.segments[i].stop;fields[i*3+2]=s->layout.segments[i].kind;
    }
    h3_sampler_hash(fields,s->layout.segment_count*3*8,parts[5]);
    for(size_t i=0;i<s->reference_count;i++) {
        const h3_layout_ref *r=&s->references[i];fields[i*5]=r->kind;fields[i*5+1]=(uint64_t)r->latent_t;
        fields[i*5+2]=(uint64_t)r->latent_h;fields[i*5+3]=(uint64_t)r->latent_w;fields[i*5+4]=(uint64_t)r->audio_t;
    }
    h3_sampler_hash(fields,s->reference_count*5*8,parts[6]);
    h3_sampler_hash(s->upscale.keyframes,sizeof(s->upscale.keyframes),parts[7]);
    h3_sampler_hash(parts,sizeof(parts),hash);
}
static void hash_hex(const uint8_t h[32],char out[65]) {for(int i=0;i<32;i++)snprintf(out+2*i,3,"%02x",h[i]);}
void h3_upscale_presentation(const h3_sampler_state *s,h3_presentation *p) {
    p->version=9;p->geometry_profile=s->params.geometry_profile;p->av_metadata_identity=1;
    p->upscale_recipe=(int)s->upscale.recipe;p->upscale_steps=(int)s->upscale.refinement_steps;p->upscale_sigma=s->upscale.sigma;
    hash_hex(s->upscale.parent_hash,p->upscale_parent_sha256);hash_hex(s->upscale.artifact_hash,p->upscale_artifact_sha256);
}
void h3_upscale_transfer_free(h3_upscale_transfer *t) {
    if(t){free(t->video);h3_upscale_conditions_free(&t->conditions);free(t);}
}
h3_upscale_transfer *h3_upscale_transfer_create(const h3_upscale_source *source,const h3_upscale_plan *p,
    const char *path,int recipe,h3_progress_callback progress,void *opaque,char *e,size_t n) {
    if(!source||!p||(recipe!=1&&recipe!=2)||(recipe==1&&(!path||!*path))){fail(e,n,"missing learned artifact or invalid transfer recipe");return NULL;}
    h3_upscale_model *model=NULL;h3_upscale_transfer *t=calloc(1,sizeof(*t));if(!t){fail(e,n,"transfer allocation failed");return NULL;}
    t->recipe=recipe;memcpy(t->parent_hash,source->state->loaded_hash,32);
    double begin=h3_av_now();
    if(recipe==1) {
        model=h3_upscale_model_load(path,e,n);if(!model)goto bad;
        const char *hex=h3_upscale_artifact_sha256();for(int i=0;i<32;i++){unsigned byte=0;sscanf(hex+2*i,"%2x",&byte);t->artifact_hash[i]=(uint8_t)byte;}
    } else {const char contract[]="h3-native-bilinear-normalized-F32-align-corners-false-v1";h3_sampler_hash(contract,sizeof(contract),t->artifact_hash);}
    t->load_seconds=h3_av_now()-begin;begin=h3_av_now();const h3_sampler_state *s=source->state;
    t->video=model?h3_upscale_volume(model,s->video,s->latent_t,s->latent_h,s->latent_w,p->info.height/16,p->info.width/16,
        progress,opaque,NULL,NULL,e,n):h3_upscale_bilinear(s->video,s->latent_t,s->latent_h,s->latent_w,p->info.height/16,p->info.width/16,e,n);
    if (!t->video)
        goto bad;
    t->forward_seconds = h3_av_now() - begin;
    begin = h3_av_now();
    if(!h3_upscale_retarget(source,p,model,&t->conditions,progress,opaque,e,n))goto bad;
    t->retarget_seconds=h3_av_now()-begin;h3_upscale_model_free(model);
    H3_VERBOSE("h3_upscale_transfer {\"recipe\":%d,\"load_seconds\":%.9g,\"forward_seconds\":%.9g,\"retarget_seconds\":%.9g}\n",
        recipe,t->load_seconds,t->forward_seconds,t->retarget_seconds);return t;
bad:h3_upscale_model_free(model);h3_upscale_transfer_free(t);return NULL;
}
static int augment_visual(h3_sampler_state *s,char *e,size_t n) {
    uint64_t seed=s->params.seed^UINT64_C(0xd1b54a32d192ed03);size_t offset=0;
    size_t count=s->ref2va?s->reference_count:s->upscale.keyframe_count;int visual=0;
    for(size_t i=0;i<count;i++)if(!s->ref2va||s->references[i].kind!=H3_LAYOUT_REF_AUDIO)visual++;
    for(size_t i=0;i<count;i++) {
        h3_layout_ref r=s->ref2va?s->references[i]:(h3_layout_ref){.kind=H3_LAYOUT_REF_IMAGE,.latent_t=1,.latent_h=s->latent_h,.latent_w=s->latent_w};
        if(r.kind==H3_LAYOUT_REF_AUDIO)continue;
        size_t len=(size_t)r.latent_t*r.latent_h*r.latent_w*24;float *v=s->condition_video+offset;
        if(s->params._arithmetic_recipe) {
            if(!h3_sglang_video_condition(v,r.latent_t,r.latent_h,r.latent_w,s->latent_t,visual,seed))return fail(e,n,"visual conditioning RNG failed");
        } else {
            h3_rng rng;h3_rng_seed(&rng,seed);
            for(size_t j=0;j<len;j++)v[j]=.999f*v[j]+.001f*h3_rng_normal(&rng);
        }
        offset+=len;
    }
    return offset==s->condition_video_elements;
}
h3_sampler_state *h3_upscale_initialize(const h3_upscale_source *source,const h3_upscale_plan *p,
    const h3_upscale_transfer *t,const h3_upscale_options *o,const char *model_dir,const h3_device_info *device,char *e,size_t n) {
    if(!h3_upscale_request_valid(o,e,n)||!o->refine_steps||!source||!p||!t||!device||
       memcmp(t->parent_hash,source->state->loaded_hash,32)||memcmp(p->source_identity,t->parent_hash,32)) {
        fail(e,n,"invalid refinement initialization or parent identity");return NULL;
    }
    h3_sigma_schedule sigmas;if(!h3_upscale_schedule(o->refine_steps,o->sigma,&sigmas))return NULL;
    h3_sampler_state *s=h3_sampler_state_create(&sigmas,p->info.video_elements,p->info.audio_elements,1);
    if(!s){fail(e,n,"refinement state allocation failed");return NULL;}
    s->upscale=source->state->upscale;s->upscale.version=2;s->upscale.stage=2;s->upscale.recipe=(unsigned)t->recipe;
    s->upscale.refinement_steps=(unsigned)o->refine_steps;s->upscale.sigma=o->sigma;
    s->upscale.seed=o->seed_set?o->seed:source->state->params.seed;
    memcpy(s->upscale.parent_hash,t->parent_hash,32);memcpy(s->upscale.artifact_hash,t->artifact_hash,32);
    h3_params params=source->state->params;params.width=p->info.width;params.height=p->info.height;
    params.render_width=params.width;params.render_height=params.height;params.geometry_profile=p->info.geometry_profile;
    params.seed=s->upscale.seed;params.steps=o->refine_steps;params._arithmetic_recipe=!strcmp(device->backend,"cuda")?H3_SGLANG_VERSION:0;
    params.use_slower_bf16_mlp=params.use_slower_bf16_qkv=params.use_slower_bf16_attention_output=1;
    uint8_t av[32];
    if(!h3_av_state_metadata_signature(model_dir,source->state->ref2va,av,e,n)||
       !h3_sampler_capture_upscale(s,source->state,model_dir,device,&params,&t->conditions.layout,p->references,
          t->conditions.video,t->conditions.video_elements,t->conditions.audio,t->conditions.audio_elements,av,e,n))goto bad;
    if(!augment_visual(s,e,n))goto bad;
    h3_upscale_condition_hash(s,s->upscale.condition_hash);
    s->original_video_noise=malloc(s->video_elements*4);if(!s->original_video_noise){fail(e,n,"video noise allocation failed");goto bad;}
    double noise_begin=h3_av_now();
    h3_rng_seed(&s->video_rng,params.seed);h3_rng_seed(&s->audio_rng,params.seed);
    if(t->noise_source) {
        const h3_sampler_state *shared=t->noise_source;
        if(!h3_sampler_state_validate(shared,e,n)||shared->upscale.stage!=2||shared->next_step||
           shared->params.seed!=params.seed||shared->rng_version!=s->rng_version||shared->video_elements!=s->video_elements||
           shared->render_width!=s->render_width||shared->render_height!=s->render_height||shared->latent_t!=s->latent_t||
           memcmp(shared->upscale.parent_hash,s->upscale.parent_hash,32)) {
            fail(e,n,"shared comparison noise has a different parent, geometry, seed or RNG recipe");goto bad;
        }
        memcpy(s->original_video_noise,shared->original_video_noise,s->video_elements*4);s->video_rng=shared->video_rng;
        H3_VERBOSE("h3cli: adopted persisted target video noise; no new video draws\n");
    }
    else if(params._arithmetic_recipe) {if(!h3_sglang_normal(params.seed,s->original_video_noise,s->video_elements)){fail(e,n,"video RNG failed");goto bad;}}
    else h3_rng_fill_normal(&s->video_rng,s->original_video_noise,s->video_elements);
    H3_VERBOSE("h3_upscale_noise_time {\"seconds\":%.9g,\"reused\":%d}\n",h3_av_now()-noise_begin,t->noise_source!=NULL);
    s->video_random_count=s->video_elements;s->audio_random_count=0;
    h3_sampler_hash(s->original_video_noise,s->video_elements*4,s->upscale.noise_hash);
    for(size_t i=0;i<s->video_elements;i++)s->video[i]=(1.f-o->sigma)*t->video[i]+o->sigma*s->original_video_noise[i];
    memcpy(s->audio,source->state->audio,s->audio_elements*4);s->full_sequence=s->layout.seq_len;
    if (!h3_sampler_state_validate(s, e, n))
        goto bad;
    return s;
bad:h3_sampler_state_free(s);return NULL;
}
