#include "src/upscale/upscale.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fail(char *e,size_t n,const char *m) {
    if (e && n)
        snprintf(e, n, "upscale: %s", m);
    return 0;
}
/* Compare existing inodes and canonical parents for not-yet-created outputs. */
static char *resolved(const char *path) {
    if(!path||!*path)return NULL;
    char *p=realpath(path,NULL);if(p)return p;
    char *copy=strdup(path);if(!copy)return NULL;
    char *slash=strrchr(copy,'/'),*name=slash?slash+1:copy;
    char *base=strdup(name);if(slash)*slash=0;
    char *parent=realpath(slash?(*copy?copy:"/"):".",NULL);
    p=NULL;if(base&&parent&&*base&&strcmp(base,".")&&strcmp(base,".."))
        if(asprintf(&p,"%s/%s",parent,base)<0)p=NULL;
    free(copy);free(base);free(parent);return p;
}
int h3_upscale_paths_alias(const char *a,const char *b) {
    if(!a||!b)return 0;
    struct stat x,y;
    if(!stat(a,&x)&&!stat(b,&y)&&x.st_dev==y.st_dev&&x.st_ino==y.st_ino)return 1;
    char *ca=resolved(a),*cb=resolved(b);
    int same=ca&&cb&&!strcmp(ca,cb);free(ca);free(cb);return same;
}
int h3_upscale_options_valid(const h3_params *p,char *e,size_t n) {
    if(!p)return fail(e,n,"missing options");
    if(p->geometry_profile && (p->geometry_profile!=1||!h3_geometry_valid(p->width,p->height,1)||
       p->still||p->continuation||p->denoise_reuse!=1||p->core_reuse!=1||p->dit_layers!=50||
       p->token_reduction||p->cuda_denoise_quant||p->cuda_attention||p->adaptive_cache||p->backend||
       p->attention_mode||p->use_int8_row_fc2||p->preview_vae||p->save_upscale_state||(p->render_width&&p->render_width!=p->width)||
       (p->render_height&&p->render_height!=p->height)))return fail(e,n,"invalid larger-canvas dense policy");
    if(p->state_only!=0&&p->state_only!=1)return fail(e,n,"invalid state-only value");
    if(!p->save_upscale_state)return p->state_only&&!p->resume_sampler_state&&!p->geometry_profile?
        fail(e,n,"state-only requires --save-upscale-state or the explicit API geometry profile"):1;
    if(p->reference_count>12||(p->reference_count&&!p->references))return fail(e,n,"invalid source references");
    char *path=resolved(p->save_upscale_state);
    if(!path)return fail(e,n,"source output needs a nonempty filename and existing parent directory");
    free(path);
    if(p->still||p->continuation||p->resume_sampler_state||p->save_sampler_state||p->stop_after_step>=0||
       p->denoise_reuse!=1||p->core_reuse!=1||p->dit_layers!=50||p->token_reduction||
       p->cuda_denoise_quant||p->cuda_attention||p->adaptive_cache||
       p->backend||p->attention_mode||p->use_int8_row_fc2||p->preview_vae||
       p->preview_on_stop||(p->render_width&&p->render_width!=p->width)||
       (p->render_height&&p->render_height!=p->height))
        return fail(e,n,"source capture requires completed dense BF16 video, full layers and exact render canvas");
    if(p->state_only&&(p->preview_denoise||p->on_frame))return fail(e,n,"state-only cannot deliver preview frames");
    const char *paths[]={p->output_path,p->first_frame,p->last_frame,p->save_sampler_state,
        p->save_conditioning,p->load_conditioning};
    for(size_t i=0;i<sizeof(paths)/sizeof(*paths);i++)
        if(h3_upscale_paths_alias(p->save_upscale_state,paths[i]))return fail(e,n,"source output aliases another input/output");
    for(size_t i=0;i<p->reference_count;i++)if(p->references&&
        (h3_upscale_paths_alias(p->save_upscale_state,p->references[i].path)||h3_upscale_paths_alias(p->save_upscale_state,p->references[i].audio_path)))
        return fail(e,n,"source output aliases a reference");
    return 1;
}
int h3_upscale_record_valid(const h3_sampler_state *s,char *e,size_t n) {
    const h3_upscale_record *u=&s->upscale;
    if(!u->stage)return !memcmp(u,&(h3_upscale_record){0},sizeof(*u))?1:fail(e,n,"metadata without stage");
    int refining=u->stage==2;
    if((u->stage!=1&&!refining)||u->version!=(refining?2u:1u)||u->semantic_policy!=1||u->metadata_identity!=1||
       !h3_geometry_valid(u->source_width,u->source_height,0)||
       (int64_t)u->source_width*(refining?2:1)!=s->render_width||(int64_t)u->source_height*(refining?2:1)!=s->render_height||
       (!refining&&(s->next_step!=s->total_steps||s->sigmas.video[s->next_step]!=0||s->sigmas.audio[s->next_step]!=0))||
       s->continuation||s->params.denoise_reuse!=1||s->params.core_reuse!=1||s->params.dit_layers!=50||
       s->params.token_reduction||s->params.cuda_denoise_quant||s->params.cuda_attention||s->params.adaptive_cache||
       s->params.backend||s->params.attention_mode||s->params.use_int8_row_fc2||s->params.width!=s->render_width||s->params.height!=s->render_height||
       (!refining&&(s->prepared.count||s->original_video_noise))||s->original_audio_noise||u->keyframe_count>2||
       (s->ref2va&&u->keyframe_count))return fail(e,n,"invalid clean source stage, policy or disposable state");
    size_t rows=0,audio=0;
    for(size_t i=0;i<12;i++) {
        if(i<s->reference_count&&s->references[i].kind!=H3_LAYOUT_REF_AUDIO) {
            const h3_layout_ref *r=&s->references[i];
            if(u->original_width[i]<1||u->original_height[i]<1||u->original_width[i]>100000||u->original_height[i]>100000||
               u->semantic_width[i]<32||u->semantic_height[i]<32||u->semantic_width[i]%32||u->semantic_height[i]%32||
               (refining?(u->semantic_width[i]>r->latent_w*16||u->semantic_height[i]>r->latent_h*16):
                   (u->semantic_width[i]!=r->latent_w*16||u->semantic_height[i]!=r->latent_h*16)))
                return fail(e,n,"missing or inconsistent original/semantic reference dimensions");
            rows+=(size_t)r->latent_t*r->latent_h*r->latent_w*24;
        } else if(u->original_width[i]||u->original_height[i]||u->semantic_width[i]||u->semantic_height[i])
            return fail(e,n,"unexpected visual reference metadata");
        if(i<s->reference_count)audio+=(size_t)s->references[i].audio_t*64;
    }
    if(!s->ref2va) {
        for(unsigned i=0;i<u->keyframe_count;i++)if((u->keyframes[i]!=0&&u->keyframes[i]!=s->aligned_frames-1)||
            (i&&u->keyframes[i]<=u->keyframes[i-1]))return fail(e,n,"invalid source keyframe endpoints");
        for(unsigned i=u->keyframe_count;i<2;i++)if(u->keyframes[i])return fail(e,n,"unused keyframe metadata");
        rows=(size_t)u->keyframe_count*s->latent_h*s->latent_w*24;
    }
    if(rows!=s->condition_video_elements||audio!=s->condition_audio_elements)return fail(e,n,"raw condition count mismatch");
    for(size_t i=0;i<s->text.tokens*s->text.width;i++)
        if((s->text.values[i]&0x7f80)==0x7f80)return fail(e,n,"nonfinite semantic conditioning");
    if(!h3_upscale_audio_intact(s))return fail(e,n,"source audio identity mismatch");
    if(refining) {
        h3_sigma_schedule expected;uint8_t hash[32],zero[32]={0};
        if((u->recipe!=1&&u->recipe!=2)||u->refinement_steps!=(unsigned)s->total_steps||u->seed!=s->params.seed||
           !h3_upscale_schedule(s->total_steps,u->sigma,&expected)||s->sampler_mode||!s->original_video_noise||
           memcmp(expected.video,s->sigmas.video,((size_t)s->total_steps+1)*4)||
           memcmp(expected.audio,s->sigmas.audio,((size_t)s->total_steps+1)*4)||
           !memcmp(u->parent_hash,zero,32)||!memcmp(u->artifact_hash,zero,32))return fail(e,n,"invalid refinement recipe, schedule or parent");
        h3_sampler_hash(s->original_video_noise,s->video_elements*4,hash);
        if(memcmp(hash,u->noise_hash,32))return fail(e,n,"refinement noise identity mismatch");
        h3_upscale_condition_hash(s,hash);
        if(memcmp(hash,u->condition_hash,32))return fail(e,n,"refinement conditions/layout identity mismatch");
    } else {
        h3_upscale_record clean=*u;clean.recipe=clean.refinement_steps=0;clean.sigma=0;clean.seed=0;
        memset(clean.parent_hash,0,32);memset(clean.artifact_hash,0,32);memset(clean.noise_hash,0,32);memset(clean.condition_hash,0,32);
        if(memcmp(&clean,u,sizeof(clean)))return fail(e,n,"refinement fields on a clean source");
    }
    return 1;
}
int h3_upscale_capture(const h3_sampler_state *s,const float *video,const float *audio,
    const h3_upscale_record *record,const char *path,char *e,size_t n) {
    if(!s||!record)return fail(e,n,"missing source capture");
    h3_sampler_state view=*s;
    view.upscale=*record;view.upscale.version=1;view.upscale.stage=1;
    view.upscale.semantic_policy=1;view.upscale.metadata_identity=1;
    view.upscale.source_width=s->render_width;view.upscale.source_height=s->render_height;
    h3_sampler_hash(s->audio,s->audio_elements*4,view.upscale.audio_hash);
    view.condition_video=(float *)video;view.condition_audio=(float *)audio;
    memset(&view.prepared,0,sizeof(view.prepared));
    view.original_video_noise=NULL;view.original_audio_noise=NULL;
    return h3_upscale_file_save(&view,path,e,n);
}
h3_upscale_source *h3_upscale_source_load(const char *path,char *e,size_t n) {
    h3_sampler_state *s=h3_upscale_file_load(path,e,n);
    if(!s)return NULL;
    h3_upscale_source *u=malloc(sizeof(*u));
    if(!u){h3_sampler_state_free(s);fail(e,n,"out of memory");return NULL;}
    u->state=s;u->path=resolved(path);
    if(!u->path){h3_upscale_source_free(u);fail(e,n,"cannot retain source path");return NULL;}
    return u;
}
int h3_upscale_source_save(const h3_upscale_source *u,const char *path,char *e,size_t n) {
    return u&&u->state?h3_upscale_file_save(u->state,path,e,n):fail(e,n,"missing source");
}
void h3_upscale_source_free(h3_upscale_source *u) {
    if(u){h3_sampler_state_free(u->state);free(u->path);free(u);}
}
h3_upscale_source *h3_upscale_source_import_sampler(const char *path,char *e,size_t n) {
    h3_sampler_state *s=h3_sampler_state_load(path,e,n);if(!s)return NULL;
    if(s->conditioned||s->reference_count||s->next_step!=s->total_steps||s->upscale.stage||!s->params._arithmetic_recipe) {
        fail(e,n,"sampler import requires a completed current dense text-only sampler with metadata identity");goto bad;
    }
    h3_prepared_cache_free(&s->prepared);
    free(s->original_video_noise);free(s->original_audio_noise);
    s->original_video_noise=NULL;s->original_audio_noise=NULL;
    s->upscale=(h3_upscale_record){.version=1,.stage=1,.semantic_policy=1,.metadata_identity=1,
        .source_width=s->render_width,.source_height=s->render_height};
    h3_sampler_hash(s->audio,s->audio_elements*4,s->upscale.audio_hash);
    if(!h3_sampler_state_validate(s,e,n))goto bad;
    h3_upscale_source *u=malloc(sizeof(*u));if(!u){fail(e,n,"out of memory");goto bad;}
    u->state=s;u->path=resolved(path);
    if(!u->path){h3_upscale_source_free(u);fail(e,n,"cannot retain imported source path");return NULL;}
    return u;
bad:h3_sampler_state_free(s);return NULL;
}
