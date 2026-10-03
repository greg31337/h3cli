#include "src/upscale/upscale.h"
#include "src/internal.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/media/delivery.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int source_compatible(h3_ctx *ctx,const h3_upscale_source *source) {
    if(!ctx||!source||!source->state)return 0;
    if(ctx->lora){h3_set_error(ctx,"upscaling does not support runtime LoRA");return 0;}
    uint8_t model[32],av[32];const h3_sampler_state *s=source->state;
    if(!h3_sampler_state_validate(s,ctx->error,sizeof(ctx->error))||s->upscale.stage!=1||
       !h3_sampler_model_metadata_effective(ctx->model_dir,NULL,s->ref2va,model,ctx->error,sizeof(ctx->error))||
       !h3_av_state_metadata_signature(ctx->model_dir,s->ref2va,av,ctx->error,sizeof(ctx->error)))return 0;
    if(memcmp(model,s->model_fingerprint,32)||memcmp(av,s->av_signature,32)) {
        h3_set_error(ctx,"upscale source model/decoder metadata differs from this installation; use the original model installation");return 0;
    }
    return 1;
}
static int source_paths(h3_ctx *ctx,const h3_upscale_source *s,const h3_upscale_options *o) {
    if(!s||!o)return 0;
    if(h3_upscale_paths_alias(s->path,o->model_path)||h3_upscale_paths_alias(s->path,o->delivery.output_path)||
       h3_upscale_paths_alias(s->path,o->save_sampler_state)) {
        h3_set_error(ctx,"upscale output/model aliases the loaded source path");return 0;
    }
    return 1;
}
h3_result *h3_upscale_execute(h3_ctx *ctx,const h3_upscale_source *source,const h3_upscale_plan *p,
    const h3_upscale_transfer *transfer,const h3_upscale_options *o) {
    if(!ctx||!h3_upscale_request_valid(o,ctx->error,sizeof(ctx->error))||!source_paths(ctx,source,o)||!source_compatible(ctx,source))return NULL;
    if(!p||!transfer||!transfer->video||memcmp(p->source_identity,source->state->loaded_hash,32)||
       memcmp(transfer->parent_hash,p->source_identity,32)) {h3_set_error(ctx,"upscale plan/transfer parent mismatch");return NULL;}
    h3_cache_clear(ctx);ctx->dit_sampler_key_ready=0;
    if(!o->refine_steps) {
        h3_av_state *av=h3_av_state_new_profile(p->info.width,p->info.height,p->info.frames,p->info.geometry_profile,
            o->seed_set?o->seed:source->state->params.seed,source->state->av_signature);
        h3_result *r=calloc(1,sizeof(*r));
        if(!av||!r){h3_av_state_free(av);free(r);h3_set_error(ctx,"cannot retain upscaled AV state");return NULL;}
        memcpy(av->video,transfer->video,av->info.video_elements*4);
        memcpy(av->audio,source->state->audio,av->info.audio_elements*4);
        r->av_state=av;r->width=p->info.width;r->height=p->info.height;r->frames=p->info.frames;
        r->fps=24;r->sample_rate=32000;r->audio_samples=p->info.audio_t*800;r->seed=av->info.seed;
        r->presentation=(h3_presentation){.version=9,.ref2va=source->state->ref2va,.render_width=r->width,
            .render_height=r->height,.width=r->width,.height=r->height,.fps=24,.sample_rate=32000,
            .codec_version=!strcmp(ctx->device.backend,"cuda")?2:1,.geometry_profile=p->info.geometry_profile,
            .av_metadata_identity=1,.upscale_recipe=transfer->recipe};
        for(int i=0;i<32;i++) {
            snprintf(r->presentation.upscale_parent_sha256+2*i,3,"%02x",transfer->parent_hash[i]);
            snprintf(r->presentation.upscale_artifact_sha256+2*i,3,"%02x",transfer->artifact_hash[i]);
        }
        if(!h3_presentation_validate(&r->presentation,&av->info,ctx->error,sizeof(ctx->error))){h3_result_free(r);return NULL;}
        fprintf(stderr,"h3cli: upscale K=0: no video/audio noise draws; no DiT evaluations\n");
        if(o->state_only)return r;
        h3_result *delivered=h3_decode_av_owned(ctx->model_dir,av,&r->presentation,&o->delivery,ctx->error,sizeof(ctx->error));
        r->av_state=NULL;h3_result_free(r);return delivered;
    }
    h3_sampler_state *state=h3_upscale_initialize(source,p,transfer,o,ctx->model_dir,&ctx->device,ctx->error,sizeof(ctx->error));
    if(!state)return NULL;
    /* Publish boundary zero before any expensive transformer admission. */
    if(o->save_sampler_state&&!h3_sampler_state_save(state,o->save_sampler_state,ctx->error,sizeof(ctx->error))) {
        h3_sampler_state_free(state);return NULL;
    }
    if(o->stop_after_step==0) {
        h3_result *r=calloc(1,sizeof(*r));if(!r){h3_sampler_state_free(state);h3_set_error(ctx,"cannot allocate paused upscale result");return NULL;}
        r->status=H3_RESULT_PAUSED;r->sampler_state=state;r->total_steps=o->refine_steps;
        r->width=p->info.width;r->height=p->info.height;r->fps=24;r->seed=state->params.seed;return r;
    }
    h3_result *r=h3_generate_upscale_state(ctx,state,o);
    if(!r||r->sampler_state!=state)h3_sampler_state_free(state);
    return r;
}
h3_result *h3_upscale(h3_ctx *ctx,const h3_upscale_source *source,const h3_upscale_options *o) {
    if(!ctx||!h3_upscale_request_valid(o,ctx->error,sizeof(ctx->error))||!source_paths(ctx,source,o)||!source_compatible(ctx,source))return NULL;
    if(!o->model_path){h3_set_error(ctx,"fresh learned upscaling requires --upscale-model");return NULL;}
    h3_upscale_plan *p=h3_upscale_plan_create(source,ctx->error,sizeof(ctx->error));if(!p)return NULL;
    h3_cache_clear(ctx);ctx->dit_sampler_key_ready=0;
    h3_upscale_transfer *t=h3_upscale_transfer_create(source,p,o->model_path,1,o->delivery.on_progress,
        o->delivery.callback_opaque,ctx->error,sizeof(ctx->error));
    h3_result *r=t?h3_upscale_execute(ctx,source,p,t,o):NULL;
    h3_upscale_transfer_free(t);h3_upscale_plan_free(p);return r;
}
