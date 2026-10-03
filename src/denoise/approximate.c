#include "src/denoise/approximate.h"
#include "src/denoise/attention.h"
#include "src/denoise/adaptive_cache.h"
#include "src/denoise/subblock.h"
#include "src/host.h"
#include "src/sampling/av_state.h"
#include "src/sampling/bridge.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#ifndef __APPLE__
#include "src/device.h"
#endif
float h3_adaptive_threshold(const h3_params *p) {
    float value=!p->adaptive_cache?0:(p->adaptive_cache_threshold_set||p->adaptive_cache_threshold!=0)?
        p->adaptive_cache_threshold:(p->adaptive_cache==1?0.04f:0.08f);
    return value==0?0:value; /* Canonicalize an API-provided negative zero. */
}
int h3_adaptive_max_hits(const h3_params *p) {
    return !p->adaptive_cache?0:(p->adaptive_cache_max_hits_set||p->adaptive_cache_max_hits)?
        p->adaptive_cache_max_hits:(p->adaptive_cache==1?1:3);
}
int h3_adaptive_controls_valid(const h3_params *p,char *e,size_t n) {
    if(!p)return 0;
    const char *why=NULL;
    int threshold=p->adaptive_cache_threshold_set||p->adaptive_cache_threshold!=0;
    int hits=p->adaptive_cache_max_hits_set||p->adaptive_cache_max_hits!=0;
    if(threshold&&(!isfinite(p->adaptive_cache_threshold)||p->adaptive_cache_threshold<0||p->adaptive_cache_threshold>1))
        why="adaptive-cache-threshold must be finite and in [0,1]";
    else if(hits&&(p->adaptive_cache_max_hits<1||p->adaptive_cache_max_hits>16))
        why="adaptive-cache-max-hits must be an integer in [1,16]";
    else if((threshold||hits)&&!p->adaptive_cache&&!p->resume_sampler_state)
        why="adaptive-cache-threshold/max-hits require an enabled --adaptive-cache";
    if(why){if(e&&n)snprintf(e,n,"%s",why);return 0;}return 1;
}
int h3_adaptive_controls_match(const h3_params *p,const h3_params *s) {
    if(!h3_adaptive_controls_valid(p,NULL,0))return 0;
    return (!(p->adaptive_cache_threshold_set||p->adaptive_cache_threshold!=0)||
        (s->adaptive_cache&&p->adaptive_cache_threshold==h3_adaptive_threshold(s)))&&
        (!(p->adaptive_cache_max_hits_set||p->adaptive_cache_max_hits)||
        (s->adaptive_cache&&p->adaptive_cache_max_hits==h3_adaptive_max_hits(s)));
}
int h3_warmups_valid(const h3_params *p,char *e,size_t n) {
    if(!p)return 0;
    const int values[]={p->adaptive_cache_warmup,p->subblock_warmup};
    const int selected[]={p->adaptive_cache_warmup_set,p->subblock_warmup_set};
    const int enabled[]={p->adaptive_cache!=0,p->cuda_attention==H3_ATTENTION_SUBBLOCK};
    const char *names[]={"adaptive-cache-warmup","subblock-warmup"};
    for(int i=0;i<2;i++)if(values[i]||selected[i]) {
        const char *why=NULL;
        if(values[i]<2||values[i]>16)why="must be an integer in [2,16]";
        else if(!p->resume_sampler_state&&!enabled[i])why=i?"requires --cuda-attention subblock":"requires an enabled --adaptive-cache";
        else if(!p->resume_sampler_state&&(int64_t)values[i]+2>p->steps)why="must leave at least two steps after warmup (N <= steps-2)";
        if(why){if(e&&n)snprintf(e,n,"%s %s",names[i],why);return 0;}
    }
    return 1;
}
int h3_warmups_match(const h3_params *p,const h3_params *s) {
    if(!h3_warmups_valid(p,NULL,0))return 0;
    if((p->adaptive_cache_warmup&&(int64_t)p->adaptive_cache_warmup+2>s->steps)||
       (p->subblock_warmup&&(int64_t)p->subblock_warmup+2>s->steps))return 0;
    return (!(p->adaptive_cache_warmup||p->adaptive_cache_warmup_set)||
            (s->adaptive_cache&&h3_adaptive_warmup(p->adaptive_cache_warmup)==h3_adaptive_warmup(s->adaptive_cache_warmup)))&&
           (!(p->subblock_warmup||p->subblock_warmup_set)||
            (s->cuda_attention==H3_ATTENTION_SUBBLOCK&&h3_subblock_warmup(p->subblock_warmup)==h3_subblock_warmup(s->subblock_warmup)));
}
int h3_adaptive_preflight(int mode,char *e,size_t n) {
    if(!mode)return 1;
    const char *why="adaptive cache requires a qualified SM120 device and CUDA 13 or newer";
#ifndef __APPLE__
    h3_device_info device;
    if(!h3_device_query(&device,e,n))return 0;
    if(mode>=1&&mode<=2&&device.cuda_compute_major==12&&device.cuda_compute_minor==0&&device.cuda_runtime_version>=13000)return 1;
#endif
    if (e && n)
        snprintf(e, n, "%s", why);
    return 0;
}
int h3_approximate_references_valid(const h3_params *p,char *e,size_t n) {
    const char *why=NULL;
    if(!p)return 0;
    if(p->adaptive_cache||p->cuda_attention==H3_ATTENTION_SUBBLOCK) {
        if(p->first_frame||p->last_frame)
            why="adaptive cache/SubBlock first/last-frame anchors are not qualified";
        else if(p->continuation&&p->cuda_denoise_quant)
            why="adaptive cache/SubBlock continuation requires BF16 projections";
        else if(p->reference_count) {
            if(p->cuda_denoise_quant)why="adaptive cache/SubBlock references require BF16; quantized references are not qualified";
            else if(p->reference_count>12||!p->references)why="adaptive cache/SubBlock requires 1..12 references with valid kinds";
            else {
                unsigned images=0,videos=0,audio=0;
                for(size_t i=0;i<p->reference_count;i++) {
                    const h3_reference *r=&p->references[i];
                    switch(r->kind) {
                    case H3_REFERENCE_IMAGE:images++;break;
                    case H3_REFERENCE_VIDEO:videos++;audio+=r->include_embedded_audio!=0;break;
                    case H3_REFERENCE_VIDEO_AUDIO:videos++;audio++;break;
                    case H3_REFERENCE_AUDIO:audio++;break;
                    default:why="SubBlock reference has an unknown kind";break;
                    }
                }
                if(!why&&(images>9||videos>3||audio>3))why="Ref2VA limits are 9 images, 3 videos, and 3 audio inputs";
                if(!why&&!images&&!videos)why="reference audio requires an image or video reference";
            }
        }
    }
    if(why){if(e&&n)snprintf(e,n,"%s",why);return 0;}return 1;
}
int h3_approximate_layout_valid(const h3_params *p,int continuation,int context,
    const h3_layout *l,char *e,size_t n) {
    if(!p||!l)return 0;
    if(!p->adaptive_cache&&p->cuda_attention!=H3_ATTENTION_SUBBLOCK)return 1;
    const char *why=NULL;h3_denoise_prefix prefix={0};
    if(l->frozen_audio||p->geometry_profile||p->still||p->save_upscale_state)
        why="adaptive cache/SubBlock require ordinary video geometry without upscale or anchors";
    else if(continuation!=0&&continuation!=1)why="invalid approximate continuation flag";
    else if(continuation) {
        if(p->cuda_denoise_quant)why="adaptive cache/SubBlock continuation requires BF16 projections";
        else if(context<39||context>362||!h3_continuation_context(context,&prefix)||
            prefix.video_prefix_t!=l->prefix.video_prefix_t||prefix.audio_prefix_t!=l->prefix.audio_prefix_t||
            context!=(p->continuation_context_frames?p->continuation_context_frames:39))
            why="invalid approximate continuation context/prefix";
        else if(p->continuation_mode==H3_CONTINUE_BRIDGE) {
            if(!l->bridge||!h3_bridge_profile_valid(l->bridge)||l->bridge->context_frames!=context||
               l->bridge->max_strength!=p->bridge_max_strength||l->bridge->type!=p->bridge_profile||
               l->bridge->video_bridge_t!=p->bridge_video_steps)why="invalid approximate bridge profile";
        } else if(p->continuation_mode!=H3_CONTINUE_HARD||l->bridge)why="invalid approximate continuation mode";
        if(!why){h3_adaptive_regions regions;if(!h3_adaptive_regions_build(l,&regions,e,n))return 0;}
    } else if(context||l->bridge||l->prefix.video_prefix_t||l->prefix.audio_prefix_t||
        p->continuation_mode==H3_CONTINUE_BRIDGE||p->keep_continuation_prefix)
        why="approximate prefix requires a continuation state";
    if(why){if(e&&n)snprintf(e,n,"%s",why);return 0;}return 1;
}
int h3_adaptive_lower_bound(const h3_params *p,char *e,size_t n) {
    if(!p||!p->adaptive_cache||p->resume_sampler_state)return 1;
    /* Geometry validation remains authoritative. Count the full target, including
     * inherited history; exact packed admission also includes text/references. */
    h3_av_state_info shape;
    int w=p->render_width?p->render_width:p->width,h=p->render_height?p->render_height:p->height;
    if(!h3_av_state_shape_profile(w,h,p->frames,p->geometry_profile,&shape))return 1;
    size_t rows=(size_t)shape.video_t*(shape.latent_h/2)*(shape.latent_w/2)+2*(size_t)shape.audio_t;
    h3_adaptive_plan plan;
    unsigned recipe=h3_adaptive_execution_recipe(p->adaptive_cache,p->cuda_denoise_quant,p->reference_count!=0,p->continuation!=NULL);
    return h3_adaptive_plan_recipe(p->adaptive_cache,recipe,rows,5376,p->adaptive_cache_max_bytes,&plan,e,n);
}
int h3_approximate_params_valid(const h3_params *p,int lora,char *e,size_t n) {
    if(p&&(!h3_warmups_valid(p,e,n)||!h3_adaptive_controls_valid(p,e,n)))return 0;
    if(p&&(p->adaptive_cache_max_bytes||p->adaptive_cache_max_bytes_set)) {
        const char *why=NULL;
        if(!p->adaptive_cache_max_bytes||p->adaptive_cache_max_bytes>SIZE_MAX)why="must be positive and fit platform size";
        else if(!p->adaptive_cache&&!p->resume_sampler_state)why="requires an enabled --adaptive-cache";
        if(why){if(e&&n)snprintf(e,n,"adaptive-cache-max-mib %s",why);return 0;}
    }
    if(p&&!h3_approximate_references_valid(p,e,n))return 0;
    if(p&&!h3_adaptive_lower_bound(p,e,n))return 0;
    const char *why=NULL;
    if(!p)why="missing approximate execution parameters";
    else if(p->adaptive_cache<0||p->adaptive_cache>2)why="adaptive-cache must be off, conservative or aggressive";
    else if(!isfinite(p->subblock_sparsity)||p->subblock_sparsity<0||p->subblock_sparsity>=1)
        why="subblock-sparsity must be finite and in [0,1)";
    else if(p->subblock_sparsity_set&&p->cuda_attention!=H3_ATTENTION_SUBBLOCK&&!p->resume_sampler_state)
        why="subblock-sparsity requires --cuda-attention subblock";
    else if(p->adaptive_cache||p->cuda_attention==H3_ATTENTION_SUBBLOCK) {
        const char *reduction=getenv("H3_TOKEN_REDUCTION");
        if(p->backend||p->backend_set||p->still||lora||
           p->denoise_reuse!=1||p->core_reuse!=1||p->dit_layers!=50||p->token_reduction||
           (reduction&&*reduction&&strcmp(reduction,"0"))||
           (p->cuda_attention!=0&&p->cuda_attention!=H3_ATTENTION_SUBBLOCK))
            why="adaptive cache/SubBlock require CUDA video, all 50 blocks, reuse/core-reuse 1, no token reduction or LoRA";
        else if(p->cuda_denoise_quant && p->adaptive_cache &&
                (p->cuda_attention || p->adaptive_cache!=H3_ADAPTIVE_CONSERVATIVE))
            why="adaptive quantization requires conservative cache and dense attention; triple combinations and aggressive cache are not qualified";
#ifdef __APPLE__
        else why="adaptive cache/SubBlock require a qualified SM120 CUDA device; Metal is unsupported";
#endif
    }
    if(why){if(e&&n)snprintf(e,n,"%s",why);return 0;}return 1;
}
