#include "src/sglang/sglang.h"
/* .h3sample v2: explicit LE scalars and lossless IEEE/BF16 tensor payloads.
 * Bounds/integrity validation precedes deserialization and model allocation. */
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/digest.h"
#include "src/weights/q8.h"
#include "src/denoise/attention.h"
#include "src/denoise/subblock.h"
#include "src/denoise/approximate.h"
#include <math.h>
#include <errno.h>
#include <float.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/mman.h>
#include <sys/stat.h>
#include <unistd.h>
#include <fcntl.h>

_Static_assert(sizeof(float)==4 && FLT_MANT_DIG==24 && sizeof(double)==8 && DBL_MANT_DIG==53,"IEEE floats required");
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"
enum { HEADER=96, ENTRY=72, SECTIONS=46, MAX_SECTIONS=256, REQUIRED=1 };
#define MAX_FILE (UINT64_C(16)*1024*1024*1024)
static const uint8_t magic[8]={'H','3','S','A','M','P','L','E'};
static const uint8_t up_magic[8]={'H','3','U','P','S','R','C',1};
static int fail(char *e,size_t n,const char *m) { if(e&&n) snprintf(e,n,"h3sample: %s",m); return 0; }
static uint64_t get(const uint8_t *p,unsigned n) { uint64_t v=0; for(unsigned i=0;i<n;i++) v|=(uint64_t)p[i]<<(i*8); return v; }
static void put(uint8_t *p,uint64_t v,unsigned n) { for(unsigned i=0;i<n;i++) p[i]=(uint8_t)(v>>(i*8)); }
static int little(void) { uint32_t one=1; return *(uint8_t *)&one==1; }
static void update(h3_sha256_ctx *h,const void *v,size_t n) {
    const uint8_t *p=v; while(n) { h3_sha256_size k=(h3_sha256_size)(n>1048576?1048576:n); h3_sha256_update(h,p,k); n-=k; p+=k; }
}
typedef struct { uint8_t *data; size_t length,offset,capacity; int read,ok,measure; } buffer;
static void bytes(buffer *b,void *p,size_t n) {
    if(!b->ok) return;
    if(n>MAX_FILE || b->offset>MAX_FILE-n || (n&&!p)) { b->ok=0; return; }
    if(b->measure) { b->offset+=n;b->length=b->offset;return; }
    if(b->read) {
        if(n>b->length-b->offset) { b->ok=0; return; }
        if(n) memcpy(p,b->data+b->offset,n);
    } else {
        if(b->offset+n>b->capacity) {
            size_t cap=b->offset+n; void *v=realloc(b->data,cap?cap:1);
            if(!v) { b->ok=0; return; } b->data=v; b->capacity=cap;
        }
        if(n) memcpy(b->data+b->offset,p,n);
        b->length=b->offset+n;
    }
    b->offset+=n;
}
static uint64_t integer(buffer *b,uint64_t v,unsigned n) {
    uint8_t raw[8]={0}; if(!b->read) put(raw,v,n); bytes(b,raw,n); return get(raw,n);
}
static void i32(buffer *b,int *v) { uint32_t bits=(uint32_t)*v; bits=(uint32_t)integer(b,bits,4); if(b->read) memcpy(v,&bits,4); }
static void u32(buffer *b,uint32_t *v) { uint32_t n=(uint32_t)integer(b,*v,4); if(b->read) *v=n; }
static void u64(buffer *b,uint64_t *v) { uint64_t n=integer(b,*v,8); if(b->read) *v=n; }
static void count(buffer *b,size_t *v,size_t max) { uint64_t n=integer(b,*v,8); if(n>max || n>SIZE_MAX) b->ok=0; else if(b->read) *v=(size_t)n; }
static void f32(buffer *b,float *v) { uint32_t n; memcpy(&n,v,4); u32(b,&n); if(b->read) memcpy(v,&n,4); }
static void f64(buffer *b,double *v) { uint64_t n; memcpy(&n,v,8); u64(b,&n); if(b->read) memcpy(v,&n,8); }
static void array(buffer *b,void **p,size_t n,size_t width) {
    if(!b->ok || !width || n>SIZE_MAX/width || n*width>MAX_FILE) { b->ok=0; return; }
    size_t length=n*width;
    if(b->read) {
        if(length>b->length-b->offset) { b->ok=0; return; }
        *p=length?malloc(length):NULL; if(length&&!*p) { b->ok=0; return; }
    }
    bytes(b,*p,length);
}
static void string(buffer *b,char **s) {
    size_t n=*s?strlen(*s):0; count(b,&n,1048576);
    if(b->read && b->ok) {
        if(n>b->length-b->offset) { b->ok=0; return; }
        *s=calloc(n+1,1); if(!*s) { b->ok=0; return; }
    }
    bytes(b,*s,n); if(b->read && b->ok && (!*s || memchr(*s,0,n))) b->ok=0;
}
#define PARAMS(X) X(width) X(height) X(frames) X(steps) X(reference_image_size) \
    X(denoise_reuse) X(dit_layers) X(core_reuse) X(token_reduction) X(use_int8_row_fc2) \
    X(use_reference_rope) X(ssd_streaming) X(render_width) X(render_height) \
    X(use_slower_bf16_mlp) X(use_slower_bf16_qkv) X(use_slower_bf16_attention_output) \
    X(use_slower_row_major_attention_output) X(use_slower_unfused_int8_inputs) \
    X(use_slower_unfused_qkv_rope) X(use_slower_scalar_qkv_rms) X(use_slower_uncached_int8_scales) \
    X(use_slower_dynamic_fc1_k) X(use_slower_grouped_quantizer) X(continuation_context_frames) \
    X(keep_continuation_prefix) X(continuation_mode) X(bridge_video_steps) X(bridge_profile)

static void rng(buffer *b,h3_rng *r) { u64(b,&r->state); u64(b,&r->increment); f32(b,&r->spare); i32(b,&r->has_spare); }
static void section(buffer *b,unsigned id,unsigned format_version,h3_sampler_state *s) {
    switch(id) {
    case 1: {
        u32(b,&s->version); i32(b,&s->total_steps); i32(b,&s->next_step); i32(b,&s->reuse_interval);
#define FIELD(x) do { int v=(int)s->params.x; i32(b,&v); s->params.x=v; } while(0);
        PARAMS(FIELD)
#undef FIELD
        u64(b,&s->params.seed); count(b,&s->params.reference_count,12); f32(b,&s->params.bridge_max_strength);
        i32(b,&s->ref2va); i32(b,&s->conditioned); i32(b,&s->render_width); i32(b,&s->render_height); i32(b,&s->aligned_frames);
        i32(b,&s->latent_t); i32(b,&s->latent_h); i32(b,&s->latent_w); i32(b,&s->audio_t); f32(b,&s->spatial_rope_scale);
        count(b,&s->video_elements,MAX_FILE/4); count(b,&s->audio_elements,MAX_FILE/4);
        count(b,&s->condition_video_elements,MAX_FILE/4); count(b,&s->condition_audio_elements,MAX_FILE/4);
        count(b,&s->text.tokens,1000000); count(b,&s->text.width,H3_TEXT_HIDDEN_SIZE); count(b,&s->reference_count,12);
        u32(b,&s->sampler_mode);
        s->params.stop_after_step=-1;
        break;
    }
    case 2:
        u32(b,&s->backend_version); string(b,&s->build_id); string(b,&s->environment);
        bytes(b,s->device.name,sizeof(s->device.name)); bytes(b,s->device.architecture,sizeof(s->device.architecture));
        u64(b,&s->device.physical_memory); u64(b,&s->device.recommended_working_set); u64(b,&s->device.max_buffer_length);
        i32(b,&s->device.apple_gpu_family); i32(b,&s->device.metal4); i32(b,&s->device.unified_memory); break;
    case 3: bytes(b,s->model_fingerprint,32); bytes(b,s->av_signature,32); break;
    case 4:
        string(b,&s->prompt); count(b,&s->provenance_bytes,1048576);
        array(b,(void **)&s->provenance,s->provenance_bytes,1); break;
    case 5:
        if(s->text.width && s->text.tokens>SIZE_MAX/s->text.width) { b->ok=0; break; }
        array(b,(void **)&s->text.values,s->text.tokens*s->text.width,2); break;
    case 6: {
        size_t n=s->text.tags?s->text.tokens:0; count(b,&n,s->text.tokens);
        if(n && n!=s->text.tokens) { b->ok=0; break; }
        array(b,(void **)&s->text.tags,n,1); break;
    }
    case 7: array(b,(void **)&s->condition_video,s->condition_video_elements,4); break;
    case 8: array(b,(void **)&s->condition_audio,s->condition_audio_elements,4); break;
    case 9:
        if(b->read && b->ok) { s->references=calloc(s->reference_count?s->reference_count:1,sizeof(*s->references)); if(!s->references) b->ok=0; }
        for(size_t i=0;i<s->reference_count && b->ok;i++) {
            h3_layout_ref *r=&s->references[i]; int kind=(int)r->kind; i32(b,&kind); if(b->read) r->kind=kind;
            i32(b,&r->latent_t); i32(b,&r->latent_h); i32(b,&r->latent_w); i32(b,&r->audio_t);
        } break;
    case 10: {
        h3_layout *l=&s->layout;
        count(b,&l->seq_len,10000000); count(b,&l->segment_count,1000);
        count(b,&l->img_cond_rows,MAX_FILE/96/4); count(b,&l->img_target_rows,MAX_FILE/96/4);
        count(b,&l->audio_cond_rows,MAX_FILE/32/4); count(b,&l->audio_target_rows,MAX_FILE/32/4);
        for(int i=0;i<5;i++) i32(b,&l->signature[i]);
        i32(b,&l->prefix.video_prefix_t); i32(b,&l->prefix.audio_prefix_t);
        if(b->read && b->ok) {
            if(l->segment_count>(b->length-b->offset)/20 || l->seq_len>(b->length-b->offset-l->segment_count*20)/24) { b->ok=0; break; }
            l->segments=calloc(l->segment_count?l->segment_count:1,sizeof(*l->segments)); l->positions=calloc(l->seq_len?l->seq_len:1,sizeof(*l->positions));
            if(!l->segments||!l->positions) { b->ok=0; break; }
        }
        for(size_t i=0;i<l->segment_count && b->ok;i++) { count(b,&l->segments[i].start,l->seq_len); count(b,&l->segments[i].stop,l->seq_len); int kind=(int)l->segments[i].kind; i32(b,&kind); if(b->read) l->segments[i].kind=kind; }
        for(size_t i=0;i<l->seq_len && b->ok;i++) { f64(b,&l->positions[i].t); f64(b,&l->positions[i].h); f64(b,&l->positions[i].w); }
        break;
    }
    case 11:
        i32(b,&s->sigmas.steps);
        if(s->sigmas.steps<1 || s->sigmas.steps>H3_MAX_STEPS || s->sigmas.steps!=s->total_steps) { b->ok=0; break; }
        bytes(b,s->sigmas.video,((size_t)s->sigmas.steps+1)*4); bytes(b,s->sigmas.audio,((size_t)s->sigmas.steps+1)*4); break;
    case 12: array(b,(void **)&s->video,s->video_elements,4); break;
    case 13: array(b,(void **)&s->audio,s->audio_elements,4); break;
    case 14:
        u32(b,&s->rng_version); rng(b,&s->video_rng); rng(b,&s->audio_rng);
        u64(b,&s->video_random_count); u64(b,&s->audio_random_count); break;
    case 15: {
        i32(b,&s->continuation); i32(b,&s->context_frames);
        bytes(b,s->source_fingerprint,32); bytes(b,s->video_tail_hash,32); bytes(b,s->audio_tail_hash,32);
        f32(b,&s->clean_coefficient); f32(b,&s->noise_coefficient); i32(b,&s->audio_preservation);
        int bridge=s->layout.bridge!=NULL; i32(b,&bridge); if(bridge<0||bridge>1) { b->ok=0; break; }
        if(bridge) {
            h3_bridge_profile *p=&s->bridge; s->layout.bridge=p;
            i32(b,&p->prefix.video_prefix_t); i32(b,&p->prefix.audio_prefix_t); i32(b,&p->context_frames);
            i32(b,&p->video_bridge_t); i32(b,&p->video_exact_t); i32(b,&p->bridge_frames); i32(b,&p->audio_bridge_t); i32(b,&p->audio_exact_t);
            int type=(int)p->type; i32(b,&type); p->type=type; f32(b,&p->max_strength); i32(b,&p->class_count);
            bytes(b,p->active,sizeof(p->active)); bytes(b,p->class_mask,sizeof(p->class_mask));
            bytes(b,p->video_classes,sizeof(p->video_classes)); bytes(b,p->audio_classes,sizeof(p->audio_classes));
        } break;
    }
    case 16:
        i32(b,&s->last_evaluated); i32(b,&s->previous_evaluated);
        if(s->total_steps<1 || s->total_steps>H3_MAX_STEPS) { b->ok=0; break; }
        bytes(b,s->selected,(size_t)s->total_steps); break;
    case 17: array(b,(void **)&s->last_video_velocity,s->video_elements,4); break;
    case 18: array(b,(void **)&s->previous_video_velocity,s->video_elements,4); break;
    case 19: array(b,(void **)&s->last_audio_velocity,s->audio_elements,4); break;
    case 20: array(b,(void **)&s->previous_audio_velocity,s->audio_elements,4); break;
    case 21: array(b,(void **)&s->original_video_noise,s->video_elements,4); break;
    case 22: array(b,(void **)&s->original_audio_noise,s->audio_elements,4); break;
    case 23: {
        count(b,&s->token_count,1000000); array(b,(void **)&s->token_ids,s->token_count,4);
        int positions=s->presentation_positions!=NULL; i32(b,&positions);
        if (positions<0 || positions>1) { b->ok=0; break; }
        if (positions) array(b,(void **)&s->presentation_positions,s->token_count*3,4);
        count(b,&s->presentation_span_count,1000000);
        array(b,(void **)&s->presentation_spans,s->presentation_span_count*2,8); break;
    }
    case 24:
        u32(b,&s->execution_version); u32(b,&s->core_forward_count); u32(b,&s->core_residual_ready);
        count(b,&s->core_rows,10000000); count(b,&s->core_columns,5376); count(b,&s->core_elements,MAX_FILE/2);
        u32(b,&s->reduction_enabled); u32(b,&s->reduction_active);
        u32(b,&s->reduction_begin); u32(b,&s->reduction_end);
        u32(b,&s->reduction_early_steps); u32(b,&s->reduction_early_end); f32(b,&s->reduction_scale);
        count(b,&s->full_sequence,10000000); count(b,&s->reduced_sequence,10000000); break;
    case 25: array(b,(void **)&s->gpu_last_video,s->video_elements,2); break;
    case 26: array(b,(void **)&s->gpu_previous_video,s->video_elements,2); break;
    case 27: array(b,(void **)&s->gpu_last_audio,s->audio_elements,2); break;
    case 28: array(b,(void **)&s->gpu_previous_audio,s->audio_elements,2); break;
    case 29: array(b,(void **)&s->core_residual,s->core_elements,2); break;
    case 30:
        u32(b,&s->prepared.version); bytes(b,s->prepared.key,32); count(b,&s->prepared.count,H3_PREPARED_MAX);
        for(size_t i=0;i<s->prepared.count && b->ok;i++) {
            h3_prepared_tensor *t=&s->prepared.tensors[i];
            u32(b,&t->id); count(b,&t->elements,MAX_FILE/2); array(b,(void **)&t->values,t->elements,2);
        } break;
    case 31:
        u32(b,&s->resume_count); u32(b,&s->resume_step); u32(b,&s->resume_format); bytes(b,s->resume_hash,32); break;
    case 32:
        u32(b,&s->refvideo_pipeline); break;
    case 46:i32(b,&s->params.geometry_profile);if(s->params.geometry_profile!=1)b->ok=0;break;
    case 45: {
        h3_upscale_record *u=&s->upscale;
        u32(b,&u->version);u32(b,&u->stage);u32(b,&u->semantic_policy);u32(b,&u->metadata_identity);
        i32(b,&u->source_width);i32(b,&u->source_height);u32(b,&u->keyframe_count);
        for(int j=0;j<2;j++)i32(b,&u->keyframes[j]);
        for(int j=0;j<12;j++) {
            i32(b,&u->original_width[j]);i32(b,&u->original_height[j]);
            i32(b,&u->semantic_width[j]);i32(b,&u->semantic_height[j]);
        }
        bytes(b,u->audio_hash,32);
        if(format_version==2) {
            u32(b,&u->recipe);u32(b,&u->refinement_steps);f32(b,&u->sigma);u64(b,&u->seed);
            bytes(b,u->parent_hash,32);bytes(b,u->artifact_hash,32);bytes(b,u->noise_hash,32);bytes(b,u->condition_hash,32);
            s->layout.frozen_audio=1;
            if(u->version!=2||u->stage!=2)b->ok=0;
        } else if(u->version!=1||u->stage!=1)b->ok=0;
        break;
    }
    case 44: {
        bytes(b,s->device.backend,sizeof(s->device.backend));
        bytes(b,s->device.cuda_uuid,sizeof(s->device.cuda_uuid));
        i32(b,&s->device.device_index);i32(b,&s->device.cuda_compute_major);i32(b,&s->device.cuda_compute_minor);
        i32(b,&s->device.cuda_runtime_version);i32(b,&s->device.cuda_driver_version);
        if(!memchr(s->device.backend,0,sizeof(s->device.backend)))b->ok=0;
        {
            int adaptive=s->params.adaptive_cache_warmup,subblock=s->params.subblock_warmup;
            i32(b,&adaptive);i32(b,&subblock);
            if((s->params.cuda_attention==H3_ATTENTION_SUBBLOCK&&
                h3_subblock_warmup(subblock)!=h3_subblock_warmup(s->params.subblock_warmup)))b->ok=0;
            s->params.adaptive_cache_warmup=adaptive;s->params.subblock_warmup=subblock;
        }
        break;
    }
    case 40:
        i32(b,&s->params.adaptive_cache);u32(b,&s->adaptive_version);
        u32(b,&s->adaptive_history.ready);u32(b,&s->adaptive_history.streak);u32(b,&s->adaptive_history.phase);
        i32(b,&s->adaptive_history.last_step);i32(b,&s->adaptive_history.last_refresh);
        count(b,&s->adaptive_elements,(size_t)(MAX_FILE/4));
        {
            uint64_t budget=b->read?0:h3_adaptive_budget(s->params.adaptive_cache_max_bytes);
            u64(b,&budget);
            if(!budget||budget>SIZE_MAX)b->ok=0;
            if(b->read)s->params.adaptive_cache_max_bytes=budget;
        }
        {
            float threshold=b->read?0:h3_adaptive_threshold(&s->params);
            int hits=b->read?0:h3_adaptive_max_hits(&s->params);
            f32(b,&threshold);i32(b,&hits);
            if(!isfinite(threshold)||threshold<0||threshold>1||hits<1||hits>16)b->ok=0;
            if(b->read){s->params.adaptive_cache_threshold=threshold;s->params.adaptive_cache_max_hits=hits;
                s->params.adaptive_cache_threshold_set=s->params.adaptive_cache_max_hits_set=1;}
        }
        if(s->params.adaptive_cache<1||s->params.adaptive_cache>2||
           s->adaptive_version!=h3_adaptive_execution_recipe(s->params.adaptive_cache,s->params.cuda_denoise_quant,s->reference_count!=0,s->continuation))b->ok=0;
        break;
    case 41: array(b,(void **)&s->adaptive_anchor,s->adaptive_elements,2);break;
    case 42: array(b,(void **)&s->adaptive_delta,s->adaptive_elements,2);break;
    case 43: {
        uint32_t expected=h3_attention_execution_recipe(H3_ATTENTION_SUBBLOCK,s->params.cuda_denoise_quant);
        uint32_t recipe=expected,plan=1,block=64,cell=16,warmup=(uint32_t)h3_subblock_warmup(s->params.subblock_warmup);
        u32(b,&recipe);u32(b,&plan);u32(b,&block);u32(b,&cell);u32(b,&warmup);
        f32(b,&s->params.subblock_sparsity);
        if(recipe!=expected||plan!=1||block!=64||cell!=16||warmup<2||warmup>16||s->params.cuda_attention!=H3_ATTENTION_SUBBLOCK||
           !isfinite(s->params.subblock_sparsity)||s->params.subblock_sparsity<0||s->params.subblock_sparsity>=1)b->ok=0;
        if(b->read)s->params.subblock_warmup=warmup==10?0:(int)warmup;
        break;
    }
    case 39:
        i32(b,&s->params._arithmetic_recipe);
        if(s->params._arithmetic_recipe!=H3_SGLANG_VERSION)b->ok=0;
        break;
    case 35:
        i32(b,&s->params.cuda_attention); u32(b,&s->attention_version); u32(b,&s->attention_plan); break;
    case 36: {
        int backend=s->params.backend, attention=s->params.attention_mode;
        i32(b,&backend);i32(b,&attention);u32(b,&s->metal_attention_version);
        s->params.backend=backend;s->params.attention_mode=attention;
        if(s->metal_attention_version==H3_METAL_ATTENTION_VERSION) {
            h3_metal_attention_options *m=&s->params.metal_attention;
            /* Keep the version-6 native-backend marker and wire layout. */
            int native_marker=1;
            i32(b,&native_marker);
            if(native_marker!=1)b->ok=0;
            i32(b,&m->candidate);i32(b,&m->q_block);i32(b,&m->kv_block);
            i32(b,&m->dense_layers);i32(b,&m->local_radius);f32(b,&m->tau);f32(b,&m->min_exact);
            i32(b,&m->precision);i32(b,&m->tier);
            i32(b,&m->dense_steps);f32(b,&m->dense_sigma);i32(b,&m->layout_fusion);
            i32(b,&m->ane_mode);i32(b,&m->ane_rows);i32(b,&m->ane_chunk);
            u64(b,&s->ane_decided);for(int j=0;j<50;j++)u32(b,&s->ane_rows[j]);
        }
        break;
    }
    case 34:
        i32(b,&s->params.cuda_denoise_quant); u32(b,&s->quant_version); break;
    case 38: {
        h3_cuda_sol_options *o=&s->params.cuda_sol;
        uint32_t recipe=H3_CUDA_SOL_VERSION,plan=H3_CUDA_SOL_PLAN_VERSION;
        u32(b,&recipe);u32(b,&plan);
        i32(b,&o->q_block);i32(b,&o->kv_block);i32(b,&o->dense_layers);i32(b,&o->dense_steps);i32(b,&o->local_radius);
        f32(b,&o->tau);f32(b,&o->min_exact);f32(b,&o->dense_sigma);
        if(s->params.cuda_attention!=H3_ATTENTION_SOL||recipe!=H3_CUDA_SOL_VERSION||plan!=H3_CUDA_SOL_PLAN_VERSION||
           !h3_cuda_sol_options_valid(*o,NULL,0))b->ok=0;
        break;
    }
    case 37: {
        int format=(int)s->params.metal_attention.weight_format;
        uint32_t version=H3_Q8_VERSION,group=H3_Q8_GROUP;
        i32(b,&format);u32(b,&version);u32(b,&group);i32(b,&s->params.metal_attention.q8_kernel);
        if(format!=H3_WEIGHT_Q8||version!=H3_Q8_VERSION||group!=H3_Q8_GROUP)b->ok=0;
        s->params.metal_attention.weight_format=(h3_weight_format)format;
        break;
    }
    default: b->ok=0;
    }
}
static unsigned dtype(unsigned id) { return (id==5 || id==41 || id==42 || (id>=25 && id<=29))?3:((id==7||id==8||(id>=12&&id<=13)||(id>=17&&id<=22))?2:1); }
static unsigned section_version(unsigned id,const h3_sampler_state *s) {
    if(id==40)return 3;
    if(id==45&&s->upscale.stage==2)return 2;
    return id==44?2u:1u;
}
static int supported_version(unsigned id,uint64_t version) { return id!=33 && (id==45 ? (version==1||version==2) : version==(id==40?3u:id==44?2u:1u)); }
static int needed(unsigned id,const h3_sampler_state *s) { return (id==21 && s->upscale.stage==2) || (id==46 && s->params.geometry_profile) || (id==45 && s->upscale.stage) || id<=16 || (id>=17 && id<=20 && s->reuse_interval>1 && !s->sampler_mode) || id==24 ||
    (id>=25 && id<=28 && s->reuse_interval>1 && s->sampler_mode) || (id==29 && s->params.core_reuse>1) || id==32 || (id==34 && s->params.cuda_denoise_quant) || (id==35 && s->params.cuda_attention) || (id==36 && s->params.backend) || (id==37 && s->params.metal_attention.weight_format) || (id==38 && s->params.cuda_attention==H3_ATTENTION_SOL) || (id==39 && s->params._arithmetic_recipe) || (id>=40 && id<=42 && s->params.adaptive_cache) || (id==43 && s->params.cuda_attention==H3_ATTENTION_SUBBLOCK) || (id==44 && (s->upscale.stage || s->params.adaptive_cache || s->params.cuda_attention==H3_ATTENTION_SUBBLOCK)); }
static int present(unsigned id,const h3_sampler_state *s) {
    return needed(id,s) || (id==21 && s->original_video_noise) || (id==22 && s->original_audio_noise) || (id==23 && s->token_ids) || (id==30 && s->prepared.count) || (id==31 && s->resume_count);
}
static int envelope(const uint8_t *data,size_t length,int source,char *error,size_t size) {
    if(length<HEADER || memcmp(data,source?up_magic:magic,8)) return fail(error,size,"truncated header or wrong magic");
    if(get(data+8,4)!=2 || get(data+12,4)!=HEADER || get(data+16,4)!=0 || get(data+32,4)!=0x01020304)
        return fail(error,size,"unsupported schema, header flags or endianness");
    for(int i=36;i<64;i++) if(data[i]) return fail(error,size,"nonzero reserved header field");
    uint64_t n=get(data+20,4);
    if(!n || n>MAX_SECTIONS || get(data+24,8)!=length || n>(length-HEADER)/ENTRY) return fail(error,size,"invalid section count/file length");
    uint8_t header[HEADER],hash[32]; memcpy(header,data,HEADER); memset(header+64,0,32);
    h3_sha256_ctx h; h3_sha256_init(&h); update(&h,header,HEADER); update(&h,data+HEADER,length-HEADER); h3_sha256_final(hash,&h);
    if(memcmp(hash,data+64,32)) return fail(error,size,"whole-file checksum mismatch");
    uint64_t end=HEADER+n*ENTRY;
    for(uint64_t i=0;i<n;i++) {
        const uint8_t *entry=data+HEADER+i*ENTRY;
        uint64_t id=get(entry,4),version=get(entry+4,4),flags=get(entry+8,4),dt=get(entry+12,4);
        uint64_t count_value=get(entry+16,8),offset=get(entry+24,8),bytes_value=get(entry+32,8);
        if(!id || flags>1 || offset!=end || offset>length || bytes_value>length-offset)
            return fail(error,size,"invalid section flags, overlapping offsets or byte count");
        for(uint64_t j=0;j<i;j++) if(get(data+HEADER+j*ENTRY,4)==id) return fail(error,size,"duplicate section");
        if(flags==REQUIRED && (id>SECTIONS || !supported_version((unsigned)id,version))) return fail(error,size,"unsupported required section/version");
        if(id<=SECTIONS && supported_version((unsigned)id,version)) {
            unsigned width=dt==2?4:dt==3?2:1;
            if(dt!=dtype((unsigned)id) || count_value>UINT64_MAX/width || count_value*width!=bytes_value)
                return fail(error,size,"section dtype/element count mismatch");
        }
        h3_sampler_hash(data+offset,(size_t)bytes_value,hash);
        if(memcmp(hash,entry+40,32)) return fail(error,size,"section checksum mismatch");
        end=offset+bytes_value;
    }
    return end==length?1:fail(error,size,"unclaimed trailing bytes");
}
static int mapped(const char *path,void **data,size_t *length,int *fd,char *error,size_t size) {
    *fd=path&&*path?open(path,O_RDONLY):-1; struct stat st;
    if(*fd<0 || fstat(*fd,&st) || !S_ISREG(st.st_mode) || st.st_size<HEADER || (uint64_t)st.st_size>MAX_FILE) {
        if (*fd >= 0)
            close(*fd);
        *fd = -1;
        return fail(error, size, "cannot open state or invalid/truncated file size");
    }
    *length=(size_t)st.st_size; *data=mmap(NULL,*length,PROT_READ,MAP_PRIVATE,*fd,0);
    if(*data==MAP_FAILED) { close(*fd); *fd=-1; return fail(error,size,"cannot map checkpoint"); }
    return 1;
}
int h3_sampler_file_validate(const char *path,int source,char *error,size_t size) {
    void *data=NULL;size_t length=0;int fd=-1;
    if(!mapped(path,&data,&length,&fd,error,size))return 0;
    int ok=envelope(data,length,source,error,size);
    munmap(data,length);close(fd);return ok;
}
int h3_sampler_model_requirements(const char *path, int source, int *ref2va,
    int *upscale, char *error, size_t size) {
    int fd = path?open(path, O_RDONLY|O_CLOEXEC):-1;
    if(fd<0)return fail(error,size,"cannot open state or invalid/truncated file size");
    struct stat st;
    uint8_t header[HEADER], table[MAX_SECTIONS*ENTRY], payload[4096];
    int ok = fd >= 0 && !fstat(fd, &st) && S_ISREG(st.st_mode) &&
        st.st_size >= HEADER && (uint64_t)st.st_size <= MAX_FILE &&
        pread(fd, header, HEADER, 0) == HEADER;
    uint64_t n = ok ? get(header+20, 4) : 0;
    ok = ok && !memcmp(header, source ? up_magic : magic, 8) &&
        get(header+8, 4) == 2 && get(header+12, 4) == HEADER &&
        get(header+24, 8) == (uint64_t)st.st_size &&
        get(header+32, 4) == 0x01020304 && n && n <= MAX_SECTIONS;
    if (ok) ok = pread(fd, table, (size_t)n*ENTRY, HEADER) == (ssize_t)(n*ENTRY);
    int found = 0;
    *ref2va = 0; *upscale = 0;
    for (uint64_t i = 0; ok && i < n; i++) {
        const uint8_t *entry = table+i*ENTRY;
        uint64_t id = get(entry, 4), offset = get(entry+24, 8), bytes_value = get(entry+32, 8);
        if (id == 45) *upscale = 1;
        if (id != 1) continue;
        if (found || !supported_version(1, get(entry+4, 4)) ||
            bytes_value > sizeof(payload) || offset < HEADER+n*ENTRY ||
            offset > (uint64_t)st.st_size || bytes_value > (uint64_t)st.st_size-offset ||
            pread(fd, payload, (size_t)bytes_value, (off_t)offset) != (ssize_t)bytes_value) {
            ok = 0; break;
        }
        uint8_t hash[32];
        h3_sampler_hash(payload, (size_t)bytes_value, hash);
        if (memcmp(hash, entry+40, 32)) { ok = 0; break; }
        h3_sampler_state state = {0};
        buffer b = {.data=payload, .length=(size_t)bytes_value, .read=1, .ok=1};
        section(&b, 1, (unsigned)get(entry+4, 4), &state);
        ok = b.ok && b.offset == b.length && (state.ref2va == 0 || state.ref2va == 1);
        *ref2va = state.ref2va; found = 1;
    }
    if (fd >= 0) close(fd);
    return ok && found ? 1 : fail(error, size, "cannot read current sampler model requirements");
}
static h3_sampler_state *load_state(const char *path,int source,uint64_t budget,int explicit_budget,char *error,size_t size) {
    double start=h3_av_now(); if(error&&size) *error=0;
    if(!little()) { fail(error,size,"unsupported host endianness"); return NULL; }
    if(explicit_budget&&(!budget||budget>SIZE_MAX)) {fail(error,size,"adaptive cache override must be positive and fit platform size");return NULL;}
    void *mapping=NULL; size_t length=0; int fd=-1;
    if(!mapped(path,&mapping,&length,&fd,error,size)) return NULL;
    const uint8_t *data=mapping; h3_sampler_state *s=NULL;
    if(source && length>(UINT64_C(2)<<30)) {fail(error,size,"clean source exceeds 2 GiB container bound");goto done;}
    if(!envelope(data,length,source,error,size)) goto done;
    s=calloc(1,sizeof(*s)); if(!s) { fail(error,size,"out of memory"); goto done; }
    const uint8_t *entries[SECTIONS+1]={0};
    for(uint64_t i=0;i<get(data+20,4);i++) {
        const uint8_t *e=data+HEADER+i*ENTRY; uint64_t id=get(e,4);
        if(id<=SECTIONS && supported_version((unsigned)id,get(e+4,4))) entries[id]=e;
    }
    if(!entries[40]&&(entries[41]||entries[42])) {fail(error,size,"orphan adaptive tensor section");goto invalid;}
    if(explicit_budget&&!entries[40]) {fail(error,size,"adaptive-cache-max-mib requires an adaptive checkpoint");goto invalid;}
    if(entries[46]) {
        const uint8_t *e=entries[46];
        buffer b={(uint8_t *)data+get(e+24,8),(size_t)get(e+32,8),0,0,1,1,0};section(&b,46,1,s);
        if(source||get(e+8,4)!=REQUIRED||!b.ok||b.offset!=b.length) {
            fail(error,size,"invalid required geometry profile");goto invalid;
        }
    }
    if(source||entries[45]) {
        const uint8_t *e=entries[45];
        if(!e||get(e+8,4)!=REQUIRED) {fail(error,size,"missing required clean source record");goto invalid;}
        buffer b={(uint8_t *)data+get(e+24,8),(size_t)get(e+32,8),0,0,1,1,0};section(&b,45,(unsigned)get(e+4,4),s);
        if(!b.ok||b.offset!=b.length||s->upscale.stage!=(source?1u:2u)||s->upscale.version!=(source?1u:2u)||
           s->upscale.semantic_policy!=1||s->upscale.metadata_identity!=1) {
            fail(error,size,"invalid clean source record");goto invalid;
        }
        if(source)for(unsigned id=17;id<=SECTIONS;id++)if(entries[id]&&id!=23&&id!=24&&id!=32&&id!=39&&id!=44&&id!=45) {
            fail(error,size,"disposable or unsupported source section");goto invalid;
        }
    }
    int adaptive_metadata_read=0;
    for(unsigned id=1;id<=SECTIONS;id++) {
        const uint8_t *e=entries[id];
        if(id==45||((id==43||id==44)&&adaptive_metadata_read))continue;
        if(!e) { if(needed(id,s)) { fail(error,size,"missing required state section"); goto invalid; } continue; }
        if ((id==34||id==35||id==36||id==37||id==38||id==39||id>=40) && get(e+8,4)!=REQUIRED) {
            fail(error,size,id==34?"quantization policy must be required":id==37?"Q8 weight policy must be required":"attention policy must be required"); goto invalid;
        }
        if(needed(id,s) && get(e+8,4)!=REQUIRED) { fail(error,size,"required section marked optional"); goto invalid; }
        buffer b={(uint8_t *)data+get(e+24,8),(size_t)get(e+32,8),0,0,1,1,0}; section(&b,id,(unsigned)get(e+4,4),s);
        if(!b.ok || b.offset!=b.length) { if(error&&size) snprintf(error,size,"h3sample: invalid section %u dimensions/length",id); goto invalid; }
        if(id==40) {
            /* Warmup/device metadata follows the cache payloads on disk. Read
             * these small records first so history can be checked before 41/42. */
            for(unsigned meta=43;meta<=44;meta++) {
                const uint8_t *m=entries[meta];
                if(!m){if(needed(meta,s)){fail(error,size,"missing required state section");goto invalid;}continue;}
                if(get(m+8,4)!=REQUIRED){fail(error,size,"attention policy must be required");goto invalid;}
                buffer small={(uint8_t *)data+get(m+24,8),(size_t)get(m+32,8),0,0,1,1,0};
                section(&small,meta,(unsigned)get(m+4,4),s);
                if(!small.ok||small.offset!=small.length){fail(error,size,"invalid adaptive warmup/device metadata before allocation");goto invalid;}
            }
            adaptive_metadata_read=1;
            h3_adaptive_plan plan;
            if(!s->full_sequence||s->full_sequence!=s->layout.seq_len||
               !h3_adaptive_plan_recipe(s->params.adaptive_cache,s->adaptive_version,s->full_sequence,5376,
                   s->params.adaptive_cache_max_bytes,&plan,error,size)||
               !h3_sampler_layout_validate(s,error,size)||
               !h3_sampler_adaptive_metadata_validate(s,error,size)) {
                if (error && size && !*error)
                    fail(error, size, "invalid adaptive layout before allocation");
                goto invalid;
            }
            if(s->adaptive_elements!=plan.elements||!entries[41]||!entries[42]||
               get(entries[41]+8,4)!=REQUIRED||get(entries[42]+8,4)!=REQUIRED||
               get(entries[41]+32,8)!=plan.tensor_bytes||get(entries[42]+32,8)!=plan.tensor_bytes) {
                fail(error,size,"invalid adaptive counts or payload lengths before allocation");goto invalid;
            }
            if(explicit_budget) {
                if(!h3_adaptive_plan_admit(&plan,budget,error,size))goto invalid;
                s->params.adaptive_cache_max_bytes=budget;
            }
            fprintf(stderr,"h3cli: adaptive checkpoint source=%s ceiling=%llu persistent=%zu rows=%zu\n",
                explicit_budget?"override":"saved",
                (unsigned long long)s->params.adaptive_cache_max_bytes,plan.persistent_bytes,s->full_sequence);
        }
        if(id==1) {
            if(source && (s->next_step!=s->total_steps||s->text.tokens>65536||
                s->condition_video_elements>(UINT64_C(1)<<28)||
                s->condition_audio_elements>(UINT64_C(1)<<28)-s->condition_video_elements)) {
                fail(error,size,"invalid or oversized source counts before allocation");goto invalid;
            }
            h3_av_state_info shape;
            if(!h3_av_state_shape_profile(s->render_width,s->render_height,s->aligned_frames,s->params.geometry_profile,&shape) ||
                shape.video_elements!=s->video_elements || shape.audio_elements!=s->audio_elements ||
                shape.video_t!=s->latent_t || shape.latent_h!=s->latent_h || shape.latent_w!=s->latent_w || shape.audio_t!=s->audio_t ||
                s->total_steps<1 || s->total_steps>H3_MAX_STEPS || s->next_step<0 || s->next_step>s->total_steps ||
                s->text.width!=H3_TEXT_HIDDEN_SIZE || s->reference_count!=s->params.reference_count) {
                fail(error,size,"invalid identity geometry/schedule"); goto invalid;
            }
        }
    }
    if((source && s->upscale.stage!=1)||(!source && s->upscale.stage==1)) { fail(error,size,"container/stage mismatch");goto invalid; }
    if(!h3_sampler_state_validate(s,error,size)) goto invalid;
    h3_sampler_hash(data,length,s->loaded_hash);
    h3_sampler_log(s,"loaded",length,h3_av_now()-start);
    goto done;
invalid: h3_sampler_state_free(s); s=NULL;
done: munmap(mapping,length); close(fd); return s;
}
static int save_state(const h3_sampler_state *state,const char *path,int source,char *error,size_t size) {
    double start=h3_av_now(); if(error&&size) *error=0;
    if(!path || !*path || !little()) return fail(error,size,"invalid path/host endianness");
    if(!h3_sampler_state_validate(state,error,size)) return 0;
    if(source&&(state->text.tokens>65536||state->condition_video_elements>(UINT64_C(1)<<28)||
        state->condition_audio_elements>(UINT64_C(1)<<28)-state->condition_video_elements))return fail(error,size,"source counts exceed bounds");
    if((source && state->upscale.stage!=1)||(!source && state->upscale.stage==1))return fail(error,size,"container/stage mismatch");
    /* Serialization reads fields through a shallow scalar copy; no state or
     * latent mutation, including during optional preview, is permitted. */
    h3_sampler_state s=*state; if(state->layout.bridge) { s.bridge=*state->layout.bridge; s.layout.bridge=&s.bridge; }
    uint64_t preflight=HEADER,writer_bytes=0;
    for(unsigned id=1;id<=SECTIONS;id++)if(present(id,state)) {
        buffer measure={.ok=1,.measure=1};
        section(&measure,id,section_version(id,state),&s);
        if(!measure.ok||preflight>MAX_FILE-ENTRY||measure.length>MAX_FILE-preflight-ENTRY)
            return fail(error,size,"checkpoint exceeds 16 GiB bound before writer allocation");
        preflight+=ENTRY+measure.length;writer_bytes+=measure.length;
    }
    if(source&&preflight>(UINT64_C(2)<<30))return fail(error,size,"clean source exceeds 2 GiB bound before writer allocation");
    if(state->params.adaptive_cache)fprintf(stderr,"h3cli: checkpoint preflight file=%llu writer-copy=%llu adaptive-export=%llu bytes\n",
        (unsigned long long)preflight,(unsigned long long)writer_bytes,(unsigned long long)state->adaptive_elements*4);
    buffer payload[SECTIONS+1]={0}; unsigned n=0; int ok=1;
    for(unsigned id=1;id<=SECTIONS;id++) if(present(id,state)) { payload[id].ok=1; section(&payload[id],id,section_version(id,state),&s); if(!payload[id].ok) ok=0; n++; }
    if(!ok) {
        for(unsigned id=1;id<=SECTIONS;id++)free(payload[id].data);
        return fail(error,size,"cannot allocate checkpoint writer copies");
    }
    uint8_t header[HEADER]={0},table[SECTIONS*ENTRY]={0};
    memcpy(header,source?up_magic:magic,8); put(header+8,2,4); put(header+12,HEADER,4); put(header+20,n,4); put(header+32,0x01020304,4);
    uint64_t end=HEADER+(uint64_t)n*ENTRY; unsigned cursor=0;
    for(unsigned id=1;id<=SECTIONS;id++) if(present(id,state)) {
        buffer *b=&payload[id]; uint8_t *e=table+cursor++*ENTRY;
        if(b->length>MAX_FILE-end) { ok=0; break; }
        put(e,id,4); put(e+4,section_version(id,state),4); put(e+8,needed(id,state)?REQUIRED:0,4); put(e+12,dtype(id),4);
        unsigned width=dtype(id)==2?4:dtype(id)==3?2:1;
        put(e+16,b->length/width,8); put(e+24,end,8); put(e+32,b->length,8);
        h3_sampler_hash(b->data,b->length,e+40); end+=b->length;
    }
    if(source&&end>(UINT64_C(2)<<30))ok=0;
    put(header+24,end,8);
    h3_sha256_ctx h; h3_sha256_init(&h); update(&h,header,HEADER); update(&h,table,(size_t)n*ENTRY);
    for(unsigned id=1;id<=SECTIONS;id++) if(present(id,state)) update(&h,payload[id].data,payload[id].length);
    h3_sha256_final(header+64,&h);
    char *temporary=NULL;
    if(asprintf(&temporary,"%s.tmp.XXXXXX",path)<0) ok=0;
    int fd=ok?mkstemp(temporary):-1; FILE *f=fd>=0?fdopen(fd,"w+b"):NULL;
    if(!f) { if(fd>=0) close(fd); ok=0; }
    if(ok) {
        uint8_t incomplete[HEADER]={0};
        ok=fwrite(incomplete,1,HEADER,f)==HEADER && fwrite(table,ENTRY,n,f)==n;
        for(unsigned id=1;id<=SECTIONS && ok;id++) if(present(id,state))
            ok=fwrite(payload[id].data,1,payload[id].length,f)==payload[id].length;
        if(ok) ok=fflush(f)==0 && fsync(fd)==0 && fseek(f,0,SEEK_SET)==0 && fwrite(header,1,HEADER,f)==HEADER && fflush(f)==0 && fsync(fd)==0;
    }
    if(f && fclose(f)) ok=0;
    if(ok) {
        /* Validate the on-disk bytes before making the destination visible. */
        void *map=NULL; size_t length=0; int checkfd=-1;
        ok=mapped(temporary,&map,&length,&checkfd,error,size);
        if(ok) { ok=envelope(map,length,source,error,size); munmap(map,length); close(checkfd); }
    }
    if(ok && rename(temporary,path)) ok=0;
    if(ok) {
        char *parent=strdup(path); if(parent) { char *slash=strrchr(parent,'/'); if(slash) { if(slash==parent) slash[1]=0; else *slash=0; } else { free(parent); parent=strdup("."); }
            if(parent) { int dir=open(parent,O_RDONLY); if(dir>=0) { fsync(dir); close(dir); } free(parent); }
        }
        h3_sampler_log(state,"saved",end,h3_av_now()-start);
        fprintf(stderr,"h3cli: checkpoint path: %s\n",path);
    } else {
        if(temporary) unlink(temporary);
        if(!error || !size || !*error) fail(error,size,"atomic checkpoint write failed");
    }
    free(temporary); for(unsigned id=1;id<=SECTIONS;id++) free(payload[id].data);
    return ok;
}

int h3_sampler_prepared_key(const h3_sampler_state *state,uint8_t key[32]) {
    h3_sampler_state s=*state; s.next_step=0;
    if(s.continuation&&s.params.continuation_mode==H3_CONTINUE_HARD&&
       (s.params.adaptive_cache||s.params.cuda_attention==H3_ATTENTION_SUBBLOCK)) {
        h3_params defaults=H3_PARAMS_DEFAULT;
        s.params.bridge_video_steps=defaults.bridge_video_steps;
        s.params.bridge_max_strength=defaults.bridge_max_strength;
        s.params.bridge_profile=defaults.bridge_profile;
    }
    /* Runtime scheduling does not change immutable conditioning/AdaLN. */
    s.ane_decided=0;memset(s.ane_rows,0,sizeof(s.ane_rows));
    const unsigned ids[]={1,2,3,5,6,7,8,9,10,11,15,32};
    h3_sha256_ctx h; h3_sha256_init(&h);
    for(size_t i=0;i<sizeof(ids)/sizeof(*ids);i++) {
        buffer b={0}; b.ok=1; section(&b,ids[i],1,&s);
        if(!b.ok) { free(b.data); return 0; }
        uint8_t length[8]; put(length,b.length,8); update(&h,length,8); update(&h,b.data,b.length); free(b.data);
    }
    if(s.params.cuda_attention) {
        buffer b={0};b.ok=1;section(&b,35,1,&s);if(!b.ok){free(b.data);return 0;}
        uint8_t length[8];put(length,b.length,8);update(&h,length,8);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params._arithmetic_recipe) {
        buffer b={0};b.ok=1;section(&b,39,1,&s);if(!b.ok){free(b.data);return 0;}
        uint8_t length[8];put(length,b.length,8);update(&h,length,8);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params.cuda_attention==H3_ATTENTION_SUBBLOCK) {
        buffer b={0};b.ok=1;section(&b,43,1,&s);if(!b.ok){free(b.data);return 0;}
        uint8_t length[8];put(length,b.length,8);update(&h,length,8);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params.cuda_attention==H3_ATTENTION_SOL) {
        buffer b={0};b.ok=1;section(&b,38,1,&s);if(!b.ok){free(b.data);return 0;}
        uint8_t length[8];put(length,b.length,8);update(&h,length,8);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params.backend) {
        buffer b={0};b.ok=1;section(&b,36,1,&s);if(!b.ok){free(b.data);return 0;}
        uint8_t length[8];put(length,b.length,8);update(&h,length,8);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params.cuda_denoise_quant) {
        buffer b={0};b.ok=1;section(&b,34,1,&s);if(!b.ok){free(b.data);return 0;}
        uint8_t length[8];put(length,b.length,8);update(&h,length,8);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params.metal_attention.weight_format) {
        buffer b={0};b.ok=1;section(&b,37,1,&s);if(!b.ok){free(b.data);return 0;}
        uint8_t length[8];put(length,b.length,8);update(&h,length,8);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params.adaptive_cache) {
        uint8_t controls[16];float threshold=h3_adaptive_threshold(&s.params);uint32_t bits;
        memcpy(&bits,&threshold,4);put(controls,(uint32_t)s.params.adaptive_cache,4);
        put(controls+4,h3_adaptive_execution_recipe(s.params.adaptive_cache,s.params.cuda_denoise_quant,s.reference_count!=0,s.continuation),4);
        put(controls+8,bits,4);put(controls+12,(uint32_t)h3_adaptive_max_hits(&s.params),4);
        update(&h,"adaptive-controls",17);update(&h,controls,sizeof(controls));
    }
    if(s.params.adaptive_cache||s.params.cuda_attention==H3_ATTENTION_SUBBLOCK) {
        uint8_t warmup[8];put(warmup,s.params.adaptive_cache?(uint32_t)h3_adaptive_warmup(s.params.adaptive_cache_warmup):0,4);put(warmup+4,s.params.cuda_attention==H3_ATTENTION_SUBBLOCK?(uint32_t)h3_subblock_warmup(s.params.subblock_warmup):0,4);
        update(&h,"warmup",6);update(&h,warmup,sizeof(warmup));
    }
    if(s.upscale.stage) {
        buffer b={0};b.ok=1;section(&b,45,section_version(45,&s),&s);if(!b.ok){free(b.data);return 0;}
        update(&h,"upscale-stage",13);update(&h,b.data,b.length);free(b.data);
    }
    if(s.params.geometry_profile) {
        buffer b={0};b.ok=1;section(&b,46,1,&s);if(!b.ok){free(b.data);return 0;}
        update(&h,"geometry-profile",16);update(&h,b.data,b.length);free(b.data);
    }
    h3_sha256_final(key,&h); return 1;
}

h3_sampler_state *h3_sampler_state_load_with_budget(const char *path,uint64_t budget,int explicit_budget,char *e,size_t n) {
    return load_state(path,0,budget,explicit_budget,e,n);
}
h3_sampler_state *h3_sampler_state_load(const char *path,char *e,size_t n) {return load_state(path,0,0,0,e,n);}
h3_sampler_state *h3_upscale_file_load(const char *path,char *e,size_t n) {return load_state(path,1,0,0,e,n);}
int h3_sampler_state_save(const h3_sampler_state *s,const char *path,char *e,size_t n) {return save_state(s,path,0,e,n);}
int h3_upscale_file_save(const h3_sampler_state *s,const char *path,char *e,size_t n) {return save_state(s,path,1,e,n);}
