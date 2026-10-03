#include "src/sglang/sglang.h"
#include "src/denoise/attention.h"
#include "src/denoise/approximate.h"
#include "src/denoise/subblock.h"
#include "src/sampling/sampler_state.h"
#include "src/execution.h"
#include "src/weights/quant.h"
#include "src/sampling/av_state.h"
#include "src/denoise/dit.h"
#include "src/platform.h"
#include "src/runtime/runtime.h"
#include "src/digest.h"
#include <dirent.h>
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

#ifndef H3_BUILD_ID
#define H3_BUILD_ID H3_VERSION
#endif
#ifndef H3_GIT_COMMIT
#define H3_GIT_COMMIT "unknown"
#endif
#ifndef H3_PORTABLE_BUILD_ID
#define H3_PORTABLE_BUILD_ID "unavailable"
#endif
#define BUILD_IDENTITY H3_VERSION ";source=" H3_BUILD_ID ";git=" H3_GIT_COMMIT H3_COMPILER_ID ";portable=" H3_PORTABLE_BUILD_ID
#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static int fail(char *error, size_t size, const char *message) {
    if (error && size) snprintf(error,size,"sampler state: %s",message);
    return 0;
}
static void update(h3_sha256_ctx *h, const void *data, size_t bytes) {
    const uint8_t *p=data;
    while (bytes) { h3_sha256_size n=(h3_sha256_size)(bytes>1048576?1048576:bytes); h3_sha256_update(h,p,n); p+=n; bytes-=n; }
}
void h3_sampler_hash(const void *data, size_t bytes, uint8_t digest[32]) {
    h3_sha256_ctx h; h3_sha256_init(&h); update(&h,data,bytes); h3_sha256_final(digest,&h);
}
static void *copy(const void *p, size_t n) {
    if (!n) return NULL;
    void *q=p?malloc(n):NULL; if (q) memcpy(q,p,n); return q;
}
void h3_sampler_state_free(h3_sampler_state *s) {
    if (!s) return;
    h3_prepared_cache_free(&s->prepared);
    free(s->gpu_last_video); free(s->gpu_previous_video);
    free(s->gpu_last_audio); free(s->gpu_previous_audio); free(s->core_residual);
    free(s->adaptive_anchor);free(s->adaptive_delta);
    free(s->video); free(s->audio);
    free(s->last_video_velocity); free(s->previous_video_velocity);
    free(s->last_audio_velocity); free(s->previous_audio_velocity);
    free(s->original_video_noise); free(s->original_audio_noise);
    h3_text_embedding_free(&s->text); h3_layout_free(&s->layout);
    free(s->condition_video); free(s->condition_audio); free(s->references);
    free(s->prompt); free(s->provenance); free(s->token_ids);
    free(s->presentation_positions); free(s->presentation_spans);
    free(s->build_id); free(s->environment); free(s);
}
h3_sampler_state *h3_sampler_state_create(const h3_sigma_schedule *sigmas,
    size_t nv, size_t na, int reuse) {
    if (!sigmas || sigmas->steps<1 || sigmas->steps>H3_MAX_STEPS || !nv || !na ||
        nv>PTRDIFF_MAX/sizeof(float) || na>PTRDIFF_MAX/sizeof(float) || reuse<1 || reuse>32) return NULL;
    h3_sampler_state *s=calloc(1,sizeof(*s)); if (!s) return NULL;
    s->version=H3_SAMPLE_VERSION; s->execution_version=1; s->sigmas=*sigmas; s->total_steps=sigmas->steps;
    s->reuse_interval=reuse; s->last_evaluated=s->previous_evaluated=-1;
    s->video_elements=nv; s->audio_elements=na;
    s->video=malloc(nv*4); s->audio=malloc(na*4);
    if (reuse>1) {
        s->last_video_velocity=calloc(nv,4); s->previous_video_velocity=calloc(nv,4);
        s->last_audio_velocity=calloc(na,4); s->previous_audio_velocity=calloc(na,4);
    }
    if (!s->video || !s->audio || (reuse>1 && (!s->last_video_velocity ||
        !s->previous_video_velocity || !s->last_audio_velocity || !s->previous_audio_velocity)) ||
        h3_dit_reuse_schedule(sigmas->steps,reuse,s->selected,sizeof(s->selected))<0) {
        h3_sampler_state_free(s); return NULL;
    }
    return s;
}

/* A cache hit is tied to file identity, size and both nanosecond timestamps.
 * Changing content while restoring mtime still changes ctime. A file modified
 * during hashing is rejected. The digest is always content-derived. */
typedef struct { uint64_t dev, ino, bytes; int64_t mt, mn, ct, cn; } stamp;
static stamp file_stamp(const struct stat *st) {
    stamp x={0}; x.dev=st->st_dev; x.ino=st->st_ino; x.bytes=(uint64_t)st->st_size;
    x.mt=h3_stat_mtime(st).tv_sec; x.mn=h3_stat_mtime(st).tv_nsec;
    x.ct=h3_stat_ctime(st).tv_sec; x.cn=h3_stat_ctime(st).tv_nsec; return x;
}
static void hex(const uint8_t hash[32], char out[65]) {
    for (int i=0;i<32;i++) snprintf(out+2*i,3,"%02x",hash[i]);
}
static int file_hash(const char *path, uint8_t digest[32], uint64_t *bytes, int cached) {
    FILE *f=fopen(path,"rb"); struct stat st;
    if (!f || fstat(fileno(f),&st) || !S_ISREG(st.st_mode) || st.st_size<0) { if(f) fclose(f); return 0; }
    stamp before=file_stamp(&st); *bytes=before.bytes;
    char cache[256]={0};
    if (cached == 1) {
        uint8_t key[32]; char name[65];
        h3_sampler_hash(&before,sizeof(before),key); hex(key,name);
        mkdir("outputs",0755); mkdir("outputs/.h3-model-hashes",0755);
        snprintf(cache,sizeof(cache),"outputs/.h3-model-hashes/%s",name);
        FILE *c=fopen(cache,"rb");
        if(c) {
            struct { stamp identity; uint8_t hash[32], check[32]; } entry={0};
            int ok=fread(&entry,1,sizeof(entry),c)==sizeof(entry) && fgetc(c)==EOF;
            fclose(c); uint8_t check[32]; h3_sampler_hash(&entry,sizeof(entry)-32,check);
            if(ok && !memcmp(&before,&entry.identity,sizeof(before)) && !memcmp(check,entry.check,32)) {
                memcpy(digest,entry.hash,32); fclose(f); return 1;
            }
        }
    }
    uint8_t *buffer=malloc(4*1048576); if(!buffer) { fclose(f); return 0; }
    h3_sha256_ctx h;
    /* Effective runtime models use fresh content reads: coarse filesystem
     * timestamps can miss rapid same-size changes. The optional CPU SHA
     * accelerator is independent of denoising policy and preserves digests. */
    if(cached>1)h3_sha256_init_fast(&h);else h3_sha256_init(&h);
    size_t n; uint64_t consumed=0;
    do { n=fread(buffer,1,4*1048576,f); if(n) { update(&h,buffer,n); consumed+=n; } } while(n==4*1048576);
    int ok=!ferror(f) && consumed==before.bytes && !fstat(fileno(f),&st);
    stamp after=file_stamp(&st); ok=ok && !memcmp(&before,&after,sizeof(before));
    free(buffer); fclose(f); if(!ok) return 0;
    h3_sha256_final(digest,&h);
    if(cache[0]) {
        struct { stamp identity; uint8_t hash[32], check[32]; } entry={0};
        entry.identity=before; memcpy(entry.hash,digest,32); h3_sampler_hash(&entry,sizeof(entry)-32,entry.check);
        char tmp[280]; snprintf(tmp,sizeof(tmp),"%s.XXXXXX",cache); int fd=mkstemp(tmp);
        if(fd>=0) { FILE *c=fdopen(fd,"wb"); if(c) { int saved=fwrite(&entry,1,sizeof(entry),c)==sizeof(entry); if(fclose(c)) saved=0; if(saved) rename(tmp,cache); } else close(fd); unlink(tmp); }
    }
    return 1;
}
static int fingerprint_tree(const char *root, const char *relative, const char *transformer, h3_sha256_ctx *h, size_t *files,int metadata) {
    char *path=NULL;
    if (transformer && (!strcmp(relative,"transformer") || !strncmp(relative,"transformer/",12))) {
        if(asprintf(&path,"%s%s",transformer,relative+11)<0) return 0;
    } else if(asprintf(&path,"%s/%s",root,relative)<0) return 0;
    struct stat st; int ok=!stat(path,&st);
    if(ok && S_ISDIR(st.st_mode)) {
        struct dirent **entries=NULL; int n=scandir(path,&entries,NULL,alphasort);
        if(n<0) ok=0;
        for(int i=0;i<n;i++) {
            const char *name=entries[i]->d_name;
            if(ok && name[0]!='.') {
                char *child=NULL;
                if(asprintf(&child,"%s%s%s",relative,*relative?"/":"",name)<0) ok=0;
                else { ok=fingerprint_tree(root,child,transformer,h,files,metadata); free(child); }
            }
            free(entries[i]);
        }
        free(entries);
    } else if(ok && S_ISREG(st.st_mode)) {
        uint8_t digest[32]; uint64_t bytes;
        if(metadata) {
            /* Presentation identity only: no model/config payload reads. A
             * separate domain distinguishes this from a content fingerprint. */
            char *canonical=realpath(path,NULL);stamp identity=file_stamp(&st);
            bytes=(uint64_t)st.st_size;
            h3_sha256_ctx entry;h3_sha256_init_fast(&entry);
            update(&entry,&identity,sizeof(identity));
            if(canonical){update(&entry,canonical,strlen(canonical)+1);free(canonical);}
            else ok=0;
            h3_sha256_final(digest,&entry);
        }else ok=file_hash(path,digest,&bytes,transformer?2:1);
        if(ok) { update(h,relative,strlen(relative)+1); uint8_t b[8]; for(int i=0;i<8;i++) b[i]=(uint8_t)(bytes>>(8*i)); update(h,b,8); update(h,digest,32); (*files)++; }
    } else ok=0;
    free(path); return ok;
}
static int model_fingerprint(const char *directory, const char *transformer, int ref2va,
    uint8_t digest[32], char *error, size_t size,int metadata) {
    if(!directory || (ref2va!=0 && ref2va!=1)) return fail(error,size,"invalid model mode");
    char *root=NULL; if(asprintf(&root,"%s/%s",directory,ref2va?"Ref2VA":"FL2VA")<0) return fail(error,size,"out of memory");
    h3_sha256_ctx h; h3_sha256_init(&h);
    const char *mode=metadata?(ref2va?"h3-model-metadata-v1/Ref2VA":"h3-model-metadata-v1/FL2VA"):
        (ref2va?"h3-model-v1/Ref2VA":"h3-model-v1/FL2VA");
    update(&h,mode,strlen(mode)); size_t files=0;
    int ok=fingerprint_tree(root,"",transformer,&h,&files,metadata); free(root);
    if(!ok || !files) return fail(error,size,"cannot fingerprint complete model tree (missing or changing file)");
    h3_sha256_final(digest,&h); return 1;
}
int h3_sampler_model_fingerprint_effective(const char *directory,const char *transformer,int ref2va,
    uint8_t digest[32],char *error,size_t size) {
    return model_fingerprint(directory,transformer,ref2va,digest,error,size,0);
}
int h3_sampler_model_metadata_effective(const char *directory,const char *transformer,int ref2va,
    uint8_t digest[32],char *error,size_t size) {
    return model_fingerprint(directory,transformer,ref2va,digest,error,size,1);
}
extern char **environ;
int h3_sampler_component_metadata(const char *directory,const char *transformer,
    int ref2va,const char *component,uint8_t digest[32],char *error,size_t size) {
    char *root=NULL;
    if(!directory||!component||asprintf(&root,"%s/%s",directory,ref2va?"Ref2VA":"FL2VA")<0)
        return fail(error,size,"invalid conditioning component");
    h3_sha256_ctx h;h3_sha256_init(&h);size_t files=0;
    const char *domain="h3-reference-component-metadata-v1";
    update(&h,domain,strlen(domain)+1);
    int ok=fingerprint_tree(root,component,transformer,&h,&files,1);free(root);
    if(!ok||!files)return fail(error,size,"cannot fingerprint conditioning component");
    h3_sha256_final(digest,&h);return 1;
}
int h3_sampler_source_fingerprint(const char *path,uint8_t digest[32],uint64_t *bytes) {
    return path&&digest&&bytes&&file_hash(path,digest,bytes,0);
}
static int compare_strings(const void *a,const void *b) { return strcmp(*(char *const *)a,*(char *const *)b); }
static int diagnostic_env(const char *s) {
    /* Model acquisition policy never changes sampler arithmetic. In particular,
     * a CLI checkpoint must remain usable in an always-offline server worker. */
    if(!strncmp(s,"H3_OFFLINE=",strlen("H3_OFFLINE=")))return 1;
    /* Cache validation changes loading cost, not denoising arithmetic. */
    if(!strncmp(s,"H3_QUANT_VERIFY=",strlen("H3_QUANT_VERIFY=")))return 1;
    const char *names[]={"H3_EXPERIMENT_TRACE=","H3_EXPERIMENT_TIMING=","H3_FULL_VAE_CUDA_GRAPH=","H3_PROFILE=","H3_PROFILE_COMPONENTS=","H3_PROFILE_REGIONS=","H3_METAL_SIGNPOSTS=","H3_DEBUG_","H3_BRIDGE_DIAGNOSTICS=","H3_PREVIEW_MODE=","H3_PREVIEW_DIAGNOSTICS=","H3_FFMPEG=","H3_FFPROBE=","H3_REGRESSION_","H3_TEST_", "H3_MODEL_DIR=", "H3_CUDA_DEVICE=", "H3_CUDA_WEIGHT_MODE=", "H3_CUDA_REGISTER_WEIGHTS=", "H3_CUDA_TEST_", "H3_CUDA_OUTPUT=", "H3_CUDA_CASES="};
    for(size_t i=0;i<sizeof(names)/sizeof(*names);i++) if(!strncmp(s,names[i],strlen(names[i]))) return 1;
    return 0;
}
static char *environment(void) {
    size_t n=0,bytes=1; for(char **e=environ;*e;e++) if(!strncmp(*e,"H3_",3) && !diagnostic_env(*e)) { n++; bytes+=strlen(h3_runtime_environment_entry(*e))+32; }
    char **list=calloc(n?n:1,sizeof(*list)); char *s=malloc(bytes); if(!list||!s) { free(list); free(s); return NULL; }
    size_t i=0; for(char **e=environ;*e;e++) if(!strncmp(*e,"H3_",3) && !diagnostic_env(*e)) list[i++]=(char *)h3_runtime_environment_entry(*e);
    qsort(list,n,sizeof(*list),compare_strings); char *p=s;
    for(i=0;i<n;i++) { size_t len=strlen(list[i]); p+=sprintf(p,"%zu:",len); memcpy(p,list[i],len); p+=len; *p++='\n'; }
    *p=0; free(list); return s;
}
int h3_sampler_checkpoint_options(const h3_params *p, const h3_device_info *device, char *error, size_t size) {
    if(!p) return fail(error,size,"missing parameters");
    if(p->_arithmetic_recipe && (p->_arithmetic_recipe!=H3_SGLANG_VERSION || p->still ||
       (device && strcmp(device->backend,"cuda"))))
        return fail(error,size,"incompatible shared SGLang CUDA policy");
    if(!h3_cuda_sol_params_valid(p,error,size))return 0;
    if(!h3_backend_preflight(H3_BACKEND_SCOPE(p),0,error,size))return 0;
    if(p->metal_attention.weight_format==H3_WEIGHT_Q8&&p->ssd_streaming)
        return fail(error,size,"Q8 uses resident packed weights; --ssd-streaming is not supported");
    if(p->backend&&device&&strcmp(device->backend,"metal"))return fail(error,size,"native Metal checkpoint requires Metal");
    if(p->backend&&(p->cuda_attention||p->cuda_denoise_quant||p->token_reduction||p->use_int8_row_fc2||
       p->dit_layers!=50||p->core_reuse!=1||p->denoise_reuse!=1))
        return fail(error,size,"Metal requires BF16 state, 50 layers, reuse 1 and token reduction off");
    const char *reduction=getenv("H3_TOKEN_REDUCTION");
    if(p->backend&&reduction&&*reduction&&strcmp(reduction,"0"))
        return fail(error,size,"Metal requires H3_TOKEN_REDUCTION off");
    if(!h3_attention_options(p->cuda_attention,error,size))return 0;
    if(p->cuda_attention&&device&&strcmp(device->backend,"cuda"))return fail(error,size,"CUDA attention checkpoint requires CUDA");
    if(!h3_quant_options(p->cuda_denoise_quant,NULL,error,size))return 0;
    if(p->cuda_denoise_quant&&device&&strcmp(device->backend,"cuda"))return fail(error,size,"quantized checkpoint requires CUDA");
    (void)device;
    if(p->core_reuse<1 || p->core_reuse>6 || (p->core_reuse>1 && p->denoise_reuse>1))
        return fail(error,size,"core reuse requires interval 1..6 and whole-denoiser reuse 1");
    uint8_t selected[H3_MAX_STEPS];
    if(h3_dit_resolve_reuse(p->steps,p->denoise_reuse,selected)<0)
        return fail(error,size,"invalid custom H3_REUSE_STEPS; include first and last step in increasing order");
    return 1;
}
char *h3_sampler_environment(void){return environment();}

/* Provenance is little-endian, independent of the host ABI. */
static int append(h3_sampler_state *s,const void *p,size_t n) {
    if(n>SIZE_MAX-s->provenance_bytes) return 0;
    void *v=realloc(s->provenance,s->provenance_bytes+n); if(!v && n) return 0;
    s->provenance=v; if(n) memcpy(s->provenance+s->provenance_bytes,p,n); s->provenance_bytes+=n; return 1;
}
static int number(h3_sampler_state *s,uint64_t n) { uint8_t b[8]; for(int i=0;i<8;i++) b[i]=(uint8_t)(n>>(i*8)); return append(s,b,8); }
static int source(h3_sampler_state *s,const char *path) {
    uint8_t digest[32]={0}; uint64_t bytes=0; size_t len=path?strlen(path):0;
    if(path && !file_hash(path,digest,&bytes,0)) return 0;
    return number(s,len) && append(s,path,len) && number(s,bytes) && append(s,digest,32);
}
static int capture_effective(h3_sampler_state *s, const h3_sampler_state *parent, const char *model_dir, const char *transformer,
    const h3_device_info *device, const h3_params *p, const char *prompt,
    const h3_text_embedding *text, const h3_layout *layout,
    const h3_layout_ref *refs, size_t nrefs, const float *cv, size_t nv,
    const float *ca, size_t na, int conditioned, const uint8_t av_signature[32],
    char *error, size_t size) {
    if(!s || !p || !device || !prompt || !text || !layout || !av_signature ||
        !h3_sampler_checkpoint_options(p,device,error,size)) return 0;
    if(h3_dit_resolve_reuse(p->steps,p->denoise_reuse,s->selected)<0) return fail(error,size,"invalid reuse bitmap");
    s->params=*p;
    s->metal_attention_version=p->backend?H3_METAL_ATTENTION_VERSION:0;
    s->params.backend_set=0;
    s->params.save_conditioning=s->params.load_conditioning=NULL;
    s->params.conditioning_schedule=0;
    s->params.save_upscale_state=NULL;s->params.state_only=0;
    s->quant_version=h3_quant_execution_recipe(p->cuda_denoise_quant,p->adaptive_cache,p->cuda_attention);
    s->attention_version=h3_attention_execution_recipe(p->cuda_attention,p->cuda_denoise_quant);
    s->attention_plan=h3_attention_plan(p->cuda_attention);
    s->params.cuda_attention_set=0;s->params.cuda_sol_set=0;
    s->params.adaptive_cache_max_bytes=p->adaptive_cache?h3_adaptive_budget(p->adaptive_cache_max_bytes):0;
    s->params.adaptive_cache_max_bytes_set=0;
    s->params.adaptive_cache_threshold=h3_adaptive_threshold(p);
    s->params.adaptive_cache_max_hits=h3_adaptive_max_hits(p);
    s->params.adaptive_cache_threshold_set=s->params.adaptive_cache_max_hits_set=p->adaptive_cache!=0;
    s->params.adaptive_cache_set=0;s->params.subblock_sparsity_set=0;
    s->params.adaptive_cache_warmup_set=0;s->params.subblock_warmup_set=0;
    s->params.cuda_denoise_quant_cache=NULL;s->params.cuda_denoise_quant_set=0;
    s->params.output_path=s->params.first_frame=s->params.last_frame=NULL;
    s->params.references=NULL; s->params.continuation=NULL;
    s->params.preview_vae=0; s->params.preview_vae_model=NULL;
    s->params.on_frame=NULL; s->params.on_progress=NULL; s->params.on_latent_step=NULL; s->params.callback_opaque=NULL;
    s->params.save_sampler_state=s->params.resume_sampler_state=NULL; s->params.stop_after_step=-1;
    s->params.preview_on_stop=s->params.preview_denoise=0;
    s->ref2va=p->reference_count!=0; s->conditioned=conditioned;
    s->render_width=p->render_width?p->render_width:p->width;
    s->render_height=p->render_height?p->render_height:p->height;
    h3_temporal_shape shape=h3_temporal(p->frames); s->aligned_frames=shape.frame_count;
    s->latent_t=shape.video_t; s->audio_t=shape.audio_t;
    h3_latent_canvas(s->render_width,s->render_height,&s->latent_w,&s->latent_h);
    s->spatial_rope_scale=!p->use_reference_rope && s->render_width==256 && s->render_height==256?0.5f:1.0f;
    s->prompt=strdup(prompt); s->device=*device; s->backend_version=H3_SAMPLE_BACKEND_VERSION;
    s->build_id=strdup(BUILD_IDENTITY); s->environment=environment();
    memcpy(s->av_signature,av_signature,32);
    s->text.tokens=text->tokens; s->text.width=text->width;
    s->text.values=copy(text->values,text->tokens*text->width*2);
    s->text.tags=text->tags?copy(text->tags,text->tokens):NULL;
    s->condition_video_elements=nv; s->condition_audio_elements=na;
    s->condition_video=copy(cv,nv*4); s->condition_audio=copy(ca,na*4);
    s->reference_count=nrefs; s->references=copy(refs,nrefs*sizeof(*refs));
    s->refvideo_pipeline=H3_REFVIDEO_NONE;
    for(size_t i=0;i<nrefs;i++) if(refs[i].kind==H3_LAYOUT_REF_VIDEO)
        s->refvideo_pipeline=H3_REFVIDEO_RELEASED_V1;
    s->layout=*layout;
    s->layout.segments=copy(layout->segments,layout->segment_count*sizeof(*layout->segments));
    s->layout.positions=copy(layout->positions,layout->seq_len*sizeof(*layout->positions));
    if(layout->bridge) { s->bridge=*layout->bridge; s->layout.bridge=&s->bridge; }
    if(!s->prompt || !s->build_id || !s->environment || !s->text.values ||
        (text->tags && !s->text.tags) || (nv && !s->condition_video) || (na && !s->condition_audio) ||
        (nrefs && !s->references) || !s->layout.segments || !s->layout.positions) return fail(error,size,"out of memory capturing conditioning");
    if(!h3_sampler_model_metadata_effective(model_dir,transformer,s->ref2va,s->model_fingerprint,error,size)) return 0;
    if(parent) {
        s->provenance_bytes=parent->provenance_bytes;s->provenance=copy(parent->provenance,parent->provenance_bytes);
        s->token_count=parent->token_count;s->presentation_span_count=parent->presentation_span_count;
        s->token_ids=copy(parent->token_ids,parent->token_count*4);
        s->presentation_positions=copy(parent->presentation_positions,parent->presentation_positions?parent->token_count*12:0);
        s->presentation_spans=copy(parent->presentation_spans,parent->presentation_span_count*16);
        if(!s->provenance||(parent->token_ids&&!s->token_ids)||(parent->presentation_positions&&!s->presentation_positions)||
           (parent->presentation_spans&&!s->presentation_spans))return fail(error,size,"cannot copy source semantic provenance");
    } else {
    size_t count=nrefs+(p->first_frame!=NULL)+(p->last_frame!=NULL);
    if(!number(s,count)) return fail(error,size,"out of memory capturing provenance");
    for(size_t i=0;i<count;i++) {
        const h3_reference *r=i<nrefs?&p->references[i]:NULL;
        const char *path=r?r->path:((i==nrefs && p->first_frame)?p->first_frame:p->last_frame);
        if(!number(s,r?(uint64_t)r->kind:(i==nrefs && p->first_frame?5:6)) ||
            !number(s,r?(uint64_t)r->include_embedded_audio:0) || !source(s,path) || !source(s,r?r->audio_path:NULL))
            return fail(error,size,"cannot fingerprint source reference");
    }
    if (text->diagnostics) {
        const h3_text_diagnostics *d=text->diagnostics;
        s->token_count=d->tokens; s->presentation_span_count=d->span_count;
        s->token_ids=copy(d->ids,d->tokens*4);
        s->presentation_positions=copy(d->positions,d->tokens*3*4);
        s->presentation_spans=copy(d->spans,d->span_count*2*8);
        if (!s->token_ids || !s->presentation_positions || (d->span_count && !s->presentation_spans)) return fail(error,size,"cannot capture presentation diagnostics");
    }
    }
    s->rng_version=p->_arithmetic_recipe?2:1; s->continuation=p->continuation!=NULL;
    s->clean_coefficient=0.999f; s->noise_coefficient=0.001f; s->audio_preservation=1;
    if(p->continuation) {
        s->context_frames=p->continuation_context_frames?p->continuation_context_frames:39;
        const h3_av_state *av=p->continuation;
        /* Canonical source .h3av digest includes its immutable header/metadata. */
        h3_av_state_fingerprint(av,s->source_fingerprint);
        h3_sha256_ctx v,a; h3_sha256_init(&v); h3_sha256_init(&a);
        size_t plane=(size_t)s->latent_h*(size_t)s->latent_w;
        for(int c=0;c<24;c++) update(&v,av->video+((size_t)c*av->info.video_t+av->info.video_t-layout->prefix.video_prefix_t)*plane,(size_t)layout->prefix.video_prefix_t*plane*4);
        for(int c=0;c<64;c++) update(&a,av->audio+(size_t)c*av->info.audio_t+av->info.audio_t-layout->prefix.audio_prefix_t,(size_t)layout->prefix.audio_prefix_t*4);
        h3_sha256_final(s->video_tail_hash,&v); h3_sha256_final(s->audio_tail_hash,&a);
    }
    return 1;
}
int h3_sampler_state_capture_effective(h3_sampler_state *s,const char *model_dir,const char *transformer,
    const h3_device_info *device,const h3_params *p,const char *prompt,const h3_text_embedding *text,
    const h3_layout *layout,const h3_layout_ref *refs,size_t nrefs,const float *cv,size_t nv,
    const float *ca,size_t na,int conditioned,const uint8_t signature[32],char *e,size_t n) {
    return capture_effective(s,NULL,model_dir,transformer,device,p,prompt,text,layout,refs,nrefs,cv,nv,ca,na,conditioned,signature,e,n);
}
int h3_sampler_capture_upscale(h3_sampler_state *s,const h3_sampler_state *parent,const char *model_dir,
    const h3_device_info *device,const h3_params *p,const h3_layout *layout,const h3_layout_ref *refs,
    const float *cv,size_t nv,const float *ca,size_t na,const uint8_t signature[32],char *e,size_t n) {
    return parent&&capture_effective(s,parent,model_dir,NULL,device,p,parent->prompt,&parent->text,layout,refs,
        parent->reference_count,cv,nv,ca,na,nv||na,signature,e,n);
}
static int device_backend(const h3_device_info *device) {
    if (!device) return 0;
    if (!strcmp(device->backend,"cuda") ||
        (!strncmp(device->architecture,"SM",2) && device->architecture[2]>='0' && device->architecture[2]<='9')) return 2;
    if (!strcmp(device->backend,"metal") || device->apple_gpu_family>0) return 1;
    return 0;
}
int h3_sampler_state_compatible_effective(const h3_sampler_state *s,const char *directory,const char *transformer,
    const h3_device_info *device,char *error,size_t size) {
    if(!h3_sampler_state_validate(s,error,size) || !h3_sampler_checkpoint_options(&s->params,device,error,size)) return 0;
    int source_backend=device_backend(&s->device), target_backend=device_backend(device);
    int cross=source_backend && target_backend && source_backend!=target_backend;
    const char *portable=strstr(s->build_id,";portable=");
    int same_portable=portable && strlen(H3_PORTABLE_BUILD_ID)==64 &&
        !strcmp(portable+10,H3_PORTABLE_BUILD_ID) && !strncmp(s->build_id,H3_VERSION ";",strlen(H3_VERSION)+1);
    if((cross ? !same_portable : strcmp(s->build_id,BUILD_IDENTITY)!=0) || s->backend_version!=H3_SAMPLE_BACKEND_VERSION)
        return fail(error,size,"incompatible engine build/backend version");
    if(!device || (!cross && (s->device.apple_gpu_family!=device->apple_gpu_family || s->device.metal4!=device->metal4 ||
        s->device.unified_memory!=device->unified_memory))) return fail(error,size,"incompatible Metal device capabilities");
    if((s->params.adaptive_cache||s->params.cuda_attention==H3_ATTENTION_SUBBLOCK||(s->upscale.stage==2&&target_backend==2))&&
       (cross||strcmp(device->backend,"cuda")||strcmp(device->name,s->device.name)||
        memcmp(device->cuda_uuid,s->device.cuda_uuid,sizeof(device->cuda_uuid))||
        device->cuda_compute_major!=s->device.cuda_compute_major||device->cuda_compute_minor!=s->device.cuda_compute_minor||
        device->cuda_runtime_version!=s->device.cuda_runtime_version||device->cuda_driver_version!=s->device.cuda_driver_version))
        return fail(error,size,"adaptive/SubBlock resume requires the recorded CUDA device and runtime");
    if(s->upscale.stage==2&&cross)return fail(error,size,"upscale refinement resume requires the original backend");
    char *env=environment(); int same=env && !strcmp(env,s->environment); free(env);
    if(!same) return fail(error,size,"numerical H3_* environment differs from checkpoint");
    uint8_t digest[32];
    if(!h3_sampler_model_metadata_effective(directory,transformer,s->ref2va,digest,error,size)) return 0;
    if(memcmp(digest,s->model_fingerprint,32)) return fail(error,size,"model metadata identity mismatch");
    return 1;
}

int h3_sampler_layout_validate(const h3_sampler_state *s,char *error,size_t size) {
    if(!s)return fail(error,size,"missing sampler layout");
    const h3_params *p=&s->params;
    if(!s->layout.segments||!s->layout.positions||!s->layout.segment_count||s->layout.segment_count>1000)
        return fail(error,size,"missing or oversized layout metadata");
    h3_av_state_info shape;
    if(!h3_av_state_shape_profile(s->render_width,s->render_height,s->aligned_frames,p->geometry_profile,&shape)||
       shape.video_elements!=s->video_elements||shape.audio_elements!=s->audio_elements||
       shape.video_t!=s->latent_t||shape.audio_t!=s->audio_t||shape.latent_h!=s->latent_h||shape.latent_w!=s->latent_w)
        return fail(error,size,"invalid adaptive target geometry before allocation");
    if(s->ref2va!=0 && s->ref2va!=1) return fail(error,size,"invalid checkpoint mode");
    if((s->reference_count!=0)!=s->ref2va || s->reference_count>12 || s->reference_count!=p->reference_count ||
        (s->reference_count && !s->references)) return fail(error,size,"checkpoint mode/reference mismatch");
    h3_params reference_policy=*p;h3_reference reference_kinds[12]={0};
    for(size_t i=0;i<s->reference_count;i++) {
        reference_kinds[i].kind=s->references[i].kind==H3_LAYOUT_REF_IMAGE?H3_REFERENCE_IMAGE:
            s->references[i].kind==H3_LAYOUT_REF_VIDEO?H3_REFERENCE_VIDEO:H3_REFERENCE_AUDIO;
        reference_kinds[i].include_embedded_audio=s->references[i].kind==H3_LAYOUT_REF_VIDEO&&s->references[i].audio_t>0;
    }
    reference_policy.references=reference_kinds;
    if(!h3_approximate_layout_valid(p,s->continuation,s->context_frames,&s->layout,error,size))return 0;
    for(size_t i=0;i<s->layout.segment_count;i++)if(s->layout.segments[i].kind==H3_SEG_REF_IMAGE&&!s->reference_count)reference_policy.first_frame="saved anchor";
    if(!h3_approximate_references_valid(&reference_policy,error,size))return 0;
    if((p->adaptive_cache||p->cuda_attention==H3_ATTENTION_SUBBLOCK)&&
       (p->core_reuse!=1||p->denoise_reuse!=1||p->dit_layers!=50||p->token_reduction))
        return fail(error,size,"invalid adaptive/SubBlock saved execution policy");
    uint32_t pipeline=H3_REFVIDEO_NONE;
    for(size_t i=0;i<s->reference_count;i++) if(s->references[i].kind==H3_LAYOUT_REF_VIDEO)
        pipeline=H3_REFVIDEO_RELEASED_V1;
    if(s->refvideo_pipeline!=pipeline)
        return fail(error,size,"incompatible Ref2VA video-pipeline provenance/version");
    if(!s->text.tokens || s->text.tokens>1000000 || s->text.width!=H3_TEXT_HIDDEN_SIZE || !s->text.values ||
        !s->prompt || !*s->prompt || !s->build_id || !s->environment || !s->provenance || s->provenance_bytes<8)
        return fail(error,size,"missing conditioning/identity metadata");
    if(s->text.tags) for(size_t i=0;i<s->text.tokens;i++) if(s->text.tags[i]>1) return fail(error,size,"invalid modality tag");
    if(s->conditioned!=!!(s->condition_video_elements || s->condition_audio_elements) ||
        (s->condition_video_elements && !s->condition_video) || (s->condition_audio_elements && !s->condition_audio))
        return fail(error,size,"missing condition rows");
    size_t provenance_offset=0;
    uint64_t records=0;
    for(int i=0;i<8;i++) records|=(uint64_t)s->provenance[i]<<(8*i);
    provenance_offset=8;
    if(records>12 || (s->ref2va && records!=s->reference_count) || (!s->ref2va && records>2))
        return fail(error,size,"invalid provenance reference count");
    for(uint64_t r=0;r<records;r++) {
        if(s->provenance_bytes-provenance_offset<16) return fail(error,size,"truncated provenance record");
        uint64_t kind=0,embedded=0;
        for(int i=0;i<8;i++) { kind|=(uint64_t)s->provenance[provenance_offset+i]<<(8*i); embedded|=(uint64_t)s->provenance[provenance_offset+8+i]<<(8*i); }
        provenance_offset+=16;
        if(kind<1 || kind>6 || embedded>1 || (s->ref2va && kind>4) || (!s->ref2va && kind<5))
            return fail(error,size,"invalid provenance kind/audio setting");
        if((p->adaptive_cache||p->cuda_attention==H3_ATTENTION_SUBBLOCK)&&s->reference_count) {
            h3_layout_ref_kind expected=kind==H3_REFERENCE_IMAGE?H3_LAYOUT_REF_IMAGE:
                kind==H3_REFERENCE_AUDIO?H3_LAYOUT_REF_AUDIO:H3_LAYOUT_REF_VIDEO;
            int has_audio=kind==H3_REFERENCE_AUDIO||kind==H3_REFERENCE_VIDEO_AUDIO||
                (kind==H3_REFERENCE_VIDEO&&embedded);
            if(expected!=s->references[r].kind||has_audio!=(s->references[r].audio_t>0))
                return fail(error,size,"approximate reference provenance kind/audio differs from layout");
        }
        for(int stream=0;stream<2;stream++) {
            if(s->provenance_bytes-provenance_offset<8) return fail(error,size,"truncated provenance path");
            uint64_t length=0; for(int i=0;i<8;i++) length|=(uint64_t)s->provenance[provenance_offset+i]<<(8*i); provenance_offset+=8;
            if(length>s->provenance_bytes-provenance_offset || s->provenance_bytes-provenance_offset-length<40 ||
                (!stream && !length) || memchr(s->provenance+provenance_offset,0,(size_t)length)) return fail(error,size,"invalid provenance path length");
            provenance_offset+=(size_t)length+40;
        }
    }
    if(provenance_offset!=s->provenance_bytes) return fail(error,size,"trailing provenance bytes");
    if(s->token_ids && s->token_count!=s->text.tokens) return fail(error,size,"invalid diagnostic token count");
    if(s->presentation_span_count && (!s->presentation_positions || !s->presentation_spans)) return fail(error,size,"missing presentation diagnostics");
    for(size_t i=0;i<s->presentation_span_count;i++) {
        uint64_t begin=s->presentation_spans[2*i],count=s->presentation_spans[2*i+1];
        if(begin>s->token_count || count>s->token_count-begin) return fail(error,size,"invalid diagnostic span");
    }
    const h3_layout *l=&s->layout;
    if(l->signature[0]!=(int)s->text.tokens || l->signature[1]!=s->latent_t || l->signature[2]!=s->latent_h ||
        l->signature[3]!=s->latent_w || l->signature[4]!=s->audio_t || l->img_target_rows!=s->video_elements/96 ||
        l->audio_target_rows!=s->audio_elements/32 || l->img_cond_rows!=s->condition_video_elements/96 ||
        l->audio_cond_rows!=s->condition_audio_elements/32 || s->condition_video_elements%96 || s->condition_audio_elements%32 ||
        !l->segments || !l->positions || !l->seq_len || l->seq_len>10000000 || !l->segment_count || l->segment_count>1000)
        return fail(error,size,"invalid resolved layout dimensions/signature");
    size_t stop=0,video=0,audio=0,cv=0,ca=0,nt=0;
    for(size_t i=0;i<l->segment_count;i++) {
        const h3_segment *g=&l->segments[i];
        if(g->start!=stop || g->stop<=g->start || g->stop>l->seq_len || g->kind<H3_SEG_TEXT || g->kind>H3_SEG_VIDEO)
            return fail(error,size,"invalid layout segment boundary/kind");
        size_t rows=g->stop-g->start; stop=g->stop;
        switch(g->kind) {
            case H3_SEG_VIDEO: video+=rows; break;
            case H3_SEG_AUDIO: audio+=rows; break;
            case H3_SEG_TEXT: nt+=rows; break;
            case H3_SEG_REF_AUDIO: ca+=rows; break;
            default: cv+=rows; break;
        }
    }
    if(stop!=l->seq_len || video!=l->img_target_rows || audio!=l->audio_target_rows || nt!=s->text.tokens ||
        cv!=l->img_cond_rows || ca!=l->audio_cond_rows) return fail(error,size,"layout row count mismatch");
    for(size_t i=0;i<l->seq_len;i++) if(!isfinite(l->positions[i].t) || !isfinite(l->positions[i].h) || !isfinite(l->positions[i].w))
        return fail(error,size,"invalid layout positions");
    for(size_t i=0;i<s->reference_count;i++) {
        const h3_layout_ref *r=&s->references[i];
        if(r->kind<H3_LAYOUT_REF_IMAGE || r->kind>H3_LAYOUT_REF_VIDEO || r->audio_t<0 || r->audio_t>1000000 ||
            (r->kind!=H3_LAYOUT_REF_AUDIO && (r->latent_t<1 || r->latent_t>10000 || r->latent_h<2 ||
                r->latent_w<2 || r->latent_h%2 || r->latent_w%2 || r->latent_h>4096 || r->latent_w>4096)))
            return fail(error,size,"invalid ordered layout reference");
    }
    if(p->adaptive_cache||p->cuda_attention==H3_ATTENTION_SUBBLOCK) {
        uint64_t visual_rows=0,audio_rows=0;
        for(size_t i=0;i<s->reference_count;i++) {
            const h3_layout_ref *r=&s->references[i];
            if(r->kind==H3_LAYOUT_REF_IMAGE&&(r->latent_t!=1||r->audio_t||
               (uint64_t)r->latent_h*(uint64_t)r->latent_w>H3_REFERENCE_IMAGE_MAX_PATCHES))
                return fail(error,size,"invalid approximate image reference shape/patch count");
            if(r->kind==H3_LAYOUT_REF_AUDIO&&(r->latent_t||r->latent_h||r->latent_w||!r->audio_t))
                return fail(error,size,"invalid approximate audio reference shape");
            if(r->kind!=H3_LAYOUT_REF_AUDIO)
                visual_rows+=(uint64_t)r->latent_t*(unsigned)(r->latent_h/2)*(unsigned)(r->latent_w/2);
            audio_rows+=(uint64_t)r->audio_t*2;
        }
        /* Bound reconstructed allocation by the already decoded layout, never
         * by unchecked reference dimensions from an adversarial descriptor. */
        if(visual_rows!=l->img_cond_rows||audio_rows!=l->audio_cond_rows)
            return fail(error,size,"reference geometry/condition rows mismatch before allocation");
        h3_layout_spec spec={(int)s->text.tokens,s->latent_t,s->latent_h,s->latent_w,
            s->audio_t,s->aligned_frames,NULL,0,s->references,s->reference_count};
        h3_layout expected={0};
        if(!h3_layout_build(&spec,&expected,error,size))return 0;
        int matches=expected.seq_len==l->seq_len&&expected.segment_count==l->segment_count&&
            expected.img_cond_rows==l->img_cond_rows&&expected.audio_cond_rows==l->audio_cond_rows&&
            !memcmp(expected.positions,l->positions,l->seq_len*sizeof(*l->positions));
        for(size_t i=0;matches&&i<l->segment_count;i++)matches=
            expected.segments[i].start==l->segments[i].start&&expected.segments[i].stop==l->segments[i].stop&&
            expected.segments[i].kind==l->segments[i].kind;
        h3_layout_free(&expected);
        if(!matches)return fail(error,size,"approximate reference provenance/layout mismatch");
    }
    return 1;
}

int h3_sampler_adaptive_metadata_validate(const h3_sampler_state *s,char *error,size_t size) {
    if(!s)return fail(error,size,"missing adaptive metadata");
    const h3_params *p=&s->params;const h3_adaptive_history *a=&s->adaptive_history;
    h3_adaptive_plan plan;
    if(p->adaptive_cache<1||p->adaptive_cache>2||
       !h3_adaptive_controls_valid(p,error,size)||!h3_warmups_valid(p,error,size)||
       s->adaptive_version!=h3_adaptive_execution_recipe(p->adaptive_cache,p->cuda_denoise_quant,s->reference_count!=0,s->continuation)||
       s->sampler_mode||p->core_reuse!=1||s->reuse_interval!=1||p->dit_layers!=50||p->token_reduction||
       !h3_adaptive_plan_recipe(p->adaptive_cache,s->adaptive_version,s->full_sequence,5376,p->adaptive_cache_max_bytes,&plan,error,size)||
       s->adaptive_elements!=plan.elements||s->next_step<0||s->next_step>s->total_steps||
       a->ready!=(unsigned)(s->next_step!=0)||
       (!a->ready&&(a->streak||a->phase||a->last_step||a->last_refresh))||
       (h3_adaptive_threshold(p)==0&&a->streak)||
       (a->ready&&(a->last_step!=s->next_step-1||a->last_refresh<0||a->last_refresh>a->last_step||
         a->last_refresh<(a->last_step<h3_adaptive_warmup(p->adaptive_cache_warmup)?a->last_step:h3_adaptive_warmup(p->adaptive_cache_warmup)-1)||
         (a->last_step==s->total_steps-1&&a->streak)||
         (a->phase&&a->last_refresh<h3_subblock_warmup(p->subblock_warmup))||
         a->streak!=(unsigned)(a->last_step-a->last_refresh)||a->streak>(unsigned)h3_adaptive_max_hits(p)||
         a->phase!=(unsigned)(p->cuda_attention==H3_ATTENTION_SUBBLOCK&&a->last_step>=h3_subblock_warmup(p->subblock_warmup)))))
        return fail(error,size,"invalid adaptive recipe, shape or committed history before allocation");
    return 1;
}
int h3_sampler_state_validate(const h3_sampler_state *s,char *error,size_t size) {
    if(!s || s->version!=H3_SAMPLE_VERSION || s->sampler_mode>1) return fail(error,size,"unsupported state schema");
    if(s->params._arithmetic_recipe && (!h3_sampler_checkpoint_options(&s->params,NULL,error,size) || s->sampler_mode))
        return fail(error,size,"unsupported SGLang reference checkpoint policy or sampler mode");
    if(s->params.metal_attention.weight_format&&
       (s->params.backend!=H3_BACKEND_METAL||s->params.ssd_streaming))
        return fail(error,size,"Q8 checkpoint requires resident Metal weights");
    if(s->ane_decided>>50)return fail(error,size,"invalid ANE decision mask");
    if(s->params.metal_attention.ane_mode&&s->next_step>0&&s->ane_decided!=((1ull<<50)-1))
        return fail(error,size,"missing completed ANE split decisions");
    for(unsigned j=0;j<50;j++) {
        uint32_t rows=s->ane_rows[j];
        if(rows>16384||rows>s->layout.seq_len/2||(rows&&(s->params.metal_attention.ane_rows<0||rows>(unsigned)s->params.metal_attention.ane_rows))||(!(s->ane_decided&(1ull<<j))&&rows)||
           (rows&&(s->params.metal_attention.ane_chunk<=0||rows%(unsigned)s->params.metal_attention.ane_chunk))||
           (s->ane_decided&&(!s->params.backend||!s->params.metal_attention.ane_mode)))
            return fail(error,size,"invalid ANE split decisions");
    }
    if((s->params.backend!=H3_BACKEND_MPSGRAPH_REFERENCE&&s->params.backend!=H3_BACKEND_METAL)||
       (s->params.attention_mode!=H3_ATTN_DENSE&&s->params.attention_mode!=H3_ATTN_SOL)||
       (s->params.backend==H3_BACKEND_MPSGRAPH_REFERENCE&&s->params.attention_mode!=H3_ATTN_DENSE)||
       s->metal_attention_version!=(s->params.backend?H3_METAL_ATTENTION_VERSION:0u)||
       (s->params.backend&&!h3_metal_options_valid(s->params.metal_attention,error,size)))
        return fail(error,size,"unsupported Metal backend or attention recipe");
    if(!h3_cuda_sol_params_valid(&s->params,error,size)||!h3_warmups_valid(&s->params,error,size)||!h3_adaptive_controls_valid(&s->params,error,size))return 0;
    if(s->params.cuda_attention<0||s->params.cuda_attention>4||
       s->attention_version!=h3_attention_execution_recipe(s->params.cuda_attention,s->params.cuda_denoise_quant)||
       s->attention_plan!=h3_attention_plan(s->params.cuda_attention))
        return fail(error,size,"unsupported attention policy, recipe or plan");
    if(s->params.cuda_denoise_quant<0||s->params.cuda_denoise_quant>2||
       s->quant_version!=h3_quant_execution_recipe(s->params.cuda_denoise_quant,s->params.adaptive_cache,s->params.cuda_attention))
        return fail(error,size,"unsupported denoiser quantization checkpoint recipe");
    if(s->params.cuda_denoise_quant && s->params.adaptive_cache &&
       (s->params.cuda_attention || s->params.adaptive_cache>1))
        return fail(error,size,"quantized adaptive checkpoints require conservative cache and dense attention");
    h3_av_state_info shape;
    const h3_params *p=&s->params;
    if(!h3_upscale_options_valid(p,error,size))return 0;
    if(!h3_av_state_shape_profile(s->render_width,s->render_height,s->aligned_frames,p->geometry_profile,&shape) ||
        s->aligned_frames!=h3_align_frame_count(p->frames) || s->aligned_frames<22 ||
        shape.video_t!=s->latent_t || shape.audio_t!=s->audio_t || shape.latent_h!=s->latent_h || shape.latent_w!=s->latent_w ||
        shape.video_elements!=s->video_elements || shape.audio_elements!=s->audio_elements ||
        !h3_geometry_valid(p->width,p->height,p->geometry_profile) ||
        s->render_width!=(p->render_width?p->render_width:p->width) || s->render_height!=(p->render_height?p->render_height:p->height) ||
        s->render_width>p->width || s->render_height>p->height || (int64_t)s->render_width*p->height!=(int64_t)s->render_height*p->width)
        return fail(error,size,"invalid latent/output geometry");
    if(s->total_steps<1 || s->total_steps>H3_MAX_STEPS || s->next_step<0 || s->next_step>s->total_steps ||
        s->sigmas.steps!=s->total_steps || p->steps!=s->total_steps || s->reuse_interval!=p->denoise_reuse ||
        s->reuse_interval<1 || s->reuse_interval>3 || p->core_reuse<1 || p->core_reuse>6 || (p->core_reuse>1 && s->reuse_interval>1) ||
        p->dit_layers<H3_MIN_DIT_LAYERS || p->dit_layers>H3_DEFAULT_DIT_LAYERS)
        return fail(error,size,"invalid schedule, step index or unsupported sampler configuration");
    for(int i=0;i<s->total_steps;i++) if(!isfinite(s->sigmas.video[i]) || !isfinite(s->sigmas.audio[i]) ||
        s->sigmas.video[i]<=s->sigmas.video[i+1] || (s->upscale.stage==2?s->sigmas.audio[i]!=0:s->sigmas.audio[i]<=s->sigmas.audio[i+1]) ||
        s->sigmas.video[i]>1 || s->sigmas.audio[i]>1 || s->selected[i]>1 || (s->reuse_interval==1 && !s->selected[i]))
        return fail(error,size,"invalid sigma/reuse schedule");
    if(s->sigmas.video[s->total_steps]!=0 || s->sigmas.audio[s->total_steps]!=0 ||
        !s->selected[0] || !s->selected[s->total_steps-1]) return fail(error,size,"invalid terminal sigma/evaluation");
    int last=-1,previous=-1;
    for(int i=0;i<s->next_step;i++) if(s->selected[i]) { previous=last; last=i; }
    if(s->reuse_interval==1 && !s->sampler_mode && (s->last_evaluated!=-1 || s->previous_evaluated!=-1)) return fail(error,size,"unexpected reuse history for reuse=1");
    if((s->reuse_interval>1 || s->sampler_mode) && (s->last_evaluated!=last || s->previous_evaluated!=previous))
        return fail(error,size,"inconsistent reuse history indices");
    if(s->reuse_interval>1 && (s->sampler_mode ?
        (!s->gpu_last_video || !s->gpu_previous_video || !s->gpu_last_audio || !s->gpu_previous_audio) :
        (!s->last_video_velocity || !s->previous_video_velocity || !s->last_audio_velocity || !s->previous_audio_velocity)))
        return fail(error,size,"missing native reuse velocity history");
    if(s->params.adaptive_cache) {
        if(!h3_sampler_adaptive_metadata_validate(s,error,size))return 0;
        if(!s->adaptive_anchor||!s->adaptive_delta)return fail(error,size,"missing adaptive tensors");
        if(s->adaptive_history.ready)for(size_t i=0;i<s->adaptive_elements;i++)
            if((s->adaptive_anchor[i]&0x7f80)==0x7f80||(s->adaptive_delta[i]&0x7f80)==0x7f80)
                return fail(error,size,"nonfinite adaptive checkpoint tensor");
    } else if(s->adaptive_version||s->adaptive_elements||s->adaptive_anchor||s->adaptive_delta)
        return fail(error,size,"unexpected adaptive state");
    if(s->execution_version!=1 || s->core_residual_ready>1 || s->reduction_enabled>1 || s->reduction_active ||
        (s->full_sequence && s->full_sequence!=s->layout.seq_len)) return fail(error,size,"invalid execution topology/version");
    if(p->core_reuse>1) {
        unsigned evaluations=0; for(int i=0;i<s->next_step;i++) evaluations+=s->selected[i]!=0;
        if(s->core_forward_count!=evaluations || s->core_residual_ready!=(evaluations!=0) ||
            s->core_rows!=s->layout.seq_len || s->core_columns!=5376 ||
            s->core_elements!=s->core_rows*s->core_columns || !s->core_residual)
            return fail(error,size,"missing or inconsistent core reuse state/topology");
    } else if(s->core_elements || s->core_residual_ready) return fail(error,size,"unexpected core residual");
    if(p->token_reduction && !s->reduction_enabled) return fail(error,size,"missing resolved token reduction");
    if(s->reduction_enabled && (s->reduction_begin>=s->reduction_end || s->reduction_end>50 ||
        s->reduction_early_end<s->reduction_end || s->reduction_early_end>50 || s->reduction_early_steps>1000 ||
        !isfinite(s->reduction_scale) || s->reduction_scale<0 || s->reduction_scale>2 ||
        s->reduced_sequence!=s->layout.seq_len-s->layout.img_target_rows+
            (size_t)s->latent_t*(s->latent_h/2)*((s->latent_w/2+1)/2)))
        return fail(error,size,"invalid resolved token reduction topology");
    if(s->prepared.count>H3_PREPARED_MAX) return fail(error,size,"too many prepared tensors");
    for(size_t i=0;i<s->prepared.count;i++) {
        const h3_prepared_tensor *t=&s->prepared.tensors[i];
        if(!t->elements || t->elements>UINT64_C(8)*1024*1024*1024/2 || !t->values)
            return fail(error,size,"invalid prepared tensor");
        for(size_t j=0;j<i;j++) if(t->id==s->prepared.tensors[j].id) return fail(error,size,"duplicate prepared tensor");
    }
    if(s->resume_count && (s->resume_step>(uint32_t)s->next_step || s->resume_format!=H3_SAMPLE_VERSION))
        return fail(error,size,"invalid resume provenance");
    const int booleans[]={p->token_reduction,p->use_int8_row_fc2,p->use_reference_rope,p->ssd_streaming,
        p->use_slower_bf16_mlp,p->use_slower_bf16_qkv,p->use_slower_bf16_attention_output,
        p->use_slower_row_major_attention_output,p->use_slower_unfused_int8_inputs,
        p->use_slower_unfused_qkv_rope,p->use_slower_scalar_qkv_rms,p->use_slower_uncached_int8_scales,
        p->use_slower_dynamic_fc1_k,p->use_slower_grouped_quantizer,p->keep_continuation_prefix};
    for(size_t i=0;i<sizeof(booleans)/sizeof(*booleans);i++) if(booleans[i]!=0 && booleans[i]!=1)
        return fail(error,size,"invalid numerical implementation flag");
    float rope=!p->use_reference_rope && s->render_width==256 && s->render_height==256?0.5f:1.0f;
    if(s->spatial_rope_scale!=rope || (p->continuation_mode!=H3_CONTINUE_HARD && p->continuation_mode!=H3_CONTINUE_BRIDGE) ||
        (p->reference_image_size!=H3_REFERENCE_IMAGE_MATCH && p->reference_image_size!=H3_REFERENCE_IMAGE_MAX &&
         p->reference_image_size!=H3_REFERENCE_IMAGE_HIGH))
        return fail(error,size,"inconsistent mode/numerical configuration");
    if(!h3_sampler_layout_validate(s,error,size))return 0;
    const h3_layout *l=&s->layout;
    h3_denoise_prefix prefix={0};
    if(s->continuation!=0 && s->continuation!=1) return fail(error,size,"invalid continuation flag");
    if(s->continuation && (!h3_continuation_context(s->context_frames,&prefix) || s->context_frames>=s->aligned_frames ||
        s->context_frames!=(p->continuation_context_frames?p->continuation_context_frames:39)))
        return fail(error,size,"invalid continuation context");
    if(s->upscale.stage==2)prefix.audio_prefix_t=s->audio_t;
    if(l->frozen_audio!=(s->upscale.stage==2))return fail(error,size,"invalid clean-audio stage tag");
    if(prefix.video_prefix_t!=l->prefix.video_prefix_t || prefix.audio_prefix_t!=l->prefix.audio_prefix_t ||
        s->clean_coefficient!=0.999f || s->noise_coefficient!=0.001f || s->audio_preservation!=1)
        return fail(error,size,"invalid continuation mask/initialization metadata");
    if(l->bridge && (!s->continuation || p->continuation_mode!=H3_CONTINUE_BRIDGE || !h3_bridge_profile_valid(l->bridge) ||
        l->bridge->prefix.video_prefix_t!=prefix.video_prefix_t || l->bridge->prefix.audio_prefix_t!=prefix.audio_prefix_t ||
        l->bridge->max_strength!=p->bridge_max_strength || l->bridge->type!=p->bridge_profile || l->bridge->video_bridge_t!=p->bridge_video_steps))
        return fail(error,size,"invalid bridge row classes");
    if(!s->continuation && (p->keep_continuation_prefix || p->continuation_mode==H3_CONTINUE_BRIDGE || s->context_frames)) return fail(error,size,"continuation configuration without a continuation mask");
    if(s->continuation && p->dit_layers!=50) return fail(error,size,"continuation requires all 50 active blocks");
    if(s->continuation && p->continuation_mode==H3_CONTINUE_BRIDGE && !l->bridge) return fail(error,size,"missing bridge mask");
    if(s->rng_version!=(p->_arithmetic_recipe?2u:1u) || s->video_random_count!=s->video_elements || s->audio_random_count!=(s->upscale.stage==2?0:s->audio_elements) ||
        !(s->video_rng.increment&1) || !(s->audio_rng.increment&1) || s->video_rng.has_spare<0 || s->video_rng.has_spare>1 ||
        s->audio_rng.has_spare<0 || s->audio_rng.has_spare>1) return fail(error,size,"invalid RNG state/version");
    const float *arrays[]={s->video,s->audio,s->condition_video,s->condition_audio,
        s->original_video_noise,s->original_audio_noise,s->last_video_velocity,s->previous_video_velocity,
        s->last_audio_velocity,s->previous_audio_velocity};
    size_t counts[]={s->video_elements,s->audio_elements,s->condition_video_elements,s->condition_audio_elements,
        s->video_elements,s->audio_elements,s->video_elements,s->video_elements,s->audio_elements,s->audio_elements};
    if(!s->video || !s->audio) return fail(error,size,"missing current AV latent");
    for(size_t a=0;a<sizeof(arrays)/sizeof(*arrays);a++) if(arrays[a])
        for(size_t i=0;i<counts[a];i++) if(!isfinite(arrays[a][i])) return fail(error,size,"non-finite F32 state");
    return h3_upscale_record_valid(s,error,size);
}

void h3_prepared_cache_free(h3_prepared_cache *cache) {
    for(size_t i=0;i<cache->count && i<H3_PREPARED_MAX;i++) free(cache->tensors[i].values);
    memset(cache,0,sizeof(*cache));
}
const h3_prepared_tensor *h3_prepared_find(const h3_prepared_cache *cache,uint32_t id,size_t elements) {
    if(!cache || cache->version!=1) return NULL;
    for(size_t i=0;i<cache->count;i++) if(cache->tensors[i].id==id && cache->tensors[i].elements==elements) return &cache->tensors[i];
    return NULL;
}
size_t h3_sampler_prepared_bytes(const h3_sampler_state *s) {
    size_t n=0; for(size_t i=0;i<s->prepared.count;i++) n+=s->prepared.tensors[i].elements*2; return n;
}
void h3_sampler_log(const h3_sampler_state *s,const char *action,uint64_t bytes,double seconds) {
    fprintf(stderr,"h3cli: checkpoint arithmetic=%d\n",s->params._arithmetic_recipe);
    if(s->params.cuda_attention)fprintf(stderr,"h3cli: checkpoint attention=%s recipe=%u plan=%u\n",h3_attention_name(s->params.cuda_attention),s->attention_version,s->attention_plan);
    if(s->params.cuda_denoise_quant)fprintf(stderr,"h3cli: checkpoint denoise quantization=%s recipe=%u\n",h3_quant_name(s->params.cuda_denoise_quant),s->quant_version);
    fprintf(stderr,"h3cli: sampler %s: %s %s, geometry %dx%d / %d frames, schedule=%d completed=%d next=%d, sigma V=%.9g A=%.9g\n"
        "h3cli: sampler prefix V=%d A=%d, conditioning=%zu prepared=%zu checkpoint=%llu bytes, %.3f s\n",
        action,s->ref2va?"Ref2VA":"FL2VA",s->sampler_mode?"GPU-state":"CPU-state",s->render_width,s->render_height,s->aligned_frames,
        s->total_steps,s->next_step,s->next_step,(double)s->sigmas.video[s->next_step],(double)s->sigmas.audio[s->next_step],
        s->layout.prefix.video_prefix_t,s->layout.prefix.audio_prefix_t,
        s->text.tokens*s->text.width*2+(s->condition_video_elements+s->condition_audio_elements)*4,
        h3_sampler_prepared_bytes(s),(unsigned long long)bytes,seconds);
}
