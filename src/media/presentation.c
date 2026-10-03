#include "src/denoise/attention.h"
#include "src/media/delivery.h"
#include "src/sampling/av_state.h"
#include "src/weights/quant.h"
#include "src/denoise/adaptive_cache.h"
#include "src/weights/lora.h"
#include <ctype.h>
#include <math.h>
#include <errno.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static void fingerprint(const h3_av_state *state,char text[65]) {
    uint8_t digest[32];h3_av_state_fingerprint(state,digest);
    for(int i=0;i<32;i++)snprintf(text+i*2,3,"%02x",digest[i]);
}
static int hex64(const char *s) {
    if(strlen(s)!=64)return 0;
    for(int i=0;i<64;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return 0;
    return 1;
}
/* Both backends use one explicit presentation schema. Codec selects current
 * backend rounding, independently of denoiser approximation and delivery. */
int h3_presentation_validate(const h3_presentation *p,const h3_av_state_info *i,char *error,size_t size) {
    if(!p||!i||p->version!=H3_PRESENTATION_VERSION||i->version!=3||p->av_metadata_identity!=1||
       (p->ref2va!=0&&p->ref2va!=1)||p->geometry_profile!=i->geometry_profile||
       (p->geometry_profile!=0&&p->geometry_profile!=1)||
       p->cuda_denoise_quant<0||p->cuda_denoise_quant>2||
       (p->cuda_denoise_quant&&(p->quant_version!=h3_quant_execution_recipe(p->cuda_denoise_quant,p->adaptive_cache,p->cuda_attention)||!hex64(p->quant_model_sha256)))||
       (!p->cuda_denoise_quant&&(p->quant_version||p->quant_model_sha256[0]||p->quant_model_metadata))||
       (p->quant_model_metadata!=0&&p->quant_model_metadata!=1)||
       (p->cuda_attention<0||p->cuda_attention>4)||
       p->attention_version!=h3_attention_execution_recipe(p->cuda_attention,p->cuda_denoise_quant)||
       p->attention_plan!=h3_attention_plan(p->cuda_attention)||
       (p->cuda_attention==H3_ATTENTION_SOL&&!h3_cuda_sol_options_valid(p->cuda_sol,NULL,0))||
       p->adaptive_cache<0||p->adaptive_cache>2||
       (p->adaptive_version==H3_ADAPTIVE_CONTINUATION_VERSION?
           (!p->adaptive_cache||p->cuda_denoise_quant||p->geometry_profile||p->upscale_recipe):
           (p->adaptive_version!=h3_adaptive_recipe(p->adaptive_cache,p->cuda_denoise_quant,p->ref2va)||
            (p->adaptive_cache&&p->trim_frames)))||
       !isfinite(p->adaptive_cache_threshold)||p->adaptive_cache_threshold<0||p->adaptive_cache_threshold>1||
       (p->adaptive_cache?(p->adaptive_cache_max_hits<1||p->adaptive_cache_max_hits>16):
          (p->adaptive_cache_threshold!=0||p->adaptive_cache_max_hits!=0))||
       (p->adaptive_cache&&p->ref2va&&p->cuda_denoise_quant)||
       (p->adaptive_cache&&p->cuda_attention&&p->cuda_attention!=H3_ATTENTION_SUBBLOCK)||
       (p->cuda_denoise_quant&&p->adaptive_cache&&(p->adaptive_cache!=1||p->cuda_attention))||
       !isfinite(p->subblock_sparsity)||p->subblock_sparsity<0||p->subblock_sparsity>=1||
       (p->cuda_attention!=H3_ATTENTION_SUBBLOCK&&p->subblock_sparsity!=0)||
       (p->adaptive_cache?(p->adaptive_cache_warmup<2||p->adaptive_cache_warmup>16):p->adaptive_cache_warmup!=0)||
       (p->cuda_attention==H3_ATTENTION_SUBBLOCK?(p->subblock_warmup<2||p->subblock_warmup>16):p->subblock_warmup!=0)||
       (p->upscale_recipe!=0&&p->upscale_recipe!=1&&p->upscale_recipe!=2)||
       (p->upscale_recipe&&(p->cuda_denoise_quant||p->cuda_attention||p->adaptive_cache||p->trim_frames||p->trim_samples||
         p->width!=p->render_width||p->height!=p->render_height||
         (p->upscale_steps!=0&&p->upscale_steps!=2&&p->upscale_steps!=3&&p->upscale_steps!=4)||
         !isfinite(p->upscale_sigma)||(p->upscale_steps?(p->upscale_sigma<=0||p->upscale_sigma>.5f):p->upscale_sigma!=0)||
         !hex64(p->upscale_parent_sha256)||!hex64(p->upscale_artifact_sha256)))||
       (!p->upscale_recipe&&(p->upscale_steps||p->upscale_sigma!=0||p->upscale_parent_sha256[0]||p->upscale_artifact_sha256[0]))||
       (p->geometry_profile&&(p->width>1920||p->height>1920))||
       p->render_width!=i->render_width||p->render_height!=i->render_height||
       p->width<32||p->height<32||p->width%2||p->height%2||
       (int64_t)p->width*p->height>(p->geometry_profile?INT64_C(2088960):H3_MAX_PIXELS)||p->width<p->render_width||p->height<p->render_height||
       (int64_t)p->width*p->render_height!=(int64_t)p->height*p->render_width||
       p->trim_frames<0||p->trim_frames>=i->frames||p->trim_samples<0||p->trim_samples>=i->audio_t*800||
       (p->trim_frames&&(p->trim_frames<39||(p->trim_frames-39)%51))||
       (int64_t)p->trim_frames*32000!=(int64_t)p->trim_samples*24||
       p->fps!=24||p->sample_rate!=32000||(p->codec_version!=1&&p->codec_version!=2)||
       (p->codec_version==1&&(p->cuda_attention||p->cuda_denoise_quant||p->adaptive_cache))||
       (p->tiny_sha256[0]&&!hex64(p->tiny_sha256))){
        snprintf(error,size,"invalid current presentation version, geometry, trimming, or codec contract");return 0;
    }return 1;
}
#define PRESENTATION_FORMAT "H3-PRESENTATION %d\nstate %s\nvariant %d\nrender %d %d\noutput %d %d\ntrim %d %d\nfps %d\naudio_rate %d\ncodec %d\ntiny %s\n" \
    "denoise_quant %d %u %s\nquant_metadata %d\nattention %d %u %u\nsol %d %d %d %d %d %a %a %a\n" \
    "adaptive %d %u\nadaptive_controls %a %d\nsubblock %a\nwarmup %d %d\ngeometry %d %d\nupscale %d %d %a %s %s\n"
static int presentation_text(char *text,size_t size,const h3_presentation *p,const char *hash) {
    const h3_cuda_sol_options *o=&p->cuda_sol;
    return snprintf(text,size,PRESENTATION_FORMAT,p->version,hash,p->ref2va,p->render_width,p->render_height,p->width,p->height,
        p->trim_frames,p->trim_samples,p->fps,p->sample_rate,p->codec_version,p->tiny_sha256[0]?p->tiny_sha256:"none",
        p->cuda_denoise_quant,p->quant_version,p->quant_model_sha256[0]?p->quant_model_sha256:"none",p->quant_model_metadata,
        p->cuda_attention,p->attention_version,p->attention_plan,o->q_block,o->kv_block,o->dense_layers,o->dense_steps,o->local_radius,
        (double)o->tau,(double)o->min_exact,(double)o->dense_sigma,p->adaptive_cache,p->adaptive_version,(double)p->adaptive_cache_threshold,p->adaptive_cache_max_hits,(double)p->subblock_sparsity,
        p->adaptive_cache_warmup,p->subblock_warmup,p->geometry_profile,p->av_metadata_identity,p->upscale_recipe,p->upscale_steps,
        (double)p->upscale_sigma,p->upscale_parent_sha256[0]?p->upscale_parent_sha256:"none",
        p->upscale_artifact_sha256[0]?p->upscale_artifact_sha256:"none");
}
int h3_presentation_load(const char *state_path,const h3_av_state *state,h3_presentation *p,char *error,size_t size) {
    if(!state_path||!state||!p){snprintf(error,size,"invalid presentation arguments");return -1;}
    char *path=malloc(strlen(state_path)+20);if(!path){snprintf(error,size,"out of memory reading presentation");return -1;}
    snprintf(path,strlen(state_path)+20,"%s.presentation",state_path);FILE *f=fopen(path,"rb");free(path);
    if(!f){snprintf(error,size,"current AV state requires presentation metadata: %s",strerror(errno));return -1;}
    char text[2048],state_hash[65];size_t n=fread(text,1,sizeof(text)-1,f);
    int extra=fgetc(f),bad=ferror(f);fclose(f);text[n]=0;int used=0;
    memset(p,0,sizeof(*p));h3_cuda_sol_options *o=&p->cuda_sol;
    int fields=sscanf(text,"H3-PRESENTATION %d\nstate %64s\nvariant %d\nrender %d %d\noutput %d %d\ntrim %d %d\nfps %d\naudio_rate %d\ncodec %d\ntiny %64s\n"
        "denoise_quant %d %u %64s\nquant_metadata %d\nattention %d %u %u\nsol %d %d %d %d %d %a %a %a\n"
        "adaptive %d %u\nadaptive_controls %a %d\nsubblock %a\nwarmup %d %d\ngeometry %d %d\nupscale %d %d %a %64s %64s\n%n",
        &p->version,state_hash,&p->ref2va,&p->render_width,&p->render_height,&p->width,&p->height,
        &p->trim_frames,&p->trim_samples,&p->fps,&p->sample_rate,&p->codec_version,p->tiny_sha256,
        &p->cuda_denoise_quant,&p->quant_version,p->quant_model_sha256,&p->quant_model_metadata,
        &p->cuda_attention,&p->attention_version,&p->attention_plan,&o->q_block,&o->kv_block,&o->dense_layers,&o->dense_steps,&o->local_radius,
        &o->tau,&o->min_exact,&o->dense_sigma,&p->adaptive_cache,&p->adaptive_version,&p->adaptive_cache_threshold,&p->adaptive_cache_max_hits,&p->subblock_sparsity,
        &p->adaptive_cache_warmup,&p->subblock_warmup,&p->geometry_profile,&p->av_metadata_identity,
        &p->upscale_recipe,&p->upscale_steps,&p->upscale_sigma,p->upscale_parent_sha256,p->upscale_artifact_sha256,&used);
    if(fields!=42||p->version!=9||!used||(size_t)used!=n||extra!=EOF||bad||memchr(text,0,n)||!hex64(state_hash)){
        snprintf(error,size,"malformed or unsupported presentation (expected current version 9)");return -1;
    }
    char *hashes[]={p->tiny_sha256,p->quant_model_sha256,p->upscale_parent_sha256,p->upscale_artifact_sha256};
    for(size_t j=0;j<sizeof(hashes)/sizeof(*hashes);j++)if(!strcmp(hashes[j],"none"))memset(hashes[j],0,65);
    char canonical[2048];int length=presentation_text(canonical,sizeof(canonical),p,state_hash);
    if(length<0||(size_t)length!=n||memcmp(text,canonical,n)){
        snprintf(error,size,"presentation has noncanonical or overflowing fields");return -1;
    }
    char actual[65];fingerprint(state,actual);
    if(strcmp(actual,state_hash)){snprintf(error,size,"stale or mismatched presentation: state fingerprint differs");return -1;}
    return h3_presentation_validate(p,&state->info,error,size)?1:-1;
}
int h3_result_save_av_state(const h3_result *result,const char *path,char *error,size_t size) {
    char local[512];if(!error||!size){error=local;size=sizeof(local);}error[0]=0;
    if(!result||!result->av_state||result->status!=H3_RESULT_COMPLETE||!path||!*path){
        snprintf(error,size,"only complete AV results can be saved with presentation");return 0;
    }
    if(!h3_presentation_validate(&result->presentation,&result->av_state->info,error,size))return 0;
    size_t capacity=strlen(path)+64;char *state_tmp=malloc(capacity),*meta_tmp=malloc(capacity),*meta=malloc(capacity);
    if(!state_tmp||!meta_tmp||!meta){free(state_tmp);free(meta_tmp);free(meta);snprintf(error,size,"cannot allocate state paths");return 0;}
    snprintf(state_tmp,capacity,"%s.stage.XXXXXX",path);snprintf(meta,capacity,"%s.presentation",path);
    snprintf(meta_tmp,capacity,"%s.stage.XXXXXX",meta);
    int fd=mkstemp(state_tmp),ok=0;if(fd<0){snprintf(error,size,"cannot stage AV state: %s",strerror(errno));goto done;}close(fd);
    if(!h3_av_state_save(result->av_state,state_tmp,error,size))goto done;
    fd=mkstemp(meta_tmp);if(fd<0){snprintf(error,size,"cannot stage presentation: %s",strerror(errno));goto done;}
    FILE *f=fdopen(fd,"wb");if(!f){close(fd);snprintf(error,size,"cannot open staged presentation");goto done;}
    char hash[65],text[2048];fingerprint(result->av_state,hash);
    int length=presentation_text(text,sizeof(text),&result->presentation,hash);
    int written=length>0&&(size_t)length<sizeof(text)&&fwrite(text,1,(size_t)length,f)==(size_t)length?length:-1;
    int failed=written<0;if(fflush(f)||fsync(fileno(f)))failed=1;if(fclose(f))failed=1;
    if(failed){snprintf(error,size,"cannot write presentation");goto done;}
    /* Each replacement is atomic. A crash between them cannot be mistaken for
     * a matching pair: readers require the canonical state fingerprint. */
    if(rename(state_tmp,path)){snprintf(error,size,"cannot commit AV state: %s",strerror(errno));goto done;}
    if(rename(meta_tmp,meta)){snprintf(error,size,"AV state saved but presentation commit failed: %s; retry saving the pair",strerror(errno));goto done;}
    ok=h3_lora_save_provenance(path,result->lora_provenance,error,size);
done:unlink(state_tmp);unlink(meta_tmp);free(state_tmp);free(meta_tmp);free(meta);return ok;
}
