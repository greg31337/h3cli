/* Bounded latent-only qualification. Each invocation executes <= six
 * transitions of the original 50-step schedule; never writes a video. */
#include "src/h3.h"
#include "src/sampling/sampler_state.h"
#include "src/denoise/adaptive_cache.h"
#include "src/weights/quant.h"
#include "src/sampling/av_state.h"
#include "src/denoise/approximate.h"
#include "src/denoise/subblock.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
typedef struct { const char *base; int cancel_at,cancelled,steps; } trace;
static int capture(int step,int total,const float *video,size_t nv,
    const float *audio,size_t na,void *opaque) {
    trace *t=opaque;char path[4096];
    if(total!=t->steps||snprintf(path,sizeof(path),"%s.step-%02d.bin",t->base,step)>=(int)sizeof(path))return 1;
    FILE *f=fopen(path,"wb");if(!f)return 1;
    uint64_t dims[2]={nv,na};int ok=fwrite(dims,sizeof(dims),1,f)==1&&
        fwrite(video,4,nv,f)==nv&&fwrite(audio,4,na,f)==na;
    return fclose(f)||!ok;
}
static int stop_decode(const char *stage,int current,int total,void *opaque) {
    (void)total;trace *t=opaque;
    if(t->cancel_at&&!t->cancelled&&!strcmp(stage,"denoise")&&current==t->cancel_at){t->cancelled=1;return 1;}
    return !strcmp(stage,"audio VAE");
}
static int save_complete_av(const h3_sampler_state *s,char *error,size_t size) {
    const char *path=getenv("H3_TEST_SAVE_AV");if(!path)return 1;
    if(s->next_step!=s->total_steps||s->params.cuda_denoise_quant)return 0;
    h3_av_state *av=h3_av_state_new(s->render_width,s->render_height,s->aligned_frames,s->params.seed,s->av_signature);
    if(!av)return 0;
    memcpy(av->video,s->video,s->video_elements*4);memcpy(av->audio,s->audio,s->audio_elements*4);
    int trim=s->continuation&&!s->params.keep_continuation_prefix;
    h3_result result={.status=H3_RESULT_COMPLETE,.av_state=av};
    result.presentation=(h3_presentation){.version=9,.av_metadata_identity=1,.ref2va=s->ref2va,
        .render_width=s->render_width,.render_height=s->render_height,.width=s->params.width,.height=s->params.height,
        .trim_frames=trim?s->context_frames:0,.trim_samples=trim?s->layout.prefix.audio_prefix_t*800:0,
        .fps=24,.sample_rate=32000,.codec_version=s->params._arithmetic_recipe?2:1,
        .cuda_attention=s->params.cuda_attention,.attention_version=s->attention_version,.attention_plan=s->attention_plan,
        .adaptive_cache=s->params.adaptive_cache,.adaptive_version=s->adaptive_version,
        .adaptive_cache_threshold=h3_adaptive_threshold(&s->params),.adaptive_cache_max_hits=h3_adaptive_max_hits(&s->params),
        .adaptive_cache_warmup=s->params.adaptive_cache?h3_adaptive_warmup(s->params.adaptive_cache_warmup):0,
        .subblock_warmup=s->params.cuda_attention==4?h3_subblock_warmup(s->params.subblock_warmup):0,
        .subblock_sparsity=s->params.cuda_attention==4?s->params.subblock_sparsity:0};
    int ok=h3_result_save_av_state(&result,path,error,size);h3_av_state_free(av);return ok;
}
int main(int argc,char **argv) {
    if(argc<5||argc>19){fprintf(stderr,"usage: adaptive_latent MODEL OUTBASE MODE STOP [STATE|- [SPARSITY|- [WIDTH HEIGHT FRAMES [QUANT CACHE [ADAPTIVE_WARMUP SUBBLOCK_WARMUP [MAX_MIB|- [REFERENCE|- REFERENCE2|- SIZE [REFERENCE3|-]]]]]]]]\n");return 2;}
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=256;p.frames=22;p.steps=50;
    if(getenv("H3_TEST_SCHEDULE_STEPS"))p.steps=atoi(getenv("H3_TEST_SCHEDULE_STEPS"));
    if(p.steps<1||p.steps>50)return 2;
    if(getenv("H3_TEST_ADAPTIVE_THRESHOLD")){
        if(!h3_adaptive_threshold_parse(getenv("H3_TEST_ADAPTIVE_THRESHOLD"),&p.adaptive_cache_threshold))return 2;
        p.adaptive_cache_threshold_set=1;
    }
    if(getenv("H3_TEST_ADAPTIVE_MAX_HITS")){
        if(!h3_adaptive_max_hits_parse(getenv("H3_TEST_ADAPTIVE_MAX_HITS"),&p.adaptive_cache_max_hits))return 2;
        p.adaptive_cache_max_hits_set=1;
    }
    p.stop_after_step=atoi(argv[4]);
    if(!h3_adaptive_parse(argv[3],&p.adaptive_cache)||p.stop_after_step<0||p.stop_after_step>p.steps)return 2;
    p.adaptive_cache_set=1;
    char path[4096];if(snprintf(path,sizeof(path),"%s.h3sample",argv[2])>=(int)sizeof(path))return 2;
    if(argc>=6&&strcmp(argv[5],"-"))p.resume_sampler_state=argv[5];
    if(argc>=7&&strcmp(argv[6],"-")){p.cuda_attention=4;p.cuda_attention_set=1;p.subblock_sparsity=strtof(argv[6],NULL);p.subblock_sparsity_set=1;}
    if(argc>=10){p.width=atoi(argv[7]);p.height=atoi(argv[8]);p.frames=atoi(argv[9]);}
    else if(argc>7)return 2;
    if(argc>=12){if(!h3_quant_parse(argv[10],&p.cuda_denoise_quant))return 2;
        p.cuda_denoise_quant_set=1;if(strcmp(argv[11],"-"))p.cuda_denoise_quant_cache=argv[11];}
    if(argc>=14){p.adaptive_cache_warmup=atoi(argv[12]);p.subblock_warmup=atoi(argv[13]);}
    if(argc>=15&&strcmp(argv[14],"-")) {
        if(!h3_adaptive_mib_parse(argv[14],&p.adaptive_cache_max_bytes))return 2;
        p.adaptive_cache_max_bytes_set=1;
    }
    h3_reference refs[12]={0};char media_paths[12][4096];
    const char *media_args[12]={0};char file_paths[12][4096];int media_count=0;
    if(!p.resume_sampler_state&&getenv("H3_TEST_REFERENCES")){
        FILE *f=fopen(getenv("H3_TEST_REFERENCES"),"r");if(!f)return 2;
        while(media_count<12&&fgets(file_paths[media_count],sizeof(file_paths[0]),f)){
            size_t len=strlen(file_paths[media_count]);
            if(!len||file_paths[media_count][len-1]!='\n'){fclose(f);return 2;}
            file_paths[media_count][len-1]=0;media_args[media_count]=file_paths[media_count];media_count++;
        }
        int valid=feof(f)||fgetc(f)==EOF;fclose(f);if(!valid||!media_count)return 2;
    }
    for(int i=15;i<argc;i++)if(i!=17&&strcmp(argv[i],"-")){
        if(media_count>=12)return 2;media_args[media_count++]=argv[i];
    }
    if(!p.resume_sampler_state)for(int i=0;i<media_count;i++) {
        if(strlen(media_args[i])>=sizeof(media_paths[0]))return 2;
        strcpy(media_paths[p.reference_count],media_args[i]);
        char *media=media_paths[p.reference_count],*soundtrack=NULL;h3_reference_kind kind=H3_REFERENCE_IMAGE;int embedded=0;
        if(!strncmp(media,"video:",6)){kind=H3_REFERENCE_VIDEO;media+=6;embedded=1;}
        else if(!strncmp(media,"silent:",7)){kind=H3_REFERENCE_VIDEO;media+=7;}
        else if(!strncmp(media,"audio:",6)){kind=H3_REFERENCE_AUDIO;media+=6;}
        else if(!strncmp(media,"paired:",7)){kind=H3_REFERENCE_VIDEO_AUDIO;media+=7;soundtrack=strchr(media,'|');if(!soundtrack)return 2;*soundtrack++=0;}
        refs[p.reference_count++]=(h3_reference){.kind=kind,.path=media,.audio_path=soundtrack,.include_embedded_audio=embedded};
    }
    p.references=refs;
    h3_av_state *continuation=NULL;
    if(!p.resume_sampler_state&&getenv("H3_TEST_CONTINUE_FROM")) {
        char error[512];continuation=h3_av_state_load(getenv("H3_TEST_CONTINUE_FROM"),error,sizeof(error));
        if(!continuation){fprintf(stderr,"%s\n",error);return 1;}
        p.continuation=continuation;
        const char *mode=getenv("H3_TEST_CONTINUE_MODE");
        if(mode&&strcmp(mode,"hard")&&strcmp(mode,"bridge"))return 2;
        p.continuation_mode=mode&&!strcmp(mode,"bridge")?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;
        if(getenv("H3_TEST_CONTINUE_CONTEXT"))p.continuation_context_frames=atoi(getenv("H3_TEST_CONTINUE_CONTEXT"));
        if(getenv("H3_TEST_BRIDGE_STRENGTH"))p.bridge_max_strength=strtof(getenv("H3_TEST_BRIDGE_STRENGTH"),NULL);
        if(getenv("H3_TEST_BRIDGE_PROFILE"))p.bridge_profile=(h3_bridge_profile_type)atoi(getenv("H3_TEST_BRIDGE_PROFILE"));
        p.keep_continuation_prefix=getenv("H3_TEST_KEEP_PREFIX")!=NULL;
    }
    if(argc>=18){if(!strcmp(argv[17],"max"))p.reference_image_size=H3_REFERENCE_IMAGE_MAX;
        else if(!strcmp(argv[17],"high"))p.reference_image_size=H3_REFERENCE_IMAGE_HIGH;
        else if(strcmp(argv[17],"match"))return 2;}
    if(p.stop_after_step==p.steps){p.save_sampler_state=path;p.on_progress=stop_decode;}
    trace t={.base=argv[2],.steps=p.steps};p.on_latent_step=capture;p.callback_opaque=&t;
    if(getenv("H3_TEST_CANCEL_ONCE")){t.cancel_at=atoi(getenv("H3_TEST_CANCEL_ONCE"));p.on_progress=stop_decode;}
    h3_ctx *ctx=h3_load_dir(argv[1]);if(!ctx){fprintf(stderr,"%s\n",h3_last_error(NULL));return 1;}
    if(getenv("H3_TEST_CONTEXT_BUDGET")||getenv("H3_TEST_BAD_AUDIO"))h3_cache_set_enabled(ctx,1);
    if(getenv("H3_TEST_BAD_AUDIO")&&!p.resume_sampler_state){
        h3_reference bad_refs[13];memcpy(bad_refs,refs,p.reference_count*sizeof(*refs));
        bad_refs[p.reference_count]=(h3_reference){.kind=H3_REFERENCE_AUDIO,.path=getenv("H3_TEST_BAD_AUDIO")};
        h3_params bad=p;bad.references=bad_refs;bad.reference_count++;bad.stop_after_step=0;
        h3_result *failed=h3_generate(ctx,"A woman plays piano in warm sunlight with flowing piano music.",&bad);
        if(failed){h3_result_free(failed);h3_free(ctx);return 1;}
        fprintf(stderr,"expected invalid-audio recovery: %s\n",h3_last_error(ctx));
    }
    h3_result *r=h3_generate(ctx,p.resume_sampler_state?NULL:"A woman plays piano in warm sunlight with flowing piano music.",&p);
    if(t.cancel_at) {
        if(r||!t.cancelled){h3_result_free(r);h3_free(ctx);return 1;}
        r=h3_generate(ctx,p.resume_sampler_state?NULL:"A woman plays piano in warm sunlight with flowing piano music.",&p);
    }
    if(!r&&p.stop_after_step==p.steps) {
        /* Final transition is saved before the decoder callback cancels. */
        char error[512];h3_sampler_state *s=h3_sampler_state_load(path,error,sizeof(error));
        int ok=s&&s->next_step==p.steps;
        if(ok){h3_prepared_cache_free(&s->prepared);ok=h3_sampler_state_save(s,path,error,sizeof(error))&&save_complete_av(s,error,sizeof(error));}
        if(!ok)fprintf(stderr,"final diagnostic save: %s\n",error);
        h3_sampler_state_free(s);h3_free(ctx);h3_av_state_free(continuation);return !ok;
    }
    if(!r){fprintf(stderr,"%s\n",h3_last_error(ctx));h3_free(ctx);h3_av_state_free(continuation);return 1;}
    int ok=r->status==H3_RESULT_PAUSED&&r->completed_steps==p.stop_after_step&&r->frames==0;
    if(ok) {
        /* Rebuildable prepared tensors are ~1 GiB for 50 steps. Test compact
         * checkpoints, retaining the complete required mutable state. */
        h3_prepared_cache_free(&r->sampler_state->prepared);
        char error[512];ok=h3_sampler_state_save(r->sampler_state,path,error,sizeof(error));
        if(!ok)fprintf(stderr,"%s\n",error);
    }
    if(ok&&getenv("H3_TEST_CONTEXT_BUDGET")) {
        uint64_t too_small=r->sampler_state->full_sequence*UINT64_C(5376)*6+6168-1;
        h3_result_free(r);r=NULL;
        p.adaptive_cache_max_bytes=too_small;p.adaptive_cache_max_bytes_set=1;
        r=h3_generate(ctx,"A woman plays piano in warm sunlight with flowing piano music.",&p);
        if(r||!strstr(h3_last_error(ctx),"minimum --adaptive-cache-max-mib"))ok=0;
        h3_result_free(r);r=NULL;
        p.adaptive_cache_max_bytes=UINT64_C(8192)*1048576;
        r=h3_generate(ctx,"A woman plays piano in warm sunlight with flowing piano music.",&p);
        if(!r||r->status!=H3_RESULT_PAUSED)ok=0;
        fprintf(stderr,"context budget rejection/recovery: %s\n",ok?"PASS":"FAIL");
    }
    h3_result_free(r);h3_free(ctx);h3_av_state_free(continuation);return !ok;
}
