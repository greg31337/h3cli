/* Real-checkpoint acceptance driver: verifies initial RNG/prefix contents and
 * every Euler transition, callback delivery, full-state capture and disk resume.
 * Outputs raw RGB/PCM for seam/color measurements alongside the MP4 and state. */
#include "src/sampling/av_state.h"
#include "src/vae/audio_vae.h"
#include "src/denoise/dit.h"
#include "src/sglang/sglang.h"
#include "src/denoise/adaptive_cache.h"
#include <inttypes.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#define REQUIRE(x) do { if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
typedef struct {
    h3_av_state *expected;
    h3_av_state *final;
    h3_denoise_prefix prefix;
    h3_bridge_profile bridge;
    int bridge_mode;
    int steps_seen, frames_seen, delivered;
    FILE *rgb;
    FILE *trace;
    FILE *oracle_trace;
    float *oracle_values;
    double started;
} audit;

/* One prompt/source/seed per process keeps the exact conditioning fixed while
 * varying sampler and bridge settings. The manifest is test-only TSV. */
typedef struct {
    char base[4096];
    h3_continuation_mode mode;
    h3_bridge_profile_type profile;
    int bridge_steps, reuse, core, gpu, keep, window, callbacks;
    float strength;
} variant;

static int read_variants(const char *path, variant **out) {
    if (!path) return 0;
    FILE *file=fopen(path,"r"); REQUIRE(file);
    variant *items=calloc(256,sizeof(*items)); REQUIRE(items);
    char line[4608]; int count=0;
    while (fgets(line,sizeof(line),file)) {
        REQUIRE(count<256 && strchr(line,'\n'));
        char *fields[9], *save=NULL;
        for (int i=0;i<9;i++) { fields[i]=strtok_r(i?NULL:line,"\t\r\n",&save); REQUIRE(fields[i]); }
        variant *v=&items[count++];
        char *window=strtok_r(NULL,"\t\r\n",&save);
        char *callbacks=window?strtok_r(NULL,"\t\r\n",&save):NULL;
        REQUIRE(!window || callbacks);
        REQUIRE(!strtok_r(NULL,"\t\r\n",&save));
        v->window=window?atoi(window):1; v->callbacks=callbacks?atoi(callbacks):1;
        REQUIRE(v->window>=0 && v->window<=1000 && (v->callbacks==0 || v->callbacks==1));
        REQUIRE(strlen(fields[0])<sizeof(v->base)); strcpy(v->base,fields[0]);
        REQUIRE(!strcmp(fields[1],"hard") || !strcmp(fields[1],"bridge"));
        v->mode=!strcmp(fields[1],"bridge")?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;
        REQUIRE(!strcmp(fields[2],"stepped") || !strcmp(fields[2],"linear") || !strcmp(fields[2],"ease-out"));
        v->profile=!strcmp(fields[2],"linear")?H3_BRIDGE_LINEAR:!strcmp(fields[2],"ease-out")?H3_BRIDGE_EASE_OUT:H3_BRIDGE_STEPPED;
        v->bridge_steps=atoi(fields[3]); v->strength=strtof(fields[4],NULL);
        v->reuse=atoi(fields[5]); v->core=atoi(fields[6]); v->gpu=atoi(fields[7]); v->keep=atoi(fields[8]);
        REQUIRE((v->gpu==0 || v->gpu==1) && (v->keep==0 || v->keep==1));
    }
    REQUIRE(!ferror(file) && !fclose(file) && count>0);
    *out=items; return count;
}

static int latent_step(int completed,int total,const float *video,size_t nv,
                        const float *audio,size_t na,void *opaque) {
    audit *a=opaque;
    REQUIRE(completed==a->steps_seen++);
    REQUIRE(nv==a->expected->info.video_elements && na==a->expected->info.audio_elements);
    if (a->trace) {
        REQUIRE(fwrite(video,4,nv,a->trace)==nv);
        REQUIRE(fwrite(audio,4,na,a->trace)==na);
    }
    if (a->oracle_trace) {
        REQUIRE(fread(a->oracle_values,4,nv+na,a->oracle_trace)==nv+na);
        if (memcmp(video,a->oracle_values,nv*4) || memcmp(audio,a->oracle_values+nv,na*4)) {
            fprintf(stderr,"GPU/CPU trajectory differs at step %d\n",completed);
            return 1;
        }
    }
    if (!completed) {
        REQUIRE(!memcmp(video,a->expected->video,nv*4));
        REQUIRE(!memcmp(audio,a->expected->audio,na*4));
    } else if (a->bridge_mode) {
        REQUIRE(h3_bridge_check_exact(&a->bridge,a->expected->info.video_t,
            a->expected->info.latent_h,a->expected->info.latent_w,a->expected->info.audio_t,
            video,audio,a->expected->video,a->expected->audio));
    } else {
        size_t hw=(size_t)a->expected->info.latent_h*(size_t)a->expected->info.latent_w;
        for (size_t c=0;c<24;c++) {
            size_t offset=c*(size_t)a->expected->info.video_t*hw;
            REQUIRE(!memcmp(video+offset,a->expected->video+offset,(size_t)a->prefix.video_prefix_t*hw*4));
        }
        for (size_t c=0;c<64;c++) {
            size_t offset=c*(size_t)a->expected->info.audio_t;
            REQUIRE(!memcmp(audio+offset,a->expected->audio+offset,(size_t)a->prefix.audio_prefix_t*4));
        }
    }
    if (completed==total) {
        for (size_t i=0;i<nv;i++) REQUIRE(isfinite(video[i]));
        for (size_t i=0;i<na;i++) REQUIRE(isfinite(audio[i]));
        memcpy(a->final->video,video,nv*4); memcpy(a->final->audio,audio,na*4);
    }
    fprintf(stderr,"h3 test: checked boundary %d/%d at %.3f s\n",
        completed,total,h3_av_now()-a->started);
    return 0;
}

static int frame(const h3_frame *f,void *opaque) {
    audit *a=opaque;
    REQUIRE(f->denoise_step==-1 && f->frame_index==a->frames_seen++ && f->frame_count==a->delivered);
    for (int row=0;row<f->height;row++) REQUIRE(fwrite(f->rgb+(size_t)row*(size_t)f->stride,
        1,(size_t)f->width*3,a->rgb)==(size_t)f->width*3);
    return 0;
}

int main(int argc,char **argv) {
#ifndef __APPLE__
    h3_sglang_exchange(H3_SGLANG_VERSION);
#endif
    if (argc<11) {
        fprintf(stderr,"usage: %s MODEL OUTBASE MODE STATE|- SEED FRAMES STEPS REUSE KEEP PROMPT [VIDEO] [CONTEXT] [WIDTH] [hard|bridge] [PROFILE] [BRIDGE_STEPS] [MAX_STRENGTH]\n",argv[0]); return 2;
    }
    const char *mode=argv[3];
    const char *video=argc>11 ? argv[11] : "outputs/continuation-validation/reference.mp4";
    h3_params p=H3_PARAMS_DEFAULT;
    p.seed=strtoull(argv[5],NULL,10); p.frames=atoi(argv[6]); p.steps=atoi(argv[7]);
    p.denoise_reuse=atoi(argv[8]); p.keep_continuation_prefix=atoi(argv[9]);
    p.continuation_context_frames=argc>12 ? atoi(argv[12]) : 39;
    p.width=p.height=argc>13 ? atoi(argv[13]) : 256;
    if (getenv("H3_TEST_HEIGHT")) p.height=atoi(getenv("H3_TEST_HEIGHT"));
    if (argc>14) {
        REQUIRE(!strcmp(argv[14],"hard") || !strcmp(argv[14],"bridge"));
        p.continuation_mode=!strcmp(argv[14],"bridge")?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;
    }
    if (argc>15) {
        REQUIRE(!strcmp(argv[15],"stepped") || !strcmp(argv[15],"linear") || !strcmp(argv[15],"ease-out"));
        p.bridge_profile=!strcmp(argv[15],"linear")?H3_BRIDGE_LINEAR:!strcmp(argv[15],"ease-out")?H3_BRIDGE_EASE_OUT:H3_BRIDGE_STEPPED;
    }
    if (argc>16) p.bridge_video_steps=atoi(argv[16]);
    if (argc>17) p.bridge_max_strength=strtof(argv[17],NULL);
    if(getenv("H3_TEST_ADAPTIVE_MODE")) {
        REQUIRE(h3_adaptive_parse(getenv("H3_TEST_ADAPTIVE_MODE"),&p.adaptive_cache));
        p.adaptive_cache_warmup=2;p.adaptive_cache_threshold=1;p.adaptive_cache_threshold_set=1;
        p.adaptive_cache_max_hits=2;p.adaptive_cache_max_hits_set=1;
    }
    if(getenv("H3_TEST_SUBBLOCK")) {
        p.cuda_attention=4;p.subblock_sparsity=.75f;p.subblock_warmup=3;
    }
    char error[512];
    h3_av_state *source=NULL;
    if (strcmp(argv[4],"-")) { source=h3_av_state_load(argv[4],error,sizeof(error)); REQUIRE(source); }
    h3_av_state *source_copy=source?h3_av_state_clone(source):NULL; REQUIRE(!source || source_copy);
    p.continuation=source;
    h3_reference refs[4]; memset(refs,0,sizeof(refs));
    if(!strcmp(mode,"reference")) {
        REQUIRE(getenv("H3_TEST_IMAGE"));
        refs[p.reference_count++]=(h3_reference){H3_REFERENCE_IMAGE,getenv("H3_TEST_IMAGE"),NULL,0};
        p.reference_image_size=H3_REFERENCE_IMAGE_MATCH;
    }
    if (!strcmp(mode,"image1") || !strcmp(mode,"image2") || !strcmp(mode,"recursive")) {
        int second=!strcmp(mode,"image2");
        refs[p.reference_count++]=(h3_reference){H3_REFERENCE_IMAGE,second?"inputs/face2.jpg":"inputs/face1.jpg",NULL,0};
        refs[p.reference_count++]=(h3_reference){H3_REFERENCE_IMAGE,second?"inputs/body2.jpg":"inputs/body1.jpg",NULL,0};
    }
    if (!strcmp(mode,"imageaudio") || !strcmp(mode,"imagevideo") || !strcmp(mode,"audio"))
        refs[p.reference_count++]=(h3_reference){H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0};
    if (!strcmp(mode,"imageaudio") || !strcmp(mode,"audio"))
        refs[p.reference_count++]=(h3_reference){H3_REFERENCE_AUDIO,"outputs/continuation-validation/reference.wav",NULL,0};
    if (!strcmp(mode,"imagevideo") || !strcmp(mode,"recursive"))
        refs[p.reference_count++]=(h3_reference){H3_REFERENCE_VIDEO,video,NULL,!strcmp(mode,"recursive")};
    if (!strcmp(mode,"video")) refs[p.reference_count++]=(h3_reference){H3_REFERENCE_VIDEO,video,NULL,1};
    if (!strcmp(mode,"replace")) refs[p.reference_count++]=(h3_reference){H3_REFERENCE_VIDEO_AUDIO,video,"outputs/continuation-validation/replacement.wav",0};
    if (!strcmp(mode,"fl2va")) p.first_frame="inputs/face1.jpg";
    p.references=refs;
    const char *paired=getenv("H3_TEST_KEEP_PAIR");
    const char *zero_pair=getenv("H3_TEST_ZERO_BRIDGE_PAIR");
    variant *variants=NULL;
    int variant_count=read_variants(getenv("H3_TEST_VARIANTS"),&variants);
    REQUIRE(!variant_count || (!paired && !zero_pair && source));
    REQUIRE(!paired || !zero_pair);
    if (zero_pair) { REQUIRE(p.continuation_mode==H3_CONTINUE_HARD); paired=zero_pair; }
    h3_ctx *ctx=h3_load_dir(argv[1]);
    if (!ctx) { fprintf(stderr,"%s\n",h3_last_error(NULL)); return 1; }
    if (paired || variant_count) { REQUIRE(source && !p.keep_continuation_prefix); h3_cache_set_enabled(ctx,1); }
    h3_av_state *pair_state=NULL;
    for (int pass=0;pass<(variant_count?variant_count:paired?2:1);pass++) {
        const char *base=pass ? paired : argv[2];
        if (variant_count) {
            const variant *v=&variants[pass]; base=v->base;
            p.continuation_mode=v->mode; p.bridge_profile=v->profile;
            p.bridge_video_steps=v->bridge_steps; p.bridge_max_strength=v->strength;
            p.denoise_reuse=v->reuse; p.core_reuse=v->core; p.keep_continuation_prefix=v->keep;
            REQUIRE(!setenv("H3_GPU_SAMPLER",v->gpu?"1":"0",1));
            REQUIRE(!setenv("H3_CPU_SAMPLER",v->gpu?"0":"1",1));
            char window[32]; snprintf(window,sizeof(window),"%d",v->window);
            REQUIRE(!setenv("H3_GPU_SAMPLER_WINDOW",window,1));
        } else if (pass) {
            if (zero_pair) { p.continuation_mode=H3_CONTINUE_BRIDGE; p.bridge_max_strength=0; }
            else p.keep_continuation_prefix=1;
        }
        char mp4[4096],state[4096],rgb[4096],pcm_path[4096],json[4096];
        REQUIRE(snprintf(mp4,sizeof(mp4),"%s.mp4",base)<(int)sizeof(mp4));
        REQUIRE(snprintf(state,sizeof(state),"%s.h3av",base)<(int)sizeof(state));
        REQUIRE(snprintf(rgb,sizeof(rgb),"%s.rgb",base)<(int)sizeof(rgb));
        REQUIRE(snprintf(pcm_path,sizeof(pcm_path),"%s.pcm",base)<(int)sizeof(pcm_path));
        REQUIRE(snprintf(json,sizeof(json),"%s.json",base)<(int)sizeof(json));
        audit a={0}; uint8_t sig[32]={0};
        int callbacks=!variant_count || variants[pass].callbacks;
        int oracle=-1;
        if (variant_count && variants[pass].gpu && getenv("H3_TEST_LATENT_TRACE")) {
            const variant *v=&variants[pass];
            for (int i=0;i<pass;i++) {
                const variant *c=&variants[i];
                if (!c->gpu && c->mode==v->mode && c->profile==v->profile &&
                    c->bridge_steps==v->bridge_steps && c->strength==v->strength &&
                    c->reuse==v->reuse && c->core==v->core && c->keep==v->keep) oracle=i;
            }
            REQUIRE(oracle>=0);
        }
        if (getenv("H3_TEST_LATENT_TRACE") && callbacks) {
            char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s.trace",base)<(int)sizeof(path));
            a.trace=fopen(path,"wb"); REQUIRE(a.trace);
        }
        a.expected=h3_av_state_new(p.width,p.height,h3_align_frame_count(p.frames),p.seed,sig);
        REQUIRE(a.expected);
        if (oracle>=0 && callbacks) {
            char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s.trace",variants[oracle].base)<(int)sizeof(path));
            a.oracle_trace=fopen(path,"rb"); REQUIRE(a.oracle_trace);
            a.oracle_values=malloc((a.expected->info.video_elements+a.expected->info.audio_elements)*4);
            REQUIRE(a.oracle_values);
        }
        if (h3_sglang_requested()) {
            size_t count=a.expected->info.audio_elements;
            float *rows=malloc(count*sizeof(*rows)); REQUIRE(rows);
            REQUIRE(h3_sglang_normal(p.seed,a.expected->video,a.expected->info.video_elements));
            REQUIRE(h3_sglang_normal(p.seed,rows,count));
            REQUIRE(h3_dit_unpack_audio(rows,32,a.expected->info.audio_t,a.expected->audio,count));
            free(rows);
        } else {
            h3_rng rng;
            h3_rng_seed(&rng,p.seed); h3_rng_fill_normal(&rng,a.expected->video,a.expected->info.video_elements);
            h3_rng_seed(&rng,p.seed); h3_rng_fill_normal(&rng,a.expected->audio,a.expected->info.audio_elements);
        }
        if (source) {
            REQUIRE(h3_continuation_context(p.continuation_context_frames,&a.prefix));
            a.bridge_mode=p.continuation_mode==H3_CONTINUE_BRIDGE;
            if (a.bridge_mode) {
                REQUIRE(h3_bridge_profile_build(p.continuation_context_frames,p.bridge_video_steps,
                    p.bridge_max_strength,p.bridge_profile,&a.bridge,NULL,0,error,sizeof(error)));
                REQUIRE(h3_av_state_insert_bridge(source,&a.expected->info,&a.bridge,1,1,a.expected->video,a.expected->audio));
            } else REQUIRE(h3_av_state_insert_prefix(source,&a.expected->info,a.prefix,a.expected->video,a.expected->audio,1));
        }
        a.final=h3_av_state_clone(a.expected); REQUIRE(a.final);
        a.delivered=a.expected->info.frames-(source && !p.keep_continuation_prefix?p.continuation_context_frames:0);
        a.rgb=fopen(rgb,"wb"); REQUIRE(a.rgb);
        p.on_frame=frame; p.on_latent_step=callbacks?latent_step:NULL; p.callback_opaque=&a; p.output_path=mp4;
        double start=h3_av_now();
        a.started=start;
        h3_result *result=h3_generate(ctx,argv[10],&p);
        if (!result) { fprintf(stderr,"%s\n",h3_last_error(ctx)); return 1; }
        double seconds=h3_av_now()-start;
        REQUIRE(!fclose(a.rgb));
        if (a.trace) REQUIRE(!fclose(a.trace));
        if (a.oracle_trace) { REQUIRE(fgetc(a.oracle_trace)==EOF); REQUIRE(!fclose(a.oracle_trace)); }
        free(a.oracle_values);
        REQUIRE(result->frames==a.delivered && a.frames_seen==a.delivered && a.steps_seen==(callbacks?p.steps+1:0));
        int trim_ticks=source && !p.keep_continuation_prefix?a.prefix.audio_prefix_t:0;
        REQUIRE(result->audio_samples==(a.expected->info.audio_t-trim_ticks)*800);
        const h3_av_state *saved=h3_result_av_state(result);
        REQUIRE(saved && saved->info.frames==a.expected->info.frames);
        if (oracle>=0) {
            char path[4096]; REQUIRE(snprintf(path,sizeof(path),"%s.h3av",variants[oracle].base)<(int)sizeof(path));
            h3_av_state *cpu=h3_av_state_load(path,error,sizeof(error)); REQUIRE(cpu);
            REQUIRE(cpu->info.video_elements==saved->info.video_elements && cpu->info.audio_elements==saved->info.audio_elements);
            REQUIRE(!memcmp(cpu->video,saved->video,saved->info.video_elements*4));
            REQUIRE(!memcmp(cpu->audio,saved->audio,saved->info.audio_elements*4));
            h3_av_state_free(cpu);
            fprintf(stderr,"h3 test: CPU/GPU parity passed before advancing to the next variant\n");
        }
        if (callbacks) {
            REQUIRE(!memcmp(saved->video,a.final->video,saved->info.video_elements*4));
            REQUIRE(!memcmp(saved->audio,a.final->audio,saved->info.audio_elements*4));
        } else {
            REQUIRE(a.bridge_mode);
            REQUIRE(h3_bridge_check_exact(&a.bridge,saved->info.video_t,saved->info.latent_h,
                saved->info.latent_w,saved->info.audio_t,saved->video,saved->audio,a.expected->video,a.expected->audio));
            for (size_t i=0;i<saved->info.video_elements;i++) REQUIRE(isfinite(saved->video[i]));
            for (size_t i=0;i<saved->info.audio_elements;i++) REQUIRE(isfinite(saved->audio[i]));
        }
        REQUIRE(h3_result_save_av_state(result,state,error,sizeof(error)));
        h3_av_state *loaded=h3_av_state_load(state,error,sizeof(error)); REQUIRE(loaded);
        REQUIRE(!memcmp(loaded->video,saved->video,saved->info.video_elements*4));
        REQUIRE(!memcmp(loaded->audio,saved->audio,saved->info.audio_elements*4));
        h3_audio_waveform waveform={0}; char vae[4096];
        snprintf(vae,sizeof(vae),"%s/%s/audio_vae",argv[1],p.reference_count?"Ref2VA":"FL2VA");
        REQUIRE(h3_audio_vae_decode(vae,"src/metal/shaders.metal",saved->audio,saved->info.audio_t,NULL,NULL,&waveform,error,sizeof(error)));
        FILE *pcm=fopen(pcm_path,"wb"); REQUIRE(pcm);
        for (int c=0;c<waveform.channels;c++) REQUIRE(fwrite(waveform.pcm+(size_t)c*(size_t)waveform.samples+(size_t)trim_ticks*800,
            4,(size_t)result->audio_samples,pcm)==(size_t)result->audio_samples);
        REQUIRE(!fclose(pcm));
        FILE *out=fopen(json,"w"); REQUIRE(out);
        const char *gpu_request=getenv("H3_GPU_SAMPLER");
        int gpu_requested=variant_count?variants[pass].gpu:
            (gpu_request && !strcmp(gpu_request,"1"));
        fprintf(out,"{\"raw_frames\":%d,\"frames\":%d,\"audio_samples\":%d,\"width\":%d,\"height\":%d,\"steps\":%d,\"reuse\":%d,\"latent_callbacks\":%d,\"references\":%zu,\"seconds\":%.6f,\"continuation_mode\":\"%s\",\"bridge_profile\":\"%s\",\"bridge_steps\":%d,\"bridge_max_strength\":%.9g,\"core_reuse\":%d,\"gpu_requested\":%d,\"seed\":%" PRIu64 "}\n",
            saved->info.frames,result->frames,result->audio_samples,result->width,result->height,p.steps,p.denoise_reuse,a.steps_seen,p.reference_count,seconds,
            a.bridge_mode?"bridge":"hard",h3_bridge_profile_name(p.bridge_profile),p.bridge_video_steps,(double)p.bridge_max_strength,
            p.core_reuse,gpu_requested,p.seed);
        REQUIRE(!fclose(out));
        printf("ok: %s, %d transitions, %d frames, %d samples, %.2f s\n",base,p.steps,result->frames,result->audio_samples,seconds); fflush(stdout);
        if (paired) {
            h3_cache_info cache; h3_cache_get_info(ctx,&cache);
            REQUIRE(cache.embedding_entries && cache.prepared_dit && cache.video_decoder);
            if (!pass) { pair_state=h3_av_state_clone(saved); REQUIRE(pair_state); }
            else {
                REQUIRE(!memcmp(pair_state->video,saved->video,saved->info.video_elements*4));
                REQUIRE(!memcmp(pair_state->audio,saved->audio,saved->info.audio_elements*4));
                puts(zero_pair ? "ok: cached conditioning, hard versus zero-strength bridge final-state parity" :
                    "ok: cached conditioning/DiT reuse and keep-prefix final-state parity");
            }
        }
        h3_audio_waveform_free(&waveform); h3_av_state_free(loaded); h3_result_free(result);
        h3_av_state_free(a.expected); h3_av_state_free(a.final);
    }
    if (source) {
        REQUIRE(!memcmp(source->video,source_copy->video,source->info.video_elements*4));
        REQUIRE(!memcmp(source->audio,source_copy->audio,source->info.audio_elements*4));
    }
    free(variants);
    h3_av_state_free(source_copy); h3_av_state_free(pair_state); h3_av_state_free(source); h3_free(ctx);
    return 0;
}
