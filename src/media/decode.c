#include "src/media/delivery.h"
#include "src/sampling/av_state.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include "src/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

typedef struct {const h3_decode_options *options;const char *phase;int cancelled;} decode_progress;
static int progress(int done,int total,void *opaque) {
    decode_progress *p=opaque;if(p->cancelled)return 1;
    if(p->options->on_progress)p->cancelled=p->options->on_progress(p->phase,done,total,p->options->callback_opaque)!=0;
    return p->cancelled;
}
static int same_file(const char *a,const char *b) {
    if (!a || !b)
        return 0;
    if (!strcmp(a, b))
        return 1;
    struct stat x,y;return !stat(a,&x)&&!stat(b,&y)&&x.st_dev==y.st_dev&&x.st_ino==y.st_ino;
}
h3_result *h3_decode_av_state(const char *model_dir,const char *state_path,
 const h3_decode_options *options,char *error,size_t size) {
    char local[512];if(!error||!size){error=local;size=sizeof(local);}error[0]=0;
    if(!model_dir||!*model_dir||!state_path||!*state_path||!options){snprintf(error,size,"invalid decode-only arguments");return NULL;}
    h3_decode_options resolved=*options;
    options=&resolved;
    if(!h3_output_encoding_valid(&options->output_encoding,error,size))return NULL;
    if(!h3_preview_vae_options(options->preview_vae,options->preview_vae_model,error,size))return NULL;
    size_t cap=strlen(state_path)+20;char *sidecar=malloc(cap);if(!sidecar){snprintf(error,size,"cannot allocate state path");return NULL;}
    snprintf(sidecar,cap,"%s.presentation",state_path);
    int collision=same_file(options->output_path,state_path)||same_file(options->output_path,sidecar)||
        same_file(options->output_path,h3_preview_vae_path(options->preview_vae_model));free(sidecar);
    if(collision){snprintf(error,size,"output must not overwrite a source state, presentation or tiny model");return NULL;}
    h3_av_state *state=h3_av_state_load(state_path,error,size);if(!state)return NULL;
    h3_presentation p={0};int found=h3_presentation_load(state_path,state,&p,error,size);
    if(found!=1){h3_av_state_free(state);return NULL;}
    return h3_decode_av_owned(model_dir,state,&p,options,error,size);
}
h3_result *h3_decode_av_owned(const char *model_dir,h3_av_state *state,
    const h3_presentation *presentation,const h3_decode_options *requested,char *error,size_t size) {
    char local[512];if(!error||!size){error=local;size=sizeof(local);}*error=0;
    if(!model_dir||!state||!requested){h3_av_state_free(state);snprintf(error,size,"invalid owned AV decode");return NULL;}
    h3_decode_options resolved=*requested;const h3_decode_options *options=&resolved;
    if(!h3_output_encoding_valid(&options->output_encoding,error,size)){h3_av_state_free(state);return NULL;}
    h3_result *result=NULL;h3_tiny_vae *tiny=NULL;h3_video_vae_decoder *full=NULL;
    h3_audio_waveform audio={0};char *audio_path=NULL,*video_path=NULL;size_t cap=0;
    int previous_reference=h3_sglang_exchange(0);
    h3_cuda_policy previous_policy=h3_cuda_policy_current();
    if(!presentation){snprintf(error,size,"current AV state requires presentation metadata");goto done;}
    h3_presentation p=*presentation;
    if(!h3_presentation_validate(&p,&state->info,error,size))goto done;
#ifdef __APPLE__
    int recipe=0;
    if(p.codec_version!=1){snprintf(error,size,"saved decode recipe requires a CUDA build");goto done;}
#else
    int recipe=H3_SGLANG_VERSION;
    if(p.codec_version!=2){snprintf(error,size,"saved decode recipe requires a Metal build");goto done;}
#endif
    if(recipe) {
        h3_params request=H3_PARAMS_DEFAULT;
        request.preview_vae=options->preview_vae;
        h3_cuda_policy policy;
        if(!h3_cuda_policy_resolve(&request,"cuda",0,&policy,error,size)||
           !h3_cuda_policy_preflight(&policy,error,size))goto done;
        h3_cuda_policy_exchange(policy);
        if(policy.preview)
            fprintf(stderr,"h3cli: preview decoding outside full-output SGLang parity; saved denoising is unchanged\n");
    }
    h3_sglang_exchange(recipe);
    if(!h3_tiny_vae_frames(state->info.video_t)){snprintf(error,size,"decode-only requires 22..362 frames (5n+2 latent frames, n >= 1)");goto done;}
    for(size_t i=0;i<state->info.video_elements;i++)if(!isfinite(state->video[i])){snprintf(error,size,"non-finite AV video state");goto done;}
    for(size_t i=0;i<state->info.audio_elements;i++)if(!isfinite(state->audio[i])){snprintf(error,size,"non-finite AV audio state");goto done;}
    decode_progress pr={options,"decode compatibility",0};if(progress(0,1,&pr)){snprintf(error,size,"decode-only cancelled");goto done;}
    uint8_t digest[32];char why[512];
    int matched=h3_av_state_metadata_signature(model_dir,p.ref2va,digest,why,sizeof(why))&&
        !memcmp(digest,state->info.compatibility,32);
    if(!matched){snprintf(error,size,"AV state is incompatible with the supplied model's decoder/geometry contract");goto done;}
    if(progress(1,1,&pr)){snprintf(error,size,"decode-only cancelled");goto done;}
    cap=strlen(model_dir)+64;audio_path=malloc(cap);video_path=malloc(cap);
    if(!audio_path||!video_path){snprintf(error,size,"cannot allocate decoder paths");goto done;}
    snprintf(audio_path,cap,"%s/%s/audio_vae",model_dir,p.ref2va?"Ref2VA":"FL2VA");
    snprintf(video_path,cap,"%s/%s/video_vae/source",model_dir,p.ref2va?"Ref2VA":"FL2VA");
    double begin=h3_av_now(),audio_begin=begin;pr.phase="audio VAE";
    if(!h3_audio_vae_decode(audio_path,"src/metal/shaders.metal",state->audio,state->info.audio_t,progress,&pr,&audio,error,size))goto done;
    double audio_seconds=h3_av_now()-audio_begin,load_begin=h3_av_now();
    if(options->preview_vae){pr.phase="tiny VAE load";if(progress(0,1,&pr))goto cancelled;
        tiny=h3_tiny_vae_load(h3_preview_vae_path(options->preview_vae_model),error,size);if(!tiny)goto done;
        memcpy(p.tiny_sha256,h3_tiny_vae_digest(tiny),65);if(progress(1,1,&pr))goto cancelled;
    }else{p.tiny_sha256[0]=0;pr.phase="video VAE load";
        full=h3_video_vae_decoder_load(video_path,"src/metal/shaders.metal",state->info.latent_h,state->info.latent_w,progress,&pr,error,size);if(!full)goto done;}
    if(getenv("H3_PROFILE"))fprintf(stderr,"h3cli: decode-only audio=%.6f decoder_cold_load=%.6f s\n",audio_seconds,h3_av_now()-load_begin);
    if(!h3_deliver_video(tiny,full,state->video,state->info.video_t,state->info.latent_h,state->info.latent_w,&p,&audio,options,error,size))goto done;
    result=calloc(1,sizeof(*result));if(!result){snprintf(error,size,"cannot allocate decode result");goto done;}
    result->width=p.width;result->height=p.height;result->frames=state->info.frames-p.trim_frames;
    result->fps=p.fps;result->sample_rate=audio.sample_rate;result->audio_samples=audio.samples;result->seed=state->info.seed;
    result->presentation=p;result->av_state=state;state=NULL;
    if(getenv("H3_PROFILE"))fprintf(stderr,"h3cli: decode-only through MP4 completion %.6f s; denoiser calls=0\n",h3_av_now()-begin);
    goto done;
cancelled:snprintf(error,size,"decoder loading cancelled");
done:h3_tiny_vae_free(tiny);h3_video_vae_decoder_free(full);h3_audio_waveform_free(&audio);h3_av_state_free(state);
    free(audio_path);free(video_path);h3_cuda_policy_exchange(previous_policy);h3_sglang_exchange(previous_reference);return result;
}

#include "src/vae/image_vae.h"
#include "src/media/ffmpeg.h"

h3_result *h3_decode_still_values(const h3_still_latent *input,const char *image_vae,
    const h3_decode_options *options,char *error,size_t size) {
    char local[512];if(!error || !size){error=local;size=sizeof(local);}*error=0;
    if(!input || !input->values || !image_vae || !options || options->preview_vae ||
        options->preview_vae_model) {
        snprintf(error,size,"direct still decode requires a latent and explicit image VAE, without video/preview options");return NULL;
    }
    const char *output=options->output_path;size_t length=output?strlen(output):0;
    if(output && (length<4 || strcmp(output+length-4,".png"))) {
        snprintf(error,size,"still output must use .png");return NULL;
    }
    if(same_file(output,image_vae)) {
        snprintf(error,size,"still output must not overwrite its latent or image VAE");return NULL;
    }
    h3_still_latent z=*input;h3_image_vae_decoder *decoder=NULL;h3_video_frames rgb={0};
    h3_result *result=NULL;uint8_t *pixels=NULL;decode_progress p={options,"image VAE load",0};
    double start=h3_av_now();

    if(!h3_still_geometry(z.height,z.width,error,size))goto done;
    h3_image_vae_info identity;
    if(!h3_memory_checkpoint(progress(0,36,&p),"image VAE inspect",error,size) ||
        !h3_image_vae_inspect(image_vae,&identity,error,size))goto done;
    if(memcmp(z.compatibility_sha256,identity.compatibility_sha256,65)) {
        snprintf(error,size,"still latent encoder/whitening compatibility differs from image VAE");goto done;
    }
    decoder=h3_image_vae_load(image_vae,"src/metal/shaders.metal",z.height,z.width,progress,&p,error,size);
    if(!decoder)goto done;
    if(strcmp(z.compatibility_sha256,h3_image_vae_info_get(decoder)->compatibility_sha256)) {
        snprintf(error,size,"still latent encoder/whitening compatibility differs from image VAE");goto done;
    }
    double loaded=h3_av_now();p.phase="image VAE decode";
    if(!h3_image_vae_decode(decoder,z.values,progress,&p,&rgb,error,size))goto done;
    double decoded=h3_av_now();
    h3_image_vae_free(decoder);decoder=NULL;
    size_t count=(size_t)rgb.width*rgb.height*3;pixels=malloc(count);
    result=calloc(1,sizeof(*result));
    if(!pixels || !result){snprintf(error,size,"cannot allocate still delivery");goto failure;}
    for(size_t i=0;i<count;i++) {
        if(!isfinite(rgb.rgb[i])){snprintf(error,size,"nonfinite image pixels");goto failure;}
        pixels[i]=(uint8_t)(fminf(1,fmaxf(0,rgb.rgb[i]))*255+.5f);
    }
    h3_frame frame={.width=rgb.width,.height=rgb.height,.stride=rgb.width*3,.rgb=pixels,
        .frame_index=0,.frame_count=1,.denoise_step=-1,.denoise_steps=0};
    if(options->on_frame && options->on_frame(&frame,options->callback_opaque)) {
        snprintf(error,size,"still delivery cancelled");goto failure;
    }
    p.phase="PNG writing";
    if(!h3_memory_checkpoint(progress(0,1,&p),"still delivery",error,size))goto failure;
    if(output && !h3_ffmpeg_write_png(output,pixels,rgb.width,rgb.height,error,size))goto failure;
    /* Completion notification follows publication; cancellation applies before it. */
    progress(1,1,&p);
    result->kind=H3_RESULT_STILL;result->width=rgb.width;result->height=rgb.height;result->frames=1;
    if(getenv("H3_PROFILE")) {
        h3_memory_snapshot mem;h3_memory_sample(&mem);
        fprintf(stderr,"h3cli: still cold_load=%.6fs decode=%.6fs delivery=%.6fs footprint=%llu denoiser_calls=0 audio_decode_calls=0 frames=1\n",
            loaded-start,decoded-loaded,h3_av_now()-decoded,(unsigned long long)mem.physical_footprint);
    }
    goto done;
failure:h3_result_free(result);result=NULL;
done:free(pixels);h3_image_vae_free(decoder);h3_video_frames_free(&rgb);return result;
}

h3_result *h3_decode_still_latent(const char *path,const char *image_vae,
    const h3_decode_options *options,char *error,size_t size) {
    if(!options || same_file(options->output_path,path)) {
        if (error && size)
            snprintf(error, size, "still output must not overwrite its latent");
        return NULL;
    }
    h3_still_latent *z=calloc(1,sizeof(*z));
    if(!z){if(error && size)snprintf(error,size,"cannot allocate still latent");return NULL;}
    if(!h3_still_latent_load(path,z,error,size)){free(z);return NULL;}
    h3_result *r=h3_decode_still_values(z,image_vae,options,error,size);
    if(r)r->still_latent=z;else {h3_still_latent_free(z);free(z);}return r;
}
