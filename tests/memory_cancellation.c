/* Linked to instrumented copies of the actual model loops. h3_test_enter is
 * inserted immediately before real layer/block/step execution, not in callbacks. */
#include "src/memory.h"
#include "src/conditioning/text_encoder.h"
#include "src/conditioning/vision_encoder.h"
#include "src/vae/audio_vae.h"
#include "src/vae/video_vae.h"
#include "src/vae/video_encoder.h"
#include "src/denoise/dit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error[512];
static const char *target;
static int entered, max_completed, callbacks, memory_mode;
#define CHECK(c) do { if (!(c)) { fprintf(stderr,"FAIL %s:%d: %s (%s)\n",__FILE__,__LINE__,#c,error); exit(1); } } while (0)
void h3_test_enter(const char *name) { if (!strcmp(name,target)) entered++; }
static void reset(const char *name, int mode) {
    target=name; entered=max_completed=callbacks=0; memory_mode=mode;
    error[0]=0; unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");
}
static int progress(int completed,int total,void *opaque) {
    (void)total; (void)opaque;
    callbacks++;
    if (completed>max_completed) max_completed=completed;
    if (!entered) return 0;
    if (memory_mode) {
        setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES","18446744073709551615",1);
        return 0;
    }
    return 1;
}
static int dit_progress(const char *phase,int completed,int total,void *opaque) {
    if (strcmp(phase,"denoise") && strcmp(phase,"denoise enqueue")) return 0;
    return progress(completed,total,opaque);
}
static void cancelled(int ok) {
    CHECK(!ok);
    CHECK(entered==1);
    CHECK(max_completed<=1);
    CHECK(strstr(error,memory_mode?"reclaimable physical memory":"cancelled"));
    printf("ok: %s %s cancellation after exactly %d entered block/step; callbacks=%d\n",
        target,memory_mode?"memory":"user",entered,callbacks);
    fflush(stdout);
    unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");
}
static void encoders(void) {
    float pixels[3*32*32];
    for (size_t i=0;i<sizeof(pixels)/sizeof(*pixels);i++) pixels[i]=(float)(i%256)/255.0f;
    for (int mode=0;mode<2;mode++) {
        reset("text",mode);
        h3_text_embedding text; uint32_t ids[]={151669,9906,151670};
        int ok=h3_text_encode_bf16("models/MiniMax-H3/FL2VA/text_encoder","src/metal/shaders.metal",
            ids,3,progress,NULL,&text,error,sizeof(error));
        cancelled(ok); CHECK(!text.values); h3_text_embedding_free(&text);
        reset("vision",mode);
        h3_vision_output vision;
        ok=h3_vision_encode_bf16("models/MiniMax-H3/FL2VA/text_encoder","src/metal/shaders.metal",
            pixels,1,32,32,progress,NULL,&vision,error,sizeof(error));
        cancelled(ok); CHECK(!vision.merged); h3_vision_output_free(&vision);
        reset("audio",mode);
        float latent[64]={0}; h3_audio_waveform waveform;
        ok=h3_audio_vae_decode("models/MiniMax-H3/FL2VA/audio_vae","src/metal/shaders.metal",
            latent,1,progress,NULL,&waveform,error,sizeof(error));
        cancelled(ok); CHECK(!waveform.pcm); h3_audio_waveform_free(&waveform);
        reset("audio-encoder",mode);
        float pcm[3200*2]={0}; h3_audio_latent audio;
        ok=h3_audio_vae_encode("models/MiniMax-H3/FL2VA/audio_vae","src/metal/shaders.metal",
            pcm,3200,progress,NULL,&audio,error,sizeof(error));
        cancelled(ok); CHECK(!audio.values); h3_audio_latent_free(&audio);
        reset("video",mode);
        float video[24*2*2*2]={0}; h3_video_frames frames;
        ok=h3_video_vae_decode("models/MiniMax-H3/FL2VA/video_vae/source","src/metal/shaders.metal",
            video,2,2,2,progress,NULL,&frames,error,sizeof(error));
        cancelled(ok); CHECK(!frames.rgb); h3_video_frames_free(&frames);
        reset("video-encoder",mode);
        h3_video_latent encoded;
        ok=h3_video_vae_encode("models/MiniMax-H3/FL2VA/video_vae/source","src/metal/shaders.metal",
            pixels,1,32,32,progress,NULL,&encoded,error,sizeof(error));
        cancelled(ok); CHECK(!encoded.values); h3_video_latent_free(&encoded);
    }
    reset("load",0);
    h3_video_vae_decoder *decoder=h3_video_vae_decoder_load(
        "models/MiniMax-H3/FL2VA/video_vae/source","src/metal/shaders.metal",2,2,NULL,NULL,error,sizeof(error));
    CHECK(decoder);
    for (int mode=0;mode<2;mode++) {
        reset("video",mode);
        float video[24*7*2*2]={0}; h3_video_frames frames;
        int ok=h3_video_vae_decoder_decode_progress(decoder,video,7,
            progress,NULL,&frames,error,sizeof(error));
        cancelled(ok); CHECK(!frames.rgb);
    }
    /* After two cancellations the cached command stream must be usable. */
    reset("unused",0);
    float video[24*7*2*2]={0}; h3_video_frames frames;
    CHECK(h3_video_vae_decoder_decode(decoder,video,7,&frames,error,sizeof(error)));
    CHECK(frames.frames==22); h3_video_frames_free(&frames);
    h3_video_vae_decoder_free(decoder);
    puts("ok: resident video decoder reused successfully after both cancellations");
}
static void denoiser(void) {
    reset("unused",0);
    uint16_t values[6*5120]={0};
    h3_text_embedding text={.tokens=6,.width=5120,.values=values};
    h3_layout_spec spec={6,2,2,2,8,5,NULL,0,NULL,0};
    h3_layout layout; h3_sigma_schedule sigmas;
    CHECK(h3_layout_build(&spec,&layout,error,sizeof(error)));
    CHECK(h3_schedule_build(4,&sigmas));
    h3_dit *dit=h3_dit_load_t2va("models/MiniMax-H3/FL2VA/transformer","src/metal/shaders.metal",
        &text,&layout,&sigmas,50,1,0,0,1.0f,0,1,1,1,1,1,1,1,0,0,0,
        NULL,NULL,error,sizeof(error));
    CHECK(dit);
    for (int backend=0;backend<3;backend++) for (int mode=0;mode<2;mode++) {
        reset("dit",mode);
        if (backend==1) { setenv("H3_GPU_SAMPLER","1",1); unsetenv("H3_CPU_SAMPLER"); }
        else { unsetenv("H3_GPU_SAMPLER"); setenv("H3_CPU_SAMPLER","1",1); }
        float video[24*2*2*2]={0}, audio[32*2*8]={0};
        CHECK(h3_dit_reset_run(dit,NULL,0,NULL,0,error,sizeof(error)));
        int ok=backend==2 ? h3_dit_denoise(dit,video,audio,dit_progress,NULL,error,sizeof(error)) :
            h3_dit_denoise_euler(dit,video,audio,1,dit_progress,NULL,error,sizeof(error));
        cancelled(ok);
    }
    h3_dit_free(dit); h3_layout_free(&layout);
    unsetenv("H3_GPU_SAMPLER"); unsetenv("H3_CPU_SAMPLER");
}
static int public_progress(const char *phase,int completed,int total,void *opaque) {
    (void)phase; (void)completed; (void)total; (void)opaque;
    callbacks++;
    return !memory_mode;
}
static int cached_progress(const char *phase,int completed,int total,void *opaque) {
    if (strcmp(phase,"denoise") && strcmp(phase,"denoise enqueue")) return 0;
    if (entered) {
        h3_cache_info info; h3_cache_get_info(opaque,&info);
        CHECK(info.prepared_dit && info.embedding_entries);
    }
    return progress(completed,total,NULL);
}
static void bridges(void) {
    reset("unused",0);
    h3_ctx *ctx=h3_load_dir("models/MiniMax-H3"); CHECK(ctx);
    h3_params params=H3_PARAMS_DEFAULT;
    params.width=params.height=256; params.frames=56; params.steps=4;
    params.on_progress=public_progress;
    for (int mode=0;mode<2;mode++) {
        reset("unused",mode);
        if(mode) setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES","18446744073709551615",1);
        h3_result *r=h3_generate(ctx,"A woman smiles.",&params);
        CHECK(!r); CHECK(callbacks==1);
        CHECK(strstr(h3_last_error(ctx),mode?"reclaimable physical memory":"cancelled"));
        puts(mode?"ok: low-memory bridge notifies user and preserves diagnostic":"ok: user cancellation stops at the initial phase");
    }
    unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");
    h3_cache_set_enabled(ctx,1);
    params.width=params.height=32; params.frames=22;
    params.on_progress=cached_progress; params.callback_opaque=ctx;
    for (int mode=0;mode<2;mode++) {
        reset("dit",mode);
        h3_result *r=h3_generate(ctx,"A woman smiles.",&params);
        snprintf(error,sizeof(error),"%s",h3_last_error(ctx));
        cancelled(r!=NULL);
        h3_cache_info info; h3_cache_get_info(ctx,&info);
        CHECK(!info.embedding_entries && !info.embedding_bytes && !info.prepared_dit && !info.video_decoder);
        puts("ok: cancelled public generation released populated conditioning/DiT caches; context reusable");
    }
    h3_free(ctx);
}
int main(void) {
    encoders(); denoiser(); bridges();
    puts("ok: all real model cancellation loops prevent the next execution");
    return 0;
}
