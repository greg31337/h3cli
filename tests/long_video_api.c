/* Bounded full-depth API context reuse and failure admission; no long media. */
#include "src/h3.h"
#include "src/upscale/upscale.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include "src/memory.h"
#include "src/device.h"
#include "src/sglang/sglang.h"
#include "src/vae/audio_vae.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static char error[1024];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);return 1;}}while(0)
static int denied(uint64_t *bytes,void *opaque){(void)opaque;*bytes=0;return 1;}
int main(int argc,char **argv) {
    CHECK(argc==3);
    h3_ctx *ctx=h3_load_dir(argv[1]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=256;p.height=256;p.frames=464;p.steps=2;p.seed=42;
    p.state_only=1;p.save_upscale_state=argv[2];
    const char *prompt="A red ceramic teapot on a wooden table, soft studio light.";
    h3_result *first=h3_generate(ctx,prompt,&p);
    snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(first&&first->av_state);
    h3_memory_snapshot first_memory={0};h3_memory_sample(&first_memory);CHECK(first_memory.process_valid);
    h3_device_info first_device;CHECK(h3_device_query(&first_device,error,sizeof(error)));
    h3_result *again=h3_generate(ctx,prompt,&p);
    snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(again&&again->av_state);
    CHECK(first->av_state->info.video_elements==again->av_state->info.video_elements);
    CHECK(!memcmp(first->av_state->video,again->av_state->video,first->av_state->info.video_elements*4));
    CHECK(!memcmp(first->av_state->audio,again->av_state->audio,first->av_state->info.audio_elements*4));
    h3_memory_snapshot repeat_memory={0};h3_memory_sample(&repeat_memory);CHECK(repeat_memory.process_valid);
    h3_device_info repeat_device;CHECK(h3_device_query(&repeat_device,error,sizeof(error)));
    CHECK(repeat_memory.resident<=first_memory.resident+(UINT64_C(2)<<30));
    CHECK(repeat_device.free_device_memory+(UINT64_C(512)<<20)>=first_device.free_device_memory);
    CHECK(repeat_device.free_device_memory>repeat_device.device_memory/10);
    printf("reuse plateau: first_RSS=%llu repeat_RSS=%llu first_free_VRAM=%llu repeat_free_VRAM=%llu bytes\n",
        (unsigned long long)first_memory.resident,(unsigned long long)repeat_memory.resident,
        (unsigned long long)first_device.free_device_memory,(unsigned long long)repeat_device.free_device_memory);
    h3_cache_clear(ctx);
    h3_memory_set_test_query(denied,NULL);
    h3_result *bad=h3_generate(ctx,prompt,&p);
    h3_memory_set_test_query(NULL,NULL);
    CHECK(!bad&&h3_memory_error(h3_last_error(ctx)));
    /* Independent audio-VAE calls must produce identical PCM from unchanged audio. */
    char audio_path[4096];snprintf(audio_path,sizeof(audio_path),"%s/FL2VA/audio_vae",argv[1]);
    h3_audio_waveform a={0},b={0};int old=h3_sglang_exchange(1);
    CHECK(h3_audio_vae_decode(audio_path,"src/metal/shaders.metal",first->av_state->audio,
        first->av_state->info.audio_t,NULL,NULL,&a,error,sizeof(error)));
    CHECK(h3_audio_vae_decode(audio_path,"src/metal/shaders.metal",again->av_state->audio,
        again->av_state->info.audio_t,NULL,NULL,&b,error,sizeof(error)));
    CHECK(a.samples==b.samples&&a.channels==2&&a.sample_rate==32000);
    CHECK(!memcmp(a.pcm,b.pcm,(size_t)a.samples*2*4));
    for(size_t i=0;i<(size_t)a.samples*2;i++)CHECK(isfinite(a.pcm[i]));
    h3_audio_waveform_free(&a);h3_audio_waveform_free(&b);h3_sglang_exchange(old);
    h3_result_free(first);h3_result_free(again);h3_free(ctx);
    puts("PASS repeated extended native API reuse, identical seeded latents/audio/PCM, memory denial");
    return 0;
}
