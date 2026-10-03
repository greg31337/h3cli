/* Production shared-policy placement, slot reuse and allocation recovery. */
#include "src/device.h"
#include "src/denoise/dit.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[1024];
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error); exit(1); } } while (0)
int main(void) {
    int layers=25;const char *requested=getenv("H3_TEST_MEMORY_LAYERS");
    if(requested){layers=atoi(requested);CHECK(layers==25||layers==50);}
    const char *root=getenv("H3_MODEL_DIR");CHECK(root);
    char weights[4096];CHECK(snprintf(weights,sizeof(weights),"%s/FL2VA/transformer",root)<(int)sizeof(weights));
    /* Placement invariance needs identical nonzero inputs, not archived text
     * encoder outputs. Every plan is compared with this run's streaming result. */
    uint16_t text_values[6*5120];
    for(size_t i=0;i<6*5120;i++){float value=(float)((int)(i%31)-15)/32;uint32_t bits;memcpy(&bits,&value,4);text_values[i]=(uint16_t)(bits>>16);}
    h3_text_embedding text={6,5120,text_values,{0},NULL,NULL};
    h3_layout_spec spec={6,2,2,2,8,5,NULL,0,NULL,0};h3_layout layout={0};h3_sigma_schedule sigmas;
    CHECK(h3_layout_build(&spec,&layout,error,sizeof(error))&&h3_serving_schedule_build(2,&sigmas));
    enum {NV=192,NA=512};float video[NV],audio[NA],expected[2][NV+NA],got[NV+NA];
    for(size_t i=0;i<NV;i++)video[i]=(float)((int)(i%31)-15)/8;
    for(size_t i=0;i<NA;i++)audio[i]=(float)((int)(i%37)-18)/8;
    const char *modes[]={"stream","auto","auto","auto","resident","auto","auto"};
    int caps[]={-1,1,layers-1,-1,-1,-1,2};
    uint64_t pinned_stream=0;
    for(int pass=0;pass<7;pass++) {
        setenv("H3_CUDA_WEIGHT_MODE",modes[pass],1);
        if(caps[pass]>=0){char s[32];snprintf(s,sizeof(s),"%d",caps[pass]);setenv("H3_TEST_CUDA_RESIDENT_BLOCKS",s,1);}
        else unsetenv("H3_TEST_CUDA_RESIDENT_BLOCKS");
        if(pass==5)setenv("H3_CUDA_TEST_MEMORY_BUDGET","3221225472",1);else unsetenv("H3_CUDA_TEST_MEMORY_BUDGET");
        h3_params p=H3_PARAMS_DEFAULT;h3_cuda_policy policy;
        CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy_exchange(policy);h3_sglang_exchange(H3_SGLANG_VERSION);
        h3_dit *dit=h3_dit_load_t2va(weights,"src/metal/shaders.metal",&text,&layout,&sigmas,
            layers,1,0,0,1.0f,1,1,1,1,1,1,1,1,0,0,0,NULL,NULL,error,sizeof(error));
        if(!dit&&pass==4&&strstr(error,"resident weights need")){puts("{\"mode\":\"resident\",\"capacity_rejection\":true}");continue;}
        CHECK(dit&&h3_dit_video_elements(dit)==NV&&h3_dit_audio_elements(dit)==NA);
        h3_gpu_stats before,after;CHECK(h3_dit_get_gpu_stats(dit,&before));
        CHECK(before.active_weight_blocks==(unsigned)layers);
        if(pass==0)CHECK(before.resident_blocks==0);
        if(caps[pass]>=0)CHECK(before.resident_blocks<=(unsigned)caps[pass]&&before.resident_blocks>0);
        if(pass==4)CHECK(before.resident_blocks==(unsigned)layers);
        /* Mutating process options cannot change an already captured plan. */
        setenv("H3_CUDA_WEIGHT_MODE","stream",1);
        for(int step=0;step<2;step++) {
            CHECK(h3_dit_forward(dit,step,video,audio,got,got+NV,error,sizeof(error)));
            if(pass)CHECK(!memcmp(expected[step],got,sizeof(got)));else memcpy(expected[step],got,sizeof(got));
        }
        CHECK(h3_dit_get_gpu_stats(dit,&after));
        const uint64_t block=770703360ull;
        CHECK(after.streamed_bytes==2*block*(layers-before.resident_blocks));
        CHECK(after.resident_blocks==before.resident_blocks&&after.resident_block_mask==before.resident_block_mask);
        CHECK(after.stream_slot_bytes==(before.resident_blocks<(unsigned)layers?2*block:0));
        CHECK(after.pinned_bytes==before.pinned_bytes);
        if(!pass)pinned_stream=after.pinned_bytes;
        if(before.resident_blocks)CHECK(after.pinned_bytes<pinned_stream);
        printf("{\"mode\":\"%s\",\"cap\":%d,\"layers\":%d,\"resident\":%u,\"injected_oom\":%s,\"peak_bytes\":%llu,\"streamed_bytes\":%llu,\"pinned_bytes\":%llu,\"passed\":true}\n",
            modes[pass],caps[pass],layers,after.resident_blocks,pass==5?"true":"false",(unsigned long long)after.peak_live_bytes,
            (unsigned long long)after.streamed_bytes,(unsigned long long)after.pinned_bytes);
        fflush(stdout);h3_dit_free(dit);
    }
    unsetenv("H3_CUDA_TEST_MEMORY_BUDGET");unsetenv("H3_CUDA_WEIGHT_MODE");unsetenv("H3_TEST_CUDA_RESIDENT_BLOCKS");
    h3_cuda_policy_exchange((h3_cuda_policy){0});h3_sglang_exchange(0);h3_layout_free(&layout);
    puts("ok: production full/partial/stream, two slot generations, gaps, immutable options and allocation recovery are bit-identical");return 0;
}
