/* Bounded temporal metadata/index checks: never allocate a movie tensor. */
#include "src/internal.h"
#include "src/geometry.h"
#include "src/sampling/av_state.h"
#include "src/sampling/bridge.h"
#include "src/vae/tiny_vae.h"
#include "src/upscale/upscale_gpu.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const int raw[]={22,362,363,464,480,481,719,720,736,737,INT_MAX};
    const int aligned[]={22,362,379,464,481,481,719,736,736,753,INT_MAX};
    char error[512];
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=256;p.steps=2;
    h3_device_info device={0};
    for(int cuda=0;cuda<2;cuda++) {
        snprintf(device.backend,sizeof(device.backend),"%s",cuda?"cuda":"metal");
        for(size_t i=0;i<sizeof(raw)/sizeof(*raw);i++) {
            assert(h3_align_frame_count(raw[i])==aligned[i]);
            int ok=aligned[i]<=h3_frame_limit(cuda);
            p.frames=raw[i];
            assert(h3_request_params_valid(&p,&device,error,sizeof(error))==ok);
            assert(h3_frames_supported(aligned[i],cuda)==ok);
        }
    }
    assert(!h3_frames_supported(INT_MIN,1));
    assert(h3_temporal(INT_MAX).audio_t==INT_MAX);
    assert(h3_video_encoder_latent_t(INT_MAX)==536870912);
    h3_temporal_shape t=h3_temporal(736);
    assert(t.frame_count==736&&t.video_t==217&&t.audio_t==1227);
    assert(h3_video_time_boundary_frames(216)==732);
    assert(h3_tiny_vae_frames(107)==362);
    assert(h3_tiny_vae_frames(217)==(H3_NATIVE_CUDA?736:0));
    assert(h3_tiny_vae_frames(218)==0);
    h3_av_state_info s;
    assert(h3_av_state_shape_profile(1920,1088,736,2,&s));
    assert(s.video_elements==(size_t)24*217*120*68&&s.audio_elements==64*1227);
    assert(!h3_av_state_shape_profile(1920,1088,737,2,&s));
    h3_denoise_prefix prefix;
    assert(h3_continuation_context(702,&prefix));
    assert(prefix.video_prefix_t==207&&prefix.audio_prefix_t==1170);
    assert(h3_prefix_row_class(prefix,0,206*2040,68,120,1227)==H3_ROW_PRESERVED_VIDEO);
    assert(h3_prefix_row_class(prefix,0,207*2040,68,120,1227)==H3_ROW_GENERATED_VIDEO);
    assert(h3_prefix_row_class(prefix,1,1227+1169,68,120,1227)==H3_ROW_PRESERVED_AUDIO);
    assert(h3_prefix_row_class(prefix,1,1227+1170,68,120,1227)==H3_ROW_GENERATED_AUDIO);
    h3_bridge_profile bridge;
    assert(h3_bridge_profile_build(702,8,.5,H3_BRIDGE_STEPPED,&bridge,NULL,0,error,sizeof(error))==H3_NATIVE_CUDA);
    if(H3_NATIVE_CUDA) {
        assert(h3_bridge_profile_valid(&bridge));
        assert(h3_bridge_row_class(&bridge,0,206*2040,68,120,1227)==H3_ROW_PRESERVED_VIDEO);
        assert(h3_bridge_row_class(&bridge,1,1227+1169,68,120,1227)==H3_ROW_PRESERVED_AUDIO);
        bridge.audio_classes[1169]=255;assert(!h3_bridge_profile_valid(&bridge));
    }
    assert(!h3_bridge_profile_build(753,8,.5,H3_BRIDGE_STEPPED,&bridge,NULL,0,error,sizeof(error)));
    h3_canvas_memory m,extra;
    const int cases[][3]={{256,256,464},{256,256,481},{256,256,736},{1344,768,719},{1344,768,736},{1920,1088,736}};
    puts("width,height,frames,rows,resident_GiB,streamed_GiB,decode_host_GiB");
    for(size_t i=0;i<sizeof(cases)/sizeof(*cases);i++) {
        int w=cases[i][0],h=cases[i][1],f=cases[i][2],profile=h3_canvas_native_profile(w,h,1);
        assert(h3_canvas_memory_estimate(w,h,f,profile,512,0,&m));
        assert(h3_canvas_sequence_valid(m.sequence_rows,1));
        assert(h3_canvas_memory_estimate(w,h,f,profile,512,1000,&extra));
        assert(extra.sequence_rows==m.sequence_rows+1000&&extra.sampler_host_bytes>m.sampler_host_bytes);
        printf("%d,%d,%d,%llu,%.3f,%.3f,%.3f\n",w,h,f,(unsigned long long)m.sequence_rows,
            (m.dit_future_bytes+m.resident_core_bytes+(2ULL<<30))/1073741824./.9,
            (m.dit_future_bytes+m.stream_core_bytes+(2ULL<<30))/1073741824./.9,m.decode_host_bytes/1073741824.);
    }
    assert(m.video_rows==442680&&m.sequence_rows==445646);
    assert(h3_canvas_memory_estimate(3840,2176,736,3,512,0,&m));
    assert(m.video_rows==1770720&&!h3_canvas_sequence_valid(m.sequence_rows,1));
    /* Logical operator counts only: even the largest full volume is not allocated. */
    size_t counts[4];h3_upscale_gpu_shape op={.time=217,.height=136,.width=240,.channels=512};
    assert(h3_upscale_gpu_counts(H3_UP_NORM,op,counts));
    assert(counts[0]==(size_t)217*136*240*512&&counts[0]*2>UINT32_MAX);
    op.time=512;op.height=op.width=256;
    assert(!h3_upscale_gpu_counts(H3_UP_NORM,op,counts));
    assert(h3_canvas_mlp_rows(445646,14336)==131072);
    uint64_t last_qkv=((uint64_t)445645*3*7168+3*7168-1)*2;
    assert(last_qkv>UINT32_MAX);
    p.frames=464;p.cuda_attention=1;snprintf(device.backend,sizeof(device.backend),"cuda");
    assert(!h3_request_params_valid(&p,&device,error,sizeof(error))&&strstr(error,"unqualified"));
    puts("PASS backend duration, overflow, anchors, bridge indices and upscale capacity metadata");
    return 0;
}
