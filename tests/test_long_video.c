/* Bounded temporal metadata/index checks: never allocate a movie tensor. */
#include "src/internal.h"
#include "src/geometry.h"
#include "src/sampling/av_state.h"
#include "src/sampling/bridge.h"
#include "src/vae/tiny_vae.h"
#include "src/upscale/upscale_gpu.h"
#include "src/weights/quant.h"
#include "src/denoise/sage3_plan.h"
#include "src/sampling/sampler_state.h"
#include <assert.h>
#include <limits.h>
#include <stdio.h>
#include <string.h>

int main(void) {
    const int raw[]={22,362,363,464,480,481,719,720,736,737,1433,1434,1440,1450,1451,INT_MAX};
    const int aligned[]={22,362,379,464,481,481,719,736,736,753,1433,1450,1450,1450,1467,INT_MAX};
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
    t=h3_temporal(1440);
    assert(t.frame_count==1450&&t.video_t==427&&t.audio_t==2417);
    assert(h3_tiny_vae_frames(427)==(H3_NATIVE_CUDA?1450:0));
    assert(h3_tiny_vae_frames(432)==0);
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
    assert(!h3_bridge_profile_build(1467,8,.5,H3_BRIDGE_STEPPED,&bridge,NULL,0,error,sizeof(error)));
    assert(h3_continuation_context(1416,&prefix));
    assert(prefix.video_prefix_t==417&&prefix.audio_prefix_t==2360);
    assert(h3_bridge_profile_build(1416,8,.5,H3_BRIDGE_STEPPED,&bridge,NULL,0,error,sizeof(error))==H3_NATIVE_CUDA);
    if(H3_NATIVE_CUDA)assert(h3_bridge_profile_valid(&bridge));
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
    assert(h3_av_state_shape_profile(1920,1088,1450,2,&s));
    assert(s.video_elements==(size_t)24*427*120*68&&s.audio_elements==64*2417);
    assert(h3_canvas_memory_estimate(1920,1088,1450,2,512,0,&m));
    assert(m.video_rows==871080&&m.sequence_rows==876426);
    assert(h3_canvas_sequence_valid(m.sequence_rows,1));
    assert(h3_canvas_memory_estimate(1920,1088,1450,2,512,123574,&extra));
    assert(extra.sequence_rows==1000000&&h3_attention_layout_valid(H3_ATTENTION_SAGE3,extra.sequence_rows,error,sizeof(error)));
    assert(h3_canvas_memory_estimate(1920,1088,1450,2,512,123575,&extra));
    assert(!h3_attention_layout_valid(H3_ATTENTION_SAGE3,extra.sequence_rows,error,sizeof(error)));
    assert(!h3_canvas_memory_estimate(1920,1088,1450,2,512,UINT64_MAX,&extra));
    assert(h3_canvas_memory_estimate(3840,2176,1450,3,512,0,&m));
    assert(m.video_rows==3484320&&!h3_canvas_sequence_valid(m.sequence_rows,1));
    assert(!h3_canvas_memory_estimate(256,256,1467,0,512,0,&m));
    h3_quant_memory q;
    const int sizes[][2]={{256,256},{640,480},{1024,768},{1344,768},{1920,1088}};
    puts("NVFP4 width,height,rows,padded,chunk,packed,cold,future,warm,host");
    for(size_t i=0;i<sizeof(sizes)/sizeof(*sizes);i++) {
        int w=sizes[i][0],h=sizes[i][1];
        assert(h3_canvas_memory_estimate(w,h,1450,h3_canvas_native_profile(w,h,1),512,0,&m));
        assert(h3_quant_memory_estimate(m.sequence_rows,50,&q));
        assert(q.chunk_rows==8192&&q.padded_rows>=q.logical_rows&&q.padded_rows-q.logical_rows<128);
        assert(q.cold_device_bytes==q.warm_device_bytes+q.cold_conversion_bytes);
        assert(q.mlp_bytes==UINT64_C(8192)*14336*6);
        printf("%d,%d,%llu,%llu,%llu,%llu,%llu,%llu,%llu,%llu\n",w,h,
            (unsigned long long)q.logical_rows,(unsigned long long)q.padded_rows,(unsigned long long)q.chunk_rows,
            (unsigned long long)q.packed_core_bytes,(unsigned long long)q.cold_conversion_bytes,
            (unsigned long long)q.future_bytes,(unsigned long long)q.warm_device_bytes,(unsigned long long)q.cold_host_bytes);
    }
    assert(h3_quant_memory_estimate(1000000,50,&q)&&q.padded_rows==1000064);
    assert(!h3_quant_memory_estimate(1000001,50,&q));
    assert(!h3_quant_memory_estimate(UINT64_MAX,50,&q));
    assert(!h3_quant_memory_estimate(1,0,&q)&&!h3_quant_memory_estimate(1,51,&q));
    assert(h3_quant_native_recipe(2,0,0,362,1344,768)==2);
    assert(h3_quant_native_recipe(2,0,0,464,256,256)==5);
    assert(h3_quant_native_recipe(2,0,0,22,1920,1088)==5);
    assert(h3_quant_native_recipe(0,0,0,1450,1920,1088)==0);
    assert(h3_quant_native_recipe(2,0,H3_ATTENTION_SAGE3,362,1344,768)==2);
    assert(h3_quant_native_recipe(2,0,H3_ATTENTION_SAGE3,363,256,256)==5);
    assert(h3_quant_native_recipe(2,0,H3_ATTENTION_SAGE3,1450,1920,1088)==5);
    assert(h3_quant_native_recipe(2,0,H3_ATTENTION_SAGE3,22,1920,1088)==5);
    h3_sage3_plan sp;
    const uint64_t sage_rows[]={1,17,127,128,129,257,4096,133446,223116,299593,299594,333282,599186,599187,876426,999999,1000000};
    for(size_t i=0;i<sizeof(sage_rows)/sizeof(*sage_rows);i++) {
        uint64_t rows=sage_rows[i];
        assert(h3_sage3_plan_build(rows,56,H3_ATTENTION_WORKSPACE_BYTES,&sp));
        assert(h3_attention_layout_valid(H3_ATTENTION_SAGE3,rows,error,sizeof(error)));
        assert(sp.padded_rows>=rows&&sp.padded_rows-rows<128&&sp.padded_rows%128==0);
        assert(sp.query_rows>=128&&sp.query_rows<=sp.padded_rows&&sp.query_rows%128==0);
        assert(sp.head_group>=1&&sp.head_group<=56&&sp.workspace_bytes<=H3_ATTENTION_WORKSPACE_BYTES);
        assert((uint64_t)sp.head_group*sp.padded_rows*128<INT32_MAX);
        assert((uint64_t)sp.head_group*(sp.query_rows/128)*sp.padded_rows<INT32_MAX);
        assert(sp.input_elements==rows*56*128);
    }
    assert(sp.input_elements==UINT64_C(7168000000)&&sp.input_elements>UINT32_MAX);
    assert(sp.head_group==1&&sp.query_rows<sp.padded_rows);
    assert(!h3_sage3_plan_build(0,56,H3_ATTENTION_WORKSPACE_BYTES,&sp));
    assert(!h3_sage3_plan_build(1000001,56,H3_ATTENTION_WORKSPACE_BYTES,&sp));
    assert(!h3_sage3_plan_build(UINT64_MAX,56,H3_ATTENTION_WORKSPACE_BYTES,&sp));
    assert(!h3_sage3_plan_build(128,55,H3_ATTENTION_WORKSPACE_BYTES,&sp));
    assert(!h3_sage3_plan_build(128,56,H3_SAGE_RESERVED_BYTES,&sp));
    assert(!h3_sage3_plan_build(1000000,56,UINT64_C(128)<<20,&sp));
    for(int version=1;version<=3;version++) {
        const size_t video[]={0,107,217,427},audio[]={0,604,1228,2418};
        assert(h3_sampler_temporal_classes((unsigned)version,0)==video[version]);
        assert(h3_sampler_temporal_classes((unsigned)version,1)==audio[version]);
    }
    assert(h3_sampler_temporal_version(362)==1&&h3_sampler_temporal_version(363)==2);
    assert(h3_sampler_temporal_version(736)==2&&h3_sampler_temporal_version(737)==3);
    assert(h3_sampler_temporal_version(1450)==3&&!h3_sampler_temporal_classes(4,0));
    p.frames=1450;p.cuda_denoise_quant=2;snprintf(device.backend,sizeof(device.backend),"cuda");
    assert(h3_request_params_valid(&p,&device,error,sizeof(error)));
    p.cuda_denoise_quant=1;assert(!h3_request_params_valid(&p,&device,error,sizeof(error)));
    p.cuda_denoise_quant=2;p.save_upscale_state="/tmp/forbidden.h3up";
    assert(!h3_request_params_valid(&p,&device,error,sizeof(error)));
    p.save_upscale_state=NULL;p.cuda_denoise_quant=0;
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
    p.cuda_attention=H3_ATTENTION_SAGE3;
    const int sage_canvases[][2]={{256,256},{640,480},{960,544},{1024,768},{1344,768},{1920,1088}};
    for(size_t i=0;i<sizeof(sage_canvases)/sizeof(*sage_canvases);i++)for(int quant=0;quant<=2;quant++) {
        p.width=sage_canvases[i][0];p.height=sage_canvases[i][1];p.frames=1450;p.cuda_denoise_quant=quant;
        assert(h3_request_params_valid(&p,&device,error,sizeof(error))==(quant!=H3_QUANT_FP8));
        p.save_upscale_state="/tmp/forbidden.h3up";
        assert(!h3_request_params_valid(&p,&device,error,sizeof(error)));p.save_upscale_state=NULL;
    }
    p.width=p.height=256;p.frames=464;p.cuda_denoise_quant=0;
    for(int mode=0;mode<=4;mode++) {
        p.cuda_attention=mode;
        assert(h3_request_params_valid(&p,&device,error,sizeof(error))==(mode==0||mode==H3_ATTENTION_SAGE3));
    }
    p.cuda_attention=H3_ATTENTION_SAGE3;
    for(int flag=0;flag<7;flag++) {
        h3_params bad=p;
        switch(flag) {
        case 0:bad.denoise_reuse=2;break;
        case 1:bad.core_reuse=4;break;
        case 2:bad.dit_layers=49;break;
        case 3:bad.token_reduction=1;break;
        case 4:bad.adaptive_cache=1;break;
        case 5:bad.backend=1;break;
        case 6:bad.geometry_profile=1;break;
        }
        assert(!h3_request_params_valid(&bad,&device,error,sizeof(error)));
    }
    puts("PASS backend duration, overflow, anchors, bridge indices and upscale capacity metadata");
    return 0;
}
