#include "src/geometry.h"
#include <assert.h>
#include <stdio.h>
#include <string.h>

int main(int argc,char **argv) {
    if(argc==2&&!strcmp(argv[1],"--report")) {
        const int duration[]={22,39,124,243,362};
        puts("[");
        for(int i=0;i<5;i++)for(int up=0;up<2;up++) {
            h3_canvas_memory m;
            assert(h3_canvas_memory_estimate(up?3840:1920,up?2176:1088,duration[i],up?3:2,512,0,&m));
            uint64_t other=UINT64_C(2)<<30;
            uint64_t k0=m.upscale_reserve_bytes>m.decode_device_bytes?m.upscale_reserve_bytes:m.decode_device_bytes;
            printf("{\"frames\":%d,\"upscale\":%d,\"rows\":%llu,\"resident_GiB\":%.6f,\"streamed_GiB\":%.6f,\"K0_GiB\":%.6f,\"decode_host_GiB\":%.6f,\"sampler_host_GiB\":%.6f}%s\n",
                duration[i],up,(unsigned long long)m.sequence_rows,
                (m.dit_future_bytes+m.resident_core_bytes+other)/1073741824./.9,
                (m.dit_future_bytes+m.stream_core_bytes+other)/1073741824./.9,
                k0/1073741824./.9,m.decode_host_bytes/1073741824.,m.sampler_host_bytes/1073741824.,i==4&&up?"":",");
        }
        puts("]");return 0;
    }
    assert(h3_canvas_native_profile(1344,768,0)==0);
    assert(h3_canvas_native_profile(1920,1088,0)==-1);
    assert(h3_canvas_native_profile(1920,1088,1)==2);
    assert(h3_canvas_native_profile(1088,1920,1)==2);
    assert(h3_canvas_native_profile(3840,2176,1)==-1);
    assert(!h3_canvas_valid(1920,1080,2));
    assert(!h3_canvas_valid(1920,1120,2));
    assert(h3_canvas_valid(3840,2176,3));
    assert(h3_canvas_valid(2176,3840,3));
    assert(!h3_canvas_valid(3840,2208,3));
    assert(!h3_canvas_valid(32,32,4));
    assert(!h3_canvas_valid(INT32_MAX,INT32_MAX,3));
    assert(!h3_canvas_backend(2,0)&&!h3_canvas_backend(3,0));
    assert(h3_canvas_backend(1,0)&&h3_canvas_backend(3,1));
    const int frames[]={22,39,124,243,362};
    const uint64_t rows[]={14280,24480,75480,146880,218280};
    for(unsigned i=0;i<5;i++) {
        h3_canvas_memory n,u,portrait;
        assert(h3_canvas_memory_estimate(1920,1088,frames[i],2,512,0,&n));
        assert(h3_canvas_memory_estimate(3840,2176,frames[i],3,512,0,&u));
        assert(h3_canvas_memory_estimate(2176,3840,frames[i],3,512,0,&portrait));
        assert(n.video_rows==rows[i]&&u.video_rows==4*rows[i]);
        assert(u.dit_future_bytes==portrait.dit_future_bytes);
        assert(n.resident_core_bytes==38535168000ULL);
        assert(n.stream_core_bytes==1541406720ULL);
        assert(n.dit_future_bytes==n.sequence_rows*358400+(1ULL<<30));
        assert(u.video_bytes==4*n.video_bytes);
        assert(u.audio_bytes==n.audio_bytes);
        assert(h3_canvas_sequence_valid(u.sequence_rows,1));
        assert((uint64_t)u.sequence_rows*7168*3*2<UINT64_MAX);
    }
    h3_canvas_memory n;
    assert(!h3_canvas_memory_estimate(1920,1088,72,2,512,0,&n));
    assert(!h3_canvas_memory_estimate(1920,1088,22,2,UINT64_MAX,1,&n));
    assert(!h3_canvas_memory_estimate(1920,1088,22,2,UINT64_MAX/358400,0,&n));
    assert(h3_canvas_dit_future(UINT64_MAX)==UINT64_MAX);
    assert(h3_canvas_sequence_valid(1000000,1));
    assert(!h3_canvas_sequence_valid(1000001,1));
    assert(!h3_canvas_sequence_valid(874838,0));
    assert(h3_canvas_text_rows_valid(UINT32_MAX/5376u));
    assert(!h3_canvas_text_rows_valid(UINT32_MAX/5376u+1));
    assert(h3_canvas_mlp_rows(14866,14336)==14866);
    assert(h3_canvas_mlp_rows(874838,14336)==131072);
    /* Logical final-row QKV offsets cross 32-bit elements without allocating. */
    uint64_t qkv=((uint64_t)874837*3*7168+7167)*2;
    assert(qkv>UINT32_MAX&&qkv/2>UINT32_MAX);
    puts("large canvas geometry/temporal/overflow/memory checks passed");
    return 0;
}
