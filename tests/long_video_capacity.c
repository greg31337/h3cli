/* Reproducible capacity arithmetic only; no model, media, or tensor allocation. */
#include "src/geometry.h"
#include "src/weights/quant.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    const int sizes[][2]={{256,256},{640,480},{1024,768},{1344,768},{1920,1088}};
    const int frames[]={379,464,736,1433,1450};
    puts("{\"schema\":1,\"kind\":\"host estimates, not measurements\",\"other_bytes\":2147483648,\"capacity_fraction\":0.9,\"cases\":[");
    for(unsigned i=0;i<5;i++)for(unsigned j=0;j<5;j++) {
        h3_canvas_memory b;h3_quant_memory q;int w=sizes[i][0],h=sizes[i][1],f=frames[j];
        assert(h3_canvas_memory_estimate(w,h,f,h3_canvas_native_profile(w,h,1),512,0,&b));
        assert(h3_quant_memory_estimate(b.sequence_rows,50,&q));
        printf("{\"width\":%d,\"height\":%d,\"frames\":%d,\"rows\":%llu,\"padded_rows\":%llu,\"chunk_rows\":%llu,",w,h,f,
            (unsigned long long)b.sequence_rows,(unsigned long long)q.padded_rows,(unsigned long long)q.chunk_rows);
#define FIELD(name,value) printf("\"%s\":%llu,",name,(unsigned long long)(value))
        FIELD("bf16_future",b.dit_future_bytes);FIELD("bf16_resident",b.resident_core_bytes);FIELD("bf16_stream_slots",b.stream_core_bytes);
        FIELD("nvfp4_packed_core",q.packed_core_bytes);FIELD("nvfp4_packed_stream",q.packed_stream_bytes);
        FIELD("nvfp4_cold_conversion",q.cold_conversion_bytes);FIELD("nvfp4_activation",q.activation_bytes);
        FIELD("nvfp4_padded_output",q.padded_output_bytes);FIELD("nvfp4_mlp",q.mlp_bytes);FIELD("nvfp4_globals",q.global_bytes);
        FIELD("nvfp4_future",q.future_bytes);FIELD("nvfp4_host_cache",q.streamed_host_bytes);FIELD("nvfp4_cold_host",q.cold_host_bytes);
        FIELD("decode_host",b.decode_host_bytes);FIELD("decode_device",b.decode_device_bytes);FIELD("sampler_host",b.sampler_host_bytes);
        const double divisor=1073741824.0*.9,other=2147483648.;
        printf("\"bf16_capacity_resident_gib\":%.6f,\"bf16_capacity_stream_gib\":%.6f,\"nvfp4_cold_capacity_resident_gib\":%.6f,\"nvfp4_cold_capacity_stream_gib\":%.6f,\"nvfp4_warm_capacity_resident_gib\":%.6f,\"nvfp4_warm_capacity_stream_gib\":%.6f}%s\n",
            ((double)b.dit_future_bytes+(double)b.resident_core_bytes+other)/divisor,
            ((double)b.dit_future_bytes+(double)b.stream_core_bytes+other)/divisor,
            ((double)q.cold_device_bytes+other)/divisor,
            ((double)q.future_bytes+(double)q.cold_conversion_bytes+other)/divisor,
            ((double)q.warm_device_bytes+other)/divisor,
            ((double)q.future_bytes+other)/divisor,i==4&&j==4?"":",");
    }
    puts("]}");return 0;
}
