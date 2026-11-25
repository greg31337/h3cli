/* Host-only shared Sage3 plan and per-phase capacity report. No tensors/models. */
#include "src/denoise/sage3_plan.h"
#include "src/weights/quant.h"
#include <assert.h>
#include <stdio.h>
int main(void) {
    const int sizes[][2]={{256,256},{640,480},{960,544},{1024,768},{1344,768},{1920,1088}};
    const int frames[]={379,464,736,1433,1450};
    puts("{\"schema\":1,\"kind\":\"host estimates, not measurements\",\"attention_recipe\":1,\"attention_plan\":1,\"workspace_bytes\":536870912,\"other_bytes\":2147483648,\"capacity_fraction\":0.9,\"cases\":[");
    for(unsigned i=0;i<6;i++)for(unsigned j=0;j<5;j++) {
        int w=sizes[i][0],h=sizes[i][1],f=frames[j];
        h3_canvas_memory b;h3_quant_memory q;h3_sage3_plan p;
        assert(h3_canvas_memory_estimate(w,h,f,h3_canvas_native_profile(w,h,1),512,0,&b));
        assert(h3_quant_memory_estimate(b.sequence_rows,50,&q));
        assert(h3_sage3_plan_build(b.sequence_rows,56,H3_ATTENTION_WORKSPACE_BYTES,&p));
        const double divisor=1073741824.*.9,other=2147483648.,workspace=H3_ATTENTION_WORKSPACE_BYTES;
        printf("{\"width\":%d,\"height\":%d,\"frames\":%d,\"rows\":%llu,\"padded_rows\":%u,\"query_rows\":%u,\"head_group\":%u,\"planned_workspace_bytes\":%llu,\"bf16_qkv_bytes\":%llu,",
            w,h,f,(unsigned long long)b.sequence_rows,p.padded_rows,p.query_rows,p.head_group,
            (unsigned long long)p.workspace_bytes,(unsigned long long)(p.input_elements*6));
        printf("\"sampler_host_bytes\":%llu,\"decode_host_bytes\":%llu,\"bf16_stream_host_cache_bytes\":%llu,\"nvfp4_stream_host_bytes\":%llu,\"nvfp4_cold_host_bytes\":%llu,\"nvfp4_mlp_bytes\":%llu,\"nvfp4_activation_bytes\":%llu,\"nvfp4_padded_output_bytes\":%llu,",
            (unsigned long long)b.sampler_host_bytes,(unsigned long long)b.decode_host_bytes,
            (unsigned long long)b.streamed_host_cache_bytes,(unsigned long long)q.streamed_host_bytes,
            (unsigned long long)q.cold_host_bytes,(unsigned long long)q.mlp_bytes,
            (unsigned long long)q.activation_bytes,(unsigned long long)q.padded_output_bytes);
        printf("\"bf16_resident_gib\":%.6f,\"bf16_stream_gib\":%.6f,\"nvfp4_cold_resident_gib\":%.6f,\"nvfp4_cold_stream_gib\":%.6f,\"nvfp4_warm_resident_gib\":%.6f,\"nvfp4_warm_stream_gib\":%.6f}%s\n",
            ((double)(b.dit_future_bytes+b.resident_core_bytes)+other+workspace)/divisor,
            ((double)(b.dit_future_bytes+b.stream_core_bytes)+other+workspace)/divisor,
            ((double)q.cold_device_bytes+other+workspace)/divisor,
            ((double)(q.future_bytes+q.cold_conversion_bytes)+other+workspace)/divisor,
            ((double)q.warm_device_bytes+other+workspace)/divisor,
            ((double)q.future_bytes+other+workspace)/divisor,i==5&&j==4?"":",");
    }
    puts("]}");return 0;
}
