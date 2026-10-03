/* Native tensor export/import tests; no model weights required. */
#include "src/sampling/sampler_state.h"
#include "src/denoise/dit.h"
#include "src/denoise/dit_schedule.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int checks;
static char error[512];
#define CHECK(x) do { checks++; if(!(x)) { fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error); exit(1); } } while(0)

int main(void) {
    h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error)); CHECK(gpu);
    uint16_t original[65536],restored[65536];
    for(size_t i=0;i<65536;i++) original[i]=(uint16_t)i;
    h3_gpu_tensor *a=h3_gpu_tensor_from_bf16(gpu,original,65536),*b=h3_gpu_tensor_new_bf16(gpu,65536); CHECK(a&&b);
    CHECK(h3_gpu_begin(gpu)); CHECK(h3_gpu_copy_bf16(gpu,b,0,a,0,65536));
    CHECK(h3_gpu_synchronize(gpu)); CHECK(h3_gpu_synchronize(gpu));
    CHECK(h3_gpu_tensor_read_bf16(b,restored,65536)); CHECK(!memcmp(original,restored,sizeof(original)));
    h3_gpu_tensor_free(a); h3_gpu_tensor_free(b);
    b=h3_gpu_tensor_new_bf16(gpu,65536); CHECK(b);
    CHECK(h3_gpu_tensor_write_bf16(b,restored,65536)); memset(restored,0,sizeof(restored));
    CHECK(h3_gpu_tensor_read_bf16(b,restored,65536)); CHECK(!memcmp(original,restored,sizeof(original)));
    h3_gpu_tensor_free(b);

    float video[24*3*4*6],packed[24*3*4*6],readback[24*3*4*6],unpacked[24*3*4*6];
    size_t n=sizeof(video)/4;
    for (size_t i = 0; i < n; i++)
        video[i] = (float)i / 127;
    video[0] = -0.0f;
    CHECK(h3_dit_patchify_video(video,24,3,4,6,packed,n));
    h3_gpu_tensor *sample=h3_gpu_tensor_new_f32(gpu,n+19); CHECK(sample);
    CHECK(h3_gpu_tensor_write_f32_range(sample,19,packed,n)); CHECK(h3_gpu_synchronize(gpu));
    CHECK(h3_gpu_tensor_read_f32_range(sample,19,readback,n)); CHECK(!memcmp(packed,readback,sizeof(packed)));
    CHECK(h3_dit_unpatchify_video(readback,24,3,4,6,unpacked,n)); CHECK(!memcmp(video,unpacked,sizeof(video)));
    h3_gpu_tensor_free(sample);

    h3_sampler_state state={0}; CHECK(h3_serving_schedule_build(2,&state.sigmas));
    h3_dit_schedule *plan=h3_dit_schedule_plan(&state.sigmas,0,0,error,sizeof(error)); CHECK(plan);
    size_t rows=h3_dit_schedule_time_rows(plan); h3_dit_schedule_free(plan);
    state.prepared.version=1; state.prepared.count=51;
    for(unsigned i=0;i<51;i++) {
        h3_prepared_tensor *t=&state.prepared.tensors[i]; t->id=i==50?2:100+i;
        t->elements=rows*(i==50?2:18)*5376; t->values=malloc(t->elements*2); CHECK(t->values);
        for(size_t j=0;j<t->elements;j++) t->values[j]=(uint16_t)(j*71+i);
    }
    h3_dit_schedule *imported=h3_dit_schedule_import(gpu,&state); CHECK(imported);
    h3_sampler_state exported={0}; CHECK(h3_dit_schedule_export(imported,&exported)); CHECK(exported.prepared.count==51);
    for(size_t i=0;i<51;i++) {
        const h3_prepared_tensor *x=&state.prepared.tensors[i],*y=&exported.prepared.tensors[i];
        CHECK(x->id==y->id && x->elements==y->elements && !memcmp(x->values,y->values,x->elements*2));
    }
    h3_dit_schedule_free(imported); h3_prepared_cache_free(&exported.prepared);
    state.prepared.tensors[10].elements--; CHECK(!h3_dit_schedule_import(gpu,&state));
    state.prepared.tensors[10].elements++; state.prepared.version=2; CHECK(!h3_dit_schedule_import(gpu,&state));
    h3_prepared_cache_free(&state.prepared); h3_gpu_free(gpu);
    printf("ok: %d sampler GPU checks; all 65536 BF16 words preserved, packed F32 exact, 51 AdaLN tensors exact\n",checks);
    return 0;
}
