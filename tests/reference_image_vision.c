/* Real vision completion plus cancellation/admission cleanup; no oracle. */
#include "src/conditioning/vision_encoder.h"
#include "src/sglang/sglang.h"
#include "src/memory.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[512];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error);exit(1);}}while(0)
static int cancel(int layer,int total,void *p){(void)total;return layer>=*(int*)p;}
static int empty(uint64_t *bytes,void *p){(void)p;*bytes=0;return 1;}
static int progress(int layer,int total,void *p){(void)p;fprintf(stderr,"vision layer %d/%d\n",layer,total);return 0;}
static void validate(h3_vision_output *out,int w,int h) {
    CHECK(out->grid_w==w/16&&out->grid_h==h/16&&out->tokens==(size_t)w*h/1024);
    const uint16_t *t[]={out->merged,out->deepstack[0],out->deepstack[1],out->deepstack[2]};
    for(int i=0;i<4;i++){CHECK(t[i]);for(size_t j=0;j<out->tokens*5120;j++)CHECK((t[i][j]&0x7f80)!=0x7f80);}
    printf("vision %dx%d patches=%d tokens=%zu peak=%llu finite PASS\n",w,h,(w/16)*(h/16),out->tokens,(unsigned long long)out->gpu_stats.peak_live_bytes);fflush(stdout);
    h3_vision_output_free(out);CHECK(!out->tokens&&!out->merged&&!out->deepstack[0]);
}
int main(int argc,char **argv) {
    if(argc!=4){fprintf(stderr,"usage: reference_image_vision WEIGHTS WIDTH HEIGHT\n");return 2;}
#ifndef __APPLE__
    h3_sglang_exchange(H3_SGLANG_VERSION);
#endif
    int w=atoi(argv[2]),h=atoi(argv[3]);CHECK(w>=32&&h>=32&&w<=8192&&h<=8192);
    float *pixels=calloc((size_t)w*h*3,4);CHECK(pixels);
    h3_vision_output out={0};h3_memory_set_test_query(empty,NULL);
    CHECK(!h3_vision_encode_bf16(argv[1],"src/metal/shaders.metal",pixels,1,32,32,NULL,NULL,&out,error,sizeof(error)));
    CHECK(h3_memory_error(error)&&!out.merged&&!out.tokens);h3_memory_set_test_query(NULL,NULL);
    for(int stop=0;stop<=1;stop++) {
        CHECK(!h3_vision_encode_bf16(argv[1],"src/metal/shaders.metal",pixels,1,32,32,cancel,&stop,&out,error,sizeof(error)));
        CHECK(strstr(error,"cancel")&&!out.merged&&!out.tokens);
    }
    for(int i=0;i<3;i++) {
        int iw=i==1?w:32,ih=i==1?h:32;
        CHECK(h3_vision_encode_bf16(argv[1],"src/metal/shaders.metal",pixels,1,ih,iw,progress,NULL,&out,error,sizeof(error)));
        validate(&out,iw,ih);
    }
    free(pixels);puts("PASS: real vision small/large/small, memory rejection and cancellation cleanup");return 0;
}
