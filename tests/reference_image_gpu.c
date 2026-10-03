/* Completion/coverage/guards only. No numerical oracle or tensor comparison. */
#include "src/gpu.h"
#include "src/sglang/sglang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static h3_gpu *g;static char error[512];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s (%s %s)\n",__LINE__,#x,error,g?h3_gpu_error(g):"");exit(1);}}while(0)
#ifndef __APPLE__
static void projection(unsigned rows,int failure) {
    size_t ni=(size_t)rows*1536,no=(size_t)rows*1152,nw=1536*1152;
    uint16_t *in=malloc(ni*2),*out=malloc((no+128)*2),*w=malloc(nw*2),bias[1152];
    CHECK(in&&out&&w);
    for(size_t i=0;i<ni;i++)in[i]=0x3d00+(i%151);
    for(size_t i=0;i<nw;i++)w[i]=0x3800+(i%37);
    for(size_t i=0;i<no+128;i++)out[i]=0x7fc1;
    memset(bias,0,sizeof(bias));
    h3_gpu_stats base,before,after;CHECK(h3_gpu_get_stats(g,&base));
    h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,in,ni),*y=h3_gpu_tensor_from_bf16(g,out,no+128),
        *weight=h3_gpu_tensor_from_bf16(g,w,nw),*b=h3_gpu_tensor_from_bf16(g,bias,1152);
    CHECK(x&&y&&weight&&b);CHECK(h3_gpu_get_stats(g,&before));CHECK(h3_gpu_begin(g));
    int ok=h3_gpu_sglang_vision_patch(g,y,x,weight,b,rows,1536,1152);
    CHECK(ok==!failure);
    if(failure){CHECK(strstr(h3_gpu_error(g),"test budget"));h3_gpu_cancel(g);}
    else {CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(y,out,no+128));
        for(size_t i=0;i<no;i++)CHECK((out[i]&0x7f80)!=0x7f80);
        for(size_t i=no;i<no+128;i++)CHECK(out[i]==0x7fc1);
    }
    CHECK(h3_gpu_get_stats(g,&after));CHECK(before.live_bytes==after.live_bytes);
    CHECK(after.peak_live_bytes<=before.peak_live_bytes || after.peak_live_bytes<=before.live_bytes+(UINT64_C(530)<<20));
    h3_gpu_tensor_free(x);h3_gpu_tensor_free(y);h3_gpu_tensor_free(weight);h3_gpu_tensor_free(b);
    CHECK(h3_gpu_get_stats(g,&after));CHECK(after.live_bytes==base.live_bytes);
    printf("projection rows=%u injected_failure=%d live=%llu peak=%llu PASS\n",rows,failure,
        (unsigned long long)after.live_bytes,(unsigned long long)after.peak_live_bytes);fflush(stdout);
    free(in);free(out);free(w);
}
static void attention_groups(void) {
    uint32_t large[]={65536,65536},small[]={4,8},bad[]={3};
    CHECK(h3_gpu_sglang_vision_groups(g,large,2));
    CHECK(h3_gpu_sglang_vision_groups(g,small,2));
    uint16_t *data=calloc(12*1152+128,2);CHECK(data);
    h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,data,12*1152),*y=h3_gpu_tensor_new_bf16(g,12*1152+128);CHECK(x&&y);
    for(int i=0;i<12*1152+128;i++)data[i]=0x7fc1;
    for(int pass=0;pass<3;pass++) {
        CHECK(h3_gpu_tensor_write_bf16(y,data,12*1152+128));
        if(pass==1){CHECK(!h3_gpu_sglang_vision_groups(g,bad,1));}
        if(pass==2){CHECK(h3_gpu_sglang_vision_groups(g,large,2));h3_gpu_cancel(g);}
        h3_gpu_stats a,b;CHECK(h3_gpu_get_stats(g,&a));CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_sdpa_bf16(g,y,x,x,x,12,16,72,0.1f));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_get_stats(g,&b));CHECK(b.attention_dispatches-a.attention_dispatches==(pass?1:2));
        CHECK(h3_gpu_tensor_read_bf16(y,data,12*1152+128));
        for(int i=0;i<12*1152;i++)CHECK((data[i]&0x7f80)!=0x7f80);
        for(int i=12*1152;i<12*1152+128;i++)CHECK(data[i]==0x7fc1);
        for(int i=0;i<12*1152+128;i++)data[i]=0x7fc1;
    }
    CHECK(h3_gpu_sglang_vision_groups(g,NULL,0));h3_gpu_tensor_free(x);h3_gpu_tensor_free(y);free(data);
    puts("group replacement, failed setter, cancellation and recovery PASS");
}
#endif
int main(void) {
#ifdef __APPLE__
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    h3_gqa_limits limits;CHECK(h3_gpu_gqa_causal_limits(g,&limits));
    unsigned lengths[]={16,(unsigned)limits.direct_max_sequence+1,16385,16};
    for(size_t c=0;c<sizeof(lengths)/sizeof(*lengths);c++) {
        unsigned rows=lengths[c];size_t count=(size_t)rows*2*128;
        uint16_t *data=malloc((count+128)*2);CHECK(data);
        for(size_t i=0;i<count;i++)data[i]=(uint16_t)(0x3c00+i%255);
        h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,data,count);
        for(size_t i=0;i<count+128;i++)data[i]=0x7fc1;
        h3_gpu_tensor *y=h3_gpu_tensor_from_bf16(g,data,count+128);CHECK(x&&y);
        CHECK(h3_gpu_begin(g));h3_gpu_cancel(g);CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_gqa_causal_bf16(g,y,x,x,x,rows,2,1,128,0.08838835f));
        CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(y,data,count+128));
        for(size_t i=0;i<count;i++)CHECK((data[i]&0x7f80)!=0x7f80);
        for(size_t i=count;i<count+128;i++)CHECK(data[i]==0x7fc1);
        h3_gpu_tensor_free(x);h3_gpu_tensor_free(y);free(data);
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));CHECK(stats.live_bytes==0);
        printf("Metal GQA rows=%u full writes, finite, guards, cancellation recovery PASS\n",rows);fflush(stdout);
    }
    /* D=72 packing exercises its padded head tail and the full 65,536-key
     * attention sequence. A second small request reuses this same context. */
    const unsigned vision_rows[]={65536,32};
    for(size_t c=0;c<2;c++) {
        size_t count=(size_t)vision_rows[c]*72;
        uint16_t *data=malloc((count+128)*2);CHECK(data);
        for(size_t i=0;i<count;i++)data[i]=(uint16_t)(0x3c00+i%255);
        h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,data,count);
        for(size_t i=0;i<count+128;i++)data[i]=0x7fc1;
        h3_gpu_tensor *y=h3_gpu_tensor_from_bf16(g,data,count+128);CHECK(x&&y);
        CHECK(h3_gpu_begin(g));CHECK(h3_gpu_sdpa_bf16(g,y,x,x,x,vision_rows[c],1,72,0.11785113f));
        CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(y,data,count+128));
        for(size_t i=0;i<count;i++)CHECK((data[i]&0x7f80)!=0x7f80);
        for(size_t i=count;i<count+128;i++)CHECK(data[i]==0x7fc1);
        h3_gpu_tensor_free(x);h3_gpu_tensor_free(y);free(data);
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));CHECK(stats.live_bytes==0);
        printf("Metal vision attention rows=%u full writes, finite, guards, cleanup PASS\n",vision_rows[c]);fflush(stdout);
    }
    h3_gpu_free(g);g=NULL;
#else
    h3_sglang_exchange(H3_SGLANG_VERSION);
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    const unsigned rows[]={4,32764,32768,32772,65540,4};
    for(size_t i=0;i<sizeof(rows)/sizeof(*rows);i++)projection(rows[i],0);
    attention_groups();h3_gpu_free(g);g=NULL;
    CHECK(!setenv("H3_CUDA_TEST_MEMORY_BUDGET","440401920",1));
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    projection(65540,1);projection(4,0);h3_gpu_free(g);g=NULL;
    unsetenv("H3_CUDA_TEST_MEMORY_BUDGET");
#endif
    puts("PASS: reference image GPU completion and lifecycle");return 0;
}
