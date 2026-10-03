#include "src/metal/ane_split.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static uint16_t bf(float f){uint32_t b;memcpy(&b,&f,4);b+=0x7fff+((b>>16)&1);return (uint16_t)(b>>16);}
static float unbf(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#x,g?h3_gpu_error(g):error);return 1;}}while(0)
int main(void) {
    const unsigned rows=1025,k=1024,n=1024;char error[4096]={0};
    h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    uint16_t *x=malloc((size_t)rows*k*2),*w=malloc((size_t)n*k*2),*ref=malloc((size_t)rows*n*2),*saved=malloc((size_t)rows*n*2),*actual=malloc((size_t)rows*n*2);CHECK(x&&w&&ref&&saved&&actual);
    for(size_t i=0;i<(size_t)rows*k;i++)x[i]=bf(sinf((float)(i%997)*.17f)*3.f);
    for(size_t i=0;i<(size_t)n*k;i++)w[i]=bf(cosf((float)(i%991)*.11f)*.03f);
    h3_gpu_tensor *tx=h3_gpu_tensor_from_bf16(g,x,(size_t)rows*k),*tw=h3_gpu_tensor_from_bf16(g,w,(size_t)n*k),*out=h3_gpu_tensor_new_bf16(g,(size_t)rows*n+128);CHECK(tx&&tw&&out);
    uint16_t *guard=(uint16_t *)h3_gpu_tensor_contents(out)+(size_t)rows*n;
    for(unsigned j=0;j<128;j++)guard[j]=0x55aa;
    CHECK(h3_gpu_begin(g));CHECK(h3_gpu_linear_bf16(g,out,tx,tw,NULL,rows,k,n));CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(out,ref,(size_t)rows*n));
    h3_metal_attention_options options=H3_METAL_ATTENTION_DEFAULT;
    options.precision=1;options.candidate=1;options.ane_rows=512;options.ane_chunk=256;
    for(int mode=1;mode<=3;mode++) {
        options.ane_mode=mode;h3_ane_split *s=h3_ane_split_create(options,k,n);CHECK(s);
        CHECK(!h3_ane_split_linear(s,g,out,tx,tw,rows+1,1,0));
        CHECK(!h3_ane_split_linear(s,g,out,tx,tw,rows,50,0));
        if(mode==3){CHECK(h3_gpu_begin(g));CHECK(h3_ane_split_linear(s,g,out,tx,tw,rows,0,0));CHECK(h3_gpu_submit(g));}
        CHECK(h3_gpu_begin(g));CHECK(h3_ane_split_linear(s,g,out,tx,tw,rows,1,0));CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(out,actual,(size_t)rows*n));
        double num=0,den=0;for(size_t i=0;i<(size_t)rows*n;i++){double r=unbf(ref[i]),v=unbf(actual[i]);CHECK(isfinite(v));num+=(v-r)*(v-r);den+=r*r;}
        double rel=sqrt(num/den);CHECK(rel<.01);
        if(mode==1)memcpy(saved,actual,(size_t)rows*n*2);
        else CHECK(!memcmp(saved,actual,(size_t)rows*n*2));
        uint64_t decided=0;uint32_t plan[50]={0};h3_ane_split_export(s,&decided,plan);CHECK((decided&2)&&plan[1]==512);
        h3_ane_split *restored=h3_ane_split_create(options,k,n);CHECK(restored);CHECK(h3_ane_split_restore(restored,decided,plan));
        CHECK(!h3_ane_split_restore(restored,decided|(1ull<<50),plan));
        plan[1]++;CHECK(!h3_ane_split_restore(restored,decided,plan));plan[1]--;
        CHECK(h3_gpu_begin(g));CHECK(h3_ane_split_linear(restored,g,out,tx,tw,rows,1,1));CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(out,actual,(size_t)rows*n));CHECK(!memcmp(saved,actual,(size_t)rows*n*2));
        h3_ane_split_free(restored);h3_ane_split_free(s);
        for(unsigned j=0;j<128;j++)CHECK(guard[j]==0x55aa);
        printf("{\"case\":\"split-and-resume\",\"mode\":%d,\"relative_l2\":%.9g,\"pass\":true}\n",mode,rel);
    }
    /* A rejected conversion must leave ANE output unpublished and run the
     * complete GPU projection. Tiny BF16 values are lost in unscaled FP16. */
    for(size_t i=0;i<(size_t)rows*k;i++)x[i]=bf(1e-30f);
    CHECK(h3_gpu_tensor_write_bf16(tx,x,(size_t)rows*k));CHECK(h3_gpu_begin(g));CHECK(h3_gpu_linear_bf16(g,out,tx,tw,NULL,rows,k,n));CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(out,ref,(size_t)rows*n));
    options.ane_mode=2;h3_ane_split *s=h3_ane_split_create(options,k,n);CHECK(s);
    CHECK(h3_gpu_begin(g));CHECK(h3_ane_split_linear(s,g,out,tx,tw,rows,1,0));CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(out,actual,(size_t)rows*n));CHECK(!memcmp(ref,actual,(size_t)rows*n*2));
    h3_ane_split_free(s);puts("{\"case\":\"range-recovery\",\"pass\":true}");
    h3_gpu_tensor_free(tx);h3_gpu_tensor_free(tw);h3_gpu_tensor_free(out);h3_gpu_free(g);free(x);free(w);free(ref);free(saved);free(actual);return 0;
}
