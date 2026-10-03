#include "src/gpu.h"
#include "src/weights/quant.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);return 1;}}while(0)
static uint16_t bf(float x){uint32_t v;memcpy(&v,&x,4);v+=0x7fff+((v>>16)&1);return (uint16_t)(v>>16);}
static float fp(uint16_t x){uint32_t v=(uint32_t)x<<16;float f;memcpy(&f,&v,4);return f;}
int main(int argc,char **argv) {
    char error[512]={0};h3_gpu *g=NULL;int mode=0;
    if(argc<3||!h3_quant_parse(argv[1],&mode)||!mode)return 2;
    h3_sglang_exchange(4);
    CHECK(h3_quant_preflight(mode,error,sizeof(error)));
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    /* Admission takes complete H3 block geometry; actual probe tensors stay small. */
    const uint64_t core_weights=2ull*(4ull*7168*5376+3ull*14336*5376);
    CHECK(h3_gpu_quant_configure(g,mode,argv[2],core_weights,1<<20,0));
    if(argc>3&&!strcmp(argv[3],"bad-range")){
        CHECK(!h3_gpu_quant_load(g,"/dev/null",1,256,256));CHECK(strstr(h3_gpu_error(g),"range"));h3_gpu_free(g);return 0;
    }
    if(argc>3&&!strcmp(argv[3],"bad-shape")){
        CHECK(!h3_gpu_quant_load(g,"/dev/null",0,255,256));CHECK(strstr(h3_gpu_error(g),"shape"));h3_gpu_free(g);return 0;
    }
    unsigned n=256,k=256,rows=129;
    uint16_t *w=malloc((size_t)n*k*2),*x=malloc((size_t)rows*k*2),*y=malloc((size_t)rows*n*2);
    CHECK(w&&x&&y);
    for(unsigned j=0;j<n;j++)for(unsigned c=0;c<k;c++)
        w[(size_t)j*k+c]=bf(ldexpf(1,(int)(j%3)-1+(int)((c/16)%3)));
    for(unsigned r=0;r<rows;r++)for(unsigned c=0;c<k;c++)
        x[(size_t)r*k+c]=bf(ldexpf((r%2)?-1:1,(int)(r%3)-1+(int)((c/16)%2)));
    if(argc>3&&!strcmp(argv[3],"zero-weight"))memset(w,0,(size_t)n*k*2);
    if(argc>3&&!strcmp(argv[3],"source-shift"))for(size_t i=0;i<(size_t)n*k;i++)w[i]=bf(fp(w[i])*2);
    char path[]="/tmp/h3-quant-weight-XXXXXX";int fd=mkstemp(path);CHECK(fd>=0);
    FILE *file=fdopen(fd,"wb");CHECK(file);CHECK(fwrite(w,2,(size_t)n*k,file)==(size_t)n*k);CHECK(!fclose(file));
    h3_gpu_tensor *weight=h3_gpu_quant_load(g,path,0,n,k);CHECK(weight);
    h3_gpu_tensor *input=h3_gpu_tensor_from_bf16(g,x,(size_t)rows*k),*output=h3_gpu_tensor_new_bf16(g,(size_t)rows*n);CHECK(input&&output);
    unsigned cases[]={1,17,129,3};
    for(unsigned c=0;c<4;c++) {
        unsigned count=cases[c];CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_linear_bf16(g,output,input,weight,NULL,count,k,n));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_bf16(output,y,(size_t)count*n));
        for(unsigned r=0;r<count;r++)for(unsigned j=0;j<n;j++) {
            float expected=0;for(unsigned i=0;i<k;i++)expected+=fp(x[(size_t)r*k+i])*fp(w[(size_t)j*k+i]);
            if(y[(size_t)r*n+j]!=bf(expected)){fprintf(stderr,"pattern mismatch %s row=%u col=%u got=%g expected=%g\n",argv[1],r,j,fp(y[(size_t)r*n+j]),fp(bf(expected)));return 1;}
        }
    }
    h3_gpu_tensor_free(weight);weight=h3_gpu_quant_load(g,path,0,n,k);CHECK(weight);
    CHECK(h3_gpu_begin(g));CHECK(h3_gpu_linear_bf16(g,output,input,weight,NULL,rows,k,n));CHECK(h3_gpu_submit(g));
    memset(x,0,(size_t)rows*k*2);
    h3_gpu_tensor_free(input);input=h3_gpu_tensor_from_bf16(g,x,(size_t)rows*k);CHECK(input);
    CHECK(h3_gpu_begin(g));CHECK(h3_gpu_linear_bf16(g,output,input,weight,NULL,rows,k,n));CHECK(h3_gpu_submit(g));
    CHECK(h3_gpu_tensor_read_bf16(output,y,(size_t)rows*n));
    for(size_t i=0;i<(size_t)rows*n;i++)CHECK((y[i]&0x7fff)==0);
    if(getenv("H3_QUANT_DIAGNOSTICS")){
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
        CHECK(!stats.quant_zeroed_values); /* All nonzero pattern inputs are exactly representable. */
        printf("packing diagnostics clipped=%llu zeroed=%llu\n",(unsigned long long)stats.quant_clipped_values,(unsigned long long)stats.quant_zeroed_values);
    }
    if(argc>3&&!strcmp(argv[3],"nonfinite")) {
        x[0]=0x7fc0;h3_gpu_tensor_free(input);input=h3_gpu_tensor_from_bf16(g,x,(size_t)rows*k);CHECK(input);
        CHECK(h3_gpu_begin(g));CHECK(h3_gpu_linear_bf16(g,output,input,weight,NULL,rows,k,n));
        CHECK(!h3_gpu_submit(g));CHECK(strstr(h3_gpu_error(g),"nonfinite"));
        h3_gpu_cancel(g);x[0]=0;h3_gpu_tensor_free(input);
        input=h3_gpu_tensor_from_bf16(g,x,(size_t)rows*k);CHECK(input);
        CHECK(h3_gpu_begin(g));CHECK(h3_gpu_linear_bf16(g,output,input,weight,NULL,rows,k,n));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_bf16(output,y,(size_t)rows*n));
        for(size_t i=0;i<(size_t)rows*n;i++)CHECK((y[i]&0x7fff)==0);
    }
    h3_gpu_tensor_free(weight);h3_gpu_tensor_free(input);h3_gpu_tensor_free(output);
    unlink(path);free(w);free(x);free(y);h3_gpu_free(g);
    printf("PASS native %s: exact representable pattern, signs, tiled scales, 1/17/129/3 row tails, plan reuse and artifact reuse\n",argv[1]);return 0;
}
