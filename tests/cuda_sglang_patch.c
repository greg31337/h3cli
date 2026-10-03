/* Projection-only regression: no models, VAE, attention or denoising. Compare
 * the bounded patch/scatter path against the original unbatched FP32 addmm. */
#include "src/gpu.h"
#include "src/sglang/sglang.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char error[1024];
static h3_gpu *gpu;
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAIL %d: %s: %s %s\n",__LINE__,#x,error,gpu?h3_gpu_error(gpu):"");exit(1); } } while(0)
static double now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec/1e9; }
static uint16_t bf16(float x) { uint32_t u;memcpy(&u,&x,4);return (uint16_t)((u+0x7fff+((u>>16)&1))>>16); }
static uint32_t random_state=723;
static float value(void) {
    random_state=random_state*1664525u+1013904223u;
    return ((float)(random_state>>8)/16777216.f-.5f)*.3f;
}

static void run(unsigned rows,unsigned k,int mapped) {
    const unsigned n=5376;
    const size_t io=mapped?0:(size_t)2*k,oo=mapped?0:7;
    unsigned output_rows=mapped?rows+(rows-1)/257+2:rows;
    size_t count=(size_t)rows*n,output_count=(size_t)output_rows*n+2*oo;
    float *input=malloc(((size_t)rows*k+io)*4),*weight=malloc((size_t)k*n*4),*bias=malloc(n*4);
    uint32_t *map=malloc((size_t)rows*4);
    uint16_t *expected=malloc(output_count*2),*got=malloc(output_count*2);
    const size_t block_rows=512;float *block=malloc(block_rows*n*4);
    CHECK(input&&weight&&bias&&map&&expected&&got&&block);
    for(size_t i=0;i<(size_t)rows*k+io;i++)input[i]=value();
    for(size_t i=0;i<(size_t)k*n;i++)weight[i]=value();
    for(unsigned i=0;i<n;i++)bias[i]=value();
    for(unsigned r=0;r<rows;r++){unsigned reverse=rows-1-r;map[r]=1+reverse+reverse/257;}
    for(size_t i=0;i<output_count;i++)expected[i]=0x3b00;
    int old=h3_sglang_exchange(H3_SGLANG_VERSION);
    gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));h3_sglang_exchange(old);CHECK(gpu);
    h3_gpu_tensor *x=h3_gpu_tensor_from_f32(gpu,input+io,(size_t)rows*k),
        *w=h3_gpu_tensor_from_f32(gpu,weight,(size_t)k*n),*b=h3_gpu_tensor_from_f32(gpu,bias,n),
        *gold=h3_gpu_tensor_new_f32(gpu,count),*y=h3_gpu_tensor_from_bf16(gpu,expected,output_count),
        *index=mapped?h3_gpu_tensor_from_u32(gpu,map,rows):NULL;
    CHECK(x&&w&&b&&gold&&y&&(!mapped||index));
    /* Unbatched, pinned cuBLAS recipe, including the original row geometry. */
    CHECK(h3_gpu_begin(gpu)&&h3_gpu_linear_f32(gpu,gold,x,w,b,rows,k,n)&&h3_gpu_submit(gpu));
    for(size_t row=0;row<rows;row+=block_rows) {
        size_t nr=rows-row<block_rows?rows-row:block_rows;
        CHECK(h3_gpu_tensor_read_f32_range(gold,row*n,block,nr*n));
        for(size_t r=0;r<nr;r++) {
            size_t dest=mapped?(size_t)map[row+r]*n:oo+(row+r)*n;
            for(unsigned col=0;col<n;col++)expected[dest+col]=bf16(block[r*n+col]);
        }
    }
    h3_gpu_tensor_free(gold);
    if(io){h3_gpu_tensor_free(x);x=h3_gpu_tensor_from_f32(gpu,input,(size_t)rows*k+io);CHECK(x);}
    free(input);free(weight);free(bias);free(block);
    h3_gpu_stats before,after;CHECK(h3_gpu_get_stats(gpu,&before));
    double start=now();uint64_t scratch=0;
    for(int repeat=0;repeat<2;repeat++) {
        CHECK(h3_gpu_begin(gpu));
        CHECK(mapped?h3_gpu_patch_linear_bf16_map(gpu,y,x,w,b,index,output_rows,rows,k,n):
            h3_gpu_patch_linear_bf16_offset(gpu,y,oo,x,io,w,b,rows,k,n));
        CHECK(h3_gpu_submit(gpu)&&h3_gpu_tensor_read_bf16(y,got,output_count));
        size_t differences=0;
        for(size_t j=0;j<output_count;j++)if(got[j]!=expected[j]) {
            if(differences<4)fprintf(stderr,"rows=%u mapped=%d element=%zu got=%04x expected=%04x\n",rows,mapped,j,got[j],expected[j]);
            differences++;
        }
        CHECK(!differences); /* Includes nonzero offsets and untouched map gaps. */
        CHECK(h3_gpu_get_stats(gpu,&after));
        if(!repeat)scratch=after.live_bytes-before.live_bytes;
        CHECK(scratch && scratch<=UINT64_C(1073741824));
        CHECK(after.live_bytes==before.live_bytes+scratch);
    }
    /* Keep dtype validation separate from geometry/capacity errors. */
    CHECK(h3_gpu_begin(gpu));
    CHECK(!(mapped?h3_gpu_patch_linear_bf16_map(gpu,y,x,w,NULL,index,output_rows,rows,k,n):
        h3_gpu_patch_linear_bf16_offset(gpu,y,oo,x,io,w,NULL,rows,k,n)));
    CHECK(strstr(h3_gpu_error(gpu),"FP32 operands and bias"));h3_gpu_cancel(gpu);
    printf("{\"rows\":%u,\"k\":%u,\"width\":%u,\"mapped\":%s,\"unbatched_fp32_bytes\":%zu,\"scratch_bytes\":%llu,\"repeated_seconds\":%.6f,\"exact_bf16\":true,\"passed\":true}\n",
        rows,k,n,mapped?"true":"false",count*4,(unsigned long long)scratch,now()-start);
    fflush(stdout);
    h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);h3_gpu_tensor_free(b);h3_gpu_tensor_free(y);h3_gpu_tensor_free(index);
    h3_gpu_free(gpu);gpu=NULL;free(map);free(expected);free(got);
}
int main(void) {
    run(257,96,1);run(257,32,0); /* Existing video/audio small paths. */
    run(49932,96,0);             /* Last row count below the old limit. */
    run(49933,96,1);             /* First rejected count; thirteen-row tail. */
    run(107856+5440,96,1);       /* 1344x768x362 plus a 2048x2720 reference. */
    run(107856,96,0);            /* Same target without mapped references. */
    return 0;
}
