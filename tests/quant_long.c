/* Small operator tensors exercise real 8192-row slabs; no movie allocations.
 * Full production N/K and maximum logical M are queried as descriptors only. */
#include "src/gpu.h"
#include "src/weights/quant.h"
#include "src/weights/quant_cache.h"
#include "src/execution.h"
#include "src/sglang/sglang.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static char error[1024];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
static uint16_t bf(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fff+((u>>16)&1);return (uint16_t)(u>>16);}
static float fp(uint16_t x){uint32_t u=(uint32_t)x<<16;float f;memcpy(&f,&u,4);return f;}
static void gpu_check(h3_gpu *g,int ok){snprintf(error,sizeof(error),"%s",h3_gpu_error(g));CHECK(ok);}
static h3_gpu *context(int bounded,const char *cache,unsigned rows) {
    h3_params p=H3_PARAMS_DEFAULT;p.frames=bounded?464:22;p.width=p.height=256;p.cuda_denoise_quant=2;
    h3_cuda_policy policy;CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));
    h3_cuda_policy_exchange(policy);h3_sglang_exchange(4);
    h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    gpu_check(g,h3_gpu_quant_configure(g,2,cache,2ull*(4ull*7168*5376+3ull*14336*5376),(uint64_t)rows*358400,0));
    return g;
}
static void compare(const uint16_t *a,const uint16_t *b,size_t count,const char *label) {
    double squared=0,energy=0;size_t changed=0;
    for(size_t i=0;i<count;i++) {
        double x=fp(a[i]),y=fp(b[i]);CHECK(isfinite(x)&&isfinite(y));
        squared+=(x-y)*(x-y);energy+=x*x;changed+=a[i]!=b[i];
        /* Distinct cuBLASLt M descriptors may differ by a final BF16 rounding.
         * This is an NVFP4-to-NVFP4 operator contract, not a golden tolerance. */
        if(fabs(x-y)>.008*fmax(fabs(x),fabs(y))+.002) {
            fprintf(stderr,"%s mismatch %zu: %g %g\n",label,i,x,y);exit(1);
        }
    }
    CHECK(squared<=1.e-6*fmax(energy,1.));
    printf("%s count=%zu changed=%zu relative_rmse=%.9g\n",label,count,changed,sqrt(squared/fmax(energy,1.)));
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    enum{ROWS=H3_QUANT_LONG_ROWS+17,WIDTH=128,HIDDEN=512};
    h3_gpu *legacy=context(0,argv[1],ROWS),*bounded=context(1,argv[1],ROWS);
    const uint64_t logical[]={1,127,128,129,8192,8193,32674,133446,333282,435762,876426,1000000};
    for(size_t i=0;i<sizeof(logical)/sizeof(*logical);i++)gpu_check(bounded,h3_gpu_quant_shape_preflight(bounded,logical[i]));
    CHECK(!h3_gpu_quant_shape_preflight(bounded,1000001));
    const unsigned nk[][2]={{768,128},{128,256},{1024,128},{128,512}};
    char paths[4][64];h3_gpu_tensor *weight[2][4];
    for(unsigned j=0;j<4;j++) {
        size_t count=(size_t)nk[j][0]*nk[j][1];uint16_t *w=malloc(count*2);CHECK(w);
        for(size_t i=0;i<count;i++)w[i]=bf((float)((int)((i*13+i/127)%31)-15)/128);
        snprintf(paths[j],sizeof(paths[j]),"/tmp/h3-quant-long-%u-XXXXXX",j);
        int fd=mkstemp(paths[j]);CHECK(fd>=0);FILE *f=fdopen(fd,"wb");CHECK(f);
        CHECK(fwrite(w,2,count,f)==count&&!fclose(f));free(w);
        weight[0][j]=h3_gpu_quant_load(legacy,paths[j],0,nk[j][0],nk[j][1]);CHECK(weight[0][j]);
        weight[1][j]=h3_gpu_quant_load(bounded,paths[j],0,nk[j][0],nk[j][1]);CHECK(weight[1][j]);
    }
    const unsigned counts[]={1,17,8193,ROWS};
    for(unsigned family=0;family<4;family++) {
        unsigned n=nk[family][0],k=nk[family][1];
        uint16_t *x=malloc((size_t)ROWS*k*2),*y[2]={malloc((size_t)ROWS*n*2),malloc((size_t)ROWS*n*2)};CHECK(x&&y[0]&&y[1]);
        /* Place the global outlier in the final chunk to detect per-chunk amax. */
        for(size_t i=0;i<(size_t)ROWS*k;i++)x[i]=bf((float)((int)(i%97)-48)/64*(i/k>=8192?3.7f:1.f));
        for(size_t c=0;c<sizeof(counts)/sizeof(*counts);c++) {
            for(unsigned mode=0;mode<2;mode++) {
                h3_gpu *g=mode?bounded:legacy;
                h3_gpu_tensor *in=h3_gpu_tensor_from_bf16(g,x,(size_t)ROWS*k),*out=h3_gpu_tensor_new_bf16(g,(size_t)ROWS*n);CHECK(in&&out);
                gpu_check(g,h3_gpu_begin(g)&&h3_gpu_linear_bf16(g,out,in,weight[mode][family],NULL,counts[c],k,n)&&h3_gpu_submit(g));
                gpu_check(g,h3_gpu_tensor_read_bf16(out,y[mode],(size_t)counts[c]*n));
                h3_gpu_tensor_free(in);h3_gpu_tensor_free(out);
            }
            compare(y[0],y[1],(size_t)counts[c]*n,"projection");
        }
        free(x);free(y[0]);free(y[1]);
    }
    uint16_t *x=malloc((size_t)ROWS*WIDTH*2),*y[2]={malloc((size_t)ROWS*WIDTH*2),malloc((size_t)ROWS*WIDTH*2)};CHECK(x&&y[0]&&y[1]);
    for(size_t i=0;i<(size_t)ROWS*WIDTH;i++)x[i]=bf((float)((int)(i%83)-41)/64*(i/WIDTH>=8192?2.7f:1.f));
    for(unsigned mode=0;mode<2;mode++) {
        h3_gpu *g=mode?bounded:legacy;
        h3_gpu_tensor *in=h3_gpu_tensor_from_bf16(g,x,(size_t)ROWS*WIDTH),*out=h3_gpu_tensor_new_bf16(g,(size_t)ROWS*WIDTH);CHECK(in&&out);
        h3_gpu_tensor *fc1=NULL,*activated=NULL;
        if(!mode){fc1=h3_gpu_tensor_new_bf16(g,(size_t)ROWS*HIDDEN*2);activated=h3_gpu_tensor_new_bf16(g,(size_t)ROWS*HIDDEN);CHECK(fc1&&activated);}
        uint64_t plateau=0;
        for(int repeat=0;repeat<3;repeat++) {
            gpu_check(g,h3_gpu_begin(g));
            int ok=mode?h3_gpu_mlp_bf16(g,out,in,weight[mode][2],weight[mode][3],ROWS,WIDTH,HIDDEN,WIDTH):
                h3_gpu_linear_bf16(g,fc1,in,weight[mode][2],NULL,ROWS,WIDTH,2*HIDDEN)&&
                h3_gpu_swiglu_bf16(g,activated,fc1,ROWS,HIDDEN)&&
                h3_gpu_linear_bf16(g,out,activated,weight[mode][3],NULL,ROWS,HIDDEN,WIDTH);
            gpu_check(g,ok&&h3_gpu_submit(g));gpu_check(g,h3_gpu_tensor_read_bf16(out,y[mode],(size_t)ROWS*WIDTH));
            h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
            if(repeat)CHECK(stats.live_bytes==plateau);plateau=stats.live_bytes;
        }
        if(mode) {
            x[0]=0x7fc0;h3_gpu_tensor_free(in);in=h3_gpu_tensor_from_bf16(g,x,(size_t)ROWS*WIDTH);CHECK(in);
            gpu_check(g,h3_gpu_begin(g)&&h3_gpu_mlp_bf16(g,out,in,weight[mode][2],weight[mode][3],ROWS,WIDTH,HIDDEN,WIDTH));
            CHECK(!h3_gpu_submit(g)&&strstr(h3_gpu_error(g),"nonfinite"));h3_gpu_cancel(g);
            memset(x,0,(size_t)ROWS*WIDTH*2);h3_gpu_tensor_free(in);in=h3_gpu_tensor_from_bf16(g,x,(size_t)ROWS*WIDTH);CHECK(in);
            gpu_check(g,h3_gpu_begin(g)&&h3_gpu_mlp_bf16(g,out,in,weight[mode][2],weight[mode][3],ROWS,WIDTH,HIDDEN,WIDTH)&&h3_gpu_submit(g));
            uint16_t *zero=calloc((size_t)ROWS*WIDTH,2);CHECK(zero);
            gpu_check(g,h3_gpu_tensor_read_bf16(out,x,(size_t)ROWS*WIDTH));CHECK(!memcmp(x,zero,(size_t)ROWS*WIDTH*2));free(zero);
        }
        h3_gpu_tensor_free(fc1);h3_gpu_tensor_free(activated);h3_gpu_tensor_free(in);h3_gpu_tensor_free(out);
    }
    compare(y[0],y[1],(size_t)ROWS*WIDTH,"two-pass MLP");
    uint64_t scale,global,bytes;CHECK(h3_weight_packed_size(nk[0][0],nk[0][1],2,&scale,&global,&bytes));
    h3_quant_cache *cache=h3_quant_cache_open(argv[1],paths[0],0,nk[0][0],nk[0][1],2,bytes,scale,global,error,sizeof(error));CHECK(cache);
    char *artifact=strdup(cache->artifact);CHECK(artifact);h3_quant_cache_close(cache);
    FILE *broken=fopen(artifact,"r+b");CHECK(broken);CHECK(fputc('!',broken)!=EOF&&!fclose(broken));
    CHECK(!h3_gpu_quant_load(bounded,paths[0],0,nk[0][0],nk[0][1]));
    CHECK(strstr(h3_gpu_error(bounded),"corrupt")||strstr(h3_gpu_error(bounded),"incompatible"));
    h3_gpu_cancel(bounded);CHECK(!unlink(artifact));free(artifact);
    h3_gpu_tensor *repaired=h3_gpu_quant_load(bounded,paths[0],0,nk[0][0],nk[0][1]);CHECK(repaired);h3_gpu_tensor_free(repaired);
    for(unsigned i=0;i<4;i++){h3_gpu_tensor_free(weight[0][i]);h3_gpu_tensor_free(weight[1][i]);unlink(paths[i]);}
    free(x);free(y[0]);free(y[1]);h3_gpu_free(legacy);h3_gpu_free(bounded);
    puts("PASS long NVFP4 descriptors, four projections, global scales, tails, two-pass MLP, cache hits, plateau and cancellation recovery");return 0;
}
