/* Real model weights, bounded rows and full conversion+GEMM+tail timings. */
#include "src/gpu.h"
#include "src/weights/quant.h"
#include "src/execution.h"
#include "src/weights/weights.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
static char error[512];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s: %s %s\n",__LINE__,#x,error,g?h3_gpu_error(g):"");exit(1);}}while(0)
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec/1.e9;}
int main(int argc,char **argv){
    h3_gpu *g=NULL;int mode;char *end=NULL;
    if(argc!=5||!h3_quant_parse(argv[1],&mode))return 2;
    unsigned long count=strtoul(argv[4],&end,10);if(!end||*end||count<1||count>16384)return 2;
    unsigned rows=(unsigned)count;
    h3_weight_store *store=h3_weight_store_open(argv[2],error,sizeof(error));CHECK(store);
    const char *names[]={"attn.qkv_proj.weight","attn.out_proj.weight","mlp.fc1.weight","mlp.fc2.weight"};
    unsigned dims[][2]={{21504,5376},{5376,7168},{28672,5376},{5376,14336}};
    for(unsigned op=0;op<4;op++){
        g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
        /* Reserve one complete H3 block for this block-0 operator probe. */
        const uint64_t core_weights=2ull*(4ull*7168*5376+3ull*14336*5376);
        if(mode)CHECK(h3_gpu_quant_configure(g,mode,argv[3],core_weights,1ull<<30,0));
        char name[128];snprintf(name,sizeof(name),"blocks.0.%s",names[op]);
        const h3_st_header *header;const h3_st_tensor *tensor=h3_weight_find(store,name,&header);CHECK(tensor);
        unsigned n=dims[op][0],k=dims[op][1];uint64_t shape[]={n,k};
        h3_gpu_tensor *w=mode?h3_gpu_quant_load(g,header->path,tensor->file_offset,n,k):h3_weight_load_bf16(store,g,name,2,shape,error,sizeof(error));CHECK(w);
        uint16_t *values=malloc((size_t)rows*k*2);CHECK(values);
        for(size_t i=0;i<(size_t)rows*k;i++){float f=(float)((int)(i%37)-18)/32;uint32_t bits;memcpy(&bits,&f,4);values[i]=(uint16_t)(bits>>16);}
        h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,values,(size_t)rows*k),*y=h3_gpu_tensor_new_bf16(g,(size_t)rows*n);free(values);CHECK(x&&y);
        double times[5];
        for(unsigned repeat=0;repeat<5;repeat++){
            double begin=now();CHECK(h3_gpu_begin(g)&&h3_gpu_linear_bf16(g,y,x,w,NULL,rows,k,n)&&h3_gpu_submit(g));times[repeat]=now()-begin;
        }
        uint16_t *result=malloc((size_t)rows*n*2);CHECK(result&&h3_gpu_tensor_read_bf16(y,result,(size_t)rows*n));
        for(size_t i=0;i<(size_t)rows*n;i++)CHECK((result[i]&0x7f80)!=0x7f80);free(result);
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
        printf("{\"mode\":\"%s\",\"matrix\":\"%s\",\"rows\":%u,\"n\":%u,\"k\":%u,\"cold\":%.9f,\"warm\":%.9f,\"peak_bytes\":%llu,\"finite\":true}\n",argv[1],name,rows,n,k,times[0],(times[1]+times[2]+times[3]+times[4])/4,(unsigned long long)stats.peak_live_bytes);
        h3_gpu_tensor_free(w);h3_gpu_tensor_free(x);h3_gpu_tensor_free(y);h3_gpu_free(g);g=NULL;
    }
    h3_weight_store_free(store);return 0;
}
