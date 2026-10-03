/* Independent Q8 packing/FP64 dot oracle, tails, range and descriptor guards.
 * --bench MODEL_DIRECTORY ROWS uses real block-zero H3 matrices and MPSGraph
 * BF16 as the timing/quality reference. No denoising occurs in this harness. */
#include "src/gpu.h"
#include "src/weights/q8.h"
#include "src/weights/weights.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <time.h>

#define CHECK(x) do {if(!(x)){fprintf(stderr,"Q8 FAIL line %d: %s (%s)\n",__LINE__,#x,g?h3_gpu_error(g):error);goto done;}}while(0)
static uint16_t bf(float x){uint32_t u;memcpy(&u,&x,4);u+=0x7fff+((u>>16)&1);return (uint16_t)(u>>16);}
static float fp(uint16_t b){uint32_t u=(uint32_t)b<<16;float x;memcpy(&x,&u,4);return x;}
static float random_float(uint32_t *s){*s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return (float)((int)(*s%8193)-4096)/4096.f;}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}
static int host(void){
    h3_gpu *g=NULL;char error[1]={0};int ok=0;size_t offset,bytes;
    uint16_t source[65]={0};int8_t q[65];float scales[2];
    float values[]={127,-127,.5f,1.5f,2.5f,-.5f,-1.5f,-2.5f};
    int expected[]={127,-127,0,2,2,0,-2,-2};
    for(size_t i=0;i<8;i++)source[i]=bf(values[i]);
    CHECK(h3_q8_pack_rows(source,q,scales,1,65));
    for(size_t i=0;i<8;i++)CHECK(q[i]==expected[i]);
    CHECK(scales[0]==1&&scales[1]==1&&q[64]==0);
    CHECK(h3_q8_layout(1,65,&offset,&bytes)&&offset==68&&bytes==76);
    CHECK(!h3_q8_layout(0,65,&offset,&bytes));
    CHECK(!h3_q8_layout(UINT32_MAX,UINT32_MAX,&offset,&bytes));
    CHECK(!h3_q8_layout(1,UINT32_MAX-31u,&offset,&bytes));
    source[64]=0x7fc0;CHECK(!h3_q8_pack_rows(source,q,scales,1,65));
    source[64]=0x7f80;CHECK(!h3_q8_pack_rows(source,q,scales,1,65));
    source[64]=1;CHECK(h3_q8_pack_rows(source,q,scales,1,65)&&isfinite(scales[1]));
    puts("{\"host_pack\":true,\"pass\":true}");ok=1;
done:return ok;
}

/* Multiple distinct matrices reuse the same dequant scratch within one command
 * buffer. A correct isolated GEMM cannot detect a missing producer/consumer
 * dependency here. Use exactly representable diagonal matrices as an oracle. */
static int queued(void){
    h3_metal_attention_options options=H3_METAL_ATTENTION_DEFAULT;
    options.weight_format=H3_WEIGHT_Q8;
    h3_backend_scope saved=h3_backend_exchange((h3_backend_scope){H3_BACKEND_METAL,H3_ATTN_DENSE,options});
    h3_gpu *g=NULL;char error[1024]={0};int ok=0,fd=-1;
    h3_gpu_tensor *x=NULL,*w[2]={0},*out[6]={0};
    uint16_t source[64*64]={0},input[3*64],actual[3*64];
    char path[]="/tmp/h3-q8-queue-XXXXXX";
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    CHECK(h3_gpu_dit_attention_configure(g,0));
    fd=mkstemp(path);CHECK(fd>=0);
    for(int j=0;j<2;j++){
        for(int k=0;k<64;k++)source[k*64+k]=bf((float)(j+1));
        CHECK(write(fd,source,sizeof(source))==(ssize_t)sizeof(source));
    }
    close(fd);fd=-1;
    for(int j=0;j<2;j++){w[j]=h3_gpu_q8_load(g,path,(uint64_t)j*sizeof(source),64,64);CHECK(w[j]);}
    for(int k=0;k<3*64;k++)input[k]=bf((float)(k-100));
    x=h3_gpu_tensor_from_bf16(g,input,3*64);CHECK(x);
    for(int j=0;j<6;j++){out[j]=h3_gpu_tensor_new_bf16(g,3*64);CHECK(out[j]);}
    h3_gpu_stats warm={0},stats={0};
    for(int repeat=0;repeat<3;repeat++){
        CHECK(h3_gpu_begin(g));
        for(int j=0;j<6;j++){
            CHECK(h3_gpu_linear_bf16(g,out[j],x,w[(j+repeat)%2],NULL,3,64,64));
            if(repeat&&j%2)CHECK(h3_gpu_continue(g));
        }
        CHECK(h3_gpu_submit(g));
        for(int j=0;j<6;j++){
            CHECK(h3_gpu_tensor_read_bf16(out[j],actual,3*64));
            for(int k=0;k<3*64;k++)CHECK(actual[k]==bf(fp(input[k])*(float)(1+(j+repeat)%2)));
        }
        CHECK(h3_gpu_get_stats(g,&stats));
        if(!repeat)warm=stats;
        else CHECK(stats.tensor_allocations==warm.tensor_allocations&&
                   stats.native_scratch_bytes==warm.native_scratch_bytes&&
                   stats.native_scratch_reuses>warm.native_scratch_reuses);
    }
    CHECK(stats.q8_weights==2&&stats.q8_dispatches==18);
    puts("{\"queued_scratch_dependencies\":true,\"pass\":true}");ok=1;
done:
    if(g)h3_gpu_cancel(g);h3_gpu_tensor_free(x);
    for(int j=0;j<2;j++)h3_gpu_tensor_free(w[j]);
    for(int j=0;j<6;j++)h3_gpu_tensor_free(out[j]);
    h3_gpu_free(g);if(fd>=0)close(fd);unlink(path);h3_backend_exchange(saved);return ok;
}

static int run(uint32_t m,uint32_t n,uint32_t k,int large,int benchmark,
               const char *directory,const char *name,int kernel){
    h3_metal_attention_options options=H3_METAL_ATTENTION_DEFAULT;options.weight_format=H3_WEIGHT_Q8;options.q8_kernel=kernel;
    h3_backend_scope saved=h3_backend_exchange((h3_backend_scope){H3_BACKEND_METAL,H3_ATTN_DENSE,options});
    h3_gpu *g=NULL;char error[1024]={0};int ok=0;
    h3_gpu_tensor *x=NULL,*w=NULL,*out=NULL,*oracle=NULL,*original=NULL,*bias=NULL;
    h3_weight_store *store=NULL;uint16_t *input=NULL,*data=NULL,*actual=NULL,*reference=NULL;
    uint16_t *cpu_weight=NULL;int8_t *packed=NULL;float *scales=NULL;
    char path[]="/tmp/h3-q8-XXXXXX";int fd=-1;size_t elements=(size_t)n*k;
    size_t output_count=(size_t)m*n,guard=64;
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);CHECK(h3_gpu_dit_attention_configure(g,0));
    input=malloc((size_t)m*k*2);actual=malloc((output_count+guard)*2);
    reference=malloc(output_count*2);CHECK(input&&actual&&reference);
    uint32_t rng=12467;
    for(size_t i=0;i<(size_t)m*k;i++)input[i]=bf(random_float(&rng)*(large?262144.f:1.f));
    x=h3_gpu_tensor_from_bf16(g,input,(size_t)m*k);CHECK(x);
    for(size_t i=0;i<output_count+guard;i++)actual[i]=0x3f80;
    out=h3_gpu_tensor_from_bf16(g,actual,output_count+guard);CHECK(out);
    if(benchmark){
        store=h3_weight_store_open(directory,error,sizeof(error));CHECK(store);
        const h3_st_header *header=NULL;const h3_st_tensor *t=h3_weight_find(store,name,&header);
        CHECK(t&&t->dtype==H3_DTYPE_BF16&&t->ndim==2&&t->shape[0]==n&&t->shape[1]==k);
        w=h3_gpu_q8_load(g,header->path,t->file_offset,n,k);CHECK(w);
        original=h3_gpu_tensor_load_bf16(g,header->path,t->file_offset,elements);CHECK(original);
        oracle=h3_gpu_tensor_new_bf16(g,output_count);CHECK(oracle);
    }else{
        data=malloc(elements*2);packed=malloc(elements);cpu_weight=malloc(elements*2);
        size_t groups=(k+63u)/64u;scales=malloc((size_t)n*groups*4);CHECK(data&&packed&&scales&&cpu_weight);
        for(size_t i=0;i<elements;i++)data[i]=bf(random_float(&rng)*.125f);
        fd=mkstemp(path);CHECK(fd>=0);CHECK(write(fd,data,elements*2)==(ssize_t)(elements*2));close(fd);fd=-1;
        w=h3_gpu_q8_load(g,path,0,n,k);CHECK(w);
        CHECK(h3_q8_pack_rows(data,packed,scales,n,k));
        size_t offset,bytes;CHECK(h3_q8_layout(n,k,&offset,&bytes));
        unsigned char *stored=h3_gpu_tensor_contents(w);CHECK(stored);
        CHECK(!memcmp(stored,packed,elements)&&!memcmp(stored+offset,scales,(size_t)n*groups*4));
        for(uint32_t j=0;j<n;j++)for(uint32_t c=0;c<k;c++)
            cpu_weight[(size_t)j*k+c]=bf((float)packed[(size_t)j*k+c]*scales[(size_t)j*groups+c/64]);
        uint16_t *bias_values=malloc(n*2);CHECK(bias_values);
        for(uint32_t j=0;j<n;j++)bias_values[j]=bf((float)(j%5)*.125f);
        bias=h3_gpu_tensor_from_bf16(g,bias_values,n);free(bias_values);CHECK(bias);
        for(uint32_t r=0;r<m;r++)for(uint32_t j=0;j<n;j++){
            double sum=(float)(j%5)*.125f;
            for(uint32_t c=0;c<k;c++)sum+=(double)fp(input[(size_t)r*k+c])*fp(cpu_weight[(size_t)j*k+c]);
            reference[(size_t)r*n+j]=bf((float)sum);
        }
        CHECK(!h3_gpu_q8_load(g,path,1,n,k));
        CHECK(!h3_gpu_q8_load(g,path,0,UINT32_MAX,UINT32_MAX));
    }
    double times[2][3]={{0}};
    int repeats=benchmark?3:1;
    for(int repeat=-1;repeat<repeats;repeat++)for(int slot=0;slot<(benchmark?2:1);slot++){
        int which=benchmark?((slot+(repeat>=0?repeat:0))%2):0;
        double start=now();CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_linear_bf16(g,which?oracle:out,x,which?original:w,bias,m,k,n));
        CHECK(h3_gpu_submit(g));
        if(repeat>=0)times[which][repeat]=now()-start;
    }
    CHECK(h3_gpu_tensor_read_bf16(out,actual,output_count+guard));
    if(benchmark)CHECK(h3_gpu_tensor_read_bf16(oracle,reference,output_count));
    double ss=0,ee=0,dot=0,aa=0;float max_error=0;
    for(size_t i=0;i<output_count;i++){
        float a=fp(actual[i]),b=fp(reference[i]);CHECK(isfinite(a)&&isfinite(b));
        double d=(double)a-b;ee+=d*d;ss+=(double)b*b;aa+=(double)a*a;dot+=(double)a*b;
        max_error=fmaxf(max_error,fabsf(a-b));
    }
    for(size_t i=output_count;i<output_count+guard;i++)CHECK(actual[i]==0x3f80);
    double relative=sqrt(ee/fmax(ss,1e-30)),cosine=dot/sqrt(fmax(ss*aa,1e-30));
    CHECK(relative<(benchmark?.03:.006));
    if(!benchmark){
        CHECK(h3_gpu_begin(g));
        CHECK(!h3_gpu_linear_bf16(g,out,x,w,NULL,m,k,n+1));
        CHECK(!h3_gpu_linear_bf16(g,x,x,w,NULL,m,k,n));
        h3_gpu_cancel(g);
        CHECK(!memcmp(h3_gpu_tensor_contents(x),input,(size_t)m*k*2));
    }
    h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
    CHECK(stats.q8_weights==1&&stats.q8_dispatches==(uint64_t)(repeats+1));
    printf("{\"m\":%u,\"n\":%u,\"k\":%u,\"large_activation\":%s,\"benchmark\":%s,\"kernel\":%d,\"relative_l2\":%.9g,\"cosine\":%.9g,\"max_error\":%.9g,\"source_bytes\":%llu,\"storage_bytes\":%llu,\"load_seconds\":%.9g,\"q8_seconds\":[%.9g,%.9g,%.9g],\"bf16_seconds\":[%.9g,%.9g,%.9g],\"pass\":true}\n",
        m,n,k,large?"true":"false",benchmark?"true":"false",kernel,relative,cosine,max_error,
        (unsigned long long)stats.q8_source_bytes,(unsigned long long)stats.q8_storage_bytes,stats.q8_load_seconds,
        times[0][0],times[0][1],times[0][2],times[1][0],times[1][1],times[1][2]);fflush(stdout);ok=1;
done:
    if(g)h3_gpu_cancel(g);h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);h3_gpu_tensor_free(out);
    h3_gpu_tensor_free(oracle);h3_gpu_tensor_free(original);h3_gpu_tensor_free(bias);
    h3_weight_store_free(store);h3_gpu_free(g);free(input);free(data);free(actual);free(reference);
    free(packed);free(scales);free(cpu_weight);if(fd>=0)close(fd);if(!benchmark)unlink(path);h3_backend_exchange(saved);return ok;
}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"--host"))return !host();
    if(argc==1){
        if(!host()||!queued())return 1;
        for(int kernel=0;kernel<2;kernel++)if(!(run(1,3,1,0,0,NULL,NULL,kernel)&&run(17,35,65,0,0,NULL,NULL,kernel)&&
            run(65,67,129,1,0,NULL,NULL,kernel)&&run(129,256,256,0,0,NULL,NULL,kernel)))return 1;
        return 0;
    }
    if(argc==6&&!strcmp(argv[1],"--bench")){
        char *end=NULL;unsigned long rows=strtoul(argv[3],&end,10);if(*end||!rows||rows>22426)return 2;
        const char *names[]={"blocks.0.attn.qkv_proj.weight","blocks.0.attn.out_proj.weight","blocks.0.mlp.fc1.weight","blocks.0.mlp.fc2.weight"};
        const uint32_t nk[][2]={{21504,5376},{5376,7168},{28672,5376},{5376,14336}};
        unsigned long index=strtoul(argv[4],&end,10);if(*end||index>3)return 2;
        int kernel=!strcmp(argv[5],"simdgroup")?1:!strcmp(argv[5],"mpsgraph")?0:-1;if(kernel<0)return 2;
        return !run((uint32_t)rows,nk[index][0],nk[index][1],0,1,argv[2],names[index],kernel);
    }
    fprintf(stderr,"Usage: %s [--bench DiT-directory rows(1..22426) projection(0..3) mpsgraph|simdgroup]\n",argv[0]);return 2;
}
