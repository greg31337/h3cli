/* Standalone H3 H=56,D=128 dense attention oracle + tile benchmark.
 * Usage: SEQUENCE ITERATIONS PATTERN INPUT_LAYOUT OUTPUT_LAYOUT [QKV.bin]
 * PATTERN random|zero|constant|outlier|file. Input file is sequence-major BF16.
 * Numerical gates are fixed: relative L2 <= 0.01, max abs <= 0.05
 * for bounded generated inputs. File inputs additionally report normalized max.
 * These are component gates; full-model acceptance is measured separately. */
#include "src/gpu.h"
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static float fp(uint16_t b){uint32_t x=(uint32_t)b<<16;float f;memcpy(&f,&x,4);return f;}
static uint16_t bf(float f){uint32_t x;memcpy(&x,&f,4);x+=0x7fff+((x>>16)&1);return (uint16_t)(x>>16);}
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}
static float random_value(uint32_t *s){*s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return (float)((int)(*s%65537)-32768)/16384.f;}
static uint64_t hash(const uint16_t *p,size_t n){uint64_t h=14695981039346656037ull;for(size_t i=0;i<n;i++){h^=p[i];h*=1099511628211ull;}return h;}
static size_t index_of(size_t i,unsigned s,int major){size_t c=i%128,h=(i/128)%56,row=i/(128*56);return major?(h*s+row)*128+c:i;}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);goto cleanup;}}while(0)
int main(int argc,char **argv){
    if(argc<6)return 2;
    unsigned seq=(unsigned)strtoul(argv[1],NULL,10);int repeats=atoi(argv[2]),il=atoi(argv[4]),ol=atoi(argv[5]);
    if(!seq||seq>100000||repeats<1||repeats>6||(il!=0&&il!=1)||(ol!=0&&ol!=1))return 2;
    const char *pattern=argv[3];int fromfile=!strcmp(pattern,"file");
    if(strcmp(pattern,"random")&&strcmp(pattern,"zero")&&strcmp(pattern,"constant")&&strcmp(pattern,"outlier")&&!fromfile)return 2;
    if(fromfile&&argc!=7)return 2;
    const size_t n=(size_t)seq*56*128,guard=256;
    char error[4096]={0};h3_gpu *g=NULL;h3_gpu_tensor *t[4]={0};uint16_t *host=NULL,*ref=NULL,*ordered=NULL;
    uint64_t hashes[3];uint32_t rng=0x5a317d6b;FILE *file=NULL;int result=1;
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    host=malloc((n+guard)*2);ref=malloc(n*2);ordered=malloc((n+guard)*2);CHECK(host&&ref&&ordered);
    if(fromfile){file=fopen(argv[6],"rb");CHECK(file);}
    for(int j=0;j<3;j++){
        for(size_t i=0;i<n;i++)host[i]=bf(!strcmp(pattern,"zero")?0:!strcmp(pattern,"constant")?.5f:random_value(&rng));
        if(fromfile)CHECK(fread(host,2,n,file)==n);
        if(!strcmp(pattern,"outlier")&&j<2)for(size_t i=0;i<n;i+=128*73)host[i]=bf(((i/(128*73))%2?-1.f:1.f)*20.f);
        for(size_t i=n;i<n+guard;i++)host[i]=0x7f7f;
        /* MPS reference starts sequence-major. */
        t[j]=h3_gpu_tensor_from_bf16(g,host,n+guard);CHECK(t[j]);
    }
    if(file){CHECK(fgetc(file)==EOF);fclose(file);file=NULL;}
    for(size_t i=0;i<n;i++)host[i]=0x7fc0;
    t[3]=h3_gpu_tensor_from_bf16(g,host,n+guard);CHECK(t[3]);
    const float scale=1.f/sqrtf(128.f);double mps[6]={0},cold=0;
    for(int r=0;r<=repeats;r++){
        double start=now();CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,56,128,scale));CHECK(h3_gpu_submit(g));
        if(r)mps[r-1]=now()-start;else cold=now()-start;
    }
    CHECK(h3_gpu_tensor_read_bf16(t[3],host,n+guard));memcpy(ref,host,n*2);
    for(size_t i=n;i<n+guard;i++)CHECK(host[i]==0x7f7f);
    /* For short inputs, independently check 16 distributed query/channel
     * positions with double QK/softmax/PV, including first and last rows. */
    double oracle_max=0;
    if(seq<=257){
        const uint16_t *q=h3_gpu_tensor_contents(t[0]),*k=h3_gpu_tensor_contents(t[1]),*v=h3_gpu_tensor_contents(t[2]);
        double *scores=malloc(seq*sizeof(double));CHECK(scores);
        for(int sample=0;sample<16;sample++){
            size_t row=(size_t)sample*(seq-1)/15,h=(size_t)(sample*7)%56,c=(size_t)(sample*19)%128;double mx=-INFINITY,sum=0,val=0;
            for(size_t x=0;x<seq;x++){double z=0;for(size_t d=0;d<128;d++)z+=(double)fp(q[(row*56+h)*128+d])*fp(k[(x*56+h)*128+d]);scores[x]=z*scale;if(scores[x]>mx)mx=scores[x];}
            for(size_t x=0;x<seq;x++){double w=exp(scores[x]-mx);sum+=w;val+=w*fp(v[(x*56+h)*128+c]);}
            double diff=fabs(val/sum-fp(ref[(row*56+h)*128+c]));if(diff>oracle_max)oracle_max=diff;
        }
        free(scores);CHECK(oracle_max<=.05);
    }
    for(int j=0;j<3;j++){
        CHECK(h3_gpu_tensor_read_bf16(t[j],host,n+guard));
        for(size_t i=0;i<n;i++)ordered[index_of(i,seq,il)]=host[i];
        for(size_t i=n;i<n+guard;i++)ordered[i]=0x7f7f;
        CHECK(h3_gpu_tensor_write_bf16(t[j],ordered,n+guard));hashes[j]=hash(ordered,n+guard);
    }
    printf("{\"sequence\":%u,\"heads\":56,\"dimension\":128,\"input_head_major\":%d,\"output_head_major\":%d,\"pattern\":\"%s\",\"mps_cold\":%.9f,\"cpu_oracle_max\":%.9f,\"mps_seconds\":[",seq,il,ol,pattern,cold,oracle_max);
    for(int r=0;r<repeats;r++)printf("%s%.9f",r?",":"",mps[r]);printf("],\"tiles\":[");fflush(stdout);
    for(int tile=0;tile<6;tile++){
        double times[6]={0};
        for(int r=0;r<=repeats;r++){
            double start=now();CHECK(h3_gpu_begin(g));
            CHECK(tile<4?h3_gpu_native_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,56,128,scale,il,ol,tile):h3_gpu_native_sdpa_candidate_b(g,t[3],t[0],t[1],t[2],seq,56,scale,il,ol,tile==4?32:64));CHECK(h3_gpu_submit(g));
            if(r)times[r-1]=now()-start;else cold=now()-start;
        }
        CHECK(h3_gpu_tensor_read_bf16(t[3],host,n+guard));
        double diff2=0,ref2=0,maxabs=0,peak=0;
        for(size_t i=0;i<n;i++){double f=fp(host[index_of(i,seq,ol)]),b=fp(ref[i]);CHECK(isfinite(f));double d=f-b;diff2+=d*d;ref2+=b*b;if(fabs(d)>maxabs)maxabs=fabs(d);if(fabs(b)>peak)peak=fabs(b);}
        double rel=sqrt(diff2/fmax(ref2,1e-30));
        for(size_t i=n;i<n+guard;i++)CHECK(host[i]==0x7f7f);
        printf("%s{\"tile\":%d,\"cold_seconds\":%.9f,\"relative_l2\":%.9g,\"max_abs\":%.9g,\"normalized_max\":%.9g,\"seconds\":[",tile?",":"",tile,cold,rel,maxabs,maxabs/fmax(sqrt(ref2/(double)n),1e-30));
        for(int r=0;r<repeats;r++)printf("%s%.9f",r?",":"",times[r]);printf("]}");fflush(stdout);
        CHECK(rel<=.01&&maxabs<=(fromfile?.02*fmax(1.,peak):.05));
    }
    for(int j=0;j<3;j++){CHECK(h3_gpu_tensor_read_bf16(t[j],host,n+guard));CHECK(hash(host,n+guard)==hashes[j]);}
    printf("]");
    if(getenv("H3_TEST_ATTENTION_BALANCED")) {
        /* All six permutations, one warmed call per implementation per round.
         * Equal sequence-major I/O includes MPSGraph's layout handling. Native
         * direct-layout preparation/projection is measured in the block tool. */
        CHECK(il==0&&ol==0);
        const int orders[6][3]={{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
        double balanced[3][6]={{0}};
        for(int round=-1;round<6;round++)for(int j=0;j<3;j++) {
            int implementation=round<0?j:orders[round][j];double start=now();
            CHECK(h3_gpu_begin(g));
            CHECK(implementation==0?h3_gpu_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,56,128,scale):
                implementation==1?h3_gpu_native_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,56,128,scale,0,0,2):
                h3_gpu_native_sdpa_candidate_b(g,t[3],t[0],t[1],t[2],seq,56,scale,0,0,32));
            CHECK(h3_gpu_submit(g));if(round>=0)balanced[implementation][round]=now()-start;
        }
        printf(",\"balanced_order\":[[0,1,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[0,2,1]],\"balanced_seconds\":{");
        const char *names[]={"mpsgraph","steel","steel-routed"};
        for(int i=0;i<3;i++){printf("%s\"%s\":[",i?",":"",names[i]);for(int r=0;r<6;r++)printf("%s%.9f",r?",":"",balanced[i][r]);printf("]");}printf("}");
    }
    printf(",\"pass\":true}\n");result=0;
cleanup:
    if(file)fclose(file);if(result&&g)h3_gpu_cancel(g);
    for(int j=0;j<4;j++)h3_gpu_tensor_free(t[j]);h3_gpu_free(g);free(host);free(ref);free(ordered);return result;
}
