/* M2 adapter-inclusive FP16 attention oracle + balanced benchmark.
 * Usage: SEQUENCE ITERATIONS PATTERN INPUT_LAYOUT OUTPUT_LAYOUT [QKV.bin]
 * Input file is sequence-major BF16. See metal_fp16_limits.json for frozen
 * operator limits; overflow/underflow/recovery and rejected inputs are covered.
 * These are component gates; full-model acceptance is measured separately. */
#include "src/metal/metal_fp16.h"
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
    if(!h3_metal_fp16_shape_valid(seq,56)||repeats<1||repeats>6||(il!=0&&il!=1)||(ol!=0&&ol!=1))return 2;
    int tile_count=2;
    const char *tile_limit=getenv("H3_TEST_FP16_TILE_COUNT");
    if(tile_limit){if(strcmp(tile_limit,"1"))return 2;tile_count=1;}
    const char *pattern=argv[3];int fromfile=!strcmp(pattern,"file");
    int invalid=!strcmp(pattern,"nonfinite")||!strcmp(pattern,"score-overflow")||!strcmp(pattern,"bf16-subnormal");
    if(strcmp(pattern,"random")&&strcmp(pattern,"zero")&&strcmp(pattern,"constant")&&strcmp(pattern,"outlier")&&strcmp(pattern,"overflow")&&strcmp(pattern,"underflow")&&strcmp(pattern,"recovery")&&strcmp(pattern,"bounded-loss")&&strcmp(pattern,"reciprocal")&&!invalid&&!fromfile)return 2;
    if(fromfile&&argc!=7)return 2;
    const size_t n=(size_t)seq*56*128,guard=256;
    char error[4096]={0};h3_gpu *g=NULL;h3_gpu_tensor *t[4]={0},*converted[4]={0};
    uint16_t *host=NULL,*ref=NULL,*ordered=NULL,*converted_ref=NULL,*refwork=NULL;
    uint64_t hashes[3];uint32_t rng=0x5a317d6b;FILE *file=NULL;int result=1;
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    host=malloc((n+guard)*2);ref=malloc(n*2);ordered=malloc((n+guard)*2);CHECK(host&&ref&&ordered);
    if(fromfile){file=fopen(argv[6],"rb");CHECK(file);}
    for(int j=0;j<3;j++){
        for(size_t i=0;i<n;i++)host[i]=bf(!strcmp(pattern,"zero")?0:!strcmp(pattern,"constant")?.5f:random_value(&rng));
        if(fromfile)CHECK(fread(host,2,n,file)==n);
        if(!strcmp(pattern,"outlier")&&j<2)for(size_t i=0;i<n;i+=128*73)host[i]=bf(((i/(128*73))%2?-1.f:1.f)*20.f);
        if(!strcmp(pattern,"overflow"))for(size_t i=0;i<n;i++)host[i]=bf(ldexpf(fp(host[i]),j==2?40:8));
        if(!strcmp(pattern,"underflow"))for(size_t i=0;i<n;i++)host[i]=bf(ldexpf(fp(host[i]),-40));
        if(!strcmp(pattern,"recovery")) {
            if(j<2)for(size_t i=0;i<n;i++)host[i]=bf(ldexpf(fp(host[i]),16));
            host[0]=bf(0x1p-60f);
        }
        if(!strcmp(pattern,"bounded-loss"))host[0]=bf(0x1p-40f);
        if(!strcmp(pattern,"reciprocal"))for(size_t i=0;i<n;i++)host[i]=bf(ldexpf(fp(host[i]),j==0?80:j==1?-80:0));
        if(!strcmp(pattern,"bf16-subnormal")&&j==0)host[0]=1;
        if(!strcmp(pattern,"score-overflow")&&j<2)for(size_t i=0;i<n;i++)host[i]=bf(0x1p100f);
        if(!strcmp(pattern,"nonfinite")&&j==0)host[0]=0x7fc0;
        for(size_t i=n;i<n+guard;i++)host[i]=0x7f7f;
        /* MPS reference starts sequence-major. */
        t[j]=h3_gpu_tensor_from_bf16(g,host,n+guard);CHECK(t[j]);
    }
    if(file){CHECK(fgetc(file)==EOF);fclose(file);file=NULL;}
    for(size_t i=0;i<n;i++)host[i]=0x3f00;
    t[3]=h3_gpu_tensor_from_bf16(g,host,n+guard);CHECK(t[3]);
    const float scale=1.f/sqrtf(128.f);double mps[6]={0},cold=0;
    for(int r=0;!invalid&&r<=repeats;r++){
        double start=now();CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,56,128,scale));CHECK(h3_gpu_submit(g));
        if(r)mps[r-1]=now()-start;else cold=now()-start;
    }
    CHECK(h3_gpu_tensor_read_bf16(t[3],host,n+guard));memcpy(ref,host,n*2);
    for(size_t i=n;i<n+guard;i++)CHECK(host[i]==0x7f7f);
    /* For short inputs, independently check 16 distributed query/channel
     * positions with double QK/softmax/PV, including first and last rows. */
    double oracle_max=0,oracle_value[16]={0},oracle_peak[16]={0};
    if(seq<=257&&!invalid){
        const uint16_t *q=h3_gpu_tensor_contents(t[0]),*k=h3_gpu_tensor_contents(t[1]),*v=h3_gpu_tensor_contents(t[2]);
        double *scores=malloc(seq*sizeof(double));CHECK(scores);
        for(int sample=0;sample<16;sample++){
            size_t row=(size_t)sample*(seq-1)/15,h=(size_t)(sample*7)%56,c=(size_t)(sample*19)%128;double mx=-INFINITY,sum=0,val=0;
            for(size_t x=0;x<seq;x++){double z=0;for(size_t d=0;d<128;d++)z+=(double)fp(q[(row*56+h)*128+d])*fp(k[(x*56+h)*128+d]);scores[x]=z*scale;if(scores[x]>mx)mx=scores[x];}
            for(size_t x=0;x<seq;x++){double w=exp(scores[x]-mx);sum+=w;val+=w*fp(v[(x*56+h)*128+c]);}
            double vpeak=0;for(size_t x=0;x<seq;x++)vpeak=fmax(vpeak,fabs(fp(v[(x*56+h)*128+c])));
            oracle_value[sample]=val/sum;oracle_peak[sample]=vpeak;
            double diff=fabs(val/sum-fp(ref[(row*56+h)*128+c]))/fmax(vpeak,1e-30);if(diff>oracle_max)oracle_max=diff;
        }
        free(scores);CHECK(oracle_max<=.005);
    }
    for(int j=0;j<3;j++){
        CHECK(h3_gpu_tensor_read_bf16(t[j],host,n+guard));
        for(size_t i=0;i<n;i++)ordered[index_of(i,seq,il)]=host[i];
        for(size_t i=n;i<n+guard;i++)ordered[i]=0x7f7f;
        CHECK(h3_gpu_tensor_write_bf16(t[j],ordered,n+guard));hashes[j]=hash(ordered,n+guard);
    }
    printf("{\"sequence\":%u,\"heads\":56,\"dimension\":128,\"input_head_major\":%d,\"output_head_major\":%d,\"pattern\":\"%s\",\"mps_cold\":%.9f,\"cpu_oracle_max\":%.9f,\"mps_seconds\":[",seq,il,ol,pattern,cold,oracle_max);
    for(int r=0;r<repeats;r++)printf("%s%.9f",r?",":"",mps[r]);printf("],\"tiles\":[");fflush(stdout);
    for(int tile=0;tile<tile_count;tile++){
        double times[6]={0};
        for(int r=0;r<=repeats;r++){
            double start=now();CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_mixed_sdpa(g,t[3],t[0],t[1],t[2],seq,56,scale,il,ol,tile));CHECK(h3_gpu_submit(g));
            if(r)times[r-1]=now()-start;else cold=now()-start;
        }
        CHECK(h3_gpu_tensor_read_bf16(t[3],host,n+guard));
        h3_metal_fp16_range ranges[56];CHECK(h3_gpu_mixed_range_report(g,ranges,56));
        unsigned recovered=0,rejected=0;float qmax=0,kmax=0,vmax=0,omax=0,score_bound=0,value_bound=0;
        uint64_t underflows=0;
        for(int h=0;h<56;h++){recovered+=(ranges[h].flags&2)!=0;rejected+=((ranges[h].flags&1)!=0)||ranges[h].output_invalid;
            qmax=fmaxf(qmax,ranges[h].q_max);kmax=fmaxf(kmax,ranges[h].k_max);vmax=fmaxf(vmax,ranges[h].v_max);omax=fmaxf(omax,ranges[h].output_max);
            if(!ranges[h].flags){score_bound=fmaxf(score_bound,ranges[h].score_error_bound);value_bound=fmaxf(value_bound,ranges[h].value_error_bound);}
            underflows+=(uint64_t)ranges[h].q_underflows+ranges[h].k_underflows+ranges[h].v_underflows;}
        CHECK(invalid?rejected>0:rejected==0);
        if(!strcmp(pattern,"recovery"))CHECK(recovered>0);
        if(!strcmp(pattern,"bounded-loss"))CHECK(underflows>0&&recovered==0);
        const uint16_t *operator_ref=ref;
        // Separate identical-converted-input operator correctness from BF16
        // conversion drift. BF16 values otherwise fit scaled normal half exactly.
        if(underflows&&!invalid) {
            converted_ref=malloc(n*2);refwork=malloc(n*2);CHECK(converted_ref&&refwork);
            for(int j=0;j<3;j++) {
                CHECK(h3_gpu_tensor_read_bf16(t[j],ordered,n+guard));
                uint64_t lost=0,reported=0;
                for(unsigned h=0;h<56;h++)reported+=j==0?ranges[h].q_underflows:j==1?ranges[h].k_underflows:ranges[h].v_underflows;
                for(size_t i=0;i<n;i++) {
                    unsigned h=(unsigned)((i/128)%56);uint16_t x=ordered[index_of(i,seq,il)];
                    int ex=j==0?ranges[h].q_exponent:j==1?ranges[h].k_exponent:ranges[h].v_exponent;
                    float scaled=ldexpf(fp(x),-ex);
                    if(!ranges[h].flags&&scaled!=0&&fabsf(scaled)<0x1p-14f){x=0;lost++;}
                    refwork[i]=x;
                }
                CHECK(lost==reported);
                converted[j]=h3_gpu_tensor_from_bf16(g,refwork,n);CHECK(converted[j]);
            }
            converted[3]=h3_gpu_tensor_new_bf16(g,n);CHECK(converted[3]);
            CHECK(h3_gpu_begin(g));CHECK(h3_gpu_sdpa_bf16(g,converted[3],converted[0],converted[1],converted[2],seq,56,128,scale));
            CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(converted[3],converted_ref,n));
            operator_ref=converted_ref;
            for(int j=0;j<4;j++){h3_gpu_tensor_free(converted[j]);converted[j]=NULL;}
            free(refwork);refwork=NULL;
        }
        double diff2=0,ref2=0,maxabs=0,peak=0;
        double original_diff=0,original_norm=0;
        for(size_t i=0;i<n;i++){double f=fp(host[index_of(i,seq,ol)]),b=fp(operator_ref[i]),orig=fp(ref[i]);CHECK(isfinite(f));if(invalid)CHECK(host[index_of(i,seq,ol)]==0x3f00);double d=f-b;diff2+=d*d;ref2+=b*b;original_diff+=(f-orig)*(f-orig);original_norm+=orig*orig;if(fabs(d)>maxabs)maxabs=fabs(d);if(fabs(b)>peak)peak=fabs(b);}
        double rel=sqrt(diff2/fmax(ref2,1e-30));
        double cpu_error=0;
        if(seq<=257&&!invalid)for(int sample=0;sample<16;sample++) {
            size_t row=(size_t)sample*(seq-1)/15,h=(size_t)(sample*7)%56,c=(size_t)(sample*19)%128;
            double d=fabs(fp(host[index_of((row*56+h)*128+c,seq,ol)])-oracle_value[sample])/fmax(oracle_peak[sample],1e-30);
            cpu_error=fmax(cpu_error,d);
        }
        CHECK(cpu_error<=.005);
        for(size_t i=n;i<n+guard;i++)CHECK(host[i]==0x7f7f);
        printf("%s{\"tile\":%d,\"cold_seconds\":%.9f,\"relative_l2\":%.9g,\"bf16_relative_l2\":%.9g,\"max_abs\":%.9g,\"normalized_max\":%.9g,\"seconds\":[",tile?",":"",tile,cold,rel,sqrt(original_diff/fmax(original_norm,1e-30)),maxabs,maxabs/fmax(sqrt(ref2/(double)n),1e-30));
        for(int r=0;r<repeats;r++)printf("%s%.9f",r?",":"",times[r]);printf("],\"cpu_error\":%.9g,\"recovered_heads\":%u,\"rejected_heads\":%u,\"ranges_max\":[%.9g,%.9g,%.9g,%.9g],\"underflows\":%llu,\"score_error_bound\":%.9g,\"value_error_bound\":%.9g}",cpu_error,recovered,rejected,qmax,kmax,vmax,omax,(unsigned long long)underflows,score_bound,value_bound);fflush(stdout);
        CHECK(invalid||(rel<=.01&&maxabs<=.02*fmax(peak,1e-30)));
        free(converted_ref);converted_ref=NULL;
    }
    for(int j=0;j<3;j++){CHECK(h3_gpu_tensor_read_bf16(t[j],host,n+guard));CHECK(hash(host,n+guard)==hashes[j]);}
    printf("]");
    if(getenv("H3_TEST_ATTENTION_BALANCED")&&!invalid) {
        /* All six permutations, one warmed call per implementation per round.
         * Equal sequence-major I/O includes MPSGraph's layout handling. Native
         * direct-layout preparation/projection is measured in the block tool. */
        CHECK(il==0&&ol==0&&tile_count==2);
        const int orders[6][3]={{0,1,2},{1,2,0},{2,0,1},{2,1,0},{1,0,2},{0,2,1}};
        double balanced[3][6]={{0}};
        for(int round=-1;round<6;round++)for(int j=0;j<3;j++) {
            int implementation=round<0?j:orders[round][j];double start=now();
            CHECK(h3_gpu_begin(g));
            CHECK(implementation==0?h3_gpu_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,56,128,scale):
                h3_gpu_mixed_sdpa(g,t[3],t[0],t[1],t[2],seq,56,scale,0,0,implementation-1));
            CHECK(h3_gpu_submit(g));if(round>=0)balanced[implementation][round]=now()-start;
        }
        printf(",\"balanced_order\":[[0,1,2],[1,2,0],[2,0,1],[2,1,0],[1,0,2],[0,2,1]],\"balanced_seconds\":{");
        const char *names[]={"mpsgraph","fp16-32x16","fp16-64x32"};
        for(int i=0;i<3;i++){printf("%s\"%s\":[",i?",":"",names[i]);for(int r=0;r<6;r++)printf("%s%.9f",r?",":"",balanced[i][r]);printf("]");}printf("}");
    }
    if(seq==1&&!invalid) {
        CHECK(h3_gpu_tensor_read_bf16(t[0],ordered,n+guard));ordered[0]=0x7fc0;
        converted[0]=h3_gpu_tensor_from_bf16(g,ordered,n+guard);CHECK(converted[0]);
        CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_mixed_sdpa(g,t[3],converted[0],t[1],t[2],seq,56,scale,il,ol,0));
        CHECK(h3_gpu_mixed_record(g,56,0,0));
        CHECK(h3_gpu_mixed_sdpa(g,t[3],t[0],t[1],t[2],seq,56,scale,il,ol,0));
        CHECK(h3_gpu_mixed_record(g,56,1,0));CHECK(h3_gpu_submit(g));
        CHECK(!h3_gpu_mixed_step_finish(g,0)); // later valid block must not hide failure
        CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_mixed_sdpa(g,t[3],t[0],t[1],t[2],seq,56,scale,il,ol,0));
        CHECK(h3_gpu_mixed_record(g,56,0,1));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_mixed_step_finish(g,1));
        h3_gpu_tensor_free(converted[0]);converted[0]=NULL;
    }
    h3_gpu_stats stats;CHECK(h3_gpu_get_stats(g,&stats));
    printf(",\"peak_live_bytes\":%llu,\"direct_dispatches\":%llu,\"submissions\":%llu,\"pass\":true}\n",
        (unsigned long long)stats.peak_live_bytes,(unsigned long long)stats.direct_dispatches,(unsigned long long)stats.submissions);result=0;
cleanup:
    if(file)fclose(file);if(result&&g)h3_gpu_cancel(g);
    for(int j=0;j<4;j++){h3_gpu_tensor_free(t[j]);h3_gpu_tensor_free(converted[j]);}
    h3_gpu_free(g);free(host);free(ref);free(ordered);free(converted_ref);free(refwork);return result;
}
