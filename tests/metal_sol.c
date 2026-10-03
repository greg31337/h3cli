/* SOL numerical/guard oracle. Generated data or a saved sequence-major H3 QKV.
 * Usage: SEQUENCE HEADS Q_BLOCK KV_BLOCK MIN_EXACT INPUT_LAYOUT PATTERN [FILE]
 * PATTERN random|constant-key|outlier|file. Diagnostic only, no denoiser steps. */
#include "src/gpu.h"
#include "src/metal/metal_fp16.h"
#include <math.h>
#include <time.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static double now(void){struct timespec ts;clock_gettime(CLOCK_MONOTONIC,&ts);return (double)ts.tv_sec+(double)ts.tv_nsec*1e-9;}
static float fp(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
static uint16_t bf(float f){uint32_t u;memcpy(&u,&f,4);u+=0x7fff+((u>>16)&1);return (uint16_t)(u>>16);}
static float random_value(uint32_t *s){*s^=*s<<13;*s^=*s>>17;*s^=*s<<5;return (float)((int)(*s%65537)-32768)/32768.f;}
static uint64_t hash(const uint16_t *p,size_t n){uint64_t x=14695981039346656037ull;for(size_t i=0;i<n;i++){x^=p[i];x*=1099511628211ull;}return x;}
static size_t at(unsigned row,unsigned h,unsigned d,unsigned seq,unsigned heads,int il){return il?((size_t)h*seq+row)*128+d:((size_t)row*heads+h)*128+d;}
/* Independent scalar double-softmax oracle for the approximate operator. It
 * explicitly enumerates exact tokens and weighted centroid entries, without
 * the shader's split partial attention or online-softmax merge. */
static double cpu_oracle(const uint16_t *inputs[3],const uint16_t *out,unsigned seq,
    unsigned heads,int il,h3_metal_attention_options o,const h3_sol_layout *layout,const h3_metal_fp16_range *ranges){
    unsigned nq=layout->query_blocks,nk=layout->key_blocks;
    float *summary[3]={0};double err=0,energy=0;
    for(unsigned j=0;j<3;j++) {
        unsigned count=j?nk:nq,bs=(unsigned)(j?o.kv_block:o.q_block);
        summary[j]=calloc((size_t)heads*count*128,sizeof(float));if(!summary[j])goto failed;
        for(unsigned h=0;h<heads;h++)for(unsigned b=0;b<count;b++)for(unsigned d=0;d<128;d++) {
            unsigned len=seq-b*bs;if(len>bs)len=bs;float sum=0;
            for(unsigned r=0;r<len;r++)sum+=fp(inputs[j][at(b*bs+r,h,d,seq,heads,il)]);
            summary[j][((size_t)h*count+b)*128+d]=fp(bf(sum/(float)len));
        }
    }
    for(unsigned sample=0;sample<48;sample++) {
        unsigned row=sample*(seq-1)/47,h=sample%heads,d=sample*19%128,qb=row/(unsigned)o.q_block;
        float mu=0,var=0,scale=1.f/sqrtf(128.f),ls=scale*1.4426950408889634f;
        for(unsigned c=0;c<128;c++) {
            float sum=0,sq=0;for(unsigned b=0;b<nk;b++){float x=summary[1][((size_t)h*nk+b)*128+c];sum+=x;sq+=x*x;}
            float mean=sum/(float)nk,q=summary[0][((size_t)h*nq+qb)*128+c];mu+=q*mean;var+=q*q*fmaxf(sq/(float)nk-mean*mean,0);
        }
        float threshold=mu*ls+o.tau*sqrtf(var*ls*ls+1e-6f);
        double maximum=-INFINITY,sum=0,value=0;
        for(unsigned b=0;b<nk;b++) {
            float dot=0;for(unsigned c=0;c<128;c++)dot+=summary[0][((size_t)h*nq+qb)*128+c]*summary[1][((size_t)h*nk+b)*128+c];
            h3_sol_block q=layout->query[qb],k=layout->key[b];unsigned floor_count=(unsigned)ceilf(o.min_exact*(float)nk);
            int local=q.first_frame>=0&&k.first_frame>=0&&q.first_frame<=k.last_frame+o.local_radius&&k.first_frame<=q.last_frame+o.local_radius;
            int exact=(ranges&&(ranges[h].flags&2))||q.protect||k.protect||local||((uint64_t)(b+1)*floor_count/nk!=(uint64_t)b*floor_count/nk)||k.rows<(unsigned)o.kv_block||dot*ls>threshold;
            unsigned count=exact?k.rows:1;
            for(unsigned r=0;r<count;r++) {
                double score=0;for(unsigned c=0;c<128;c++)score+=(double)fp(inputs[0][at(row,h,c,seq,heads,il)])*(exact?fp(inputs[1][at(b*(unsigned)o.kv_block+r,h,c,seq,heads,il)]):summary[1][((size_t)h*nk+b)*128+c]);
                score*=scale;if(!exact)score+=log((double)o.kv_block);
                double v=exact?fp(inputs[2][at(b*(unsigned)o.kv_block+r,h,d,seq,heads,il)]):summary[2][((size_t)h*nk+b)*128+d];
                double next=fmax(maximum,score),a=exp(maximum-next),w=exp(score-next);sum=sum*a+w;value=value*a+w*v;maximum=next;
            }
        }
        double ref=value/sum,diff=ref-fp(out[((size_t)row*heads+h)*128+d]);err+=diff*diff;energy+=ref*ref;
    }
    for(unsigned j=0;j<3;j++)free(summary[j]);return sqrt(err/fmax(energy,1e-30));
failed:
    for(unsigned j=0;j<3;j++)free(summary[j]);return INFINITY;
}
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);goto cleanup;}}while(0)
int main(int argc,char **argv) {
    if(argc<8)return 2;
    unsigned seq=(unsigned)strtoul(argv[1],0,10),heads=(unsigned)strtoul(argv[2],0,10);
    h3_metal_attention_options options=H3_METAL_ATTENTION_DEFAULT;
    options.q_block=atoi(argv[3]);options.kv_block=atoi(argv[4]);options.min_exact=strtof(argv[5],0);
    int mixed=getenv("H3_TEST_SOL_FP16")!=NULL;
    if(mixed){options.precision=1;options.candidate=1;}
    int recovery=!strcmp(argv[7],"recovery"),invalid=!strcmp(argv[7],"nonfinite");
    h3_metal_fp16_range ranges[56]={0};
    int il=atoi(argv[6]),file_mode=!strcmp(argv[7],"file");
    if(!h3_metal_fp16_shape_valid(seq,heads)||heads>56||(il!=0&&il!=1)||(file_mode&&argc!=9)||
       (!file_mode&&!recovery&&!invalid&&strcmp(argv[7],"random")&&strcmp(argv[7],"constant-key")&&strcmp(argv[7],"outlier")))return 2;
    char error[4096]={0};h3_gpu *g=NULL;h3_gpu_tensor *t[5]={0};uint16_t *data=NULL,*ordered=NULL,*ref=NULL;FILE *f=NULL;
    h3_sol_layout layout={0};int result=1;size_t n=(size_t)seq*heads*128,guard=128;
    CHECK(h3_metal_options_valid(options,error,sizeof(error)));
    layout.sequence=seq;layout.query_blocks=(seq+(unsigned)options.q_block-1)/(unsigned)options.q_block;
    layout.key_blocks=(seq+(unsigned)options.kv_block-1)/(unsigned)options.kv_block;
    layout.query=calloc(layout.query_blocks,sizeof(*layout.query));layout.key=calloc(layout.key_blocks,sizeof(*layout.key));CHECK(layout.query&&layout.key);
    // Distinct protected rows deliberately straddle routing/tile boundaries.
    for(unsigned r=0;r<seq;r++) {
        unsigned protected_row=r<7||seq-r<=7||(r>=seq/3&&r<seq/3+9);
        layout.protected_rows+=protected_row;
        h3_sol_block *meta[]={&layout.query[r/(unsigned)options.q_block],&layout.key[r/(unsigned)options.kv_block]};
        for(unsigned j=0;j<2;j++){if(!meta[j]->rows)meta[j]->first_frame=(int)(r/16);meta[j]->last_frame=(int)(r/16);meta[j]->protect|=protected_row;meta[j]->rows++;}
    }
    g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);CHECK(h3_gpu_native_attention_layout(g,&layout));
    data=malloc((n+guard)*2);ordered=malloc((n+guard)*2);ref=malloc(n*2);CHECK(data&&ordered&&ref);
    if(file_mode){f=fopen(argv[8],"rb");CHECK(f);}
    uint32_t rng=0x731add12;uint64_t hashes[3];
    for(unsigned j=0;j<3;j++) {
        for(size_t i=0;i<n;i++)data[i]=bf(j==1&&!strcmp(argv[7],"constant-key")?.125f:random_value(&rng));
        if(!strcmp(argv[7],"outlier")&&j<2)for(size_t i=0;i<n;i+=128*73)data[i]=bf((i/(128*73))%2?-20.f:20.f);
        if(recovery&&j==0){data[0]=bf(0x1p30f);data[128*heads]=bf(0x1p-20f);}
        if(invalid&&j==0)data[0]=0x7fc0;
        if(file_mode)CHECK(fread(data,2,n,f)==n);
        for(size_t i=n;i<n+guard;i++)data[i]=0x7f7f;
        for(size_t i=0;i<n;i++){size_t row=i/(heads*128),h=(i/128)%heads,d=i%128;ordered[il?(h*seq+row)*128+d:i]=data[i];}
        memcpy(ordered+n,data+n,guard*2);hashes[j]=hash(ordered,n+guard);t[j]=h3_gpu_tensor_from_bf16(g,ordered,n+guard);CHECK(t[j]);
    }
    if(f){CHECK(fgetc(f)==EOF);fclose(f);f=NULL;}
    for(size_t i=0;i<n;i++)data[i]=0x7fc0;
    t[3]=h3_gpu_tensor_from_bf16(g,data,n+guard);t[4]=h3_gpu_tensor_from_bf16(g,data,n+guard);CHECK(t[3]&&t[4]);
    float scale=1.f/sqrtf(128.f);
    CHECK(h3_gpu_begin(g));CHECK(mixed?h3_gpu_mixed_sdpa(g,t[3],t[0],t[1],t[2],seq,heads,scale,il,0,options.q_block==64):h3_gpu_native_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,heads,128,scale,il,0,-1));CHECK(h3_gpu_submit(g));
    CHECK(h3_gpu_tensor_read_bf16(t[3],ref,n));
    // Exercise scratch reuse across in-flight layers in one command buffer.
    CHECK(h3_gpu_begin(g));
    CHECK(mixed?h3_gpu_mixed_sol(g,t[4],t[0],t[1],t[2],seq,heads,scale,il,0,options,1,0):h3_gpu_native_sol_bf16(g,t[4],t[0],t[1],t[2],seq,heads,scale,il,0,options,1,0));
    CHECK(mixed?h3_gpu_mixed_sol(g,t[4],t[0],t[1],t[2],seq,heads,scale,il,0,options,2,0):h3_gpu_native_sol_bf16(g,t[4],t[0],t[1],t[2],seq,heads,scale,il,0,options,2,0));
    CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(t[4],data,n+guard));
    if(mixed) {
        CHECK(h3_gpu_mixed_range_report(g,ranges,heads));
        if(invalid) {
            CHECK(ranges[0].flags&1);
            for(size_t i=0;i<n;i++)CHECK(data[i]==0x7fc0);
            for(size_t i=n;i<n+guard;i++)CHECK(data[i]==0x7f7f);
            printf("{\"invalid_commit_rejected\":true}\n");result=0;goto cleanup;
        }
        for(unsigned h=0;h<heads;h++)CHECK(!(ranges[h].flags&1)&&!ranges[h].output_invalid);
        if(recovery)CHECK(ranges[0].flags&2);
    }
    double dd=0,xx=0,yy=0,xy=0,pdd=0,pxx=0,pmax=0,ppeak=0,maximum=0,peak=0;
    for(size_t i=0;i<n;i++) {
        double a=fp(ref[i]),b=fp(data[i]),d=a-b;CHECK(isfinite(b));
        dd+=d*d;xx+=a*a;yy+=b*b;xy+=a*b;maximum=fmax(maximum,fabs(d));peak=fmax(peak,fabs(a));
        unsigned row=(unsigned)(i/(heads*128));if(layout.query[row/(unsigned)options.q_block].protect||(mixed&&(ranges[(i/128)%heads].flags&2))){pdd+=d*d;pxx+=a*a;pmax=fmax(pmax,fabs(d));ppeak=fmax(ppeak,fabs(a));}
    }
    for(size_t i=n;i<n+guard;i++)CHECK(data[i]==0x7f7f);
    const uint16_t *inputs[3];for(unsigned j=0;j<3;j++){inputs[j]=h3_gpu_tensor_contents(t[j]);CHECK(inputs[j]&&hash(inputs[j],n+guard)==hashes[j]);}
    double oracle=seq<=4097?cpu_oracle(inputs,data,seq,heads,il,options,&layout,mixed?ranges:NULL):-1;
    CHECK(oracle<=.01);
    double rel=sqrt(dd/fmax(xx,1e-30)),cosine=xx*yy>0?xy/sqrt(xx*yy):1,protected_rel=sqrt(pdd/fmax(pxx,1e-30));
    int all_exact=options.min_exact==1||!strcmp(argv[7],"constant-key");
    int pass=protected_rel<=.01&&pmax<=.02*fmax(1.,ppeak)&&(!all_exact||(rel<=.01&&maximum<=.02*fmax(1.,peak)));
    /* Output layout changes only indexing, including tail rows. Reuse the
     * former dense destination and require bit-identical logical SOL output. */
    CHECK(h3_gpu_begin(g));
    CHECK(mixed?h3_gpu_mixed_sol(g,t[3],t[0],t[1],t[2],seq,heads,scale,il,1,options,3,0):h3_gpu_native_sol_bf16(g,t[3],t[0],t[1],t[2],seq,heads,scale,il,1,options,3,0));
    CHECK(h3_gpu_submit(g));CHECK(h3_gpu_tensor_read_bf16(t[3],ordered,n+guard));
    for(unsigned r=0;r<seq;r++)for(unsigned h=0;h<heads;h++)for(unsigned d=0;d<128;d++)
        CHECK(data[((size_t)r*heads+h)*128+d]==ordered[((size_t)h*seq+r)*128+d]);
    for(size_t i=n;i<n+guard;i++)CHECK(ordered[i]==0x7f7f);
    printf("{\"sequence\":%u,\"heads\":%u,\"q_block\":%d,\"kv_block\":%d,\"min_exact\":%.9g,\"pattern\":\"%s\",\"relative_l2\":%.9g,\"cosine\":%.9g,\"protected_relative_l2\":%.9g,\"cpu_oracle_relative_l2\":%.9g,\"max_abs\":%.9g,\"correctness_pass\":%s,\"conservative_quality_pass\":%s}\n",
        seq,heads,options.q_block,options.kv_block,(double)options.min_exact,argv[7],rel,cosine,protected_rel,oracle,maximum,pass?"true":"false",rel<=.05&&cosine>=.998?"true":"false");
    if(mixed)printf("{\"mixed_sol_recipe\":%d,\"recovery_head0\":%s}\n",H3_METAL_SOL_VERSION,(ranges[0].flags&2)?"true":"false");
    h3_gpu_native_sol_report(g,0);CHECK(pass);
    const char *repeat_env=getenv("H3_SOL_BENCH_REPEATS");int repeats=repeat_env?atoi(repeat_env):0;
    CHECK(repeats>=0&&repeats<=5);
    if(repeats) {
        double dense_times[5],sol_times[5];
        for(int rep=0;rep<repeats;rep++)for(int order=0;order<2;order++) {
            int route=(rep+order)%2;double start=now();CHECK(h3_gpu_begin(g));
            if(route)CHECK(mixed?h3_gpu_mixed_sol(g,t[4],t[0],t[1],t[2],seq,heads,scale,il,0,options,4,0):
                h3_gpu_native_sol_bf16(g,t[4],t[0],t[1],t[2],seq,heads,scale,il,0,options,4,0));
            else CHECK(mixed?h3_gpu_mixed_sdpa(g,t[3],t[0],t[1],t[2],seq,heads,scale,il,0,options.q_block==64):
                h3_gpu_native_sdpa_bf16(g,t[3],t[0],t[1],t[2],seq,heads,128,scale,il,0,-1));
            CHECK(h3_gpu_submit(g));double elapsed=now()-start;
            if(route)sol_times[rep]=elapsed;else dense_times[rep]=elapsed;
            if(mixed){CHECK(h3_gpu_mixed_range_report(g,ranges,heads));for(unsigned h=0;h<heads;h++)CHECK(!(ranges[h].flags&1)&&!ranges[h].output_invalid);}
        }
        printf("{\"dense_seconds\":[");for(int i=0;i<repeats;i++)printf("%s%.9f",i?",":"",dense_times[i]);
        printf("],\"sol_seconds\":[");for(int i=0;i<repeats;i++)printf("%s%.9f",i?",":"",sol_times[i]);printf("]}\n");
    }
    result=0;
cleanup:
    if(f)fclose(f);if(result&&g)h3_gpu_cancel(g);
    for(unsigned j=0;j<5;j++)h3_gpu_tensor_free(t[j]);h3_gpu_free(g);
    h3_sol_layout_free(&layout);free(data);free(ordered);free(ref);return result;
}
