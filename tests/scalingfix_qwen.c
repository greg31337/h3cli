#include "src/conditioning/text_encoder.h"
#include "src/conditioning/tokenizer.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>
static void die(const char *s) { fprintf(stderr,"%s\n",s);exit(1); }
static void *read_data(const char *dir,const char *name,size_t n) {
    char p[4096];snprintf(p,sizeof(p),"%s/%s",dir,name);FILE *f=fopen(p,"rb");void *v=malloc(n?n:1);
    if(!f||!v||fread(v,1,n,f)!=n||fgetc(f)!=EOF)die(p);fclose(f);return v;
}
static double now(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+t.tv_nsec*1e-9; }
int main(int argc,char **argv) {
    if(argc!=4)die("usage: scalingfix_qwen WEIGHTS FIXTURE_OR_--plain PROMPT_OR_OUTPUT");
    char error[512];h3_text_embedding out;double start=now();int ok;
    if(!strcmp(argv[2],"--plain")) {
        char path[4096];snprintf(path,sizeof(path),"%s/tokenizer.json",argv[1]);
        h3_tokenizer *tok=h3_tokenizer_load(path,error,sizeof(error));if(!tok)die(error);
        uint32_t *ids;size_t n;if(!h3_tokenizer_encode(tok,argv[3],1,&ids,&n,error,sizeof(error)))die(error);
        ok=h3_text_encode_bf16(argv[1],"src/metal/shaders.metal",ids,n,NULL,NULL,&out,error,sizeof(error));free(ids);h3_tokenizer_free(tok);
    } else {
        setenv("H3_TEST_QWEN_DUMP",argv[3],1);
        uint64_t *spec=read_data(argv[2],"spec.u64",32);size_t n=spec[0],ns=spec[1];
        uint32_t *ids=read_data(argv[2],"ids.u32",n*4),*pos=spec[2]?read_data(argv[2],"positions.u32",n*12):NULL;
        uint8_t *tags=spec[3]?read_data(argv[2],"tags.u8",n):NULL;
        h3_text_vision_span *spans=calloc(ns?ns:1,sizeof(*spans));
        for(size_t i=0;i<ns;i++) {char name[64];snprintf(name,sizeof(name),"span-%zu.u64",i);uint64_t *ss=read_data(argv[2],name,16);spans[i].start=ss[0];spans[i].tokens=ss[1];free(ss);
            snprintf(name,sizeof(name),"vision-%zu.bf16",i);spans[i].embeddings=read_data(argv[2],name,spans[i].tokens*5120*2);
            for(int j=0;j<3;j++){snprintf(name,sizeof(name),"deepstack-%zu-%d.bf16",i,j);spans[i].deepstack[j]=read_data(argv[2],name,spans[i].tokens*5120*2);}
        }
        ok=pos ? h3_text_encode_multimodal_layers_bf16(argv[1],"src/metal/shaders.metal",ids,n,spans,ns,pos,tags,50,NULL,NULL,&out,error,sizeof(error)) : h3_text_encode_bf16(argv[1],"src/metal/shaders.metal",ids,n,NULL,NULL,&out,error,sizeof(error));
        for(size_t i=0;i<ns;i++){free((void*)spans[i].embeddings);for(int j=0;j<3;j++)free((void*)spans[i].deepstack[j]);}
        free(spec);free(spans);free(ids);free(pos);free(tags);
    }
    if(!ok)die(error);
    if(strcmp(argv[2],"--plain")) {
        char path[4096];snprintf(path,sizeof(path),"%s/final.bf16",argv[3]);FILE *f=fopen(path,"wb");
        size_t n=out.tokens*out.width;if(!f || fwrite(out.values,2,n,f)!=n || fclose(f))die("cannot save final conditioning");
    }
    struct rusage r;getrusage(RUSAGE_SELF,&r);
    printf("{\"seconds\":%.9g,\"tokens\":%zu,\"gpu_seconds\":%.9g,\"peak_live_bytes\":%llu,\"allocated_bytes\":%llu,\"max_rss\":%ld}\n",now()-start,out.tokens,out.gpu_stats.gpu_seconds,(unsigned long long)out.gpu_stats.peak_live_bytes,(unsigned long long)out.gpu_stats.allocated_bytes,r.ru_maxrss);
    h3_text_embedding_free(&out);return 0;
}
