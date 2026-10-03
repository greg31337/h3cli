/* Bounded test-only native operation replay. Inputs retain checkpoint dtype. */
#include "src/gpu.h"
#include "src/sglang/sglang.h"
#include "src/conditioning/text_encoder.h"
#include <stdio.h>
#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <math.h>
static unsigned number(const char *s) {
    char *end;errno=0;unsigned long v=strtoul(s,&end,10);
    if(errno||!*s||*end||!v||v>1000000){fprintf(stderr,"invalid dimension\n");exit(2);}return (unsigned)v;
}
int main(int argc,char **argv) {
    if(argc==7 && !strcmp(argv[1],"norm")) {
        unsigned rows=number(argv[5]),width=number(argv[6]);size_t count=(size_t)rows*width;
        if(count>128*1024*1024||width%4)return 2;
        char error[1024]={0};int previous=h3_sglang_exchange(1);
        h3_gpu*g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));h3_sglang_exchange(previous);
        if(!g){fprintf(stderr,"%s\n",error);return 1;}
        h3_gpu_tensor*x=h3_gpu_tensor_load_bf16(g,argv[2],0,count),*w=h3_gpu_tensor_load_bf16(g,argv[3],0,width),
            *y=h3_gpu_tensor_new_bf16(g,count);uint16_t*host=malloc(count*2);
        int ok=x&&w&&y&&host&&h3_gpu_begin(g)&&h3_gpu_rms_norm_bf16(g,y,x,w,rows,width,1e-5f)&&
            h3_gpu_submit(g)&&h3_gpu_tensor_read_bf16(y,host,count);
        if(ok){FILE*f=fopen(argv[4],"wbx");ok=f&&fwrite(host,2,count,f)==count;if(f&&fclose(f))ok=0;}
        if(!ok)fprintf(stderr,"norm replay failed: %s\n",h3_gpu_error(g));
        free(host);h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);h3_gpu_tensor_free(y);h3_gpu_free(g);return ok?0:1;
    }
    if(argc==10 && !strcmp(argv[1],"qkv-norm")) {
        unsigned rows=number(argv[7]),heads=number(argv[8]),dim=number(argv[9]);
        size_t count=(size_t)rows*heads*dim;
        if(dim!=128||count>128*1024*1024)return 2;
        char error[1024]={0};int previous=h3_sglang_exchange(1);
        h3_gpu*g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));h3_sglang_exchange(previous);
        if(!g){fprintf(stderr,"%s\n",error);return 1;}
        h3_gpu_tensor*x=h3_gpu_tensor_load_bf16(g,argv[2],0,3*count),
            *qw=h3_gpu_tensor_load_bf16(g,argv[3],0,dim),*kw=h3_gpu_tensor_load_bf16(g,argv[4],0,dim),
            *q=h3_gpu_tensor_new_bf16(g,count),*k=h3_gpu_tensor_new_bf16(g,count),*v=h3_gpu_tensor_new_bf16(g,count);
        uint16_t*host=malloc(count*2);
        int ok=x&&qw&&kw&&q&&k&&v&&host&&h3_gpu_begin(g)&&
            h3_gpu_grouped_qkv_rope_bf16(g,q,k,v,x,qw,kw,qw,qw,rows,heads,dim,0,1e-5f)&&h3_gpu_submit(g);
        h3_gpu_tensor*t[2]={q,k};
        for(int i=0;ok&&i<2;i++){
            ok=h3_gpu_tensor_read_bf16(t[i],host,count);FILE*f=ok?fopen(argv[5+i],"wbx"):NULL;
            ok=f&&fwrite(host,2,count,f)==count;if(f&&fclose(f))ok=0;
        }
        if(!ok)fprintf(stderr,"Q/K norm replay failed: %s\n",h3_gpu_error(g));
        free(host);h3_gpu_tensor_free(x);h3_gpu_tensor_free(qw);h3_gpu_tensor_free(kw);
        h3_gpu_tensor_free(q);h3_gpu_tensor_free(k);h3_gpu_tensor_free(v);h3_gpu_free(g);return ok?0:1;
    }
    if(argc==10 && (!strcmp(argv[1],"attention")||!strcmp(argv[1],"gqa"))) {
        unsigned rows=number(argv[6]),heads=number(argv[7]),kv=number(argv[8]),dim=number(argv[9]);
        size_t nq=(size_t)rows*heads*dim,nk=(size_t)rows*kv*dim;
        if(nq>512*1024*1024||nk>512*1024*1024||heads%kv)return 2;
        char error[1024]={0};int previous=h3_sglang_exchange(1);
        h3_gpu*g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));h3_sglang_exchange(previous);
        if(!g){fprintf(stderr,"%s\n",error);return 1;}
        h3_gpu_tensor*q=h3_gpu_tensor_load_bf16(g,argv[2],0,nq),*k=h3_gpu_tensor_load_bf16(g,argv[3],0,nk),
            *v=h3_gpu_tensor_load_bf16(g,argv[4],0,nk),*y=h3_gpu_tensor_new_bf16(g,nq);
        uint16_t *host=malloc(nq*2);int ok=q&&k&&v&&y&&host&&h3_gpu_begin(g);
        if(ok)ok=!strcmp(argv[1],"gqa")?h3_gpu_gqa_causal_bf16(g,y,q,k,v,rows,heads,kv,dim,1/sqrtf(dim)):
            kv==heads&&h3_gpu_sdpa_bf16(g,y,q,k,v,rows,heads,dim,1/sqrtf(dim));
        ok=ok&&h3_gpu_submit(g)&&h3_gpu_tensor_read_bf16(y,host,nq);
        if(ok){FILE*f=fopen(argv[5],"wbx");ok=f&&fwrite(host,2,nq,f)==nq;if(f&&fclose(f))ok=0;}
        if(!ok)fprintf(stderr,"attention replay failed: %s\n",h3_gpu_error(g));
        free(host);h3_gpu_tensor_free(q);h3_gpu_tensor_free(k);h3_gpu_tensor_free(v);h3_gpu_tensor_free(y);h3_gpu_free(g);return ok?0:1;
    }
    if(argc==5 && !strcmp(argv[1],"text")) {
        struct stat st;if(stat(argv[3],&st)||st.st_size<=0||st.st_size>32768*4||st.st_size%4)return 2;
        size_t rows=(size_t)st.st_size/4;uint32_t *ids=malloc(rows*4);FILE*f=fopen(argv[3],"rb");
        int ok=ids&&f&&fread(ids,4,rows,f)==rows;if(f&&fclose(f))ok=0;
        if(!ok){free(ids);return 1;}
        char error[1024]={0};h3_text_embedding text={0};int previous=h3_sglang_exchange(1);
        ok=h3_text_encode_bf16(argv[2],"src/metal/shaders.metal",ids,rows,NULL,NULL,&text,error,sizeof(error));
        h3_sglang_exchange(previous);free(ids);
        if(ok){f=fopen(argv[4],"wbx");ok=f&&fwrite(text.values,2,text.tokens*text.width,f)==text.tokens*text.width;if(f&&fclose(f))ok=0;}
        if(!ok)fprintf(stderr,"text replay failed: %s\n",error);
        h3_text_embedding_free(&text);return ok?0:1;
    }
    int head_mode=argc>1&&!strcmp(argv[1],"linear-f32-head");
    int bias_mode=argc>1 && (!strcmp(argv[1],"linear-bias")||!strcmp(argv[1],"linear-f32-bias")||head_mode);
    int f32_mode=argc>1 && (!strcmp(argv[1],"linear-f32-bias")||head_mode);
    if((!bias_mode&&(argc!=9||strcmp(argv[1],"linear")))||(bias_mode&&argc!=(head_mode?11:10))) {
        fprintf(stderr,"usage: %s linear input.bf16 checkpoint offset output.bf16 rows input_dim output_dim\n       %s text text_encoder tokens.u32 output.bf16\n",argv[0],argv[0]);return 2;
    }
    int shift=bias_mode?1:0;
    unsigned rows=number(argv[6+shift]),k=number(argv[7+shift]),n=number(argv[8+shift]);
    if((uint64_t)rows*k>512*1024*1024 || (uint64_t)rows*n>512*1024*1024 || (uint64_t)k*n>512*1024*1024)return 2;
    char *end;errno=0;uint64_t offset=strtoull(argv[4],&end,10);if(errno||!*argv[4]||*end)return 2;
    uint64_t bias_offset=0;
    if(bias_mode){errno=0;bias_offset=strtoull(argv[5],&end,10);if(errno||!*argv[5]||*end)return 2;}
    char error[1024]={0};int previous=h3_sglang_exchange(1);
    h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));h3_sglang_exchange(previous);
    if(!g){fprintf(stderr,"%s\n",error);return 1;}
    h3_gpu_tensor *(*load)(h3_gpu*,const char*,uint64_t,size_t)=f32_mode?h3_gpu_tensor_load_f32:h3_gpu_tensor_load_bf16;
    h3_gpu_tensor *x=load(g,argv[2],0,(size_t)rows*k),*w=load(g,argv[3],offset,(size_t)k*n);
    h3_gpu_tensor *b=bias_mode?load(g,argv[3],bias_offset,n):NULL;
    h3_gpu_tensor *y=f32_mode?h3_gpu_tensor_new_f32(g,(size_t)rows*n):h3_gpu_tensor_new_bf16(g,(size_t)rows*n);
    size_t count=(size_t)rows*n,item=f32_mode?4:2;void *host=malloc(count*item);
    int ok=x&&w&&y&&(!bias_mode||b)&&host&&h3_gpu_begin(g);
#ifndef __APPLE__
    if(ok&&head_mode)ok=h3_gpu_sglang_head_f32(g,y,x,w,b,rows,k,n,number(argv[10]));
    else
#endif
    if(ok)ok=f32_mode?h3_gpu_linear_f32(g,y,x,w,b,rows,k,n):h3_gpu_linear_bf16(g,y,x,w,b,rows,k,n);
    ok=ok&&h3_gpu_submit(g)&&
        (f32_mode?h3_gpu_tensor_read_f32(y,host,count):h3_gpu_tensor_read_bf16(y,host,count));
    if(ok){FILE*f=fopen(argv[5+shift],"wbx");ok=f&&fwrite(host,item,count,f)==count;if(f&&fclose(f))ok=0;}
    if(!ok)fprintf(stderr,"replay failed: %s\n",h3_gpu_error(g));
    free(host);h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);h3_gpu_tensor_free(b);h3_gpu_tensor_free(y);h3_gpu_free(g);
    return ok?0:1;
}
