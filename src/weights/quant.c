#include "src/weights/quant.h"
#include "src/denoise/attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <sys/stat.h>
#include <unistd.h>
static _Thread_local h3_quant_scope request;
int h3_quant_verify(void) {
    const char *value=getenv("H3_QUANT_VERIFY");
    return value&&!strcmp(value,"1");
}
const char *h3_quant_name(int mode) {
    return mode==0?"off":mode==1?"fp8":mode==2?"nvfp4":"invalid";
}
int h3_quant_parse(const char *text,int *mode) {
    if(!text||!mode)return 0;
    for(int i=0;i<3;i++)if(!strcmp(text,h3_quant_name(i))){*mode=i;return 1;}
    return 0;
}
int h3_quant_projection(const char *name,int requested) {
    if(!name||strncmp(name,"blocks.",7))return H3_QUANT_OFF;
    const char *p=name+7;
    if(*p<'0'||*p>'9')return H3_QUANT_OFF;
    while(*p>='0'&&*p<='9')p++;
    const char *suffixes[]={".attn.qkv_proj.weight",".attn.out_proj.weight",".mlp.fc1.weight",".mlp.fc2.weight"};
    for(size_t i=0;i<4;i++)if(!strcmp(p,suffixes[i]))return requested;
    return H3_QUANT_OFF;
}
unsigned h3_quant_recipe(int mode,int adaptive_cache) {
    return mode?(adaptive_cache?H3_QUANT_ADAPTIVE_VERSION:H3_QUANT_VERSION):0;
}
unsigned h3_quant_execution_recipe(int mode,int adaptive_cache,int attention) {
    return mode&&attention==H3_ATTENTION_SUBBLOCK?H3_QUANT_SUBBLOCK_VERSION:h3_quant_recipe(mode,adaptive_cache);
}
int h3_quant_projection_policy(const char *name,int requested,int adaptive_cache) {
    if(adaptive_cache&&name&&!strncmp(name,"blocks.0.",9))return H3_QUANT_OFF;
    return h3_quant_projection(name,requested);
}
int h3_quant_options(int mode,const char *cache,char *error,size_t size) {
    const char *message=NULL;
    const char *verify=getenv("H3_QUANT_VERIFY");
    if(mode<0||mode>2)message="cuda-denoise-quant must be off, fp8 or nvfp4";
    else if(cache&&(!*cache||!mode))message="cuda-denoise-quant-cache requires a nonempty path and quantized denoising";
    else if(mode&&verify&&strcmp(verify,"0")&&strcmp(verify,"1"))message="H3_QUANT_VERIFY must be 0 or 1";
    if(message){if(error&&size)snprintf(error,size,"%s",message);return 0;}
    /* Check the nearest existing parent without creating files or loading a
     * backend. The actual mkdir/open still checks for races at preparation. */
    if(mode&&cache){
        char *parent=strdup(cache);struct stat st;int ok=parent!=NULL;
        while(ok&&stat(parent,&st)){
            if(errno!=ENOENT){ok=0;break;}
            char *slash=strrchr(parent,'/');
            if(!slash){strcpy(parent,".");break;}
            if(slash==parent){parent[1]=0;break;}*slash=0;
        }
        if(ok)ok=!stat(parent,&st)&&S_ISDIR(st.st_mode)&&!access(parent,W_OK|X_OK);
        free(parent);
        if(!ok){if(error&&size)snprintf(error,size,"quantization cache path is not a writable directory: %s",cache);return 0;}
    }
    return 1;
}
h3_quant_scope h3_quant_exchange(h3_quant_scope next){h3_quant_scope old=request;request=next;return old;}
h3_quant_scope h3_quant_current(void){return request;}
#ifdef __APPLE__
int h3_quant_preflight(int mode,char *error,size_t size) {
    if(!mode)return 1;
    if(error&&size)snprintf(error,size,"cuda-denoise-quant requires the CUDA backend; Metal is unsupported");
    return 0;
}
#endif
