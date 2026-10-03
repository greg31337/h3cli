#include "src/denoise/attention.h"
#include "src/denoise/subblock.h"
#include "src/cuda/cuda_sol_policy.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#if (defined(H3_CUDA_USE_SAGE) || defined(H3_CUDA_USE_SOL) || defined(H3_CUDA_USE_SUBBLOCK)) && !defined(__APPLE__)
#include "src/device.h"
#endif
static _Thread_local int requested;
const char *h3_attention_name(int mode) {
    return mode==0?"default":mode==1?"sage2++":mode==2?"sage3":mode==3?"sol":mode==4?"subblock":"invalid";
}
int h3_attention_parse(const char *text,int *mode) {
    if(!text||!mode)return 0;
    for(int i=0;i<5;i++)if(!strcmp(text,h3_attention_name(i))){*mode=i;return 1;}
    return 0;
}
int h3_attention_options(int mode,char *error,size_t size) {
    const char *message=NULL;
    if(mode<0||mode>4)message="cuda-attention must be default, sage2++, sage3, sol or subblock";
    if(message){if(error&&size)snprintf(error,size,"%s",message);return 0;}
    return 1;
}
int h3_attention_preflight(int mode,char *error,size_t size) {
    if(!h3_attention_options(mode,error,size))return 0;
    if(!mode)return 1;
#ifdef __APPLE__
    if(error&&size)snprintf(error,size,"cuda-attention requires CUDA; Metal attention is unchanged");
#else
#ifndef H3_CUDA_USE_SUBBLOCK
    if(mode==H3_ATTENTION_SUBBLOCK){if(error&&size)snprintf(error,size,"SubBlock requires a CUDA_SUBBLOCK=1 build");return 0;}
#endif
#ifndef H3_CUDA_USE_SOL
    if(mode==H3_ATTENTION_SOL){if(error&&size)snprintf(error,size,"CUDA SOL requires a CUDA_SOL=1 build");return 0;}
#endif
#ifndef H3_CUDA_USE_SAGE
    if(mode==H3_ATTENTION_SAGE2||mode==H3_ATTENTION_SAGE3){if(error&&size)snprintf(error,size,"Sage attention requires a CUDA_SAGE=1 build");return 0;}
#endif
#if defined(H3_CUDA_USE_SAGE) || defined(H3_CUDA_USE_SOL) || defined(H3_CUDA_USE_SUBBLOCK)
    h3_device_info device;
    if(!h3_device_query(&device,error,size))return 0;
    int minimum=mode==H3_ATTENTION_SUBBLOCK?13000:12080;
    if(device.cuda_compute_major==12&&device.cuda_compute_minor==0&&device.cuda_runtime_version>=minimum)return 1;
    if(error&&size)snprintf(error,size,"cuda-attention requires a qualified SM120 device and CUDA %s or newer",mode==H3_ATTENTION_SUBBLOCK?"13.0":"12.8");
#endif
#endif
    return 0;
}
int h3_attention_exchange(int mode){int previous=requested;requested=mode;return previous;}
int h3_attention_current(void){return requested;}

unsigned h3_attention_recipe(int mode){return mode==H3_ATTENTION_SOL?H3_CUDA_SOL_VERSION:mode?H3_ATTENTION_VERSION:0;}
unsigned h3_attention_execution_recipe(int mode,int quant){return mode==H3_ATTENTION_SUBBLOCK&&quant?H3_SUBBLOCK_QUANT_VERSION:h3_attention_recipe(mode);}
unsigned h3_attention_plan(int mode){return mode==H3_ATTENTION_SOL?H3_CUDA_SOL_PLAN_VERSION:mode?H3_ATTENTION_PLAN_VERSION:0;}
