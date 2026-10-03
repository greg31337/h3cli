#include "src/backend.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static _Thread_local h3_backend_scope current;
const char *h3_weight_format_name(h3_weight_format f) {
    return f==H3_WEIGHT_BF16?"bf16":f==H3_WEIGHT_Q8?"q8":"invalid";
}
int h3_weight_format_parse(const char *s,h3_weight_format *f) {
    if(!s||!f)return 0;
    for(int i=0;i<2;i++)if(!strcmp(s,h3_weight_format_name((h3_weight_format)i))){*f=(h3_weight_format)i;return 1;}
    return 0;
}
const char *h3_metal_candidate_name(int c) {
    return c==0?"steel":c==1?"steel-routed":"invalid";
}
const char *h3_metal_precision_name(int p){return p==0?"bf16":p==1?"fp16":"invalid";}
const char *h3_metal_tier_name(int t){return t==0?"diagnostic":t==1?"reference-candidate":t==2?"preview-candidate":"invalid";}
int h3_metal_options_equal(h3_metal_attention_options a,h3_metal_attention_options b) {
    return a.candidate==b.candidate&&
        a.q_block==b.q_block&&a.kv_block==b.kv_block&&a.dense_layers==b.dense_layers&&
        a.local_radius==b.local_radius&&a.tau==b.tau&&a.min_exact==b.min_exact&&
        a.precision==b.precision&&a.tier==b.tier&&
        a.dense_steps==b.dense_steps&&a.dense_sigma==b.dense_sigma&&a.layout_fusion==b.layout_fusion&&
        a.ane_mode==b.ane_mode&&a.ane_rows==b.ane_rows&&a.ane_chunk==b.ane_chunk&&
        a.weight_format==b.weight_format&&a.q8_kernel==b.q8_kernel;
}
int h3_metal_options_valid(h3_metal_attention_options a,char *e,size_t n) {
    if((a.weight_format!=H3_WEIGHT_BF16&&a.weight_format!=H3_WEIGHT_Q8)||
       (a.weight_format==H3_WEIGHT_Q8&&a.ane_mode)||a.q8_kernel<0||a.q8_kernel>1||
       (a.q8_kernel&&a.weight_format!=H3_WEIGHT_Q8)) {
        if(e&&n)snprintf(e,n,"Metal weight format must be bf16 or q8; Q8 requires --metal-ane off; q8-kernel mpsgraph|simdgroup requires q8");
        return 0;
    }
    if(a.candidate<0||a.candidate>1||
       (a.q_block!=32&&a.q_block!=64)||
       (a.kv_block!=32&&a.kv_block!=64&&a.kv_block!=128)||
       a.dense_layers<0||a.dense_layers>50||a.local_radius<0||a.local_radius>10000||
       !isfinite(a.tau)||a.tau<0||a.tau>16||
       !isfinite(a.min_exact)||a.min_exact<0||a.min_exact>1||
       a.dense_steps<0||a.dense_steps>1000||!isfinite(a.dense_sigma)||
       (a.dense_sigma!=-1&&(a.dense_sigma<0||a.dense_sigma>1))||
       a.precision<0||a.precision>1||a.tier<0||a.tier>2||
       (a.precision==1&&a.candidate!=1)||a.layout_fusion<0||a.layout_fusion>1||
       (a.layout_fusion&&a.precision!=1)||a.ane_mode<0||a.ane_mode>3||
       a.ane_rows<0||a.ane_rows>16384||
       (a.ane_chunk!=256&&a.ane_chunk!=512&&a.ane_chunk!=1024)||
       (a.ane_mode&&(a.precision!=1||a.ane_rows<a.ane_chunk||a.ane_rows%a.ane_chunk))) {
        if(e&&n)snprintf(e,n,"invalid Metal options (fp16 requires steel-routed); ANE requires FP16 attention, mode off|serial|static|dynamic, chunk 256|512|1024 and a positive multiple-of-chunk row budget <=16384");
        return 0;
    }
    return 1;
}
const char *h3_metal_sol_dense_reason(h3_metal_attention_options o,unsigned block,
    int step,float video_sigma,float audio_sigma) {
    if(step<0||!isfinite(video_sigma)||!isfinite(audio_sigma)||
       video_sigma<0||video_sigma>1||audio_sigma<0||audio_sigma>1)return "invalid-noise";
    if(o.min_exact>=1)return "all-exact";
    if(step<o.dense_steps)return "early-evaluation";
    if(o.dense_sigma>=0&&fmaxf(video_sigma,audio_sigma)>=o.dense_sigma)return "high-noise";
    if(block<(unsigned)o.dense_layers)return "early-layer";
    return NULL;
}
const char *h3_backend_name(h3_backend b) {
    return b==H3_BACKEND_MPSGRAPH_REFERENCE?"mpsgraph":b==H3_BACKEND_METAL?"metal":"invalid";
}
const char *h3_attention_mode_name(h3_attention_mode a) {
    return a==H3_ATTN_DENSE?"dense":a==H3_ATTN_SOL?"sol":"invalid";
}
int h3_backend_parse(const char *s,h3_backend *b) {
    if(!s||!b)return 0;
    for(int i=0;i<2;i++)if(!strcmp(s,h3_backend_name((h3_backend)i))){*b=(h3_backend)i;return 1;}
    return 0;
}
int h3_attention_mode_parse(const char *s,h3_attention_mode *a) {
    if(!s||!a)return 0;
    for(int i=0;i<2;i++)if(!strcmp(s,h3_attention_mode_name((h3_attention_mode)i))){*a=(h3_attention_mode)i;return 1;}
    return 0;
}
int h3_backend_preflight(h3_backend_scope s,int explicit_selection,char *e,size_t n) {
    const char *message=NULL;
    if(s.backend!=H3_BACKEND_MPSGRAPH_REFERENCE&&s.backend!=H3_BACKEND_METAL)
        message="backend must be mpsgraph or metal";
    else if(s.attention!=H3_ATTN_DENSE&&s.attention!=H3_ATTN_SOL)
        message="attention must be dense or sol";
    else if(s.backend==H3_BACKEND_MPSGRAPH_REFERENCE&&
            (s.attention!=H3_ATTN_DENSE||s.metal.weight_format||s.metal.q8_kernel||explicit_selection&8))
        message="Metal/SOL options require --backend metal";
#ifndef __APPLE__
    else if(explicit_selection||s.backend!=H3_BACKEND_MPSGRAPH_REFERENCE)
        message="--backend and --metal-attention select Apple Metal execution; use CUDA options on Linux";
#else
    (void)explicit_selection;
    if(!message&&s.backend==H3_BACKEND_METAL) {
        if(!h3_metal_options_valid(s.metal,e,n))return 0;
    }
#endif
    if(message){if(e&&n)snprintf(e,n,"%s",message);return 0;}
    return 1;
}
h3_backend_scope h3_backend_current(void){return current;}
h3_backend_scope h3_backend_exchange(h3_backend_scope s){h3_backend_scope old=current;current=s;return old;}
int h3_test_evaluation_budget(int evaluations,char *e,size_t n) {
    const char *limit=getenv("H3_TEST_MAX_EVALUATIONS");
    if(!limit)return 1;
    int maximum=!strcmp(limit,"6")?6:!strcmp(limit,"50")?50:0;
    if(!maximum||evaluations<0||evaluations>maximum){
        if(e&&n)snprintf(e,n,"test evaluation budget exceeded: requested %d; H3_TEST_MAX_EVALUATIONS must be 6 or explicitly 50",evaluations);
        return 0;
    }
    return 1;
}
