#ifndef H3_PROFILE_H
#define H3_PROFILE_H
#include "src/gpu.h"
void h3_profile_memory(h3_gpu *gpu,const char *phase,int step,int block,double seconds);
typedef struct { double started; uint64_t signpost; h3_gpu_stats before; } h3_profile_span;
int h3_profile_steps_enabled(void);
h3_profile_span h3_profile_component_begin(h3_gpu *gpu,const char *name,unsigned block,int step);
int h3_profile_component_end(h3_gpu *gpu,h3_profile_span span,const char *name,unsigned block,int step);
void h3_profile_step(h3_gpu *gpu,double started,int step,int total,int evaluated,h3_gpu_stats before);
#endif
