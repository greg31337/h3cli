#ifndef H3_ANE_SPLIT_H
#define H3_ANE_SPLIT_H
#include "src/gpu.h"
typedef struct h3_ane_split h3_ane_split;
h3_ane_split *h3_ane_split_create(h3_metal_attention_options options,int k,int n);
void h3_ane_split_free(h3_ane_split *split);
int h3_ane_split_linear(h3_ane_split *split,h3_gpu *gpu,h3_gpu_tensor *out,
    h3_gpu_tensor *input,h3_gpu_tensor *weight,unsigned rows,unsigned block,int step);
void h3_ane_split_export(h3_ane_split *split,uint64_t *decided,uint32_t rows[50]);
int h3_ane_split_restore(h3_ane_split *split,uint64_t decided,const uint32_t rows[50]);
#endif
