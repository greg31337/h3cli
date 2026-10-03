#ifndef H3_ATTENTION_H
#define H3_ATTENTION_H
#include <stddef.h>
#ifdef __cplusplus
extern "C" {
#endif
enum { H3_ATTENTION_DEFAULT=0, H3_ATTENTION_SAGE2=1, H3_ATTENTION_SAGE3=2, H3_ATTENTION_SOL=3, H3_ATTENTION_SUBBLOCK=4 };
#define H3_ATTENTION_VERSION 1
#define H3_SUBBLOCK_QUANT_VERSION 2
/* Shape-only, fixed-budget plan. No free-memory-dependent arithmetic choices. */
#define H3_ATTENTION_PLAN_VERSION 1
#define H3_ATTENTION_WORKSPACE_BYTES ((size_t)512*1024*1024)
const char *h3_attention_name(int mode);
unsigned h3_attention_recipe(int mode);
/* SubBlock execution recipe 2 requires the quantization policy section. */
unsigned h3_attention_execution_recipe(int mode,int quant);
unsigned h3_attention_plan(int mode);
int h3_attention_parse(const char *text,int *mode);
int h3_attention_options(int mode,char *error,size_t size);
int h3_attention_preflight(int mode,char *error,size_t size);
int h3_attention_exchange(int mode);
int h3_attention_current(void);
#ifdef __cplusplus
}
#endif
#endif
