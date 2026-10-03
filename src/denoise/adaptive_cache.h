#ifndef H3_ADAPTIVE_CACHE_H
#define H3_ADAPTIVE_CACHE_H
#include <stddef.h>
#include <stdint.h>
#include "src/host.h"
#ifdef __cplusplus
extern "C" {
#endif
#define H3_ADAPTIVE_VERSION 1
#define H3_ADAPTIVE_QUANT_VERSION 2
#define H3_ADAPTIVE_REFERENCE_VERSION 3
#define H3_ADAPTIVE_CONTINUATION_VERSION 4
#define H3_ADAPTIVE_REGIONS 23
/* Slots: global, generated video/audio, ten bridge classes per stream.
 * Temporal slot zero contributes only to the global score. */
typedef struct {
    uint32_t rows, video_start, video_rows, audio_start, audio_rows;
    uint32_t spatial, video_t, audio_t;
    uint32_t counts[H3_ADAPTIVE_REGIONS];
    uint8_t video[107], audio[604];
} h3_adaptive_regions;
#define H3_ADAPTIVE_DEFAULT_BYTES (UINT64_C(4096) * 1024 * 1024)
enum { H3_ADAPTIVE_OFF, H3_ADAPTIVE_CONSERVATIVE, H3_ADAPTIVE_AGGRESSIVE };
typedef struct {
    uint32_t ready, streak, phase;
    int last_step, last_refresh;
} h3_adaptive_history;
typedef struct { size_t elements, tensor_bytes, scratch_bytes, bytes, persistent_bytes; } h3_adaptive_plan;
const char *h3_adaptive_name(int mode);
int h3_adaptive_parse(const char *text, int *mode);
/* Recipe 2 binds the unchanged cache arithmetic to quantized suffix recipe 3. */
unsigned h3_adaptive_recipe(int mode,int quant,int references);
unsigned h3_adaptive_execution_recipe(int mode,int quant,int references,int continuation);
int h3_adaptive_regions_build(const h3_layout *layout,h3_adaptive_regions *regions,char *error,size_t size);
int h3_adaptive_regions_score(const h3_adaptive_regions *regions,const float scores[H3_ADAPTIVE_REGIONS],float *score);
int h3_adaptive_threshold_parse(const char *text,float *threshold);
int h3_adaptive_max_hits_parse(const char *text,int *max_hits);
int h3_adaptive_score(unsigned recipe,const float scores[3],float *score);
int h3_adaptive_warmup(int configured);
uint64_t h3_adaptive_budget(uint64_t bytes);
int h3_adaptive_mib_parse(const char *text,uint64_t *bytes);
int h3_adaptive_plan_size(int mode,size_t rows,size_t columns,
    h3_adaptive_plan *plan,char *error,size_t size);
int h3_adaptive_plan_admit(const h3_adaptive_plan *plan,uint64_t budget,char *error,size_t size);
int h3_adaptive_plan_budget(int mode,size_t rows,size_t columns,uint64_t budget,
    h3_adaptive_plan *plan,char *error,size_t size);
int h3_adaptive_plan_recipe(int mode,unsigned recipe,size_t rows,size_t columns,uint64_t budget,
    h3_adaptive_plan *plan,char *error,size_t size);
int h3_adaptive_plan_make(int mode, size_t rows, size_t columns,
    h3_adaptive_plan *plan, char *error, size_t size);
/* Returns 1 for a refresh, 0 for a hit, -1 for invalid input. No mutation. */
int h3_adaptive_decide(int mode, const h3_adaptive_history *history,
    int step, int steps, int warmup, unsigned phase, float score,
    float threshold, int max_hits, const char **reason);
void h3_adaptive_commit(h3_adaptive_history *history, int step,
    unsigned phase, int refresh);
#ifdef __cplusplus
}
#endif
#endif
