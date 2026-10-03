/* Cached reference requests and protected fast contexts in one process.
 * Small geometries qualify lifetime/isolation, not production content/speed. */
#define main existing_isolation_main
#include "cuda_sglang_isolation.c"
#undef main
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include "src/internal.h"
#include "src/denoise/dit.h"
#include "src/memory.h"

typedef struct { int steps, cancel; } session_trace;
static int progress(const char *phase,int done,int total,void *opaque) {
    session_trace *t=opaque;
    if(!strcmp(phase,"denoise")) {
        CHECK(total==6);t->steps=done;
        if(t->cancel&&done>=1)return 1;
    }
    return 0;
}
static int same(const h3_av_state *a,const h3_av_state *b) {
    return a&&b&&a->info.video_elements==b->info.video_elements&&a->info.audio_elements==b->info.audio_elements&&
        !memcmp(a->video,b->video,a->info.video_elements*4)&&
        !memcmp(a->audio,b->audio,a->info.audio_elements*4);
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    h3_ctx *ctx=h3_load_dir(argv[1]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=64;p.height=64;p.frames=22;p.steps=6;
    p._arithmetic_recipe=H3_SGLANG_VERSION;p.seed=723;p.on_progress=progress;
    const char *prompt="A grand piano in a sunlit room, slow camera movement.";
    h3_av_state *gold[2]={NULL,NULL};uint16_t *fast_gold=NULL;
    uint64_t device[2]={0,0},pinned[2]={0,0};
    for(int i=0;i<5;i++) {
        int shape=i>=2&&i<4;
        p.width=shape?96:64;p.frames=shape?39:22;
        session_trace t={0};p.callback_opaque=&t;
        h3_result *r=h3_generate(ctx,prompt,&p);
        if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));
        CHECK(r&&r->status==H3_RESULT_COMPLETE&&t.steps==6);
        if(!gold[shape])gold[shape]=h3_av_state_clone(r->av_state);
        else CHECK(same(gold[shape],r->av_state));
        CHECK(gold[shape]);h3_result_free(r);
        h3_cache_info cache;h3_cache_get_info(ctx,&cache);
        CHECK(cache.embedding_entries==1&&cache.prepared_dit&&cache.video_decoder);
        h3_gpu_stats stats;CHECK(h3_dit_get_gpu_stats(ctx->dit,&stats));
        if(device[shape])CHECK(stats.live_bytes==device[shape]&&stats.pinned_bytes==pinned[shape]);
        else {device[shape]=stats.live_bytes;pinned[shape]=stats.pinned_bytes;}
        h3_memory_snapshot memory;h3_memory_sample(&memory);
        /* Linux includes process VmSwap in footprint; swap_valid describes
         * the separate macOS system-swap field. */
        CHECK(memory.process_valid&&memory.physical_footprint==memory.resident);
        uint16_t *fast=evaluate(0);
        if(fast_gold){CHECK(!memcmp(fast,fast_gold,E*2));free(fast);}else fast_gold=fast;
        printf("{\"request\":%d,\"frames\":%d,\"width\":%d,\"evaluations\":6,\"cache_present\":true,\"same_state\":true,\"fast_unchanged\":true}\n",i+1,p.frames,p.width);fflush(stdout);
        printf("{\"request\":%d,\"live_dit_bytes\":%llu,\"pinned_dit_bytes\":%llu,\"rss_bytes\":%llu,\"swap_bytes\":%llu}\n",i+1,
            (unsigned long long)stats.live_bytes,(unsigned long long)stats.pinned_bytes,
            (unsigned long long)memory.resident,
            (unsigned long long)(memory.physical_footprint-memory.resident));fflush(stdout);
    }
    session_trace t={.cancel=1};p.callback_opaque=&t;
    h3_result *cancelled=h3_generate(ctx,prompt,&p);CHECK(!cancelled);
    h3_cache_info cache;h3_cache_get_info(ctx,&cache);
    CHECK(!cache.embedding_entries&&!cache.prepared_dit&&!cache.video_decoder);
    t=(session_trace){0};
    h3_result *recovered=h3_generate(ctx,prompt,&p);
    if(!recovered)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));
    CHECK(recovered&&t.steps==6&&same(gold[0],recovered->av_state));h3_result_free(recovered);
    h3_cache_clear(ctx);h3_cache_get_info(ctx,&cache);
    CHECK(!cache.embedding_entries&&!cache.prepared_dit&&!cache.video_decoder);
    h3_free(ctx);h3_av_state_free(gold[0]);h3_av_state_free(gold[1]);free(fast_gold);
    puts("PASS cached reference repeats, shape invalidation, mixed fast contexts, cancellation/clear and exact recovery");
    return 0;
}
