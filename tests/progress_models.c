/* Opt-in real-model progress lifecycle checks, including cancellation and cache. */
#include "src/h3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); exit(1); } } while (0)
typedef struct {
    int mode, references, reference_done, dit_done, tiles_done, denoise;
} trace;
static int progress(const char *phase, int completed, int total, void *opaque) {
    trace *t = opaque;
    if (!strcmp(phase, "reference vision preparation") && completed == total) {
        CHECK(total == t->references && total > 0);
        t->reference_done++;
        if (t->mode == 1) return 1;
    }
    if (!strcmp(phase, "video VAE encoder") && total > 0 && completed == total)
        t->tiles_done += total;
    if (!strcmp(phase, "DiT initialization") && completed == 1 && total == 1) {
        t->dit_done++;
        if (t->mode == 2) return 1;
    }
    if (!strcmp(phase, "denoise") || !strcmp(phase, "denoise enqueue")) {
        t->denoise++;
        return t->mode != 3;
    }
    return 0;
}
static void run(h3_ctx *ctx, h3_params *params, trace *t) {
    params->on_progress = progress; params->callback_opaque = t;
    h3_result *result = h3_generate(ctx, "The woman smiles and turns toward the camera. Steady camera.", params);
    if (t->mode == 3) {
        CHECK(result && result->status == H3_RESULT_COMPLETE);
        h3_result_free(result);
        return;
    }
    CHECK(!result);
    if (!strstr(h3_last_error(ctx), "cancel")) fprintf(stderr,"unexpected error: %s\n",h3_last_error(ctx));
    CHECK(strstr(h3_last_error(ctx), "cancel"));
}
int main(void) {
    setvbuf(stdout, NULL, _IOLBF, 0);
    h3_ctx *ctx = h3_load_dir("models/MiniMax-H3"); CHECK(ctx);
    h3_params p = H3_PARAMS_DEFAULT;
    p.width = p.height = 128; p.frames = 56; p.steps = 2;
    h3_reference refs[] = {{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0},
                           {H3_REFERENCE_IMAGE,"inputs/body1.jpg",NULL,0}};
    p.references = refs; p.reference_count = 2;
    trace t = {.mode=1, .references=2};
    run(ctx, &p, &t);
    CHECK(t.reference_done == 1 && !t.dit_done && !t.denoise && t.tiles_done == 2);
    puts("PASS two-reference completion and cancellation before text/DiT");

    /* The user's canvas produces a 3x3 encoder grid. Stop before expensive
     * DiT/text work, after all nine tiles and Qwen vision have completed. */
    p.width = 480; p.height = 640; p.reference_count = 1;
    t = (trace){.mode=1, .references=1}; run(ctx, &p, &t);
    CHECK(t.reference_done == 1 && t.tiles_done == 9 && !t.dit_done && !t.denoise);
    puts("PASS 480x640 nine-tile reference preparation");

    p.width = p.height = 128;
    t = (trace){.mode=2, .references=1}; run(ctx, &p, &t);
    CHECK(t.reference_done == 1 && t.dit_done == 1 && !t.denoise);
    puts("PASS fresh DiT completion and cancellation before sampling");

    h3_cache_set_enabled(ctx, 1);
    t = (trace){.mode=3, .references=1}; run(ctx, &p, &t);
    CHECK(t.dit_done == 1 && t.denoise > 0);
    h3_cache_info info; h3_cache_get_info(ctx, &info); CHECK(info.prepared_dit);
    t = (trace){.mode=3, .references=1}; run(ctx, &p, &t);
    CHECK(t.dit_done == 1 && t.denoise > 0 && !t.reference_done);
    h3_cache_get_info(ctx, &info); CHECK(info.prepared_dit);
    t = (trace){.mode=2, .references=1}; run(ctx, &p, &t);
    CHECK(t.dit_done == 1 && !t.denoise);
    h3_cache_get_info(ctx, &info); CHECK(!info.prepared_dit && !info.embedding_entries);
    puts("PASS cached DiT completion, successful reuse and cancellation cleanup");
    h3_free(ctx);
    return 0;
}
