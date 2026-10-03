#include "src/denoise/sol.h"
#include "src/backend.h"
#include <stdio.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
static unsigned checks;
static char error[256];
#define CHECK(x) do { checks++;if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,error);exit(1);} } while(0)
int main(void) {
    h3_temporal_shape shape=h3_temporal(243);
    CHECK(shape.video_t==72&&shape.audio_t==405);
    const int contexts[]={0,39,90,141,192,243,294,345};
    const int frames[]={243,362};
    const h3_layout_ref refs[]={
        {H3_LAYOUT_REF_IMAGE,1,8,8,0},
        {H3_LAYOUT_REF_VIDEO,7,8,8,37},
        {H3_LAYOUT_REF_AUDIO,0,0,0,80}};
    for(unsigned geometry=0;geometry<2;geometry++) {
    shape=h3_temporal(frames[geometry]);
    int anchors[]={0,frames[geometry]-1};
    for(unsigned context=0;context<sizeof(contexts)/sizeof(*contexts);context++)for(int reference=0;reference<2;reference++)
    for(unsigned qb=32;qb<=64;qb*=2)for(unsigned kb=32;kb<=128;kb*=2) {
        if(contexts[context]>=frames[geometry])continue;
        h3_layout_spec spec={.text_len=7,.latent_t=shape.video_t,.latent_h=8,.latent_w=8,.audio_t=shape.audio_t,.frame_count=frames[geometry]};
        if(reference){spec.references=refs;spec.reference_count=3;}
        else {spec.keyframes=anchors;spec.keyframe_count=2;}
        h3_layout l={0};CHECK(h3_layout_build(&spec,&l,error,sizeof(error)));
        if(context)CHECK(h3_continuation_context(contexts[context],&l.prefix));
        h3_sol_layout p={0};CHECK(h3_sol_layout_build(&l,qb,kb,&p,error,sizeof(error)));
        CHECK(p.query_blocks==(l.seq_len+qb-1)/qb&&p.key_blocks==(l.seq_len+kb-1)/kb);
        unsigned expected_protected=0;
        for(size_t s=0;s<l.segment_count;s++)for(size_t r=l.segments[s].start;r<l.segments[s].stop;r++) {
            int frame=l.segments[s].kind==H3_SEG_VIDEO?(int)((r-l.segments[s].start)/16):-1;
            int protect=frame<0||frame<l.prefix.video_prefix_t||frame==0||frame==shape.video_t-1;
            if(protect){CHECK(p.query[r/qb].protect&&p.key[r/kb].protect);expected_protected++;}
            else {CHECK(p.query[r/qb].first_frame<=frame&&p.query[r/qb].last_frame>=frame);}
        }
        CHECK(expected_protected==p.protected_rows);
        CHECK(p.key[p.key_blocks-1].rows==(l.seq_len%kb?l.seq_len%kb:kb));
        h3_sol_layout_free(&p);
        size_t saved=l.segments[1].start;l.segments[1].start++;
        CHECK(!h3_sol_layout_build(&l,qb,kb,&p,error,sizeof(error)));CHECK(!p.query&&!p.key);
        l.segments[1].start=saved;h3_layout_free(&l);
    }
    }
    h3_metal_attention_options a=H3_METAL_ATTENTION_DEFAULT;
    CHECK(a.min_exact==0.f&&h3_metal_options_valid(a,error,sizeof(error)));
    h3_backend_scope scope={H3_BACKEND_METAL,H3_ATTN_DENSE,a};
#ifdef __APPLE__
    CHECK(h3_backend_preflight(scope,1,error,sizeof(error)));
    scope.attention=H3_ATTN_SOL;
    CHECK(h3_backend_preflight(scope,3,error,sizeof(error)));
#else
    CHECK(!h3_backend_preflight(scope,1,error,sizeof(error)));
#endif
    scope.backend=H3_BACKEND_MPSGRAPH_REFERENCE;scope.attention=H3_ATTN_DENSE;
    CHECK(h3_backend_preflight(scope,0,error,sizeof(error)));
    CHECK(!h3_backend_preflight(scope,8,error,sizeof(error)));
    h3_metal_attention_options b=a;b.min_exact=1.01f;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    b=a;b.q_block=16;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    b=a;CHECK(h3_metal_options_equal(a,b));b.tau=2;CHECK(!h3_metal_options_equal(a,b));
    a.dense_steps=2;a.dense_sigma=.9f;
    CHECK(!strcmp(h3_metal_sol_dense_reason(a,2,0,1,1),"early-evaluation"));
    CHECK(!strcmp(h3_metal_sol_dense_reason(a,2,1,.5f,.5f),"early-evaluation"));
    CHECK(!strcmp(h3_metal_sol_dense_reason(a,2,2,.5f,.95f),"high-noise"));
    CHECK(!strcmp(h3_metal_sol_dense_reason(a,2,2,.9f,.5f),"high-noise"));
    CHECK(!strcmp(h3_metal_sol_dense_reason(a,0,2,.5f,.5f),"early-layer"));
    CHECK(!h3_metal_sol_dense_reason(a,2,2,.5f,.5f));
    /* A resumed schedule retains its absolute index, even with a low sigma. */
    CHECK(!h3_metal_sol_dense_reason(a,2,37,.1f,.1f));
    CHECK(!strcmp(h3_metal_sol_dense_reason(a,2,37,NAN,.1f),"invalid-noise"));
    a.min_exact=1;CHECK(!strcmp(h3_metal_sol_dense_reason(a,2,37,.1f,.1f),"all-exact"));
    a.min_exact=.1f;a.dense_steps=0;a.dense_sigma=-1;CHECK(!h3_metal_sol_dense_reason(a,2,0,1,1));
    b=a;b.dense_sigma=NAN;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    b=a;b.dense_sigma=-.5f;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    b=a;b.dense_steps=1001;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    b=a;b.layout_fusion=1;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    b.precision=1;b.candidate=1;CHECK(h3_metal_options_valid(b,error,sizeof(error)));
    b.layout_fusion=2;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    b.layout_fusion=-1;CHECK(!h3_metal_options_valid(b,error,sizeof(error)));
    printf("PASS: %u SOL layout/config assertions, all Ref2VA modalities and 39/90/141/192/243/294/345-frame prefixes at 243/362 frames\n",checks);
    return 0;
}
