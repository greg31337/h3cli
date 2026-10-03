#include "src/denoise/adaptive_cache.h"
#include "src/denoise/subblock.h"
#include "src/denoise/attention.h"
#include "src/denoise/approximate.h"
#include "src/weights/quant.h"
#include "src/internal.h"
#include "src/sampling/av_state.h"
#include "src/sampling/bridge.h"
#include "src/denoise/sol.h"
#include <stdlib.h>
#include <math.h>
#include <stdio.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);return 1;}}while(0)
static int continuation_regions(void) {
    char error[512];
    for(int context=39;context<=90;context+=51)for(int mode=0;mode<=3;mode++)for(int strength=0;strength<=2;strength++) {
        h3_av_state_info shape;CHECK(h3_av_state_shape(256,256,124,&shape));
        h3_layout_ref refs[]={{H3_LAYOUT_REF_IMAGE,1,32,32,0},{H3_LAYOUT_REF_AUDIO,0,0,0,80}};
        h3_layout_spec spec={65,shape.video_t,shape.latent_h,shape.latent_w,shape.audio_t,124,NULL,0,refs,2};
        h3_layout l={0};CHECK(h3_layout_build(&spec,&l,error,sizeof(error)));
        CHECK(h3_continuation_context(context,&l.prefix));
        h3_params p=H3_PARAMS_DEFAULT;p.adaptive_cache=1;p.continuation_context_frames=context;
        p.continuation_mode=mode?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;
        p.bridge_max_strength=(float)strength*.5f;p.bridge_profile=(h3_bridge_profile_type)(mode?mode-1:0);
        h3_bridge_profile bridge;
        if(mode){CHECK(h3_bridge_profile_build(context,8,p.bridge_max_strength,p.bridge_profile,&bridge,NULL,0,error,sizeof(error)));l.bridge=&bridge;}
        if(!h3_approximate_layout_valid(&p,1,context,&l,error,sizeof(error))){fprintf(stderr,"context=%d mode=%d strength=%d: %s\n",context,mode,strength,error);return 1;}
        for(int cache=0;cache<=2;cache++)for(int sparse=0;sparse<=1;sparse++)if(cache||sparse) {
            h3_params variant=p;variant.adaptive_cache=cache;variant.cuda_attention=sparse?H3_ATTENTION_SUBBLOCK:0;
            variant.adaptive_cache_threshold=cache?.12f:0;variant.adaptive_cache_max_hits=cache?2:0;
            variant.adaptive_cache_warmup=cache?2:0;variant.subblock_warmup=sparse?3:0;variant.steps=6;
            CHECK(h3_approximate_layout_valid(&variant,1,context,&l,error,sizeof(error)));
#ifdef __APPLE__
            CHECK(!h3_approximate_params_valid(&variant,0,error,sizeof(error))&&strstr(error,"Metal"));
#else
            CHECK(h3_approximate_params_valid(&variant,0,error,sizeof(error)));
#endif
            variant.geometry_profile=1;CHECK(!h3_approximate_layout_valid(&variant,1,context,&l,error,sizeof(error)));
            variant.geometry_profile=0;variant.save_upscale_state="excluded";
            CHECK(!h3_approximate_layout_valid(&variant,1,context,&l,error,sizeof(error)));
        }
        h3_adaptive_regions r;CHECK(h3_adaptive_regions_build(&l,&r,error,sizeof(error)));
        size_t spatial=(size_t)shape.latent_h*shape.latent_w/4;
        CHECK(r.counts[0]==l.seq_len&&r.counts[1]==(shape.video_t-l.prefix.video_prefix_t)*spatial);
        CHECK(r.counts[2]==2u*(unsigned)(shape.audio_t-l.prefix.audio_prefix_t));
        for(unsigned t=0;t<r.video_t;t++)CHECK(r.video[t]==(t>=(unsigned)l.prefix.video_prefix_t?1:
            (!mode||!strength||bridge.class_mask[bridge.video_classes[t]]==0)?0:
            bridge.video_classes[t]==H3_ROW_GENERATED_VIDEO?12:3+bridge.video_classes[t]-H3_ROW_BRIDGE_VIDEO_FIRST));
        for(unsigned t=0;t<r.audio_t;t++)CHECK(r.audio[t]==(t>=(unsigned)l.prefix.audio_prefix_t?2:
            (!mode||!strength||bridge.class_mask[bridge.audio_classes[t]]==0)?0:
            bridge.audio_classes[t]==H3_ROW_GENERATED_AUDIO?22:13+bridge.audio_classes[t]-H3_ROW_BRIDGE_AUDIO_FIRST));
        float scores[H3_ADAPTIVE_REGIONS]={.001f,.2f,.3f},selected;
        CHECK(h3_adaptive_regions_score(&r,scores,&selected)&&selected==.3f);
        for(unsigned c=3;c<H3_ADAPTIVE_REGIONS;c++)if(r.counts[c]) {
            scores[c]=.75f;CHECK(h3_adaptive_regions_score(&r,scores,&selected)&&selected==.75f);scores[c]=0;
        }
        scores[0]=NAN;CHECK(!h3_adaptive_regions_score(&r,scores,&selected));
        h3_adaptive_plan mem;CHECK(h3_adaptive_plan_recipe(1,4,l.seq_len,5376,0,&mem,error,sizeof(error)));
        CHECK(mem.bytes==l.seq_len*5376*6+47288&&mem.scratch_bytes==47288);
        CHECK(!h3_adaptive_plan_recipe(1,4,l.seq_len,5376,mem.bytes-1,&mem,error,sizeof(error)));
        h3_sol_layout sol={0};CHECK(h3_sol_layout_build(&l,64,64,&sol,error,sizeof(error)));
        for(unsigned row=0;row<r.rows;row++) {
            int protected=row<r.video_start||row>=r.video_start+r.video_rows||
                (row-r.video_start)/r.spatial<(unsigned)l.prefix.video_prefix_t;
            if(protected)CHECK(sol.query[row/64].protect&&sol.key[row/64].protect);
        }
        h3_sol_layout_free(&sol);
        p.cuda_denoise_quant=1;CHECK(!h3_approximate_layout_valid(&p,1,context,&l,error,sizeof(error)));p.cuda_denoise_quant=0;
        CHECK(!h3_approximate_layout_valid(&p,0,0,&l,error,sizeof(error)));
        l.prefix.audio_prefix_t++;CHECK(!h3_approximate_layout_valid(&p,1,context,&l,error,sizeof(error)));l.prefix.audio_prefix_t--;
        l.frozen_audio=1;CHECK(!h3_adaptive_regions_build(&l,&r,error,sizeof(error)));l.frozen_audio=0;
        h3_layout_free(&l);
    }
    CHECK(h3_adaptive_execution_recipe(1,0,0,1)==4&&h3_adaptive_execution_recipe(2,0,1,1)==4);
    CHECK(h3_adaptive_execution_recipe(0,0,1,1)==0);
    uint8_t signature[32]={0};h3_av_state *source=h3_av_state_new(32,32,124,42,signature);CHECK(source);
    memset(source->video,0,source->info.video_elements*4);memset(source->audio,0,source->info.audio_elements*4);
    h3_params identity=H3_PARAMS_DEFAULT;identity.continuation=source;identity.adaptive_cache=1;identity.cuda_attention=H3_ATTENTION_SUBBLOCK;
    char *first=h3_prepared_key("conditioning",&identity,32,32);CHECK(first);
    identity.bridge_max_strength=.9f;identity.bridge_video_steps=2;identity.bridge_profile=H3_BRIDGE_LINEAR;
    char *next=h3_prepared_key("conditioning",&identity,32,32);CHECK(next&&!strcmp(first,next));free(next);
    identity.continuation_mode=H3_CONTINUE_BRIDGE;
    next=h3_prepared_key("conditioning",&identity,32,32);CHECK(next&&strcmp(first,next));free(next);
    identity.continuation_mode=H3_CONTINUE_HARD;identity.continuation_context_frames=90;
    next=h3_prepared_key("conditioning",&identity,32,32);CHECK(next&&strcmp(first,next));free(next);
    identity.continuation_context_frames=0;source->video[0]=1;
    next=h3_prepared_key("conditioning",&identity,32,32);CHECK(next&&strcmp(first,next));free(next);
    free(first);h3_av_state_free(source);
    return 0;
}
int main(void) {
    CHECK(!continuation_regions());
    char error[512];h3_adaptive_plan plan;h3_subblock_plan sp;const char *why;
    CHECK(h3_adaptive_plan_make(0,SIZE_MAX,SIZE_MAX,&plan,error,sizeof(error))&&plan.bytes==0);
    CHECK(h3_adaptive_plan_make(1,9000,5376,&plan,error,sizeof(error))&&plan.bytes<(UINT64_C(512)*1024*1024));
    CHECK(!h3_adaptive_plan_make(1,SIZE_MAX,2,&plan,error,sizeof(error)));
    CHECK(h3_adaptive_plan_make(1,100000,5376,&plan,error,sizeof(error)));
    CHECK(!h3_adaptive_plan_admit(&plan,(UINT64_C(512)*1024*1024),error,sizeof(error)));
    const size_t rows[]={11572,37768,109120,110086};
    const uint64_t expected[]={373272600,1218250776,3519780888,3550940184};
    for(size_t i=0;i<4;i++) {
        CHECK(h3_adaptive_plan_make(1,rows[i],5376,&plan,error,sizeof(error)));
        CHECK(plan.bytes==expected[i]&&plan.persistent_bytes==rows[i]*5376*4);
        CHECK(h3_adaptive_plan_admit(&plan,plan.bytes,error,sizeof(error)));
        CHECK(!h3_adaptive_plan_admit(&plan,plan.bytes-1,error,sizeof(error)));
        CHECK(strstr(error,"minimum --adaptive-cache-max-mib"));
        CHECK(h3_adaptive_plan_admit(&plan,UINT64_C(8192)*1048576,error,sizeof(error)));
    }
    /* Capacity-only mixed reference topology; no long video is rendered. */
    h3_av_state_info shape;CHECK(h3_av_state_shape(1344,768,362,&shape));
    h3_layout_ref mixed_refs[]={{H3_LAYOUT_REF_IMAGE,1,128,96,0},{H3_LAYOUT_REF_VIDEO,13,48,80,0},
        {H3_LAYOUT_REF_VIDEO,13,48,80,80},{H3_LAYOUT_REF_AUDIO,0,0,0,80}};
    h3_layout_spec mixed_spec={58,shape.video_t,shape.latent_h,shape.latent_w,shape.audio_t,362,NULL,0,mixed_refs,4};
    h3_layout mixed_layout={0};CHECK(h3_layout_build(&mixed_spec,&mixed_layout,error,sizeof(error)));
    size_t target_rows=shape.video_elements/96+shape.audio_elements/32;
    size_t reference_rows=128*96/4+2*13*48*80/4+4*80;
    CHECK(mixed_layout.seq_len==58+target_rows+reference_rows);
    CHECK(h3_adaptive_plan_size(1,mixed_layout.seq_len,5376,&plan,error,sizeof(error)));
    CHECK(plan.bytes==mixed_layout.seq_len*5376*6+6168);
    CHECK(h3_adaptive_plan_admit(&plan,UINT64_C(8192)*1048576,error,sizeof(error)));
    CHECK(!h3_adaptive_plan_admit(&plan,plan.bytes-1,error,sizeof(error)));
    for(int context=39;context<362;context+=51) {
        CHECK(h3_continuation_context(context,&mixed_layout.prefix));
        h3_adaptive_regions regions;h3_bridge_profile bridge;
        for(int mode=0;mode<2;mode++) {
            mixed_layout.bridge=NULL;
            if(mode){CHECK(h3_bridge_profile_build(context,8,.5f,H3_BRIDGE_STEPPED,&bridge,NULL,0,error,sizeof(error)));mixed_layout.bridge=&bridge;}
            CHECK(h3_adaptive_regions_build(&mixed_layout,&regions,error,sizeof(error)));
            CHECK(h3_adaptive_plan_recipe(1,4,mixed_layout.seq_len,5376,UINT64_C(8192)*1048576,&plan,error,sizeof(error)));
            CHECK(plan.bytes==mixed_layout.seq_len*5376*6+47288);
            CHECK(regions.counts[1]>0&&regions.counts[2]>0);
        }
    }
    mixed_layout.bridge=NULL;
    h3_layout_free(&mixed_layout);
    uint64_t budget=0;
    const char *invalid[]={"","0","-1","+1","1M","1.0"," 1","1 ","auto","17592186044416","18446744073709551616"};
    for(size_t i=0;i<sizeof(invalid)/sizeof(*invalid);i++)CHECK(!h3_adaptive_mib_parse(invalid[i],&budget));
    CHECK(h3_adaptive_mib_parse("4096",&budget)&&budget==H3_ADAPTIVE_DEFAULT_BYTES);
    CHECK(h3_adaptive_mib_parse("000512",&budget)&&budget==(UINT64_C(512)*1024*1024));
    CHECK(!h3_adaptive_plan_size(1,SIZE_MAX/6,1,&plan,error,sizeof(error)));
    for(int mode=1;mode<=2;mode++) {
        h3_adaptive_history h={0};
        for(int step=0;step<4;step++) {
            CHECK(h3_adaptive_decide(mode,&h,step,50,0,0,0,mode==1?.04f:.08f,mode==1?1:3,&why)==1);
            h3_adaptive_commit(&h,step,0,1);
        }
        int limit=mode==1?1:3;
        for(int j=0;j<limit;j++){
            CHECK(h3_adaptive_decide(mode,&h,4+j,50,0,0,0,mode==1?.04f:.08f,mode==1?1:3,&why)==0);
            h3_adaptive_commit(&h,4+j,0,0);
        }
        CHECK(h3_adaptive_decide(mode,&h,4+limit,50,0,0,0,mode==1?.04f:.08f,mode==1?1:3,&why)==1&&!strcmp(why,"streak"));
        h3_adaptive_commit(&h,9,0,1);
        CHECK(h3_adaptive_decide(mode,&h,10,50,0,1,0,mode==1?.04f:.08f,mode==1?1:3,&why)==1&&!strcmp(why,"attention-phase"));
        CHECK(h3_adaptive_decide(mode,&h,10,50,0,0,mode==1?.04f:.08f,mode==1?.04f:.08f,mode==1?1:3,&why)==1&&!strcmp(why,"score"));
        CHECK(h3_adaptive_decide(mode,&h,10,50,0,0,NAN,mode==1?.04f:.08f,mode==1?1:3,&why)==-1);
        CHECK(h3_adaptive_decide(mode,&h,10,50,0,0,INFINITY,mode==1?.04f:.08f,mode==1?1:3,&why)==-1);
        CHECK(h3_adaptive_decide(mode,&h,10,50,0,0,-1,mode==1?.04f:.08f,mode==1?1:3,&why)==-1);
        h3_adaptive_commit(&h,48,1,1);
        CHECK(h3_adaptive_decide(mode,&h,49,50,0,1,0,mode==1?.04f:.08f,mode==1?1:3,&why)==1&&!strcmp(why,"final"));
    }
    CHECK(h3_subblock_budget(590,.75f)==152);
    for(int warmup=2;warmup<=16;warmup++)for(int mode=1;mode<=2;mode++) {
        h3_adaptive_history h={0};
        for(int step=0;step<warmup;step++) {
            CHECK(h3_adaptive_decide(mode,&h,step,warmup+2,warmup,0,0,mode==1?.04f:.08f,mode==1?1:3,&why)==1);
            CHECK(!strcmp(why,step?"warmup":"empty"));
            h3_adaptive_commit(&h,step,0,1);
        }
        CHECK(h3_adaptive_decide(mode,&h,warmup,warmup+2,warmup,0,0,mode==1?.04f:.08f,mode==1?1:3,&why)==0);
        h3_adaptive_commit(&h,warmup,0,0);
        CHECK(h3_adaptive_decide(mode,&h,warmup+1,warmup+2,warmup,0,0,mode==1?.04f:.08f,mode==1?1:3,&why)==1&&!strcmp(why,"final"));
        CHECK(h3_adaptive_decide(mode,&h,warmup+1,50,warmup,1,0,mode==1?.04f:.08f,mode==1?1:3,&why)==1&&!strcmp(why,"attention-phase"));
        CHECK(!strcmp(h3_subblock_dense_reason(9000,1,warmup-1,.75f,warmup),"warmup"));
        CHECK(!h3_subblock_dense_reason(9000,1,warmup,.75f,warmup));
        CHECK(!strcmp(h3_subblock_dense_reason(9000,0,warmup,.75f,warmup),"probe"));
    }
    CHECK(h3_subblock_budget(590,.8f)==120);
    CHECK(h3_subblock_budget(7,.99f)==7);
    CHECK(h3_subblock_budget(590,0)==590);
    CHECK(h3_subblock_budget(590,NAN)==0);
    CHECK(h3_subblock_plan_make(9000,56,.75f,H3_ATTENTION_WORKSPACE_BYTES,&sp,error,sizeof(error)));
    CHECK(sp.bytes<=H3_ATTENTION_WORKSPACE_BYTES&&sp.keep==40);
    CHECK(!h3_subblock_plan_make(9000,56,.75f,1024,&sp,error,sizeof(error)));
    CHECK(!h3_subblock_plan_make(UINT32_MAX,56,.75f,H3_ATTENTION_WORKSPACE_BYTES,&sp,error,sizeof(error)));
    CHECK(!strcmp(h3_subblock_dense_reason(9000,1,9,.75f,0),"warmup"));
    CHECK(!strcmp(h3_subblock_dense_reason(9000,0,10,.75f,0),"probe"));
    CHECK(!strcmp(h3_subblock_dense_reason(4095,1,10,.75f,0),"short"));
    CHECK(!strcmp(h3_subblock_dense_reason(9000,1,10,0,0),"full-budget"));
    CHECK(!h3_subblock_dense_reason(9000,1,10,.75f,0));
    h3_params p=H3_PARAMS_DEFAULT;
    h3_params saved=H3_PARAMS_DEFAULT;saved.adaptive_cache=1;saved.cuda_attention=H3_ATTENTION_SUBBLOCK;
    for(int n=2;n<=16;n++) {
        p=saved;p.adaptive_cache_warmup=p.subblock_warmup=n;p.steps=n+2;
        CHECK(h3_warmups_valid(&p,error,sizeof(error)));
        p.steps=n+1;CHECK(!h3_warmups_valid(&p,error,sizeof(error)));
        p.resume_sampler_state="deferred";CHECK(h3_warmups_valid(&p,error,sizeof(error)));
        saved.adaptive_cache_warmup=saved.subblock_warmup=n;
        CHECK(h3_warmups_match(&p,&saved));saved.subblock_warmup=n==2?3:2;
        CHECK(!h3_warmups_match(&p,&saved));
    }
    for(int n=-1;n<=17;n++)if(n<2||n>16) {
        p=saved;p.adaptive_cache_warmup=n;p.adaptive_cache_warmup_set=1;
        CHECK(!h3_warmups_valid(&p,error,sizeof(error)));
        p=saved;p.subblock_warmup=n;p.subblock_warmup_set=1;
        CHECK(!h3_warmups_valid(&p,error,sizeof(error)));
    }
    p=(h3_params)H3_PARAMS_DEFAULT;p.adaptive_cache_warmup=2;CHECK(!h3_warmups_valid(&p,error,sizeof(error)));
    p.adaptive_cache_warmup=0;p.subblock_warmup=2;CHECK(!h3_warmups_valid(&p,error,sizeof(error)));
    saved=(h3_params)H3_PARAMS_DEFAULT;saved.adaptive_cache=1;saved.cuda_attention=H3_ATTENTION_SUBBLOCK;saved.steps=2;
    CHECK(h3_warmups_valid(&saved,error,sizeof(error))); /* unchanged short defaults */
    p=saved;p.resume_sampler_state="deferred";p.adaptive_cache_warmup=4;CHECK(!h3_warmups_match(&p,&saved));
    saved.steps=50;p.subblock_warmup=10;CHECK(h3_warmups_match(&p,&saved));
    p=(h3_params)H3_PARAMS_DEFAULT;
    CHECK(h3_approximate_params_valid(&p,0,error,sizeof(error)));
    p.subblock_sparsity_set=1;CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
    p.subblock_sparsity_set=0;p.adaptive_cache=1;p.core_reuse=4;
    CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
    p.core_reuse=1;p.denoise_reuse=2;CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
    p.denoise_reuse=1;CHECK(!h3_approximate_params_valid(&p,1,error,sizeof(error)));
    p.reference_count=1;CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error))&&strstr(error,"valid kinds"));
    p.reference_count=0;p.first_frame="unqualified";CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error))&&strstr(error,"not qualified"));
    p.first_frame=NULL;p.last_frame="unqualified";CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error))&&strstr(error,"not qualified"));
    p.last_frame=NULL;p.continuation=(const h3_av_state *)(uintptr_t)1;
#ifdef __APPLE__
    CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error))&&strstr(error,"Metal"));
#else
    CHECK(h3_approximate_params_valid(&p,0,error,sizeof(error)));
#endif
    p.cuda_denoise_quant=1;CHECK(!h3_approximate_references_valid(&p,error,sizeof(error))&&strstr(error,"BF16"));
    const char *families[]={"attn.qkv_proj.weight","attn.out_proj.weight","mlp.fc1.weight","mlp.fc2.weight"};
    for(int mode=1;mode<=2;mode++) {
        int ordinary=0,adaptive=0;
        for(int block=0;block<50;block++)for(int family=0;family<4;family++) {
            char name[128];snprintf(name,sizeof(name),"blocks.%d.%s",block,families[family]);
            CHECK(h3_quant_projection_policy(name,mode,0)==mode);ordinary++;
            int selected=h3_quant_projection_policy(name,mode,1);
            CHECK(selected==(block?mode:0));adaptive+=selected!=0;
        }
        CHECK(ordinary==200&&adaptive==196);
        CHECK(h3_quant_recipe(mode,0)==H3_QUANT_VERSION);
        CHECK(h3_quant_recipe(mode,1)==H3_QUANT_ADAPTIVE_VERSION);
        CHECK(!h3_quant_projection_policy("video_head.proj.weight",mode,1));
        p=(h3_params)H3_PARAMS_DEFAULT;p.adaptive_cache=1;p.cuda_denoise_quant=mode;
#ifdef __APPLE__
        CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error))&&strstr(error,"Metal"));
#else
        CHECK(h3_approximate_params_valid(&p,0,error,sizeof(error)));
#endif
        p.adaptive_cache=2;CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
        p.adaptive_cache=1;p.cuda_attention=H3_ATTENTION_SUBBLOCK;
        CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
        p.adaptive_cache=0;
#ifdef __APPLE__
        CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error))&&strstr(error,"Metal"));
#else
        CHECK(h3_approximate_params_valid(&p,0,error,sizeof(error)));
#endif
        CHECK(h3_quant_execution_recipe(mode,0,H3_ATTENTION_SUBBLOCK)==H3_QUANT_SUBBLOCK_VERSION);
        CHECK(h3_attention_execution_recipe(H3_ATTENTION_SUBBLOCK,mode)==H3_SUBBLOCK_QUANT_VERSION);
        CHECK(h3_quant_execution_recipe(mode,0,0)==H3_QUANT_VERSION);
        CHECK(h3_quant_execution_recipe(mode,1,0)==H3_QUANT_ADAPTIVE_VERSION);
        CHECK(h3_attention_execution_recipe(H3_ATTENTION_SUBBLOCK,0)==H3_ATTENTION_VERSION);
        p.first_frame="unqualified";CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
    }
    h3_reference refs[13]={0};
    for(int i=0;i<13;i++)refs[i].kind=H3_REFERENCE_IMAGE;
    p=(h3_params)H3_PARAMS_DEFAULT;p.cuda_attention=H3_ATTENTION_SUBBLOCK;p.references=refs;
    for(int mode=0;mode<3;mode++)for(size_t count=1;count<=9;count++) {
        p.reference_image_size=mode;p.reference_count=count;
        CHECK(h3_approximate_references_valid(&p,error,sizeof(error)));
        for(size_t j=0;j<count;j++) {
            refs[j].kind=H3_REFERENCE_AUDIO;CHECK(h3_approximate_references_valid(&p,error,sizeof(error))==(count>1));
            refs[j].kind=H3_REFERENCE_VIDEO;CHECK(h3_approximate_references_valid(&p,error,sizeof(error)));
            refs[j].kind=H3_REFERENCE_VIDEO_AUDIO;CHECK(h3_approximate_references_valid(&p,error,sizeof(error)));
            refs[j].kind=(h3_reference_kind)99;CHECK(!h3_approximate_references_valid(&p,error,sizeof(error)));
            refs[j].kind=H3_REFERENCE_IMAGE;
        }
    }
    for(size_t count=10;count<=13;count++){p.reference_count=count;CHECK(!h3_approximate_references_valid(&p,error,sizeof(error)));}
    p.reference_count=12;
    for(int i=9;i<12;i++)refs[i]=(h3_reference){.kind=H3_REFERENCE_VIDEO,.include_embedded_audio=1};
    CHECK(h3_approximate_references_valid(&p,error,sizeof(error)));
    refs[8].kind=H3_REFERENCE_AUDIO;CHECK(!h3_approximate_references_valid(&p,error,sizeof(error)));refs[8].kind=H3_REFERENCE_IMAGE;
    refs[8].kind=H3_REFERENCE_VIDEO;CHECK(!h3_approximate_references_valid(&p,error,sizeof(error)));refs[8].kind=H3_REFERENCE_IMAGE;
    p.reference_count=1;p.cuda_denoise_quant=1;CHECK(!h3_approximate_references_valid(&p,error,sizeof(error)));
    p.cuda_denoise_quant=0;p.adaptive_cache=1;CHECK(h3_approximate_references_valid(&p,error,sizeof(error)));
    p.adaptive_cache=0;p.cuda_attention=0;
    for(int reuse=2;reuse<=3;reuse++){p.denoise_reuse=reuse;CHECK(h3_approximate_params_valid(&p,0,error,sizeof(error)));}
    p=(h3_params)H3_PARAMS_DEFAULT;p.adaptive_cache_max_bytes_set=1;
    CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
    p.adaptive_cache_max_bytes=H3_ADAPTIVE_DEFAULT_BYTES;CHECK(!h3_approximate_params_valid(&p,0,error,sizeof(error)));
    p.resume_sampler_state="deferred";CHECK(h3_approximate_params_valid(&p,0,error,sizeof(error)));
    /* Policy expectations are explicit, independent of the production resolver. */
    const char *bad_threshold[]={""," "," 0.04","0.04 ","-0","+0.1","NaN","inf","0x1p-4",".e1","1e","1e+","1.00000000000000000001","1.01","1e999","1e-999","1e-46","0.1junk"};
    float threshold=0;
    for(size_t i=0;i<sizeof(bad_threshold)/sizeof(*bad_threshold);i++)CHECK(!h3_adaptive_threshold_parse(bad_threshold[i],&threshold));
    const char *good_threshold[]={"0","0.0",".04","4e-2","1","1.0","1e+0","000.080"};
    const float values[]={0,0,.04f,.04f,1,1,1,.08f};
    for(size_t i=0;i<sizeof(values)/sizeof(*values);i++)CHECK(h3_adaptive_threshold_parse(good_threshold[i],&threshold)&&threshold==values[i]);
    const char *bad_hits[]={"","0","17","-1","+1","1.0","1e0","1 "," 1","9999999999999999999999"};int hits=0;
    for(size_t i=0;i<sizeof(bad_hits)/sizeof(*bad_hits);i++)CHECK(!h3_adaptive_max_hits_parse(bad_hits[i],&hits));
    CHECK(h3_adaptive_max_hits_parse("00016",&hits)&&hits==16);
    for(int limit=1;limit<=16;limit++){
        h3_adaptive_history h={.ready=1,.last_step=3,.last_refresh=3};
        for(int j=0;j<limit;j++){
            CHECK(h3_adaptive_decide(1,&h,4+j,50,4,0,.05f,.06f,limit,&why)==0);
            h3_adaptive_commit(&h,4+j,0,0);
        }
        CHECK(h3_adaptive_decide(1,&h,4+limit,50,4,0,0,.06f,limit,&why)==1&&!strcmp(why,"streak"));
    }
    h3_adaptive_history exact={.ready=1,.last_step=3,.last_refresh=3};
    CHECK(h3_adaptive_decide(1,&exact,4,50,4,0,0,0,16,&why)==1&&!strcmp(why,"score"));
    CHECK(h3_adaptive_decide(1,&exact,4,50,4,0,1,1,16,&why)==1);
    CHECK(h3_adaptive_decide(1,&exact,4,50,4,0,.06f,.06f,2,&why)==1);
    CHECK(h3_adaptive_decide(1,&exact,5,50,4,0,0,1,16,&why)==1&&!strcmp(why,"discontinuity"));
    h3_adaptive_history empty={0};
    for(int step=0;step<50;step++)CHECK(h3_adaptive_decide(1,&empty,step,50,4,0,NAN,1,16,&why)==-1);
    float scores[]={.001f,.2f,.7f},score=0;
    CHECK(h3_adaptive_score(1,scores,&score)&&score==.001f);
    CHECK(h3_adaptive_score(2,scores,&score)&&score==.001f);
    CHECK(h3_adaptive_score(3,scores,&score)&&score==.7f);
    scores[1]=NAN;CHECK(!h3_adaptive_score(3,scores,&score));
    for(int mode=1;mode<=2;mode++){
        h3_params configured=H3_PARAMS_DEFAULT;configured.adaptive_cache=mode;
        CHECK(h3_adaptive_threshold(&configured)==(mode==1?.04f:.08f));
        CHECK(h3_adaptive_max_hits(&configured)==(mode==1?1:3));
        configured.adaptive_cache_threshold_set=1;configured.adaptive_cache_threshold=0;
        configured.adaptive_cache_max_hits=16;
        CHECK(h3_adaptive_threshold(&configured)==0&&h3_adaptive_max_hits(&configured)==16);
        CHECK(h3_adaptive_controls_valid(&configured,error,sizeof(error)));
        h3_params request=H3_PARAMS_DEFAULT;request.resume_sampler_state="saved";
        CHECK(h3_adaptive_controls_match(&request,&configured));
        request.adaptive_cache_threshold_set=1;CHECK(h3_adaptive_controls_match(&request,&configured));
        request.adaptive_cache_threshold=.04f;CHECK(!h3_adaptive_controls_match(&request,&configured));
        request.adaptive_cache_threshold=0;request.adaptive_cache_max_hits=15;CHECK(!h3_adaptive_controls_match(&request,&configured));
        request.adaptive_cache_max_hits=16;CHECK(h3_adaptive_controls_match(&request,&configured));
        configured.adaptive_cache=0;CHECK(!h3_adaptive_controls_valid(&configured,error,sizeof(error)));
    }
    p=(h3_params)H3_PARAMS_DEFAULT;p.adaptive_cache=1;p.cuda_attention=4;
    char *omitted=h3_prepared_key("same references",&p,256,256);CHECK(omitted);
    p.adaptive_cache_threshold=.04f;p.adaptive_cache_threshold_set=1;p.adaptive_cache_max_hits=1;
    p.adaptive_cache_warmup=4;p.subblock_warmup=10;
    char *explicit_defaults=h3_prepared_key("same references",&p,256,256);CHECK(explicit_defaults&&!strcmp(omitted,explicit_defaults));free(explicit_defaults);
    p.adaptive_cache_threshold=0;
    char *zero=h3_prepared_key("same references",&p,256,256);CHECK(zero&&strcmp(omitted,zero));
    p.adaptive_cache_threshold=-0.0f;char *negative_zero=h3_prepared_key("same references",&p,256,256);
    CHECK(negative_zero&&!strcmp(zero,negative_zero));free(negative_zero);free(zero);free(omitted);
    CHECK(!h3_quant_recipe(0,1));
    puts("PASS adaptive decisions, strict thresholds, phase/final boundaries, memory plans, mixed precision and policy conflicts");return 0;
}
