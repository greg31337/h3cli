#include "src/sampling/av_state.h"
#include "src/denoise/attention.h"
#include "src/denoise/dit.h"
#include "src/denoise/dit_schedule.h"
#include "src/internal.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL %s:%d: %s\n",__FILE__,__LINE__,#x); exit(1); } } while (0)
static uint8_t signature[32]={1,2,3};
static char error[512];

static void timing(void) {
    for (int k=0;k<4;k++) {
        int frames=39+51*k;
        h3_denoise_prefix p;
        CHECK(h3_continuation_context(frames,&p));
        CHECK(p.video_prefix_t==12+15*k && p.audio_prefix_t==65+85*k);
        CHECK(frames*40==p.audio_prefix_t*24);
        CHECK(h3_video_latent_t(frames)==p.video_prefix_t);
        CHECK(p.audio_prefix_t*800*24==frames*32000);
    }
    const int bad[]={INT_MIN,-1,0,5,22,38,40,56,73,89,91,INT_MAX};
    h3_denoise_prefix p;
    for (size_t i=0;i<sizeof(bad)/sizeof(*bad);i++) CHECK(!h3_continuation_context(bad[i],&p));
    CHECK(!h3_continuation_context(39,NULL));
    CHECK(h3_align_frame_count(INT_MAX)==INT_MAX);
}

static void state_io(void) {
    h3_av_state_info info;
    CHECK(h3_av_state_shape(256,256,56,&info));
    CHECK(info.video_elements==24*17*16*16 && info.audio_elements==64*93);
    CHECK(!h3_av_state_shape(INT_MAX,INT_MAX,INT_MAX,&info));
    CHECK(!h3_av_state_shape(32,32,INT_MAX,&info));
    CHECK(!h3_av_state_shape(31,32,56,&info));
    CHECK(!h3_av_state_shape(32,32,55,&info));
    CHECK(!h3_av_state_shape(32,32,413,&info));
    h3_av_state *s=h3_av_state_new(32,32,56,UINT64_MAX,signature); CHECK(s);
    const uint32_t bits[]={0,0x80000000,1,0x3f800000,0x7f800000,0xff800000,0x7fc12345};
    for (size_t i=0;i<s->info.video_elements;i++) memcpy(s->video+i,bits+i%7,4);
    for (size_t i=0;i<s->info.audio_elements;i++) memcpy(s->audio+i,bits+(i+3)%7,4);
    h3_av_state *copy=h3_av_state_clone(s); CHECK(copy && copy->video!=s->video && copy->audio!=s->audio);
    CHECK(!memcmp(copy->video,s->video,s->info.video_elements*4));
    char path[4096];
    CHECK(snprintf(path,sizeof(path),"%s/h3av-test-XXXXXX",getenv("TMPDIR")?getenv("TMPDIR"):"/tmp")<(int)sizeof(path));
    int fd=mkstemp(path); CHECK(fd>=0); close(fd);
    CHECK(h3_av_state_save(s,path,error,sizeof(error)));
    h3_av_state *loaded=h3_av_state_load(path,error,sizeof(error));
    CHECK(loaded && loaded->info.seed==UINT64_MAX);
    CHECK(!memcmp(loaded->video,s->video,s->info.video_elements*4));
    CHECK(!memcmp(loaded->audio,s->audio,s->info.audio_elements*4));
    CHECK(!memcmp(loaded->info.compatibility,signature,32));
    h3_av_state_free(loaded);
    const long offsets[]={0,8,12,16,20,24,36,48,52,64,72,80,88,120,128,160,200};
    for (size_t i=0;i<sizeof(offsets)/sizeof(*offsets);i++) {
        CHECK(h3_av_state_save(s,path,error,sizeof(error)));
        FILE *f=fopen(path,"r+b"); CHECK(f);
        CHECK(!fseek(f,offsets[i],SEEK_SET)); int byte=fgetc(f); CHECK(byte!=EOF);
        CHECK(!fseek(f,offsets[i],SEEK_SET)); CHECK(fputc(byte^0x80,f)!=EOF); CHECK(!fclose(f));
        CHECK(!h3_av_state_load(path,error,sizeof(error)) && *error);
    }
    CHECK(h3_av_state_save(s,path,error,sizeof(error)));
    CHECK(!truncate(path,159)); CHECK(!h3_av_state_load(path,error,sizeof(error)));
    CHECK(h3_av_state_save(s,path,error,sizeof(error)));
    FILE *f=fopen(path,"ab"); CHECK(f); fputc(0,f); fclose(f);
    CHECK(!h3_av_state_load(path,error,sizeof(error)));
    CHECK(!h3_av_state_save(s,"/no-such-h3av-directory/state",error,sizeof(error)));
    CHECK(!h3_av_state_load("/no-such-h3av-directory/state",error,sizeof(error)));
    CHECK(!h3_av_state_save(NULL,path,error,sizeof(error)));
    h3_av_state_free(s); h3_av_state_free(copy); h3_av_state_free(NULL); unlink(path);
}

static void check_prefix(const h3_av_state *target,const h3_av_state *expected,h3_denoise_prefix p) {
    size_t hw=(size_t)target->info.latent_h*(size_t)target->info.latent_w;
    for (size_t c=0;c<24;c++) CHECK(!memcmp(target->video+c*(size_t)target->info.video_t*hw,
        expected->video+c*(size_t)expected->info.video_t*hw,(size_t)p.video_prefix_t*hw*4));
    for (size_t c=0;c<64;c++) CHECK(!memcmp(target->audio+c*(size_t)target->info.audio_t,
        expected->audio+c*(size_t)expected->info.audio_t,(size_t)p.audio_prefix_t*4));
}

static void prefixes(void) {
    h3_av_state *source=h3_av_state_new(64,32,192,99,signature);
    h3_av_state *target=h3_av_state_new(64,32,243,100,signature); CHECK(source && target);
    for (size_t i=0;i<source->info.video_elements;i++) source->video[i]=(float)i/8192.0f;
    for (size_t i=0;i<source->info.audio_elements;i++) source->audio[i]=(float)i/16384.0f;
    h3_av_state *pristine=h3_av_state_clone(source); CHECK(pristine);
    for (int k=0;k<4;k++) {
        h3_denoise_prefix p;
        CHECK(h3_av_state_validate_continuation(source,64,32,243,39+51*k,signature,&p,error,sizeof(error)));
        h3_rng rng;
        h3_rng_seed(&rng,100); h3_rng_fill_normal(&rng,target->video,target->info.video_elements);
        h3_rng_seed(&rng,100); h3_rng_fill_normal(&rng,target->audio,target->info.audio_elements);
        h3_av_state *noise=h3_av_state_clone(target); CHECK(noise);
        CHECK(h3_av_state_insert_prefix(source,&target->info,p,target->video,target->audio,0));
        size_t hw=(size_t)target->info.latent_h*(size_t)target->info.latent_w;
        for (size_t c=0;c<24;c++) {
            size_t dst=c*(size_t)target->info.video_t*hw;
            size_t src=(c*(size_t)source->info.video_t+(size_t)(source->info.video_t-p.video_prefix_t))*hw;
            CHECK(!memcmp(target->video+dst,source->video+src,(size_t)p.video_prefix_t*hw*4));
            size_t suffix=dst+(size_t)p.video_prefix_t*hw;
            CHECK(!memcmp(target->video+suffix,noise->video+suffix,(size_t)(target->info.video_t-p.video_prefix_t)*hw*4));
        }
        for (size_t c=0;c<64;c++) {
            size_t dst=c*(size_t)target->info.audio_t;
            size_t src=c*(size_t)source->info.audio_t+(size_t)(source->info.audio_t-p.audio_prefix_t);
            CHECK(!memcmp(target->audio+dst,source->audio+src,(size_t)p.audio_prefix_t*4));
            CHECK(!memcmp(target->audio+dst+(size_t)p.audio_prefix_t,noise->audio+dst+(size_t)p.audio_prefix_t,
                (size_t)(target->info.audio_t-p.audio_prefix_t)*4));
        }
        memcpy(target->video,noise->video,target->info.video_elements*4);
        CHECK(h3_av_state_insert_prefix(source,&target->info,p,target->video,target->audio,1));
        for (size_t c=0;c<24;c++) for (size_t i=0;i<(size_t)p.video_prefix_t*hw;i++) {
            size_t dst=c*(size_t)target->info.video_t*hw+i;
            size_t src=(c*(size_t)source->info.video_t+(size_t)(source->info.video_t-p.video_prefix_t))*hw+i;
            volatile float a=0.999f*source->video[src],b=0.001f*noise->video[dst];
            float expected=a+b; CHECK(!memcmp(target->video+dst,&expected,4));
        }
        h3_av_state *initial=h3_av_state_clone(target); CHECK(initial);
        initial->video[0]=target->video[0]=-0.0f; initial->audio[0]=target->audio[0]=-0.0f;
        float *vv=malloc(target->info.video_elements*4),*av=malloc(target->info.audio_elements*4); CHECK(vv && av);
        for (int step=0;step<20;step++) {
            for (size_t i=0;i<target->info.video_elements;i++) vv[i]=(float)(i%97)+(float)step;
            for (size_t i=0;i<target->info.audio_elements;i++) av[i]=(float)(i%31)-(float)step;
            h3_prefix_mask_velocity(p,target->info.video_t,target->info.latent_h,target->info.latent_w,target->info.audio_t,vv,av);
            CHECK(h3_prefix_euler_step(p,target->info.video_t,target->info.latent_h,target->info.latent_w,
                target->info.audio_t,target->video,target->audio,vv,av,1,0.9f,1,0.8f));
            check_prefix(target,initial,p);
        }
        CHECK(memcmp(target->video,initial->video,target->info.video_elements*4));
        CHECK(memcmp(target->audio,initial->audio,target->info.audio_elements*4));
        free(vv); free(av); h3_av_state_free(initial); h3_av_state_free(noise);
    }
    h3_denoise_prefix p; uint8_t bad[32]={9};
    CHECK(h3_continuation_context(39,&p));
    h3_av_state_info malformed=target->info; malformed.video_t=INT_MAX;
    CHECK(!h3_av_state_insert_prefix(source,&malformed,p,target->video,target->audio,1));
    malformed=target->info; malformed.audio_elements=SIZE_MAX;
    CHECK(!h3_av_state_insert_prefix(source,&malformed,p,target->video,target->audio,1));
    CHECK(!h3_av_state_validate_continuation(source,32,32,243,39,signature,&p,error,sizeof(error)));
    CHECK(!h3_av_state_validate_continuation(source,64,32,243,40,signature,&p,error,sizeof(error)));
    CHECK(!h3_av_state_validate_continuation(source,64,32,243,243,signature,&p,error,sizeof(error)));
    CHECK(!h3_av_state_validate_continuation(source,64,32,39,39,signature,&p,error,sizeof(error)));
    CHECK(!h3_av_state_validate_continuation(source,64,32,243,39,bad,&p,error,sizeof(error)));
    CHECK(!memcmp(source->video,pristine->video,source->info.video_elements*4));
    CHECK(!memcmp(source->audio,pristine->audio,source->info.audio_elements*4));
    h3_av_state_free(pristine); h3_av_state_free(source); h3_av_state_free(target);
}

static void row_maps(int oracle) {
    h3_sigma_schedule sigmas={0}; sigmas.steps=5;
    float sv[]={1,0.7f,0.2f,0.0001f,0},sa[]={1,0.4f,0.1f,0,0};
    memcpy(sigmas.video,sv,sizeof(sv)); memcpy(sigmas.audio,sa,sizeof(sa));
    h3_dit_schedule *schedule=h3_dit_schedule_plan(&sigmas,1,1,error,sizeof(error)); CHECK(schedule);
    h3_layout_ref refs[]={{H3_LAYOUT_REF_IMAGE,1,4,4,0},{H3_LAYOUT_REF_VIDEO,7,4,4,80},{H3_LAYOUT_REF_AUDIO,0,0,0,80}};
    h3_layout_spec spec={3,17,4,4,93,56,NULL,0,refs,3};
    h3_layout layout; CHECK(h3_layout_build(&spec,&layout,error,sizeof(error)));
    CHECK(h3_continuation_context(39,&layout.prefix));
    uint8_t tags[]={1,0,1}; uint32_t *map=malloc(layout.seq_len*sizeof(*map)); CHECK(map);
    float vv[24*17*4*4], av[64*93], vp[24*17*4*4], ap[64*93];
    for (size_t i=0;i<sizeof(vv)/sizeof(*vv);i++) vv[i]=2.0f;
    for (size_t i=0;i<sizeof(av)/sizeof(*av);i++) av[i]=3.0f;
    h3_prefix_mask_velocity(layout.prefix,17,4,4,93,vv,av);
    CHECK(h3_dit_patchify_video(vv,24,17,4,4,vp,sizeof(vp)/sizeof(*vp)));
    CHECK(h3_dit_pack_audio(av,32,93,ap,sizeof(ap)/sizeof(*ap)));
    if (oracle) printf("[");
    for (int step=0;step<5;step++) {
        CHECK(h3_dit_schedule_row_map(schedule,step,&layout,tags,3,map,layout.seq_len));
        if (oracle) printf("%s{\"sigma_v\":%.9g,\"sigma_a\":%.9g,\"rows\":[",step?",":"",(double)sv[step],(double)sa[step]);
        size_t emitted=0;
        for (size_t s=0;s<layout.segment_count;s++) {
            const h3_segment *seg=&layout.segments[s];
            for (size_t row=seg->start;row<seg->stop;row++) {
                float time=h3_dit_schedule_timestep(schedule,map[row]/3);
                if (seg->kind==H3_SEG_VIDEO || seg->kind==H3_SEG_AUDIO) {
                    int audio=seg->kind==H3_SEG_AUDIO;
                    h3_target_row_class kind=h3_prefix_row_class(layout.prefix,audio,row-seg->start,4,4,93);
                    CHECK(time==h3_prefix_timestep(kind,sv[step],sa[step])); CHECK(map[row]%3==(audio?2u:0u));
                    float velocity=audio ? ap[(row-seg->start)*32] : vp[(row-seg->start)*96];
                    CHECK(velocity==(kind>=H3_ROW_PRESERVED_VIDEO ? 0.0f : audio ? 3.0f : 2.0f));
                    size_t columns=audio?32:96;
                    const float *values=audio?ap:vp;
                    for (size_t column=0;column<columns;column++)
                        CHECK(values[(row-seg->start)*columns+column]==velocity);
                    if (oracle) printf("%s[%d,%zu,%d,%.9g,%.9g]",emitted++?",":"",audio,row-seg->start,kind,(double)time,(double)velocity);
                } else if (seg->kind==H3_SEG_TEXT) CHECK(time==1.0f-sv[step]);
                else if (seg->kind==H3_SEG_REF_AUDIO) CHECK(time==1.0f);
                else CHECK(time==fmaxf(1.0f-sv[step],0.999f));
            }
        }
        if (oracle) printf("]}");
    }
    if (oracle) puts("]");
    if (!oracle) {
        uint32_t *masked=malloc(layout.seq_len*sizeof(*masked)); CHECK(masked);
        CHECK(h3_dit_schedule_row_map(schedule,1,&layout,tags,3,masked,layout.seq_len));
        layout.prefix=(h3_denoise_prefix){0};
        CHECK(h3_dit_schedule_row_map(schedule,1,&layout,tags,3,map,layout.seq_len));
        for (size_t s=0;s<layout.segment_count;s++) {
            const h3_segment *seg=&layout.segments[s];
            for (size_t row=seg->start;row<seg->stop;row++) {
                if (seg->kind==H3_SEG_VIDEO)
                    CHECK(map[row]==h3_dit_schedule_video_row(schedule,1)*3);
                else if (seg->kind==H3_SEG_AUDIO)
                    CHECK(map[row]==h3_dit_schedule_audio_row(schedule,1)*3+2);
                else CHECK(map[row]==masked[row]);
            }
        }
        free(masked);
    }
    free(map); h3_layout_free(&layout); h3_dit_schedule_free(schedule);
}

static void validation(void) {
    h3_ctx ctx={0}; h3_params p=H3_PARAMS_DEFAULT;
    p.steps=6;
    h3_av_state *s=h3_av_state_new(256,256,56,42,signature); CHECK(s);
    p.width=p.height=256; p.continuation=s;
    p.first_frame="inputs/face1.jpg";
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"anchors"));
    p.first_frame=NULL; p.last_frame="inputs/body1.jpg";
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"anchors"));
    p.last_frame=NULL; p.core_reuse=4;
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"core-reuse"));
    p.core_reuse=1; p.token_reduction=1;
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"token reduction"));
    p.token_reduction=0; p.dit_layers=45;
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"layers"));
    p.dit_layers=50; p.continuation_context_frames=40;
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"39 + 51"));
    p.continuation_context_frames=39; p.frames=39;
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"suffix"));
    p.frames=56; p.reference_count=13;
    h3_reference refs[13]={{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0}}; p.references=refs;
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"12 references"));
    p.reference_count=2;
    for (int kind=H3_REFERENCE_IMAGE;kind<=H3_REFERENCE_VIDEO_AUDIO;kind++) {
        refs[1]=(h3_reference){(h3_reference_kind)kind,"fixture",kind==H3_REFERENCE_VIDEO_AUDIO?"audio":NULL,1};
        CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"Ref2VA checkpoint"));
    }
    for (int reuse=2;reuse<=3;reuse++) {
        p.denoise_reuse=reuse;
        CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"Ref2VA checkpoint"));
    }
    p.denoise_reuse=1;
    p.reference_count=10;
    for (size_t i=0;i<10;i++) refs[i]=(h3_reference){H3_REFERENCE_IMAGE,"fixture",NULL,0};
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"9 images"));
    p.reference_count=12;
    for (size_t i=9;i<12;i++) refs[i]=(h3_reference){H3_REFERENCE_VIDEO,"fixture",NULL,1};
    CHECK(!h3_generate(&ctx,"prompt",&p) && strstr(ctx.error,"Ref2VA checkpoint"));
    h3_av_state_free(s);
}

static void cache_keys(void) {
    const char *old_scale = getenv("H3_QWEN_GQA_SCALE_MODE");
    char *saved_scale = old_scale ? strdup(old_scale) : NULL;
    h3_ctx invalid_ctx={0}; h3_params invalid_params=H3_PARAMS_DEFAULT;
    invalid_params.steps=6;
    invalid_params.resume_sampler_state="missing-checkpoint";
    const char *old_preview=getenv("H3_PREVIEW_MODE");
    char *saved_preview=old_preview?strdup(old_preview):NULL;
    setenv("H3_PREVIEW_MODE","invalid",1);
    CHECK(!h3_generate(&invalid_ctx,"prompt",&invalid_params));
    CHECK(strstr(invalid_ctx.error,"H3_PREVIEW_MODE"));
    unsetenv("H3_PREVIEW_MODE");
    setenv("H3_QWEN_GQA_SCALE_MODE","invalid",1);
    CHECK(!h3_generate(&invalid_ctx,"prompt",&invalid_params));
    CHECK(strstr(invalid_ctx.error,"H3_QWEN_GQA_SCALE_MODE"));
    setenv("H3_QWEN_GQA_SCALE_MODE", "reference", 1);
    h3_params p=H3_PARAMS_DEFAULT; p.frames=141;
    h3_reference refs[]={{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0},
                         {H3_REFERENCE_IMAGE,"inputs/body1.jpg",NULL,0}};
    p.references=refs; p.reference_count=2;
    h3_av_state *one=h3_av_state_new(32,32,141,42,signature);
    h3_av_state *two=h3_av_state_new(32,32,141,43,signature); CHECK(one && two);
    memset(one->video,0,one->info.video_elements*4); memset(one->audio,0,one->info.audio_elements*4);
    memset(two->video,1,two->info.video_elements*4); memset(two->audio,1,two->info.audio_elements*4);
    p.continuation=one;
    char *condition=h3_conditioning_key("same prompt",&p,32,32,1);
    char *prepared=h3_prepared_key(condition,&p,32,32); CHECK(condition && prepared);
    p.cuda_attention=H3_ATTENTION_SAGE2;
    char *optional_condition=h3_conditioning_key("same prompt",&p,32,32,1);
    CHECK(optional_condition && strcmp(condition,optional_condition));
    char *optional_prepared=h3_prepared_key(optional_condition,&p,32,32);
    CHECK(optional_prepared && strcmp(prepared,optional_prepared));
    free(optional_condition);free(optional_prepared);p.cuda_attention=H3_ATTENTION_DEFAULT;
    for(int mode=0;mode<2;mode++) {
        setenv("H3_PREVIEW_MODE",mode?"noisy":"denoised",1);
        char *same_preview=h3_conditioning_key("same prompt",&p,32,32,1);
        CHECK(same_preview && !strcmp(condition,same_preview));free(same_preview);
    }
    if(saved_preview) {setenv("H3_PREVIEW_MODE",saved_preview,1);free(saved_preview);}
    else unsetenv("H3_PREVIEW_MODE");
    for(int empty=0;empty<2;empty++) {
        if(empty) setenv("H3_QWEN_GQA_SCALE_MODE","",1);
        else unsetenv("H3_QWEN_GQA_SCALE_MODE");
        char *default_key=h3_conditioning_key("same prompt",&p,32,32,1);
        CHECK(default_key && !strcmp(condition,default_key)); free(default_key);
    }
    for (int mode=0;mode<2;mode++) {
        setenv("H3_QWEN_GQA_SCALE_MODE",mode ? "scaled-q" : "legacy",1);
        char *changed=h3_conditioning_key("same prompt",&p,32,32,1);
        CHECK(changed && strcmp(condition,changed)); free(changed);
    }
    setenv("H3_QWEN_GQA_SCALE_MODE","reference",1);
    p.continuation=two; p.seed=43;
    char *same=h3_conditioning_key("same prompt",&p,32,32,1);
    char *same_prepared=h3_prepared_key(same,&p,32,32);
    CHECK(same && same_prepared && !strcmp(condition,same) && !strcmp(prepared,same_prepared));
    free(same); free(same_prepared);
    p.continuation_context_frames=90;
    same=h3_conditioning_key("same prompt",&p,32,32,1);
    same_prepared=h3_prepared_key(same,&p,32,32);
    CHECK(same && same_prepared && !strcmp(condition,same) && strcmp(prepared,same_prepared));
    free(same); free(same_prepared);
    refs[0].path="inputs/face2.jpg";
    same=h3_conditioning_key("same prompt",&p,32,32,1); CHECK(same && strcmp(condition,same)); free(same);
    refs[0].path="inputs/face1.jpg";
    same=h3_conditioning_key("different prompt",&p,32,32,1); CHECK(same && strcmp(condition,same)); free(same);
    free(condition); free(prepared); h3_av_state_free(one); h3_av_state_free(two);
    if (saved_scale) { setenv("H3_QWEN_GQA_SCALE_MODE",saved_scale,1); free(saved_scale); }
    else unsetenv("H3_QWEN_GQA_SCALE_MODE");
}

int main(int argc,char **argv) {
    if (argc==2 && !strcmp(argv[1],"--oracle")) { row_maps(1); return 0; }
    timing(); state_io(); prefixes(); row_maps(0); validation(); cache_keys();
    printf("ok: %d continuation checks\n",checks); return 0;
}
