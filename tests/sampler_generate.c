/* Real-checkpoint CPU oracle. One model job at a time. Latent callbacks compare
 * all completed boundaries, including those restored in a fresh process. */
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error[512];
#define CHECK(x) do { if(!(x)) { fprintf(stderr,"FAIL %s:%d: %s (%s)\n",__FILE__,__LINE__,#x,error); exit(1); } } while(0)
typedef struct { FILE *trace; int compare, calls, first, last, frames, resuming; } audit;
static int latent(int completed,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque) {
    audit *test=opaque; CHECK(total==20); CHECK(completed>=0 && completed<=20);
    if(!test->calls) test->first=completed; test->last=completed; test->calls++;
    if(test->compare) {
        CHECK(!fseeko(test->trace,(off_t)((nv+na)*4*(size_t)completed),SEEK_SET));
        float *expected=malloc((nv+na)*4); CHECK(expected);
        CHECK(fread(expected,4,nv+na,test->trace)==nv+na);
        if(memcmp(expected,v,nv*4)||memcmp(expected+nv,a,na*4)) {
            fprintf(stderr,"latent mismatch at completed step %d\n",completed); free(expected); return 1;
        }
        free(expected);
    } else { CHECK(fwrite(v,4,nv,test->trace)==nv); CHECK(fwrite(a,4,na,test->trace)==na); }
    fprintf(stderr,"h3cli: oracle %s boundary %d/%d\n",test->compare?"compared":"recorded",completed,total);
    return 0;
}
static int progress(const char *phase,int completed,int total,void *opaque) {
    (void)completed; (void)total; audit *test=opaque;
    if(test->resuming) CHECK(!strstr(phase,"encoder") && !strstr(phase,"tokenizer"));
    return 0;
}
static int frame(const h3_frame *f,void *opaque) {
    audit *test=opaque; CHECK(f->frame_index==test->frames); test->frames++;
    CHECK(f->denoise_step==-1 && f->denoise_steps==0); return 0;
}
static char *path(const char *base,const char *extension) { char *s=NULL; CHECK(asprintf(&s,"%s.%s",base,extension)>=0); return s; }
static void generated(h3_ctx *ctx,const char *prompt,h3_params *p,int paused,const char *av_path) {
    audit *test=p->callback_opaque; test->frames=0;
    h3_result *r=h3_generate(ctx,prompt,p); if(!r) snprintf(error,sizeof(error),"%s",h3_last_error(ctx)); CHECK(r);
    CHECK(r->status==(paused?H3_RESULT_PAUSED:H3_RESULT_COMPLETE));
    CHECK(r->frames==test->frames);
    if(test->resuming) CHECK(r->resume_count>0 && r->resume_format==1);
    CHECK(r->total_steps==20 && r->completed_steps==(paused?p->stop_after_step:20));
    if(paused) {
        CHECK(!r->av_state && r->sampler_state); CHECK(r->audio_samples==0);
        /* Preview output is a copy; the owned mathematical state serializes to
         * exactly the same bytes after VideoVAE decoding. */
        CHECK(h3_sampler_state_save(r->sampler_state,av_path,error,sizeof(error)));
    } else { CHECK(r->av_state && !r->sampler_state); CHECK(h3_av_state_save(r->av_state,av_path,error,sizeof(error))); }
    h3_result_free(r);
}
int main(int argc,char **argv) {
    if(argc<5) { fprintf(stderr,"usage: sampler_generate MODEL OUTBASE MODE SOURCE|- [REUSE]\n       sampler_generate MODEL OUTBASE resume STATE TRACE [STOP]\n"); return 2; }
    h3_ctx *ctx=h3_load_dir(argv[1]); if(!ctx) snprintf(error,sizeof(error),"%s",h3_last_error(NULL)); CHECK(ctx);
    h3_params p=H3_PARAMS_DEFAULT; p.width=p.height=256; p.frames=90; p.steps=20; p.seed=72;
    if(getenv("H3_TEST_FRAMES")) p.frames=atoi(getenv("H3_TEST_FRAMES"));
    if(getenv("H3_TEST_CORE")) p.core_reuse=atoi(getenv("H3_TEST_CORE"));
    if(getenv("H3_TEST_REDUCTION")) p.token_reduction=atoi(getenv("H3_TEST_REDUCTION"));
    char *mp4=path(argv[2],"mp4"), *av=path(argv[2],"h3av"),*trace=path(argv[2],"trace");
    char *sample=path(argv[2],"h3sample"),*full=path(argv[2],"full.h3sample"),*preview=path(argv[2],"preview.mp4"),*after=path(argv[2],"after-preview.h3sample");
    audit test={0}; p.on_latent_step=latent; p.on_frame=frame; p.on_progress=progress; p.callback_opaque=&test;
    if(!strcmp(argv[3],"resume")) {
        CHECK(argc>=6); test.resuming=1; test.trace=fopen(argv[5],"rb"); CHECK(test.trace); test.compare=1;
        p.resume_sampler_state=argv[4]; p.output_path=mp4;
        if(argc>=7) { p.stop_after_step=atoi(argv[6]); p.save_sampler_state=sample; }
        generated(ctx,NULL,&p,p.stop_after_step>=0 && p.stop_after_step<20,p.stop_after_step>=0 && p.stop_after_step<20?after:av);
        CHECK(test.calls==test.last-test.first+1); fclose(test.trace);
    } else {
        h3_cache_set_enabled(ctx,1);
        int reuse=argc>=6?atoi(argv[5]):1; p.denoise_reuse=reuse;
        h3_reference refs[]={{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0},{H3_REFERENCE_IMAGE,"inputs/body1.jpg",NULL,0},
            {H3_REFERENCE_VIDEO,"outputs/continuation-validation/reference.mp4",NULL,0},
            {H3_REFERENCE_AUDIO,"outputs/continuation-validation/reference.wav",NULL,0}};
        if(strcmp(argv[3],"t2va")) { p.references=refs; p.reference_count=!strcmp(argv[3],"mixed")?4:2; }
        h3_av_state *source=NULL;
        if(strcmp(argv[4],"-")) { source=h3_av_state_load(argv[4],error,sizeof(error)); CHECK(source); p.continuation=source; }
        if(!strcmp(argv[3],"bridge")) p.continuation_mode=H3_CONTINUE_BRIDGE;
        const char *prompt="The woman walks slowly through a sunlit garden and waves. Steady camera and quiet outdoor ambience.";
        test.trace=fopen(trace,"w+b"); CHECK(test.trace); p.output_path=mp4; p.save_sampler_state=full;
        generated(ctx,prompt,&p,0,av); CHECK(test.calls==21 && test.first==0 && test.last==20); CHECK(!fflush(test.trace));
        test.compare=1; test.calls=0; p.stop_after_step=4; p.save_sampler_state=sample; p.preview_on_stop=1; p.output_path=preview;
        generated(ctx,prompt,&p,1,after); CHECK(test.calls==5 && test.first==0 && test.last==4);
        if(getenv("H3_TEST_ALT_SOURCE")) {
            h3_av_state *alternative=h3_av_state_load(getenv("H3_TEST_ALT_SOURCE"),error,sizeof(error)); CHECK(alternative);
            char *alternate_mp4=path(argv[2],"alternate.mp4"),*alternate_av=path(argv[2],"alternate.h3av");
            p.continuation=alternative; p.stop_after_step=-1; p.save_sampler_state=NULL; p.preview_on_stop=0;
            p.output_path=alternate_mp4; test.calls=0; generated(ctx,prompt,&p,0,alternate_av);
            CHECK(test.calls==21); p.continuation=source; h3_av_state_free(alternative); free(alternate_mp4); free(alternate_av);
        }
        if(getenv("H3_TEST_BOUNDARIES")) {
            char *list=strdup(getenv("H3_TEST_BOUNDARIES")); CHECK(list);
            /* Produce independent pause boundaries in the same prepared context;
             * each restart will be checked in a separate process by the runner. */
            for(char *value=strtok(list,",");value;value=strtok(NULL,",")) {
                int boundary=atoi(value); if(boundary==4) continue;
                char suffix[64]; snprintf(suffix,sizeof(suffix),"step%d.h3sample",boundary);
                char *checkpoint=path(argv[2],suffix);
                p.stop_after_step=boundary; p.save_sampler_state=checkpoint; p.preview_on_stop=0;
                test.calls=0; generated(ctx,prompt,&p,1,checkpoint);
                CHECK(test.calls==boundary+1); free(checkpoint);
            }
            free(list);
            char *same_mp4=path(argv[2],"same.mp4"),*same_av=path(argv[2],"same.h3av");
            p.resume_sampler_state=sample; p.stop_after_step=-1; p.save_sampler_state=NULL;
            p.preview_on_stop=0; p.output_path=same_mp4; test.calls=0; test.resuming=1;
            generated(ctx,NULL,&p,0,same_av); CHECK(test.first==4 && test.last==20 && test.calls==17);
            free(same_mp4); free(same_av);
        }
        fclose(test.trace); h3_av_state_free(source);
    }
    h3_free(ctx); free(mp4); free(av); free(trace); free(sample); free(full); free(preview); free(after);
    fprintf(stderr,"ok: full-schedule sampler boundaries and result status\n"); return 0;
}
