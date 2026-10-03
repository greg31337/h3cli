/* Bounded reference-family coverage through a reusable public-API session.
 * All outputs are two-step functional fixtures, never quality evidence. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>

static char error[1024];
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error); exit(1); } } while (0)
static double now(void) {
    struct timespec t; clock_gettime(CLOCK_MONOTONIC,&t);
    return (double)t.tv_sec+(double)t.tv_nsec/1e9;
}
static int latent(int step,int total,const float *v,size_t nv,
                  const float *a,size_t na,void *opaque) {
    int *next=opaque; CHECK(total==2 && step==(*next)++);
    for (size_t i=0;i<nv;i++) CHECK(isfinite(v[i]));
    for (size_t i=0;i<na;i++) CHECK(isfinite(a[i]));
    return 0;
}
int main(int argc,char **argv) {
    CHECK(argc==5 || argc==6); /* MODEL, OUTPUT_DIR, VIDEO, AUDIO, optional CASE */
    const char *names[]={"t2va","first","last","first-last",
                         "video","silent","video-audio","audio"};
    int selected=-1;
    if (argc==6) {
        for (int which=0;which<8;which++) if (!strcmp(argv[5],names[which])) selected=which;
        CHECK(selected>=0);
    }
    h3_ctx *ctx=h3_load_dir(argv[1]); CHECK(ctx);
    h3_cache_set_enabled(ctx,1);
    const char *prompt="A woman smiles and turns toward the camera in a sunlit forest.";
    for (int which=0;which<8;which++) {
        if (selected>=0 && which!=selected) continue;
        char *movie=NULL,*state=NULL;
        CHECK(asprintf(&movie,"%s/%s.mp4",argv[2],names[which])>0);
        CHECK(asprintf(&state,"%s/%s.h3av",argv[2],names[which])>0);
        h3_params p=H3_PARAMS_DEFAULT;
        p.width=p.height=128; p.frames=which>=4 && which<=6?56:22;
        p.steps=2; p.seed=1001; p.dit_layers=50;
        p.denoise_reuse=p.core_reuse=1;
        p.output_path=movie;
        if (which==1 || which==3) p.first_frame="inputs/face1.jpg";
        if (which==2 || which==3) p.last_frame="inputs/face2.jpg";
        h3_reference refs[2]={{H3_REFERENCE_VIDEO,argv[3],NULL,1},{0}};
        if (which>=4) { p.references=refs; p.reference_count=1; }
        if (which==5) refs[0].include_embedded_audio=0;
        if (which==6) {
            refs[0].kind=H3_REFERENCE_VIDEO_AUDIO;
            refs[0].audio_path=argv[4]; refs[0].include_embedded_audio=0;
        }
        if (which==7) {
            refs[0]=(h3_reference){H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0};
            refs[1]=(h3_reference){H3_REFERENCE_AUDIO,argv[4],NULL,0};
            p.reference_count=2;
        }
        int boundaries=0; p.on_latent_step=latent; p.callback_opaque=&boundaries;
        double start=now();
        h3_result *r=h3_generate(ctx,prompt,&p);
        if (!r) snprintf(error,sizeof(error),"%s",h3_last_error(ctx));
        CHECK(r && r->status==H3_RESULT_COMPLETE && r->completed_steps==2 && boundaries==3);
        CHECK(h3_av_state_save(r->av_state,state,error,sizeof(error)));
        printf("{\"case\":\"%s\",\"frames\":%d,\"wall_seconds\":%.9f,\"valid\":true}\n",
               names[which],p.frames,now()-start); fflush(stdout);
        h3_result_free(r); free(movie); free(state);
    }
    h3_free(ctx);
    return 0;
}
