/* Released-model preview invariance and visual fixtures through public API. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <sys/resource.h>

static char error[512];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error);exit(1);}}while(0)
typedef struct {
    FILE *trace;const char *base;int compare,steps,calls,previews,frames,cancel,low_memory,inner_calls;
    size_t nv,na;double raw_time,preview_seconds,step_seconds,last_boundary;
} audit;
static double now(void){struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9;}
static char *path(const char *base,const char *suffix){char *p=NULL;CHECK(asprintf(&p,"%s.%s",base,suffix)>=0);return p;}
static int latent(int step,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque){
    audit *s=opaque;CHECK(total==s->steps);s->nv=nv;s->na=na;
    double t=now();if(step && s->last_boundary!=0)s->step_seconds+=t-s->last_boundary;s->raw_time=t;
    CHECK(!fseeko(s->trace,(off_t)((nv+na)*sizeof(float)*(size_t)step),SEEK_SET));
    if(s->compare){
        float *want=malloc((nv+na)*sizeof(float));CHECK(want);CHECK(fread(want,4,nv+na,s->trace)==nv+na);
        if(memcmp(want,v,nv*4)||memcmp(want+nv,a,na*4))fprintf(stderr,"trajectory mismatch at step %d (%s)\n",step,s->base);
        CHECK(!memcmp(want,v,nv*4));CHECK(!memcmp(want+nv,a,na*4));free(want);
    }else{CHECK(fwrite(v,4,nv,s->trace)==nv);CHECK(fwrite(a,4,na,s->trace)==na);CHECK(!fflush(s->trace));}
    s->calls++;s->last_boundary=now();return 0;
}
static int frame(const h3_frame *f,void *opaque){
    audit *s=opaque;
    if(f->denoise_step>=0){
        s->preview_seconds+=now()-s->raw_time;s->previews++;
        char suffix[64];snprintf(suffix,sizeof(suffix),"step-%02d.ppm",f->denoise_step+1);char *name=path(s->base,suffix);
        FILE *out=fopen(name,"wb");CHECK(out);fprintf(out,"P6\n%d %d\n255\n",f->width,f->height);
        for(int y=0;y<f->height;y++)CHECK(fwrite(f->rgb+(size_t)y*(size_t)f->stride,3,(size_t)f->width,out)==(size_t)f->width);
        CHECK(!fclose(out));free(name);s->last_boundary=now();
        return s->cancel && f->denoise_step+1==2;
    }
    CHECK(f->frame_index==s->frames);s->frames++;return 0;
}
static int progress(const char *phase,int completed,int total,void *opaque){
    (void)total;audit *s=opaque;
    /* Inject pressure inside the second preview's real VideoVAE loop. The
     * failed block must propagate through preview delivery and generation. */
    if(s->low_memory && s->calls==3 && completed>=1 && !strcmp(phase,"preview VAE decode")){
        s->inner_calls++;
        setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES","18446744073709551615",1);
    }
    return 0;
}
static void equal_files(const char *a,const char *b){
    FILE *x=fopen(a,"rb"),*y=fopen(b,"rb");CHECK(x&&y);unsigned char u[65536],v[65536];size_t n;
    do{n=fread(u,1,sizeof(u),x);size_t m=fread(v,1,sizeof(v),y);if(n!=m||memcmp(u,v,n))fprintf(stderr,"file mismatch: %s versus %s\n",a,b);CHECK(n==m && !memcmp(u,v,n));}while(n);
    CHECK(!ferror(x)&&!ferror(y));fclose(x);fclose(y);
}
static void equal_suffix(const char *a,const char *b,const char *suffix){char *x=path(a,suffix),*y=path(b,suffix);equal_files(x,y);free(x);free(y);}
static h3_result *generate(h3_ctx *ctx,const char *prompt,h3_params *p,audit *s,const char *base,int paused){
    s->base=base;s->calls=s->previews=s->frames=0;s->preview_seconds=s->step_seconds=s->last_boundary=0;
    double start=now();h3_result *r=h3_generate(ctx,prompt,p);
    if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r);
    CHECK(r->status==(paused?H3_RESULT_PAUSED:H3_RESULT_COMPLETE));
    if(!paused){char *av=path(base,"h3av");CHECK(h3_av_state_save(r->av_state,av,error,sizeof(error)));free(av);}
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    char *stats=path(base,"json");FILE *f=fopen(stats,"w");CHECK(f);
    fprintf(f,"{\"seconds\":%.9g,\"preview_seconds\":%.9g,\"step_seconds\":%.9g,\"preview_calls\":%d,\"raw_calls\":%d,\"max_rss\":%ld,\"video_elements\":%zu,\"audio_elements\":%zu}\n",now()-start,s->preview_seconds,s->step_seconds,s->previews,s->calls,usage.ru_maxrss,s->nv,s->na);
    CHECK(!fclose(f));free(stats);return r;
}
int main(int argc,char **argv){
    if(argc!=10){fprintf(stderr,"usage: denoise_generate MODEL OUT CASE cpu|gpu SIZE FRAMES STEPS REUSE SOURCE|-\n");return 2;}
    int gpu=!strcmp(argv[4],"gpu");setenv(gpu?"H3_GPU_SAMPLER":"H3_CPU_SAMPLER","1",1);unsetenv(gpu?"H3_CPU_SAMPLER":"H3_GPU_SAMPLER");
    h3_ctx *ctx=h3_load_dir(argv[1]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=atoi(argv[5]);p.frames=atoi(argv[6]);p.steps=atoi(argv[7]);p.denoise_reuse=atoi(argv[8]);p.seed=72;
    const char *prompt="The woman looks into the camera and says <d>[English] Where are you going? To the garden.</d> She smiles and nods gently. Steady camera.";
    h3_reference refs[2]={{H3_REFERENCE_IMAGE,"inputs/2.jpg",NULL,0},{H3_REFERENCE_IMAGE,"inputs/body1.jpg",NULL,0}};
    if(!strcmp(argv[3],"fl2va"))p.first_frame="inputs/face1.jpg";
    else{
        p.references=refs;p.reference_count=1;
        prompt="The woman in <Picture 1> stands on the left of a man in a blue shirt. A red book rests on the table between them. She raises her right hand and turns toward him. A third person stands near the window. Steady camera.";
        if(!strcmp(argv[3],"video")){
            refs[0]=(h3_reference){H3_REFERENCE_VIDEO,"outputs/denoise-validation/reference.mp4",NULL,0};
            prompt="The woman in <Video 1> walks toward the camera and turns her head toward a second woman on her right. A man remains behind them near the window. Keep the same main woman and clothes. Steady camera.";
        }
    }
    h3_av_state *source=NULL;
    if(strcmp(argv[9],"-")){
        source=h3_av_state_load(argv[9],error,sizeof(error));CHECK(source);p.continuation=source;p.keep_continuation_prefix=1;
        if(!strcmp(argv[3],"bridge"))p.continuation_mode=H3_CONTINUE_BRIDGE;
        refs[0].path="inputs/face1.jpg";
        prompt="The woman in <Picture 1> continues walking slowly through the garden and turns toward the camera. A second person stands behind her near a window. Steady camera and natural movement.";
    }
    int cancel_only=getenv("H3_TEST_PREVIEW_CANCEL_ONLY")!=NULL;
    char *trace_path=path(argv[2],"trace");audit s={0};s.steps=p.steps;s.trace=fopen(trace_path,cancel_only?"r+b":"w+b");CHECK(s.trace);
    p.on_latent_step=latent;p.on_frame=frame;p.on_progress=progress;p.callback_opaque=&s;
    const char *modes[]={"off","noisy","denoised","default","empty"};const char *only=getenv("H3_TEST_PREVIEW_ONLY");
    int mode_count=getenv("H3_TEST_PREVIEW_DEFAULT")?5:3;
    char *baseline=cancel_only?path(argv[2],"off"):NULL,*pause_baseline=NULL;
    for(int mode=0;mode<mode_count && !cancel_only;mode++){
        if(only && strcmp(only,modes[mode]))continue;
        if(mode==3)unsetenv("H3_PREVIEW_MODE");
        else setenv("H3_PREVIEW_MODE",mode==4?"":mode==2?"denoised":"noisy",1);
        p.preview_denoise=mode!=0;
        char *base=path(argv[2],modes[mode]),*mp4=path(base,"mp4"),*checkpoint=path(base,"h3sample");
        p.output_path=mp4;p.save_sampler_state=checkpoint;p.stop_after_step=-1;s.compare=baseline!=NULL;
        fprintf(stderr,"RUN %s %s %s\n",argv[3],argv[4],modes[mode]);
        h3_result *r=generate(ctx,prompt,&p,&s,base,0);CHECK(s.calls==p.steps+1);CHECK(s.previews==(mode?p.steps:0));h3_result_free(r);
        if(baseline){equal_suffix(baseline,base,"mp4");equal_suffix(baseline,base,"h3av");equal_suffix(baseline,base,"h3sample");}
        else baseline=strdup(base);
        if(mode>=3){
            char *denoised=path(argv[2],"denoised");
            for(int step=1;step<=p.steps;step++){
                char suffix[64];snprintf(suffix,sizeof(suffix),"step-%02d.ppm",step);
                equal_suffix(denoised,base,suffix);
            }
            free(denoised);
        }
        if(getenv("H3_TEST_PREVIEW_PAUSE")){
            char *pause=path(base,"pause"),*file=path(pause,"h3sample");p.stop_after_step=2;p.output_path=NULL;p.save_sampler_state=file;s.compare=1;
            r=generate(ctx,prompt,&p,&s,pause,1);CHECK(s.calls==3);CHECK(s.previews==(mode?2:0));h3_result_free(r);
            if(pause_baseline)equal_files(pause_baseline,file);else pause_baseline=strdup(file);
            free(pause);free(file);
        }
        free(base);free(mp4);free(checkpoint);
    }
    CHECK(baseline);
    if(pause_baseline){
        char *base=path(argv[2],"resumed"),*mp4=path(base,"mp4");
        setenv("H3_PREVIEW_MODE","denoised",1);p.preview_denoise=1;p.resume_sampler_state=pause_baseline;p.save_sampler_state=NULL;p.stop_after_step=-1;p.output_path=mp4;s.compare=1;
        h3_result *r=generate(ctx,NULL,&p,&s,base,0);CHECK(s.calls==p.steps-1);h3_result_free(r);
        equal_suffix(baseline,base,"mp4");equal_suffix(baseline,base,"h3av");free(base);free(mp4);p.resume_sampler_state=NULL;
    }
    if(getenv("H3_TEST_PREVIEW_CANCEL"))for(int mode=1;mode<=2;mode++)for(int memory=0;memory<2;memory++){
        setenv("H3_PREVIEW_MODE",modes[mode],1);p.preview_denoise=1;p.save_sampler_state=NULL;p.output_path=NULL;p.stop_after_step=-1;
        s.base=baseline;s.compare=1;s.cancel=!memory;s.low_memory=memory;s.calls=s.previews=s.frames=s.inner_calls=0;
        h3_result *r=h3_generate(ctx,prompt,&p);CHECK(!r);CHECK(strstr(h3_last_error(ctx),memory?"reclaimable physical memory":"cancelled during denoising preview"));
        CHECK(s.calls==3 && s.previews==(memory?1:2));CHECK(!memory || s.inner_calls>0);
        unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");
    }
    fclose(s.trace);h3_av_state_free(source);h3_free(ctx);free(baseline);free(pause_baseline);free(trace_path);
    fprintf(stderr,"PASS real %s %s preview invariance/fixtures\n",argv[3],argv[4]);return 0;
}
