/* Real sampler-loop regression; tests/denoise_sampler.py replaces predictions. */
#include "src/denoise/dit.h"
#include "src/sampling/sampler_state.h"
#include "src/memory.h"
#include "src/cli/cli_progress.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int preview_test_forward(h3_dit *,int,const float *,const float *,float *,float *,char *,size_t);
static int preview_test_encode(h3_dit *,int,int,int,int,char *,size_t);
#include "preview_test_dit.c"

/* Exercise every preview/reuse/resume case within the suite's evaluation cap. */
enum { VT=17,LH=4,LW=6,AT=96,NV=24*VT*LH*LW,NA=64*AT,STEPS=6 };
static size_t checks;
static char detail[512];
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,detail);exit(1);}}while(0)
static float raw_video[STEPS][NV],raw_audio[STEPS][NA];
static float video_trace[STEPS+1][NV],audio_trace[STEPS+1][NA],effective[STEPS][NV];
static h3_gpu_tensor *pred_video[STEPS],*pred_audio[STEPS];
static int forwards,kind,preview_mode,cancel_step,low_memory,raw_calls,preview_calls,last_raw;

static int preview_test_forward(h3_dit *dit,int step,const float *v,const float *a,
    float *vv,float *av,char *error,size_t size) {
    (void)v;(void)a;(void)error;(void)size;forwards++;
    memcpy(vv,raw_video[step],sizeof(raw_video[step]));memcpy(av,raw_audio[step],sizeof(raw_audio[step]));
    if(kind==1)h3_prefix_mask_velocity(dit->layout.prefix,VT,LH,LW,AT,vv,av);
    return 1;
}
static int preview_test_encode(h3_dit *dit,int step,int begin,int submit,int split,char *error,size_t size) {
    (void)begin;(void)submit;(void)split;(void)error;(void)size;forwards++;
    return h3_gpu_copy_bf16(dit->gpu,dit->video_output_bf16,0,pred_video[step],0,NV) &&
           h3_gpu_copy_bf16(dit->gpu,dit->audio_output_bf16,0,pred_audio[step],0,NA);
}
static int raw_callback(int step,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque) {
    (void)opaque;CHECK(total==STEPS && nv==NV && na==NA);
    CHECK(!memcmp(v,video_trace[step],sizeof(video_trace[step])));
    CHECK(!memcmp(a,audio_trace[step],sizeof(audio_trace[step])));
    raw_calls++;last_raw=step;return 0;
}
static int display_callback(int step,int total,const float *v,size_t nv,void *opaque) {
    (void)opaque;CHECK(total==STEPS && nv==NV && step==last_raw);
    h3_sigma_schedule schedule;CHECK(h3_serving_schedule_build(STEPS,&schedule));
    for(int i=0;i<NV;i++) {
        float want=video_trace[step][i];
        if(preview_mode && schedule.video[step]!=0 && effective[step-1][i]!=0)
            want=fmaf(schedule.video[step],effective[step-1][i],want);
        CHECK(!memcmp(v+i,&want,4));
    }
    preview_calls++;return step==cancel_step;
}
static int progress_callback(const char *phase,int step,int total,void *opaque) {
    (void)phase;(void)total;(void)opaque;
    if(low_memory && step==3)setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES","18446744073709551615",1);
    return 0;
}
static void oracle(h3_dit *dit,int reuse) {
    uint8_t selected[STEPS];CHECK(h3_dit_reuse_schedule(STEPS,reuse,selected,STEPS)>0);
    int last=-1,previous=-1;
    for(int i=0;i<NV;i++)video_trace[0][i]=(float)(i%129-64)/32;
    for(int i=0;i<NA;i++)audio_trace[0][i]=(float)(i%73-36)/16;
    video_trace[0][0]=audio_trace[0][0]=-0.0f;
    for(int step=0;step<STEPS;step++) {
        if(selected[step]){previous=last;last=step;}
        float rv=0,ra=0;
        if(!selected[step] && previous>=0) {
            rv=(dit->sigmas.video[step]-dit->sigmas.video[last])/(dit->sigmas.video[last]-dit->sigmas.video[previous]);
            ra=(dit->sigmas.audio[step]-dit->sigmas.audio[last])/(dit->sigmas.audio[last]-dit->sigmas.audio[previous]);
        }
        for(int stream=0;stream<2;stream++)for(int i=0;i<(stream?NA:NV);i++) {
            float v=stream?raw_audio[last][i]:raw_video[last][i];
            if(!selected[step] && previous>=0)v=fmaf(stream?ra:rv,v-(stream?raw_audio[previous][i]:raw_video[previous][i]),v);
            int t=stream?i%AT:(i/(LH*LW))%VT;
            float mask=1;
            if(kind==1 && t<(stream?65:12))mask=0;
            if(kind==2 && t<(stream?65:12))mask=dit->bridge.class_mask[stream?dit->bridge.audio_classes[t]:dit->bridge.video_classes[t]];
            v=mask==0?0:v*mask;
            float before=stream?audio_trace[step][i]:video_trace[step][i];
            float delta=stream?dit->sigmas.audio[step]-dit->sigmas.audio[step+1]:dit->sigmas.video[step]-dit->sigmas.video[step+1];
            float next=kind && mask==0?before:fmaf(delta,v,before);
            if(stream)audio_trace[step+1][i]=next;
            else {video_trace[step+1][i]=next;effective[step][i]=v;}
        }
    }
}
static void configure(h3_dit *dit,h3_gpu *gpu) {
    memset(dit,0,sizeof(*dit));dit->gpu=gpu;dit->latent_t=VT;dit->latent_h=LH;dit->latent_w=LW;dit->audio_t=AT;
    dit->video_rows=NV/96;dit->audio_rows=NA/32;dit->core_reuse_interval=1;
    dit->video_condition_rows=3;dit->audio_condition_rows=2;
    CHECK(h3_serving_schedule_build(STEPS,&dit->sigmas));
    if(kind)CHECK(h3_continuation_context(39,&dit->layout.prefix));
    if(kind==2){CHECK(h3_bridge_profile_build(39,4,.5f,H3_BRIDGE_LINEAR,&dit->bridge,NULL,0,detail,sizeof(detail)));dit->layout.bridge=&dit->bridge;}
    dit->video_input=h3_gpu_tensor_new_f32(gpu,NV+3*96);dit->audio_input=h3_gpu_tensor_new_f32(gpu,NA+2*32);
    dit->qkv=h3_gpu_tensor_new_bf16(gpu,2*NV);
    dit->video_output_bf16=h3_gpu_tensor_new_bf16(gpu,NV);dit->audio_output_bf16=h3_gpu_tensor_new_bf16(gpu,NA);
    CHECK(dit->video_input&&dit->audio_input&&dit->qkv&&dit->video_output_bf16&&dit->audio_output_bf16);
    h3_dit_set_latent_callback(dit,raw_callback,NULL);
}
static void release(h3_dit *dit) {
    h3_gpu_tensor_free(dit->video_input);h3_gpu_tensor_free(dit->audio_input);h3_gpu_tensor_free(dit->qkv);
    h3_gpu_tensor_free(dit->video_output_bf16);h3_gpu_tensor_free(dit->audio_output_bf16);
    h3_gpu_tensor_free(dit->previous_video_velocity);h3_gpu_tensor_free(dit->previous_audio_velocity);
}
static h3_sampler_state *run(h3_dit *dit,int gpu,int reuse,int mode,int split) {
    preview_mode=mode>=2;forwards=raw_calls=preview_calls=0;last_raw=-1;
    if(mode==3)unsetenv("H3_PREVIEW_MODE");
    else setenv("H3_PREVIEW_MODE",mode==4?"":preview_mode?"denoised":"noisy",1);
    h3_sampler_state *state=h3_sampler_state_create(&dit->sigmas,NV,NA,reuse);CHECK(state);state->sampler_mode=(uint32_t)gpu;
    memcpy(state->video,video_trace[0],sizeof(video_trace[0]));memcpy(state->audio,audio_trace[0],sizeof(audio_trace[0]));
    int ok=h3_dit_denoise_euler_range(dit,state,split?4:STEPS,progress_callback,NULL,mode?display_callback:NULL,NULL,detail,sizeof(detail));
    if(cancel_step || low_memory) {
        CHECK(!ok && state->next_step==3);CHECK(strstr(detail,low_memory?"reclaimable physical memory":"preview stopped"));
        CHECK(!memcmp(state->video,video_trace[3],sizeof(video_trace[3])));
        CHECK(!memcmp(state->audio,audio_trace[3],sizeof(audio_trace[3])));
    } else {
        CHECK(ok);
        if(split) {
            /* Cross-mode resume must preserve the exact numerical history. */
            preview_mode=!preview_mode;setenv("H3_PREVIEW_MODE",preview_mode?"denoised":"noisy",1);
            CHECK(h3_dit_denoise_euler_range(dit,state,STEPS,progress_callback,NULL,mode?display_callback:NULL,NULL,detail,sizeof(detail)));
        }
        CHECK(state->next_step==STEPS);
        CHECK(!memcmp(state->video,video_trace[STEPS],sizeof(video_trace[STEPS])));
        CHECK(!memcmp(state->audio,audio_trace[STEPS],sizeof(audio_trace[STEPS])));
        CHECK(raw_calls==STEPS+1+!!split);CHECK(preview_calls==(mode?STEPS:0));
        int evaluations=0;for(int i=0;i<STEPS;i++)evaluations+=state->selected[i];CHECK(forwards==evaluations);
    }
    unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");return state;
}
typedef struct {
    h3_cli_progress_state cli;
    FILE *stream;
    int completed, completions, previews, cancel;
} progress_trace;
static int traced_progress(const char *phase,int step,int total,void *opaque) {
    progress_trace *t=opaque;
    int completed=!strcmp(phase,"denoise")&&step>t->completed;
    if(completed){t->completed=step;t->completions++;}
    h3_cli_progress_update(&t->cli,t->stream,phase,step,total);
    return completed&&step==t->cancel;
}
static int traced_preview(int step,int total,const float *video,size_t count,void *opaque) {
    progress_trace *t=opaque;(void)video;
    CHECK(total==STEPS&&count==NV);
    CHECK(t->completed==step&&!strcmp(t->cli.phase,"denoise"));
    /* Stand in for expensive display work. The next start must reset this
     * timer, and this step's completion must already have been reported. */
    t->cli.step_started-=3600;
    t->previews++;return 0;
}
static void progress_case(h3_gpu *gpu,int window,int reuse,int preview,int split,int cancel) {
    char value[16];snprintf(value,sizeof(value),"%d",window);setenv("H3_GPU_SAMPLER_WINDOW",value,1);
    setenv("H3_PREVIEW_MODE","noisy",1);
    h3_dit dit;kind=0;configure(&dit,gpu);h3_dit_set_latent_callback(&dit,NULL,NULL);oracle(&dit,reuse);
    h3_sampler_state *state=h3_sampler_state_create(&dit.sigmas,NV,NA,reuse);CHECK(state);
    state->sampler_mode=1;memcpy(state->video,video_trace[0],sizeof(video_trace[0]));
    memcpy(state->audio,audio_trace[0],sizeof(audio_trace[0]));
    char *raw=NULL;size_t size=0;progress_trace t={.cancel=cancel};
    t.stream=open_memstream(&raw,&size);CHECK(t.stream);
    int ok=h3_dit_denoise_euler_range(&dit,state,split?2:6,traced_progress,&t,
        preview?traced_preview:NULL,&t,detail,sizeof(detail));
    if(cancel)CHECK(!ok&&state->next_step==cancel&&t.completed==cancel);
    else {
        CHECK(ok);
        if(split)CHECK(h3_dit_denoise_euler_range(&dit,state,6,traced_progress,&t,
            preview?traced_preview:NULL,&t,detail,sizeof(detail)));
        CHECK(t.completed==6&&state->next_step==6);
        CHECK(!memcmp(state->video,video_trace[6],sizeof(video_trace[6])));
        CHECK(!memcmp(state->audio,audio_trace[6],sizeof(audio_trace[6])));
    }
    CHECK(fclose(t.stream)==0);
    int timed=0;const char *p=raw;
    while((p=strstr(p,"last step: "))) {
        double seconds=-1;CHECK(sscanf(p,"last step: %lf",&seconds)==1&&seconds>=0&&seconds<60);
        timed++;p++;
    }
    if(preview||(window==1&&reuse==1))CHECK(timed==(cancel?cancel:6)&&t.completions==timed);
    else if(window!=1)CHECK(timed==0); /* Enqueue/wait windows are not single steps. */
    else CHECK(timed>0&&timed<6); /* Reuse may queue cheap steps together. */
    CHECK(t.previews==(preview?6:0));
    printf("PASS GPU progress window=%d reuse=%d preview=%d resume=%d cancel=%d: %d timed completions\n",
        window,reuse,preview,split,cancel,timed);fflush(stdout);
    free(raw);h3_sampler_state_free(state);release(&dit);
}
int main(int argc,char **argv) {
    int progress_only=argc==2&&!strcmp(argv[1],"--progress-only");
    CHECK(argc==1||progress_only);
    h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",detail,sizeof(detail));CHECK(gpu);
    uint16_t vb[NV],ab[NA];float packed[NV];
    for(int step=0;step<STEPS;step++) {
        for(int i=0;i<NV;i++){uint32_t b=(uint32_t)(0x3d80+(i*17+step*29)%700+(i%2?0x8000:0))<<16;memcpy(&raw_video[step][i],&b,4);}
        for(int i=0;i<NA;i++){uint32_t b=(uint32_t)(0x3d80+(i*23+step*41)%700+(i%2?0x8000:0))<<16;memcpy(&raw_audio[step][i],&b,4);}
        CHECK(h3_dit_patchify_video(raw_video[step],24,VT,LH,LW,packed,NV));
        for(int i=0;i<NV;i++){uint32_t b;memcpy(&b,packed+i,4);vb[i]=(uint16_t)(b>>16);}
        CHECK(h3_dit_pack_audio(raw_audio[step],32,AT,packed,NA));
        for(int i=0;i<NA;i++){uint32_t b;memcpy(&b,packed+i,4);ab[i]=(uint16_t)(b>>16);}
        pred_video[step]=h3_gpu_tensor_from_bf16(gpu,vb,NV);pred_audio[step]=h3_gpu_tensor_from_bf16(gpu,ab,NA);CHECK(pred_video[step]&&pred_audio[step]);
    }
    if(progress_only) {
        unsetenv("H3_PROFILE");
        progress_case(gpu,1,1,0,0,0);
        progress_case(gpu,1,1,0,1,0);
        progress_case(gpu,1,2,0,0,0);
        progress_case(gpu,2,1,0,0,0);
        progress_case(gpu,0,1,0,0,0);
        progress_case(gpu,0,2,1,1,0);
        progress_case(gpu,1,1,0,0,2);
    } else for(int backend=0;backend<2;backend++)for(kind=0;kind<3;kind++)for(int reuse=1;reuse<=3;reuse++) {
        h3_dit dit;configure(&dit,gpu);oracle(&dit,reuse);
        h3_sampler_state *baseline=run(&dit,backend,reuse,0,0);
        for(int mode=1;mode<=4;mode++)for(int split=0;split<2;split++) {
            h3_sampler_state *s=run(&dit,backend,reuse,mode,split);
            if(reuse>1) {
                const void *a[]={s->last_video_velocity,s->previous_video_velocity,s->last_audio_velocity,s->previous_audio_velocity,s->gpu_last_video,s->gpu_previous_video,s->gpu_last_audio,s->gpu_previous_audio};
                const void *b[]={baseline->last_video_velocity,baseline->previous_video_velocity,baseline->last_audio_velocity,baseline->previous_audio_velocity,baseline->gpu_last_video,baseline->gpu_previous_video,baseline->gpu_last_audio,baseline->gpu_previous_audio};
                for(int j=backend?4:0;j<(backend?8:4);j++)CHECK(!memcmp(a[j],b[j],(size_t)(j%4<2?NV:NA)*(backend?2:4)));
            }
            h3_sampler_state_free(s);
        }
        for(int mode=1;mode<=2;mode++) {
            cancel_step=3;h3_sampler_state *s=run(&dit,backend,reuse,mode,0);h3_sampler_state_free(s);cancel_step=0;
            low_memory=1;s=run(&dit,backend,reuse,mode,0);h3_sampler_state_free(s);low_memory=0;
        }
        h3_sampler_state_free(baseline);release(&dit);
        printf("PASS %s kind=%d reuse=%d: off/noisy/denoised/default/empty trajectories, histories, resume and cancellation\n",backend?"GPU":"CPU",kind,reuse);fflush(stdout);
    }
    for(int step=0;step<STEPS;step++){h3_gpu_tensor_free(pred_video[step]);h3_gpu_tensor_free(pred_audio[step]);}
    h3_gpu_free(gpu);printf("ok: %zu sampler preview checks\n",checks);return 0;
}
