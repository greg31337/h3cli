/* Full model, public API, every completed Euler boundary, fresh-process resume. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include "src/device.h"
#include "src/gpu.h"
#include "src/internal.h"
#include "src/denoise/dit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[1024];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
typedef struct {FILE *trace,*frames;int compare,first,calls,steps,frame_calls;} audit;
static int decoded_frame(const h3_frame *frame,void *opaque) {
    audit *s=opaque;CHECK(frame->denoise_step<0&&frame->frame_index==s->frame_calls++);
    size_t bytes=(size_t)frame->height*(size_t)frame->stride;
    if(s->compare){uint8_t *ref=malloc(bytes);CHECK(ref&&fread(ref,1,bytes,s->frames)==bytes);CHECK(!memcmp(ref,frame->rgb,bytes));free(ref);}
    else CHECK(fwrite(frame->rgb,1,bytes,s->frames)==bytes);
    return 0;
}
static int latent(int step,int total,const float *v,size_t nv,const float *a,size_t na,void *opaque) {
    audit *s=opaque;CHECK(total==s->steps);CHECK(!fseeko(s->trace,(off_t)((nv+na)*4*(size_t)step),SEEK_SET));
    if(s->compare) {
        float *ref=malloc((nv+na)*4);CHECK(ref&&fread(ref,4,nv+na,s->trace)==nv+na);
        CHECK(!memcmp(ref,v,nv*4));CHECK(!memcmp(ref+nv,a,na*4));free(ref);
    } else {CHECK(fwrite(v,4,nv,s->trace)==nv&&fwrite(a,4,na,s->trace)==na);CHECK(!fflush(s->trace));}
    if(!s->calls) s->first=step;
    s->calls++;
    fprintf(stderr,"checked boundary %d/%d\n",step,total);
    return 0;
}
static char *name(const char *base,const char *suffix){char *s=NULL;CHECK(asprintf(&s,"%s.%s",base,suffix)>0);return s;}
static h3_result *run(h3_ctx *ctx,const char *prompt,h3_params *p){h3_result*r=h3_generate(ctx,prompt,p);if(!r)snprintf(error,sizeof(error),"%s",h3_last_error(ctx));CHECK(r);return r;}
int main(int argc,char **argv) {
    CHECK(argc==5 || argc==7);const char *mode=argv[1],*base=argv[3];
    /* Bound the prepared cache before reserving VRAM. On a large card,
     * evicting resident weights can otherwise free tens of GiB, defeating
     * the intended low-memory decoder-eviction scenario. */
    if(getenv("H3_TEST_CACHE_PRESSURE")) {
        CHECK(!setenv("H3_CUDA_WEIGHT_MODE","stream",1));
    }
    int steps=atoi(argv[4]);CHECK(steps>=2);int stop=steps>3?3:1;
    if(getenv("H3_TEST_STOP_AFTER"))stop=atoi(getenv("H3_TEST_STOP_AFTER"));
    CHECK(stop>=0 && stop<steps);
    h3_ctx *ctx=h3_load_dir(argv[2]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    char *trace=name(base,"trace"),*sample=name(base,"h3sample"),*full=name(base,"mp4"),*av=name(base,"h3av");
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=128;p.frames=22;p.steps=steps;p.denoise_reuse=1;p.core_reuse=1;p.seed=72;
    if(getenv("H3_TEST_WIDTH"))p.width=atoi(getenv("H3_TEST_WIDTH"));
    if(getenv("H3_TEST_HEIGHT"))p.height=atoi(getenv("H3_TEST_HEIGHT"));
    if(getenv("H3_TEST_FRAMES"))p.frames=atoi(getenv("H3_TEST_FRAMES"));
    if(getenv("H3_TEST_REUSE"))p.denoise_reuse=atoi(getenv("H3_TEST_REUSE"));
    if(getenv("H3_TEST_CORE"))p.core_reuse=atoi(getenv("H3_TEST_CORE"));
    if(getenv("H3_TEST_REDUCTION"))p.token_reduction=atoi(getenv("H3_TEST_REDUCTION"));
    h3_reference refs[2]={{H3_REFERENCE_IMAGE,"inputs/face1.jpg",NULL,0},
                          {H3_REFERENCE_IMAGE,"inputs/body1.jpg",NULL,0}};
    p.references=refs;p.reference_count=1;
    h3_av_state *continuation=NULL;
    const char *source=getenv("H3_TEST_CONTINUATION");
    if(source && !strcmp(mode,"record")) {
        CHECK(!getenv("H3_TEST_CACHE_PRESSURE"));
        continuation=h3_av_state_load(source,error,sizeof(error));CHECK(continuation);
        p.width=continuation->info.render_width;p.height=continuation->info.render_height;
        p.frames=141;p.continuation_context_frames=39;p.seed=76;p.continuation=continuation;
        p.reference_count=2;
    }
    audit test={0};test.steps=steps;p.on_latent_step=latent;p.callback_opaque=&test;p.output_path=full;
    if(!strcmp(mode,"record")) {
        test.trace=fopen(trace,"w+b");CHECK(test.trace);
        if(getenv("H3_TEST_CACHE_PRESSURE")){char *rgb=name(base,"rgb");test.frames=fopen(rgb,"w+b");free(rgb);CHECK(test.frames);p.on_frame=decoded_frame;}
        h3_result *r=run(ctx,"The woman smiles and turns toward the camera. Steady camera.",&p);
        CHECK(r->status==H3_RESULT_COMPLETE&&test.calls==steps+1);CHECK(h3_av_state_save(r->av_state,av,error,sizeof(error)));h3_result_free(r);


        if(getenv("H3_TEST_CACHE_PRESSURE")) {
            h3_device_info device;CHECK(h3_device_query(&device,error,sizeof(error))&&!strcmp(device.backend,"cuda"));
            h3_cache_info cached;h3_cache_get_info(ctx,&cached);CHECK(cached.prepared_dit&&cached.video_decoder);
            CHECK(test.frame_calls==22&&!fflush(test.frames)&&!fseeko(test.frames,0,SEEK_SET));
            test.compare=1;test.calls=0;test.frame_calls=0;
            r=run(ctx,"The woman smiles and turns toward the camera. Steady camera.",&p);
            CHECK(r->status==H3_RESULT_COMPLETE&&test.calls==steps+1&&test.frame_calls==22&&fgetc(test.frames)==EOF);h3_result_free(r);
            CHECK(!fclose(test.frames));test.frames=NULL;p.on_frame=NULL;
            CHECK(h3_device_query(&device,error,sizeof(error))&&device.free_device_memory>(UINT64_C(4)<<30));
            h3_gpu *pressure=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(pressure);
            h3_gpu_tensor *reservation=h3_gpu_tensor_new_i8(pressure,(size_t)(device.free_device_memory-(UINT64_C(4)<<30)));
            CHECK(reservation);
            char *pressure_output=name(base,"pressure.mp4");p.output_path=pressure_output;p.steps=2;p.on_latent_step=NULL;
            r=run(ctx,"The woman waves toward the camera. Quiet outdoor ambience.",&p);
            CHECK(r->status==H3_RESULT_COMPLETE);h3_result_free(r);
            h3_cache_get_info(ctx,&cached);CHECK(cached.video_decoder&&!cached.prepared_dit);
            h3_gpu_tensor_free(reservation);h3_gpu_free(pressure);free(pressure_output);
            p.output_path=full;p.steps=steps;p.on_latent_step=latent;
            fprintf(stderr,"ok: exact conditioning/prepared/decoder reuse and cache eviction under device-memory pressure\n");
        }
        test.compare=1;test.calls=0;p.save_sampler_state=sample;p.stop_after_step=stop;
        r=run(ctx,"The woman smiles and turns toward the camera. Steady camera.",&p);
        CHECK(r->status==H3_RESULT_PAUSED&&r->completed_steps==stop&&test.calls==stop+1);h3_result_free(r);
    } else {
        CHECK(argc==7&&!strcmp(mode,"resume"));
        test.trace=fopen(argv[6],"rb");CHECK(test.trace);test.compare=1;
        p.resume_sampler_state=argv[5];p.references=NULL;p.reference_count=0;
        h3_result *r=run(ctx,NULL,&p);CHECK(r->status==H3_RESULT_COMPLETE&&test.first==stop&&test.calls==steps-stop+1);
        CHECK(h3_av_state_save(r->av_state,av,error,sizeof(error)));h3_result_free(r);
    }
    CHECK(!fclose(test.trace));h3_free(ctx);h3_av_state_free(continuation);
    free(trace);free(sample);free(full);free(av);
    fprintf(stderr,"ok: completed-boundary resume audit\n");return 0;
}
