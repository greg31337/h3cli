/* Retained-context invalidation and failure recovery, compared with fresh
 * contexts. Every request executes at most five of six scheduled transitions. */
#include "src/h3.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include "src/denoise/adaptive_cache.h"
#include "src/internal.h"
#include "src/denoise/dit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do { if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,h3_last_error(ctx));return 1;} }while(0)
static int cancel(const char *stage,int n,int total,void *opaque) {
    (void)total;(void)opaque;return !strcmp(stage,"denoise")&&n==1;
}
static int equal(const h3_sampler_state *a,const h3_sampler_state *b) {
    if(!a||!b||a->next_step!=b->next_step||a->adaptive_version!=b->adaptive_version||
       a->video_elements!=b->video_elements||a->audio_elements!=b->audio_elements||
       a->adaptive_elements!=b->adaptive_elements||
       memcmp(&a->adaptive_history,&b->adaptive_history,sizeof(a->adaptive_history))||
       memcmp(a->video,b->video,a->video_elements*4)||memcmp(a->audio,b->audio,a->audio_elements*4))return 0;
    return !a->adaptive_elements||(!memcmp(a->adaptive_anchor,b->adaptive_anchor,a->adaptive_elements*2)&&
                                 !memcmp(a->adaptive_delta,b->adaptive_delta,a->adaptive_elements*2));
}
int main(int argc,char **argv) {
    if(argc!=5&&(argc!=6||strcmp(argv[5],"live"))){fprintf(stderr,"usage: adaptive_continuation_context MODEL SOURCE.h3av IMAGE1 IMAGE2 [live]\n");return 2;}
    int live=argc==6;
    char error[512];h3_ctx *ctx=NULL;
    h3_av_state *source=h3_av_state_load(argv[2],error,sizeof(error));CHECK(source);
    h3_av_state *original=h3_av_state_clone(source);CHECK(original);
    /* Two independent contexts coexist; cap residency so both fit the device.
     * Keep the retained context alive across every policy and media change. */
    if(live) {
        /* Automatic partial plans deliberately reconsider capacity each time.
         * Explicit streaming allows actual live DiT reuse while two contexts fit. */
        CHECK(!setenv("H3_CUDA_WEIGHT_MODE","stream",1));
        CHECK(!unsetenv("H3_TEST_CUDA_RESIDENT_BLOCKS"));
    } else CHECK(!setenv("H3_TEST_CUDA_RESIDENT_BLOCKS","8",1));
    ctx=h3_load_dir(argv[1]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_reference ref={.kind=H3_REFERENCE_IMAGE,.path=argv[3]};
    const char *prompt="A woman plays piano in warm sunlight with flowing piano music.";
    for(int pass=0;pass<7;pass++) {
        if(live&&pass>1)continue;
        h3_params p=H3_PARAMS_DEFAULT;
        p.width=source->info.render_width;p.height=source->info.render_height;p.frames=124;
        p.steps=6;p.stop_after_step=(live||pass==1)?3:5;p.continuation=source;
        p.adaptive_cache=pass==3?0:pass==2?2:1;p.adaptive_cache_warmup=p.adaptive_cache?2:0;
        p.adaptive_cache_threshold=1;p.adaptive_cache_threshold_set=p.adaptive_cache!=0;
        p.adaptive_cache_max_hits=p.adaptive_cache?2:0;p.adaptive_cache_max_hits_set=p.adaptive_cache!=0;
        if(!p.adaptive_cache)p.adaptive_cache_threshold=0;
        p.cuda_attention=pass?4:0;p.subblock_warmup=pass?3:0;p.subblock_sparsity=pass?.75f:0;
        p.continuation_mode=pass==1||pass==2?H3_CONTINUE_BRIDGE:H3_CONTINUE_HARD;
        p.continuation_context_frames=pass==2?90:39;p.bridge_profile=pass==2?H3_BRIDGE_LINEAR:H3_BRIDGE_STEPPED;
        ref.path=pass==2||pass==3?argv[4]:argv[3];p.references=&ref;p.reference_count=1;
        p.reference_image_size=H3_REFERENCE_IMAGE_MATCH;
        if(pass==5)source->video[source->info.video_elements-1]+=0.25f;
        if(pass==4) {
            h3_params bad=p;bad.on_progress=cancel;
            h3_result *r=h3_generate(ctx,prompt,&bad);CHECK(!r);
            bad=p;bad.adaptive_cache_max_bytes=1;bad.adaptive_cache_max_bytes_set=1;
            r=h3_generate(ctx,prompt,&bad);CHECK(!r&&strstr(h3_last_error(ctx),"minimum --adaptive-cache-max-mib"));
            h3_reference missing={.kind=H3_REFERENCE_IMAGE,.path="/missing-continuation-media"};
            bad=p;bad.references=&missing;r=h3_generate(ctx,prompt,&bad);CHECK(!r);
            bad=p;bad.resume_sampler_state="/missing-continuation-sampler";
            r=h3_generate(ctx,NULL,&bad);CHECK(!r);
            char invalid[]="/tmp/h3-adaptive-invalid-XXXXXX";
            int invalid_fd=mkstemp(invalid);CHECK(invalid_fd>=0);
            CHECK(write(invalid_fd,"malformed sampler",17)==17);CHECK(!close(invalid_fd));
            bad=p;bad.resume_sampler_state=invalid;
            r=h3_generate(ctx,NULL,&bad);CHECK(!r);CHECK(!unlink(invalid));
            CHECK(!setenv("H3_CUDA_TEST_MEMORY_BUDGET","1",1));
            r=h3_generate(ctx,prompt,&p);CHECK(!r);
            CHECK(!unsetenv("H3_CUDA_TEST_MEMORY_BUDGET"));
        }
        h3_result *retained=h3_generate(ctx,prompt,&p);CHECK(retained&&retained->status==H3_RESULT_PAUSED);
        h3_ctx *fresh=h3_load_dir(argv[1]);CHECK(fresh);
        h3_result *reference=h3_generate(fresh,prompt,&p);CHECK(reference&&reference->status==H3_RESULT_PAUSED);
        CHECK(equal(retained->sampler_state,reference->sampler_state));
        h3_result_free(reference);h3_free(fresh);
        if(live||pass==1) {
            char path[]="/tmp/h3-adaptive-context-XXXXXX";int fd=mkstemp(path);CHECK(fd>=0);CHECK(!close(fd));
            h3_prepared_cache_free(&retained->sampler_state->prepared);
            CHECK(h3_sampler_state_save(retained->sampler_state,path,error,sizeof(error)));
            h3_params resume=H3_PARAMS_DEFAULT;resume.resume_sampler_state=path;resume.stop_after_step=5;
            struct h3_dit *prepared=ctx->dit;
            if(live) {
                uint8_t key[32];CHECK(prepared&&ctx->dit_sampler_key_ready);
                CHECK(h3_sampler_prepared_key(retained->sampler_state,key));
                CHECK(!memcmp(key,ctx->dit_sampler_key,32));
                CHECK(h3_dit_placement_compatible(prepared,0));
            }
            h3_result *continued=h3_generate(ctx,NULL,&resume);CHECK(continued&&continued->status==H3_RESULT_PAUSED);
            if(live)CHECK(ctx->dit==prepared);
            p.stop_after_step=5;fresh=h3_load_dir(argv[1]);CHECK(fresh);
            reference=h3_generate(fresh,prompt,&p);CHECK(reference&&reference->status==H3_RESULT_PAUSED);
            CHECK(equal(continued->sampler_state,reference->sampler_state));h3_result_free(continued);
            continued=h3_generate(ctx,prompt,&p);CHECK(continued&&continued->status==H3_RESULT_PAUSED);
            if(live)CHECK(ctx->dit==prepared);
            CHECK(equal(continued->sampler_state,reference->sampler_state));h3_result_free(continued);
            h3_result_free(reference);h3_free(fresh);CHECK(!unlink(path));
            fprintf(stderr,"PASS continuation resume, sparse transition, new-request reset; live=%d\n",live);
        }
        h3_result_free(retained);
        fprintf(stderr,"PASS continuation context pass=%d mode=%d context=%d cache=%d sparse=%d\n",
            pass,p.continuation_mode,p.continuation_context_frames,p.adaptive_cache,p.cuda_attention);
    }
    source->video[source->info.video_elements-1]=original->video[original->info.video_elements-1];
    CHECK(!memcmp(source->video,original->video,source->info.video_elements*4)&&
          !memcmp(source->audio,original->audio,source->info.audio_elements*4));
    h3_free(ctx);h3_av_state_free(source);h3_av_state_free(original);puts("PASS retained continuation context");return 0;
}
