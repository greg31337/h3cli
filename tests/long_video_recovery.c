/* Full delivery failure/cancellation on an inspected 256x256/464-frame state. */
#include "src/h3.h"
#include "src/gpu.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include "src/memory.h"
#include "src/upscale/upscale_network.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[1024];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);return 1;}}while(0)
static int cancel(const char *phase,int done,int total,void *opaque) {
    (void)done;(void)total;(void)opaque;return !strcmp(phase,"video VAE decode");
}
static int original(const char *path) {
    char bytes[9]={0};FILE *f=fopen(path,"rb");if(!f)return 0;
    int ok=fread(bytes,1,8,f)==8&&!memcmp(bytes,"original",8)&&fgetc(f)==EOF;fclose(f);return ok;
}
static int cancel_frame(const h3_frame *frame,void *opaque) {
    int *calls=opaque;(*calls)++;return frame->width>0;
}
int main(int argc,char **argv) {
    CHECK(argc==5||argc==6);
    /* Deterministic allocation failure without exhausting physical VRAM. */
    setenv("H3_CUDA_TEST_MEMORY_BUDGET","65536",1);
    h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(gpu);
    CHECK(!h3_gpu_tensor_new_bf16(gpu,65536));
    h3_gpu_cancel(gpu);
    h3_gpu_tensor *small=h3_gpu_tensor_new_bf16(gpu,32);CHECK(small);
    h3_gpu_tensor_free(small);h3_gpu_free(gpu);unsetenv("H3_CUDA_TEST_MEMORY_BUDGET");
    if(!strcmp(argv[1],"preview")) {
        h3_av_state *av=h3_av_state_load(argv[3],error,sizeof(error));CHECK(av);
        CHECK(av->info.frames==464&&av->info.render_width==256&&av->info.render_height==256);
        h3_av_state_free(av);int calls=0;
        h3_decode_options o={.preview_vae=1,.preview_vae_model=argv[4],.on_frame=cancel_frame,.callback_opaque=&calls};
        h3_result *r=h3_decode_av_state(argv[2],argv[3],&o,error,sizeof(error));
        CHECK(!r&&calls==1&&strstr(error,"cancel"));
        puts("PASS maximum native preview callback cancellation");return 0;
    }
    h3_av_state *state=h3_av_state_load(argv[2],error,sizeof(error));CHECK(state);
    int frames=state->info.frames;CHECK(frames==464&&state->info.render_width==256&&state->info.render_height==256);h3_av_state_free(state);
    FILE *f=fopen(argv[3],"wb");CHECK(f&&fwrite("original",1,8,f)==8&&!fclose(f));
    h3_decode_options o={.output_path=argv[3],.on_progress=cancel};
    h3_result *r=h3_decode_av_state(argv[1],argv[2],&o,error,sizeof(error));
    CHECK(!r&&strstr(error,"cancel")&&original(argv[3]));
    char *ff=getenv("H3_FFMPEG")?strdup(getenv("H3_FFMPEG")):NULL;
    setenv("H3_FFMPEG","/bin/false",1);o.on_progress=NULL;
    r=h3_decode_av_state(argv[1],argv[2],&o,error,sizeof(error));CHECK(!r&&original(argv[3]));
    if(argc==6) {
        h3_sampler_state *completed=h3_sampler_state_load(argv[5],error,sizeof(error));CHECK(completed);
        CHECK(completed->aligned_frames==464&&completed->render_width==256&&completed->render_height==256&&completed->next_step==completed->total_steps);
        h3_sampler_state_free(completed);
        char saved[4096];CHECK(snprintf(saved,sizeof(saved),"%s.h3sample",argv[4])<(int)sizeof(saved));
        h3_ctx *ctx=h3_load_dir(argv[1]);CHECK(ctx);
        h3_params p=H3_PARAMS_DEFAULT;p.resume_sampler_state=argv[5];p.save_sampler_state=saved;p.output_path=argv[3];
        r=h3_generate(ctx,NULL,&p);snprintf(error,sizeof(error),"%s",h3_last_error(ctx));
        CHECK(!r&&original(argv[3]));
        completed=h3_sampler_state_load(saved,error,sizeof(error));CHECK(completed);
        CHECK(completed->next_step==completed->total_steps);
        h3_sampler_state_free(completed);h3_free(ctx);
        puts("PASS completed checkpoint survives failed post-denoise delivery");
    }
    if(ff){setenv("H3_FFMPEG",ff,1);free(ff);}else unsetenv("H3_FFMPEG");
    o.output_path=argv[4];
    r=h3_decode_av_state(argv[1],argv[2],&o,error,sizeof(error));CHECK(r&&r->frames==frames);
    h3_result_free(r);CHECK(original(argv[3]));
    puts("PASS full-VAE cancellation, encoder failure, output preservation and recovery");return 0;
}
