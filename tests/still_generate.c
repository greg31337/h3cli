#include "src/h3.h"
#include "src/host.h"
#include "src/vae/image_vae.h"
#include "src/digest.h"
#include "src/memory.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int callbacks,transitions;static char prefix[1024];static uint64_t peak;
static double denoise_begin,denoise_end;
static void sample_memory(void) {h3_memory_snapshot m;h3_memory_sample(&m);if(m.physical_footprint>peak)peak=m.physical_footprint;}
static int progress(const char *phase,int a,int b,void *opaque) {
    (void)opaque;sample_memory();
    if(!strcmp(phase,"denoise")) {
        if(!a && !denoise_begin)denoise_begin=h3_av_now();
        if(b>0 && a==b)denoise_end=h3_av_now();
    }
    if(a==0||a==b||!strcmp(phase,"denoise"))fprintf(stderr,"%s %d/%d\n",phase,a,b);
    return 0;
}
static int frame(const h3_frame *f,void *p) {(void)p;callbacks++;return f->frame_index!=0 || f->frame_count!=1 || f->denoise_step!=-1;}
static int latent(int done,int total,const float *v,size_t nv,const float *a,size_t na,void *p) {
    (void)p;sample_memory();if(nv!=24*30*40 || na!=128 || total!=6)return 1;
    for(size_t i=0;i<nv;i++)if(!isfinite(v[i]))return 1;
    for(size_t i=0;i<na;i++)if(!isfinite(a[i]))return 1;
    char path[1200];snprintf(path,sizeof(path),"%s/step-%d.f32",prefix,done);FILE *f=fopen(path,"wb");
    if(!f)return 1;int ok=fwrite(v,4,nv,f)==nv && fwrite(a,4,na,f)==na;ok=!fclose(f)&&ok;
    transitions++;return !ok;
}
int main(int argc,char **argv) {
    if(argc<3)return 2;snprintf(prefix,sizeof(prefix),"%s",argv[1]);
    h3_ctx *ctx=h3_load_dir("models/MiniMax-H3");if(!ctx){fprintf(stderr,"%s\n",h3_last_error(NULL));return 1;}
    h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.still=1;p.frames=1;p.width=640;p.height=480;p.steps=6;
    p.image_vae="models/image-vae/minimax_h3_t1_image_vae_step1597.safetensors";
    char output[1200],zpath[1200];snprintf(output,sizeof(output),"%s/output.png",prefix);snprintf(zpath,sizeof(zpath),"%s/latent.safetensors",prefix);
    p.output_path=output;p.save_still_latent=zpath;p.on_progress=progress;p.on_frame=frame;p.on_latent_step=latent;
    h3_reference ref={.kind=H3_REFERENCE_IMAGE,.path="inputs/2.jpg"};if(argc>3){p.references=&ref;p.reference_count=1;}
    double start=h3_av_now();h3_result *r=h3_generate(ctx,argv[2],&p);
    if(!r){fprintf(stderr,"generation: %s\n",h3_last_error(ctx));h3_free(ctx);return 1;}
    int ok=r->kind==H3_RESULT_STILL && r->still_latent && !r->av_state && !r->sampler_state && r->frames==1 && !r->fps && !r->audio_samples && !r->sample_rate && callbacks==1 && transitions==7;
    printf("{\"pass\":%s,\"wall_seconds\":%.6f,\"denoise_seconds\":%.6f,\"peak_footprint\":%llu,\"callbacks\":%d,\"latent_callbacks\":%d,\"video_T\":1,\"auxiliary_audio_T\":2,\"audio_decode\":false,\"steps\":6}\n",ok?"true":"false",h3_av_now()-start,denoise_end>denoise_begin?denoise_end-denoise_begin:0,(unsigned long long)peak,callbacks,transitions);
    h3_result_free(r);h3_free(ctx);return !ok;
}
