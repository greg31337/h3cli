/* Real-model stock still qualification; records every finite sampler boundary. */
#include "src/h3.h"
#include "src/vae/image_vae.h"
#include "src/digest.h"
#include "src/memory.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
static int callbacks,transitions,steps,width,height;
static uint64_t peak;
static h3_sha256_ctx trajectory;
static int progress(const char *phase,int a,int b,void *opaque) {
    (void)opaque;h3_memory_snapshot m;h3_memory_sample(&m);if(m.physical_footprint>peak)peak=m.physical_footprint;
    if(a==0||a==b)fprintf(stderr,"%s %d/%d\n",phase,a,b);
    return 0;
}
static int frame(const h3_frame *f,void *p) {
    (void)p;callbacks++;return f->frame_index!=0 || f->frame_count!=1 || f->denoise_step!=-1 || f->width!=width || f->height!=height;
}
static int latent(int done,int total,const float *v,size_t nv,const float *a,size_t na,void *p) {
    (void)p;if(nv!=(size_t)24*(height/16)*(width/16) || na!=128 || total!=steps || done!=transitions)return 1;
    for(size_t i=0;i<nv;i++)if(!isfinite(v[i]))return 1;
    for(size_t i=0;i<na;i++)if(!isfinite(a[i]))return 1;
    h3_sha256_update(&trajectory,v,(h3_sha256_size)(nv*4));h3_sha256_update(&trajectory,a,(h3_sha256_size)(na*4));transitions++;return 0;
}
int main(int argc,char **argv) {
    if(argc<8){fprintf(stderr,"usage: still_generate MODEL OUTDIR W H STEPS match|high|max PROMPT [REF...]\n");return 2;}
    width=atoi(argv[3]);height=atoi(argv[4]);steps=atoi(argv[5]);
    h3_ctx *ctx=h3_load_dir(argv[1]);if(!ctx){fprintf(stderr,"%s\n",h3_last_error(NULL));return 1;}
    h3_params p=H3_PARAMS_DEFAULT;p.still=1;p.frames=1;p.width=width;p.height=height;p.steps=steps;p.seed=42;
    p.reference_image_size=!strcmp(argv[6],"max")?H3_REFERENCE_IMAGE_MAX:!strcmp(argv[6],"high")?H3_REFERENCE_IMAGE_HIGH:H3_REFERENCE_IMAGE_MATCH;
    h3_reference *refs=calloc((size_t)argc,sizeof(*refs));if(!refs)return 1;
    for(int i=8;i<argc;i++)refs[i-8]=(h3_reference){.kind=H3_REFERENCE_IMAGE,.path=argv[i]};
    p.references=refs;p.reference_count=(size_t)(argc-8);
    char output[2048],zpath[2048];snprintf(output,sizeof(output),"%s/output.png",argv[2]);snprintf(zpath,sizeof(zpath),"%s/latent.safetensors",argv[2]);
    p.output_path=output;p.save_still_latent=zpath;p.on_progress=progress;p.on_frame=frame;p.on_latent_step=latent;
    h3_sha256_init(&trajectory);double start=h3_av_now();h3_result *r=h3_generate(ctx,argv[7],&p);
    if(!r){fprintf(stderr,"generation: %s\n",h3_last_error(ctx));h3_free(ctx);free(refs);return 1;}
    int ok=r->kind==H3_RESULT_STILL && r->still_latent && !r->av_state && !r->sampler_state && r->frames==1 && !r->fps && !r->audio_samples && !r->sample_rate && callbacks==1 && transitions==steps+1;
    uint8_t bytes[32];char digest[65];h3_sha256_final(bytes,&trajectory);for(int i=0;i<32;i++)snprintf(digest+i*2,3,"%02x",bytes[i]);
    printf("{\"passed\":%s,\"wall_seconds\":%.6f,\"peak_footprint\":%llu,\"callbacks\":%d,\"latent_callbacks\":%d,\"video_T\":1,\"auxiliary_audio_T\":2,\"target_rows\":%d,\"steps\":%d,\"trajectory_sha256\":\"%s\"}\n",ok?"true":"false",h3_av_now()-start,(unsigned long long)peak,callbacks,transitions,(width/32)*(height/32),steps,digest);
    h3_result_free(r);h3_free(ctx);free(refs);return !ok;
}
