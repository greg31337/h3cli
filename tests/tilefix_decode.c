/* Development-only F32 encoder/decode-only tool; private plan inspection. */
#include "../src/vae/video_vae.c"
#include "src/vae/video_encoder.h"
#include "src/sampling/av_state.h"
#include "src/sampling/sampler_state.h"
#include "src/execution.h"
#include <sys/resource.h>
#include <time.h>
static void die(const char *s) { fprintf(stderr,"%s\n",s);exit(1); }
static double seconds(void) { struct timespec t;clock_gettime(CLOCK_MONOTONIC,&t);return (double)t.tv_sec+(double)t.tv_nsec*1e-9; }
static void axis_json(const tile_axis *a) {
    printf("{\"length\":%d,\"starts\":[",a->length);
    for(int i=0;i<a->count;i++) printf("%s%d",i?",":"",a->starts[i]);
    printf("],\"overlaps\":[");
    for(int i=0;i<a->count-1;i++) printf("%s%d",i?",":"",a->overlaps[i]);
    printf("]}");
}
int main(int argc,char **argv) {
    /* Development-only candidate selection; production callers use params. */
    if(argc!=8) die("usage: tilefix_decode encode|decode|resident|av|sample|plan T H W INPUT OUTPUT WEIGHTS");
    int t=atoi(argv[2]),h=atoi(argv[3]),w=atoi(argv[4]);
    if(t<1||t>1024||h<16||h>4096||w<16||w>4096||h%16||w%16) die("invalid dimensions (H/W are output pixels)");
    int encode=!strcmp(argv[1],"encode"),plan=!strcmp(argv[1],"plan");
    int av=!strcmp(argv[1],"av"),sample=!strcmp(argv[1],"sample");
    if(!encode&&!plan&&!av&&!sample&&strcmp(argv[1],"decode")&&strcmp(argv[1],"resident")) die("invalid mode");
    char error[512];tile_axis y,x;int tile=h3_video_vae_tile_pixels(h,w,error,sizeof(error));
    if(!tile) die(error);
    if(!tile_axis_build(h,tile,&y,error,sizeof(error))||!tile_axis_build(w,tile,&x,error,sizeof(error))) die(error);
    float *input=NULL,*values=NULL;size_t count=0;double load=0,decode=0;
    h3_video_frames frames={0};h3_video_latent latent={0};h3_gpu_stats stats={0};
    if(!plan) {
        size_t n=encode?(size_t)3*t*h*w:(size_t)24*t*(h/16)*(w/16);
        input=malloc(n*sizeof(float));if(!input) die("allocation failed");
        if(av) {
            h3_av_state *s=h3_av_state_load(argv[5],error,sizeof(error));
            if(!s) die(error);
            if(s->info.video_t!=t||s->info.render_height!=h||s->info.render_width!=w||s->info.video_elements!=n) die("AV state shape mismatch");
            memcpy(input,s->video,n*sizeof(float));h3_av_state_free(s);
        } else if(sample) {
            h3_sampler_state *s=h3_sampler_state_load(argv[5],error,sizeof(error));
            if(!s) die(error);
            if(s->latent_t!=t||s->render_height!=h||s->render_width!=w||s->video_elements!=n) die("sampler state shape mismatch");
            memcpy(input,s->video,n*sizeof(float));h3_sampler_state_free(s);
        } else {
            FILE *f=fopen(argv[5],"rb");
            if(!f||fread(input,sizeof(float),n,f)!=n||fgetc(f)!=EOF) die("invalid F32 input size");
            fclose(f);
        }
        double start=seconds();
        if(encode) {
            if(!h3_ref2va_video_vae_encode(argv[7],"src/metal/shaders.metal",input,t,t,h,w,NULL,NULL,&latent,error,sizeof(error))) die(error);
            count=(size_t)24*latent.time*latent.height*latent.width;values=latent.values;stats=latent.gpu_stats;
        } else if(!strcmp(argv[1],"resident")) {
            h3_video_vae_decoder *d=h3_video_vae_decoder_load(argv[7],"src/metal/shaders.metal",h/16,w/16,NULL,NULL,error,sizeof(error));
            if(!d) die(error);
            load=seconds()-start;start=seconds();
            if(!h3_video_vae_decoder_decode(d,input,t,&frames,error,sizeof(error))) die(error);
            decode=seconds()-start;h3_video_vae_decoder_free(d);
        } else if(!h3_video_vae_decode(argv[7],"src/metal/shaders.metal",input,t,h/16,w/16,NULL,NULL,&frames,error,sizeof(error))) die(error);
        if(decode==0.0) decode=seconds()-start;
        if(!encode) {count=(size_t)frames.frames*h*w*3;values=frames.rgb;stats=frames.gpu_stats;}
        for(size_t i=0;i<count;i++) if(!isfinite(values[i])) die("nonfinite output");
        FILE *f=fopen(argv[6],"wb");if(!f||fwrite(values,sizeof(float),count,f)!=count||fclose(f)) die("write failed");
    }
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
    printf("{\"tile\":%d,\"y\":",tile);axis_json(&y);printf(",\"x\":");axis_json(&x);
    printf(",\"spatial_tiles\":%d,\"frames\":%d,\"latent_time\":%d,\"load_seconds\":%.9f,\"decode_seconds\":%.9f,\"peak_rss_bytes\":%ld,\"peak_metal_tensor_bytes\":%llu}\n",y.count*x.count,frames.frames,latent.time,load,decode,usage.ru_maxrss,(unsigned long long)stats.peak_live_bytes);
    free(input);h3_video_frames_free(&frames);h3_video_latent_free(&latent);tile_axis_free(&y);tile_axis_free(&x);return 0;
}
