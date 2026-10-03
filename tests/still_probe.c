#include "src/vae/image_vae.h"
#include "src/weights/weights.h"
#include "src/media/ffmpeg.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int cancel(int a,int b,void *p) {(void)b;(void)p;return a>=1;}
static int progress(int a,int b,void *p) { (void)p;if(a==b)fprintf(stderr,"phase %d/%d\n",a,b);return 0; }
static int write_raw(const char *p,const float *v,size_t n) {FILE *f=fopen(p,"wb");if(!f)return 0;int ok=fwrite(v,4,n,f)==n;return !fclose(f)&&ok;}
int main(int argc,char **argv) {
    if(argc<3){fprintf(stderr,"usage: still_probe inspect MODEL | encode MODEL IMAGE W H LATENT | decode MODEL LATENT RGB [repeat]\n");return 2;}
    char error[512]={0};h3_image_vae_info info;
    if(!strcmp(argv[1],"validate")) {h3_st_header h={0};int ok=h3_st_read_header(argv[2],&h,error,sizeof(error)) && h3_image_vae_validate(&h,&info,error,sizeof(error));h3_st_free_header(&h);if(!ok)goto fail;return 0;}
    if(!strcmp(argv[1],"store")) {h3_weight_store *store=h3_weight_store_open(argv[2],error,sizeof(error));if(!store)goto fail;h3_weight_store_free(store);return 0;}
    if(!h3_image_vae_inspect(argv[2],&info,error,sizeof(error)))goto fail;
    if(!strcmp(argv[1],"inspect")) {printf("artifact=%s\ncompatibility=%s\nidentity=%s\nslice=%d\n",info.artifact_sha256,info.compatibility_sha256,info.identity,info.output_slice);return 0;}
    if(!strcmp(argv[1],"encode") && argc==7) {
        int w=atoi(argv[4]),h=atoi(argv[5]);float *pixels=NULL;h3_video_latent z={0};
        if(!h3_ffmpeg_read_image_f32(argv[3],w,h,H3_IMAGE_FIT_COVER,&pixels,error,sizeof(error)))goto fail;
        double t=h3_av_now();
        int ok=h3_image_vae_encode(argv[2],"src/metal/shaders.metal",pixels,h,w,progress,NULL,&z,error,sizeof(error));free(pixels);
        if(!ok)goto fail;
        h3_still_latent out={.height=z.height,.width=z.width,.values=z.values};strcpy(out.compatibility_sha256,info.compatibility_sha256);
        ok=h3_still_latent_save(argv[6],&out,error,sizeof(error));h3_video_latent_free(&z);
        fprintf(stderr,"encoder_seconds=%.6f\n",h3_av_now()-t);if(!ok)goto fail;return 0;
    }
    if(!strcmp(argv[1],"decode") && argc>=5) {
        h3_still_latent z={0};h3_video_frames out={0};
        if(!h3_still_latent_load(argv[3],&z,error,sizeof(error)))goto fail;
        double start=h3_av_now();h3_image_vae_decoder *d=h3_image_vae_load(argv[2],"src/metal/shaders.metal",z.height,z.width,progress,NULL,error,sizeof(error));
        if(!d){h3_still_latent_free(&z);goto fail;}double loaded=h3_av_now();
        int repeat=argc>5?atoi(argv[5]):1,ok=1;
        if(h3_image_vae_decode(d,z.values,cancel,NULL,&out,error,sizeof(error)) || out.rgb || !strstr(error,"cancel")) {fprintf(stderr,"cancellation failed\n");return 1;}
        fprintf(stderr,"cancelled first block; retrying resident decoder\n");
        for(int i=0;ok && i<repeat;i++) {double t=h3_av_now();h3_video_frames_free(&out);
            ok=h3_image_vae_decode(d,z.values,progress,NULL,&out,error,sizeof(error));
            fprintf(stderr,"decode_%d_seconds=%.6f gpu_live=%llu peak=%llu\n",i,h3_av_now()-t,(unsigned long long)out.gpu_stats.live_bytes,(unsigned long long)out.gpu_stats.peak_live_bytes);
        }
        if(ok)ok=write_raw(argv[4],out.rgb,(size_t)out.width*out.height*3);
        fprintf(stderr,"load_seconds=%.6f\n",loaded-start);
        h3_video_frames_free(&out);h3_still_latent_free(&z);h3_image_vae_free(d);if(!ok)goto fail;return 0;
    }
    return 2;
fail:fprintf(stderr,"still_probe: %s\n",error);return 1;
}
