/* Stock decode normalization control and atomic PNG failure. */
#include "../src/vae/video_vae.c"
#include "src/media/ffmpeg.h"
#include <unistd.h>
static int low(uint64_t *bytes,void *opaque){(void)opaque;*bytes=1;return 1;}
int main(int argc,char **argv) {
    if(argc!=4)return 2;char error[512];h3_still_latent z={0};h3_video_frames out={0};
    if(!h3_still_latent_load(argv[2],&z,error,sizeof(error)))goto fail;
    h3_video_vae_decoder *d=h3_video_vae_still_load(argv[1],"src/metal/shaders.metal",z.height,z.width,NULL,NULL,error,sizeof(error));if(!d)goto fail;
    h3_memory_set_test_query(low,NULL);
    int rejected=!h3_video_vae_still_decode(d,z.values,NULL,NULL,&out,error,sizeof(error)) && !out.rgb;
    h3_memory_set_test_query(NULL,NULL);if(!rejected)goto fail;
    uint64_t live=0;
    for(int i=0;i<3;i++) {
        if(!h3_video_vae_still_decode(d,z.values,NULL,NULL,&out,error,sizeof(error)))goto fail;
        h3_gpu_stats stats;if(!h3_gpu_get_stats(d->vae.gpu,&stats) || (i && stats.live_bytes!=live))goto fail;
        live=stats.live_bytes;h3_video_frames_free(&out);
    }
    size_t area=(size_t)z.height*z.width;
    for(int c=0;c<24;c++)for(size_t i=0;i<area;i++)z.values[c*area+i]=z.values[c*area+i]*d->latent_std[c]+d->latent_mean[c];
    int ok=h3_video_vae_still_decode(d,z.values,NULL,NULL,&out,error,sizeof(error));
    char path[1024];snprintf(path,sizeof(path),"%s/wrong-normalization.f32",argv[3]);
    FILE *f=ok?fopen(path,"wb"):NULL;if(f){ok=fwrite(out.rgb,4,(size_t)out.width*out.height*3,f)==(size_t)out.width*out.height*3;ok=!fclose(f)&&ok;}else ok=0;
    h3_video_frames_free(&out);h3_video_vae_decoder_free(d);h3_still_latent_free(&z);if(!ok)goto fail;
    char output[]="/tmp/h3-still-png-XXXXXX";int fd=mkstemp(output);if(fd<0)return 1;
    if(write(fd,"sentinel",8)!=8)return 1;close(fd);setenv("H3_FFMPEG","/usr/bin/false",1);
    uint8_t rgb[3]={0};ok=!h3_ffmpeg_write_png(output,rgb,1,1,error,sizeof(error));f=fopen(output,"rb");char old[9]={0};
    ok=ok && f && fread(old,1,8,f)==8 && !strcmp(old,"sentinel");if(f)fclose(f);unlink(output);
    if(!ok)goto fail;puts("normalization control and atomic PNG failure passed");return 0;
fail:fprintf(stderr,"%s\n",error);return 1;
}
