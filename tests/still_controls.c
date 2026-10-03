/* Private test hooks only: public decoder always uses the declared slice. */
#include "../src/vae/video_vae.c"
#include "src/media/ffmpeg.h"
#include <unistd.h>
static int write_values(const char *path,const float *v,size_t count) {FILE *f=fopen(path,"wb");if(!f)return 0;int ok=fwrite(v,4,count,f)==count;return !fclose(f)&&ok;}
int main(int argc,char **argv) {
    if(argc!=4)return 2;char error[512];h3_still_latent z={0};h3_video_frames out={0};
    if(!h3_still_latent_load(argv[2],&z,error,sizeof(error)))goto fail;
    h3_image_vae_decoder *d=h3_image_vae_load(argv[1],"src/metal/shaders.metal",z.height,z.width,NULL,NULL,error,sizeof(error));if(!d)goto fail;
    int ok=1;
    for(int t=0;t<4 && ok;t++) {
        d->info.output_slice=d->core.vae.image_slice=t;
        ok=h3_image_vae_decode(d,z.values,NULL,NULL,&out,error,sizeof(error));
        char path[1024];snprintf(path,sizeof(path),"%s/candidate-slice-%d.f32",argv[3],t);
        if(ok)ok=write_values(path,out.rgb,(size_t)out.width*out.height*3);h3_video_frames_free(&out);
    }
    /* Reversible diagnostic input error: supply raw instead of normalized. */
    d->info.output_slice=d->core.vae.image_slice=3;
    size_t area=(size_t)z.height*z.width;
    for(int c=0;c<24;c++)for(size_t i=0;i<area;i++)z.values[c*area+i]=z.values[c*area+i]*d->info.deviation[c]+d->info.mean[c];
    if(ok)ok=h3_image_vae_decode(d,z.values,NULL,NULL,&out,error,sizeof(error));
    char path[1024];snprintf(path,sizeof(path),"%s/wrong-normalization.f32",argv[3]);
    if(ok)ok=write_values(path,out.rgb,(size_t)out.width*out.height*3);
    h3_video_frames_free(&out);h3_image_vae_free(d);h3_still_latent_free(&z);if(!ok)goto fail;
    /* Failed PNG encoder leaves an existing destination intact. */
    char output[]="/tmp/h3-still-png-XXXXXX";int fd=mkstemp(output);if(fd<0)return 1;
    if(write(fd,"sentinel",8)!=8)return 1;close(fd);setenv("H3_FFMPEG","/usr/bin/false",1);
    uint8_t rgb[3]={0};ok=!h3_ffmpeg_write_png(output,rgb,1,1,error,sizeof(error));FILE *f=fopen(output,"rb");char old[9]={0};
    ok=ok && f && fread(old,1,8,f)==8 && !strcmp(old,"sentinel");if(f)fclose(f);unlink(output);
    if(!ok)goto fail;puts("all four slices, normalization control and atomic PNG failure passed");return 0;
fail:fprintf(stderr,"%s\n",error);return 1;
}
