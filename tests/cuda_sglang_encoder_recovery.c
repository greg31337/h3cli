/* In-process reference encoder cancellation/recovery between GPU calls. */
#define main existing_isolation_main
#include "cuda_sglang_isolation.c"
#undef main
#include "src/vae/video_encoder.h"
#include "src/media/ffmpeg.h"
extern size_t cudnnGetVersion(void);
static int cancel_tile(int complete,int total,void *opaque){(void)opaque;return total>0&&complete>0;}
int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: encoder-recovery WEIGHTS IMAGE ORACLE_MOMENTS.f32\n");return 2;}
    size_t linked=cudnnGetVersion();CHECK(linked==92000);
    uint16_t *before=evaluate(0);float *pixels=NULL;h3_video_moments moments={0};
    CHECK(h3_ffmpeg_read_image_f32(argv[2],640,480,H3_IMAGE_FIT_STRETCH,&pixels,error,sizeof(error)));
    h3_sglang_exchange(1);h3_sglang_vae_byte_pixels(pixels,3u*640u*480u,0);
    CHECK(!h3_video_vae_encode_moments(argv[1],"src/metal/shaders.metal",pixels,1,480,640,cancel_tile,NULL,&moments,error,sizeof(error)));
    CHECK(!moments.values&&strstr(error,"cancelled"));
    CHECK(h3_video_vae_encode_moments(argv[1],"src/metal/shaders.metal",pixels,1,480,640,NULL,NULL,&moments,error,sizeof(error)));
    CHECK(moments.time==1&&moments.height==30&&moments.width==40);
    size_t count=48u*30u*40u;float *gold=malloc(count*4);CHECK(gold);FILE*f=fopen(argv[3],"rb");CHECK(f);
    CHECK(fread(gold,4,count,f)==count&&fgetc(f)==EOF);CHECK(!fclose(f));CHECK(!memcmp(gold,moments.values,count*4));
    h3_video_moments_free(&moments);free(gold);free(pixels);h3_sglang_exchange(0);
    CHECK(cudnnGetVersion()==linked);uint16_t *after=evaluate(0);CHECK(!memcmp(before,after,E*2));free(before);free(after);
    puts("PASS: encoder cancellation, exact recovery moments, stable cuDNN and surrounding GPU output");return 0;
}
