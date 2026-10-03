/* Real image/video CNN moments replay. No sampling or language encoder. */
#include "src/sglang/sglang.h"
#include "src/vae/video_encoder.h"
#include "src/media/ffmpeg.h"
#include "src/media/refvideo.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv){
    if(argc!=4 && (argc!=5 || strcmp(argv[4],"video"))){
        fprintf(stderr,"usage: encoder WEIGHTS INPUT OUTPUT.f32 [video]\n");return 2;
    }
    h3_sglang_exchange(1);char error[1024]={0};float *pixels=NULL;h3_video_moments out={0};
    int video=argc==5,total=0;
    h3_refvideo_plan plan={0};
    int ok=video ? h3_refvideo_read_sglang(argv[2],640,480,124,&total,&pixels,&plan,error,sizeof(error)) :
        h3_ffmpeg_read_image_f32(argv[2],640,480,H3_IMAGE_FIT_STRETCH,&pixels,error,sizeof(error));
    if(ok)h3_sglang_vae_byte_pixels(pixels,3u*640u*480u*(video?plan.frames:1),0);
    if(ok && video)ok=h3_ref2va_video_vae_moments(argv[1],"src/metal/shaders.metal",pixels,
        plan.frames,plan.vae_frames,480,640,NULL,NULL,&out,error,sizeof(error));
    else if(ok)ok=h3_video_vae_encode_moments(argv[1],"src/metal/shaders.metal",pixels,1,480,640,NULL,NULL,&out,error,sizeof(error));
    free(pixels);
    if(ok){size_t n=(size_t)48*out.time*out.height*out.width;FILE*f=fopen(argv[3],"wbx");if(!f)ok=0;else{ok=fwrite(out.values,4,n,f)==n;if(fclose(f))ok=0;}}
    h3_video_moments_free(&out);if(!ok)fprintf(stderr,"encoder diagnostic failed: %s\n",error);return ok?0:1;
}
