/* Native reference vision diagnostic. Fixed production fixture bounds the
 * optional intermediate capture; no language encoder or denoising runs. */
#include "src/sglang/sglang.h"
#include "src/conditioning/vision_encoder.h"
#include "src/media/ffmpeg.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static void die(const char *s){fprintf(stderr,"%s\n",s);exit(1);}
int main(int argc,char **argv){
    if(argc!=4)die("usage: bin/cuda_sglang_vision WEIGHTS IMAGE OUTDIR");
    h3_sglang_exchange(1);char error[1024];float *pixels=NULL;
    if(!h3_ffmpeg_read_image_f32(argv[2],640,480,H3_IMAGE_FIT_STRETCH,&pixels,error,sizeof(error)))die(error);
    h3_vision_output out={0};
    if(!h3_vision_encode_bf16(argv[1],"src/metal/shaders.metal",pixels,1,480,640,NULL,NULL,&out,error,sizeof(error)))die(error);
    free(pixels);
    const uint16_t *tensors[]={out.merged,out.deepstack[0],out.deepstack[1],out.deepstack[2]};
    const char *names[]={"merged","deepstack-0","deepstack-1","deepstack-2"};
    for(int i=0;i<4;i++){char path[4096];snprintf(path,sizeof(path),"%s/%s.bf16",argv[3],names[i]);FILE *f=fopen(path,"wbx");
        if(!f||fwrite(tensors[i],2,out.tokens*5120,f)!=out.tokens*5120||fclose(f))die("cannot retain vision diagnostic");}
    h3_vision_output_free(&out);puts("native reference vision capture complete");return 0;
}
