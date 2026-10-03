#include "src/media/ffmpeg.h"
#include "src/sglang/sglang.h"
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

/* Tiny deterministic media fixture, independent of model inference. */
int main(int argc,char **argv) {
    if(argc!=7)return 2;
    h3_output_encoding encoding={0};char error[512];
    if(strcmp(argv[2],"auto")&&!h3_output_quality_parse(argv[2],&encoding.quality))return 2;
    if(strcmp(argv[3],"auto")){encoding.crf=atoi(argv[3]);encoding.crf_set=1;}
    encoding.lossless_video=atoi(argv[4]);h3_sglang_exchange(atoi(argv[5]));
    enum {W=32,H=32,F=4,N=6400};
    uint8_t rgb[W*H*F*3];float pcm[2*N];
    for(size_t i=0;i<sizeof(rgb);i++)rgb[i]=(uint8_t)((i*37+i/19)%256);
    /* Solid primaries test BT.709 conversion independently of chroma edges. */
    for(int f=0;f<F;f++)for(int y=0;y<H;y++)for(int x=0;x<16;x++) {
        size_t i=((size_t)f*H*W+(size_t)y*W+(size_t)x)*3;
        rgb[i]=(uint8_t)(f==0?255:0);rgb[i+1]=(uint8_t)(f==1?255:0);rgb[i+2]=(uint8_t)(f==2?255:0);
    }
    for(size_t i=0;i<2*N;i++)pcm[i]=(float)((int)(i%100)-50)/500;
    const float *audio=strcmp(argv[6],"silent")?pcm:NULL;
    int ok;
    if(!strcmp(argv[6],"buffered"))ok=h3_ffmpeg_write_av_rgb24_f32(argv[1],rgb,F,W,H,24,pcm,N,2,32000,&encoding,error,sizeof(error));
    else if(!audio)ok=h3_ffmpeg_write_rgb24(argv[1],rgb,F,W,H,24,&encoding,error,sizeof(error));
    else {
        h3_ffmpeg_writer *w=h3_ffmpeg_writer_open(argv[1],F,W,H,24,audio,N,2,32000,&encoding,error,sizeof(error));
        ok=w!=NULL;
        for(int f=0;ok&&f<F;f++)ok=h3_ffmpeg_writer_write(w,rgb+(size_t)f*W*H*3,1,error,sizeof(error));
        if(ok)ok=h3_ffmpeg_writer_finish(w,error,sizeof(error));else h3_ffmpeg_writer_abort(w);
    }
    if(!ok){fprintf(stderr,"%s\n",error);return 1;}
    return 0;
}
