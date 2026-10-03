/* Public decode-only failure handling; creates no denoising fixtures. */
#include "src/h3.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);return 1;}}while(0)
static int cancel_frame(const h3_frame *f,void *p){(void)f;(*(int*)p)++;return 1;}
static int cancel_mux(const char *phase,int done,int total,void *p){(void)total;if(!strcmp(phase,"FFmpeg")&&!done){(*(int*)p)++;return 1;}return 0;}
int main(int argc,char **argv){
    if (argc != 3)
        return 2;
    char error[512] = {0}, dir[] = "/tmp/h3-fast-vae-delivery-XXXXXX", path[512];
    CHECK(mkdtemp(dir));
    snprintf(path,sizeof(path),"%s/output.mp4",dir);int called=0;
    h3_decode_options opt={.output_path=path,.on_frame=cancel_frame,.callback_opaque=&called};
    CHECK(!h3_decode_av_state(argv[1],argv[2],&opt,error,sizeof(error)));CHECK(called>0);CHECK(access(path,F_OK));
    opt.on_frame=NULL;opt.on_progress=cancel_mux;called=0;
    CHECK(!h3_decode_av_state(argv[1],argv[2],&opt,error,sizeof(error)));CHECK(called>0);CHECK(access(path,F_OK));
    opt.on_progress=NULL;CHECK(!setenv("H3_FFMPEG","/usr/bin/false",1));
    CHECK(!h3_decode_av_state(argv[1],argv[2],&opt,error,sizeof(error)));CHECK(access(path,F_OK));unsetenv("H3_FFMPEG");
    h3_result *result=h3_decode_av_state(argv[1],argv[2],&opt,error,sizeof(error));CHECK(result);h3_result_free(result);CHECK(!access(path,R_OK));
    CHECK(!unlink(path));CHECK(!rmdir(dir));puts("PASS public default decode: frame cancellation, mux cancellation, FFmpeg failure, no partial output, successful retry");return 0;
}
