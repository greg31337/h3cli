#include "../src/vae/video_vae.c"
#include "src/sampling/av_state.h"
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error);return 1;}}while(0)
static int cancel(int done,int total,void *p){(void)total;(void)p;return done>=3;}
static int low(uint64_t *bytes,void *p){(void)p;*bytes=1;return 1;}
typedef struct{int first;const float *want;int cancel;} sink;
static int frames(const float *rgb,int first,int n,int w,int h,void *opaque){sink *s=opaque;size_t count=(size_t)n*w*h*3;if(s->first!=first)return 1;if(s->want&&memcmp(rgb,s->want+(size_t)first*w*h*3,count*4))return 1;s->first+=n;return s->cancel;}
int main(int argc,char **argv){
    if (argc != 3)
        return 2;
    char error[512];
#ifndef __APPLE__
    h3_sglang_exchange(H3_SGLANG_VERSION);
#endif
    h3_av_state *s=h3_av_state_load(argv[2],error,sizeof(error));CHECK(s);
    h3_video_vae_decoder *d=h3_video_vae_decoder_load(argv[1],"src/metal/shaders.metal",s->info.latent_h,s->info.latent_w,NULL,NULL,error,sizeof(error));CHECK(d);
    h3_video_frames reference={0};CHECK(h3_video_vae_decoder_decode(d,s->video,s->info.video_t,&reference,error,sizeof(error)));
    h3_video_frames standalone={0};CHECK(h3_video_vae_decode(argv[1],"src/metal/shaders.metal",s->video,s->info.video_t,s->info.latent_h,s->info.latent_w,NULL,NULL,&standalone,error,sizeof(error)));
    CHECK(standalone.frames==reference.frames&&standalone.width==reference.width&&standalone.height==reference.height);
    CHECK(!memcmp(reference.rgb,standalone.rgb,(size_t)reference.frames*reference.width*reference.height*3*4));h3_video_frames_free(&standalone);
    uint64_t live=0;
    for(int i=0;i<4;i++){
        sink out={.want=reference.rgb};CHECK(h3_video_vae_decoder_stream_progress(d,s->video,s->info.video_t,frames,&out,NULL,NULL,error,sizeof(error)));CHECK(out.first==s->info.frames);
        h3_gpu_stats stats;CHECK(h3_gpu_get_stats(d->vae.gpu,&stats));if(i)CHECK(live==stats.live_bytes);live=stats.live_bytes;
    }
    sink out={0};CHECK(!h3_video_vae_decoder_stream_progress(d,s->video,s->info.video_t,frames,&out,cancel,NULL,error,sizeof(error)));CHECK(strstr(error,"cancelled"));
    float saved=s->video[0];s->video[0]=NAN;out=(sink){0};
    CHECK(!h3_video_vae_decoder_stream_progress(d,s->video,s->info.video_t,frames,&out,NULL,NULL,error,sizeof(error)));
    s->video[0]=saved;
    h3_memory_set_test_query(low,NULL);CHECK(!h3_video_vae_decoder_stream_progress(d,s->video,s->info.video_t,frames,&out,NULL,NULL,error,sizeof(error)));h3_memory_set_test_query(NULL,NULL);
    out=(sink){.cancel=1};CHECK(!h3_video_vae_decoder_stream_progress(d,s->video,s->info.video_t,frames,&out,NULL,NULL,error,sizeof(error)));
    out=(sink){.want=reference.rgb};CHECK(h3_video_vae_decoder_stream_progress(d,s->video,s->info.video_t,frames,&out,NULL,NULL,error,sizeof(error)));
    CHECK(out.first==s->info.frames);h3_video_frames_free(&reference);h3_video_vae_decoder_free(d);h3_av_state_free(s);
    puts("PASS streamed/materialized bitwise equality, repeated allocation plateau, mid-tile cancellation, failed admission, sink failure and clean retry");return 0;
}
