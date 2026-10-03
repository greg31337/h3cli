/* Independent decoder contexts/streams: actual 36-block tiles, ordered host
 * stitching and all readbacks are timed. Weights are duplicated and reported. */
#include "../src/vae/video_vae.c"
#include "src/sampling/av_state.h"
#include <pthread.h>
typedef struct{h3_video_vae_decoder *d;const float *z;h3_video_frames out;int ok;char error[512];} worker;
static void *run(void *p){worker *w=p;w->ok=h3_video_vae_decoder_decode(w->d,w->z,7,&w->out,w->error,sizeof(w->error));return NULL;}
int main(int argc,char **argv){
    if(argc!=4){fprintf(stderr,"usage: fast_vae_batch WEIGHTS STATE BATCH\n");return 2;}
    int batch=atoi(argv[3]);if(batch!=1&&batch!=2&&batch!=4)return 2;
    char error[512];h3_av_state *s=h3_av_state_load(argv[2],error,sizeof(error));if(!s){fprintf(stderr,"%s\n",error);return 1;}
    float *z=extract_latent_tile(s->video,s->info.video_t,s->info.latent_h,s->info.latent_w,0,0,0,7,16,16,error,sizeof(error));if(!z)return 1;
    worker workers[4]={0};pthread_t threads[4];
    for(int i=0;i<batch;i++){workers[i].z=z;workers[i].d=h3_video_vae_decoder_load(argv[1],"src/metal/shaders.metal",16,16,NULL,NULL,error,sizeof(error));if(!workers[i].d){fprintf(stderr,"%s\n",error);return 1;}}
    int starts[]={0,192,384,576},overlap[]={64,64,64};tile_axis x={0},y={0};
    x.count=batch;x.length=256;x.starts=starts;x.overlaps=overlap;y.count=1;y.length=256;y.starts=starts;y.overlaps=overlap;
    for(int r=0;r<4;r++){
        double start=vae_clock();for(int i=0;i<batch;i++)if(pthread_create(&threads[i],NULL,run,&workers[i]))return 1;
        float *tiles[4];uint64_t peak=0;
        for(int i=0;i<batch;i++){pthread_join(threads[i],NULL);if(!workers[i].ok){fprintf(stderr,"%s\n",workers[i].error);return 1;}tiles[i]=workers[i].out.rgb;peak+=workers[i].out.gpu_stats.peak_live_bytes;}
        h3_video_frames result={0};if(!stitch_tiles(tiles,&y,&x,22,0,&result,error,sizeof(error)))return 1;
        printf("{\"batch\":%d,\"repeat\":%d,\"complete_seconds\":%.9f,\"sum_peak_tensor_bytes\":%llu,\"includes\":\"independent tile contexts, copy, unpack and ordered stitching\"}\n",batch,r,vae_clock()-start,(unsigned long long)peak);fflush(stdout);
        h3_video_frames_free(&result);for(int i=0;i<batch;i++)h3_video_frames_free(&workers[i].out);
    }
    for(int i=0;i<batch;i++)h3_video_vae_decoder_free(workers[i].d);free(z);h3_av_state_free(s);return 0;
}
