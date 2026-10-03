/* Decode-only benchmark. Same production tile/decode code and resident plans.
 * A cropped tile is diagnostic; full reports require verified six-step states. */
#include "../src/vae/video_vae.c"
#include "tests/reference_av_fixture.h"
#include "src/execution.h"
#include <sys/resource.h>
static void die(const char *s){fprintf(stderr,"%s\n",s);exit(1);}
typedef struct { FILE *f;int next;double write_seconds; } sink;
static int receive(const float *rgb,int first,int count,int w,int h,void *p){
    sink *s=p;if(s->next!=first)return 1;
    double t=vae_clock();size_t n=(size_t)count*w*h*3;
    int ok=!s->f||fwrite(rgb,4,n,s->f)==n;s->write_seconds+=vae_clock()-t;s->next+=count;return !ok;
}
int main(int argc,char **argv){
    if(argc!=7)die("usage: fast_vae_bench WEIGHTS STATE MODE OUTPUT.f32 REPEATS tile|full");
    if(strcmp(argv[3],"default"))die("only the current default decoder is available");
    char error[512];h3_av_state *s=reference_av_load(argv[2],error,sizeof(error));if(!s)die(error);
    int tile=!strcmp(argv[6],"tile"),repeats=atoi(argv[5]);if(repeats<1||repeats>10)die("bad repeats");
    int h=tile?16:s->info.latent_h,w=tile?16:s->info.latent_w,t=tile?7:s->info.video_t;
    if(h>s->info.latent_h||w>s->info.latent_w||t>s->info.video_t)die("state smaller than tile");
    float *input=tile?extract_latent_tile(s->video,s->info.video_t,s->info.latent_h,s->info.latent_w,0,0,0,t,h,w,error,sizeof(error)):s->video;
    if(!input)die(error);
    double start=vae_clock();
    h3_video_vae_decoder *d=h3_video_vae_decoder_load(argv[1],"src/metal/shaders.metal",h,w,NULL,NULL,error,sizeof(error));
    if (!d)
        die(error);
    double load = vae_clock() - start;
    for(int r=0;r<repeats;r++){
        const char *name=argv[3];const char *filename=argv[4];
        sink out={0};if(r==0 && strcmp(argv[4],"-")){out.f=fopen(filename,"wb");if(!out.f)die("open output failed");}
        start=vae_clock();int ok=h3_video_vae_decoder_stream_progress(d,input,t,receive,&out,NULL,NULL,error,sizeof(error));
        if (out.f && fclose(out.f))
            die("write failed");
        if (!ok)
            die(error);
        h3_gpu_stats stats;h3_gpu_get_stats(d->vae.gpu,&stats);
        struct rusage ru;getrusage(RUSAGE_SELF,&ru);
        h3_memory_snapshot memory={0};h3_memory_sample(&memory);
        uint64_t device_bytes=0;
#ifdef __APPLE__
        device_bytes=h3_gpu_device_allocated_bytes();
#endif
        printf("{\"mode\":\"%s\",\"repeat\":%d,\"tile\":%s,\"load_seconds\":%.9f,\"decode_seconds\":%.9f,\"write_seconds\":%.9f,\"frames\":%d,\"width\":%d,\"height\":%d,\"peak_tensor_bytes\":%llu,\"live_tensor_bytes\":%llu,\"peak_rss_native\":%ld,\"physical_footprint_bytes\":%llu,\"process_compressed_bytes\":%llu,\"metal_current_allocated_bytes\":%llu,\"swap_used_bytes\":%llu,\"gemm_seconds_cumulative\":%.9f,\"attention_seconds_cumulative\":%.9f}\n",
            name,r,tile?"true":"false",load,vae_clock()-start,out.write_seconds,out.next,w*16,h*16,(unsigned long long)stats.peak_live_bytes,(unsigned long long)stats.live_bytes,ru.ru_maxrss,(unsigned long long)memory.physical_footprint,(unsigned long long)memory.process_compressed,(unsigned long long)device_bytes,(unsigned long long)memory.swap_used,stats.linear_seconds,stats.attention_seconds);fflush(stdout);
        if(tile && r==0 && strcmp(argv[4],"-")){
            h3_gpu_tensor *parts[]={d->vae.latent,d->vae.rope_cos,d->vae.rope_sin};const char *names[]={"input","cos","sin"};
            for(int j=0;j<3;j++){
                size_t m=h3_gpu_tensor_elements(parts[j]);float *values=malloc(m*4);char tensor_path[4096];snprintf(tensor_path,sizeof(tensor_path),"%s.%s",argv[4],names[j]);
                FILE *stream=fopen(tensor_path,"wb");if(!values||!stream||!h3_gpu_tensor_read_f32(parts[j],values,m)||fwrite(values,4,m,stream)!=m||fclose(stream))die("tile input dump failed");free(values);
            }
            size_t n=(size_t)d->vae.sequence*OUTPUT_PATCH;float *raw=malloc(n*4);char path[4096];snprintf(path,sizeof(path),"%s.projected",argv[4]);
            FILE *f=fopen(path,"wb");if(!raw||!f||!h3_gpu_tensor_read_f32(d->vae.projected,raw,n)||fwrite(raw,4,n,f)!=n||fclose(f))die("projected dump failed");free(raw);
        }
    }
    h3_gpu_profile_mark(d->vae.gpu,"benchmark");h3_video_vae_decoder_free(d);if(tile)free(input);h3_av_state_free(s);return 0;
}
