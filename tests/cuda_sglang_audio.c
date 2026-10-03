/* Crossed full-audio-decoder replay; no sampling or lossy encoding. */
#include "src/vae/audio_vae.h"
#include "tests/reference_av_fixture.h"
#include "src/sglang/sglang.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>

int main(int argc,char **argv) {
    if(argc!=4){fprintf(stderr,"usage: cuda_sglang_audio WEIGHTS STATE OUTPUT.f32\n");return 2;}
    char error[512]={0};h3_sglang_exchange(1);
    h3_av_state *s=reference_av_load(argv[2],error,sizeof(error));
    if(!s){fprintf(stderr,"%s\n",error);return 1;}
    h3_audio_waveform out={0};struct timespec begin,end;clock_gettime(CLOCK_MONOTONIC,&begin);
    int ok=h3_audio_vae_decode(argv[1],"src/metal/shaders.metal",s->audio,s->info.audio_t,
                             NULL,NULL,&out,error,sizeof(error));
    clock_gettime(CLOCK_MONOTONIC,&end);
    if(ok){
        size_t n=(size_t)out.channels*out.samples;
        for(size_t i=0;i<n;i++)if(!isfinite(out.pcm[i])){ok=0;break;}
        FILE *f=ok?fopen(argv[3],"wbx"):NULL;
        if(!f)ok=0;else{if(fwrite(out.pcm,4,n,f)!=n)ok=0;if(fclose(f))ok=0;}
        if(ok)printf("{\"channels\":%d,\"samples\":%d,\"sample_rate\":%d,\"decode_seconds\":%.9f,\"peak_tensor_bytes\":%llu}\n",
            out.channels,out.samples,out.sample_rate,(double)(end.tv_sec-begin.tv_sec)+(double)(end.tv_nsec-begin.tv_nsec)/1e9,
            (unsigned long long)out.gpu_stats.peak_live_bytes);
    }
    if(!ok)fprintf(stderr,"audio replay failed: %s\n",error);
    h3_audio_waveform_free(&out);h3_av_state_free(s);return ok?0:1;
}
