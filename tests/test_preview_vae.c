#include "src/denoise/adaptive_cache.h"
#include "src/denoise/attention.h"
#include "src/weights/quant.h"
#include "src/media/delivery.h"
#include "src/sampling/av_state.h"
#include "src/media/ffmpeg.h"
#include "src/memory.h"
#include "src/execution.h"
#include <assert.h>
#include <math.h>
#include <signal.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/resource.h>
#include <unistd.h>
#include <errno.h>
#include <sys/wait.h>

static char error[512];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
#ifdef __APPLE__
uint64_t h3_test_metal_allocated(void);
#else
#include <cuda_runtime_api.h>
static uint64_t h3_test_metal_allocated(void){size_t available=0,total=0;
    return cudaMemGetInfo(&available,&total)==cudaSuccess?(uint64_t)(total-available):0;}
#endif
typedef struct {int next,cancel,pattern;const float *expected;FILE *out;uint64_t peak;} sink;
static int frames(const float *rgb,int first,int count,int w,int h,void *opaque) {
    sink *s=opaque;CHECK(first==s->next);size_t frame=(size_t)w*(size_t)h*3,n=(size_t)count*frame;
    uint64_t allocated=h3_test_metal_allocated();if(allocated>s->peak)s->peak=allocated;
    for(size_t i=0;i<n;i++)CHECK(isfinite(rgb[i])&&rgb[i]>=0&&rgb[i]<=1);
    if(s->expected)CHECK(!memcmp(rgb,s->expected+(size_t)first*frame,n*sizeof(float)));
    if(s->pattern==1)for(int t=0;t<count;t++)for(int y=0;y<h;y++)for(int x=0;x<w;x++)for(int c=0;c<3;c++){
        float want=(float)(c*4+(y%2)*2+x%2)/16;CHECK(rgb[((size_t)t*(size_t)h*(size_t)w+(size_t)y*(size_t)w+(size_t)x)*3+(size_t)c]==want);
    }
    if(s->pattern==2)for(int t=0;t<count;t++){
        int kept=0,raw=0;while(1){if(raw%20>=3&&kept++==first+t)break;raw++;}
        int latent=raw/4;
        float current=3*tanhf((float)(latent+1)/96),past=latent?3*tanhf((float)latent/96):0;
        for(int y=0;y<h;y++)for(int x=0;x<w;x++)for(int c=0;c<3;c++){
            float want=(current+past)*0.125f+(float)(c*4+(y%2)*2+x%2)/16;
            float got=rgb[((size_t)t*(size_t)h*(size_t)w+(size_t)y*(size_t)w+(size_t)x)*3+(size_t)c];
            CHECK(fabsf(got-want)<0.004f); /* FP16 rounding in this structural oracle only. */
        }
    }
    if (s->out)
        CHECK(fwrite(rgb, 4, n, s->out) == n);
    s->next += count;
    return s->cancel;
}
static int low_memory(uint64_t *bytes,void *opaque){(void)opaque;*bytes=1;return 1;}
static int cancel_progress(int done,int total,void *opaque){(void)done;(void)total;(void)opaque;return 1;}
static int cancel_frame(const h3_frame *f,void *opaque){(void)f;if(opaque)(*(int *)opaque)++;return 1;}
static int cancel_mux(const char *phase,int done,int total,void *opaque){(void)total;int stop=!strcmp(phase,"FFmpeg")&&!done;if(stop&&opaque)(*(int *)opaque)++;return stop;}
static void temporal_input(float *z,int time,int h,int w){
    for(int t=0;t<time;t++)for(int p=0;p<h*w;p++)z[t*h*w+p]=(float)(t+1)/32;
}
static void host(void) {
    h3_params p=H3_PARAMS_DEFAULT,z={0};CHECK(!p.preview_vae&&!p.preview_vae_model&&!z.preview_vae);
    CHECK(h3_tiny_vae_frames(7)==22&&h3_tiny_vae_frames(17)==56&&h3_tiny_vae_frames(27)==90&&h3_tiny_vae_frames(72)==243&&h3_tiny_vae_frames(107)==362);
    CHECK(!h3_tiny_vae_frames(2)&&!h3_tiny_vae_frames(8)&&!h3_tiny_vae_frames(112));
    CHECK(!h3_preview_vae_options(0,"missing",error,sizeof(error)));
    char directory[]="/tmp/h3-preview-vae-XXXXXX";CHECK(mkdtemp(directory));char path[512],meta[540],video[512];
    snprintf(path,sizeof(path),"%s/test.h3av",directory);snprintf(meta,sizeof(meta),"%s.presentation",path);snprintf(video,sizeof(video),"%s/test.mp4",directory);
    uint8_t sig[32]={0};h3_av_state *state=h3_av_state_new(32,32,90,42,sig);CHECK(state);
    memset(state->video,0,state->info.video_elements*4);memset(state->audio,0,state->info.audio_elements*4);
    h3_result r={.av_state=state,.presentation={.version=9,.av_metadata_identity=1,.ref2va=1,
        .render_width=32,.render_height=32,.width=64,.height=64,.trim_frames=39,.trim_samples=52000,
        .fps=24,.sample_rate=32000,.codec_version=2}};
    h3_presentation base=r.presentation,read;
    for(int attention=0;attention<=4;attention++)for(int quant=0;quant<=2;quant++)for(int adaptive=0;adaptive<=2;adaptive++) {
        if(adaptive&&attention&&attention!=H3_ATTENTION_SUBBLOCK)continue;
        if(quant&&adaptive)continue;
        r.presentation=base;r.presentation.cuda_attention=attention;r.presentation.cuda_denoise_quant=quant;
        r.presentation.adaptive_cache=adaptive;r.presentation.adaptive_version=h3_adaptive_execution_recipe(adaptive,quant,r.presentation.ref2va,1);
        r.presentation.attention_version=h3_attention_execution_recipe(attention,quant);
        r.presentation.attention_plan=h3_attention_plan(attention);
        r.presentation.quant_version=h3_quant_execution_recipe(quant,adaptive,attention);
        r.presentation.quant_model_metadata=quant!=0;if(quant)memset(r.presentation.quant_model_sha256,'a',64);
        if(attention==H3_ATTENTION_SOL)r.presentation.cuda_sol=(h3_cuda_sol_options)H3_CUDA_SOL_DEFAULT;
        if(attention==H3_ATTENTION_SUBBLOCK){r.presentation.subblock_sparsity=.75f;r.presentation.subblock_warmup=2;}
        if(adaptive){r.presentation.adaptive_cache_warmup=2;r.presentation.adaptive_cache_threshold=.04f;r.presentation.adaptive_cache_max_hits=2;}
        CHECK(h3_result_save_av_state(&r,path,error,sizeof(error)));
        CHECK(h3_presentation_load(path,state,&read,error,sizeof(error))==1&&!memcmp(&read,&r.presentation,sizeof(read)));
        if(adaptive) {
            r.presentation.adaptive_version=H3_ADAPTIVE_REFERENCE_VERSION;
            CHECK(!h3_result_save_av_state(&r,path,error,sizeof(error)));
            r.presentation.adaptive_version=H3_ADAPTIVE_CONTINUATION_VERSION;
            r.presentation.trim_frames=r.presentation.trim_samples=0;
            CHECK(h3_result_save_av_state(&r,path,error,sizeof(error))); /* keep-prefix */
            r.presentation.adaptive_version=H3_ADAPTIVE_REFERENCE_VERSION;
            CHECK(h3_result_save_av_state(&r,path,error,sizeof(error))); /* ordinary */
            r.presentation.trim_frames=base.trim_frames;r.presentation.trim_samples=base.trim_samples;
            r.presentation.adaptive_version=H3_ADAPTIVE_CONTINUATION_VERSION;
        }
        r.presentation.version=7;CHECK(!h3_result_save_av_state(&r,path,error,sizeof(error)));r.presentation.version=9;
        r.presentation.adaptive_cache_warmup=17;CHECK(!h3_result_save_av_state(&r,path,error,sizeof(error)));
    }
    r.presentation=base;r.presentation.codec_version=1;
    CHECK(h3_result_save_av_state(&r,path,error,sizeof(error)));
    CHECK(h3_presentation_load(path,state,&read,error,sizeof(error))==1&&!memcmp(&read,&r.presentation,sizeof(read)));
    r.presentation.codec_version=3;CHECK(!h3_result_save_av_state(&r,path,error,sizeof(error)));r.presentation=base;
    CHECK(h3_result_save_av_state(&r,path,error,sizeof(error)));
    h3_decode_options wrong_backend={0};
#ifdef __APPLE__
    r.presentation.codec_version=2;
#else
    r.presentation.codec_version=1;
#endif
    CHECK(h3_result_save_av_state(&r,path,error,sizeof(error)));
    CHECK(!h3_decode_av_state("missing",path,&wrong_backend,error,sizeof(error)));
    CHECK(strstr(error,"saved decode recipe requires"));
    r.presentation=base;CHECK(h3_result_save_av_state(&r,path,error,sizeof(error)));
    state->video[0]=1;CHECK(h3_presentation_load(path,state,&read,error,sizeof(error))==-1);state->video[0]=0;
    r.presentation.trim_samples++;CHECK(!h3_result_save_av_state(&r,path,error,sizeof(error)));r.presentation.trim_samples--;
    FILE *f=fopen(meta,"ab");CHECK(f);fputs("trailing",f);fclose(f);CHECK(h3_presentation_load(path,state,&read,error,sizeof(error))==-1);
    CHECK(!unlink(meta));CHECK(h3_presentation_load(path,state,&read,error,sizeof(error))==-1);
    h3_decode_options opt={.output_path=video};CHECK(!h3_decode_av_state("missing",path,&opt,error,sizeof(error)));CHECK(strstr(error,"presentation"));
    opt.output_path=path;CHECK(!h3_decode_av_state("missing",path,&opt,error,sizeof(error)));CHECK(strstr(error,"overwrite"));
    uint8_t rgb[32*32*3]={0};float pcm[3200*2]={0};
    h3_ffmpeg_writer *w=h3_ffmpeg_writer_open(video,2,32,32,24,pcm,3200,2,32000,NULL,error,sizeof(error));CHECK(w);
    CHECK(h3_ffmpeg_writer_write(w,rgb,1,error,sizeof(error)));CHECK(h3_ffmpeg_writer_write(w,rgb,1,error,sizeof(error)));
    CHECK(h3_ffmpeg_writer_finish(w,error,sizeof(error)));CHECK(!access(video,R_OK));
    w=h3_ffmpeg_writer_open(video,2,32,32,24,NULL,0,0,0,NULL,error,sizeof(error));CHECK(w);
    CHECK(!h3_ffmpeg_writer_finish(w,error,sizeof(error)));CHECK(!access(video,R_OK));
    w=h3_ffmpeg_writer_open(video,1,32,32,24,NULL,0,0,0,NULL,error,sizeof(error));CHECK(w);h3_ffmpeg_writer_abort(w);
    /* Force an audio write larger than the pipe capacity while FFmpeg waits
     * for the first video frame. Abort must survive positive partial writes
     * that also generate SIGPIPE, and must join/reap every writer/child. */
    float *large_pcm=calloc(65536,sizeof(float));CHECK(large_pcm);
    for(int attempt=0;attempt<8;attempt++){
        w=h3_ffmpeg_writer_open(video,20,32,32,24,large_pcm,32768,2,32000,NULL,error,sizeof(error));CHECK(w);
        usleep(10000);h3_ffmpeg_writer_abort(w);
    }
    free(large_pcm);
    CHECK(!setenv("H3_FFMPEG","/usr/bin/false",1));
    for(int ignore=0;ignore<3;ignore++){
        signal(SIGPIPE,ignore==1?SIG_IGN:SIG_DFL);
        sigset_t set,old;sigemptyset(&set);sigaddset(&set,SIGPIPE);
        if(ignore==2)CHECK(!sigprocmask(SIG_BLOCK,&set,&old));
        w=h3_ffmpeg_writer_open(video,20,32,32,24,pcm,3200,2,32000,NULL,error,sizeof(error));CHECK(w);
        int ok=1;for(int i=0;i<20&&ok;i++)ok=h3_ffmpeg_writer_write(w,rgb,1,error,sizeof(error));
        if(ok)CHECK(!h3_ffmpeg_writer_finish(w,error,sizeof(error)));else h3_ffmpeg_writer_abort(w);
        if(ignore==2){sigset_t pending;CHECK(!sigpending(&pending));CHECK(!sigismember(&pending,SIGPIPE));CHECK(!sigprocmask(SIG_SETMASK,&old,NULL));}
    }
    signal(SIGPIPE,SIG_DFL);unsetenv("H3_FFMPEG");
    h3_av_state_free(state);CHECK(!unlink(path));CHECK(!unlink(video));CHECK(!rmdir(directory));
    puts("ok: defaults, temporal contract, presentation, backend dispatch, streaming audio, incomplete output, abort and SIGPIPE");
}
static void gpu(const char *weights,int pattern) {
    h3_tiny_vae *d=h3_tiny_vae_load(weights,error,sizeof(error));CHECK(d);float z[24*27*6*6];
    CHECK(h3_tiny_vae_cache_matches(d,h3_tiny_vae_digest(d)));
#ifndef __APPLE__
    const char *original=getenv("H3_PREVIEW_CUDA_WORKSPACE_MB");char *saved=original?strdup(original):NULL;
    setenv("H3_PREVIEW_CUDA_WORKSPACE_MB",original&&!strcmp(original,"1")?"2":"1",1);
    CHECK(!h3_tiny_vae_cache_matches(d,h3_tiny_vae_digest(d)));
    if(saved){setenv("H3_PREVIEW_CUDA_WORKSPACE_MB",saved,1);free(saved);}else unsetenv("H3_PREVIEW_CUDA_WORKSPACE_MB");
    CHECK(h3_tiny_vae_cache_matches(d,h3_tiny_vae_digest(d)));
#endif
    for(size_t i=0;i<sizeof(z)/sizeof(*z);i++)z[i]=sinf((float)i);
#ifndef __APPLE__
    setenv("H3_TEST_TINY_CUDA_MAX_BYTES","1",1);sink failed={0};
    CHECK(!h3_tiny_vae_stream(d,z,7,2,2,5,-1,frames,&failed,NULL,NULL,error,sizeof(error)));
    CHECK(!failed.next&&strstr(error,"allocation budget"));unsetenv("H3_TEST_TINY_CUDA_MAX_BYTES");
#endif
    const int times[]={7,17,27};for(int n=0;n<3;n++)for(int batch=1;batch<=5;batch++){
        if(pattern==2)temporal_input(z,times[n],2,2);
        sink s={.pattern=pattern};CHECK(h3_tiny_vae_stream(d,z,times[n],2,2,batch,-1,frames,&s,NULL,NULL,error,sizeof(error)));
        CHECK(s.next==h3_tiny_vae_frames(times[n]));
    }
    for(int geometry=4;geometry<=6;geometry+=2){sink resized={.pattern=pattern};
        if(pattern==2)temporal_input(z,7,geometry,geometry);
        CHECK(h3_tiny_vae_stream(d,z,7,geometry,geometry,5,-1,frames,&resized,NULL,NULL,error,sizeof(error)));CHECK(resized.next==22);}
    if(pattern==2)temporal_input(z,7,2,2);
    sink s={.cancel=1};CHECK(!h3_tiny_vae_stream(d,z,7,2,2,3,-1,frames,&s,NULL,NULL,error,sizeof(error)));
    s=(sink){0};CHECK(!h3_tiny_vae_stream(d,z,7,2,2,3,-1,frames,&s,cancel_progress,NULL,error,sizeof(error)));
    h3_memory_set_test_query(low_memory,NULL);s=(sink){0};CHECK(!h3_tiny_vae_stream(d,z,7,2,2,5,-1,frames,&s,NULL,NULL,error,sizeof(error)));CHECK(!s.next);h3_memory_set_test_query(NULL,NULL);
    h3_video_frames a={0},b={0};CHECK(h3_tiny_vae_decode(d,z,7,2,2,-1,&a,NULL,NULL,error,sizeof(error)));
    CHECK(h3_tiny_vae_decode(d,z,7,2,2,21,&b,NULL,NULL,error,sizeof(error)));CHECK(!memcmp(a.rgb+21*32*32*3,b.rgb,32*32*3*4));
    sink again={.expected=a.rgb,.pattern=pattern};CHECK(h3_tiny_vae_stream(d,z,7,2,2,5,-1,frames,&again,NULL,NULL,error,sizeof(error)));
    z[0]=NAN;s=(sink){0};CHECK(!h3_tiny_vae_stream(d,z,7,2,2,5,-1,frames,&s,NULL,NULL,error,sizeof(error)));CHECK(!s.next);
    h3_video_frames_free(&a);h3_video_frames_free(&b);h3_tiny_vae_free(d);
    puts("ok: 7/17/27 latents x batches 1..5, global frame selection, pixel shuffle, reset, selected-frame context, cancellation, memory rejection, finite input");
}
static void replay(const char *weights,const char *state_path,const char *raw) {
    h3_av_state *state=h3_av_state_load(state_path,error,sizeof(error));CHECK(state);double begin=h3_av_now();
    h3_tiny_vae *tiny=NULL;h3_video_vae_decoder *full=NULL;
    if(!strcmp(weights,"full"))full=h3_video_vae_decoder_load("models/MiniMax-H3/Ref2VA/video_vae/source","src/metal/shaders.metal",state->info.latent_h,state->info.latent_w,NULL,NULL,error,sizeof(error));
    else
        tiny = h3_tiny_vae_load(weights, error, sizeof(error));
    CHECK(tiny || full);
    printf("load %.9f s\n",h3_av_now()-begin);
    for(int run=0;run<2;run++){
        sink s={0};if(!run&&strcmp(raw,"-")){s.out=fopen(raw,"wb");CHECK(s.out);}
        begin=h3_av_now();int ok=tiny?h3_tiny_vae_stream(tiny,state->video,state->info.video_t,state->info.latent_h,state->info.latent_w,5,-1,frames,&s,NULL,NULL,error,sizeof(error)):
            h3_video_vae_decoder_stream_progress(full,state->video,state->info.video_t,frames,&s,NULL,NULL,error,sizeof(error));
        CHECK(ok&&s.next==state->info.frames);if(s.out)CHECK(!fclose(s.out));
        printf("%s decode_sink %.9f s device_allocated_at_delivery_peak %llu\n",run?"warm":"cold",h3_av_now()-begin,(unsigned long long)s.peak);
    }
    if(full){h3_video_frames buffered={0};CHECK(h3_video_vae_decoder_decode(full,state->video,state->info.video_t,&buffered,error,sizeof(error)));
        sink s={.expected=buffered.rgb};CHECK(h3_video_vae_decoder_stream_progress(full,state->video,state->info.video_t,frames,&s,NULL,NULL,error,sizeof(error)));h3_video_frames_free(&buffered);
        puts("ok: full streamed and buffered frames exactly equal including all spatial/temporal seams and tail");}
    struct rusage usage;getrusage(RUSAGE_SELF,&usage);
#ifdef __APPLE__
    printf("max_rss %ld\n",usage.ru_maxrss);
#else
    printf("max_rss %llu\n",(unsigned long long)usage.ru_maxrss*1024);
#endif
    h3_tiny_vae_free(tiny);h3_video_vae_decoder_free(full);h3_av_state_free(state);
}
static void delivery_errors(const char *state) {
    char path[]="/tmp/h3-tiny-delivery-XXXXXX";int fd=mkstemp(path);CHECK(fd>=0);CHECK(write(fd,"preserve",8)==8);close(fd);
    h3_decode_options options={.output_path=path};
    for(int backend=0;backend<2;backend++)for(int kind=0;kind<3;kind++){
        int calls=0;options.preview_vae=backend==0;options.callback_opaque=&calls;
        options.on_frame=kind==0?cancel_frame:NULL;options.on_progress=kind==1?cancel_mux:NULL;
        unsetenv("H3_FFMPEG");if(kind==2)setenv("H3_FFMPEG","/usr/bin/false",1);
        CHECK(!h3_decode_av_state("models/MiniMax-H3",state,&options,error,sizeof(error)));
        CHECK(strstr(error,kind<2?"cancelled":"FFmpeg"));if(kind<2)CHECK(calls==1);
        FILE *f=fopen(path,"rb");CHECK(f);char data[9]={0};CHECK(fread(data,1,9,f)==8&&!memcmp(data,"preserve",8));fclose(f);
        int status;CHECK(waitpid(-1,&status,WNOHANG)==-1&&errno==ECHILD);
    }
    unsetenv("H3_FFMPEG");unlink(path);puts("ok: public decode cancellation during frames/mux and encoder failure preserve target and reap encoder children");
}
int main(int argc,char **argv){
    if(argc==2&&!strcmp(argv[1],"host"))host();
    else if(argc==3&&!strcmp(argv[1],"validate")){char digest[65];if(!h3_tiny_vae_validate(argv[2],digest,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}puts(digest);}
    else if(argc==4&&!strcmp(argv[1],"gpu"))gpu(argv[2],atoi(argv[3]));
    else if(argc==5&&!strcmp(argv[1],"replay"))replay(argv[2],argv[3],argv[4]);
    else if(argc==3&&!strcmp(argv[1],"delivery-errors"))delivery_errors(argv[2]);
    else
        return 2;
    return 0;
}
