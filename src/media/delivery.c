#include "src/media/delivery.h"
#include "src/sampling/av_state.h"
#include "src/media/ffmpeg.h"
#include "src/memory.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

const char *h3_preview_vae_path(const char *path) {
    return path ? path : "models/preview-vae/taeh3.safetensors";
}
int h3_preview_vae_options(int enabled,const char *path,char *error,size_t size) {
    if((enabled!=0&&enabled!=1)||(!enabled&&path)){
        snprintf(error,size,"preview-vae-model requires --preview-vae (boolean selection)");return 0;
    }
    if(!enabled)return 1;
    char digest[65];return h3_tiny_vae_validate(h3_preview_vae_path(path),digest,error,size);
}

typedef struct {
    const h3_presentation *p;
    const h3_decode_options *options;
    h3_ffmpeg_writer *writer;
    int next, total, failed, tiny;
    char *error;size_t size;
    double start, first, conversion, write;
} delivery;
static int progress(int done,int total,void *opaque) {
    delivery *d=opaque;
    if(d->failed)return 1;
    if(!h3_memory_check(0,"video delivery",d->error,d->size) ||
       (d->options->on_progress&&d->options->on_progress(d->tiny?"tiny video VAE":"video VAE decode",done,total,d->options->callback_opaque))){
        if (!d->error[0])
            snprintf(d->error, d->size, "video delivery cancelled");
        d->failed = 1;
        return 1;
    }return 0;
}
static int frames(const float *rgb,int first,int count,int width,int height,void *opaque) {
    delivery *d=opaque;const h3_presentation *p=d->p;
    int skip=p->trim_frames-first;if(skip<0)skip=0;if(skip>=count)return 0;
    first+=skip;count-=skip;
    if(first-p->trim_frames!=d->next || width!=p->render_width || height!=p->render_height){
        snprintf(d->error,d->size,"invalid decoder frame ordering/geometry");return d->failed=1;
    }
    double begin=h3_av_now();size_t frame_bytes=(size_t)width*(size_t)height*3;
    rgb+=(size_t)skip*frame_bytes;size_t n=(size_t)count*frame_bytes;
    uint8_t *bytes=malloc(n);if(!bytes){snprintf(d->error,d->size,"cannot allocate bounded RGB batch");return d->failed=1;}
    for(size_t i=0;i<n;i++){
        if(!isfinite(rgb[i])){free(bytes);snprintf(d->error,d->size,"non-finite decoded RGB");return d->failed=1;}
        float scaled=fminf(1,fmaxf(0,rgb[i]))*255;
        bytes[i]=p->codec_version==2?(uint8_t)scaled:(uint8_t)lrintf(scaled);
    }
    if(width!=p->width||height!=p->height){uint8_t *resized=NULL;
        if(!h3_resize_rgb24_high_quality(bytes,count,width,height,p->width,p->height,&resized)){
            free(bytes);snprintf(d->error,d->size,"cannot resize decoded batch");return d->failed=1;
        }free(bytes);bytes=resized;
    }
    d->conversion+=h3_av_now()-begin;if(!d->next)d->first=h3_av_now()-d->start;
    frame_bytes=(size_t)p->width*(size_t)p->height*3;
    for(int i=0;i<count;i++)if(d->options->on_frame){
        h3_frame f={p->width,p->height,p->width*3,bytes+(size_t)i*frame_bytes,d->next+i,d->total,-1,0};
        if(d->options->on_frame(&f,d->options->callback_opaque)){
            free(bytes);snprintf(d->error,d->size,"frame delivery cancelled");return d->failed=1;
        }
    }
    begin=h3_av_now();
    if(d->writer&&!h3_ffmpeg_writer_write(d->writer,bytes,count,d->error,d->size))d->failed=1;
    d->write+=h3_av_now()-begin;free(bytes);d->next+=count;return d->failed;
}
int h3_deliver_video(h3_tiny_vae *tiny,h3_video_vae_decoder *full,
 const float *latent,int time,int height,int width,const h3_presentation *p,
 h3_audio_waveform *audio,const h3_decode_options *options,char *error,size_t size) {
    if(error&&size)error[0]=0;
    int total=h3_tiny_vae_frames(time)-p->trim_frames;
    if((!tiny==!full)||!options||total<1||p->trim_samples<0||
       (audio->pcm&&audio->samples<=p->trim_samples)){
        snprintf(error,size,"invalid delivery geometry or audio prefix");return 0;
    }
    if(audio->pcm&&p->trim_samples){int remain=audio->samples-p->trim_samples;
        for(int c=0;c<audio->channels;c++)memmove(audio->pcm+(size_t)c*(size_t)remain,
            audio->pcm+(size_t)c*(size_t)audio->samples+(size_t)p->trim_samples,(size_t)remain*sizeof(float));
        audio->samples=remain;
    }
    delivery d={.p=p,.options=options,.total=total,.tiny=tiny!=NULL,.error=error,.size=size,.start=h3_av_now()};
    if(options->output_path&&*options->output_path){
        d.writer=h3_ffmpeg_writer_open(options->output_path,total,p->width,p->height,p->fps,
            audio->pcm,audio->samples,audio->channels,audio->sample_rate,&options->output_encoding,error,size);
        if(!d.writer)return 0;
    }
    char decoder_error[512]={0};
    int ok=tiny?h3_tiny_vae_stream(tiny,latent,time,height,width,5,-1,frames,&d,progress,&d,decoder_error,sizeof(decoder_error)):
        h3_video_vae_decoder_stream_progress(full,latent,time,frames,&d,progress,&d,decoder_error,sizeof(decoder_error));
    if(!ok||d.failed||d.next!=total){
        if(!error[0])snprintf(error,size,"%s",decoder_error[0]?decoder_error:"incomplete decoder output");
        h3_ffmpeg_writer_abort(d.writer);return 0;
    }
    double finish=h3_av_now();
    if(d.writer&&options->on_progress&&options->on_progress("FFmpeg",0,1,options->callback_opaque)){
        snprintf(error,size,"media finalization cancelled");h3_ffmpeg_writer_abort(d.writer);return 0;
    }
    if(d.writer&&!h3_ffmpeg_writer_finish(d.writer,error,size))return 0;
    if(d.writer&&options->on_progress)options->on_progress("FFmpeg",1,1,options->callback_opaque);
    if(getenv("H3_PROFILE"))fprintf(stderr,"h3cli: %s delivery: first_frame=%.6f decode_compute_transfer=%.6f conversion=%.6f pipe=%.6f mux_finish=%.6f total=%.6f s\n",
        tiny?"tiny":"full",d.first,finish-d.start-d.conversion-d.write,d.conversion,d.write,h3_av_now()-finish,h3_av_now()-d.start);
    return 1;
}
