/* Released-weight progress regression: multiple tiles/chunks, cache, cancellation.
 * Build with VAE_PROGRESS_BEFORE and the old archive for an independent RGB baseline. */
#include "src/vae/video_vae.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static char error[512];
#define CHECK(x) do { if (!(x)) { fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error); exit(1); } } while (0)
typedef struct { int expected, last, done, calls, cancel_at; } trace;
typedef struct { trace load, decode; } phases;
static int update(trace *t, int completed, int total) {
    CHECK(total == t->expected && completed >= t->last && completed <= total);
    if (completed == total) t->done++;
    t->last = completed; t->calls++;
    return t->cancel_at > 0 && completed >= t->cancel_at;
}
static int loading(int completed, int total, void *opaque) {
    return update(&((phases *)opaque)->load,completed,total);
}
static int decoding(int completed, int total, void *opaque) {
    return update(&((phases *)opaque)->decode,completed,total);
}
static void save(const char *root, const char *name, const h3_video_frames *frames) {
    char path[4096]; CHECK(snprintf(path,sizeof(path),"%s/%s.f32",root,name)<(int)sizeof(path));
    FILE *f=fopen(path,"wb");CHECK(f);
    size_t n=(size_t)frames->frames*(size_t)frames->height*(size_t)frames->width*3;
    CHECK(fwrite(frames->rgb,sizeof(float),n,f)==n);CHECK(!fclose(f));
}
int main(int argc,char **argv) {
    CHECK(argc==2);
    const char *weights="models/MiniMax-H3/FL2VA/video_vae/source";
    enum { T=17,H=2,W=20 }; /* 56 frames, 32x320, two spatial tiles. */
    float video[24*T*H*W];
    for(size_t i=0;i<sizeof(video)/sizeof(*video);i++) video[i]=(float)((int)(i%37)-18)/100;
    h3_video_frames batch={0},resident={0},preview={0};
    phases p={.load={.expected=36},.decode={.expected=216}};
#ifdef VAE_PROGRESS_BEFORE
    CHECK(h3_video_vae_decode(weights,"src/metal/shaders.metal",video,T,H,W,NULL,NULL,&batch,error,sizeof(error)));
#else
    CHECK(h3_video_vae_decode_phased(weights,"src/metal/shaders.metal",video,T,H,W,loading,decoding,&p,&batch,error,sizeof(error)));
    CHECK(p.load.done==1 && p.decode.done==1 && p.decode.last==216);
#endif
    save(argv[1],"batch",&batch);
    p=(phases){.load={.expected=36},.decode={.expected=216}};
    h3_video_vae_decoder *decoder=h3_video_vae_decoder_load(weights,"src/metal/shaders.metal",H,W,
        loading,&p,error,sizeof(error));CHECK(decoder);CHECK(p.load.done==1);
#ifdef VAE_PROGRESS_BEFORE
    CHECK(h3_video_vae_decoder_decode(decoder,video,T,&resident,error,sizeof(error)));
#else
    CHECK(h3_video_vae_decoder_decode_progress(decoder,video,T,decoding,&p,&resident,error,sizeof(error)));
    CHECK(p.decode.done==1 && p.decode.last==216);
#endif
    CHECK(batch.frames==resident.frames && batch.width==resident.width && batch.height==resident.height);
    CHECK(!memcmp(batch.rgb,resident.rgb,(size_t)batch.frames*H*16*W*16*3*sizeof(float)));
    save(argv[1],"resident",&resident);
    h3_video_frames_free(&batch);h3_video_frames_free(&resident);
    int index=-1;
    p.decode=(trace){.expected=72};
#ifdef VAE_PROGRESS_BEFORE
    CHECK(h3_video_vae_decoder_preview(decoder,video,T,&preview,&index,error,sizeof(error)));
#else
    CHECK(h3_video_vae_decoder_preview_progress(decoder,video,T,decoding,&p,&preview,&index,error,sizeof(error)));
    CHECK(p.decode.done==1 && p.decode.last==72);
#endif
    CHECK(preview.frames==1 && index==28);save(argv[1],"preview",&preview);h3_video_frames_free(&preview);
#ifndef VAE_PROGRESS_BEFORE
    /* Cancel within the second tile, and again at final delivery. Reuse after
     * each cancellation ensures completed/cancelled calls own no stale state. */
    const int stops[]={37,216};
    for(size_t i=0;i<sizeof(stops)/sizeof(*stops);i++) {
        p.decode=(trace){.expected=216,.cancel_at=stops[i]};
        CHECK(!h3_video_vae_decoder_decode_progress(decoder,video,T,decoding,&p,&resident,error,sizeof(error)));
        CHECK(strstr(error,"cancelled") && !resident.rgb && p.decode.last==stops[i]);
        p.decode=(trace){.expected=72};
        CHECK(h3_video_vae_decoder_preview_progress(decoder,video,T,decoding,&p,&preview,&index,error,sizeof(error)));
        CHECK(p.decode.done==1);h3_video_frames_free(&preview);
    }
    p.decode=(trace){.expected=72,.cancel_at=72};
    CHECK(!h3_video_vae_decoder_preview_progress(decoder,video,T,decoding,&p,&preview,&index,error,sizeof(error)));
    CHECK(!preview.rgb && strstr(error,"cancelled"));
#endif
    h3_video_vae_decoder_free(decoder);
    /* Exercise the separate, nonresident single-tile streaming path too. */
    p=(phases){.load={.expected=36},.decode={.expected=36}};
#ifdef VAE_PROGRESS_BEFORE
    CHECK(h3_video_vae_decode(weights,"src/metal/shaders.metal",video,7,2,2,NULL,NULL,&batch,error,sizeof(error)));
#else
    CHECK(h3_video_vae_decode_phased(weights,"src/metal/shaders.metal",video,7,2,2,loading,decoding,&p,&batch,error,sizeof(error)));
    CHECK(!p.load.calls && p.decode.done==1);
#endif
    save(argv[1],"streaming",&batch);h3_video_frames_free(&batch);
    puts("PASS VAE multi-tile/chunk, resident, preview, streaming and cancellation progress");
    return 0;
}
