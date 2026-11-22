/* Reused production context: video -> still -> video and recoverable failures. */
#include "src/h3.h"
#include "src/digest.h"
#include "src/memory.h"
#include <stdio.h>
#include <string.h>
#include <stdlib.h>
static const char *cancel_phase;
static int frames;
static h3_sha256_ctx pixels;
static int progress(const char *phase,int done,int total,void *opaque) {
    (void)total;(void)opaque;return cancel_phase && strstr(phase,cancel_phase) && done>=1;
}
static int frame(const h3_frame *f,void *opaque){(void)opaque;frames++;h3_sha256_update(&pixels,f->rgb,(h3_sha256_size)(f->height*f->stride));return 0;}
static int low(uint64_t *bytes,void *opaque){(void)opaque;*bytes=1;return 1;}
int main(int argc,char **argv){
    if(argc!=2)return 2;h3_ctx *ctx=h3_load_dir(argv[1]);if(!ctx)return 1;h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=256;p.height=256;p.frames=22;p.steps=2;p.seed=42;p.on_progress=progress;p.on_frame=frame;
    unsigned char video[32],still[32],got[32];
    for(int i=0;i<4;i++){
        p.still=i%2;p.frames=p.still?1:22;frames=0;h3_sha256_init(&pixels);
        h3_result *r=h3_generate(ctx,"A red teapot on a wooden table.",&p);
        if(!r||r->frames!=p.frames||frames!=p.frames||(p.still&&(!r->still_latent||r->av_state||r->audio_samples)))goto fail;
        h3_sha256_final(got,&pixels);h3_result_free(r);
        if(i<2)memcpy(i?still:video,got,32);else if(memcmp(i%2?still:video,got,32))goto fail;
    }
    p.still=1;p.frames=1;
    const char *phases[]={"load transformer","denoise","stock still VAE decode"};
    for(int i=0;i<3;i++){
        cancel_phase=phases[i];frames=0;h3_result *r=h3_generate(ctx,"A red teapot on a wooden table.",&p);
        if(r||frames){h3_result_free(r);goto fail;}cancel_phase=NULL;
        frames=0;h3_sha256_init(&pixels);r=h3_generate(ctx,"A red teapot on a wooden table.",&p);if(!r||frames!=1)goto fail;
        h3_sha256_final(got,&pixels);h3_result_free(r);if(memcmp(still,got,32))goto fail;
    }
    h3_memory_set_test_query(low,NULL);h3_result *r=h3_generate(ctx,"A red teapot on a wooden table.",&p);h3_memory_set_test_query(NULL,NULL);
    if(r){h3_result_free(r);goto fail;}frames=0;h3_sha256_init(&pixels);r=h3_generate(ctx,"A red teapot on a wooden table.",&p);
    if(!r||frames!=1)goto fail;h3_sha256_final(got,&pixels);h3_result_free(r);if(memcmp(still,got,32))goto fail;
    h3_free(ctx);puts("PASS video/still cache isolation, repeated identical pixels, load/sampling/tile cancellation, memory admission failure and clean retries");return 0;
fail:fprintf(stderr,"still context failed (%s): %s\n",cancel_phase?cancel_phase:"normal",h3_last_error(ctx));h3_free(ctx);return 1;
}
