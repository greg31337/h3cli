/* Test actual private decoder policy/geometry without a model or GPU. */
#include "../src/vae/video_vae.c"
#include "src/vae/video_encoder.h"
#include "src/host.h"
static int checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL line %d: %s\n",__LINE__,#x); exit(1); } } while(0)
static int configured_tile_pixels(int h, int w) { return h3_video_vae_tile_pixels(h,w,NULL,0); }
typedef struct { int total, last, finishes, cancel_at; } progress_trace;
static int check_progress(int completed, int total, void *opaque) {
    progress_trace *t = opaque;
    CHECK(total == t->total && completed >= t->last && completed <= total);
    t->last = completed;
    t->finishes += completed == total;
    return t->cancel_at > 0 && completed >= t->cancel_at;
}
static void test_decode_progress(void) {
    const int cases[][3] = {{1,1,1}, {3,1,1}, {3,2,2}, {21,3,3}, {1,3,3}};
    char error[512];
    for (size_t c = 0; c < sizeof(cases)/sizeof(*cases); c++) {
        int tiles = cases[c][0] * cases[c][1] * cases[c][2];
        progress_trace t = {.total=tiles*36, .last=-1};
        decode_progress work;
        CHECK(decode_progress_init(&work, check_progress, &t,
            cases[c][0], cases[c][1], cases[c][2], error, sizeof(error)));
        for (int tile = 0; tile < tiles; tile++) {
            for (int block = 0; block <= 36; block++)
                CHECK(!decode_progress_block(block,36,&work));
            work.offset += 36;
        }
        CHECK(!t.finishes && t.last == t.total-1);
        CHECK(decode_progress_finish(&work,error,sizeof(error)));
        CHECK(t.finishes == 1 && t.last == t.total);
    }
    decode_progress work;
    CHECK(!decode_progress_init(&work,NULL,NULL,INT32_MAX,2,2,error,sizeof(error)));
    CHECK(!decode_progress_init(&work,NULL,NULL,1,0,1,error,sizeof(error)));
    progress_trace t = {.total=72, .last=-1, .cancel_at=37};
    CHECK(decode_progress_init(&work,check_progress,&t,1,1,2,error,sizeof(error)));
    work.offset = 36;
    CHECK(decode_progress_block(1,36,&work)); /* Still cancel within later tiles. */
    CHECK(!t.finishes);
    t.cancel_at = 72;
    CHECK(!decode_progress_finish(&work,error,sizeof(error)));
    CHECK(strstr(error,"cancelled"));
}
int main(void) {
    test_decode_progress();
    /* Captured from unmodified 59b64c9 before editing (baseline/policy.txt). */
    const int matrix[][3]={{320,320,320},{512,512,288},{480,864,272},
        {576,1024,320},{768,768,304},{768,1024,304},{768,1344,320},
        {256,256,256},{272,272,272},{288,288,288},{304,304,304}};
    unsetenv("H3_PROFILE"); unsetenv("H3_VAE_TILE_PIXELS");
    for(int h=16;h<=2048;h+=16) for(int w=16;w<=2048;w+=16)
        CHECK(configured_tile_pixels(h,w)==256);
    /* Exhaust every canvas accepted by h3_valid_params, including extremes. */
    for(int h=32;h<=H3_MAX_PIXELS/32;h+=32)
        for(int w=32;w<=H3_MAX_PIXELS/h;w+=32)
            CHECK(configured_tile_pixels(h,w)==256);
    for(size_t i=0;i<sizeof(matrix)/sizeof(*matrix);i++) {
        int h=matrix[i][0],w=matrix[i][1],old=matrix[i][2];
        setenv("H3_VAE_TILE_PIXELS","auto",1);
        CHECK(configured_tile_pixels(h,w)==old); CHECK(configured_tile_pixels(w,h)==old);
        for(int tile=256;tile<=320;tile+=16) {
            char value[16];snprintf(value,sizeof(value),"%d",tile);
            setenv("H3_VAE_TILE_PIXELS",value,1); CHECK(configured_tile_pixels(h,w)==tile);
        }
        const char *invalid[]={"","no","AUTO","320px","256.0","255","240","336","384","512","1088","513","528","257","-256","0","256 ","99999999999999999999999999999999999999999999"};
        for(size_t j=0;j<sizeof(invalid)/sizeof(*invalid);j++) {
            setenv("H3_VAE_TILE_PIXELS",invalid[j],1); CHECK(configured_tile_pixels(h,w)==0);
            char error[512];
            CHECK(!h3_video_vae_decoder_load("/no-weights", "missing.metal", 34, 60, NULL, NULL, error, sizeof(error)));
            float latent=0;h3_video_frames frames;
            CHECK(!h3_video_vae_decode("/no-weights", "missing.metal", &latent, 7, 34, 60, NULL, NULL, &frames, error, sizeof(error)));
            CHECK(strstr(error,"H3_VAE_TILE_PIXELS=") && strstr(error,"256..320") && strstr(error,"default is 256"));
        }
    }
    setenv("H3_VAE_TILE_PIXELS","auto",1);
    for(int h=32;h<=H3_MAX_PIXELS/32;h+=32)
        for(int w=32;w<=H3_MAX_PIXELS/h;w+=32) {
            int tile=configured_tile_pixels(h,w);
            CHECK(tile>=256 && tile<=320 && tile%16==0);
        }
    setenv("H3_VAE_TILE_PIXELS"," +0320",1); CHECK(configured_tile_pixels(768,768)==320);
    char error[512];
    for(int extent=16;extent<=2048;extent+=16) {
        tile_axis a; h3_video_tile_axis encoder;
        CHECK(tile_axis_build(extent,256,&a,error,sizeof(error)));
        CHECK(h3_video_tile_axis_build(extent,&encoder,error,sizeof(error)));
        CHECK(a.count==encoder.count && a.length==encoder.length);
        CHECK(!memcmp(a.starts,encoder.starts,(size_t)a.count*sizeof(int)));
        CHECK(a.starts[a.count-1]+a.length==extent);
        for(int i=0;i<a.count-1;i++) {
            CHECK(a.overlaps[i]>=64 && a.overlaps[i]%16==0);
            CHECK(a.overlaps[i]==encoder.overlaps[i]);
        }
        tile_axis_free(&a);h3_video_tile_axis_free(&encoder);
    }
    unsetenv("H3_VAE_TILE_PIXELS"); printf("ok: %d tile policy/overlap checks\n",checks);return 0;
}
