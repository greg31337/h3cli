#include "src/vae/video_encoder.h"
#include "src/weights/safetensors.h"
#include "src/host.h"
#include "src/denoise/dit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static void die(const char *s) { fprintf(stderr, "FAIL: %s\n", s); exit(1); }

static void write_values(const char *path, const float *values, size_t count) {
    FILE *f = fopen(path, "wb");
    if (!f || fwrite(values, 4, count, f) != count || fclose(f)) die("write failed");
}

#ifndef H3_ENCODER_LEGACY_ONLY
static void check_axis(h3_st_header *fixture, int extent, const char *prefix) {
    char error[512], name[64]; h3_video_tile_axis axis;
    if (!h3_video_tile_axis_build(extent, &axis, error, sizeof(error))) die(error);
    const char *suffix[] = {"starts", "overlaps"};
    for (int i = 0; i < 2; i++) {
        snprintf(name, sizeof(name), "x.%s_%s", prefix, suffix[i]);
        const h3_st_tensor *t = h3_st_find(fixture, name);
        size_t n = (size_t)(axis.count - i);
        int *want = malloc(n*4);
        if (!t || h3_st_tensor_elements(t) != n || !want ||
            !h3_st_read_data(fixture, t, want, n*4, error, sizeof(error)) ||
            memcmp(want, i ? axis.overlaps : axis.starts, n*4)) die("spatial tile plan differs from oracle");
        free(want);
    }
    h3_video_tile_axis_free(&axis);
}

static void stitch(h3_st_header *fixture, const char *path) {
    char error[512]; int canvas[2];
    const h3_st_tensor *c = h3_st_find(fixture, "x.canvas"), *t = h3_st_find(fixture, "x.tiles");
    if (!c || !t || t->ndim != 6 || t->shape[1] != 1 || t->shape[2] != 48 ||
        !h3_st_read_data(fixture, c, canvas, sizeof(canvas), error, sizeof(error))) die("invalid stitch fixture");
    check_axis(fixture, canvas[0], "y"); check_axis(fixture, canvas[1], "x");
    size_t n = h3_st_tensor_elements(t), count = (size_t)t->shape[0];
    float *values = malloc(n*4); float **tiles = malloc(count*sizeof(*tiles));
    if (!values || !tiles || !h3_st_read_data(fixture, t, values, n*4, error, sizeof(error))) die("cannot load tiles");
    for (size_t i = 0; i < count; i++) tiles[i] = values + i*(n/count);
    h3_video_moments output;
    if (!h3_video_moments_stitch(tiles, (int)t->shape[3], canvas[0], canvas[1], &output, error, sizeof(error))) die(error);
    write_values(path, output.values, (size_t)48*output.time*output.height*output.width);
    printf("{\"channels\":48,\"time\":%d,\"height\":%d,\"width\":%d}\n", output.time, output.height, output.width);
    h3_video_moments_free(&output); free(values); free(tiles);
}

typedef struct { int completed, total; } progress_state;
static int progress(int completed, int total, void *opaque) {
    progress_state *state = opaque;
    if (completed != state->completed + 1 || total < completed || (state->total && state->total != total))
        die("invalid chunk/tile progress");
    state->completed = completed; state->total = total;
    return 0;
}
#endif

int main(int argc, char **argv) {
    if (argc != 5) die("usage: refvideo_encoder MODE WEIGHTS FIXTURE OUTPUT");
    char error[512]; h3_st_header fixture;
    if (!h3_st_read_header(argv[3], &fixture, error, sizeof(error))) die(error);
#ifndef H3_ENCODER_LEGACY_ONLY
    if (!strcmp(argv[1], "stitch")) { stitch(&fixture, argv[4]); h3_st_free_header(&fixture); return 0; }
#endif
    const h3_st_tensor *p = h3_st_find(&fixture, "x.pixels");
    if (!p || p->ndim != 5 || p->dtype != H3_DTYPE_F32 || p->shape[0] != 1 || p->shape[1] != 3)
        die("invalid pixel fixture");
    size_t count = h3_st_tensor_elements(p);
    float *pixels = malloc(count * sizeof(float));
    if (!pixels || !h3_st_read_data(&fixture, p, pixels, count*4, error, sizeof(error))) die(error);
    int frames=(int)p->shape[2], height=(int)p->shape[3], width=(int)p->shape[4];
    h3_video_latent latent = {0};
    float *values = NULL; size_t output_count = 0;
    if (!strcmp(argv[1], "legacy")) {
        if (!h3_video_vae_encode(argv[2], "src/metal/shaders.metal", pixels, frames, height, width,
            NULL, NULL, &latent, error, sizeof(error))) die(error);
        values = latent.values; output_count = (size_t)24*latent.time*latent.height*latent.width;
        printf("{\"channels\":24,\"time\":%d,\"height\":%d,\"width\":%d}\n", latent.time, latent.height, latent.width);
    }
#ifndef H3_ENCODER_LEGACY_ONLY
    else if (!strcmp(argv[1], "raw")) {
        h3_video_moments moments = {0};
        if (!h3_video_vae_encode_moments(argv[2], "src/metal/shaders.metal", pixels, frames, height, width,
            NULL, NULL, &moments, error, sizeof(error))) die(error);
        values = moments.values; output_count = (size_t)48*moments.time*moments.height*moments.width;
        printf("{\"channels\":48,\"time\":%d,\"height\":%d,\"width\":%d}\n", moments.time, moments.height, moments.width);
    }
    else if (!strcmp(argv[1], "temporal") || !strcmp(argv[1], "released")) {
        int selected;
        const h3_st_tensor *v = h3_st_find(&fixture, "x.vae_frames");
        if (!v || !h3_st_read_data(&fixture, v, &selected, 4, error, sizeof(error))) die("missing VAE frame count");
        float *before = malloc(count*4); if (!before) die("allocation failed"); memcpy(before, pixels, count*4);
        progress_state state = {0}; h3_gpu_stats stats;
        h3_rng request;
        const char *seed = getenv("H3_TEST_REQUEST_SEED");
        h3_rng_seed(&request, seed ? strtoull(seed, NULL, 10) : 0);
        for (int i = 0; i < 1000; i++) (void)h3_rng_normal(&request);
        if (!strcmp(argv[1], "temporal")) {
            h3_video_moments moments = {0};
            if (!h3_ref2va_video_vae_moments(argv[2], "src/metal/shaders.metal", pixels, frames, selected, height, width,
                progress, &state, &moments, error, sizeof(error))) die(error);
            values = moments.values; output_count = (size_t)48*moments.time*moments.height*moments.width;
            stats = moments.gpu_stats;
            printf("{\"channels\":48,\"time\":%d,\"height\":%d,\"width\":%d}\n", moments.time, moments.height, moments.width);
        } else {
            if (!h3_ref2va_video_vae_encode(argv[2], "src/metal/shaders.metal", pixels, frames, selected, height, width,
                progress, &state, &latent, error, sizeof(error))) die(error);
            values = latent.values; output_count = (size_t)24*latent.time*latent.height*latent.width;
            stats = latent.gpu_stats;
            printf("{\"channels\":24,\"time\":%d,\"height\":%d,\"width\":%d}\n", latent.time, latent.height, latent.width);
        }
        if (!state.total || state.completed != state.total || stats.mps_conv_dispatches != (uint64_t)34*state.total ||
            stats.submissions != (uint64_t)18*state.total) die("invalid released encoder dispatch totals");
        if (memcmp(before, pixels, count*4)) die("encoder mutated normalized RGB input");
        free(before);
    }
#endif
    else die("unknown encoder mode");
    write_values(argv[4], values, output_count);
#ifndef H3_ENCODER_LEGACY_ONLY
    if (!strcmp(argv[1], "released")) {
        float *rows = malloc(output_count*4);
        char path[4096];
        if (!rows || !h3_dit_patchify_video(values,24,latent.time,latent.height,latent.width,
                rows,output_count)) die("cannot patchify released latent");
        if (snprintf(path,sizeof(path),"%s.rows",argv[4]) >= (int)sizeof(path)) die("output path too long");
        write_values(path,rows,output_count); free(rows);
    }
#endif
    free(values); free(pixels); h3_st_free_header(&fixture);
    return 0;
}
