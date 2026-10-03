#include "src/vae/video_posterior.h"
#include "src/weights/safetensors.h"
#include "src/host.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static size_t checks;
static char error[512];
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr, "FAIL %s:%d: %s (%s)\n", __FILE__, __LINE__, #x, error); exit(1); } } while (0)

static float *tensor(h3_st_header *f, const char *name, size_t *count) {
    const h3_st_tensor *t = h3_st_find(f, name); CHECK(t && t->dtype == H3_DTYPE_F32);
    *count = h3_st_tensor_elements(t); float *data = malloc(*count * 4); CHECK(data);
    CHECK(h3_st_read_data(f, t, data, *count * 4, error, sizeof(error))); return data;
}

static void half_roundtrip(void) {
    for (unsigned bits = 0; bits < 65536; bits++) {
        float x = h3_video_f16_to_f32((uint16_t)bits);
        if ((bits & 0x7c00) == 0x7c00 && (bits & 1023)) { CHECK(isnan(x)); continue; }
        CHECK(h3_video_f32_to_f16(x) == bits);
        if ((bits & 0x7fff) < 0x7bff) {
            float next = h3_video_f16_to_f32((uint16_t)(bits + 1));
            float middle = (x + next) * .5f;
            CHECK(h3_video_f32_to_f16(middle) == ((bits & 1) ? bits + 1 : bits));
        }
    }
    CHECK(h3_video_f32_to_f16(NAN) == 0x7e00);
}

static void temporal(void) {
    const int frames[] = {39, 56, 73, 107, 124};
    const int times[] = {12, 17, 22, 32, 37};
    for (int f = 0; f < 5; f++) {
        h3_video_chunk_plan plan;
        CHECK(h3_video_chunk_plan_build(frames[f], &plan));
        CHECK(plan.padding_frames == 12 && plan.padded_frames == frames[f] + 12);
        CHECK(plan.chunks == (frames[f]+12)/17 && plan.raw_time == times[f]+3 && plan.time == times[f]);
        int source_frames = frames[f] + 4;
        size_t n = (size_t)3 * source_frames * 6;
        float *pixels = malloc(n*4), *before = malloc(n*4); CHECK(pixels && before);
        for (size_t i = 0; i < n; i++) pixels[i] = (float)i;
        memcpy(before, pixels, n*4);
        h3_video_moments *chunks = calloc((size_t)plan.chunks, sizeof(*chunks)); CHECK(chunks);
        for (int k = 0; k < plan.chunks; k++) {
            float *chunk = h3_video_chunk_extract(pixels, source_frames, 2, 3, &plan, k); CHECK(chunk);
            for (int c = 0; c < 3; c++) for (int t = 0; t < 17; t++) for (int p = 0; p < 6; p++) {
                int from = k * 17 + t; if (from >= frames[f]) from = frames[f] - 1;
                CHECK(chunk[(c*17+t)*6+p] == pixels[(c*source_frames+from)*6+p]);
            }
            free(chunk);
            chunks[k].time = 5; chunks[k].height = 2; chunks[k].width = 3;
            chunks[k].values = malloc(48*5*6*4); CHECK(chunks[k].values);
            for (int c = 0; c < 48; c++) for (int t = 0; t < 5; t++) for (int p = 0; p < 6; p++)
                chunks[k].values[(c*5+t)*6+p] = (float)(c*10000+(k*5+t)*10+p);
        }
        CHECK(!memcmp(before, pixels, n*4));
        CHECK(!h3_video_chunk_extract(pixels, source_frames, 2, 3, &plan, plan.chunks));
        h3_video_moments joined = {0};
        CHECK(h3_video_moments_join(chunks, plan.chunks, frames[f], &joined, error, sizeof(error)));
        CHECK(joined.time == times[f] && joined.height == 2 && joined.width == 3);
        for (int c = 0; c < 48; c++) for (int t = 0; t < times[f]; t++) for (int p = 0; p < 6; p++)
            CHECK(joined.values[(c*times[f]+t)*6+p] == (float)(c*10000+t*10+p));
        h3_video_moments_free(&joined);
        chunks[0].time = 4;
        CHECK(!h3_video_moments_join(chunks, plan.chunks, frames[f], &joined, error, sizeof(error)));
        CHECK(!joined.values);
        for (int k = 0; k < plan.chunks; k++) h3_video_moments_free(&chunks[k]);
        free(chunks); free(pixels); free(before);
    }
    h3_video_chunk_plan plan;
    for (int frames_n = 5; frames_n <= 360; frames_n++)
        CHECK(h3_video_chunk_plan_build(frames_n, &plan) == ((frames_n-5)%17 == 0));
    CHECK(!h3_video_chunk_plan_build(361, &plan));
    h3_video_moments output;
    float dummy = 0;
    CHECK(!h3_ref2va_video_vae_moments("unused", "unused", &dummy, 60, 60, 64, 64, NULL, NULL, &output, error, sizeof(error)));
    CHECK(!output.values);
    h3_video_latent latent;
    CHECK(!h3_ref2va_video_vae_encode("unused", "unused", &dummy, 55, 56, 64, 64, NULL, NULL, &latent, error, sizeof(error)));
    CHECK(!latent.values);
}

static void oracle(const char *path) {
    h3_st_header f; CHECK(h3_st_read_header(path, &f, error, sizeof(error)));
    const size_t sizes[] = {1,2,7,15,16,17,24,96,216,1920,5952,131071};
    double rng_max = 0;
    for (size_t j = 0; j < sizeof(sizes)/sizeof(*sizes); j++) {
        char name[64]; snprintf(name, sizeof(name), "epsilon.%zu", sizes[j]);
        if (!h3_st_find(&f, name)) { CHECK(sizes[j] == 131071); continue; }
        size_t n; float *want = tensor(&f, name, &n), *got = malloc(n*4), *repeat = malloc(n*4); CHECK(got && repeat);
        CHECK(n == sizes[j] && h3_video_posterior_epsilon(got, n));
        for (size_t i = 0; i < n; i++) {
            double d = fabs((double)got[i]-want[i]); if (d > rng_max) rng_max = d;
            CHECK(d <= 1e-6); /* Scalar libm vs CPU-vector elementary functions. */
        }
        for (uint64_t seed = 0; seed < 3; seed++) {
            h3_rng request; h3_rng_seed(&request, seed); (void)h3_rng_normal(&request);
            CHECK(h3_video_posterior_epsilon(repeat, n) && !memcmp(got, repeat, n*4));
        }
        if (n == 96) {
            h3_rng original; h3_rng_seed(&original, 42);
            CHECK(h3_rng_normal(&original) != got[0]);
        }
        free(want); free(got); free(repeat);
    }
    size_t half_n, n;
    float *half_input = tensor(&f, "half.input", &half_n), *half_output = tensor(&f, "half.output", &n); CHECK(n == half_n);
    for (size_t i = 0; i < n; i++) {
        float got = h3_video_f16_to_f32(h3_video_f32_to_f16(half_input[i]));
        CHECK(!memcmp(&got, half_output+i, 4));
    }
    free(half_input); free(half_output);
    float *moments = tensor(&f, "x.moments", &n);
    size_t count; float *epsilon = tensor(&f, "x.epsilon", &count); CHECK(n == 2*count && count%24 == 0);
    float *want = tensor(&f, "x.sample", &n); CHECK(n == count);
    float *got = malloc(n*4); CHECK(got && h3_video_posterior_sample(moments, epsilon, n, got));
    double sample_max = 0;
    for (size_t i = 0; i < n; i++) {
        double d = fabs((double)got[i]-want[i]); if (d > sample_max) sample_max = d;
        CHECK(isfinite(got[i]) && d <= 2e-3 + 3e-7*fabs(want[i]));
    }
    float *rounded = tensor(&f, "x.rounded", &n);
    for (size_t i = 0; i < n; i++) CHECK(h3_video_f16_to_f32(h3_video_f32_to_f16(got[i])) == rounded[i]);
    float *mean = tensor(&f, "x.mean", &n); CHECK(n == 24);
    float *std = tensor(&f, "x.std", &n); CHECK(n == 24);
    float *normalized = tensor(&f, "x.normalized", &n); CHECK(n == count);
    CHECK(h3_video_posterior_normalize(got, count/24, mean, std, got));
    for (size_t i = 0; i < n; i++) CHECK(got[i] == normalized[i]);
    std[0] = 0; CHECK(!h3_video_posterior_normalize(got, count/24, mean, std, got));
    free(moments); free(epsilon); free(want); free(got); free(rounded); free(mean); free(std); free(normalized);
    h3_st_free_header(&f);
    printf("oracle: RNG max abs %.9g; posterior sample max abs %.9g; FP16/normalization exact\n", rng_max, sample_max);
}

int main(int argc, char **argv) {
    half_roundtrip(); temporal();
    oracle(argc > 1 ? argv[1] : "tests/fixtures/refvideo-posterior.safetensors");
    printf("ok: %zu released video posterior, FP16, RNG and temporal checks\n", checks);
    return 0;
}
