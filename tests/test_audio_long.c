/* Released AudioVAE weights at the representative 243-frame duration, with synthetic latents.
 * Exercises all seven stages without text encoding, denoising or video decode. */
#include "src/vae/audio_vae.h"

#include <math.h>
#include <stdio.h>
#include <stdlib.h>

enum { LATENT_LENGTH = 405, LATENT_COUNT = 32 * 2 * LATENT_LENGTH,
       SAMPLES = LATENT_LENGTH * 800 };

static void die(const char *message) {
    fprintf(stderr, "FAIL tests/test_audio_long.c: %s\n", message);
    exit(1);
}

static int progress(int completed, int total, void *opaque) {
    int *next = opaque;
    if (total != 7 || completed != (*next)++) die("unexpected decoder progress");
    fprintf(stderr, "long AudioVAE stage: %d/%d\n", completed, total);
    return 0;
}

int main(int argc, char **argv) {
    if (argc > 2) die("usage: bin/audio_long_test [model-root]");
    const char *root = argc == 2 ? argv[1] : "models/MiniMax-H3";
    char weights[4096], error[1024];
    if (snprintf(weights, sizeof(weights), "%s/FL2VA/audio_vae", root) >=
        (int)sizeof(weights)) die("model path too long");
    float *latent = malloc(LATENT_COUNT * sizeof(*latent));
    if (!latent) die("latent allocation failed");
    for (size_t i = 0; i < LATENT_COUNT; i++)
        latent[i] = (float)((int)((i * 17 + 3) % 67) - 33) / 128.0f;
    int next = 0;
    h3_audio_waveform output = {0};
    if (!h3_audio_vae_decode(weights, "src/metal/shaders.metal", latent,
                             LATENT_LENGTH, progress, &next, &output,
                             error, sizeof(error))) die(error);
    if (next != 8 || output.channels != 2 || output.samples != SAMPLES ||
        output.sample_rate != 32000 || !output.pcm) die("incorrect waveform shape");
    double energy = 0.0;
    for (size_t i = 0; i < (size_t)output.channels * output.samples; i++) {
        float value = output.pcm[i];
        if (!isfinite(value) || fabsf(value) > 1.0f) die("invalid decoded PCM");
        energy += (double)value * value;
    }
    if (energy <= 0.0) die("empty decoded waveform");
    printf("ok: seven AudioVAE stages, %d stereo samples at 32000 Hz (%.3f s)\n",
           SAMPLES, (double)SAMPLES / 32000.0);
    h3_audio_waveform_free(&output);
    free(latent);
    return 0;
}
