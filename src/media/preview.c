#include "src/media/preview.h"
#include <math.h>
#include <stdio.h>
#include <string.h>

int h3_preview_mode_parse(const char *value, h3_preview_mode *mode,
                          char *error, size_t error_size) {
    if (!mode) return 0;
    if (!value || !*value) *mode = H3_PREVIEW_DENOISED;
    else if (!strcmp(value, "denoised")) *mode = H3_PREVIEW_DENOISED;
    else if (!strcmp(value, "noisy")) *mode = H3_PREVIEW_NOISY;
    else {
        if (error && error_size) snprintf(error, error_size,
            "invalid H3_PREVIEW_MODE '%s'; expected denoised or noisy", value);
        return 0;
    }
    return 1;
}

const char *h3_preview_mode_name(h3_preview_mode mode) {
    return mode == H3_PREVIEW_DENOISED ? "denoised" : "noisy";
}

void h3_preview_denoised_f32(float *velocity, const float *sample,
                            size_t elements, float sigma) {
    /* Preserve the final step and zero-velocity prefixes bitwise, including
     * signed zero, without consulting an unused nonfinite velocity. */
    if (sigma == 0.0f) {
        memcpy(velocity, sample, elements * sizeof(*velocity));
        return;
    }
    for (size_t i = 0; i < elements; i++)
        velocity[i] = velocity[i] == 0.0f ? sample[i] :
            fmaf(sigma, velocity[i], sample[i]);
}
