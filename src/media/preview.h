#ifndef H3_PREVIEW_H
#define H3_PREVIEW_H
#include <stddef.h>

/* Display semantics only; never part of numerical sampler/checkpoint state. */
typedef enum { H3_PREVIEW_DENOISED, H3_PREVIEW_NOISY } h3_preview_mode;
int h3_preview_mode_parse(const char *value, h3_preview_mode *mode,
                          char *error, size_t error_size);
const char *h3_preview_mode_name(h3_preview_mode mode);
/* Reuse the effective velocity workspace only after history/state consumers.
 * sample is x_next, sigma is sigma_next. The sample must remain untouched. */
void h3_preview_denoised_f32(float *velocity, const float *sample,
                            size_t elements, float sigma);
#endif
