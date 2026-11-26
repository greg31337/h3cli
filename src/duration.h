#ifndef H3_DURATION_H
#define H3_DURATION_H
#include <stdio.h>

/* Human-readable durations only. Serialized timings retain full precision. */
typedef struct { char text[64]; } h3_duration_text;
static inline h3_duration_text h3_duration_format(double seconds, int digits) {
    h3_duration_text value;
    snprintf(value.text, sizeof(value.text), "%.*f", seconds > 5.0 ? 0 : digits, seconds);
    return value;
}
#define H3_SECONDS(seconds, digits) (h3_duration_format((seconds), (digits)).text)
#endif
