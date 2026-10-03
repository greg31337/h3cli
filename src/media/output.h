#ifndef H3_MEDIA_OUTPUT_H
#define H3_MEDIA_OUTPUT_H

#include <stddef.h>
#include <stdio.h>
#include <string.h>

typedef enum {
    H3_OUTPUT_PRODUCTION = 0, /* Omitted quality: CRF 18. */
    H3_OUTPUT_DEFAULT,        /* Explicit SGLang default: CRF 25. */
    H3_OUTPUT_MAXIMUM,
    H3_OUTPUT_HIGH,
    H3_OUTPUT_MEDIUM,
    H3_OUTPUT_LOW
} h3_output_quality;

/* Delivery-only settings; never part of conditioning or sampler identity.
 * Zero initialization selects the production default. */
typedef struct {
    h3_output_quality quality;
    int crf, crf_set;
    int lossless_video; /* libx264rgb CRF 0; preserves delivered RGB24 exactly. */
} h3_output_encoding;

static inline int h3_output_quality_parse(const char *name, h3_output_quality *out) {
    static const char *const names[] = {"default", "maximum", "high", "medium", "low"};
    for (size_t i=0; name && i<sizeof(names)/sizeof(*names); i++)
        if (!strcmp(name,names[i])) { *out=(h3_output_quality)(i+1); return 1; }
    return 0;
}
static inline int h3_output_encoding_selected(const h3_output_encoding *o) {
    return o && (o->quality || o->crf_set || o->lossless_video);
}
static inline int h3_output_encoding_valid(const h3_output_encoding *o, char *error, size_t size) {
    const char *why=NULL;
    if (!o) return 1;
    if (o->quality<H3_OUTPUT_PRODUCTION || o->quality>H3_OUTPUT_LOW)
        why="output-quality must be default, maximum, high, medium or low";
    else if ((o->crf_set!=0 && o->crf_set!=1) || (o->lossless_video!=0 && o->lossless_video!=1))
        why="invalid output encoding selection";
    else if (o->crf_set && (o->crf<0 || o->crf>51))
        why="ffmpeg-crf must be an integer from 0 to 51";
    else if (o->lossless_video && o->crf_set && o->crf!=0)
        why="lossless-video requires ffmpeg-crf 0 (or omit ffmpeg-crf)";
    if (why && error && size) snprintf(error,size,"%s",why);
    return why==NULL;
}
/* Caller validates first. This follows SGLang's output_quality mapping. */
static inline int h3_output_crf(const h3_output_encoding *o) {
    static const int crf[] = {18,25,0,5,22,33};
    return !o ? 18 : o->lossless_video ? 0 : o->crf_set ? o->crf : crf[o->quality];
}

#endif
