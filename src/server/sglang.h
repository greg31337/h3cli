#ifndef H3_SERVER_SGLANG_H
#define H3_SERVER_SGLANG_H
#include "src/request.h"
#include "src/server/json.h"
typedef struct {
    h3_request request;
    sj_value *transport;
    uint64_t seeds[10];
    int variants, canonical, native_flags;
    int width_needed, height_needed, duration_needed;
    int needs_embedded_audio;
    unsigned char required_audio[512];
    char task[16];
    char *model;
} h3_submission;
void h3_submission_free(h3_submission *submission);
int h3_submission_parse(const sj_value *body,h3_submission *submission,char *error,size_t size);
/* Resolve auto geometry/audio duration after importing only the effective media. */
int h3_submission_geometry(h3_submission *submission,int image_width,int image_height,
                           double audio_seconds,char *error,size_t size);
sj_value *h3_request_json(const h3_request *request);
int h3_request_from_json(const sj_value *json,h3_request *request,char *error,size_t size);
sj_value *h3_option_schema(void);
sj_value *h3_submission_schema(void);
#endif
