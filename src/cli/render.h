#ifndef H3_CLI_RENDER_H
#define H3_CLI_RENDER_H
#include "src/request.h"
#include "src/server/json.h"

/* The planner never starts workers, reads models or creates outputs. */
typedef struct {
    char *name, *prompt, *preamble, *sections[6];
    int line, continuing, skipped, frames, delivered, width, height;
    h3_request settings;
} h3_render_segment;
typedef struct {
    char *source, *source_text, *output, *work, *merged;
    h3_render_segment *segments;
    size_t count;
    int nomerge, dry_run;
} h3_render_plan;

int h3_render_plan_read(h3_render_plan *plan, const h3_request *request,
                        char *error, size_t size);
void h3_render_plan_free(h3_render_plan *plan);
int h3_render_request(const h3_render_plan *plan, size_t index, h3_request *request,
                      char *error, size_t size);
char *h3_render_chain_name(const h3_render_segment *segment);
sj_value *h3_render_plan_json(const h3_render_plan *plan, const char *binary);
int h3_render_run(const h3_request *request, const char *binary,
                  int (*validate)(const h3_request *));

/* Owned strings; allocation failures terminate the one-shot command. */
char *h3_render_format(const char *format, ...) __attribute__((format(printf, 1, 2)));
char *h3_render_absolute(const char *base, const char *value);
#endif
