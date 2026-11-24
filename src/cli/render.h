#ifndef H3_CLI_RENDER_H
#define H3_CLI_RENDER_H
#include "src/request.h"
#include "src/server/json.h"

/* The planner never starts workers, reads models or creates outputs. */
typedef struct {
    char *name, *label, *prompt, *preamble, *sections[6];
    int line, continuing, skipped, frames, delivered, width, height;
    uint64_t seed_offset;
    h3_request settings;
} h3_render_segment;
typedef struct { char *name, *path; } h3_render_reference;
typedef struct {
    char *source, *source_text, *output, *work, *merged;
    h3_render_segment *segments;
    size_t count;
    uint64_t *seed_set;
    size_t seed_count;
    h3_render_reference *references;
    size_t reference_count;
    int nomerge, dry_run, overlay_segment_name, upscale, upscale_refine_steps;
} h3_render_plan;

static inline int h3_render_width(const h3_render_plan *p,const h3_render_segment *s) {return s->width*(p->upscale?2:1);}
static inline int h3_render_height(const h3_render_plan *p,const h3_render_segment *s) {return s->height*(p->upscale?2:1);}

int h3_render_plan_read(h3_render_plan *plan, const h3_request *request,
                        char *error, size_t size);
void h3_render_plan_free(h3_render_plan *plan);
int h3_render_request(const h3_render_plan *plan, size_t index, h3_request *request,
                      char *error, size_t size);
int h3_render_upscale_request(const h3_render_plan *plan,size_t index,h3_request *request,
                              char *error,size_t size);
char *h3_render_chain_name(const h3_render_segment *segment);
sj_value *h3_render_plan_json(const h3_render_plan *plan, const char *binary);
int h3_render_run(const h3_request *request, const char *binary,
                  int (*validate)(const h3_request *));

/* Owned strings; allocation failures terminate the one-shot command. */
char *h3_render_format(const char *format, ...) __attribute__((format(printf, 1, 2)));
char *h3_render_absolute(const char *base, const char *value);
int h3_render_badge(const char *path, const char *label, int width, int height,
                    int *margin, char *error, size_t size);
#endif
