#ifndef H3_MODELS_H
#define H3_MODELS_H

#include "src/request.h"
#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

enum {
    H3_MODEL_BASE = 1, H3_MODEL_REFERENCES = 2, H3_MODEL_PREVIEW = 4,
    H3_MODEL_IMAGE = 8, H3_MODEL_UPSCALE = 16, H3_MODEL_ALL = 31
};
typedef struct {
    const char *path, *url, *sha256, *repository, *revision;
    uint64_t bytes;
    unsigned groups;
} h3_model_artifact;
extern const h3_model_artifact h3_model_catalog[];
extern const size_t h3_model_catalog_count;
extern const char h3_model_catalog_sha256[];

typedef struct {
    char *root, *main, *preview, *image, *upscale;
} h3_model_paths;
typedef struct {
    const h3_model_artifact *artifact;
    char *destination;
} h3_model_item;
typedef struct {
    h3_model_paths paths;
    h3_model_item *items;
    size_t count;
    int offline, verify, inspect;
} h3_model_plan;
typedef struct {
    const char *phase, *component;
    uint64_t completed, total, reused;
    double elapsed;
} h3_model_progress;
typedef int (*h3_model_callback)(const h3_model_progress *progress, void *opaque);

int h3_models_offline(void);
int h3_models_global_init(char *error, size_t size);
char *h3_models_absolute(const char *path, char *error, size_t size);
int h3_models_paths(const h3_request *request, h3_model_paths *paths,
                    char *error, size_t size);
void h3_models_paths_free(h3_model_paths *paths);
int h3_models_groups(const char *text, unsigned *groups, char *error, size_t size);
int h3_models_resolve(const h3_request *request, h3_model_plan *plan,
                      char *error, size_t size);
void h3_models_plan_free(h3_model_plan *plan);
int h3_models_apply_paths(h3_request *request, const h3_model_plan *plan,
                          char *error, size_t size);
int h3_models_prepare(const h3_model_plan *plan, h3_model_callback callback,
                      void *opaque, char *error, size_t size);
int h3_models_list(const h3_model_plan *plan, FILE *stream);
int h3_models_missing(const h3_model_plan *plan);
int h3_models_catalog_valid(char *error, size_t size);

#endif
