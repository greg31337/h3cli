#include "src/models/models.h"
#include <string.h>
#include "src/models/catalog.inc"

int h3_models_catalog_valid(char *error, size_t size) {
    for (size_t i = 0; i < h3_model_catalog_count; i++) {
        const h3_model_artifact *a = &h3_model_catalog[i];
        if (!a->bytes || !a->groups || (a->groups & ~H3_MODEL_ALL) ||
            strlen(a->sha256) != 64 || strlen(a->revision) != 40 ||
            strncmp(a->url, "https://", 8) || !strstr(a->url, a->revision) ||
            a->path[0] == '/' || strstr(a->path, "..") || strchr(a->path, '\\')) {
            snprintf(error, size, "invalid embedded model artifact: %s", a->path);
            return 0;
        }
        for (size_t j = 0; j < i; j++) {
            if (!strcmp(a->path, h3_model_catalog[j].path)) {
                snprintf(error, size, "duplicate embedded model artifact: %s", a->path);
                return 0;
            }
        }
    }
    return 1;
}
