#include "src/models/models.h"
#include "src/media/delivery.h"
#include "src/sampling/sampler_state.h"
#include <errno.h>
#include <inttypes.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

static const char *value(const h3_request *r, const char *key) {
    const h3_option_value *v = h3_request_get(r, key);
    return v ? v->value : NULL;
}
static char *join(const char *root, const char *leaf) {
    char *path = NULL;
    if (asprintf(&path, "%s/%s", root, leaf) < 0) return NULL;
    return path;
}
void h3_models_paths_free(h3_model_paths *p) {
    free(p->root); free(p->main); free(p->preview); free(p->image); free(p->upscale);
    memset(p, 0, sizeof(*p));
}
static char *resolve_path(const h3_request *r, const char *key, const char *root,
                          const char *leaf, char *error, size_t size) {
    const char *explicit_path = value(r, key);
    char *candidate = explicit_path ? strdup(explicit_path) : join(root, leaf);
    char *result = candidate ? h3_models_absolute(candidate, error, size) : NULL;
    free(candidate);
    return result;
}
int h3_models_paths(const h3_request *r, h3_model_paths *p, char *error, size_t size) {
    memset(p, 0, sizeof(*p));
    const char *root = value(r, "models-path");
    p->root = h3_models_absolute(root ? root : "models", error, size);
    if (!p->root) goto failed;
    const char *main_leaf = "MiniMaxH3";
    if (!value(r, "model-dir")) {
        char *canonical = join(p->root, "MiniMaxH3");
        char *old = join(p->root, "MiniMax-H3");
        struct stat st;
        if (!canonical || !old) { free(canonical); free(old); goto failed; }
        if (lstat(canonical, &st) && errno == ENOENT && !lstat(old, &st)) main_leaf = "MiniMax-H3";
        free(canonical); free(old);
    }
    p->main = resolve_path(r, "model-dir", p->root, main_leaf, error, size);
    p->preview = resolve_path(r, "preview-vae-model", p->root, "preview-vae/taeh3.safetensors", error, size);
    p->image = resolve_path(r, "image-vae", p->root, "image-vae/minimax_h3_t1_image_vae_step1597.safetensors", error, size);
    p->upscale = resolve_path(r, "upscale-model", p->root, "latent-upscale/minimax_h3_latent_upscaler_3d_conv_v1_bf16.safetensors", error, size);
    if (p->main && p->preview && p->image && p->upscale) return 1;
failed:
    if (!*error) snprintf(error, size, "cannot resolve model paths");
    h3_models_paths_free(p); return 0;
}
int h3_models_groups(const char *text, unsigned *groups, char *error, size_t size) {
    const char *names[] = {"base", "references", "preview", "image-vae", "upscale", "all"};
    const unsigned bits[] = {1, 2, 4, 8, 16, 31};
    *groups = 0;
    if (!text || !*text) goto bad;
    const char *start = text;
    do {
        const char *end = strchr(start, ',');
        size_t length = end ? (size_t)(end-start) : strlen(start);
        unsigned selected = 0;
        for (size_t i = 0; i < 6; i++)
            if (strlen(names[i]) == length && !memcmp(start, names[i], length)) selected = bits[i];
        if (!selected) goto bad;
        *groups |= selected;
        if (!end) return 1;
        start = end+1;
    } while (*start);
bad:
    snprintf(error, size, "model groups must be base,references,preview,image-vae,upscale or all");
    return 0;
}
static int preview_enabled(const h3_request *r) {
    const char *quality = value(r, "quality");
    int enabled = quality && (!strcmp(quality, "preview") || !strcmp(quality, "fast-preview"));
    for (size_t i = 0; i < r->count; i++) {
        if (!strcmp(r->values[i].option->name, "preview-vae")) enabled = 1;
        if (!strcmp(r->values[i].option->name, "no-preview-vae")) enabled = 0;
    }
    return enabled;
}
static int decode_dependency(const char *path) {
    const char *relative = strchr(path, '/');
    if (!relative) return 0;
    relative++;
    const char *files[] = {"transformer/config.json", "video_vae/config.json",
        "video_vae/source/config.json", "video_vae/source/model.safetensors",
        "audio_vae/config.json", "audio_vae/config.yaml", "audio_vae/metadata.json",
        "audio_vae/model.safetensors"};
    for (size_t i = 0; i < sizeof(files)/sizeof(*files); i++)
        if (!strcmp(relative, files[i])) return 1;
    return 0;
}
static int presentation_mode(const char *path, int *mode, char *error, size_t size) {
    struct stat st;
    if(!path||stat(path,&st)||!S_ISREG(st.st_mode)){snprintf(error,size,"cannot open h3av state: %.180s",path?path:"");return 0;}
    char *sidecar = NULL;
    if (asprintf(&sidecar, "%s.presentation", path) < 0) return 0;
    FILE *f = fopen(sidecar, "rb"); free(sidecar);
    char text[256] = {0}, hash[65];
    size_t n = f ? fread(text, 1, sizeof(text)-1, f) : 0;
    if (f) fclose(f);
    int version = 0;
    if (n && sscanf(text, "H3-PRESENTATION %d\nstate %64s\nvariant %d", &version, hash, mode) == 3 &&
        version == H3_PRESENTATION_VERSION && (*mode == 0 || *mode == 1)) return 1;
    snprintf(error, size, "decode requires current presentation metadata: %.180s", path);
    return 0;
}
int h3_models_resolve(const h3_request *r, h3_model_plan *p, char *error, size_t size) {
    memset(p, 0, sizeof(*p));
    if (!h3_models_paths(r, &p->paths, error, size)) return 0;
    p->offline = h3_models_offline() || h3_request_get(r, "offline") != NULL;
    const char *op = h3_request_operation(r), *prefetch = value(r, "download-models");
    unsigned groups = 0;
    int decode = !strcmp(op, "decode_av"), ref = 0, up = 0;
    p->inspect = !strcmp(op, "info") || !strcmp(op, "help") || !strcmp(op, "inspect_upscale");
    int list = h3_request_get(r, "list-models") != NULL;
    if (prefetch || list) {
        const char *allowed[] = {"models-path", "model-dir", "preview-vae-model", "image-vae",
            "upscale-model", "download-models", "list-models", "offline", "help"};
        for (size_t i = 0; i < r->count; i++) {
            int good = 0;
            for (size_t j = 0; j < sizeof(allowed)/sizeof(*allowed); j++)
                if (!strcmp(r->values[i].option->name, allowed[j])) good = 1;
            if (!good) { snprintf(error, size, "model prefetch/list cannot be combined with --%s", r->values[i].option->name); goto failed; }
        }
        if (prefetch && list) { snprintf(error, size, "choose either --download-models or --list-models"); goto failed; }
        if (!h3_models_groups(prefetch ? prefetch : "all", &groups, error, size)) goto failed;
        p->verify = prefetch != NULL;
        p->inspect = list;
    } else if (!p->inspect) {
        if (h3_request_get(r, "ref-image") || h3_request_get(r, "ref-audio") ||
            h3_request_get(r, "ref-video") || h3_request_get(r, "ref-silent-video") ||
            h3_request_get(r, "ref-video-audio")) ref = 1;
        if (!strcmp(op, "resume") || !strcmp(op, "upscale")) {
            int source = !strcmp(op, "upscale") && !h3_request_get(r, "upscale-import-sampler");
            const char *state = value(r, !strcmp(op, "resume") ? "resume-sampler-state" : "upscale-state");
            if (!h3_sampler_model_requirements(state, source, &ref, &up, error, size)) goto failed;
        }
        if (decode && !presentation_mode(value(r, "decode-av-state"), &ref, error, size)) goto failed;
        if (strcmp(op, "decode_still")) groups |= ref ? H3_MODEL_REFERENCES : H3_MODEL_BASE;
        if (!strcmp(op, "still") || !strcmp(op, "decode_still")) groups |= H3_MODEL_IMAGE;
        if (!strcmp(op, "upscale")) groups |= H3_MODEL_UPSCALE;
        if (preview_enabled(r)) groups |= H3_MODEL_PREVIEW;
    }
    p->items = calloc(h3_model_catalog_count, sizeof(*p->items));
    if (!p->items) goto failed;
    for (size_t i = 0; i < h3_model_catalog_count; i++) {
        const h3_model_artifact *a = &h3_model_catalog[i];
        if (!(a->groups & groups)) continue;
        if (!prefetch && !list && (a->groups & 3u) &&
            (!strchr(a->path, '/') || (decode && !decode_dependency(a->path)))) continue;
        const char *aux = a->groups == H3_MODEL_PREVIEW ? p->paths.preview :
            a->groups == H3_MODEL_IMAGE ? p->paths.image :
            a->groups == H3_MODEL_UPSCALE ? p->paths.upscale : NULL;
        char *dest = aux ? strdup(aux) : join(p->paths.main, a->path);
        if (!dest) goto failed;
        p->items[p->count++] = (h3_model_item){a, dest};
    }
    return 1;
failed:
    if (!*error) snprintf(error, size, "cannot plan model dependencies");
    h3_models_plan_free(p); return 0;
}
void h3_models_plan_free(h3_model_plan *p) {
    for (size_t i = 0; i < p->count; i++) free(p->items[i].destination);
    free(p->items); h3_models_paths_free(&p->paths); memset(p, 0, sizeof(*p));
}
int h3_models_apply_paths(h3_request *r, const h3_model_plan *p, char *error, size_t size) {
    unsigned groups = 0;
    for (size_t i = 0; i < p->count; i++) groups |= p->items[i].artifact->groups;
    const char *names[] = {"model-dir", "preview-vae-model", "image-vae", "upscale-model"};
    const char *paths[] = {p->paths.main, p->paths.preview, p->paths.image, p->paths.upscale};
    const unsigned masks[] = {3, 4, 8, 16};
    for (size_t i = 0; i < 4; i++) {
        if (h3_request_get(r, names[i]) || (groups & masks[i]) || (!i && p->inspect)) {
            h3_request_remove(r, names[i]);
            if (!h3_request_add(r, names[i], paths[i], NULL, H3_SOURCE_MANAGED, error, size)) return 0;
        }
    }
    return 1;
}
int h3_models_missing(const h3_model_plan *p) {
    for (size_t i = 0; i < p->count; i++) {
        struct stat st;
        if (stat(p->items[i].destination, &st) || !S_ISREG(st.st_mode)) return 1;
    }
    return 0;
}
int h3_models_list(const h3_model_plan *p, FILE *stream) {
    fprintf(stream, "Catalog %s\nModels root: %s\nMain model: %s\n", h3_model_catalog_sha256, p->paths.root, p->paths.main);
    uint64_t missing = 0, logical = 0;
    for (size_t i = 0; i < p->count; i++) {
        const h3_model_item *item = &p->items[i];
        struct stat st;
        int present = !stat(item->destination, &st) && S_ISREG(st.st_mode);
        fprintf(stream, "%s\t%" PRIu64 "\t%s\t%s@%s\n", present ? "present (unverified)" : "missing",
            item->artifact->bytes, item->destination, item->artifact->repository, item->artifact->revision);
        logical += item->artifact->bytes;
        int first = 1;
        for (size_t j = 0; j < p->count; j++) {
            if (strcmp(item->artifact->sha256, p->items[j].artifact->sha256)) continue;
            if (j < i || (!stat(p->items[j].destination, &st) && S_ISREG(st.st_mode))) first = 0;
        }
        if (!present && first) missing += item->artifact->bytes;
    }
    fprintf(stream, "Logical bytes: %" PRIu64 "; estimated unique missing transfer bytes: %" PRIu64 "\n", logical, missing);
    return !ferror(stream);
}
