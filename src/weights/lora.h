#ifndef H3_LORA_H
#define H3_LORA_H
#include "src/h3.h"
#include <stdio.h>
typedef struct h3_lora_selection h3_lora_selection;
typedef struct {
    char *transformer;
    char key[65];
    int lease;
    int cache_hit;
    double hash_seconds, verify_seconds, prepare_seconds;
    size_t scratch_peak;
} h3_lora_variant;
/* Host-only API: no GPU, tokenizer or model allocations. NULL options is a no-op. */
int h3_lora_selection_create(const h3_lora_options *options,
    h3_lora_selection **selection, char *error, size_t error_size);
void h3_lora_selection_free(h3_lora_selection *selection);
void h3_lora_selection_print(const h3_lora_selection *selection, FILE *out);
int h3_lora_prepare(const h3_lora_selection *selection, const char *transformer,
    int ref2va, h3_progress_callback progress, void *opaque,
    h3_lora_variant *variant, char *error, size_t error_size);
void h3_lora_variant_free(h3_lora_variant *variant);
/* Informational sidecars never participate in sampler/AV compatibility. */
char *h3_lora_provenance(const h3_lora_selection *selection,
    const h3_lora_variant *variant, const char *base, int ref2va);
int h3_lora_save_provenance(const char *state_path, const char *json,
    char *error, size_t error_size);
#endif
