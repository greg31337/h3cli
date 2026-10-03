#ifndef H3_LORA_JSON_H
#define H3_LORA_JSON_H
#include <stddef.h>
#include <stdio.h>
typedef enum { LJ_NULL, LJ_BOOL, LJ_NUMBER, LJ_STRING, LJ_ARRAY, LJ_OBJECT } lj_type;
typedef struct lj_value {
    lj_type type;
    char *key, *text;
    double number;
    struct lj_value **items;
    size_t count;
} lj_value;
lj_value *lj_parse(const char *text, size_t size, char *error, size_t error_size);
lj_value *lj_file(const char *path, char *error, size_t error_size);
void lj_free(lj_value *value);
const lj_value *lj_get(const lj_value *object, const char *key);
int lj_number(const char *text, double *value);
int lj_quote(FILE *file, const char *text);
#endif
