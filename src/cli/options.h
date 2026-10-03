#ifndef H3_CLI_OPTIONS_H
#define H3_CLI_OPTIONS_H
#include <getopt.h>
#include <stddef.h>
#include <stdint.h>
typedef enum { H3_OPT_BOOL, H3_OPT_INT, H3_OPT_U64, H3_OPT_FLOAT,
               H3_OPT_STRING, H3_OPT_ENUM } h3_option_type;
typedef enum { H3_PATH_NONE, H3_PATH_MEDIA, H3_PATH_STATE, H3_PATH_MODEL,
               H3_PATH_OUTPUT, H3_PATH_CACHE } h3_path_role;
typedef struct {
    const char *name;
    int id, arity;
    h3_option_type type;
    h3_path_role path;
    int repeatable;
    const char *group;
    double minimum, maximum;
    const char *choices, *default_value, *platform, *operations;
} h3_option_descriptor;
extern const struct option h3_cli_getopt_options[];
extern const h3_option_descriptor h3_cli_options[];
extern const size_t h3_cli_option_count;
typedef struct {const char *name;int steps,reuse,adaptive,preview;} h3_quality_preset;
extern const h3_quality_preset h3_quality_presets[5];
const h3_option_descriptor *h3_option_find(const char *name);
const h3_option_descriptor *h3_option_short(char name);
int h3_option_validate(const h3_option_descriptor *option, const char *value,
                       char *error, size_t size);
#endif
