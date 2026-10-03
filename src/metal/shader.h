#ifndef H3_SHADER_H
#define H3_SHADER_H
#include <stddef.h>
/* Returns an owned path. Explicit overrides fail without falling through. */
char *h3_shader_resolve(const char *supplied, char *error, size_t error_size);
#endif
