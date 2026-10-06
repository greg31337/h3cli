#include "src/log.h"
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static _Thread_local int verbose_override = -1;
static _Thread_local void (*before_line)(void *);
static _Thread_local void *line_opaque;

int h3_log_verbose(void) {
    if (verbose_override >= 0) return verbose_override;
    const char *value = getenv("H3_VERBOSE");
    return (value && !strcmp(value, "1")) || getenv("H3_PROFILE") != NULL;
}
int h3_log_exchange_verbose(int enabled) {
    int previous = verbose_override;
    verbose_override = enabled;
    return previous;
}
void h3_log_line_callback(void (*callback)(void *), void *opaque) {
    before_line = callback;
    line_opaque = opaque;
}
void h3_log_printf(const char *format, ...) {
    if (before_line) before_line(line_opaque);
    va_list args;
    va_start(args, format);
    vfprintf(stderr, format, args);
    va_end(args);
}
