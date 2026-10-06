#ifndef H3_LOG_H
#define H3_LOG_H

#ifdef __cplusplus
extern "C" {
#endif

/* Diagnostics never enable profiling or alter numerical execution. The CLI
 * scopes the override to one request; library callers can use H3_VERBOSE=1. */
int h3_log_verbose(void);
int h3_log_exchange_verbose(int enabled); /* -1 restores environment defaults. */
void h3_log_line_callback(void (*callback)(void *), void *opaque);
void h3_log_printf(const char *format, ...) __attribute__((format(printf, 1, 2)));

#ifdef __cplusplus
}
#endif

#define H3_VERBOSE(...) do { if (h3_log_verbose()) h3_log_printf(__VA_ARGS__); } while (0)
#endif
