#ifndef H3_CLI_PROGRESS_H
#define H3_CLI_PROGRESS_H

#include <stdio.h>

typedef struct {
    char phase[64];
    double started;
} h3_cli_phase_time;

typedef struct {
    char phase[64];
    int active;
    int completed;
    int total;
    double phase_started;
    double step_started;
    int phase_timing;
    int terminal;
    int line_open;
    double last_draw;
    h3_cli_phase_time pending[32];
} h3_cli_progress_state;

/* CLI progress output. Zero totals are indeterminate. */
void h3_cli_progress_update(h3_cli_progress_state *state, FILE *stream,
                            const char *phase, int completed, int total);
/* End a live line for a frame/error message, without claiming completion. */
void h3_cli_progress_finish(h3_cli_progress_state *state, FILE *stream);

#endif
