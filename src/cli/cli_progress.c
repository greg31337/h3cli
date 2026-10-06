#include "src/cli/cli_progress.h"
#include <string.h>
#include <time.h>

static double progress_now(void) {
    struct timespec t;
    if (clock_gettime(CLOCK_MONOTONIC, &t)) return 0;
    return (double)t.tv_sec + (double)t.tv_nsec*1e-9;
}

void h3_cli_progress_finish(h3_cli_progress_state *state, FILE *stream) {
    if (state->line_open) fputc('\n', stream);
    state->line_open = 0;
    state->line_width = 0;
    state->active = 0;
}

/* Parent phases can emit child phases before their completion callback. Keep
 * their start times so the final parent duration includes that work. */
static h3_cli_phase_time *phase_time(h3_cli_progress_state *state, const char *phase) {
    h3_cli_phase_time *empty = NULL;
    for (size_t i = 0; i < sizeof(state->pending)/sizeof(state->pending[0]); i++) {
        h3_cli_phase_time *entry = &state->pending[i];
        if (!strcmp(entry->phase, phase)) return entry;
        if (!entry->phase[0] && !empty) empty = entry;
    }
    return empty;
}

void h3_cli_progress_update(h3_cli_progress_state *state, FILE *stream,
                            const char *phase, int completed, int total) {
    double now = progress_now(), step_seconds = -1;
    int same_phase = !strcmp(state->phase, phase);
    int denoise = !strcmp(phase, "denoise");
    if (denoise) {
        if (same_phase && total > 0 && state->total == total &&
            completed == state->completed + 1 && state->step_started > 0)
            step_seconds = now - state->step_started;
        /* Repeated step starts exclude preview time from the next step. */
        state->step_started = now;
    } else state->step_started = 0;
    if (same_phase && state->completed == completed && state->total == total) return;
    if (!same_phase) {
        /* A parent announces 0/1 before its child stages. Reuse that initial
         * row; its timed completion will get a row after the children finish. */
        int reuse_parent = state->terminal && state->line_open &&
            state->completed == 0 && state->total == 1 && !state->phase_timing;
        if (!reuse_parent) h3_cli_progress_finish(state, stream);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
    }
    h3_cli_phase_time *timer = phase_time(state, phase);
    if (timer && !timer->phase[0]) {
        snprintf(timer->phase, sizeof(timer->phase), "%s", phase);
        timer->started = now;
        if (state->phase_timing)
            fprintf(stream, "h3cli: phase start %s: monotonic %.6f\n", phase, now);
    }
    state->phase_started = timer ? timer->started : now;
    double elapsed = now - state->phase_started;
    state->completed = completed;
    state->total = total;
    state->active = total <= 0 || completed < total;
    int done = !state->active;
    /* Redraws are throttled even through pipes. Explicit plain mode retains
     * stage boundaries, denoising steps, and occasional load/decode updates. */
    int draw = !same_phase || done || (!state->terminal && denoise) ||
        now-state->last_draw >= (state->terminal ? 0.1 : 5.0);
    if (draw) {
        char status[64], timing[96], line[256];
        if (total <= 0) snprintf(status, sizeof(status), "loading...");
        else if (!completed) snprintf(status, sizeof(status), "starting...");
        else snprintf(status, sizeof(status), "%d/%d", completed, total);
        if (step_seconds >= 0)
            snprintf(timing, sizeof(timing), "%.2f s; last step: %.2f s", elapsed, step_seconds);
        else snprintf(timing, sizeof(timing), "%.2f s", elapsed);
        snprintf(line, sizeof(line), "%-28s %-12s (%s)", phase, status, timing);
        if (state->terminal) {
            int width = (int)strlen(line);
            if (width < state->line_width) width = state->line_width;
            /* Erase only the previous row's tail. Fixed 100-column padding
             * wraps on ordinary terminals and defeats in-place updates. The
             * final CR also delimits each update for streaming log readers. */
            fprintf(stream, "\r%-*s\r", width, line);
            state->line_width = width;
            state->line_open = 1;
            if (done) h3_cli_progress_finish(state, stream);
        } else fprintf(stream, "%s\n", line);
        state->last_draw = now;
    }
    if (done) {
        if (state->phase_timing)
            fprintf(stream, "h3cli: phase duration %s: %.6f s\n", phase, elapsed);
        if (timer) memset(timer, 0, sizeof(*timer));
        state->phase_started = 0;
    }
    fflush(stream);
}
