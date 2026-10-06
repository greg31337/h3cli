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
        h3_cli_progress_finish(state, stream);
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
    /* Logs get stage boundaries, completed denoising steps, and a heartbeat
     * for long loads/decodes. Terminals get a throttled in-place counter. */
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
            fprintf(stream, "\r%-100s", line);
            state->line_open = 1;
            if (done) { fputc('\n', stream); state->line_open = 0; }
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
