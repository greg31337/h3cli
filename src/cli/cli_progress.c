#include "src/cli/cli_progress.h"

#include <string.h>
#include <stdlib.h>
#include <time.h>

static double progress_now(void) {
    struct timespec t;if(clock_gettime(CLOCK_MONOTONIC,&t))return 0;
    return (double)t.tv_sec+(double)t.tv_nsec*1e-9;
}
static void profile_finish(h3_cli_progress_state *state,FILE *stream) {
    if(state->phase_started>0) {
        fprintf(stream,"\nh3cli: phase duration %s: %.6f s\n",state->phase,progress_now()-state->phase_started);
        state->phase_started=0;
    }
}

void h3_cli_progress_finish(h3_cli_progress_state *state, FILE *stream) {
    if (state->active) fputc('\n', stream);
    state->active = 0;
}

void h3_cli_progress_update(h3_cli_progress_state *state, FILE *stream,
                            const char *phase, int completed, int total) {
    int same_phase = !strcmp(state->phase, phase);
    int denoise = !strcmp(phase, "denoise");
    double step_seconds = -1;
    if (denoise) {
        double now = progress_now();
        if (same_phase && total > 0 && state->total == total &&
            completed == state->completed + 1 && state->step_started > 0 &&
            now >= state->step_started)
            step_seconds = now - state->step_started;
        /* The repeated counter at the start of the next step resets its
         * timer without redrawing. Enqueue progress is not GPU completion. */
        state->step_started = now;
    } else state->step_started = 0;
    if (same_phase && state->completed == completed &&
        state->total == total) return;
    if (!same_phase) {
        h3_cli_progress_finish(state, stream);
        profile_finish(state,stream);
        snprintf(state->phase, sizeof(state->phase), "%s", phase);
        /* Reference qualification needs ordinary phase timing without the
         * CUDA event instrumentation enabled by H3_PROFILE. */
        if(getenv("H3_PROFILE") || state->phase_timing) {
            state->phase_started=progress_now();
            fprintf(stream,"h3cli: phase start %s: monotonic %.6f\n",phase,state->phase_started);
        }
    }
    state->completed = completed;
    state->total = total;
    state->active = total <= 0 || completed < total;
    char status[64];
    if (total <= 0) snprintf(status, sizeof(status), "loading...");
    else if (completed == 0) snprintf(status, sizeof(status), "starting...");
    else snprintf(status, sizeof(status), "%4d/%-4d", completed, total);
    /* Pad the status so a shorter counter erases the previous loading label. */
    fprintf(stream, "\r%-25s %-12s", phase, status);
    if (denoise) {
        char timing[64] = "";
        if (step_seconds >= 0)
            snprintf(timing, sizeof(timing), "(last step: %.2f s)", step_seconds);
        fprintf(stream, "%-32s", timing);
    }
    if (!state->active) fputc('\n', stream);
    if(!state->active)profile_finish(state,stream);
    fflush(stream);
}
