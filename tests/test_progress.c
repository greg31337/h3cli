#include "src/cli/cli_progress.h"
#include <stdlib.h>
#include <string.h>

static int checks;
#define CHECK(x) do { checks++; if (!(x)) { fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x); exit(1); } } while (0)

/* Interpret carriage returns to check the visible terminal lines, including
 * stale text left behind when a status becomes a shorter numeric counter. */
static void screen(const char *raw, char *out) {
    char line[256] = {0}; size_t col = 0, length = 0, used = 0;
    for (; *raw; raw++) {
        if (*raw == '\r') col = 0;
        else if (*raw == '\n') {
            memcpy(out + used, line, length); used += length; out[used++] = '\n';
            memset(line, 0, sizeof(line)); col = length = 0;
        } else {
            CHECK(col < sizeof(line)); line[col++] = *raw;
            if (col > length) length = col;
        }
    }
    memcpy(out + used, line, length); out[used + length] = '\0';
}

int main(void) {
    h3_cli_progress_state state = {.terminal = 1}; char *raw = NULL; size_t size = 0;
    FILE *f = open_memstream(&raw, &size); CHECK(f);
    h3_cli_progress_update(&state, f, "reference vision preparation", 0, 1);
    CHECK(state.active && strstr(raw, "starting...") && !strstr(raw, "0/1"));
    h3_cli_progress_update(&state, f, "video VAE encoder", 0, 0);
    CHECK(state.active && strstr(raw, "loading...") && !strstr(raw, "0/0"));
    size_t before = size;
    h3_cli_progress_update(&state, f, "video VAE encoder", 0, 0);
    CHECK(size == before); /* Repeated memory/cancellation checkpoints stay quiet. */
    for (int i = 0; i <= 9; i++)
        h3_cli_progress_update(&state, f, "video VAE encoder", i, 9);
    CHECK(!state.active);
    h3_cli_progress_update(&state, f, "Qwen vision", 27, 27);
    h3_cli_progress_update(&state, f, "reference vision preparation", 1, 1);
    h3_cli_progress_update(&state, f, "DiT initialization", 0, 1);
    h3_cli_progress_update(&state, f, "load transformer core", 50, 50);
    h3_cli_progress_update(&state, f, "DiT initialization", 1, 1);
    before = size; h3_cli_progress_finish(&state, f); fflush(f); CHECK(before == size);
    CHECK(fclose(f) == 0);
    char visible[4096]; screen(raw, visible);
    CHECK(strstr(visible, "video VAE encoder            9/9"));
    CHECK(!strstr(visible, "9/9   ng"));
    CHECK(!strstr(visible, "loading..."));
    CHECK(strstr(visible, "reference vision preparation 1/1"));
    CHECK(strstr(visible, "DiT initialization           1/1"));
    free(raw);

    /* Interruption/failure terminates an indeterminate line without reporting
     * success. A later update may resume on a new line, as with frame previews. */
    raw = NULL; size = 0; memset(&state, 0, sizeof(state));
    f = open_memstream(&raw, &size); CHECK(f);
    h3_cli_progress_update(&state, f, "video VAE encoder", 0, 0);
    h3_cli_progress_finish(&state, f); fflush(f);
    CHECK(!state.active && raw[size-1] == '\n' && !strstr(raw, "1/1"));
    before = size; h3_cli_progress_finish(&state, f); fflush(f); CHECK(before == size);
    h3_cli_progress_update(&state, f, "video VAE encoder", 1, 9);
    CHECK(state.active);
    CHECK(fclose(f) == 0); free(raw);

    /* Completed GPU boundaries use the same timer as CPU steps. Repeated
     * starts exclude intervening previews; enqueued windows cannot claim a
     * per-step duration, including when the final window finally completes. */
    raw = NULL; size = 0; memset(&state, 0, sizeof(state));
    f = open_memstream(&raw, &size); CHECK(f);
    h3_cli_progress_update(&state, f, "denoise", 2, 6); /* Resumed range. */
    state.step_started -= 2;
    h3_cli_progress_update(&state, f, "denoise", 3, 6);
    CHECK(strstr(raw, "last step: 2."));
    before = size;
    state.step_started -= 3600; /* Preview work must not enter the next step. */
    h3_cli_progress_update(&state, f, "denoise", 3, 6);
    CHECK(size == before);
    h3_cli_progress_update(&state, f, "denoise", 4, 6);
    CHECK(strstr(raw + before, "last step: 0."));
    before = size;
    h3_cli_progress_update(&state, f, "denoise enqueue", 5, 6);
    h3_cli_progress_update(&state, f, "denoise", 6, 6);
    CHECK(!strstr(raw + before, "last step:"));
    CHECK(fclose(f) == 0); free(raw);
    /* Explicit plain mode has no carriage returns or profiling records, and a
     * large decode cannot flood the log with thousands of tiny updates. */
    raw = NULL; size = 0; memset(&state, 0, sizeof(state));
    f = open_memstream(&raw, &size); CHECK(f);
    h3_cli_progress_update(&state, f, "video VAE decode", 0, 14000);
    for (int i = 1; i <= 14000; i++)
        h3_cli_progress_update(&state, f, "video VAE decode", i, 14000);
    CHECK(!strchr(raw, '\r') && !strstr(raw, "monotonic") && !strstr(raw, "phase duration"));
    CHECK(strstr(raw, "14000/14000") && strstr(raw, " s)") && size < 1000);
    CHECK(fclose(f) == 0); free(raw);

    /* Nested stages retain their original start, and verbose phase timing is
     * available without turning on GPU profiling. */
    raw = NULL; size = 0; memset(&state, 0, sizeof(state));
    state.phase_timing = 1;
    f = open_memstream(&raw, &size); CHECK(f);
    h3_cli_progress_update(&state, f, "reference preparation", 0, 1);
    state.pending[0].started -= 2;
    h3_cli_progress_update(&state, f, "vision encoder", 0, 27);
    h3_cli_progress_update(&state, f, "vision encoder", 27, 27);
    h3_cli_progress_update(&state, f, "reference preparation", 1, 1);
    CHECK(strstr(raw, "phase duration reference preparation: 2."));
    CHECK(strstr(raw, "phase duration vision encoder:"));
    CHECK(!state.pending[0].phase[0] && !state.pending[1].phase[0]);
    CHECK(fclose(f) == 0); free(raw);

    printf("ok: %d progress rendering and lifecycle checks\n", checks);
    return 0;
}
