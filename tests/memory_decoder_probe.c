/* Current decoder failure/ownership probe used by memory_decoder.py.
 * Allocation tracking covers only the decoder, excluding the FFmpeg child. */
#include "src/memory.h"
#include <inttypes.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/wait.h>
#include <errno.h>
#include <unistd.h>

static struct { void *pointer; size_t bytes; } allocations[128];
static size_t live, peak, largest, calls, fail_at, first, second;
static void *tracked_malloc(size_t bytes) {
    calls++;
    if (calls == 1) first = bytes;
    if (calls == 2) second = bytes;
    if (fail_at && calls == fail_at) return NULL;
    void *p = malloc(bytes);
    if (!p) return NULL;
    size_t i = 0;
    while (i < 128 && allocations[i].pointer) i++;
    if (i == 128) abort();
    allocations[i].pointer = p; allocations[i].bytes = bytes;
    live += bytes;
    if (live > peak) peak = live;
    if (bytes > largest) largest = bytes;
    return p;
}
static void tracked_free(void *p) {
    if (!p) return;
    for (size_t i = 0; i < 128; i++) if (allocations[i].pointer == p) {
        live -= allocations[i].bytes; allocations[i].pointer = NULL;
        free(p); return;
    }
    abort();
}
static int read_calls;
static ssize_t tracked_read(int fd, void *data, size_t size) {
    const char *failure = getenv("H3_TEST_READ_FAILURE_AT");
    if (failure && ++read_calls == atoi(failure)) { errno = EIO; return -1; }
    return read(fd, data, size);
}
#define read tracked_read
#define malloc tracked_malloc
#define free tracked_free
#include H3_DECODER_SOURCE
#undef read
#undef malloc
#undef free

static int query_calls, cancel_at;
static uint64_t first_available, minimum_available = UINT64_MAX, test_floor, allocation_reserve;
static int memory_query(uint64_t *bytes, void *opaque) {
    (void)opaque;
    if (cancel_at > 0) { *bytes = ++query_calls >= cancel_at ? 0 : UINT64_MAX; return 1; }
    /* Safe macOS stress: use live OS measurements and raise the test floor,
     * instead of driving the machine anywhere near physical exhaustion. */
    h3_memory_set_test_query(NULL, NULL);
    int ok = h3_memory_available_bytes(bytes);
    h3_memory_set_test_query(memory_query, NULL);
    if (!ok) return 0;
    if (!query_calls) first_available = *bytes;
    if (*bytes < minimum_available) minimum_available = *bytes;
    query_calls++;
    if (query_calls == 1 || query_calls == -cancel_at) {
        test_floor = H3_MIN_AVAILABLE_MEMORY;
        uint64_t margin = UINT64_C(64) * 1024 * 1024;
        if (query_calls == 1 && *bytes > allocation_reserve &&
            *bytes - allocation_reserve > H3_MIN_AVAILABLE_MEMORY + margin)
            test_floor = *bytes - allocation_reserve - margin;
        if (query_calls != 1) test_floor = *bytes + 2 * 1024 * 1024;
        char floor[32]; snprintf(floor, sizeof(floor), "%" PRIu64, test_floor);
        setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES", floor, 1);
    }
    return 1;
}
int main(int argc, char **argv) {
    if (argc != 7) return 2;
    const char *allocation = getenv("H3_TEST_FAIL_ALLOCATION_AT");
    if (allocation) fail_at = (size_t)strtoull(allocation, NULL, 10);
    cancel_at = atoi(argv[6]);
    if (cancel_at) h3_memory_set_test_query(memory_query, NULL);
    int width = atoi(argv[2]), height = atoi(argv[3]), frames = -1;
    int cap = atoi(argv[4]);
    allocation_reserve = (uint64_t)width * (uint64_t)height * 3 * (uint64_t)cap * 4;
    float *pixels = (void *)1;
    char error[512] = {0};
    int ok = h3_ffmpeg_read_normalized_video_f32(
        argv[1], width, height, cap, &pixels, &frames, error, sizeof(error));
    size_t bytes = ok ? (size_t)frames * (size_t)width * (size_t)height * 3 * sizeof(float) : 0;
    if (ok) {
        FILE *out = fopen(argv[5], "wb");
        if (!out || fwrite(pixels, 1, bytes, out) != bytes || fclose(out)) return 3;
        tracked_free(pixels);
    } else if (pixels || frames) return 4;
    int status;
    int child = (int)waitpid(-1, &status, WNOHANG);
    /* Darwin may briefly report the failed spawn while the kernel retires it. */
    for (int retry = 0; child == 0 && retry < 1000; retry++) {
        usleep(1000); child = (int)waitpid(-1, &status, WNOHANG);
    }
    if (child != -1 || errno != ECHILD || live) { fprintf(stderr,"cleanup child=%d errno=%d live=%zu error=%s\n",child,errno,live,error); return 5; }
    if (cancel_at < 0) fprintf(stderr,
        "stress actual first=%" PRIu64 " minimum=%" PRIu64 " test-floor=%" PRIu64 "\n",
        first_available, minimum_available, test_floor);
    fprintf(stderr, "%s\n", error);
    printf("{\"ok\":%d,\"frames\":%d,\"bytes\":%zu,\"peak\":%zu,\"live\":%zu,"
           "\"calls\":%zu,\"first\":%zu,\"second\":%zu,\"largest\":%zu,\"queries\":%d}\n",
        ok, frames, bytes, peak, live, calls, first, second, largest, query_calls);
    return 0;
}
