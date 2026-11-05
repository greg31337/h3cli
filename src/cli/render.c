#include "src/cli/render.h"
#include <errno.h>
#include <fcntl.h>
#include <math.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>
#ifndef __APPLE__
#include <linux/fs.h>
#include <sys/syscall.h>
#endif

extern char **environ;
static volatile sig_atomic_t interrupted;
static void stop(int signal_number) { interrupted = signal_number; }
static double now(void) {
    struct timespec t;
    clock_gettime(CLOCK_MONOTONIC, &t);
    return (double)t.tv_sec + (double)t.tv_nsec / 1e9;
}
/* Every tool is an argv invocation in an owned process group, never a shell. */
static int process(char *const *argv, const char *log_path, int live, char **capture,
                   double timeout, char *error, size_t size) {
    if (interrupted) { snprintf(error, size, "render interrupted"); return 0; }
    FILE *log = log_path ? fopen(log_path, "wb") : NULL;
    if (log_path && !log) { snprintf(error, size, "cannot open log %s: %s", log_path, strerror(errno)); return 0; }
    int pipes[2];
    if (pipe(pipes)) { if (log) fclose(log); snprintf(error, size, "cannot create worker pipe"); return 0; }
    fcntl(pipes[0], F_SETFD, FD_CLOEXEC); fcntl(pipes[1], F_SETFD, FD_CLOEXEC);
    if (log) fcntl(fileno(log), F_SETFD, FD_CLOEXEC);
    posix_spawn_file_actions_t actions;
    posix_spawnattr_t attrs;
    posix_spawn_file_actions_init(&actions); posix_spawnattr_init(&attrs);
    posix_spawn_file_actions_addopen(&actions, STDIN_FILENO, "/dev/null", O_RDONLY, 0);
    posix_spawn_file_actions_adddup2(&actions, pipes[1], STDOUT_FILENO);
    posix_spawn_file_actions_adddup2(&actions, pipes[1], STDERR_FILENO);
    posix_spawn_file_actions_addclose(&actions, pipes[0]);
    posix_spawn_file_actions_addclose(&actions, pipes[1]);
    sigset_t signals, mask;
    sigemptyset(&signals); sigaddset(&signals, SIGINT); sigaddset(&signals, SIGTERM); sigaddset(&signals, SIGPIPE);
    sigemptyset(&mask);
    posix_spawnattr_setsigdefault(&attrs, &signals); posix_spawnattr_setsigmask(&attrs, &mask);
    posix_spawnattr_setpgroup(&attrs, 0);
    posix_spawnattr_setflags(&attrs, POSIX_SPAWN_SETPGROUP | POSIX_SPAWN_SETSIGDEF | POSIX_SPAWN_SETSIGMASK);
    pid_t pid = 0;
    int rc = posix_spawnp(&pid, argv[0], &actions, &attrs, argv, environ);
    posix_spawnattr_destroy(&attrs); posix_spawn_file_actions_destroy(&actions); close(pipes[1]);
    if (rc) {
        close(pipes[0]); if (log) fclose(log);
        snprintf(error, size, "cannot execute %s: %s", argv[0], strerror(rc)); return 0;
    }
    fcntl(pipes[0], F_SETFL, O_NONBLOCK);
    double started = now(), stopping = 0; int status = 0, exited = 0, eof = 0, failed = 0;
    size_t length = 0;
    if (capture) *capture = NULL;
    while (!exited || !eof) {
        double elapsed = now() - started;
        if ((interrupted || failed || (timeout > 0 && elapsed > timeout)) && stopping == 0) {
            kill(-pid, SIGTERM); stopping = now();
            if (!interrupted && !failed) { snprintf(error, size, "%s timed out", argv[0]); failed = 1; }
        }
        if (stopping > 0 && now() - stopping > 2.0) kill(-pid, SIGKILL);
        struct pollfd ready = {.fd = pipes[0], .events = POLLIN};
        if (!eof) poll(&ready, 1, 100);
        else { struct timespec delay = {0, 10000000}; nanosleep(&delay, NULL); }
        for (;;) {
            char buffer[8192]; ssize_t n = read(pipes[0], buffer, sizeof(buffer));
            if (!n) { eof = 1; break; }
            if (n < 0) {
                if (errno != EAGAIN && errno != EWOULDBLOCK && errno != EINTR) { failed = 1; eof = 1; }
                break;
            }
            if (log && fwrite(buffer, 1, (size_t)n, log) != (size_t)n) failed = 1;
            if (live) { fwrite(buffer, 1, (size_t)n, stderr); fflush(stderr); }
            if (capture && !failed) {
                if (length + (size_t)n > 2*1024*1024) { snprintf(error, size, "media probe output is too large"); failed = 1; }
                else {
                    char *grown = realloc(*capture, length + (size_t)n + 1);
                    if (!grown) { failed = 1; snprintf(error, size, "out of memory"); }
                    else { *capture = grown; memcpy(grown + length, buffer, (size_t)n); length += (size_t)n; grown[length] = 0; }
                }
            }
        }
        if (!exited) {
            pid_t waited = waitpid(pid, &status, WNOHANG);
            if (waited == pid) exited = 1;
            else if (waited < 0 && errno != EINTR) { exited = 1; failed = 1; }
        }
        if (stopping > 0 && exited && now() - stopping > 2.5) { kill(-pid, SIGKILL); break; }
    }
    /* A terminated leader may have left children holding GPU/media resources. */
    if (stopping > 0) kill(-pid, SIGKILL);
    close(pipes[0]);
    if (log && fclose(log)) failed = 1;
    if (interrupted) { snprintf(error, size, "render interrupted by signal %d", (int)interrupted); return 0; }
    if (failed || !WIFEXITED(status) || WEXITSTATUS(status)) {
        if (!*error) snprintf(error, size, "%s failed%s%s", argv[0], log_path ? "; log: " : "", log_path ? log_path : "");
        return 0;
    }
    return 1;
}
static int directory(const char *path, int fresh, char *error, size_t size) {
    char *parent = strdup(path);
    if (!parent) return 0;
    for (char *p = parent + 1; *p; p++) if (*p == '/') {
        *p = 0;
        if (mkdir(parent, 0755) && errno != EEXIST) { free(parent); goto fail; }
        *p = '/';
    }
    free(parent);
    if (!mkdir(path, 0755) || (!fresh && errno == EEXIST)) return 1;
fail:
    snprintf(error, size, "cannot create fresh output directory %s: %s", path, strerror(errno)); return 0;
}
static int write_text(const char *path, const char *text, char *error, size_t size) {
    FILE *f = fopen(path, "wb");
    if (!f) { snprintf(error, size, "cannot write %s: %s", path, strerror(errno)); return 0; }
    int ok = fwrite(text, 1, strlen(text), f) == strlen(text);
    if (fclose(f)) ok = 0;
    if (!ok) snprintf(error, size, "cannot write %s", path);
    return ok;
}
static int publish_video(const char *source, const char *target, char *error, size_t size) {
    int status;
#ifdef __APPLE__
    status = renamex_np(source, target, RENAME_EXCL);
#else
    status = (int)syscall(SYS_renameat2, AT_FDCWD, source, AT_FDCWD, target, RENAME_NOREPLACE);
#endif
    if (status) snprintf(error, size, "cannot publish video %s without overwriting an existing file: %s", target, strerror(errno));
    return status == 0;
}
/* Check the destination filesystem's actual name equivalence before rendering.
 * APFS, for example, can alias differently encoded Unicode names. Dry runs
 * stay read-only; this check runs in the fresh output's private work directory. */
static int check_names(const h3_render_plan *p, const sj_value *report, char *error, size_t size) {
    char *folder = h3_render_format("%s/name-check", p->work);
    if (!directory(folder, 1, error, size)) { free(folder); return 0; }
    const sj_value *collections[] = {sj_get(report, "segments"), sj_get(report, "merged")};
    int ok = 1;
    for (size_t group = 0; group < 2 && ok; group++) {
        for (size_t i = 0; i < collections[group]->count; i++) {
            const sj_value *row = collections[group]->items[i];
            if (!strcmp(sj_field(row, "status"), "skipped")) continue;
            char *name = h3_render_format("%s/%s", folder, sj_field(row, "name"));
            int fd = open(name, O_WRONLY | O_CREAT | O_EXCL | O_CLOEXEC, 0600);
            if (fd < 0) {
                snprintf(error, size, "output filename collides or is unavailable on this filesystem: %s: %s", sj_field(row, "name"), strerror(errno));
                ok = 0;
            } else close(fd);
            free(name); if (!ok) break;
        }
    }
    for (size_t group = 0; group < 2; group++) for (size_t i = 0; i < collections[group]->count; i++) {
        char *name = h3_render_format("%s/%s", folder, sj_field(collections[group]->items[i], "name"));
        unlink(name); free(name);
    }
    rmdir(folder); free(folder); return ok;
}
static int report_write(const h3_render_plan *p, const sj_value *report, char *error, size_t size) {
    char *temporary = h3_render_format("%s/run.json.tmp", p->output), *target = h3_render_format("%s/run.json", p->output);
    char *text = sj_dump(report);
    int ok = text && write_text(temporary, text, error, size);
    if (ok && rename(temporary, target)) { snprintf(error, size, "cannot publish run.json: %s", strerror(errno)); ok = 0; }
    free(text); free(target); free(temporary); return ok;
}
static double number(const sj_value *object, const char *key) {
    const sj_value *v = sj_get(object, key); double result = -1;
    if (v && v->text) lj_number(v->text, &result);
    return result;
}
typedef struct {
    double scan_timeout;
    int verify_frames;
} probe_options;
static int probe_options_read(probe_options *options, char *error, size_t size) {
    options->scan_timeout = 600;
    options->verify_frames = 0;
    const char *timeout = getenv("H3_RENDER_SCAN_TIMEOUT"), *verify = getenv("H3_RENDER_VERIFY_FRAMES");
    if (timeout && *timeout && (!lj_number(timeout, &options->scan_timeout) || options->scan_timeout <= 0)) {
        snprintf(error, size, "H3_RENDER_SCAN_TIMEOUT must be a positive finite number of seconds"); return 0;
    }
    if (verify && *verify) {
        if (strcmp(verify, "0") && strcmp(verify, "1")) {
            snprintf(error, size, "H3_RENDER_VERIFY_FRAMES must be 0 or 1"); return 0;
        }
        options->verify_frames = !strcmp(verify, "1");
    }
    return 1;
}
static const sj_value *media_stream(const sj_value *root, const char *kind) {
    const sj_value *streams = sj_get(root, "streams");
    if (streams && streams->type == LJ_ARRAY) for (size_t i = 0; i < streams->count; i++) {
        const char *type = sj_field(streams->items[i], "codec_type");
        if (type && !strcmp(type, kind)) return streams->items[i];
    }
    return NULL;
}
static sj_value *probe_media(const char *probe, const char *path, int scan, double timeout,
                            const char *log, char *error, size_t size) {
    char *metadata[] = {(char *)probe, "-v", "error", "-show_streams", "-of", "json", (char *)path, NULL};
    /* Only decode video when the container has no frame count or verification
     * was requested. Counting frames without selecting a stream decodes audio too. */
    char *frames[] = {(char *)probe, "-v", "error", "-select_streams", "v:0", "-count_frames",
        "-show_entries", "stream=codec_type,nb_read_frames", "-of", "json", (char *)path, NULL};
    char *text = NULL, reason[1024] = {0};
    int ok = process(scan ? frames : metadata, log, 0, &text, timeout, reason, sizeof(reason));
    sj_value *root = ok && text ? sj_parse(text, strlen(text), reason, sizeof(reason)) : NULL;
    free(text);
    if (!root) snprintf(error, size, "%s: %s: %s", path, scan ? "frame scan" : "metadata probe",
                        *reason ? reason : "no stream metadata returned");
    return root;
}
static sj_value *inspect(const char *probe, const char *path, int64_t frames, int width, int height,
                         const probe_options *options, const char *log, char *error, size_t size) {
    /* MP4/MOV sample tables already contain the frame count. Reading that
     * metadata avoids decoding every clip, prepared input and complete merge. */
    sj_value *root = probe_media(probe, path, 0, 120, log, error, size);
    if (!root) return NULL;
    const sj_value *video = media_stream(root, "video"), *audio = media_stream(root, "audio");
    const char *fps = video ? sj_field(video, "avg_frame_rate") : NULL;
    double numerator = 0, denominator = 1;
    if (fps) sscanf(fps, "%lf/%lf", &numerator, &denominator);
    if (!video || !audio || number(video, "width") != width || number(video, "height") != height ||
        denominator <= 0 || numerator / denominator != 24 ||
        fabs(number(video, "duration") - (double)frames / 24.0) > .1 ||
        fabs(number(audio, "duration") - (double)frames / 24.0) > .1) goto invalid;
    double stored = number(video, "nb_frames");
    int missing = stored <= 0 || floor(stored) != stored;
    if (!missing && stored != (double)frames) goto invalid;
    if (missing || options->verify_frames) {
        fprintf(stderr, "h3cli: %s: %s; scanning video frames (timeout %.3g s)\n", path,
                missing ? "container frame count unavailable" : "full frame verification requested", options->scan_timeout);
        sj_value *scanned = probe_media(probe, path, 1, options->scan_timeout, log, error, size);
        if (!scanned) { sj_free(root); return NULL; }
        double counted = number(media_stream(scanned, "video"), "nb_read_frames");
        sj_free(scanned);
        if (counted != (double)frames) goto invalid;
    }
    sj_value *result = sj_clone(video); sj_free(root); return result;
invalid:
    snprintf(error, size, "%s: expected %dx%d, %lld frames at 24 fps with synchronized audio", path, width, height, (long long)frames);
    sj_free(root); return NULL;
}
static int signature_equal(const sj_value *a, const sj_value *b) {
    const char *keys[] = {"codec_name", "profile", "pix_fmt", "width", "height", "color_range",
                          "color_space", "color_transfer", "color_primaries", "sample_aspect_ratio"};
    for (size_t i = 0; i < sizeof(keys)/sizeof(*keys); i++) {
        const sj_value *x = sj_get(a, keys[i]), *y = sj_get(b, keys[i]);
        const char *xs = x && x->text ? x->text : "", *ys = y && y->text ? y->text : "";
        if (strcmp(xs, ys)) return 0;
    }
    return 1;
}
static int merge(const h3_render_plan *p, size_t merge_index, sj_value *row, sj_value **videos,
                 const char *ffmpeg, const char *probe, const probe_options *options, char *error, size_t size) {
    const sj_value *inputs = sj_get(row, "segments");
    size_t *indices = calloc(inputs->count, sizeof(*indices));
    if (!indices) { snprintf(error, size, "out of memory"); return 0; }
    int normalize = 0; int64_t frames = 0;
    for (size_t j = 0; j < inputs->count; j++) {
        for (size_t i = 0; i < p->count; i++) if (!strcmp(p->segments[i].name, sj_text(inputs->items[j]))) { indices[j] = i; break; }
        frames += p->segments[indices[j]].delivered;
        if (j && !signature_equal(videos[indices[0]], videos[indices[j]])) normalize = 1;
    }
    const h3_render_segment *first = &p->segments[indices[0]];
    char *listing = strdup("ffconcat version 1.0\n");
    char *list_path = h3_render_format("%s/merge-%04zu.txt", p->work, merge_index);
    char *partial = h3_render_format("%s/merge-%04zu.mp4", p->work, merge_index);
    char *target = h3_render_format("%s/%s", p->output, sj_field(row, "name"));
    int ok = 0;
    if (normalize) fprintf(stderr, "h3cli: normalizing %s to %dx%d lossless RGB for merging\n", sj_field(row, "name"), first->width, first->height);
    for (size_t j = 0; j < inputs->count; j++) {
        const h3_render_segment *s = &p->segments[indices[j]];
        char *source = h3_render_format("%s/%s", p->output, s->name);
        char *name = h3_render_format("merge-%04zu-%04zu.mov", merge_index, j);
        char *prepared = h3_render_format("%s/%s", p->work, name);
        char *log = h3_render_format("%s/merge-%04zu-%04zu.log", p->work, merge_index, j);
        char *audio = h3_render_format("aresample=32000,apad=whole_len=%lld,atrim=end_sample=%lld,asetpts=PTS-STARTPTS",
                                       (long long)s->delivered*32000/24, (long long)s->delivered*32000/24);
        char *canvas = h3_render_format("scale=%d:%d:force_original_aspect_ratio=decrease:force_divisible_by=2,pad=%d:%d:(ow-iw)/2:(oh-ih)/2,setsar=1",
                                        first->width, first->height, first->width, first->height);
        char *argv[64]; size_t n = 0;
#define ARG(x) argv[n++] = (char *)(x)
        ARG(ffmpeg); ARG("-nostdin"); ARG("-hide_banner"); ARG("-loglevel"); ARG("warning"); ARG("-n");
        ARG("-i"); ARG(source); ARG("-map"); ARG("0:v:0"); ARG("-map"); ARG("0:a:0"); ARG("-map_metadata"); ARG("-1");
        ARG("-c:v"); ARG(normalize ? "libx264rgb" : "copy");
        if (normalize) {
            ARG("-crf"); ARG("0"); ARG("-preset"); ARG("fast"); ARG("-pix_fmt"); ARG("rgb24");
            if (s->width != first->width || s->height != first->height) { ARG("-vf"); ARG(canvas); }
        }
        ARG("-af"); ARG(audio); ARG("-c:a"); ARG("pcm_s16le"); ARG("-ar"); ARG("32000"); ARG("-ac"); ARG("2");
        ARG("-video_track_timescale"); ARG("24000"); ARG(prepared); argv[n] = NULL;
#undef ARG
        int prepared_ok = process(argv, log, 0, NULL, 0, error, size);
        sj_value *verified = prepared_ok ? inspect(probe, prepared, s->delivered, first->width, first->height, options, NULL, error, size) : NULL;
        if (verified) {
            char *next = h3_render_format("%sfile '%s'\nduration %.12f\n", listing, name, (double)s->delivered / 24.0);
            free(listing); listing = next;
        }
        free(source); free(name); free(prepared); free(log); free(audio); free(canvas);
        if (!verified) goto done;
        sj_free(verified);
    }
    if (!write_text(list_path, listing, error, size)) goto done;
    char *log = h3_render_format("%s/merge-%04zu.log", p->work, merge_index);
    char *audio = h3_render_format("aresample=32000:async=1:first_pts=0,apad,atrim=duration=%.12f", (double)frames / 24.0);
    char *argv[] = {(char *)ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "warning", "-n", "-f", "concat", "-safe", "1",
        "-i", list_path, "-map", "0:v:0", "-map", "0:a:0", "-map_metadata", "-1", "-c:v", "copy", "-af", audio,
        "-c:a", "aac", "-b:a", "192k", "-movflags", "+faststart", partial, NULL};
    int merged = process(argv, log, 0, NULL, 0, error, size); free(log); free(audio);
    sj_value *verified = merged ? inspect(probe, partial, frames, first->width, first->height, options, NULL, error, size) : NULL;
    if (!verified) goto done;
    sj_free(verified);
    if (interrupted || !publish_video(partial, target, error, size)) goto done;
    sj_set(row, "status", sj_string("complete")); sj_add(row, "video", sj_string(target));
    sj_add(row, "frames", sj_int(frames)); sj_add(row, "seconds", sj_real((double)frames / 24.0));
    sj_add(row, "lossless_normalization", sj_bool(normalize));
    sj_add(row, "width", sj_int(first->width)); sj_add(row, "height", sj_int(first->height));
    fprintf(stderr, "h3cli: merged video: %s\n", target); ok = 1;
done:
    free(indices); free(listing); free(list_path); free(partial); free(target); return ok;
}
static int preflight(const h3_render_plan *p, const char *ffmpeg, const char *probe,
                     const probe_options *options, char *error, size_t size) {
    char *path = h3_render_format("%s/probe.mp4", p->work), *log = h3_render_format("%s/preflight.log", p->work);
    char *argv[] = {(char *)ffmpeg, "-nostdin", "-hide_banner", "-loglevel", "error", "-n", "-f", "lavfi", "-i",
        "color=c=gray:s=96x96:r=24:d=0.25", "-f", "lavfi", "-i", "anullsrc=r=32000:cl=stereo", "-t", "0.25",
        "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", path, NULL};
    int ok = process(argv, log, 0, NULL, 30, error, size);
    sj_value *video = ok ? inspect(probe, path, 6, 96, 96, options, NULL, error, size) : NULL;
    ok = video != NULL; sj_free(video); free(path); free(log); return ok;
}
int h3_render_run(const h3_request *request, const char *binary, int (*validate)(const h3_request *)) {
    h3_render_plan plan = {0}; sj_value *report = NULL, **videos = NULL, *running = NULL;
    char error[2048] = {0}; int status = 2, created = 0, signals_set = 0;
    struct sigaction old_int, old_term, action = {0};
    double started = now();
    if (!h3_render_plan_read(&plan, request, error, sizeof(error))) goto done;
    for (size_t i = 0; i < plan.count; i++) {
        if (plan.segments[i].skipped || !validate) continue;
        h3_request segment = {0};
        int valid = h3_render_request(&plan, i, &segment, error, sizeof(error)) && !validate(&segment);
        h3_request_free(&segment);
        if (!valid) { if (!*error) snprintf(error, sizeof(error), "%s:%d: invalid generation options", plan.source, plan.segments[i].line); goto done; }
    }
    report = h3_render_plan_json(&plan, binary);
    if (!report) { snprintf(error, sizeof(error), "cannot allocate render report"); goto done; }
    if (plan.dry_run) { sj_write(stdout, report); putchar('\n'); status = 0; goto done; }
    size_t active = 0;
    for (size_t i = 0; i < plan.count; i++) active += !plan.segments[i].skipped;
    if (!active) { puts("All segments skipped; nothing to render."); status = 0; goto done; }
    probe_options options;
    if (!probe_options_read(&options, error, sizeof(error))) goto done;
    const char *ffmpeg = getenv("H3_FFMPEG"), *probe = getenv("H3_FFPROBE");
    if (!ffmpeg || !*ffmpeg) ffmpeg = "ffmpeg";
    if (!probe || !*probe) probe = "ffprobe";
    sj_value *tools = sj_object(); sj_add(tools, "h3cli", sj_string(binary));
    sj_add(tools, "ffmpeg", sj_string(ffmpeg)); sj_add(tools, "ffprobe", sj_string(probe)); sj_add(report, "tools", tools);
    sj_value *validation = sj_object();
    sj_add(validation, "verify_frames", sj_bool(options.verify_frames));
    sj_add(validation, "scan_timeout_seconds", sj_real(options.scan_timeout));
    sj_add(report, "validation", validation);
    interrupted = 0; action.sa_handler = stop; sigemptyset(&action.sa_mask);
    sigaction(SIGINT, &action, &old_int); sigaction(SIGTERM, &action, &old_term); signals_set = 1;
    if (!directory(plan.output, 1, error, sizeof(error))) goto done;
    created = 1; status = 1;
    if (!directory(plan.work, 1, error, sizeof(error))) goto done;
    char *source = h3_render_format("%s/source.txt", plan.work);
    int saved = write_text(source, plan.source_text, error, sizeof(error)); free(source);
    if (!saved) goto done;
    sj_set(report, "status", sj_string("running"));
    if (!report_write(&plan, report, error, sizeof(error))) goto done;
    if (!check_names(&plan, report, error, sizeof(error))) goto done;
    if (!preflight(&plan, ffmpeg, probe, &options, error, sizeof(error))) goto done;
    videos = calloc(plan.count, sizeof(*videos));
    if (!videos) { snprintf(error, sizeof(error), "out of memory"); goto done; }
    sj_value *segments = (sj_value *)sj_get(report, "segments");
    size_t completed = 0;
    for (size_t i = 0; i < plan.count; i++) {
        const h3_render_segment *s = &plan.segments[i]; if (s->skipped) continue;
        running = segments->items[i]; sj_set(running, "status", sj_string("running"));
        if (!report_write(&plan, report, error, sizeof(error))) goto done;
        fprintf(stderr, "h3cli: segment %zu/%zu: %s (%d delivered frames)\n", ++completed, active, s->name, s->delivered);
        const sj_value *command = sj_get(running, "command");
        char **argv = calloc(command->count + 1, sizeof(*argv));
        if (!argv) { snprintf(error, sizeof(error), "out of memory"); goto done; }
        for (size_t j = 0; j < command->count; j++) argv[j] = (char *)sj_text(command->items[j]);
        char *log = h3_render_format("%s/%04zu.log", plan.work, i);
        int rendered = process(argv, log, 1, NULL, 0, error, sizeof(error)); free(argv); free(log);
        if (!rendered) goto done;
        char *partial = h3_render_format("%s/%04zu.mp4", plan.work, i);
        videos[i] = inspect(probe, partial, s->delivered, s->width, s->height, &options, NULL, error, sizeof(error));
        char *state = h3_render_format("%s/%04zu.h3av", plan.work, i), *sidecar = h3_render_format("%s.presentation", state);
        struct stat st;
        int state_ok = !stat(state, &st) && S_ISREG(st.st_mode) && st.st_size > 0 && !stat(sidecar, &st) && S_ISREG(st.st_mode) && st.st_size > 0;
        sj_add(running, "state", sj_string(state)); free(state); free(sidecar);
        if (!videos[i] || !state_ok || interrupted || !publish_video(partial, sj_field(running, "video"), error, sizeof(error))) {
            if (!*error) snprintf(error, sizeof(error), "segment %s did not produce a complete clip and continuation state", s->name);
            free(partial); goto done;
        }
        free(partial); sj_set(running, "status", sj_string("complete")); running = NULL;
        if (!report_write(&plan, report, error, sizeof(error))) goto done;
    }
    sj_value *merges = (sj_value *)sj_get(report, "merged");
    for (size_t i = 0; i < merges->count; i++) {
        running = merges->items[i]; sj_set(running, "status", sj_string("running"));
        if (!report_write(&plan, report, error, sizeof(error)) ||
            !merge(&plan, i, running, videos, ffmpeg, probe, &options, error, sizeof(error))) goto done;
        running = NULL;
        if (!report_write(&plan, report, error, sizeof(error))) goto done;
    }
    status = 0;
done:
    if (signals_set && interrupted) status = 128 + interrupted;
    if (created && report) {
        const char *state = status >= 128 ? "interrupted" : status ? "failed" : "complete";
        sj_set(report, "status", sj_string(state));
        sj_add(report, "elapsed_seconds", sj_real(now() - started));
        if (*error) sj_add(report, "error", sj_string(error));
        if (running) sj_set(running, "status", sj_string(state));
        char report_error[2048] = {0};
        if (!report_write(&plan, report, report_error, sizeof(report_error))) {
            fprintf(stderr, "h3cli: render: %s\n", report_error); if (!status) status = 1;
        }
    }
    if (signals_set) { sigaction(SIGINT, &old_int, NULL); sigaction(SIGTERM, &old_term, NULL); }
    if (*error) fprintf(stderr, "h3cli: render: %s\n", error);
    if (!status && created) fprintf(stderr, "h3cli: render complete: %s\n", plan.output);
    if (videos) { for (size_t i = 0; i < plan.count; i++) sj_free(videos[i]); free(videos); }
    sj_free(report); h3_render_plan_free(&plan); return status;
}
