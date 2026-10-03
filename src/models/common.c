#include "src/models/internal.h"
#include "src/platform.h"
#include <curl/curl.h>
#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <stdarg.h>
#include <stdlib.h>
#include <string.h>
#include <time.h>
#include <unistd.h>

#pragma GCC diagnostic ignored "-Wdeprecated-declarations"

static pthread_once_t curl_once = PTHREAD_ONCE_INIT;
static CURLcode curl_status;
static void initialize_curl(void) { curl_status = curl_global_init(CURL_GLOBAL_DEFAULT); }
int h3_models_global_init(char *error, size_t size) {
    pthread_once(&curl_once, initialize_curl);
    if (curl_status == CURLE_OK) return 1;
    snprintf(error, size, "cannot initialize model HTTPS transport");
    return 0;
}
int h3_models_offline(void) {
    const char *value = getenv("H3_OFFLINE");
    return value && !strcmp(value, "1");
}
double model_now(void) {
    struct timespec now;
    clock_gettime(CLOCK_MONOTONIC, &now);
    return (double)now.tv_sec + (double)now.tv_nsec / 1e9;
}
int model_fail(model_job *job, const char *format, ...) {
    pthread_mutex_lock(&job->mutex);
    if (!job->error[0]) {
        va_list args;
        va_start(args, format);
        vsnprintf(job->error, sizeof(job->error), format, args);
        va_end(args);
    }
    atomic_store(&job->stopped, 1);
    pthread_mutex_unlock(&job->mutex);
    return 0;
}
int model_progress(model_job *job, const char *phase, const char *component,
                   uint64_t downloaded, uint64_t reused, int force) {
    pthread_mutex_lock(&job->mutex);
    job->downloaded += downloaded;
    job->reused += reused;
    double now = model_now();
    if (job->callback && (force || now - job->last_event >= .2)) {
        h3_model_progress event = {phase, component, job->downloaded, job->total,
                                   job->reused, now - job->began};
        job->last_event = now;
        if (job->callback(&event, job->opaque)) atomic_store(&job->stopped, 1);
    }
    int ok = !atomic_load(&job->stopped);
    pthread_mutex_unlock(&job->mutex);
    return ok;
}
static void digest_hex(const uint8_t raw[32], char text[65]) {
    for (int i = 0; i < 32; i++) snprintf(text + 2*i, 3, "%02x", raw[i]);
}
void model_digest(const void *data, size_t length, char out[65]) {
    h3_sha256_ctx hash;
    h3_sha256_init_fast(&hash);
    const unsigned char *p = data;
    while (length) {
        size_t n = length > 1048576 ? 1048576 : length;
        h3_sha256_update(&hash, p, (h3_sha256_size)n);
        p += n; length -= n;
    }
    uint8_t raw[32];
    h3_sha256_final(raw, &hash);
    digest_hex(raw, out);
}
int model_hash_fd(int fd, const h3_model_artifact *artifact, model_job *job,
                   char hash[65]) {
    struct stat before, after;
    if (fstat(fd, &before) || !S_ISREG(before.st_mode) || before.st_size < 0 ||
        (uint64_t)before.st_size != artifact->bytes || lseek(fd, 0, SEEK_SET) < 0)
        return 0;
    unsigned char *buffer = malloc(4u*1024u*1024u);
    if (!buffer) return 0;
    h3_sha256_ctx ctx;
    h3_sha256_init_fast(&ctx);
    uint64_t total = 0;
    int ok = 1;
    for (;;) {
        ssize_t got = read(fd, buffer, 4u*1024u*1024u);
        if (got < 0 && errno == EINTR) continue;
        if (got < 0) { ok = 0; break; }
        if (!got) break;
        total += (uint64_t)got;
        h3_sha256_update(&ctx, buffer, (h3_sha256_size)got);
        if (job && !model_progress(job, "verifying_models", artifact->path, 0, 0, 0)) {
            ok = 0; break;
        }
    }
    free(buffer);
    uint8_t raw[32];
    h3_sha256_final(raw, &ctx);
    digest_hex(raw, hash);
    if (fstat(fd, &after)) return 0;
    struct timespec bm = h3_stat_mtime(&before), am = h3_stat_mtime(&after);
    struct timespec bc = h3_stat_ctime(&before), ac = h3_stat_ctime(&after);
    if (before.st_dev != after.st_dev ||
        before.st_ino != after.st_ino || before.st_size != after.st_size ||
        bm.tv_sec != am.tv_sec || bm.tv_nsec != am.tv_nsec ||
        bc.tv_sec != ac.tv_sec || bc.tv_nsec != ac.tv_nsec)
        ok = 0;
    return ok && total == artifact->bytes;
}

/* Resolve the existing prefix without creating it. The returned absolute path
 * has no symlink components, including for a not-yet-created model root. */
char *h3_models_absolute(const char *path, char *error, size_t size) {
    if (!path || !*path) { snprintf(error, size, "model path must not be empty"); return NULL; }
    char *full = NULL;
    if (path[0] == '/') full = strdup(path);
    else {
        char *cwd = getcwd(NULL, 0);
        if (cwd && asprintf(&full, "%s/%s", cwd, path) < 0) full = NULL;
        free(cwd);
    }
    if (!full || strlen(full) >= PATH_MAX) {
        free(full); snprintf(error, size, "model path is too long or cannot be allocated"); return NULL;
    }
    size_t length = strlen(full);
    while (length > 1 && full[length-1] == '/') full[--length] = 0;
    char *resolved = realpath(full, NULL);
    if (resolved) { free(full); return resolved; }
    char *prefix = strdup(full);
    if (!prefix) { free(full); return NULL; }
    for (;;) {
        struct stat st;
        if (!lstat(prefix, &st) || errno != ENOENT) break;
        char *slash = strrchr(prefix, '/');
        if (!slash) break;
        if (slash == prefix) prefix[1] = 0;
        else *slash = 0;
        resolved = realpath(prefix, NULL);
        if (resolved) {
            const char *suffix = full + strlen(prefix);
            if (*suffix == '/') suffix++;
            char *check = strdup(suffix), *save = NULL;
            int safe = check != NULL;
            for (char *part = check ? strtok_r(check, "/", &save) : NULL;
                 part; part = strtok_r(NULL, "/", &save)) {
                if (!strcmp(part, ".") || !strcmp(part, "..")) safe = 0;
            }
            free(check);
            char *joined = NULL;
            if (safe && asprintf(&joined, "%s%s%s", resolved,
                                  strcmp(resolved, "/") ? "/" : "", suffix) < 0) joined = NULL;
            free(resolved); resolved = joined;
            break;
        }
        if (!strcmp(prefix, "/")) break;
    }
    free(prefix); free(full);
    if (!resolved) snprintf(error, size, "model path is inaccessible, dangling or has an unresolved parent: %.200s", path);
    return resolved;
}

/* O_NOFOLLOW on every resolved component also protects against substitution
 * between resolution and publication. Caller owns parent fd and leaf name. */
int model_parent(const char *absolute, int create, char **name, char *error, size_t size) {
    *name = NULL;
    if (!absolute || *absolute != '/') return -1;
    char *copy = strdup(absolute + 1);
    if (!copy) return -1;
    int dir = open("/", O_RDONLY|O_DIRECTORY|O_CLOEXEC);
    char *save = NULL, *part = strtok_r(copy, "/", &save);
    while (dir >= 0 && part) {
        char *next = strtok_r(NULL, "/", &save);
        if (!strcmp(part, ".") || !strcmp(part, "..")) { close(dir); dir = -1; break; }
        if (!next) { *name = strdup(part); break; }
        if (create && mkdirat(dir, part, !strcmp(part, ".h3cli-downloads") ? 0700 : 0755) && errno != EEXIST) {
            close(dir); dir = -1; break;
        }
        int child = openat(dir, part, O_RDONLY|O_DIRECTORY|O_NOFOLLOW|O_CLOEXEC);
        close(dir); dir = child; part = next;
    }
    free(copy);
    if (dir < 0 || !*name) {
        if (dir >= 0) close(dir);
        free(*name); *name = NULL;
        snprintf(error, size, "cannot access model destination parent: %.240s (%s)", absolute, strerror(errno));
        return -1;
    }
    return dir;
}
int model_write_atomic(int dir, const char *name, const void *data, size_t length) {
    static _Atomic unsigned serial;
    char temp[160];
    int fd = -1;
    for (unsigned tries = 0; tries < 100 && fd < 0; tries++) {
        snprintf(temp, sizeof(temp), "%s.tmp-%ld-%u", name, (long)getpid(), atomic_fetch_add(&serial, 1));
        fd = openat(dir, temp, O_WRONLY|O_CREAT|O_EXCL|O_NOFOLLOW|O_CLOEXEC, 0600);
        if (fd < 0 && errno != EEXIST) break;
    }
    if (fd < 0) return 0;
    size_t at = 0;
    while (at < length) {
        ssize_t n = write(fd, (const char *)data + at, length - at);
        if (n < 0 && errno == EINTR) continue;
        if (n <= 0) break;
        at += (size_t)n;
    }
    int ok = at == length && !fsync(fd);
    if (close(fd)) ok = 0;
    if (ok) ok = !renameat(dir, temp, dir, name) && !fsync(dir);
    if (!ok) unlinkat(dir, temp, 0);
    return ok;
}
