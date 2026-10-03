#include "src/memory.h"
#include "src/media/ffmpeg.h"
#include "src/sglang/sglang.h"

#include <errno.h>
#include <fcntl.h>
#include <limits.h>
#include <pthread.h>
#include <poll.h>
#include <signal.h>
#include <spawn.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <sys/wait.h>
#include <time.h>
#include <unistd.h>

extern char **environ;

static const char *ffmpeg_program(void) {
    const char *override = getenv("H3_FFMPEG");
    return override && *override ? override : "ffmpeg";
}

static const char *ffprobe_program(void) {
    const char *override = getenv("H3_FFPROBE");
    return override && *override ? override : "ffprobe";
}

static void fail(char *error, size_t error_size, const char *format, ...) {
    if (!error || !error_size) return;
    va_list arguments;
    va_start(arguments, format);
    vsnprintf(error, error_size, format, arguments);
    va_end(arguments);
}

static int make_parents(const char *path, char *error, size_t error_size) {
    char *copy = strdup(path);
    if (!copy) {
        fail(error, error_size, "out of memory resolving output directory");
        return 0;
    }
    for (char *cursor = copy + 1; *cursor; cursor++) {
        if (*cursor != '/') continue;
        *cursor = '\0';
        if (mkdir(copy, 0755) != 0 && errno != EEXIST) {
            fail(error, error_size, "cannot create output directory %s: %s",
                 copy, strerror(errno));
            free(copy);
            return 0;
        }
        *cursor = '/';
    }
    free(copy);
    return 1;
}

int h3_ffprobe_visual_size(const char *path, int *width, int *height,
                           char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (width) *width = 0;
    if (height) *height = 0;
    if (!path || !*path || !width || !height) {
        fail(error, error_size, "invalid FFprobe visual-size arguments");
        return 0;
    }
    int stream[2];
    if (pipe(stream) != 0) {
        fail(error, error_size, "cannot create FFprobe pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffprobe", "-v", "error", "-select_streams", "v:0",
        "-show_entries", "stream=width,height", "-of", "csv=p=0:s=x",
        (char *)path, NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffprobe_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        fail(error, error_size, "cannot start FFprobe: %s", strerror(code));
        return 0;
    }
    char output[128];
    size_t received = 0;
    int overflow = 0;
    while (1) {
        char byte;
        ssize_t amount = read(stream[0], &byte, 1);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        if (received + 1 < sizeof(output)) output[received++] = byte;
        else overflow = 1;
    }
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        fail(error, error_size, "cannot wait for FFprobe: %s", strerror(errno));
        return 0;
    }
    output[received] = '\0';
    int parsed_width = 0, parsed_height = 0, consumed = 0;
    if (overflow || !WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        sscanf(output, "%dx%d%n", &parsed_width, &parsed_height, &consumed) != 2) {
        fail(error, error_size, "FFprobe could not inspect visual stream %s",
             path);
        return 0;
    }
    for (char *cursor = output + consumed; *cursor; cursor++) {
        if (*cursor != ' ' && *cursor != '\t' && *cursor != '\r' &&
            *cursor != '\n') {
            fail(error, error_size, "FFprobe returned an invalid visual size");
            return 0;
        }
    }
    if (parsed_width < 1 || parsed_height < 1) {
        fail(error, error_size, "visual stream has invalid dimensions %dx%d",
             parsed_width, parsed_height);
        return 0;
    }
    *width = parsed_width;
    *height = parsed_height;
    return 1;
}

int h3_ffmpeg_read_image_f32(const char *path, int width, int height,
                             h3_image_fit fit, float **pixels,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (!path || !*path || !pixels || width < 1 || height < 1 ||
        (fit != H3_IMAGE_FIT_STRETCH && fit != H3_IMAGE_FIT_COVER)) {
        fail(error, error_size, "invalid FFmpeg image input arguments");
        return 0;
    }
    if ((size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3 ||
        area * 3 > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded image size overflows");
        return 0;
    }
    size_t bytes = area * 3;
    uint8_t *rgb = malloc(bytes);
    float *channel_major = malloc(bytes * sizeof(*channel_major));
    if (!rgb || !channel_major) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "out of memory decoding input image");
        return 0;
    }
    char filter[256];
    if (fit == H3_IMAGE_FIT_STRETCH) {
        snprintf(filter, sizeof(filter), "scale=%d:%d:flags=lanczos",
                 width, height);
    } else {
        snprintf(filter, sizeof(filter),
                 "scale=%d:%d:force_original_aspect_ratio=increase:flags=lanczos,"
                 "crop=%d:%d", width, height, width, height);
    }
    int stream[2];
    if (pipe(stream) != 0) {
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot create FFmpeg image pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-frames:v", "1", "-vf", filter,
        "-f", "rawvideo", "-pix_fmt", "rgb24", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t received = 0;
    while (received < bytes) {
        ssize_t amount = read(stream[0], rgb + received, bytes - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(rgb);
        free(channel_major);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received != bytes || trailing != 0) {
        free(rgb);
        free(channel_major);
        if (!WIFEXITED(status) || WEXITSTATUS(status) != 0)
            fail(error, error_size, "FFmpeg could not decode image %s (status %d)",
                 path, WIFEXITED(status) ? WEXITSTATUS(status) : -1);
        else
            fail(error, error_size,
                 "FFmpeg decoded %zu bytes for %dx%d image, expected %zu",
                 received + (trailing > 0 ? 1u : 0u), width, height, bytes);
        return 0;
    }
    const float scale = 1.0f / 255.0f;
    const int reference=h3_sglang_requested();
    for (size_t pixel = 0; pixel < area; pixel++) {
        channel_major[pixel] = reference ? (float)rgb[3 * pixel]/255.f : (float)rgb[3 * pixel] * scale;
        channel_major[area + pixel] = reference ? (float)rgb[3 * pixel + 1]/255.f : (float)rgb[3 * pixel + 1] * scale;
        channel_major[2 * area + pixel] = reference ? (float)rgb[3 * pixel + 2]/255.f : (float)rgb[3 * pixel + 2] * scale;
    }
    free(rgb);
    *pixels = channel_major;
    return 1;
}

static int read_video_f32(const char *path, int width, int height,
                             int max_frames, float **pixels, int *frames,
                             char *error, size_t error_size) {
    if (error && error_size) error[0] = '\0';
    if (pixels) *pixels = NULL;
    if (frames) *frames = 0;
    if (!path || !*path || !pixels || !frames || width < 1 || height < 1 ||
        max_frames < 1 || (size_t)width > SIZE_MAX / (size_t)height) {
        fail(error, error_size, "invalid FFmpeg video input arguments");
        return 0;
    }
    size_t area = (size_t)width * (size_t)height;
    if (area > SIZE_MAX / 3) {
        fail(error, error_size, "decoded video frame size overflows");
        return 0;
    }
    size_t frame_bytes = area * 3;
    if ((size_t)max_frames > SIZE_MAX / frame_bytes) {
        fail(error, error_size, "decoded video size overflows");
        return 0;
    }
    size_t capacity = (size_t)max_frames * frame_bytes;
    if (capacity > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "converted video size overflows");
        return 0;
    }
    size_t float_bytes = capacity * sizeof(float);
    if (float_bytes > UINT64_MAX - frame_bytes) {
        fail(error, error_size, "reference video allocation reserve overflows");
        return 0;
    }
    if (!h3_memory_check((uint64_t)float_bytes + frame_bytes,
                         "reference video decode allocation", error, error_size))
        return 0;
    float *channel_major = malloc(float_bytes);
    uint8_t *rgb = malloc(frame_bytes);
    if (!channel_major || !rgb) {
        free(channel_major); free(rgb);
        fail(error, error_size, "out of memory decoding input video");
        return 0;
    }
    char filter[256], frame_limit[32];
    snprintf(filter, sizeof(filter),
             "fps=24,scale=%d:%d:flags=lanczos,setsar=1", width, height);
    snprintf(frame_limit, sizeof(frame_limit), "%d", max_frames);
    int stream[2];
    if (pipe(stream) != 0) {
        free(channel_major); free(rgb);
        fail(error, error_size, "cannot create FFmpeg video pipe: %s",
             strerror(errno));
        return 0;
    }
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:v:0", "-an", "-vf", filter,
        "-frames:v", frame_limit, "-f", "rawvideo", "-pix_fmt", "rgb24",
        "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    int actions_ready = !code;
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, ffmpeg_program(), &actions, NULL,
                                    arguments, environ);
    if (actions_ready) posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(channel_major); free(rgb);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    int frame_count = 0, ok = 1;
    const float scale = 1.0f / 255.0f;
    const int reference=h3_sglang_requested();
    while (frame_count < max_frames && ok) {
        size_t received = 0;
        while (received < frame_bytes) {
            /* A stalled child must not prevent a dynamic memory cancellation. */
            if (!h3_memory_check(0, "reference video decode", error, error_size)) {
                ok = 0;
                break;
            }
            struct pollfd ready = {stream[0], POLLIN, 0};
            int polled = poll(&ready, 1, 100);
            if (polled < 0 && errno == EINTR) continue;
            if (polled == 0) continue;
            if (polled < 0) {
                fail(error, error_size, "cannot poll FFmpeg video pipe: %s", strerror(errno));
                ok = 0;
                break;
            }
            ssize_t amount = read(stream[0], rgb + received, frame_bytes - received);
            if (amount < 0 && errno == EINTR) continue;
            if (amount < 0) {
                fail(error, error_size, "cannot read FFmpeg video pipe: %s", strerror(errno));
                ok = 0;
                break;
            }
            if (amount == 0) {
                if (received) {
                    fail(error, error_size, "FFmpeg returned a partial video frame for %s", path);
                    ok = 0;
                }
                break;
            }
            received += (size_t)amount;
        }
        if (!ok || received == 0) break;
        for (size_t pixel = 0; pixel < area; pixel++)
            for (size_t channel = 0; channel < 3; channel++) {
                size_t destination = (channel * (size_t)max_frames +
                                      (size_t)frame_count) * area + pixel;
                channel_major[destination] = reference ? (float)rgb[pixel * 3 + channel]/255.f : (float)rgb[pixel * 3 + channel] * scale;
            }
        frame_count++;
    }
    /* Verify EOF with at most one sentinel byte, as the old bounded reader did.
     * Never decode or retain another frame after max_frames. */
    while (ok && frame_count == max_frames) {
        if (!h3_memory_check(0, "reference video decode completion", error, error_size)) {
            ok = 0; break;
        }
        struct pollfd ready = {stream[0], POLLIN, 0};
        int polled = poll(&ready, 1, 100);
        if (polled < 0 && errno == EINTR) continue;
        if (!polled) continue;
        uint8_t extra;
        ssize_t trailing = polled < 0 ? -1 : read(stream[0], &extra, 1);
        if (trailing < 0 && errno == EINTR) continue;
        if (trailing != 0) {
            fail(error, error_size, "FFmpeg could not decode bounded video %s", path);
            ok = 0;
        }
        break;
    }
    close(stream[0]);
    int status = 0, cancellation_waits = 0;
    pid_t waited;
    for (;;) {
        if (ok) ok = h3_memory_check(0, "reference video decode completion", error, error_size);
        if (!ok && !cancellation_waits) kill(child, SIGTERM);
        if (!ok && ++cancellation_waits == 50) kill(child, SIGKILL);
        waited = waitpid(child, &status, WNOHANG);
        if (waited < 0 && errno == EINTR) continue;
        if (waited != 0) break;
        struct timespec pause = {0, 20000000};
        nanosleep(&pause, NULL);
    }
    if (ok && (waited < 0 || !WIFEXITED(status) || WEXITSTATUS(status) != 0)) {
        fail(error, error_size, "FFmpeg could not decode bounded video %s", path);
        ok = 0;
    }
    if (ok && frame_count < 1) {
        fail(error, error_size,
             "normalized reference video has no decoded frames");
        ok = 0;
    }
    if (ok) ok = h3_memory_check(0, "after reference video decode", error, error_size);
    free(rgb);
    if (!ok) { free(channel_major); return 0; }
    /* Compact the channel strides for short clips.
     * Keep the allocation capacity: realloc could temporarily duplicate F32. */
    if (frame_count != max_frames)
        for (size_t channel = 1; channel < 3; channel++)
            memmove(channel_major + channel * (size_t)frame_count * area,
                    channel_major + channel * (size_t)max_frames * area,
                    (size_t)frame_count * area * sizeof(*channel_major));
    *pixels = channel_major;
    *frames = frame_count;
    return 1;
}

int h3_ffmpeg_read_normalized_video_f32(const char *path, int width, int height,
    int max_frames, float **pixels, int *frames, char *error, size_t error_size) {
    return read_video_f32(path, width, height, max_frames, pixels, frames, error, error_size);
}

static int read_audio_rate(const char *path, int max_samples,
                             int truncate_at_limit,
                             float **pcm, int *samples,
                             char *error, size_t error_size,int AUDIO_RATE,int MIN_SAMPLES,
                             const char *program) {
    enum { AUDIO_CHANNELS = 2 };
    if (error && error_size) error[0] = '\0';
    if (pcm) *pcm = NULL;
    if (samples) *samples = 0;
    if (!path || !*path || !pcm || !samples || max_samples < MIN_SAMPLES ||
        max_samples > AUDIO_RATE * 15 ||
        (truncate_at_limit != 0 && truncate_at_limit != 1)) {
        fail(error, error_size, "invalid FFmpeg audio input arguments");
        return 0;
    }
    size_t elements = (size_t)max_samples * AUDIO_CHANNELS;
    if (elements > SIZE_MAX / sizeof(float)) {
        fail(error, error_size, "decoded audio size overflows");
        return 0;
    }
    float *interleaved = malloc(elements * sizeof(*interleaved));
    if (!interleaved) {
        fail(error, error_size, "out of memory decoding reference audio");
        return 0;
    }
    char duration[64];
    double seconds = (double)max_samples / (double)AUDIO_RATE;
    if (!truncate_at_limit) seconds += 1.0 / (double)AUDIO_RATE;
    snprintf(duration, sizeof(duration), "%.9f", seconds);
    int stream[2];
    if (pipe(stream) != 0) {
        free(interleaved);
        fail(error, error_size, "cannot create FFmpeg audio pipe: %s",
             strerror(errno));
        return 0;
    }
    char rate[24];snprintf(rate,sizeof(rate),"%d",AUDIO_RATE);
    char *arguments[] = {
        "ffmpeg", "-v", "error", "-i", (char *)path,
        "-map", "0:a:0", "-vn", "-ac", "2", "-ar", rate,
        "-t", duration, "-f", "f32le", "pipe:1", NULL
    };
    posix_spawn_file_actions_t actions;
    int code = posix_spawn_file_actions_init(&actions);
    if (!code) code = posix_spawn_file_actions_adddup2(
        &actions, stream[1], STDOUT_FILENO);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[0]);
    if (!code) code = posix_spawn_file_actions_addclose(&actions, stream[1]);
    pid_t child = -1;
    if (!code) code = posix_spawnp(&child, program, &actions, NULL,
                                    arguments, environ);
    posix_spawn_file_actions_destroy(&actions);
    close(stream[1]);
    if (code) {
        close(stream[0]);
        free(interleaved);
        fail(error, error_size, "cannot start FFmpeg: %s", strerror(code));
        return 0;
    }
    size_t capacity = elements * sizeof(*interleaved);
    size_t received = 0;
    while (received < capacity) {
        ssize_t amount = read(stream[0], (uint8_t *)interleaved + received,
                              capacity - received);
        if (amount < 0 && errno == EINTR) continue;
        if (amount <= 0) break;
        received += (size_t)amount;
    }
    uint8_t extra;
    ssize_t trailing;
    do trailing = read(stream[0], &extra, 1);
    while (trailing < 0 && errno == EINTR);
    close(stream[0]);
    int status = 0;
    while (waitpid(child, &status, 0) < 0) {
        if (errno == EINTR) continue;
        free(interleaved);
        fail(error, error_size, "cannot wait for FFmpeg: %s", strerror(errno));
        return 0;
    }
    size_t frame_bytes = AUDIO_CHANNELS * sizeof(float);
    if (!truncate_at_limit && trailing > 0) {
        free(interleaved);
        fail(error, error_size,
             "reference audio exceeds the 15 second total limit");
        return 0;
    }
    if (!WIFEXITED(status) || WEXITSTATUS(status) != 0 ||
        received % frame_bytes) {
        free(interleaved);
        fail(error, error_size,
             "FFmpeg could not decode a stereo soundtrack from %s", path);
        return 0;
    }
    int sample_count = (int)(received / frame_bytes);
    if (sample_count < MIN_SAMPLES) {
        free(interleaved);
        fail(error, error_size,
             "reference audio requires at least 2 seconds at 32 kHz");
        return 0;
    }
    size_t output_elements = (size_t)sample_count * AUDIO_CHANNELS;
    float *channel_major = malloc(output_elements * sizeof(*channel_major));
    if (!channel_major) {
        free(interleaved);
        fail(error, error_size, "out of memory converting reference audio");
        return 0;
    }
    for (int sample = 0; sample < sample_count; sample++)
        for (int channel = 0; channel < AUDIO_CHANNELS; channel++)
            channel_major[(size_t)channel * (size_t)sample_count +
                          (size_t)sample] =
                interleaved[(size_t)sample * AUDIO_CHANNELS +
                            (size_t)channel];
    free(interleaved);
    *pcm = channel_major;
    *samples = sample_count;
    return 1;
}

int h3_ffmpeg_write_rgb24(const char *path, const uint8_t *frames,
    int count, int width, int height, int fps, const h3_output_encoding *encoding,
    char *error, size_t size) {
    h3_ffmpeg_writer *w=h3_ffmpeg_writer_open(path,count,width,height,fps,
        NULL,0,0,0,encoding,error,size);
    if (!w) return 0;
    if (!h3_ffmpeg_writer_write(w,frames,count,error,size)) {
        h3_ffmpeg_writer_abort(w); return 0;
    }
    return h3_ffmpeg_writer_finish(w,error,size);
}

typedef struct {
    int descriptor;
    const uint8_t *data;
    size_t bytes;
    int error;
} stream_writer;

static int close_action(posix_spawn_file_actions_t *actions, int descriptor,
                        int keep_a, int keep_b) {
    return descriptor == keep_a || descriptor == keep_b ? 0 :
           posix_spawn_file_actions_addclose(actions, descriptor);
}

struct h3_ffmpeg_writer {
    pid_t child;
    int descriptor, expected, written, audio_started;
    size_t frame_bytes;
    pthread_t audio_thread;
    stream_writer audio;
    float *pcm;
    char *path, *temporary;
};

/* Darwin suppresses pipe signals per descriptor. The portable path blocks
 * them on the writer thread. Neither changes the process-wide handler. */
static ssize_t safe_pipe_write(int fd,const void *data,size_t bytes) {
#ifdef F_SETNOSIGPIPE
    return write(fd,data,bytes);
#else
    sigset_t set,old,pending;sigemptyset(&set);sigaddset(&set,SIGPIPE);
    pthread_sigmask(SIG_BLOCK,&set,&old);sigpending(&pending);
    ssize_t result=write(fd,data,bytes);int saved=errno;
    /* A pipe closed during a large write can return a positive partial count
     * AND queue SIGPIPE (Darwin). Drain any newly generated signal before
     * restoring the caller's mask, including after such partial writes. */
    if(!sigismember(&pending,SIGPIPE)){
        sigset_t after;sigpending(&after);
        if(sigismember(&after,SIGPIPE)){int sig=0;sigwait(&set,&sig);}
    }
    pthread_sigmask(SIG_SETMASK,&old,NULL);errno=saved;return result;
#endif
}
static void *bounded_audio_thread(void *opaque) {
    stream_writer *w=opaque;size_t done=0;
    while(done<w->bytes){size_t n=w->bytes-done;if(n>65536)n=65536;
        ssize_t written=safe_pipe_write(w->descriptor,w->data+done,n);
        if(written<0&&errno==EINTR)continue;
        if(written<=0){w->error=written<0?errno:EIO;break;}done+=(size_t)written;}
    close(w->descriptor);w->descriptor=-1;return NULL;
}
static void writer_release(h3_ffmpeg_writer *w) {
    free(w->pcm);free(w->path);free(w->temporary);free(w);
}
void h3_ffmpeg_writer_abort(h3_ffmpeg_writer *w) {
    if(!w)return;
    if(w->child>0)kill(w->child,SIGKILL);
    if(w->descriptor>=0)close(w->descriptor);
    if(w->audio_started)pthread_join(w->audio_thread,NULL);
    else if(w->audio.descriptor>=0)close(w->audio.descriptor);
    if(w->child>0){int status;while(waitpid(w->child,&status,0)<0&&errno==EINTR){}}
    if (w->temporary)
        unlink(w->temporary);
    writer_release(w);
}
h3_ffmpeg_writer *h3_ffmpeg_writer_open(const char *path,int frames,int width,int height,int fps,
 const float *pcm,int samples,int channels,int rate,const h3_output_encoding *encoding,char *error,size_t size) {
    if(!path||!*path||frames<1||width<2||height<2||width%2||height%2||fps<1||
       (pcm&&(samples<1||channels<1||channels>8||rate<1))||
       (uint64_t)(unsigned)width*(unsigned)height>SIZE_MAX/3){fail(error,size,"invalid streaming media arguments");return NULL;}
    if(!h3_output_encoding_valid(encoding,error,size))return NULL;
    /* Only the immutable regression harness selects this profile. CUDA
     * arithmetic and delivery compression are independent in production. */
    const char *test=getenv("H3_TEST_REFERENCE_ENCODING");
    int reference=test && !strcmp(test,"1");
    if(reference && (!h3_sglang_requested() || h3_output_encoding_selected(encoding))) {
        fail(error,size,"recorded encoding requires CUDA reference arithmetic and no output overrides");return NULL;
    }
    int lossless=encoding && encoding->lossless_video;
    if(!make_parents(path,error,size))return NULL;
    h3_ffmpeg_writer *w=calloc(1,sizeof(*w));if(!w){fail(error,size,"out of memory creating media writer");return NULL;}
    w->descriptor=w->audio.descriptor=-1;w->expected=frames;w->frame_bytes=(size_t)width*(size_t)height*3;
    w->path=strdup(path);w->temporary=malloc(strlen(path)+32);
    if(!w->path||!w->temporary){fail(error,size,"out of memory creating media path");goto failed;}
    snprintf(w->temporary,strlen(path)+32,"%s.partial.XXXXXX.mp4",path);
    int temporary=mkstemps(w->temporary,4);if(temporary<0){fail(error,size,"cannot create temporary media: %s",strerror(errno));goto failed;}close(temporary);
    if(pcm){size_t elements=(size_t)samples*(size_t)channels;
        if(elements>SIZE_MAX/sizeof(float)){fail(error,size,"audio stream size overflows");goto failed;}
        w->pcm=malloc(elements*sizeof(float));if(!w->pcm){fail(error,size,"out of memory copying audio");goto failed;}
        for(int s=0;s<samples;s++)for(int c=0;c<channels;c++)w->pcm[(size_t)s*(size_t)channels+(size_t)c]=pcm[(size_t)c*(size_t)samples+(size_t)s];
        w->audio.data=(const uint8_t *)w->pcm;w->audio.bytes=elements*sizeof(float);
    }
    int vp[2]={-1,-1},ap[2]={-1,-1};
    if(pipe(vp)|| (pcm&&pipe(ap))){if(vp[0]>=0)close(vp[0]);if(vp[1]>=0)close(vp[1]);fail(error,size,"cannot create media pipes");goto failed;}
#ifdef F_SETNOSIGPIPE
    if(fcntl(vp[1],F_SETNOSIGPIPE,1)||(pcm&&fcntl(ap[1],F_SETNOSIGPIPE,1))){
        int saved=errno;close(vp[0]);close(vp[1]);if(ap[0]>=0)close(ap[0]);if(ap[1]>=0)close(ap[1]);
        fail(error,size,"cannot suppress media pipe signals: %s",strerror(saved));goto failed;
    }
#endif
    int target=vp[1]>vp[0]?vp[1]:vp[0];if(ap[0]>target)target=ap[0];if(ap[1]>target)target=ap[1];target++;
    char dimensions[64],fps_text[32],rate_text[32],channels_text[32],input[32];
    snprintf(dimensions,sizeof(dimensions),"%dx%d",width,height);snprintf(fps_text,sizeof(fps_text),"%d",fps);
    snprintf(rate_text,sizeof(rate_text),"%d",rate);snprintf(channels_text,sizeof(channels_text),"%d",channels);snprintf(input,sizeof(input),"pipe:%d",target);
    char *args[80];int n=0;
#define ARG(x) args[n++]=(char *)(x)
    ARG("ffmpeg");ARG("-y");ARG("-loglevel");ARG("error");ARG("-f");ARG("rawvideo");ARG("-pixel_format");ARG("rgb24");
    ARG("-video_size");ARG(dimensions);ARG("-framerate");ARG(fps_text);ARG("-i");ARG("pipe:0");
    if(pcm){ARG("-f");ARG("f32le");ARG("-ar");ARG(rate_text);ARG("-ac");ARG(channels_text);ARG("-i");ARG(input);}
    ARG("-map");ARG("0:v:0");if(pcm){ARG("-map");ARG("1:a:0");}
    char threads[32],crf[16];snprintf(crf,sizeof(crf),"%d",reference?25:h3_output_crf(encoding));
    ARG("-c:v");ARG(lossless?"libx264rgb":"libx264");ARG("-preset");ARG("fast");
    ARG("-crf");ARG(crf);ARG("-pix_fmt");ARG(lossless?"rgb24":"yuv420p");
    if(!reference) {
        ARG("-vf");ARG(lossless?
            "setparams=range=full:color_primaries=bt709:color_trc=bt709:colorspace=gbr":
            "scale=in_range=full:out_range=limited:out_color_matrix=bt709,setparams=range=limited:color_primaries=bt709:color_trc=bt709:colorspace=bt709");
        ARG("-color_range");ARG(lossless?"pc":"tv");
        ARG("-colorspace");ARG(lossless?"rgb":"bt709");
        ARG("-color_primaries");ARG("bt709");ARG("-color_trc");ARG("bt709");
    }
    if(reference){
        long cpus=sysconf(_SC_NPROCESSORS_ONLN);if(cpus<1)cpus=1;
        int count=(height+15)/32;if(count<1)count=1;if(count>128)count=128;
        if(count>cpus*3/2)count=(int)(cpus*3/2);
        snprintf(threads,sizeof(threads),"%d",count);ARG("-threads");ARG(threads);
    }
    if(pcm){ARG("-c:a");ARG("aac");if(!reference){ARG("-b:a");ARG("192k");}}
    if(!reference){ARG("-movflags");ARG("+faststart");}ARG(w->temporary);args[n]=NULL;
#undef ARG
    posix_spawn_file_actions_t actions;int code=posix_spawn_file_actions_init(&actions),initialized=!code;
    if(!code)code=posix_spawn_file_actions_adddup2(&actions,vp[0],STDIN_FILENO);
    if(!code&&pcm)code=posix_spawn_file_actions_adddup2(&actions,ap[0],target);
    int descriptors[]={vp[0],vp[1],ap[0],ap[1]};
    for(int i=0;i<4&&!code;i++)if(descriptors[i]>=0)code=close_action(&actions,descriptors[i],STDIN_FILENO,pcm?target:-1);
    if(!code)code=posix_spawnp(&w->child,ffmpeg_program(),&actions,NULL,args,environ);
    if (initialized)
        posix_spawn_file_actions_destroy(&actions);
    close(vp[0]);
    if (ap[0] >= 0)
        close(ap[0]);
    w->descriptor=vp[1];w->audio.descriptor=ap[1];
    if(code){fail(error,size,"cannot start streaming FFmpeg: %s",strerror(code));goto failed;}
    if(pcm){code=pthread_create(&w->audio_thread,NULL,bounded_audio_thread,&w->audio);
        if(code){fail(error,size,"cannot start audio writer: %s",strerror(code));goto failed;}w->audio_started=1;}
    return w;
failed:h3_ffmpeg_writer_abort(w);return NULL;
}
int h3_ffmpeg_writer_write(h3_ffmpeg_writer *w,const uint8_t *rgb,int frames,char *error,size_t size) {
    if(!w||!rgb||frames<1||frames>w->expected-w->written||
       (size_t)frames>SIZE_MAX/w->frame_bytes){fail(error,size,"invalid media batch/frame count");return 0;}
    size_t bytes=(size_t)frames*w->frame_bytes,done=0;
    while(done<bytes){size_t n=bytes-done;if(n>65536)n=65536;ssize_t count=safe_pipe_write(w->descriptor,rgb+done,n);
        if (count < 0 && errno == EINTR)
            continue;
        if (count <= 0) {
            fail(error, size, "streaming FFmpeg pipe failed: %s", strerror(errno));
            return 0;
        }
        done += (size_t)count;
    }
    w->written+=frames;return 1;
}
int h3_ffmpeg_writer_finish(h3_ffmpeg_writer *w,char *error,size_t size) {
    if(!w)return 0;
    if(w->written!=w->expected){fail(error,size,"incomplete video: %d/%d frames",w->written,w->expected);h3_ffmpeg_writer_abort(w);return 0;}
    close(w->descriptor);w->descriptor=-1;if(w->audio_started){pthread_join(w->audio_thread,NULL);w->audio_started=0;}
    int status=0;pid_t result;do result=waitpid(w->child,&status,0);while(result<0&&errno==EINTR);w->child=0;
    int ok=result>0&&WIFEXITED(status)&&WEXITSTATUS(status)==0&&!w->audio.error;
    if(!ok)fail(error,size,"streaming FFmpeg failed (status %d, audio error %d)",status,w->audio.error);
    if(ok&&rename(w->temporary,w->path)){ok=0;fail(error,size,"cannot publish completed media: %s",strerror(errno));}
    if (!ok)
        unlink(w->temporary);
    writer_release(w);
    return ok;
}

int h3_ffmpeg_write_av_rgb24_f32(const char *path, const uint8_t *frames,
    int count, int width, int height, int fps, const float *pcm, int samples,
    int channels, int rate, const h3_output_encoding *encoding, char *error, size_t size) {
    if (!pcm) { fail(error,size,"missing audio samples"); return 0; }
    h3_ffmpeg_writer *w=h3_ffmpeg_writer_open(path,count,width,height,fps,
        pcm,samples,channels,rate,encoding,error,size);
    if (!w) return 0;
    if (!h3_ffmpeg_writer_write(w,frames,count,error,size)) {
        h3_ffmpeg_writer_abort(w); return 0;
    }
    return h3_ffmpeg_writer_finish(w,error,size);
}

int h3_ffmpeg_write_png(const char *path,const uint8_t *rgb,int width,int height,
    char *error,size_t size) {
    if(!path || !*path || !rgb || width<1 || height<1 ||
        (uint64_t)(unsigned)width*(unsigned)height>UINT64_C(1032192)) {
        fail(error,size,"invalid still PNG arguments");return 0;
    }
    if(!make_parents(path,error,size))return 0;
    char *tmp=malloc(strlen(path)+20);
    if(!tmp){fail(error,size,"cannot allocate PNG staging path");return 0;}
    sprintf(tmp,"%s.part-XXXXXX",path);
    int fd=mkstemp(tmp);if(fd<0){free(tmp);fail(error,size,"cannot stage PNG");return 0;}close(fd);
    int pipes[2];if(pipe(pipes)){unlink(tmp);free(tmp);fail(error,size,"cannot create PNG pipe");return 0;}
#ifdef F_SETNOSIGPIPE
    if(fcntl(pipes[1],F_SETNOSIGPIPE,1)){close(pipes[0]);close(pipes[1]);unlink(tmp);free(tmp);return 0;}
#endif
    char dimensions[64];snprintf(dimensions,sizeof(dimensions),"%dx%d",width,height);
    char *args[]={"ffmpeg","-v","error","-y","-f","rawvideo","-pixel_format","rgb24",
        "-video_size",dimensions,"-i","pipe:0","-frames:v","1","-an","-c:v","png","-f","image2","-update","1",tmp,NULL};
    posix_spawn_file_actions_t actions;int code=posix_spawn_file_actions_init(&actions),initialized=!code;
    if(!code)code=posix_spawn_file_actions_adddup2(&actions,pipes[0],STDIN_FILENO);
    if(!code)code=posix_spawn_file_actions_addclose(&actions,pipes[0]);
    if(!code)code=posix_spawn_file_actions_addclose(&actions,pipes[1]);
    pid_t child=-1;if(!code)code=posix_spawnp(&child,ffmpeg_program(),&actions,NULL,args,environ);
    if (initialized)
        posix_spawn_file_actions_destroy(&actions);
    close(pipes[0]);
    int ok=!code;size_t at=0,bytes=(size_t)width*height*3;
    while(ok && at<bytes) {
        ssize_t n=safe_pipe_write(pipes[1],rgb+at,bytes-at>65536?65536:bytes-at);
        if(n<0 && errno==EINTR)continue;
        if(n<=0)ok=0;else at+=(size_t)n;
    }
    close(pipes[1]);int status=0;
    if(child>0) {pid_t got;do {got=waitpid(child,&status,0);}while(got<0 && errno==EINTR);
        ok=ok && got==child && WIFEXITED(status) && !WEXITSTATUS(status);}
    if(ok)ok=rename(tmp,path)==0;
    if(!ok){unlink(tmp);fail(error,size,"PNG encoding failed%s%s",code?": ":"",code?strerror(code):"");}
    free(tmp);return ok;
}

int h3_ffmpeg_read_audio_f32(const char *path,int max_samples,int truncate,float **pcm,int *samples,char *error,size_t size) {
    return read_audio_rate(path,max_samples,truncate,pcm,samples,error,size,32000,64000,ffmpeg_program());
}
int h3_ffmpeg_read_soundtrack_44100(const char *path,int max_samples,float **pcm,int *samples,char *error,size_t size) {
    /* SGLang reads reference audio with PATH FFmpeg, independently of the
     * imageio FFmpeg used for final delivery. AAC float decoding differs even
     * when both versions decode identical RGB. Keep the output H3_FFMPEG override intact. */
    const char *program=getenv("H3_SGLANG_INPUT_FFMPEG");
    return read_audio_rate(path,max_samples,1,pcm,samples,error,size,44100,1,
                           program&&*program?program:"ffmpeg");
}
