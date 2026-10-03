#ifndef H3_PLATFORM_H
#define H3_PLATFORM_H
#include <sys/stat.h>
#include <time.h>
static inline struct timespec h3_stat_mtime(const struct stat *s) {
#ifdef __APPLE__
    return s->st_mtimespec;
#else
    return s->st_mtim;
#endif
}
static inline struct timespec h3_stat_ctime(const struct stat *s) {
#ifdef __APPLE__
    return s->st_ctimespec;
#else
    return s->st_ctim;
#endif
}
static inline struct timespec h3_stat_atime(const struct stat *s) {
#ifdef __APPLE__
    return s->st_atimespec;
#else
    return s->st_atim;
#endif
}
#ifdef __clang__
#define H3_COMPILER_ID ";clang=" __clang_version__
#else
#define H3_COMPILER_ID ";gcc=" __VERSION__
#endif
char *h3_executable_path(void);
#endif
