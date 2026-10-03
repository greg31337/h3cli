#ifndef H3_TEST_TEACHER_H
#define H3_TEST_TEACHER_H
#include <stddef.h>
/* Test-only CPU Euler input replacement. Both files validate before either
 * destination changes. step is the zero-based next evaluation (1..5). */
int h3_test_teacher_load(const char *directory,int step,float *video,size_t nv,
                         float *audio,size_t na,char *error,size_t size);
/* Explicit SGLang reference replay also imports the first stochastic input
 * and permits the complete 50-evaluation qualification schedule. */
int h3_test_teacher_load_sglang(const char *directory,int step,float *video,size_t nv,
                               float *audio,size_t na,char *error,size_t size);
#endif
