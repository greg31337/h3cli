#ifndef H3_MODELS_INTERNAL_H
#define H3_MODELS_INTERNAL_H
#include "src/models/models.h"
#include "src/digest.h"
#include <pthread.h>
#include <stdatomic.h>
#include <sys/stat.h>

typedef struct {
    const h3_model_plan *plan;
    h3_model_callback callback;
    void *opaque;
    pthread_mutex_t mutex;
    atomic_int stopped;
    uint64_t downloaded, reused, total;
    double began, last_event;
    char error[1024];
    size_t next;
    unsigned char *verified;
} model_job;
double model_now(void);
int model_fail(model_job *job, const char *format, ...);
int model_progress(model_job *job, const char *phase, const char *component,
                   uint64_t downloaded, uint64_t reused, int force);
void model_digest(const void *data, size_t length, char out[65]);
int model_hash_fd(int fd, const h3_model_artifact *artifact, model_job *job,
                   char hash[65]);
int model_parent(const char *absolute, int create, char **name,
                  char *error, size_t size);
int model_write_atomic(int dir, const char *name, const void *data, size_t length);
int model_transfer(model_job *job, const h3_model_artifact *artifact, int fd,int metadata_dir,const char *part);
#endif
