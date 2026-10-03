#ifndef H3_CLI_DISPATCH_H
#define H3_CLI_DISPATCH_H
#include "src/h3.h"
#include "src/models/models.h"
typedef struct {
    int validate_only;
    int prepare_only;
    int (*model_progress)(const h3_model_progress *progress, void *opaque);
    int (*progress)(const char *phase,int completed,int total,void *opaque);
    int (*frame)(const h3_frame *frame,void *opaque);
    void (*result)(const h3_result *result,void *opaque);
    void *opaque;
} h3_cli_hooks;
/* Shared one-shot parser/preflight/dispatcher. Used in fresh worker processes;
 * neither it nor GPU initialization runs on HTTP handler threads. */
int h3_cli_run(int argc,char **argv,const h3_cli_hooks *hooks);
#endif
