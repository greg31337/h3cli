#ifndef H3_REQUEST_H
#define H3_REQUEST_H
#include "src/cli/options.h"
#include <stdio.h>
typedef enum { H3_SOURCE_DEFAULT, H3_SOURCE_SGLANG, H3_SOURCE_NATIVE, H3_SOURCE_MANAGED } h3_option_source;
typedef struct {
    const h3_option_descriptor *option;
    char *value, *second;
    h3_option_source source;
} h3_option_value;
typedef struct {
    h3_option_value *values;
    size_t count, capacity;
} h3_request;
void h3_request_free(h3_request *request);
int h3_request_add(h3_request *request,const char *name,const char *value,
                   const char *second,h3_option_source source,char *error,size_t size);
const h3_option_value *h3_request_get(const h3_request *request,const char *name);
void h3_request_remove(h3_request *request,const char *name);
void h3_request_remove_group(h3_request *request,const char *group);
int h3_request_parse_argv(h3_request *request,int argc,char *const *argv,
                         h3_option_source source,char *error,size_t size);
int h3_request_parse_string(h3_request *request,const char *text,char *error,size_t size);
char **h3_request_argv(const h3_request *request,int *argc);
void h3_argv_free(char **argv);
const char *h3_request_operation(const h3_request *request);
#endif
