#include "src/request.h"
#include <stdlib.h>
#include <string.h>
void h3_request_free(h3_request *r) {
    if (!r)
        return;
    for (size_t i = 0; i < r->count; i++) {
        free(r->values[i].value);
        free(r->values[i].second);
    }
    free(r->values);memset(r,0,sizeof(*r));
}
const h3_option_value *h3_request_get(const h3_request *r,const char *name) {
    for(size_t i=r->count;i>0;i--)if(!strcmp(r->values[i-1].option->name,name))return &r->values[i-1];
    return NULL;
}
static void remove_at(h3_request *r,size_t i) {
    free(r->values[i].value);free(r->values[i].second);
    memmove(r->values+i,r->values+i+1,(r->count-i-1)*sizeof(*r->values));r->count--;
}
void h3_request_remove(h3_request *r,const char *name) {
    for(size_t i=0;i<r->count;)if(!strcmp(r->values[i].option->name,name))remove_at(r,i);else i++;
}
void h3_request_remove_group(h3_request *r,const char *group) {
    for(size_t i=0;i<r->count;)if(!strcmp(r->values[i].option->group,group))remove_at(r,i);else i++;
}
int h3_request_add(h3_request *r,const char *name,const char *value,const char *second,
                   h3_option_source source,char *error,size_t size) {
    const h3_option_descriptor *o=h3_option_find(name);
    if(!o){snprintf(error,size,"unknown option --%.128s",name);return 0;}
    if(!h3_option_validate(o,value,error,size))return 0;
    if(o->arity==2&&(!second||!*second)){snprintf(error,size,"--%s requires two paths",name);return 0;}
    if(r->count>=512){snprintf(error,size,"too many request options");return 0;}
    if(r->count==r->capacity){size_t cap=r->capacity?r->capacity*2:32;
        h3_option_value *p=realloc(r->values,cap*sizeof(*p));if(!p)goto oom;r->values=p;r->capacity=cap;}
    h3_option_value v={.option=o,.source=source};
    if(value&&!(v.value=strdup(value)))goto oom;
    if(second&&!(v.second=strdup(second))){free(v.value);goto oom;}
    r->values[r->count++]=v;return 1;
oom:snprintf(error,size,"out of memory");return 0;
}
int h3_request_parse_argv(h3_request *r,int argc,char *const *argv,h3_option_source source,char *error,size_t size) {
    for(int i=0;i<argc;i++){
        const char *arg=argv[i],*value=NULL;const h3_option_descriptor *o=NULL;
        if(!strncmp(arg,"--",2)){
            const char *eq=strchr(arg+2,'=');size_t n=eq?(size_t)(eq-arg-2):strlen(arg+2);char name[128];
            if (!n || n >= sizeof(name))
                goto unknown;
            memcpy(name, arg + 2, n);
            name[n] = 0;
            o = h3_option_find(name);
            if (eq)
                value = eq + 1;
        }else if(arg[0]=='-'&&arg[1]){o=h3_option_short(arg[1]);if(arg[2])value=arg+2;}
        if(!o)goto unknown;
        if(!o->arity&&value){snprintf(error,size,"--%s takes no value",o->name);return 0;}
        if(o->arity&&!value){if(++i>=argc){snprintf(error,size,"--%s needs a value",o->name);return 0;}value=argv[i];}
        const char *second=NULL;if(o->arity==2){if(++i>=argc){snprintf(error,size,"--%s needs VIDEO AUDIO",o->name);return 0;}second=argv[i];}
        if(!h3_request_add(r,o->name,value,second,source,error,size))return 0;
        continue;
unknown:snprintf(error,size,"unknown option or unexpected operand at token %d: %.128s",i,arg);return 0;
    }return 1;
}
void h3_argv_free(char **argv){if(argv){for(size_t i=0;argv[i];i++)free(argv[i]);free(argv);}}
char **h3_request_argv(const h3_request *r,int *argc) {
    char **v=calloc(2+r->count*3,sizeof(*v));if(!v)return NULL;size_t n=0;
    v[n++]=strdup("h3cli");if(!v[0])goto fail;
    for(size_t i=0;i<r->count;i++){
        const h3_option_value *x=&r->values[i];
        if (asprintf(&v[n], "--%s", x->option->name) < 0)
            goto fail;
        n++;
        if(x->value){v[n++]=strdup(x->value);if(!v[n-1])goto fail;}
        if(x->second){v[n++]=strdup(x->second);if(!v[n-1])goto fail;}
    }*argc=(int)n;return v;
fail:h3_argv_free(v);return NULL;
}
const char *h3_request_operation(const h3_request *r){
    const char *names[]={"inspect-upscale-state","upscale-state","resume-sampler-state","decode-av-state","decode-still-latent","still","info","help"};
    const char *ops[]={"inspect_upscale","upscale","resume","decode_av","decode_still","still","info","help"};
    for(size_t i=0;i<sizeof(names)/sizeof(names[0]);i++)if(h3_request_get(r,names[i]))return ops[i];
    return "generate";
}
