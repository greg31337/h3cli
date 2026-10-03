#include "src/server/json.h"
#include <errno.h>
#include <inttypes.h>
#include <math.h>
#include <stdlib.h>
#include <string.h>
static sj_value *value(lj_type type,const char *text){sj_value *v=calloc(1,sizeof(*v));if(!v)return NULL;v->type=type;
    if(text&&!(v->text=strdup(text))){free(v);return NULL;}return v;}
sj_value *sj_object(void){return value(LJ_OBJECT,NULL);}
sj_value *sj_array(void){return value(LJ_ARRAY,NULL);}
sj_value *sj_null(void){return value(LJ_NULL,NULL);}
sj_value *sj_string(const char *s){return s?value(LJ_STRING,s):sj_null();}
sj_value *sj_bool(int n){sj_value *v=value(LJ_BOOL,n?"true":"false");if(v)v->number=!!n;return v;}
static sj_value *number(const char *s){sj_value *v=value(LJ_NUMBER,s);if(v)lj_number(s,&v->number);return v;}
sj_value *sj_int(int64_t n){char b[32];snprintf(b,sizeof(b),"%"PRId64,n);return number(b);}
sj_value *sj_uint(uint64_t n){char b[32];snprintf(b,sizeof(b),"%"PRIu64,n);return number(b);}
sj_value *sj_real(double n){char b[64];if(!isfinite(n))return NULL;snprintf(b,sizeof(b),"%.17g",n);return number(b);}
int sj_add(sj_value *p,const char *key,sj_value *v){
    if(!p||!v||(p->type!=LJ_ARRAY&&p->type!=LJ_OBJECT)){sj_free(v);return 0;}
    if(p->type==LJ_OBJECT&&(!key||sj_get(p,key))){sj_free(v);return 0;}
    if(key){v->key=strdup(key);if(!v->key){sj_free(v);return 0;}}
    sj_value **items=realloc(p->items,(p->count+1)*sizeof(*items));if(!items){sj_free(v);return 0;}
    p->items=items;p->items[p->count++]=v;return 1;
}
int sj_set(sj_value *p,const char *key,sj_value *v){
    if(p&&p->type==LJ_OBJECT)for(size_t i=0;i<p->count;i++)if(!strcmp(p->items[i]->key,key)){
                if (!v)
                    return 0;
                v->key = strdup(key);
                if (!v->key) {
                    sj_free(v);
                    return 0;
                }
                sj_free(p->items[i]);
                p->items[i] = v;
                return 1;
            }
    return sj_add(p,key,v);
}
int sj_write(FILE *f,const sj_value *v){
    if(!v)return 0;
    if(v->type==LJ_STRING)return lj_quote(f,v->text);
    if(v->type==LJ_BOOL)return fputs(v->number!=0?"true":"false",f)>=0;
    if(v->type==LJ_NUMBER)return v->text&&fprintf(f,"%s",v->text)>=0;
    if(v->type==LJ_NULL)return fputs("null",f)>=0;
    int object=v->type==LJ_OBJECT;if(fputc(object?'{':'[',f)==EOF)return 0;
    for(size_t i=0;i<v->count;i++){if(i&&fputc(',',f)==EOF)return 0;
        if(object&&(!lj_quote(f,v->items[i]->key)||fputc(':',f)==EOF))return 0;
        if(!sj_write(f,v->items[i]))return 0;}
    return fputc(object?'}':']',f)!=EOF;
}
char *sj_dump(const sj_value *v){char *s=NULL;size_t n=0;FILE *f=open_memstream(&s,&n);if(!f)return NULL;
    int ok=sj_write(f,v);if(fclose(f))ok=0;if(!ok){free(s);return NULL;}return s;}
sj_value *sj_clone(const sj_value *v){char *s=sj_dump(v);if(!s)return NULL;char error[128];sj_value *copy=sj_parse(s,strlen(s),error,sizeof(error));free(s);return copy;}
const char *sj_text(const sj_value *v){return v&&v->type==LJ_STRING?v->text:NULL;}
const char *sj_field(const sj_value *v,const char *key){return sj_text(sj_get(v,key));}
int sj_u64(const sj_value *v,uint64_t *out){
    if(!v||v->type!=LJ_NUMBER||!v->text||!*v->text)return 0;
    for(const char *p=v->text;*p;p++)if(*p<'0'||*p>'9')return 0;
    char *end;errno=0;unsigned long long n=strtoull(v->text,&end,10);if(errno||*end)return 0;*out=(uint64_t)n;return 1;
}
int sj_i64(const sj_value *v,int64_t *out){
    if(!v||v->type!=LJ_NUMBER||!v->text||!*v->text)return 0;
    const char *p=v->text;if(*p=='-')p++;if(!*p)return 0;for(;*p;p++)if(*p<'0'||*p>'9')return 0;
    char *end;errno=0;long long n=strtoll(v->text,&end,10);if(errno||*end)return 0;*out=(int64_t)n;return 1;
}

static int compare_keys(const void *a,const void *b){const sj_value *x=*(sj_value *const *)a,*y=*(sj_value *const *)b;return strcmp(x->key,y->key);}
void sj_sort(sj_value *v){if(!v)return;for(size_t i=0;i<v->count;i++)sj_sort(v->items[i]);if(v->type==LJ_OBJECT&&v->count>1)qsort(v->items,v->count,sizeof(*v->items),compare_keys);}
