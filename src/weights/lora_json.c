#include "src/weights/lora_json.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <locale.h>
#include <math.h>
#include <errno.h>
#include <limits.h>
#include <fenv.h>
#include <stdint.h>
#if defined(__SSE__)
#include <xmmintrin.h>
#endif
#ifdef __APPLE__
#include <xlocale.h>
#endif
typedef struct {
    const char *at;
    const char *end;
    char *error;
    size_t error_size;
} lj_cursor;

static int lj_fail(lj_cursor *cursor, const char *message) {
    if (cursor->error && cursor->error_size) {
        snprintf(cursor->error, cursor->error_size, "%s", message);
    }
    return 0;
}

static void lj_ws(lj_cursor *cursor) {
    while (cursor->at < cursor->end &&
           (*cursor->at == ' ' || *cursor->at == '\n' ||
            *cursor->at == '\r' || *cursor->at == '\t')) cursor->at++;
}

static int lj_take(lj_cursor *cursor, char expected) {
    lj_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != expected) {
        return lj_fail(cursor, "malformed JSON");
    }
    cursor->at++;
    return 1;
}

static int lj_hex(char value) {
    if (value >= '0' && value <= '9') return value - '0';
    if (value >= 'a' && value <= 'f') return value - 'a' + 10;
    if (value >= 'A' && value <= 'F') return value - 'A' + 10;
    return -1;
}

static char *lj_string(lj_cursor *cursor) {
    lj_ws(cursor);
    if (cursor->at >= cursor->end || *cursor->at != '"') {
        lj_fail(cursor, "expected JSON string");
        return NULL;
    }
    cursor->at++;
    size_t maximum = (size_t)(cursor->end - cursor->at);
    char *result = malloc(maximum + 1);
    if (!result) {
        lj_fail(cursor, "out of memory parsing JSON");
        return NULL;
    }
    size_t length = 0;
    while (cursor->at < cursor->end && *cursor->at != '"') {
        unsigned char value = (unsigned char)*cursor->at++;
        if (value == '\\') {
            if (cursor->at >= cursor->end) goto malformed;
            value = (unsigned char)*cursor->at++;
            switch (value) {
                case '"': result[length++] = '"'; break;
                case '\\': result[length++] = '\\'; break;
                case '/': result[length++] = '/'; break;
                case 'b': result[length++] = '\b'; break;
                case 'f': result[length++] = '\f'; break;
                case 'n': result[length++] = '\n'; break;
                case 'r': result[length++] = '\r'; break;
                case 't': result[length++] = '\t'; break;
                case 'u': {
                    if (cursor->end - cursor->at < 4) goto malformed;
                    int codepoint = 0;
                    for (int index = 0; index < 4; index++) {
                        int digit = lj_hex(cursor->at[index]);
                        if (digit < 0) goto malformed;
                        codepoint = codepoint * 16 + digit;
                    }
                    cursor->at += 4;
                    if (codepoint >= 0xd800 && codepoint <= 0xdbff) {
                        if (cursor->end - cursor->at < 6 || cursor->at[0] != '\\' ||
                            cursor->at[1] != 'u') goto malformed;
                        int low = 0;
                        for (int index = 2; index < 6; index++) {
                            int digit = lj_hex(cursor->at[index]);
                            if (digit < 0) goto malformed;
                            low = low * 16 + digit;
                        }
                        if (low < 0xdc00 || low > 0xdfff) goto malformed;
                        cursor->at += 6;
                        codepoint = 0x10000 + ((codepoint - 0xd800) << 10) + low - 0xdc00;
                    } else if (codepoint >= 0xdc00 && codepoint <= 0xdfff) goto malformed;
                    /* Public tensor names/metadata use C strings. */
                    if (!codepoint) goto malformed;
                    if (codepoint < 0x80) {
                        result[length++] = (char)codepoint;
                    } else if (codepoint < 0x800) {
                        result[length++] = (char)(0xc0 | (codepoint >> 6));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    } else if (codepoint < 0x10000) {
                        result[length++] = (char)(0xe0 | (codepoint >> 12));
                        result[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    } else {
                        result[length++] = (char)(0xf0 | (codepoint >> 18));
                        result[length++] = (char)(0x80 | ((codepoint >> 12) & 0x3f));
                        result[length++] = (char)(0x80 | ((codepoint >> 6) & 0x3f));
                        result[length++] = (char)(0x80 | (codepoint & 0x3f));
                    }
                    break;
                }
                default: goto malformed;
            }
        } else {
            if (value < 0x20) goto malformed;
            result[length++] = (char)value;
            if (value >= 0x80) {
                int extra = value >= 0xc2 && value <= 0xdf ? 1 :
                            value >= 0xe0 && value <= 0xef ? 2 :
                            value >= 0xf0 && value <= 0xf4 ? 3 : -1;
                if (extra < 0 || cursor->end - cursor->at < extra) goto malformed;
                unsigned char first = (unsigned char)*cursor->at;
                if ((value == 0xe0 && first < 0xa0) || (value == 0xed && first >= 0xa0) ||
                    (value == 0xf0 && first < 0x90) || (value == 0xf4 && first >= 0x90)) goto malformed;
                for (int i = 0; i < extra; i++) {
                    unsigned char next = (unsigned char)*cursor->at++;
                    if ((next & 0xc0) != 0x80) goto malformed;
                    result[length++] = (char)next;
                }
            }
        }
    }
    if (cursor->at >= cursor->end) goto malformed;
    cursor->at++;
    result[length] = '\0';
    char *compact = realloc(result, length + 1);
    return compact ? compact : result;

malformed:
    free(result);
    lj_fail(cursor, "malformed JSON string escape");
    return NULL;
}


void lj_free(lj_value *v) {
    if (!v) return;
    for (size_t i=0;i<v->count;i++) lj_free(v->items[i]);
    free(v->items); free(v->key); free(v->text); free(v);
}
const lj_value *lj_get(const lj_value *v, const char *key) {
    if (v && v->type==LJ_OBJECT) for(size_t i=0;i<v->count;i++)
        if(!strcmp(v->items[i]->key,key)) return v->items[i];
    return NULL;
}
/* Deliberately accepts decimal syntax only, regardless of the process locale. */
int lj_number(const char *s, double *value) {
    const char *p=s;
    if(*p=='-' || *p=='+') p++;
    int digits=0;
    while(*p>='0'&&*p<='9') {p++;digits++;}
    if(*p=='.') {p++;while(*p>='0'&&*p<='9'){p++;digits++;}}
    if(!digits) return 0;
    if(*p=='e'||*p=='E') {p++;if(*p=='-'||*p=='+')p++;digits=0;
        while(*p>='0'&&*p<='9'){p++;digits++;} if(!digits)return 0;}
    if(*p)return 0;
    locale_t loc=newlocale(LC_NUMERIC_MASK,"C",(locale_t)0);
    if(!loc)return 0;
    fenv_t env;
    if(fegetenv(&env)||fesetround(FE_TONEAREST)){freelocale(loc);return 0;}
#if defined(__SSE__)
    unsigned mxcsr=_mm_getcsr();_mm_setcsr(mxcsr&~0x8040u);
#endif
#if defined(__aarch64__)
    uint64_t fpcr;__asm__ volatile("mrs %0, fpcr":"=r"(fpcr));
    uint64_t gradual=fpcr&~((UINT64_C(1)<<24)|(UINT64_C(1)<<19)|UINT64_C(3));
    __asm__ volatile("msr fpcr, %0"::"r"(gradual));
#endif
    char *end=NULL; errno=0; double n=strtod_l(s,&end,loc);
#if defined(__SSE__)
    _mm_setcsr(mxcsr);
#endif
#if defined(__aarch64__)
    __asm__ volatile("msr fpcr, %0"::"r"(fpcr));
#endif
    fesetenv(&env);
    freelocale(loc);
    if(!end || *end || !isfinite(n))return 0;
    *value=n;return 1;
}
static lj_value *lj_value_read(lj_cursor *c, int depth) {
    if(depth>64) {lj_fail(c,"JSON nesting exceeds 64");return NULL;}
    lj_ws(c); if(c->at==c->end)return NULL;
    lj_value *v=calloc(1,sizeof(*v)); if(!v)return NULL;
    char ch=*c->at;
    if(ch=='"') {v->type=LJ_STRING;v->text=lj_string(c);if(!v->text)goto bad;}
    else if(ch=='{'||ch=='[') {
        v->type=ch=='{'?LJ_OBJECT:LJ_ARRAY;char close=ch=='{'?'}':']'; c->at++;
        lj_ws(c);if(c->at<c->end&&*c->at==close){c->at++;return v;}
        for(;;) {
            char *key=NULL;
            if(v->type==LJ_OBJECT) {
                key=lj_string(c);
                if(!key||!lj_take(c,':')){free(key);goto bad;}
                if(lj_get(v,key)){free(key);lj_fail(c,"duplicate JSON key");goto bad;}
            }
            lj_value *child=lj_value_read(c,depth+1);
            if(!child){free(key);goto bad;}
            child->key=key;
            lj_value **next=realloc(v->items,(v->count+1)*sizeof(*next));
            if(!next){lj_free(child);goto bad;}
            v->items=next;v->items[v->count++]=child;
            lj_ws(c);if(c->at==c->end)goto bad;
            if(*c->at==close){c->at++;break;}
            if(!lj_take(c,','))goto bad;
        }
    } else if(ch=='t'||ch=='f'||ch=='n') {
        const char *lit=ch=='t'?"true":ch=='f'?"false":"null";size_t n=strlen(lit);
        if((size_t)(c->end-c->at)<n||memcmp(c->at,lit,n))goto bad;
        c->at+=n;v->type=ch=='n'?LJ_NULL:LJ_BOOL;v->number=ch=='t';
    } else {
        const char *start=c->at;
        if(*c->at=='-')c->at++;
        if(c->at==c->end||*c->at<'0'||*c->at>'9')goto bad;
        if(*c->at=='0')c->at++;else while(c->at<c->end&&*c->at>='0'&&*c->at<='9')c->at++;
        if(c->at<c->end&&*c->at=='.'){
            c->at++;const char *digits=c->at;
            while(c->at<c->end&&*c->at>='0'&&*c->at<='9')c->at++;
            if(c->at==digits)goto bad;
        }
        if(c->at<c->end&&(*c->at=='e'||*c->at=='E')){
            c->at++;if(c->at<c->end&&(*c->at=='+'||*c->at=='-'))c->at++;
            const char *digits=c->at;
            while(c->at<c->end&&*c->at>='0'&&*c->at<='9')c->at++;
            if(c->at==digits)goto bad;
        }
        v->text=strndup(start,(size_t)(c->at-start));v->type=LJ_NUMBER;
        if(!v->text||!lj_number(v->text,&v->number))goto bad;
    }
    return v;
bad:lj_fail(c,"invalid JSON or allocation failure");lj_free(v);return NULL;
}
lj_value *lj_parse(const char *s,size_t n,char *error,size_t cap) {
    if(error&&cap)error[0]=0;
    lj_cursor c={s,s+n,error,cap};lj_value *v=lj_value_read(&c,0);lj_ws(&c);
    if(c.at!=c.end){lj_free(v);v=NULL;lj_fail(&c,"trailing JSON data");}return v;
}
lj_value *lj_file(const char *path,char *error,size_t cap) {
    FILE *f=fopen(path,"rb");if(!f){snprintf(error,cap,"%s: %s",path,strerror(errno));return NULL;}
    long length;
    if(fseek(f,0,SEEK_END)||(length=ftell(f))<0||length>64*1024*1024){fclose(f);snprintf(error,cap,"JSON unreadable or exceeds 64 MiB");return NULL;}
    if(fseek(f,0,SEEK_SET)){fclose(f);snprintf(error,cap,"cannot seek JSON input");return NULL;}
    size_t n=(size_t)length;char *s=malloc(n+1);
    if(!s){fclose(f);return NULL;}size_t got=fread(s,1,n,f);fclose(f);
    lj_value *v=got==n?lj_parse(s,n,error,cap):NULL;free(s);return v;
}
int lj_quote(FILE *f,const char *s) {
    if(fputc('"',f)==EOF)return 0;
    for(const unsigned char *p=(const unsigned char *)s;*p;p++) {
        if(*p=='"'||*p=='\\'){if(fputc('\\',f)==EOF)return 0;}
        if(*p<32){if(fprintf(f,"\\u%04x",*p)<0)return 0;}
        else if(fputc(*p,f)==EOF)return 0;
    }
    return fputc('"',f)!=EOF;
}
