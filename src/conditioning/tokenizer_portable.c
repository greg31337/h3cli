/* Linux equivalent of src/conditioning/tokenizer.m. ICU supplies the same Unicode category
 * and NFC operations; byte-level BPE and H3 special IDs remain native C. */
#include "src/conditioning/tokenizer.h"
#include <json-c/json.h>
#include <unicode/uchar.h>
#include <unicode/unorm2.h>
#include <unicode/ustring.h>
#include <unicode/utf8.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
struct h3_tokenizer {
    struct json_object *config,*vocab,*ranks,*added,*registered,*inverse;
    char encoder[256][4];
    int16_t decoder[324];
    uint32_t maximum;
};
typedef struct { uint32_t *ids;size_t count,capacity; } tokens;
typedef struct { uint32_t value;size_t start,end; } point;
static int fail(char *error,size_t size,const char *message){if(error&&size)snprintf(error,size,"%s",message);return 0;}
static struct json_object *get(struct json_object *o,const char *key){struct json_object *v=NULL;if(o)json_object_object_get_ex(o,key,&v);return v;}
static int type(struct json_object *o,enum json_type t){return o&&json_object_is_type(o,t);}
static int is_string(struct json_object *o,const char *s){return type(o,json_type_string)&&!strcmp(json_object_get_string(o),s);}
static int push(tokens *t,uint32_t id){
    if(t->count==t->capacity){size_t cap=t->capacity?t->capacity*2:32;if(cap<t->capacity||cap>SIZE_MAX/sizeof(*t->ids))return 0;void*p=realloc(t->ids,cap*sizeof(*t->ids));if(!p)return 0;t->ids=p;t->capacity=cap;}
    t->ids[t->count++]=id;return 1;
}
static int register_token(h3_tokenizer*t,const char *s,struct json_object *id,char*e,size_t z){
    if(!s||!*s||!type(id,json_type_int)||json_object_get_int64(id)<0||json_object_get_uint64(id)>UINT32_MAX)return fail(e,z,"invalid tokenizer string or token ID");
    uint32_t n=(uint32_t)json_object_get_uint64(id);char key[32];snprintf(key,sizeof(key),"%u",n);
    struct json_object *old=get(t->registered,s),*inverse=get(t->inverse,key);
    if((old&&json_object_get_uint64(old)!=n)||(inverse&&strcmp(json_object_get_string(inverse),s)))return fail(e,z,"tokenizer collision registering token");
    json_object_object_add(t->registered,s,json_object_get(id));
    json_object_object_add(t->inverse,key,json_object_new_string(s));
    if (n > t->maximum)
        t->maximum = n;
    return 1;
}
static char *pair_key(const char*a,const char*b){size_t x=strlen(a),y=strlen(b);if(x>SIZE_MAX-y-4)return NULL;char*r=malloc(x+y+4);if(r){memcpy(r,a,x);memcpy(r+x,"\xef\xbf\xbf",3);memcpy(r+x+3,b,y+1);}return r;}
h3_tokenizer *h3_tokenizer_load(const char *path,char *error,size_t size){
    if(error&&size)*error=0;
    if(!path){fail(error,size,"tokenizer path is required");return NULL;}
    h3_tokenizer*t=calloc(1,sizeof(*t));if(!t)return NULL;
    t->config=json_object_from_file(path);
    struct json_object *model=get(t->config,"model"),*normalizer=get(t->config,"normalizer"),*merges=get(model,"merges");
    t->vocab=get(model,"vocab");
    if(!is_string(get(model,"type"),"BPE") || get(model,"unk_token") || !is_string(get(normalizer,"type"),"NFC") || !type(t->vocab,json_type_object)||!type(merges,json_type_array)){
        fail(error,size,"unexpected tokenizer specification");goto bad;
    }
    t->registered=json_object_new_object();t->inverse=json_object_new_object();t->added=json_object_new_object();t->ranks=json_object_new_object();
    if(!t->registered||!t->inverse||!t->added||!t->ranks)goto bad;
    {json_object_object_foreach(t->vocab,key,id){if(!register_token(t,key,id,error,size))goto bad;}}
    struct json_object *added=get(t->config,"added_tokens");
    if(added&&!type(added,json_type_array)){fail(error,size,"invalid added-token collection");goto bad;}
    for(size_t i=0;added&&i<json_object_array_length(added);i++){
        struct json_object *a=json_object_array_get_idx(added,i),*id=get(a,"id"),*content=get(a,"content");
        if(!type(a,json_type_object)||!type(content,json_type_string)||json_object_get_boolean(get(a,"single_word"))||json_object_get_boolean(get(a,"lstrip"))||json_object_get_boolean(get(a,"rstrip"))||json_object_get_boolean(get(a,"normalized"))){fail(error,size,"unsupported added-token policy");goto bad;}
        const char*s=json_object_get_string(content);
        if(!register_token(t,s,id,error,size))goto bad;
        json_object_object_add(t->added,s,json_object_get(id));
    }
    {static const char *special[]={"<d>","</d>","<|cutoff|>","<|lyrics_start|>","<|lyrics_end|>","<|caption_start|>","<|caption_end|>"};
    for(size_t i=0;i<sizeof(special)/sizeof(*special);i++){
        struct json_object *id=json_object_new_int64(151669+(int64_t)i);
        int ok=register_token(t,special[i],id,error,size);
        if(ok)json_object_object_add(t->added,special[i],json_object_get(id));
        json_object_put(id);if(!ok)goto bad;
    }}
    for(size_t i=0;i<json_object_array_length(merges);i++){
        struct json_object *entry=json_object_array_get_idx(merges,i);char *left=NULL;const char *right=NULL;
        if(type(entry,json_type_string)){
            const char*s=json_object_get_string(entry),*space=strchr(s,' ');
            if(space){left=strndup(s,(size_t)(space-s));right=space+1;}
        }else if(type(entry,json_type_array)&&json_object_array_length(entry)==2){
            struct json_object*a=json_object_array_get_idx(entry,0),*b=json_object_array_get_idx(entry,1);
            if(type(a,json_type_string)&&type(b,json_type_string)){left=strdup(json_object_get_string(a));right=json_object_get_string(b);}
        }
        if(!left||!right){free(left);fail(error,size,"invalid tokenizer merge");goto bad;}
        char*key=pair_key(left,right);free(left);if(!key)goto bad;
        json_object_object_add(t->ranks,key,json_object_new_int64((int64_t)i));free(key);
    }
    for(size_t i=0;i<324;i++)t->decoder[i]=-1;
    {uint32_t extra=0;
     for(uint32_t b=0;b<256;b++){
        int visible=(b>='!'&&b<='~')||(b>=0xa1&&b<=0xac)||(b>=0xae&&b<=0xff);
        uint32_t cp=visible?b:256+extra++;int32_t n=0;
        U8_APPEND_UNSAFE(t->encoder[b],n,cp);t->encoder[b][n]=0;t->decoder[cp]=(int16_t)b;
     }}
    return t;
bad:
    if(error&&size&&!*error)fail(error,size,"out of memory loading tokenizer");
    h3_tokenizer_free(t);return NULL;
}
void h3_tokenizer_free(h3_tokenizer*t){if(t){json_object_put(t->config);json_object_put(t->ranks);json_object_put(t->added);json_object_put(t->registered);json_object_put(t->inverse);free(t);}}
static char *normalize(const char *s,size_t bytes){
    if (bytes > INT32_MAX)
        return NULL;
    UErrorCode status = U_ZERO_ERROR;
    int32_t units = 0;
    u_strFromUTF8(NULL,0,&units,s,(int32_t)bytes,&status);
    if(status!=U_BUFFER_OVERFLOW_ERROR&&U_FAILURE(status))return NULL;
    UChar *input=malloc(((size_t)units+1)*sizeof(*input));if(!input)return NULL;
    status=U_ZERO_ERROR;u_strFromUTF8(input,units+1,NULL,s,(int32_t)bytes,&status);
    const UNormalizer2*nfc=unorm2_getNFCInstance(&status);
    int32_t needed=unorm2_normalize(nfc,input,units,NULL,0,&status);
    if(status!=U_BUFFER_OVERFLOW_ERROR&&U_FAILURE(status)){free(input);return NULL;}
    UChar *normalized=malloc(((size_t)needed+1)*sizeof(*normalized));if(!normalized){free(input);return NULL;}
    status=U_ZERO_ERROR;unorm2_normalize(nfc,input,units,normalized,needed+1,&status);free(input);
    int32_t length=0;u_strToUTF8(NULL,0,&length,normalized,needed,&status);
    if(status!=U_BUFFER_OVERFLOW_ERROR&&U_FAILURE(status)){free(normalized);return NULL;}
    char*out=malloc((size_t)length+1);if(!out){free(normalized);return NULL;}
    status=U_ZERO_ERROR;u_strToUTF8(out,length+1,NULL,normalized,needed,&status);free(normalized);
    if(U_FAILURE(status)){free(out);return NULL;}return out;
}
static int bpe(const h3_tokenizer*t,const char*s,size_t bytes,tokens*out){
    char **symbols=calloc(bytes?bytes:1,sizeof(*symbols));if(!symbols)return 0;
    size_t count=bytes;int ok=1;
    for(size_t i=0;i<count;i++){symbols[i]=strdup(t->encoder[(unsigned char)s[i]]);if(!symbols[i])ok=0;}
    while(ok&&count>1){
        int64_t best=INT64_MAX;size_t at=0;
        for(size_t i=0;i+1<count;i++){
            char*key=pair_key(symbols[i],symbols[i+1]);if(!key){ok=0;break;}
            struct json_object*r=get(t->ranks,key);free(key);
            if(r&&json_object_get_int64(r)<best){best=json_object_get_int64(r);at=i;}
        }
        if(!ok||best==INT64_MAX)break;
        char *left=strdup(symbols[at]),*right=strdup(symbols[at+1]);
        if(!left||!right){free(left);free(right);ok=0;break;}
        size_t write=0;
        for(size_t i=0;i<count;){
            if(i+1<count&&!strcmp(symbols[i],left)&&!strcmp(symbols[i+1],right)){
                size_t n=strlen(left),m=strlen(right);char*merged=malloc(n+m+1);
                if(!merged){ok=0;while(i<count)symbols[write++]=symbols[i++];break;}
                memcpy(merged,left,n);memcpy(merged+n,right,m+1);
                free(symbols[i]);free(symbols[i+1]);symbols[write++]=merged;i+=2;
            }else symbols[write++]=symbols[i++];
        }
        free(left);free(right);count=write;
    }
    for(size_t i=0;i<count;i++){
        struct json_object*id=ok?get(t->vocab,symbols[i]):NULL;
        if(ok&&(!id||!push(out,(uint32_t)json_object_get_uint64(id))))ok=0;
        free(symbols[i]);
    }
    free(symbols);return ok;
}
static int letter(uint32_t v){int8_t c=u_charType((UChar32)v);return c==U_UPPERCASE_LETTER||c==U_LOWERCASE_LETTER||c==U_TITLECASE_LETTER||c==U_MODIFIER_LETTER||c==U_OTHER_LETTER;}
static int number(uint32_t v){int8_t c=u_charType((UChar32)v);return c==U_DECIMAL_DIGIT_NUMBER||c==U_LETTER_NUMBER||c==U_OTHER_NUMBER;}
static int space(uint32_t v){return u_isUWhiteSpace((UChar32)v)||(v>=0x1c&&v<=0x1f);}
static size_t contraction(const point*p,size_t n,size_t i){
    static const char*v[]={"'s","'t","'re","'ve","'m","'ll","'d"};if(p[i].value!='\'')return 0;
    for(size_t k=0;k<sizeof(v)/sizeof(*v);k++){size_t len=strlen(v[k]);if(i+len>n)continue;int match=1;
        for(size_t j=1;j<len;j++){uint32_t cp=p[i+j].value;if(cp>='A'&&cp<='Z')cp+='a'-'A';if(cp!=(unsigned char)v[k][j])match=0;}
        if(match)return len;
    }return 0;
}
static int plain(const h3_tokenizer*t,const char*s,size_t bytes,tokens*out){
    if (!bytes)
        return 1;
    char *text = normalize(s, bytes);
    if (!text)
        return 0;
    size_t length=strlen(text);point*p=calloc(length?length:1,sizeof(*p));if(!p){free(text);return 0;}
    int32_t cursor=0;size_t n=0;
    while((size_t)cursor<length){int32_t start=cursor;UChar32 cp;U8_NEXT((const uint8_t *)text,cursor,(int32_t)length,cp);if(cp<0){free(text);free(p);return 0;}p[n++]=(point){(uint32_t)cp,(size_t)start,(size_t)cursor};}
    int ok=1;
    for(size_t i=0;i<n&&ok;){size_t stop=i,contract=contraction(p,n,i);uint32_t v=p[i].value;
        if(contract)stop=i+contract;
        else if(letter(v)||(v!='\r'&&v!='\n'&&!number(v)&&i+1<n&&letter(p[i+1].value))){stop=i+(size_t)!letter(v);while(stop<n&&letter(p[stop].value))stop++;}
        else if(number(v))stop=i+1;
        else {
            size_t start=i+(size_t)(v==' '&&i+1<n&&!space(p[i+1].value)&&!letter(p[i+1].value)&&!number(p[i+1].value));
            stop=start;while(stop<n&&!space(p[stop].value)&&!letter(p[stop].value)&&!number(p[stop].value))stop++;
            if(stop>start){while(stop<n&&(p[stop].value=='\r'||p[stop].value=='\n'))stop++;}
            else if(space(v)){
                size_t end=i+1,newline=0;while(end<n&&space(p[end].value))end++;
                for(size_t j=i;j<end;j++)if(p[j].value=='\r'||p[j].value=='\n')newline=j+1;
                stop=newline?newline:end==n?end:end-i>1?end-1:i+1;
            }else {ok=0;break;}
        }
        ok=bpe(t,text+p[i].start,p[stop-1].end-p[i].start,out);i=stop;
    }
    free(p);free(text);return ok;
}
int h3_tokenizer_encode(const h3_tokenizer*t,const char*s,int pad,uint32_t**ids,size_t*count,char*error,size_t size){
    if (error && size)
        *error = 0;
    if (!t || !s || !ids || !count)
        return 0;
    *ids = NULL;
    *count = 0;
    // Validate the complete input before matching special tokens.
    char*valid=normalize(s,strlen(s));if(!valid)return fail(error,size,"prompt is not valid UTF-8");free(valid);
    tokens out={0};const char*start=s;int ok=1;
    while(*start&&ok){const char*match=NULL,*special=NULL;uint32_t id=0;
        {json_object_object_foreach(t->added,key,value){const char*found=strstr(start,key);
            if(found&&(!match||found<match||(found==match&&strlen(key)>strlen(special)))){match=found;special=key;id=(uint32_t)json_object_get_uint64(value);}}}
        if(!match){ok=plain(t,start,strlen(start),&out);break;}
        ok=plain(t,start,(size_t)(match-start),&out)&&push(&out,id);start=match+strlen(special);
    }
    if(ok&&!out.count&&pad)ok=push(&out,H3_PAD_TOKEN_ID);
    if(!ok){free(out.ids);return fail(error,size,"unable to encode BPE prompt");}
    *ids=out.ids;*count=out.count;return 1;
}
void h3_tokenizer_ids_free(uint32_t*ids){free(ids);}
typedef struct {char *p;size_t used,cap;} buffer;
static int append(buffer*b,const char*s,size_t n){if(n>SIZE_MAX-b->used-1)return 0;size_t need=b->used+n+1;if(need>b->cap){size_t cap=need>SIZE_MAX/2?need:need*2;void*p=realloc(b->p,cap);if(!p)return 0;b->p=p;b->cap=cap;}memcpy(b->p+b->used,s,n);b->used+=n;b->p[b->used]=0;return 1;}
static int flush(buffer*out,buffer*bytes){
    if (!bytes->used)
        return 1;
    int32_t i = 0;
    int valid = bytes->used <= INT32_MAX;
    while(valid&&(size_t)i<bytes->used){UChar32 cp;U8_NEXT((const uint8_t *)bytes->p,i,(int32_t)bytes->used,cp);if(cp<0)valid=0;}
    int ok=valid?append(out,bytes->p,bytes->used):append(out,"\xef\xbf\xbd",3);bytes->used=0;return ok;
}
char*h3_tokenizer_decode(const h3_tokenizer*t,const uint32_t*ids,size_t count,char*error,size_t size){
    if (error && size)
        *error = 0;
    if (!t || (!ids && count))
        return NULL;
    buffer out = {0}, bytes = {0};
    int ok = 1;
    for(size_t i=0;i<count&&ok;i++){
        char key[32];snprintf(key,sizeof(key),"%u",ids[i]);struct json_object*v=get(t->inverse,key);
        if(!v){fail(error,size,ids[i]>t->maximum?"token ID is out of range":"unknown token ID");ok=0;break;}
        const char*s=json_object_get_string(v);if(get(t->added,s)){ok=flush(&out,&bytes)&&append(&out,s,strlen(s));continue;}
        size_t len=strlen(s);int32_t p=0;
        while((size_t)p<len&&ok){UChar32 cp;U8_NEXT((const uint8_t *)s,p,(int32_t)len,cp);
            if(cp<0||cp>=324||t->decoder[cp]<0){fail(error,size,"invalid byte-level token");ok=0;break;}
            char b=(char)t->decoder[cp];ok=append(&bytes,&b,1);
        }
    }
    if (ok)
        ok = flush(&out, &bytes) && append(&out, "", 0);
    free(bytes.p);
    if(!ok){free(out.p);return NULL;}return out.p;
}
