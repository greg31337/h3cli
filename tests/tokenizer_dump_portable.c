#include "src/conditioning/tokenizer.h"
#include <json-c/json.h>
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv){
    if(argc!=3)return 2;
    char error[512];h3_tokenizer*t=h3_tokenizer_load(argv[1],error,sizeof(error));
    if(!t){fprintf(stderr,"%s\n",error);return 1;}
    struct json_object*corpus=json_object_from_file(argv[2]);
    if(!corpus||!json_object_is_type(corpus,json_type_array))return 2;
    struct json_object*out=json_object_new_array();
    for(size_t i=0;i<json_object_array_length(corpus);i++){
        struct json_object*item=json_object_array_get_idx(corpus,i),*text,*name;
        if(!json_object_object_get_ex(item,"text",&text)||!json_object_object_get_ex(item,"name",&name))return 2;
        uint32_t*ids=NULL;size_t n=0;
        if(!h3_tokenizer_encode(t,json_object_get_string(text),0,&ids,&n,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
        char*decoded=h3_tokenizer_decode(t,ids,n,error,sizeof(error));
        if(!decoded){fprintf(stderr,"%s\n",error);return 1;}
        struct json_object*row=json_object_new_object(),*values=json_object_new_array();
        for(size_t j=0;j<n;j++)json_object_array_add(values,json_object_new_int64(ids[j]));
        json_object_object_add(row,"name",json_object_get(name));json_object_object_add(row,"text",json_object_get(text));
        json_object_object_add(row,"ids",values);json_object_object_add(row,"decoded",json_object_new_string(decoded));
        json_object_array_add(out,row);free(ids);free(decoded);
    }
    puts(json_object_to_json_string_ext(out,JSON_C_TO_STRING_PRETTY));
    json_object_put(out);json_object_put(corpus);h3_tokenizer_free(t);return 0;
}
