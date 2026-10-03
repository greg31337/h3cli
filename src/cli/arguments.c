#include "src/request.h"
#include <ctype.h>
#include <stdlib.h>
#include <string.h>
int h3_request_parse_string(h3_request *r,const char *text,char *error,size_t size) {
    if(!text||strlen(text)>65536){snprintf(error,size,"h3cli must be a string of at most 65536 bytes");return 0;}
    char **argv=calloc(1025,sizeof(*argv));char *token=malloc(strlen(text)+1);
    if(!argv||!token){free(argv);free(token);snprintf(error,size,"out of memory");return 0;}
    int argc=0,ok=0;const char *p=text;
    while(*p){
        while (isspace((unsigned char)*p))
            p++;
        if (!*p)
            break;
        size_t n=0;char quote=0;
        while(*p&&(quote||!isspace((unsigned char)*p))){
            char c=*p++;
            if(c=='\''&&quote!='"'){quote=quote?0:c;continue;}
            if(c=='"'&&quote!='\''){quote=quote?0:c;continue;}
            if(c=='\\'&&quote!='\''){
                if(!*p)goto malformed;
                if(quote=='"'&&*p!='"'&&*p!='\\'&&*p!='$'&&*p!='`'&&*p!='\n'){token[n++]=c;continue;}
                c=*p++;if(c=='\n')continue;
            }token[n++]=c;
        }
        if(quote)goto malformed;
        if(argc>=1024){snprintf(error,size,"too many h3cli tokens");goto done;}
        token[n]=0;argv[argc++]=strdup(token);if(!argv[argc-1]){snprintf(error,size,"out of memory");goto done;}
    }
    ok=h3_request_parse_argv(r,argc,argv,H3_SOURCE_NATIVE,error,size);goto done;
malformed:snprintf(error,size,"unterminated quote/escape in h3cli at byte %zu",(size_t)(p-text));
done:h3_argv_free(argv);free(token);return ok;
}
