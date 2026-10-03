#include "src/server/sglang.h"
#include <stdlib.h>
#include <string.h>
int main(int argc,char **argv){
    char error[1024]={0};sj_value *out=NULL;
    if(argc==2&&!strcmp(argv[1],"registry"))out=h3_option_schema();
    else {char *text=malloc(1024*1024+1);if(!text)return 1;size_t n=fread(text,1,1024*1024,stdin);sj_value *body=sj_parse(text,n,error,sizeof(error));free(text);h3_submission s={0};
        int ok=body&&h3_submission_parse(body,&s,error,sizeof(error));
        if(ok&&!(argc==2&&!strcmp(argv[1],"parse-only")))ok=h3_submission_geometry(&s,682,1024,4,error,sizeof(error));
        if(ok){out=h3_request_json(&s.request);sj_add(out,"operation",sj_string(h3_request_operation(&s.request)));sj_add(out,"task",sj_string(s.task));sj_add(out,"variants",sj_int(s.variants));sj_value *seeds=sj_array();for(int i=0;i<s.variants;i++)sj_add(seeds,NULL,sj_uint(s.seeds[i]));sj_add(out,"seeds",seeds);}
        h3_submission_free(&s);sj_free(body);if(!ok){fprintf(stderr,"%s\n",error);return 2;}}
    if (!out)
        return 1;
    int ok = sj_write(stdout, out);
    fputc('\n', stdout);
    sj_free(out);
    return ok ? 0 : 1;
}
