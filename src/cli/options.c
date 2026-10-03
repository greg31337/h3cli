#include "src/cli/options.h"
#include "src/cli/option_ids.h"
#include "src/weights/lora_json.h"
#include "src/denoise/adaptive_cache.h"
#include <errno.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
const h3_quality_preset h3_quality_presets[5]={
    {"lossless",50,1,H3_ADAPTIVE_OFF,0},
    {"extra-high",50,1,H3_ADAPTIVE_OFF,0},
    {"high",50,1,H3_ADAPTIVE_CONSERVATIVE,0},
    {"preview",12,2,H3_ADAPTIVE_OFF,1},
    {"fast-preview",6,3,H3_ADAPTIVE_OFF,1}
};
#define H3_OPTION(name,id,arity,type,path,repeat,group,min,max,choices,def,platform,ops) \
    {name, (arity) ? required_argument : no_argument, NULL, id},
const struct option h3_cli_getopt_options[] = {
#include "src/cli/options.def"
    {NULL, 0, NULL, 0}
};
#undef H3_OPTION
#define H3_OPTION(name,id,arity,type,path,repeat,group,min,max,choices,def,platform,ops) \
    {name,id,arity,H3_OPT_##type,H3_PATH_##path,repeat,group,min,max,choices,def,platform,ops},
const h3_option_descriptor h3_cli_options[] = {
#include "src/cli/options.def"
};
#undef H3_OPTION
const size_t h3_cli_option_count=sizeof(h3_cli_options)/sizeof(h3_cli_options[0]);
const h3_option_descriptor *h3_option_find(const char *name) {
    for(size_t i=0;i<h3_cli_option_count;i++)if(!strcmp(name,h3_cli_options[i].name))return &h3_cli_options[i];
    return NULL;
}
const h3_option_descriptor *h3_option_short(char name) {
    for(size_t i=0;i<h3_cli_option_count;i++)if(h3_cli_options[i].id==(unsigned char)name)return &h3_cli_options[i];
    return NULL;
}
int h3_option_validate(const h3_option_descriptor *o,const char *s,char *error,size_t size) {
    int ok=1;
    if(!o||(!s&&o->arity)){snprintf(error,size,"missing option/value");return 0;}
    if(!o->arity)return 1;
    if(o->type==H3_OPT_ENUM){
        ok=0;size_t n=strlen(s);
        for(const char *p=o->choices;p&&*p;){const char *end=strchr(p,'|');size_t len=end?(size_t)(end-p):strlen(p);
            if(n==len&&!memcmp(p,s,n)){ok=1;break;}p=end?end+1:NULL;}
    }else if(o->type==H3_OPT_INT||o->type==H3_OPT_U64){
        if (!*s)
            ok = 0;
        for (const char *p = s; *p; p++)
            if (*p < '0' || *p > '9')
                ok = 0;
        char *end;errno=0;unsigned long long v=strtoull(s,&end,10);
        if(errno||*end||((double)v<o->minimum)||
           (o->type==H3_OPT_INT&&(double)v>o->maximum))ok=0;
    }else if(o->type==H3_OPT_FLOAT){double v;
        ok=lj_number(s,&v)&&isfinite(v)&&v>=o->minimum&&v<=o->maximum;
        if(!strcmp(o->name,"seconds")&&ok&&v<=0)ok=0;
    }else if(o->path!=H3_PATH_NONE&&! *s)ok=0;
    if(!ok)snprintf(error,size,"invalid --%s value: %.160s%s%s",o->name,s,
        o->choices?"; expected ":"",o->choices?o->choices:"");
    return ok;
}
