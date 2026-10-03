/* Test-only transport/catalog entry point. No fixture controls enter h3cli. */
#include "src/models/models.h"
#include "src/server/json.h"
#include <signal.h>
#include <stdlib.h>
#include <string.h>
static volatile sig_atomic_t cancelled;
static void stop(int sig){(void)sig;cancelled=1;}
static double cancel_after;
static int progress(const h3_model_progress *p,void *opaque){
    (void)opaque;
    sj_value *e=sj_object();sj_add(e,"phase",sj_string(p->phase));sj_add(e,"component",sj_string(p->component));
    sj_add(e,"downloaded",sj_uint(p->completed));sj_add(e,"reused",sj_uint(p->reused));sj_add(e,"seconds",sj_real(p->elapsed));
    char *text=sj_dump(e);if(text){puts(text);free(text);}sj_free(e);fflush(stdout);
    return cancelled||(cancel_after>0&&p->elapsed>=cancel_after);
}
int main(int argc,char **argv){
    char error[1024]={0};h3_model_plan plan={0};h3_request request={0};int ok=0;
    if(argc>1&&!strcmp(argv[1],"plan")){
        ok=h3_request_parse_argv(&request,argc-2,argv+2,H3_SOURCE_NATIVE,error,sizeof(error))&&
            h3_models_catalog_valid(error,sizeof(error))&&h3_models_resolve(&request,&plan,error,sizeof(error));
        if(ok){sj_value *a=sj_array();for(size_t i=0;i<plan.count;i++){
            sj_value *v=sj_object();sj_add(v,"path",sj_string(plan.items[i].artifact->path));sj_add(v,"destination",sj_string(plan.items[i].destination));sj_add(a,NULL,v);
        }char *text=sj_dump(a);if(text){puts(text);free(text);}sj_free(a);}
    }else if(argc==3&&!strcmp(argv[1],"fetch")){
        FILE *f=fopen(argv[2],"rb");char buffer[65536];size_t n=f?fread(buffer,1,sizeof(buffer)-1,f):0;if(f)fclose(f);
        sj_value *v=sj_parse(buffer,n,error,sizeof(error));const sj_value *files=sj_get(v,"files");
        if(!v||!files||files->type!=LJ_ARRAY||files->count>32)goto done;
        plan.paths.main=h3_models_absolute(sj_field(v,"root"),error,sizeof(error));
        plan.offline=h3_models_offline();plan.verify=sj_get(v,"verify")&&sj_get(v,"verify")->number!=0;
        const sj_value *delay=sj_get(v,"cancel_after");if(delay&&delay->text)cancel_after=strtod(delay->text,NULL);
        plan.count=files->count;plan.items=calloc(plan.count,sizeof(*plan.items));
        h3_model_artifact *artifacts=calloc(plan.count,sizeof(*artifacts));
        if(!plan.paths.main||!plan.items||!artifacts){free(artifacts);sj_free(v);goto done;}
        for(size_t i=0;i<plan.count;i++){
            const sj_value *a=files->items[i];artifacts[i]=(h3_model_artifact){.path=sj_field(a,"path"),.url=sj_field(a,"url"),.sha256=sj_field(a,"sha256"),.groups=H3_MODEL_BASE};
            sj_u64(sj_get(a,"bytes"),&artifacts[i].bytes);plan.items[i].artifact=&artifacts[i];
            if(asprintf(&plan.items[i].destination,"%s/%s",plan.paths.main,artifacts[i].path)<0)plan.items[i].destination=NULL;
        }
        signal(SIGINT,stop);signal(SIGTERM,stop);signal(SIGXFSZ,SIG_IGN);
        ok=h3_models_prepare(&plan,progress,NULL,error,sizeof(error));
        free(artifacts);sj_free(v);
    }
done:
    if(!ok)fprintf(stderr,"%s\n",*error?error:"invalid test invocation");
    h3_models_plan_free(&plan);h3_request_free(&request);return ok?0:1;
}
