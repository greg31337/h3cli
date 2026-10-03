/* Real-backend lifecycle test: owned options, both modes, prepared DiT reuse,
 * repeated content verification, deterministic latents, and cancellation. */
#include "../src/h3.h"
#include "../src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int cancel;
static int progress(const char *phase,int completed,int total,void *opaque) {
    (void)completed;(void)total;(void)opaque;
    return cancel&&!strncmp(phase,"LoRA",4);
}
int main(int argc,char **argv) {
    if(argc!=6){fprintf(stderr,"usage: lora_runtime_generate model cache adapter reference output-prefix\n");return 2;}
    char *argument=strdup(argv[3]),*cache=strdup(argv[2]);
    const char *adapters[]={argument};
    h3_lora_options selection={adapters,1,cache,512};
    h3_ctx *ctx=h3_load_dir_with_lora(argv[1],&selection);
    /* The context must not borrow any caller-owned option strings or arrays. */
    memset(argument,'x',strlen(argument));memset(cache,'x',strlen(cache));
    free(argument);free(cache);adapters[0]=NULL;memset(&selection,0,sizeof(selection));
    if(!ctx){fprintf(stderr,"%s\n",h3_last_error(NULL));return 1;}
    h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;p.width=p.height=128;p.frames=22;p.steps=2;p.seed=4142;
    p.preview_vae=1;p.on_progress=progress;
    h3_reference reference={.kind=H3_REFERENCE_IMAGE,.path=argv[4]};
    unsigned char first[32];int ok=1;
    for(int request=0;request<4&&ok;request++) {
        int mode=request==2;
        p.references=mode?&reference:NULL;p.reference_count=(size_t)mode;
        h3_result *result=h3_generate(ctx,"A person wearing a blue jacket walks through a sunny park.",&p);
        if(!result){fprintf(stderr,"request %d: %s\n",request,h3_last_error(ctx));ok=0;break;}
        unsigned char hash[32];h3_av_state_fingerprint(result->av_state,hash);
        if(!request)memcpy(first,hash,32);
        else if(!mode&&memcmp(first,hash,32)){fprintf(stderr,"repeated FL2VA latents differ\n");ok=0;}
        char path[4096],error[512];snprintf(path,sizeof(path),"%s-%d.h3av",argv[5],request);
        if(!h3_result_save_av_state(result,path,error,sizeof(error))){fprintf(stderr,"%s\n",error);ok=0;}
        h3_cache_info info;h3_cache_get_info(ctx,&info);
        printf("request %d mode=%s prepared_dit=%d lora_provenance=%d hash=",request,mode?"Ref2VA":"FL2VA",info.prepared_dit,result->lora_provenance!=NULL);
        for(int i=0;i<32;i++)printf("%02x",hash[i]);putchar('\n');fflush(stdout);
        if(!result->lora_provenance)ok=0;
        h3_result_free(result);
    }
    cancel=1;
    h3_result *cancelled=h3_generate(ctx,"A sunny park.",&p);
    if(cancelled){h3_result_free(cancelled);ok=0;}
    else if(!strstr(h3_last_error(ctx),"cancel")){fprintf(stderr,"unexpected cancellation error: %s\n",h3_last_error(ctx));ok=0;}
    h3_free(ctx);return ok?0:1;
}
