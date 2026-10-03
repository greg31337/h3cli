/* Model-backed cache regression: reference contents change while path, size and
 * mtime remain identical. Three preparation-only requests, zero evaluations.
 * Usage: h3_conditioning_context MODEL_DIR NEW_OUTPUT_DIRECTORY */
#include "src/h3.h"
#include "src/conditioning/conditioning.h"
#include "src/sampling/sampler_state.h"
#include "src/denoise/approximate.h"
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static int fixture(const char *path,int blue){
    FILE *f=fopen(path,"wb");if(!f)return 0;
    int ok=fprintf(f,"P6\n256 192\n255\n")>0;
    unsigned char pixel[3]={blue?20:220,40,blue?220:20};
    for(int i=0;i<256*192&&ok;i++)ok=fwrite(pixel,1,3,f)==3;
    return fclose(f)==0&&ok;
}
static int prepare(h3_ctx *ctx,h3_params *p,const char *path){
    p->save_conditioning=path;
    h3_result *r=h3_generate(ctx,"A person in a blue jacket walks through a sunny park.",p);
    if(!r){fprintf(stderr,"%s\n",h3_last_error(ctx));return 0;}
    int ok=r->sampler_state!=NULL;
    if(ok&&p->adaptive_cache)ok=!r->sampler_state->adaptive_history.ready&&
        r->sampler_state->params.adaptive_cache_threshold==h3_adaptive_threshold(p)&&
        r->sampler_state->params.adaptive_cache_max_hits==h3_adaptive_max_hits(p);
    h3_result_free(r);return ok;
}
int main(int argc,char **argv){
    if(argc!=3)return 2;
    if(mkdir(argv[2],0700)){perror("new output directory required");return 2;}
    char image[4096],a[4096],b[4096],c[4096],pause[4096],error[512]={0};
    const char *names[]={"reference.ppm","a.h3cond","b.h3cond","cold-b.h3cond","pause.h3sample"};
    char *paths[]={image,a,b,c,pause};
    for(int i=0;i<5;i++)if(snprintf(paths[i],4096,"%s/%s",argv[2],names[i])>=4096)return 2;
    setenv("H3_TEST_MAX_EVALUATIONS","6",1);setenv("H3_CPU_SAMPLER","1",1);
    h3_ctx *ctx=NULL;h3_conditioning *first=NULL,*warm=NULL,*cold=NULL;int result=1;
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL line %d: %s (%s)\n",__LINE__,#x,error);goto cleanup;}}while(0)
    CHECK(fixture(image,0));struct stat initial;CHECK(!stat(image,&initial));
    h3_reference ref={H3_REFERENCE_IMAGE,image,NULL,0};h3_params p=H3_PARAMS_DEFAULT;
    p.width=p.height=256;p.frames=22;p.steps=1;p.dit_layers=50;p.denoise_reuse=p.core_reuse=1;
    p.use_slower_bf16_mlp=p.use_slower_bf16_qkv=p.use_slower_bf16_attention_output=1;
    p.references=&ref;p.reference_count=1;p.stop_after_step=0;p.save_sampler_state=pause;
    if(getenv("H3_TEST_SUBBLOCK_CONTEXT")){p.steps=50;p.cuda_attention=4;p.subblock_sparsity=.75f;p.subblock_warmup=2;}
    if(getenv("H3_TEST_ADAPTIVE_CONTEXT")){p.steps=6;p.adaptive_cache=1;p.adaptive_cache_threshold=.06f;p.adaptive_cache_max_hits=2;}
    ctx=h3_load_dir(argv[1]);CHECK(ctx);h3_cache_set_enabled(ctx,1);CHECK(prepare(ctx,&p,a));
    CHECK(fixture(image,1));
#ifdef __APPLE__
    struct timespec times[2]={initial.st_atimespec,initial.st_mtimespec};
#else
    struct timespec times[2]={initial.st_atim,initial.st_mtim};
#endif
    CHECK(!utimensat(AT_FDCWD,image,times,0));CHECK(prepare(ctx,&p,b));
    h3_free(ctx);ctx=h3_load_dir(argv[1]);CHECK(ctx);CHECK(prepare(ctx,&p,c));
    first=h3_conditioning_load(a,error,sizeof(error));warm=h3_conditioning_load(b,error,sizeof(error));
    cold=h3_conditioning_load(c,error,sizeof(error));CHECK(first&&warm&&cold);
    CHECK(strcmp(first->identity,warm->identity));CHECK(!strcmp(warm->identity,cold->identity));
    CHECK(first->video_elements==warm->video_elements&&warm->video_elements==cold->video_elements);
    CHECK(memcmp(first->video,warm->video,warm->video_elements*sizeof(float)));
    CHECK(!memcmp(warm->video,cold->video,warm->video_elements*sizeof(float)));
    CHECK(warm->text.tokens==cold->text.tokens);
    CHECK(!memcmp(warm->text.values,cold->text.values,warm->text.tokens*warm->text.width*2));
    const h3_prepared_tensor *wr=h3_prepared_find(&warm->prepared,1,warm->text.tokens*5376);
    const h3_prepared_tensor *cr=h3_prepared_find(&cold->prepared,1,cold->text.tokens*5376);
    CHECK(wr&&cr&&!memcmp(wr->values,cr->values,wr->elements*2));
    if(p.adaptive_cache){
        char policy[4096];CHECK(snprintf(policy,sizeof(policy),"%s/policy.h3cond",argv[2])<(int)sizeof(policy));
        h3_cache_set_enabled(ctx,1);CHECK(prepare(ctx,&p,policy));
        h3_cache_info retained;h3_cache_get_info(ctx,&retained);CHECK(retained.prepared_dit);
        p.adaptive_cache_threshold=0;p.adaptive_cache_threshold_set=1;p.adaptive_cache_max_hits=16;
        CHECK(prepare(ctx,&p,policy));
        p.first_frame="invalid-anchor";h3_result *bad=h3_generate(ctx,"invalid",&p);CHECK(!bad);
        p.first_frame=NULL;CHECK(prepare(ctx,&p,policy));
    }
    puts("PASS: same-path/size/mtime reference replacement invalidates live conditioning and DiT; warm and cold BF16/F32 conditioning are exact; zero denoising evaluations");
    result=0;
cleanup:
    h3_conditioning_free(first);h3_conditioning_free(warm);h3_conditioning_free(cold);h3_free(ctx);return result;
}
