/* Exercise the production batch coordinator with deterministic model/decoder
 * substitutes: seed order, bounded staging, phase separation and failures. */
#include "src/cli/still_batch.h"
#include "src/internal.h"
#include "src/vae/image_vae.h"
#include <errno.h>
#include <fcntl.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static unsigned checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s\n",__LINE__,#x);exit(1);}} while(0)
static int generated,decoded,results,loads,decoder_loads,fail_generate,fail_decode,bad_shape,staged_fd;
static int expected_count;
static uint64_t initial_seed;
static h3_ctx *active;

static void fake_clear(h3_ctx *ctx){ctx->dit=NULL;}
static void fake_enable(h3_ctx *ctx,int enabled){if(!enabled)fake_clear(ctx);ctx->cache_enabled=enabled;}
static const char *fake_error(const h3_ctx *ctx){return ctx->error;}
static FILE *fake_tmpfile(void){FILE *f=tmpfile();if(f)staged_fd=fileno(f);return f;}
static h3_result *fake_generate(h3_ctx *ctx,const char *prompt,const h3_params *p) {
    CHECK(!strcmp(prompt,"batch prompt")&&ctx->cache_enabled&&decoded==0);
    CHECK(p->seed==initial_seed+(uint64_t)generated);
    if(generated++==fail_generate){snprintf(ctx->error,sizeof(ctx->error),"injected sampling failure");return NULL;}
    if(!ctx->dit){loads++;ctx->dit=(struct h3_dit *)(uintptr_t)1;}
    h3_result *r=calloc(1,sizeof(*r));CHECK(r);
    h3_still_latent *z=calloc(1,sizeof(*z));CHECK(z);
    z->height=2;z->width=2;z->values=malloc(96*sizeof(float));CHECK(z->values);
    memset(z->compatibility_sha256,'a',64);
    for(int i=0;i<96;i++)z->values[i]=(float)(p->seed%1000)+(float)i/128;
    if(bad_shape)z->width=4;
    r->kind=H3_RESULT_STILL;r->frames=1;r->still_latent=z;r->seed=p->seed;
    return r;
}
static h3_result *fake_decode(const h3_still_latent *z,const char *directory,
    const h3_decode_options *options,h3_still_decode_cache *cache,char *error,size_t size) {
    CHECK(active->dit==NULL&&generated==expected_count);
    CHECK(!strcmp(directory,"model/FL2VA/video_vae/source"));
    CHECK(z->width==2&&z->height==2);
    for(int i=0;i<96;i++)CHECK(z->values[i]==(float)((initial_seed+(uint64_t)decoded)%1000)+(float)i/128);
    char *path=h3_still_batch_path("images/teapot.png",expected_count,initial_seed+(uint64_t)decoded);
    CHECK(path&&!strcmp(path,options->output_path));free(path);
    if(decoded++==fail_decode){snprintf(error,size,"injected decode failure");return NULL;}
    if(!cache->decoder){decoder_loads++;cache->decoder=(h3_video_vae_decoder *)(uintptr_t)1;}
    h3_result *r=calloc(1,sizeof(*r));CHECK(r);r->kind=H3_RESULT_STILL;r->frames=1;
    return r;
}
static void fake_decoder_clear(h3_still_decode_cache *cache){memset(cache,0,sizeof(*cache));}

#define h3_cache_clear fake_clear
#define h3_cache_set_enabled fake_enable
#define h3_last_error fake_error
#define h3_generate_still_latent fake_generate
#define h3_decode_still_values_cached fake_decode
#define h3_still_decode_cache_clear fake_decoder_clear
#define tmpfile fake_tmpfile
#include "src/cli/still_batch.c"
#undef h3_cache_clear
#undef h3_cache_set_enabled
#undef h3_last_error
#undef h3_generate_still_latent
#undef h3_decode_still_values_cached
#undef h3_still_decode_cache_clear
#undef tmpfile

static void item(int index,int count,uint64_t seed,int decoding,void *opaque) {
    CHECK(opaque==active&&count==expected_count&&seed==initial_seed+(uint64_t)index);
    CHECK(index==(decoding?decoded:generated));
}
static void result(const h3_result *r,const char *output,void *opaque) {
    CHECK(opaque==active&&output&&r->still_latent);
    CHECK(r->seed==initial_seed+(uint64_t)results++&&r->completed_steps==2&&r->total_steps==2);
}
static void paths(void) {
    const char *source[]={"a.png","dir.with.dot/name.png","dir.with.dot/latent","a.tar.safetensors",".png","a path/猫.png"};
    const char *expected[]={"a-seed-42.png","dir.with.dot/name-seed-42.png","dir.with.dot/latent-seed-42","a.tar-seed-42.safetensors","-seed-42.png","a path/猫-seed-42.png"};
    for(size_t i=0;i<sizeof(source)/sizeof(*source);i++) {
        char *p=h3_still_batch_path(source[i],3,42);CHECK(p&&!strcmp(p,expected[i]));free(p);
        p=h3_still_batch_path(source[i],1,42);CHECK(p&&!strcmp(p,source[i]));free(p);
    }
    char *p=h3_still_batch_path("a.png",2,UINT64_MAX);CHECK(p&&!strcmp(p,"a-seed-18446744073709551615.png"));free(p);
    CHECK(!h3_still_batch_path(NULL,2,42));
}
static void run(int count,int gen_failure,int dec_failure,int malformed,int cache_enabled) {
    generated=decoded=results=loads=decoder_loads=0;staged_fd=-1;
    fail_generate=gen_failure;fail_decode=dec_failure;bad_shape=malformed;
    expected_count=count;initial_seed=42;
    h3_ctx ctx={.model_dir="model",.cache_enabled=cache_enabled};active=&ctx;
    h3_params p=H3_PARAMS_DEFAULT;p.still=1;p.frames=1;p.width=32;p.height=32;p.steps=2;
    p.seed=initial_seed;p.output_path="images/teapot.png";
    h3_still_batch_hooks hooks={item,result,&ctx};char error[512];
    h3_cuda_policy previous=h3_cuda_policy_exchange((h3_cuda_policy){.adaptive_cache_max_bytes=1234});
    int ok=h3_still_batch_run(&ctx,"batch prompt",&p,count,&hooks,error,sizeof(error));
    CHECK(h3_cuda_policy_current().adaptive_cache_max_bytes==1234);
    h3_cuda_policy_exchange(previous);
    CHECK(ok==(gen_failure<0&&dec_failure<0&&!malformed));
    CHECK(ctx.cache_enabled==cache_enabled&&ctx.dit==NULL&&loads==(gen_failure==0?0:1));
    CHECK(staged_fd>=0&&fcntl(staged_fd,F_GETFD)==-1&&errno==EBADF);
    if(ok)CHECK(generated==count&&decoded==count&&results==count&&decoder_loads==1);
    else {
        CHECK(error[0]);
        if(gen_failure>=0||malformed)CHECK(decoded==0&&results==0&&decoder_loads==0);
        else CHECK(results==dec_failure);
    }
}
int main(void) {
    paths();char error[256];h3_params p=H3_PARAMS_DEFAULT;
    CHECK(!h3_still_batch_validate(&p,2,error,sizeof(error))&&strstr(error,"requires --still"));
    p.still=1;
    CHECK(!h3_still_batch_validate(&p,0,error,sizeof(error)));
    CHECK(!h3_still_batch_validate(&p,-1,error,sizeof(error)));
    p.seed=UINT64_MAX;
    CHECK(h3_still_batch_validate(&p,1,error,sizeof(error)));
    CHECK(!h3_still_batch_validate(&p,2,error,sizeof(error))&&strstr(error,"overflow"));
    p.seed=UINT64_MAX-2;CHECK(h3_still_batch_validate(&p,3,error,sizeof(error)));
    run(3,-1,-1,0,0);run(1,-1,-1,0,1);
    run(3,0,-1,0,0);run(3,1,-1,0,0);
    run(3,-1,0,0,0);run(3,-1,1,0,0);run(3,-1,-1,1,0);
    printf("PASS: %u still batch checks\n",checks);return 0;
}
