/* Frozen reference probes use production APIs; fast outputs are compared only
 * within the same process, so later fast improvements do not redefine goldens. */
#define main isolation_main
#include "cuda_sglang_isolation.c"
#undef main
#include "src/host.h"
static void save(const char *root,const char *name,const void *p,size_t bytes) {
    char path[4096];CHECK(snprintf(path,sizeof(path),"%s/%s",root,name)<(int)sizeof(path));
    FILE *f=fopen(path,"wbx");CHECK(f);CHECK(fwrite(p,1,bytes,f)==bytes);CHECK(!fclose(f));
}
static void load(const char *root,const char *name,void *p,size_t bytes) {
    char path[4096];CHECK(snprintf(path,sizeof(path),"%s/%s",root,name)<(int)sizeof(path));
    FILE *f=fopen(path,"rb");CHECK(f);CHECK(fread(p,1,bytes,f)==bytes);CHECK(!fclose(f));
}
int main(int argc,char **argv) {
    CHECK(argc==3);const char *out=argv[1],*fixtures=argv[2];
    h3_params p=H3_PARAMS_DEFAULT;CHECK(h3_sglang_resolve(&p,"cuda",0)==H3_SGLANG_VERSION);
    CHECK(H3_SGLANG_VERSION==4);
    float noise[4096];CHECK(h3_sglang_normal(42,noise,4096));save(out,"noise.f32",noise,sizeof(noise));
    for(int n=6;n<=50;n+=44) {
        h3_sigma_schedule s;CHECK(h3_sglang_schedule(n,&s));char name[128];
        snprintf(name,sizeof(name),"schedule-%d.f32",n);save(out,name,s.video,(size_t)(n+1)*4);
        snprintf(name,sizeof(name),"audio-schedule-%d.f32",n);save(out,name,s.audio,(size_t)(n+1)*4);
        for(int step=0;step<n;step++) {
            float state[4096],v[4096];memcpy(state,noise,sizeof(state));
            for(int i=0;i<4096;i++)v[i]=noise[4095-i]*.125f;
            CHECK(h3_sglang_euler(state,v,4096,s.video[step],s.video[step+1]));
            snprintf(name,sizeof(name),"euler-%d-%d.f32",n,step);save(out,name,state,sizeof(state));
        }
    }
    h3_sigma_schedule s;CHECK(h3_sglang_schedule(50,&s));
    for(int step=25;step<=49;step+=24)for(int modality=0;modality<2;modality++) {
        float x[4096],v[4096];char name[128];const char *kind=modality?"audio":"video";
        snprintf(name,sizeof(name),"late-%d-%s-input.f32",step,kind);load(fixtures,name,x,sizeof(x));
        snprintf(name,sizeof(name),"late-%d-%s-velocity.f32",step,kind);load(fixtures,name,v,sizeof(v));
        const float *sigma=modality?s.audio:s.video;
        CHECK(h3_sglang_euler(x,v,4096,sigma[step],sigma[step+1]));
        snprintf(name,sizeof(name),"late-%d-%s-output.f32",step,kind);save(out,name,x,sizeof(x));
    }
    float rows[96*12];for(int i=0;i<96*12;i++)rows[i]=noise[i];
    CHECK(h3_sglang_video_condition(rows,3,4,4,7,2,42));save(out,"condition-packed.f32",rows,sizeof(rows));
    CHECK(h3_sglang_audio_condition(rows,64,42));
    uint16_t *ref=evaluate(1),*fast=evaluate(0),*again=evaluate(1);
    CHECK(!memcmp(ref,again,E*2));save(out,"reference-ops.bf16",ref,E*2);free(again);
    again=evaluate(0);CHECK(!memcmp(fast,again,E*2));free(again);free(fast);free(ref);
    /* Head-major reference norm/RoPE and production dense attention at two sizes. */
    int old=h3_sglang_exchange(1);h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));h3_sglang_exchange(old);CHECK(g);
    for(unsigned rows_n=17;rows_n<=513;rows_n+=496) {
        const unsigned heads=2,dim=128;size_t count=(size_t)rows_n*heads*dim;
        uint16_t *qkv=malloc(count*6),*host=malloc(count*2),*trig=malloc((size_t)rows_n*64*2),norm[128];CHECK(qkv&&host&&trig);
        for(size_t i=0;i<(size_t)rows_n*64;i++)trig[i]=(uint16_t)(0x3e00+i%128);
        for(size_t i=0;i<count*3;i++)qkv[i]=(uint16_t)(0x3c00+i%511);
        for(int i=0;i<128;i++)norm[i]=0x3f80;
        h3_gpu_tensor *x=h3_gpu_tensor_from_bf16(g,qkv,count*3),*w=h3_gpu_tensor_from_bf16(g,norm,128),
            *q=h3_gpu_tensor_new_bf16(g,count),*k=h3_gpu_tensor_new_bf16(g,count),*v=h3_gpu_tensor_new_bf16(g,count),
            *y=h3_gpu_tensor_new_bf16(g,count),*c=h3_gpu_tensor_from_bf16(g,trig,(size_t)rows_n*64);
        CHECK(x&&w&&q&&k&&v&&y&&c);CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_grouped_qkv_rope_bf16(g,q,k,v,x,w,w,c,c,rows_n,heads,dim,64,1e-5f));
        CHECK(h3_gpu_sdpa_bf16(g,y,q,k,v,rows_n,heads,dim,1.f/sqrtf(dim)));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_bf16(y,host,count));char name[64];snprintf(name,sizeof(name),"attention-%u.bf16",rows_n);save(out,name,host,count*2);
        h3_gpu_tensor_free(x);h3_gpu_tensor_free(w);h3_gpu_tensor_free(q);h3_gpu_tensor_free(k);h3_gpu_tensor_free(v);h3_gpu_tensor_free(y);h3_gpu_tensor_free(c);free(qkv);free(host);free(trig);
    }
    h3_gpu_free(g);decoder_failure_recovery();host_weight_cache();puts("PASS frozen host/operator/isolation probes");return 0;
}
