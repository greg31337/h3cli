#include "src/sglang/sglang.h"
#include "src/execution.h"
#include <math.h>
#include <string.h>
#include <stdio.h>
#include <stdlib.h>
static _Thread_local int requested;
static _Thread_local size_t capture_bytes;
int h3_sglang_exchange(int enabled) { int old=requested; requested=enabled; capture_bytes=0; return old; }
int h3_sglang_requested(void) { return requested; }
int h3_sglang_exact_requested(void) { return requested && !h3_cuda_policy_current().attention && !h3_cuda_policy_current().projection_precision; }
int h3_sglang_resolve(const h3_params *p,const char *backend,int lora_active) {
    (void)lora_active;
    /* Shared preparation/sampler recipe, independent of attention, precision and delivery.
     * Validation reports incompatible options instead of silently downgrading. */
    if(!backend || strcmp(backend,"cuda") || p->still)return 0;
    return H3_SGLANG_VERSION;
}
int h3_sglang_reference_image_canvas(int width,int height,int *out_width,int *out_height) {
    h3_reference_image_shape shape;
    if(!out_width||!out_height||!h3_reference_image_resolve(width,height,0,0,
        H3_REFERENCE_IMAGE_MAX,&shape,NULL,0))return 0;
    *out_width=shape.width;*out_height=shape.height;return 1;
}
void h3_sglang_vae_byte_pixels(float *pixels,size_t count,int restore) {
    for(size_t i=0;i<count;i++) {
        float byte=roundf(pixels[i]*255.f);
        pixels[i]=restore?byte/255.f:byte*(1.f/255.f);
    }
}
int h3_sglang_capture_step(int step,int total) {
    const char *selection=getenv("H3_SGLANG_CAPTURE_STEPS");
    if(!selection)return step==0||step==total/2||step==total-1;
    if(!strcmp(selection,"none"))return 0;
    if(!*selection)return -1;
    int matched=0;
    while(*selection) {
        if(*selection<'0'||*selection>'9')return -1;
        unsigned value=0;
        while(*selection>='0'&&*selection<='9') {
            value=value*10u+(unsigned)(*selection++-'0');if(value>=1000)return -1;
        }
        if(value>=(unsigned)total)return -1;
        matched|=value==(unsigned)step;
        if(!*selection)break;
        if(*selection++!=','||!*selection)return -1;
    }
    return matched;
}
int h3_sglang_dump(const char *name,const void *data,size_t bytes) {
    const char *directory=getenv("H3_TEST_SGLANG_DIR");
    if(!directory||!*directory)return 1;
    if(!name||strchr(name,'/')||(!data&&bytes)||bytes>UINT64_C(1024)*1024*1024)return 0;
    if(capture_bytes>UINT64_C(8)*1024*1024*1024-bytes)return 0;
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",directory,name);
    if(n<0||n>=(int)sizeof(path))return 0;
    FILE *f=fopen(path,"wbx");if(!f)return 0;
    int ok=fwrite(data,1,bytes,f)==bytes;if(fclose(f))ok=0;
    if (ok)
        capture_bytes += bytes;
    return ok;
}
static int read_fixture(const char *directory,const char *name,void *data,size_t bytes) {
    char path[4096];int n=snprintf(path,sizeof(path),"%s/%s",directory,name);
    if(n<0||n>=(int)sizeof(path))return 0;
    FILE *f=fopen(path,"rb");if(!f)return 0;
    int ok=fread(data,1,bytes,f)==bytes && fgetc(f)==EOF && !ferror(f);
    if (fclose(f))
        ok = 0;
    return ok;
}
int h3_sglang_import_text(const uint32_t *tokens,size_t rows,uint16_t *text,size_t width) {
    const char *directory=getenv("H3_TEST_SGLANG_INPUT_DIR");
    if(!directory||!*directory)return 1;
    const char *budget=getenv("H3_TEST_MAX_EVALUATIONS");
    if(!h3_sglang_requested()||!budget||(strcmp(budget,"6")&&strcmp(budget,"50"))||
       !tokens||!text||!rows||rows>100000||width!=5120)return 0;
    uint32_t *ids=malloc(rows*sizeof(*ids));
    uint16_t *values=malloc(rows*width*sizeof(*values));
    int ok=ids&&values&&read_fixture(directory,"tokens.u32",ids,rows*sizeof(*ids))&&
        !memcmp(ids,tokens,rows*sizeof(*ids))&&read_fixture(directory,"text.bf16",values,rows*width*sizeof(*values));
    for(size_t i=0;ok&&i<rows*width;i++)if((values[i]&0x7f80)==0x7f80)ok=0;
    if(ok)memcpy(text,values,rows*width*sizeof(*values));
    free(ids);free(values);return ok;
}
int h3_sglang_import_conditions(float *video,size_t nv,float *audio,size_t na) {
    const char *directory=getenv("H3_TEST_SGLANG_CONDITION_DIR");
    if(!directory||!*directory)return 1;
    const char *budget=getenv("H3_TEST_MAX_EVALUATIONS");
    if(!h3_sglang_requested()||!budget||(strcmp(budget,"6")&&strcmp(budget,"50"))||
       nv>(256u<<20)/sizeof(float)||na>(64u<<20)/sizeof(float)||(!video&&nv)||(!audio&&na))return 0;
    if((nv&&!read_fixture(directory,"condition-video.f32",video,nv*sizeof(float)))||
       (na&&!read_fixture(directory,"condition-audio.f32",audio,na*sizeof(float))))return 0;
    for(size_t i=0;i<nv;i++)if(!isfinite(video[i]))return 0;
    for(size_t i=0;i<na;i++)if(!isfinite(audio[i]))return 0;
    fprintf(stderr,"h3cli: test-only imported conditioned rows; native conditioning qualification bypassed\n");
    return 1;
}

int h3_sglang_schedule(int evaluations,h3_sigma_schedule *out) {
    if(!out || evaluations<1 || evaluations>H3_MAX_STEPS)return 0;
    memset(out,0,sizeof(*out));out->steps=evaluations;
    float step=-1.0f/(float)evaluations;
    for(int i=0;i<=evaluations;i++) {
        /* CPU torch.linspace computes the second half from the endpoint. */
        float t=i<(evaluations+1)/2 ? fmaf(step,(float)i,1.0f) : -step*(float)(evaluations-i);
        out->video[i]=(12.0f*t)/(1.0f+11.0f*t);
        out->audio[i]=(3.0f*t)/(1.0f+2.0f*t);
    }
    return 1;
}
int h3_sglang_euler(float *state,const float *velocity,size_t count,float sigma,float next) {
    if(!state||!velocity||!isfinite(sigma)||!isfinite(next)||sigma<=next||next<0)return 0;
    float ratio=next/sigma, complement=1.0f-ratio;
    /* Upstream computes sigma_t through its stored FP32 conditioning time. */
    float t=1.0f-sigma, sigma_t=1.0f-t;
    for(size_t i=0;i<count;i++) {
        float scaled=sigma_t*velocity[i];
        float clean=state[i]+scaled;
        float a=ratio*state[i], b=complement*clean;
        state[i]=a+b;
    }
    return 1;
}
#ifdef __APPLE__
int h3_sglang_normal(uint64_t seed,float *out,size_t count) {
    (void)seed;(void)out;(void)count;return 0;
}
#endif
int h3_sglang_video_condition(float *rows,int time,int height,int width,
                              int target_time,int visual_conditions,uint64_t seed) {
    if(!rows||time<1||height<2||width<2||(height%2)||(width%2)||target_time<1||
       target_time>1000||visual_conditions<1||visual_conditions>100||time>target_time+visual_conditions)return 0;
    size_t full_time=(size_t)target_time+(size_t)visual_conditions,area=(size_t)height*(size_t)width;
    if(area>(512u<<20)/sizeof(float)/24/full_time)return 0;
    size_t n=24*full_time*area;float *noise=malloc(n*sizeof(float));if(!noise)return 0;
    int ok=h3_sglang_normal(seed,noise,n);
    const float timestep=.999f,complement=1.f-timestep;
    if(ok)for(int t=0;t<time;t++)for(int y=0;y<height/2;y++)for(int x=0;x<width/2;x++) {
        size_t row=((size_t)t*(size_t)(height/2)+(size_t)y)*(size_t)(width/2)+(size_t)x;
        for(int c=0;c<96;c++) {
            size_t source=((size_t)(c/4)*full_time+(size_t)t)*area+
                (size_t)(2*y+c/2%2)*(size_t)width+(size_t)(2*x+c%2);
            float a=timestep*rows[row*96+(size_t)c],b=complement*noise[source];rows[row*96+(size_t)c]=a+b;
        }
    }
    free(noise);return ok;
}
int h3_sglang_audio_condition(float *rows,size_t elements,uint64_t seed) {
    /* Pinned MINIMAX_H3_AUDIO_REF_COND_TIMESTEP is 1.0. Visual conditions
     * use 0.999, but audio references remain clean and consume no draw. */
    (void)seed;
    if(!rows||!elements||elements%64||elements>(64u<<20)/sizeof(float))return 0;
    for(size_t i=0;i<elements;i++)if(!isfinite(rows[i]))return 0;
    return 1;
}
