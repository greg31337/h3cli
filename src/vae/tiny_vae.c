/* TAEH3 decoder architecture: Ollin Boer Bohan, MIT (third_party/taeh3/LICENSE).
 * Native implementation; normalized H3 diffusion latents, RGB in [0,1]. */
#include "src/vae/tiny_vae_internal.h"
#include "src/vae/video_posterior.h"
#include "src/digest.h"
#include "src/memory.h"
#include "src/host.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

struct h3_tiny_vae { void *backend; char digest[65], policy[256]; };
static int policy(char out[256]) {
#ifdef __APPLE__
    out[0]=0;return 1;
#else
    const char *device=getenv("H3_CUDA_DEVICE"),*conv=getenv("H3_PREVIEW_CUDA_CONV"),*scratch=getenv("H3_PREVIEW_CUDA_WORKSPACE_MB");
    int n=snprintf(out,256,"%s|%s|%s",device?device:"0",conv?conv:"auto",scratch?scratch:"32");
    return n>=0&&n<256;
#endif
}
static int fail(char *e,size_t n,const char *s) { if(e&&n)snprintf(e,n,"%s",s);return 0; }
int h3_tiny_vae_frames(int t) { return t>=7 && t<=107 && t%5==2 ? (t-2)/5*17+5 : 0; }
uint64_t h3_tiny_vae_memory_reserve(int h,int w,int batch) {
    if(h<2||w<2||(int64_t)h*w>H3_MAX_PIXELS/256||batch<1||batch>5)return UINT64_MAX;
#ifdef __APPLE__
    const uint64_t fixed=UINT64_C(128)*1024*1024;
#else
    /* Also covers the largest optional (128 MiB) CUDA convolution workspace. */
    const uint64_t fixed=UINT64_C(256)*1024*1024;
#endif
    return fixed+(uint64_t)(unsigned)h*(unsigned)w*(unsigned)batch*262144;
}
static void weight(h3_tiny_weight *a,int *n,const char *name,int out,int in,int k,int bias) {
    h3_tiny_weight *v=&a[(*n)++];
    snprintf(v->name,sizeof(v->name),"%s.weight",name);v->ndim=4;
    v->shape[0]=(uint64_t)out;v->shape[1]=(uint64_t)in;v->shape[2]=(uint64_t)k;v->shape[3]=(uint64_t)k;
    if(bias){v=&a[(*n)++];snprintf(v->name,sizeof(v->name),"%s.bias",name);v->ndim=1;v->shape[0]=(uint64_t)out;}
}
int h3_tiny_weights(h3_tiny_weight a[64]) {
    memset(a,0,64*sizeof(*a));int n=0;
    weight(a,&n,"decoder.1",256,24,3,1);
    const int ids[]={3,4,5,9,10,11,15,16,17};
    for(int i=0;i<9;i++){int c=i<3?256:i<6?128:64;
        for(int j=0;j<3;j++){char name[64];snprintf(name,sizeof(name),"decoder.%d.conv.%d",ids[i],j*2);weight(a,&n,name,c,j?c:2*c,3,1);}}
    weight(a,&n,"decoder.7.conv",256,256,1,0);weight(a,&n,"decoder.8",128,256,3,0);
    weight(a,&n,"decoder.13.conv",256,128,1,0);weight(a,&n,"decoder.14",64,128,3,0);
    weight(a,&n,"decoder.19.conv",128,64,1,0);weight(a,&n,"decoder.20",64,64,3,0);
    weight(a,&n,"decoder.22",12,64,3,1);return n;
}
int h3_tiny_vae_validate(const char *path,char digest[65],char *error,size_t size) {
    if(!path||!*path)return fail(error,size,"preview VAE requires a nonempty model path");
    h3_st_header h={0};if(!h3_st_read_header(path,&h,error,size))return 0;
    h3_tiny_weight expected[64];int count=h3_tiny_weights(expected),ok=1,actual=0;
    for(size_t i=0;i<h.tensor_count;i++){
        if(!strncmp(h.tensors[i].name,"decoder.",8))actual++;
        else if(strncmp(h.tensors[i].name,"encoder.",8))ok=0;
    }
    if(actual!=count)ok=0;
    for(int i=0;i<count&&ok;i++){
        const h3_st_tensor *v=h3_st_find(&h,expected[i].name);
        if(!v||v->ndim!=expected[i].ndim||v->dtype!=H3_DTYPE_F16){ok=0;break;}
        for(int j=0;j<v->ndim;j++)if(v->shape[j]!=expected[i].shape[j])ok=0;
        if(!ok)break;
        size_t bytes=(size_t)(v->data_end-v->data_begin);
        uint16_t *raw=malloc(bytes);if(!raw){ok=0;break;}
        ok=h3_st_read_data(&h,v,raw,bytes,error,size);
        for(size_t j=0;j<bytes/2&&ok;j++)if((raw[j]&0x7c00)==0x7c00)ok=0;
        free(raw);
    }
    h3_st_free_header(&h);
    if(!ok)return fail(error,size,"incompatible or non-finite TAEH3 decoder weights (expected pinned F16 architecture)");
    FILE *f=fopen(path,"rb");if(!f)return fail(error,size,"cannot hash preview VAE");
    h3_sha256_ctx hash;h3_sha256_init(&hash);uint8_t buf[65536],out[32];size_t n;
    while((n=fread(buf,1,sizeof(buf),f)))h3_sha256_update(&hash,buf,(h3_sha256_size)n);
    ok=!ferror(f);fclose(f);if(!ok)return fail(error,size,"cannot read preview VAE digest");
    h3_sha256_final(out,&hash);if(digest)for(int i=0;i<32;i++)snprintf(digest+2*i,3,"%02x",out[i]);
    return 1;
}
h3_tiny_vae *h3_tiny_vae_load(const char *path,char *error,size_t size){
    h3_tiny_vae *d=calloc(1,sizeof(*d));if(!d){fail(error,size,"out of memory creating preview VAE");return NULL;}
    if(!policy(d->policy)){free(d);fail(error,size,"preview VAE policy exceeds supported length");return NULL;}
    if(!h3_tiny_vae_validate(path,d->digest,error,size)||(d->backend=h3_tiny_backend_load(path,error,size))==NULL){h3_tiny_vae_free(d);return NULL;}
    return d;
}
void h3_tiny_vae_free(h3_tiny_vae *d){if(d){h3_tiny_backend_free(d->backend);free(d);}}
const char *h3_tiny_vae_digest(const h3_tiny_vae *d){return d?d->digest:"";}
int h3_tiny_vae_cache_matches(const h3_tiny_vae *d,const char *digest){
    char current[256];return d&&digest&&!strcmp(d->digest,digest)&&policy(current)&&!strcmp(current,d->policy);
}
int h3_tiny_vae_stream(h3_tiny_vae *d,const float *z,int t,int h,int w,int batch,int stop,
 h3_video_batch_callback callback,void *opaque,h3_video_vae_progress progress,void *po,char *error,size_t size){
    int frames=h3_tiny_vae_frames(t);
    if(!d||!z||!callback||!frames||h<2||w<2||(int64_t)h*w>H3_MAX_PIXELS/256||batch<1||batch>5||stop>=frames||stop< -1)
        return fail(error,size,"invalid preview VAE decode geometry/batch");
    size_t hw=(size_t)h*(size_t)w, pixels=hw*256*3;
    for(size_t i=0;i<24*(size_t)t*hw;i++)if(!isfinite(z[i]))return fail(error,size,"non-finite preview VAE input");
    float *input=malloc((size_t)batch*hw*24*sizeof(float));
    if(!input)return fail(error,size,"out of memory preparing preview VAE input");
    h3_tiny_backend_reset(d->backend);int ok=1,delivered=0;
    for(int start=0;start<t&&ok;start+=batch){
        int count=t-start<batch?t-start:batch;
        /* Conservative activation/graph allowance, proportional to this batch,
         * plus immutable weights and at most two compiled plans. */
        uint64_t reserve=h3_tiny_vae_memory_reserve(h,w,count);
        if(!h3_memory_check(reserve,"preview VAE decode",error,size)){ok=0;break;}
        for(int b=0;b<count;b++)for(size_t p=0;p<hw;p++)for(int c=0;c<24;c++)
            input[((size_t)b*hw+p)*24+(size_t)c]=z[((size_t)c*(size_t)t+(size_t)(start+b))*hw+p];
        float *rgb=NULL;ok=h3_tiny_backend_chunk(d->backend,input,count,h,w,&rgb,error,size);
        int kept=0;
        if(ok)for(int i=0;i<count*4;i++)if((start*4+i)%20>=3){
            if(stop>=0 && delivered+kept>stop)break;
            float *src=rgb+(size_t)i*pixels;
            for(size_t p=0;p<pixels;p++)if(!isfinite(src[p])){ok=fail(error,size,"non-finite preview VAE output");break;}
            if(!ok)break;
            memmove(rgb+(size_t)kept*pixels,src,pixels*sizeof(float));kept++;
        }
        if(ok&&kept&&callback(rgb,delivered,kept,w*16,h*16,opaque))ok=fail(error,size,"preview VAE delivery cancelled");
        delivered+=kept;free(rgb);
        if(ok&&progress&&progress(delivered,stop>=0?stop+1:frames,po))ok=fail(error,size,"preview VAE decode cancelled");
        if(stop>=0&&delivered>stop)break;
    }
    h3_tiny_backend_profile(d->backend);
    h3_tiny_backend_reset(d->backend);free(input);return ok;
}
typedef struct{h3_video_frames *out;int selected;} gather;
static int collect(const float *rgb,int first,int count,int w,int h,void *opaque){
    gather *g=opaque;size_t frame=(size_t)w*(size_t)h*3;
    if(g->selected>=0){if(g->selected>=first&&g->selected<first+count)memcpy(g->out->rgb,rgb+(size_t)(g->selected-first)*frame,frame*sizeof(float));
    } else
        memcpy(g->out->rgb + (size_t)first * frame, rgb, (size_t)count * frame * sizeof(float));
    return 0;
}
int h3_tiny_vae_decode(h3_tiny_vae *d,const float *z,int t,int h,int w,int selected,h3_video_frames *out,
 h3_video_vae_progress progress,void *opaque,char *error,size_t size){
    if (!out)
        return fail(error, size, "missing preview VAE output");
    memset(out, 0, sizeof(*out));
    int frames=h3_tiny_vae_frames(t);if(!frames||h<2||w<2||(int64_t)h*w>H3_MAX_PIXELS/256||selected< -1||selected>=frames)return fail(error,size,"invalid preview geometry");
    out->frames=selected>=0?1:frames;out->height=h*16;out->width=w*16;
    size_t n=(size_t)out->frames*(size_t)h*(size_t)w*256*3;
    if(!h3_memory_check(n*sizeof(float),"preview VAE output",error,size))return 0;
    out->rgb=malloc(n*sizeof(float));if(!out->rgb)return fail(error,size,"out of memory collecting preview frames");
    gather g={out,selected};int ok=h3_tiny_vae_stream(d,z,t,h,w,5,selected,collect,&g,progress,opaque,error,size);
    if (!ok)
        h3_video_frames_free(out);
    return ok;
}
