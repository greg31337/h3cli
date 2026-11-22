#include "src/vae/image_vae.h"
#include "src/host.h"
#include "src/memory.h"
#include <errno.h>
#include <math.h>
#include <stdarg.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#include <sys/stat.h>

static int fail(char *error, size_t size, const char *format, ...) {
    if (error && size) { va_list ap; va_start(ap, format); vsnprintf(error, size, format, ap); va_end(ap); }
    return 0;
}
static const char *meta(const h3_st_header *h, const char *key) {
    for (size_t i = 0; i < h->metadata_count; i++)
        if (!strcmp(h->metadata_keys[i], key)) return h->metadata_values[i];
    return NULL;
}
static int eq(const char *a, const char *b) { return a && !strcmp(a,b); }
int h3_still_protect_vae(const char *directory,const char *output,
    const char *latent_output,char *error,size_t size) {
    const char *files[]={"model.safetensors","config.json","../config.json"};
    const char *targets[]={output,latent_output};
    for(size_t i=0;i<3;i++) {
        char *path=NULL;
        if(asprintf(&path,"%s/%s",directory,files[i])<0)
            return fail(error,size,"cannot validate still output destination");
        for(size_t j=0;j<2;j++)if(targets[j]) {
            struct stat a,b;
            int collision=!strcmp(path,targets[j]) ||
                (!stat(path,&a) && !stat(targets[j],&b) && a.st_dev==b.st_dev && a.st_ino==b.st_ino);
            if(collision){free(path);return fail(error,size,"still output must not overwrite stock VAE weights or configuration");}
        }
        free(path);
    }
    return 1;
}
int h3_still_geometry(int h, int w, char *error, size_t size) {
    if (h < 2 || w < 2 || h % 2 || w % 2 ||
        h > H3_STILL_MAX_DIMENSION / 16 || w > H3_STILL_MAX_DIMENSION / 16 ||
        (uint64_t)(unsigned)h * (unsigned)w > H3_STILL_MAX_PIXELS / 256)
        return fail(error,size,"still canvas must be multiples of 32, at least 32x32, <= %d pixels and <= %d per side",H3_STILL_MAX_PIXELS,H3_STILL_MAX_DIMENSION);
    return 1;
}
static int digest_valid(const char *s) {
    if(!s || strnlen(s,65)!=64)return 0;
    for(int i=0;i<64;i++)if(!((s[i]>='0'&&s[i]<='9')||(s[i]>='a'&&s[i]<='f')))return 0;
    return 1;
}
int h3_still_latent_load(const char *path,h3_still_latent *out,char *error,size_t size) {
    if(!out)return fail(error,size,"missing still latent output");
    memset(out,0,sizeof(*out));h3_st_header h={0};
    if(!h3_st_read_header(path,&h,error,size))return 0;
    const h3_st_tensor *t=h3_st_find(&h,"latent");
    const char *compat=meta(&h,"h3_vae_compatibility");
    int ok=h.tensor_count==1 && t && t->dtype==H3_DTYPE_F32 && t->ndim==5 &&
        t->shape[0]==1 && t->shape[1]==24 && t->shape[2]==1 && t->shape[3]<=INT32_MAX && t->shape[4]<=INT32_MAX &&
        eq(meta(&h,"h3_still_schema"),"2") &&
        eq(meta(&h,"h3_decoder_recipe"),H3_STILL_DECODER_RECIPE) &&
        (eq(meta(&h,"h3_model_variant"),"FL2VA") || eq(meta(&h,"h3_model_variant"),"Ref2VA")) && eq(meta(&h,"h3_latent_space"),"h3-normalized-v1") && digest_valid(compat);
    if(!ok)fail(error,size,"invalid still latent: expected schema 2 stock-video recipe, normalized F32 [1,24,1,h,w] and VAE compatibility; schema 1 belongs to the retired image VAE, regenerate with --still");
    if(ok)ok=h3_still_geometry((int)t->shape[3],(int)t->shape[4],error,size);
    size_t count=ok?(size_t)(24*t->shape[3]*t->shape[4]):0;
    if(ok)ok=h3_memory_check(count*4,"still latent load",error,size);
    if(ok) {out->values=malloc(count*4);if(!out->values)ok=fail(error,size,"cannot allocate still latent");}
    if(ok)ok=h3_st_read_data(&h,t,out->values,count*4,error,size);
    for(size_t i=0;ok&&i<count;i++)if(!isfinite(out->values[i]))ok=fail(error,size,"nonfinite still latent");
    if(ok) {out->reference_model=eq(meta(&h,"h3_model_variant"),"Ref2VA");out->height=(int)t->shape[3];out->width=(int)t->shape[4];memcpy(out->compatibility_sha256,compat,65);}
    h3_st_free_header(&h);if(!ok)h3_still_latent_free(out);return ok;
}
int h3_still_latent_save(const char *path,const h3_still_latent *z,char *error,size_t size) {
    if(!path || !*path || !z || !z->values || !digest_valid(z->compatibility_sha256) || (z->reference_model!=0 && z->reference_model!=1))return fail(error,size,"invalid still latent save arguments");
    if(!h3_still_geometry(z->height,z->width,error,size))return 0;
    size_t count=(size_t)24*z->height*z->width;
    for(size_t i=0;i<count;i++)if(!isfinite(z->values[i]))return fail(error,size,"nonfinite still latent");
    char header[768];int n=snprintf(header,sizeof(header),"{\"__metadata__\":{\"h3_still_schema\":\"2\",\"h3_latent_space\":\"h3-normalized-v1\",\"h3_vae_compatibility\":\"%s\",\"h3_decoder_recipe\":\"%s\",\"h3_model_variant\":\"%s\",\"producer\":\"h3cli\"},\"latent\":{\"dtype\":\"F32\",\"shape\":[1,24,1,%d,%d],\"data_offsets\":[0,%zu]}}",z->compatibility_sha256,H3_STILL_DECODER_RECIPE,z->reference_model?"Ref2VA":"FL2VA",z->height,z->width,count*4);
    if(n<0 || (size_t)n+8>=sizeof(header))return fail(error,size,"still latent header overflow");
    while(n%8)header[n++]=' ';
    char *parent=strdup(path);if(!parent)return fail(error,size,"out of memory resolving still latent path");
    for(char *p=parent+1;*p;p++)if(*p=='/') {
        *p=0;int bad=mkdir(parent,0755) && errno!=EEXIST;*p='/';
        if(bad){free(parent);return fail(error,size,"cannot create still latent directory: %s",strerror(errno));}
    }
    free(parent);
    char *tmp=malloc(strlen(path)+16);if(!tmp)return fail(error,size,"out of memory staging still latent");
    sprintf(tmp,"%s.part-XXXXXX",path);int fd=mkstemp(tmp);FILE *f=fd>=0?fdopen(fd,"wb"):NULL;
    uint64_t length=(uint64_t)n;
    int ok=f && fwrite(&length,8,1,f)==1 && fwrite(header,1,(size_t)n,f)==(size_t)n && fwrite(z->values,4,count,f)==count && !fflush(f) && !fsync(fd);
    if(f) {if(fclose(f))ok=0;} else if(fd>=0)close(fd);
    if(ok)ok=rename(tmp,path)==0;
    if(!ok){unlink(tmp);fail(error,size,"cannot publish still latent: %s",strerror(errno));}
    free(tmp);return ok;
}
void h3_still_latent_free(h3_still_latent *z) {if(z){free(z->values);memset(z,0,sizeof(*z));}}

int h3_still_options(const h3_params *p,char *error,size_t size) {
    if(!p)return fail(error,size,"missing still parameters");
    if(p->image_vae)return fail(error,size,"--image-vae is retired; use --still or --decode-still-latent with --model-dir and the stock video VAE");
    if(!p->still) {
        if(p->save_still_latent)return fail(error,size,"still latent output requires --still");
        return 1;
    }
    if(p->still!=1 || p->frames!=1)
        return fail(error,size,"still generation requires exactly one frame");
    if(p->width%16 || p->height%16)
        return fail(error,size,"still canvas dimensions must be multiples of 32");
    if(!h3_still_geometry(p->height/16,p->width/16,error,size))return 0;
    const char *path=p->output_path;size_t n=path?strlen(path):0;
    if(path && (n<4 || strcmp(path+n-4,".png")))return fail(error,size,"still output must use .png");
    if(p->preview_vae || p->preview_vae_model || p->preview_denoise || p->first_frame || p->last_frame ||
        p->continuation || p->continuation_mode || p->keep_continuation_prefix || p->stop_after_step>=0 ||
        p->save_sampler_state || p->resume_sampler_state || p->preview_on_stop || p->save_conditioning || p->load_conditioning || p->conditioning_schedule || p->render_width || p->render_height)
        return fail(error,size,"still mode does not support video preview, anchors, continuation, sampler/conditioning files or a separate render canvas");
    if(p->dit_layers!=50 || p->denoise_reuse!=1 || p->core_reuse!=1 || p->token_reduction ||
        p->backend || p->attention_mode || p->metal_attention.weight_format || p->metal_attention.ane_mode ||
        p->cuda_attention || p->cuda_denoise_quant || p->use_int8_row_fc2)
        return fail(error,size,"still mode requires dense BF16, all 50 blocks, no reuse/quantization/acceleration");
    if(p->reference_count && !p->references)return fail(error,size,"missing still references");
    for(size_t i=0;i<p->reference_count;i++) {
        if(p->references[i].kind!=H3_REFERENCE_IMAGE)return fail(error,size,"still mode currently supports only ordered image references");
        const char *source=p->references[i].path,*targets[]={p->output_path,p->save_still_latent};
        for(int k=0;source && k<2;k++)if(targets[k]) {
            struct stat a,b;
            if(!strcmp(source,targets[k]) || (!stat(source,&a) && !stat(targets[k],&b) && a.st_dev==b.st_dev && a.st_ino==b.st_ino))
                return fail(error,size,"still output must not overwrite a reference image");
        }
    }
    if(p->save_still_latent) {
        if(!*p->save_still_latent)return fail(error,size,"still latent output path must not be empty");
        const char *other[]={p->output_path};
        for(int i=0;i<1;i++)if(other[i]) {
            struct stat a,b;
            if(!strcmp(p->save_still_latent,other[i]) || (!stat(p->save_still_latent,&a) && !stat(other[i],&b) && a.st_dev==b.st_dev && a.st_ino==b.st_ino))
                return fail(error,size,"still latent output must not overwrite PNG output");
        }
    }
    const char *blocks=getenv("H3_TEST_DIT_BLOCKS"),*reduction=getenv("H3_TOKEN_REDUCTION");
    if((blocks && strcmp(blocks,"50")) || (reduction && strcmp(reduction,"0")) || getenv("H3_REUSE_STEPS"))
        return fail(error,size,"still baseline requires all 50 blocks, no token reduction or custom reuse schedule");
    const char *tile=getenv("H3_VAE_TILE_PIXELS");
    if(tile && strcmp(tile,"256"))return fail(error,size,"still mode requires released 256/64 VAE tiles");
    return 1;
}
