#include "src/vae/image_vae.h"
#include "src/weights/lora_json.h"
#include "src/digest.h"
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
static int number(const lj_value *o, const char *key, double n) {
    const lj_value *v = lj_get(o,key); return v && v->type == LJ_NUMBER && v->number == n;
}
static int string(const lj_value *o, const char *key, const char *s) {
    const lj_value *v = lj_get(o,key); return v && v->type == LJ_STRING && eq(v->text,s);
}
static int boolean(const lj_value *o, const char *key, int b) {
    const lj_value *v = lj_get(o,key); return v && v->type == LJ_BOOL && (v->number != 0) == b;
}
static int null_value(const lj_value *o, const char *key) {
    const lj_value *v = lj_get(o,key); return v && v->type == LJ_NULL;
}
static int array(const lj_value *o, const char *key, const int *values, size_t n) {
    const lj_value *v = lj_get(o,key);
    if (!v || v->type != LJ_ARRAY || v->count != n) return 0;
    for (size_t i=0;i<n;i++) if (v->items[i]->type != LJ_NUMBER || v->items[i]->number != values[i]) return 0;
    return 1;
}
int h3_still_geometry(int h, int w, char *error, size_t size) {
    if (h < 2 || w < 2 || h % 2 || w % 2 ||
        (uint64_t)(unsigned)h * (unsigned)w > H3_MAX_PIXELS / 256)
        return fail(error,size,"still canvas must be multiples of 32, at least 32x32, and <= %d pixels",H3_MAX_PIXELS);
    return 1;
}
static const struct { const char *name; int ndim; uint64_t shape[8]; } schema[] = {
#include "src/vae/image_vae_schema.inc"
};
int h3_image_vae_validate(const h3_st_header *h, h3_image_vae_info *info, char *error, size_t size) {
    if (error && size) *error=0;
    if (!h || !info) return fail(error,size,"image VAE descriptor is required");
    memset(info,0,sizeof(*info));
    if ((!eq(meta(h,"h3_t1_direct"),"true") && !eq(meta(h,"h3_t1_direct"),"1")) ||
        !eq(meta(h,"h3_t1_format"),"full_decoder_v1"))
        return fail(error,size,"image VAE requires h3_t1_direct=true and full_decoder_v1; stock video VAE cannot decode a still");
    const char *slice=meta(h,"h3_t1_output_slice"), *config=meta(h,"minimax_h3_video_vae");
    if (!slice || strlen(slice)!=1 || *slice<'0' || *slice>'3')
        return fail(error,size,"image VAE requires explicit output slice 0..3");
    info->output_slice=*slice-'0';
    if (!config || strlen(config)>65536) return fail(error,size,"missing or oversized image VAE config");
    lj_value *root=lj_parse(config,strlen(config),error,size);
    if (!root) return 0;
    const lj_value *c=lj_get(root,"source_config"), *vit=lj_get(c,"vit_decoder_kwargs");
    int ok=number(c,"embed_dim",24) && number(c,"z_channels",24) && number(c,"vae_ratio",16) &&
        number(c,"vae_ratio_t",4) && number(c,"in_channels",3) && number(c,"out_ch",3) &&
        number(c,"ch",128) && number(c,"num_res_blocks",2) && number(c,"scaling_factor",1) && number(c,"shift_factor",0) &&
        null_value(c,"num_res_blocks_decoder") && null_value(c,"padding_mode_t") &&
        null_value(c,"zq_ch_encoder") && null_value(c,"zq_ch_decoder") && null_value(c,"time_up") &&
        string(c,"pixel_norm_type","imagenet") && string(c,"padding_mode","reflect") &&
        boolean(c,"causal_encoder",1) && boolean(c,"causal_decoder",0) && boolean(c,"use_vit_decoder",1) &&
        boolean(c,"use_3d_conv",1) && boolean(c,"use_t_isolated_gn",1) &&
        array(c,"ch_mult",(int[]){1,2,2,4,4,8},6) && array(c,"space_down",(int[]){2,2,2,2,1,1},6) &&
        array(c,"time_down",(int[]){1,2,2,1,1,1},6) &&
        array(c,"space_up",(int[]){1,2,2,2,2,1},6) &&
        number(vit,"num_layers",36) && number(vit,"heads",32) && number(vit,"dim_head",64) &&
        number(vit,"rope_dim_ratio",.75) && number(vit,"rope_theta",100) &&
        string(vit,"norm_type","rms_norm") && boolean(vit,"norm_affine",1) &&
        string(vit,"qk_norm_type","rms_norm") && boolean(vit,"qk_norm_affine",0) &&
        string(vit,"ffn_activation_fn","silu") && boolean(vit,"ffn_use_gated",1);
    if (!ok) fail(error,size,"unsupported image VAE architecture");
    const char *keys[]={"latents_mean","latents_std"};
    for (int a=0;ok && a<2;a++) {
        const lj_value *v=lj_get(root,keys[a]);
        if (!v || v->type!=LJ_ARRAY || v->count!=24) { ok=fail(error,size,"image VAE requires 24 %s values",keys[a]); break; }
        for (int i=0;i<24;i++) {
            float n=(float)v->items[i]->number;
            if(v->items[i]->type!=LJ_NUMBER || !isfinite(n) || (a && n<=0)) { ok=fail(error,size,"invalid image VAE whitening");break; }
            (a?info->deviation:info->mean)[i]=n;
        }
    }
    lj_free(root);
    if (!ok) return 0;
    if(h->tensor_count!=sizeof(schema)/sizeof(*schema)) return fail(error,size,"image VAE requires complete 562-tensor inventory");
    for(size_t i=0;i<sizeof(schema)/sizeof(*schema);i++) {
        const h3_st_tensor *t=h3_st_find(h,schema[i].name);
        if(!t || t->dtype!=H3_DTYPE_F16 || t->ndim!=schema[i].ndim ||
           memcmp(t->shape,schema[i].shape,(size_t)schema[i].ndim*sizeof(uint64_t)))
            return fail(error,size,"unsupported image VAE tensor: %s",schema[i].name);
    }
    return 1;
}
static void hex(const uint8_t *bytes,char out[65]) {
    for(int i=0;i<32;i++) snprintf(out+2*i,3,"%02x",bytes[i]);
}
static int hash_range(FILE *f,uint64_t offset,uint64_t bytes,h3_sha256_ctx *hash,char *error,size_t size) {
    unsigned char *buffer=malloc(1024*1024);
    if(!buffer || fseeko(f,(off_t)offset,SEEK_SET)) {free(buffer);return fail(error,size,"cannot hash image VAE");}
    int ok=1;
    while(bytes && ok) {
        size_t n=bytes>1024*1024?1024*1024:(size_t)bytes;
        if(fread(buffer,1,n,f)!=n) {ok=fail(error,size,"truncated image VAE while hashing");break;}
        h3_sha256_update(hash,buffer,(h3_sha256_size)n);bytes-=n;
    }
    free(buffer);return ok;
}
int h3_image_vae_inspect(const char *path,h3_image_vae_info *info,char *error,size_t size) {
    h3_st_header h={0};
    if(!h3_st_read_header(path,&h,error,size))return 0;
    int ok=h3_image_vae_validate(&h,info,error,size);
    FILE *f=ok?fopen(path,"rb"):NULL;
    if(ok && !f)ok=fail(error,size,"cannot open image VAE");
    h3_sha256_ctx hash;uint8_t digest[32];
    if(ok) {
        h3_sha256_init(&hash);ok=hash_range(f,0,h.file_size,&hash,error,size);
        h3_sha256_final(digest,&hash);hex(digest,info->artifact_sha256);
    }
    if(ok) {
        h3_sha256_init(&hash);
        static const char recipe[]="h3-image-encoder-f16-normalized-v1";
        h3_sha256_update(&hash,recipe,sizeof(recipe)-1);
        /* Both supported platforms are little endian IEEE F32. */
        h3_sha256_update(&hash,info->mean,sizeof(info->mean));
        h3_sha256_update(&hash,info->deviation,sizeof(info->deviation));
        for(size_t i=0;ok && i<sizeof(schema)/sizeof(*schema);i++) {
            const char *n=schema[i].name;
            if(strncmp(n,"encoder.",8) && strncmp(n,"quant_conv.",11))continue;
            const h3_st_tensor *t=h3_st_find(&h,n);
            h3_sha256_update(&hash,n,(h3_sha256_size)strlen(n)+1);
            ok=hash_range(f,t->file_offset,t->data_end-t->data_begin,&hash,error,size);
        }
        h3_sha256_final(digest,&hash);hex(digest,info->compatibility_sha256);
        char identity[256];int n=snprintf(identity,sizeof(identity),"image:full_decoder_v1:f16-to-f32-v1:256/64:%d:%s:%s",info->output_slice,info->artifact_sha256,info->compatibility_sha256);
        h3_sha256_init(&hash);h3_sha256_update(&hash,identity,(h3_sha256_size)n);h3_sha256_final(digest,&hash);hex(digest,info->identity);
    }
    if (f)
        fclose(f);
    h3_st_free_header(&h);
    return ok;
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
    const char *compat=meta(&h,"h3_encoder_compatibility");
    int ok=h.tensor_count==1 && t && t->dtype==H3_DTYPE_F32 && t->ndim==5 &&
        t->shape[0]==1 && t->shape[1]==24 && t->shape[2]==1 && t->shape[3]<=INT32_MAX && t->shape[4]<=INT32_MAX &&
        eq(meta(&h,"h3_still_schema"),"1") && eq(meta(&h,"h3_latent_space"),"h3-normalized-v1") && digest_valid(compat);
    if(!ok)fail(error,size,"invalid still latent: expected schema 1, normalized F32 [1,24,1,h,w], encoder compatibility digest");
    if(ok)ok=h3_still_geometry((int)t->shape[3],(int)t->shape[4],error,size);
    size_t count=ok?(size_t)(24*t->shape[3]*t->shape[4]):0;
    if(ok)ok=h3_memory_check(count*4,"still latent load",error,size);
    if(ok) {out->values=malloc(count*4);if(!out->values)ok=fail(error,size,"cannot allocate still latent");}
    if(ok)ok=h3_st_read_data(&h,t,out->values,count*4,error,size);
    for(size_t i=0;ok&&i<count;i++)if(!isfinite(out->values[i]))ok=fail(error,size,"nonfinite still latent");
    if(ok) {out->height=(int)t->shape[3];out->width=(int)t->shape[4];memcpy(out->compatibility_sha256,compat,65);}
    h3_st_free_header(&h);if(!ok)h3_still_latent_free(out);return ok;
}
int h3_still_latent_save(const char *path,const h3_still_latent *z,char *error,size_t size) {
    if(!path || !*path || !z || !z->values || !digest_valid(z->compatibility_sha256))return fail(error,size,"invalid still latent save arguments");
    if(!h3_still_geometry(z->height,z->width,error,size))return 0;
    size_t count=(size_t)24*z->height*z->width;
    for(size_t i=0;i<count;i++)if(!isfinite(z->values[i]))return fail(error,size,"nonfinite still latent");
    char header[512];int n=snprintf(header,sizeof(header),"{\"__metadata__\":{\"h3_still_schema\":\"1\",\"h3_latent_space\":\"h3-normalized-v1\",\"h3_encoder_compatibility\":\"%s\",\"producer\":\"h3cli\"},\"latent\":{\"dtype\":\"F32\",\"shape\":[1,24,1,%d,%d],\"data_offsets\":[0,%zu]}}",z->compatibility_sha256,z->height,z->width,count*4);
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
    if(!p->still) {
        if(p->image_vae || p->save_still_latent)return fail(error,size,"image VAE and still latent output require --still");
        return 1;
    }
    if(p->still!=1 || p->frames!=1 || !p->image_vae || !*p->image_vae)
        return fail(error,size,"still generation requires --image-vae and exactly one frame");
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
        return fail(error,size,"initial still mode requires dense BF16, all 50 blocks, no reuse/quantization/acceleration");
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
        const char *other[]={p->image_vae,p->output_path};
        for(int i=0;i<2;i++)if(other[i]) {
            struct stat a,b;
            if(!strcmp(p->save_still_latent,other[i]) || (!stat(p->save_still_latent,&a) && !stat(other[i],&b) && a.st_dev==b.st_dev && a.st_ino==b.st_ino))
                return fail(error,size,"still latent output must not overwrite image VAE or PNG output");
        }
    }
    const char *blocks=getenv("H3_TEST_DIT_BLOCKS"),*reduction=getenv("H3_TOKEN_REDUCTION");
    if((blocks && strcmp(blocks,"50")) || (reduction && strcmp(reduction,"0")) || getenv("H3_REUSE_STEPS"))
        return fail(error,size,"still baseline requires all 50 blocks, no token reduction or custom reuse schedule");
    const char *tile=getenv("H3_VAE_TILE_PIXELS");
    if(tile && strcmp(tile,"256"))return fail(error,size,"still mode requires released 256/64 VAE tiles");
    return 1;
}
