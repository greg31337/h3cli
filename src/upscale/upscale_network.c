/* Native LBH full-context graph. Weight/source notices: THIRD_PARTY_NOTICES.md. */
#include "src/upscale/upscale_network.h"
#include "src/upscale/upscale_gpu.h"
#include "src/weights/safetensors.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/memory.h"
#include "src/sglang/sglang.h"
#include <limits.h>
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>

#ifdef __clang__
#pragma STDC FP_CONTRACT OFF
#endif
#include "src/upscale/upscale_schema.inc"
#define UP_TENSORS (sizeof(up_schema)/sizeof(*up_schema))
#define UP_TILE 256u
struct h3_upscale_model {
    h3_gpu *gpu;
    h3_gpu_tensor *weights[UP_TENSORS];
    h3_upscale_stats stats;
};
static int fail(char *e,size_t n,const char *m) {if(e&&n)snprintf(e,n,"upscaler: %s",m);return 0;}
static uint16_t bf(float v) {uint32_t b;memcpy(&b,&v,4);return (uint16_t)((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t v) {uint32_t b=(uint32_t)v<<16;float f;memcpy(&f,&b,4);return f;}
static float round_bf(float v) {return fp(bf(v));}

int h3_upscale_gpu_counts(h3_upscale_gpu_op op,h3_upscale_gpu_shape p,size_t n[4]) {
    if(!n||!p.time||p.time>512||!p.height||p.height>256||!p.width||p.width>256||
       !p.channels||p.channels>512||op<H3_UP_NORM||op>H3_UP_BIAS)return 0;
    uint64_t count=(uint64_t)p.time*p.height*p.width*p.channels,out=count,a=0,b=0;
    switch(op) {
    case H3_UP_NORM:if(p.channels%32)return 0;a=b=p.channels;break;
    case H3_UP_COLUMNS:
        if(!p.rows||p.rows>1024||(p.kernel_time!=1&&p.kernel_time!=3)||
           (p.kernel_space!=1&&p.kernel_space!=3)||p.first>count/p.channels)return 0;
        out=(uint64_t)p.rows*p.channels*p.kernel_time*p.kernel_space*p.kernel_space;
        if (out > (32u << 20))
            return 0;
        break;
    case H3_UP_DEPTHWISE:if(p.kernel_time!=5)return 0;a=(uint64_t)p.channels*5;b=p.channels;break;
    case H3_UP_MODULATE:a=p.channels*2u;break;
    case H3_UP_RESIZE:
        if(!p.out_height||p.out_height>256||!p.out_width||p.out_width>256)return 0;
        out=(uint64_t)p.time*p.out_height*p.out_width*p.channels;break;
    case H3_UP_BIAS:a=p.channels;break;
    }
    if(count>UINT32_MAX||out>UINT32_MAX)return 0;
    n[0]=(size_t)out;n[1]=(size_t)count;n[2]=(size_t)a;n[3]=(size_t)b;return 1;
}
static h3_gpu_tensor *weight(h3_upscale_model *m,const char *name) {
    for(size_t i=0;i<UP_TENSORS;i++)if(!strcmp(name,up_schema[i].name))return m->weights[i];
    return NULL;
}
static h3_gpu_tensor *suffix(h3_upscale_model *m,const char *prefix,const char *tail) {
    char name[160];if(snprintf(name,sizeof(name),"%s.%s",prefix,tail)>=(int)sizeof(name))return NULL;
    return weight(m,name);
}
void h3_upscale_model_free(h3_upscale_model *m) {
    if(!m)return;
    h3_gpu_cancel(m->gpu);
    for(size_t i=0;i<UP_TENSORS;i++)h3_gpu_tensor_free(m->weights[i]);
    h3_gpu_free(m->gpu);free(m);
}
const char *h3_upscale_artifact_sha256(void) {return up_artifact_hash;}
h3_upscale_model *h3_upscale_model_load(const char *path,char *e,size_t n) {
    double begin=h3_av_now();h3_st_header header={0};h3_upscale_model *m=NULL;
    if(!path||!*path||!h3_st_read_header(path,&header,e,n))return NULL;
    if(header.tensor_count!=UP_TENSORS||header.file_size!=UINT64_C(690592992)) {
        fail(e,n,"expected the pinned 322-tensor BF16 artifact");goto done;
    }
    for(size_t i=0;i<UP_TENSORS;i++) {
        const h3_st_tensor *t=h3_st_find(&header,up_schema[i].name);
        if(!t||t->dtype!=H3_DTYPE_BF16||t->ndim!=up_schema[i].ndim||
           t->data_begin!=up_schema[i].begin||t->data_end!=up_schema[i].end) {
            fail(e,n,"tensor name/dtype/shape/offset differs from the pinned schema");goto done;
        }
        for(int j=0;j<t->ndim;j++)if(t->shape[j]!=up_schema[i].shape[j]) {
            fail(e,n,"tensor dimensions differ from pinned schema");goto done;
        }
    }
    /* Hash only this small independently acquired artifact. Never base H3 weights. */
    uint8_t hash[32];uint64_t bytes;char hex[65];
    if(!h3_sampler_source_fingerprint(path,hash,&bytes)){fail(e,n,"cannot verify upscaler artifact");goto done;}
    for(int i=0;i<32;i++)snprintf(hex+2*i,3,"%02x",hash[i]);
    if(bytes!=header.file_size||strcmp(hex,up_artifact_hash)){fail(e,n,"upscaler checksum mismatch");goto done;}
    if(!h3_memory_check(UINT64_C(2)<<30,"latent upscaler load",e,n))goto done;
    m=calloc(1,sizeof(*m));if(!m){fail(e,n,"out of memory");goto done;}
    /* The graph's recipe is independent of the request's DiT math policy. */
    int previous=h3_sglang_exchange(0);
    m->gpu=h3_gpu_create("src/metal/shaders.metal",e,n);h3_sglang_exchange(previous);
    if(!m->gpu)goto bad;
    h3_gpu_profile_set_label(m->gpu,"latent upscaler BF16 recipe 1");
    for(size_t i=0;i<UP_TENSORS;i++) {
        const h3_st_tensor *t=h3_st_find(&header,up_schema[i].name);
        m->weights[i]=h3_gpu_tensor_load_bf16(m->gpu,path,t->file_offset,(size_t)h3_st_tensor_elements(t));
        if(!m->weights[i]){fail(e,n,h3_gpu_error(m->gpu));goto bad;}
        m->stats.weights_bytes+=t->data_end-t->data_begin;
    }
    m->stats.load_seconds=h3_av_now()-begin;
    fprintf(stderr,"h3cli: latent upscaler: recipe=1 full-context BF16 weights=%llu load=%.6fs sha256=%s\n",
        (unsigned long long)m->stats.weights_bytes,m->stats.load_seconds,up_artifact_hash);
    goto done;
bad:h3_upscale_model_free(m);m=NULL;
done:h3_st_free_header(&header);return m;
}
int h3_upscale_model_stats(const h3_upscale_model *m,h3_upscale_stats *s) {
    if (!m || !s)
        return 0;
    *s = m->stats;
    h3_gpu_stats gpu;
    if (h3_gpu_get_stats(m->gpu, &gpu))
        s->peak_gpu_bytes = gpu.peak_live_bytes;
    return 1;
}
static int emit(h3_upscale_model *m,h3_gpu_tensor *x,const char *name,int c,int t,int h,int w,
    h3_upscale_trace trace,void *opaque) {
    if(!trace)return 1;
    size_t spatial=(size_t)t*h*w,count=spatial*(size_t)c;
    uint16_t *raw=malloc(count*2);float *host=malloc(count*4);
    int ok=raw&&host&&h3_gpu_tensor_read_bf16(x,raw,count);
    if(ok) {
        for(size_t j=0;j<spatial;j++)for(int k=0;k<c;k++)host[(size_t)k*spatial+j]=fp(raw[j*(size_t)c+k]);
        ok=!trace(name,host,c,t,h,w,opaque);
    }
    (void)m;free(raw);free(host);return ok;
}
static int conv(h3_upscale_model *m,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const char *prefix,h3_upscale_gpu_shape p,unsigned co,unsigned kernel,
    h3_gpu_tensor *columns,h3_gpu_tensor *tile) {
    h3_gpu_tensor *w=suffix(m,prefix,"weight"),*b=suffix(m,prefix,"bias");
    if(!w||!b)return 0;
    unsigned spatial=p.time*p.height*p.width,filter=p.channels*kernel*kernel*kernel;
    p.kernel_time=p.kernel_space=kernel;p.rows=UP_TILE;
    for(unsigned first=0;first<spatial;first+=UP_TILE) {
        unsigned count=spatial-first<UP_TILE?spatial-first:UP_TILE;p.first=first;
        h3_upscale_gpu_shape bias={.time=1,.height=1,.width=UP_TILE,.channels=co};
        if(!h3_gpu_upscale(m->gpu,columns,in,NULL,NULL,H3_UP_COLUMNS,p)||
           !h3_gpu_upscale_linear(m->gpu,tile,columns,w,UP_TILE,filter,co)||
           !h3_gpu_upscale(m->gpu,tile,tile,b,NULL,H3_UP_BIAS,bias)||
           !h3_gpu_copy_bf16(m->gpu,out,(size_t)first*co,tile,0,(size_t)count*co))return 0;
        m->stats.convolution_tiles++;
        /* Bound MPSGraph command temporaries as well as explicit packing. */
        if((first/UP_TILE)%4==3&&!h3_gpu_continue(m->gpu))return 0;
    }
    m->stats.convolutions++;return 1;
}
static int norm(h3_upscale_model *m,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const char *prefix,h3_upscale_gpu_shape p) {
    return h3_gpu_upscale(m->gpu,out,in,suffix(m,prefix,"weight"),suffix(m,prefix,"bias"),H3_UP_NORM,p);
}
static int run_block(h3_upscale_model *m,h3_gpu_tensor *x,h3_gpu_tensor *a,h3_gpu_tensor *b,
    h3_gpu_tensor *embedding,h3_gpu_tensor *temp,h3_gpu_tensor *mod,h3_gpu_tensor *columns,h3_gpu_tensor *tile,
    h3_upscale_gpu_shape p,int half,int block) {
    h3_gpu *g=m->gpu;size_t count=(size_t)p.time*p.height*p.width*512;
    char prefix[64],name[96];snprintf(prefix,sizeof(prefix),"%s_blocks.%d",half?"out":"in",block);
            if(block%3==1) {
                snprintf(name,sizeof(name),"%s.norm",prefix);
                if(!norm(m,a,x,name,p)||!h3_gpu_silu_bf16(g,a,a,(uint32_t)count))return 0;
                snprintf(name,sizeof(name),"%s.dwconv",prefix);p.kernel_time=5;
                if(!h3_gpu_upscale(g,b,a,suffix(m,name,"weight"),suffix(m,name,"bias"),H3_UP_DEPTHWISE,p))return 0;
                snprintf(name,sizeof(name),"%s.pwconv",prefix);
                if(!conv(m,a,b,name,p,512,1,columns,tile)||!h3_gpu_add_bf16(g,x,x,a,(uint32_t)count))return 0;
            } else {
                snprintf(name,sizeof(name),"%s.in_layers.0",prefix);
                if(!norm(m,a,x,name,p)||!h3_gpu_silu_bf16(g,a,a,(uint32_t)count))return 0;
                snprintf(name,sizeof(name),"%s.in_layers.2",prefix);
                if(!conv(m,b,a,name,p,512,3,columns,tile))return 0;
                if(!h3_gpu_silu_bf16(g,temp,embedding,64))return 0;
                snprintf(name,sizeof(name),"%s.emb_layers.1",prefix);
                if(!h3_gpu_linear_bf16(g,mod,temp,suffix(m,name,"weight"),suffix(m,name,"bias"),1,64,1024))return 0;
                snprintf(name,sizeof(name),"%s.out_norm",prefix);
                if(!norm(m,a,b,name,p)||!h3_gpu_upscale(g,a,a,mod,NULL,H3_UP_MODULATE,p)||!h3_gpu_silu_bf16(g,a,a,(uint32_t)count))return 0;
                snprintf(name,sizeof(name),"%s.out_layers.2",prefix);
                if(!conv(m,b,a,name,p,512,3,columns,tile)||!h3_gpu_add_bf16(g,x,x,b,(uint32_t)count))return 0;
            }
    return 1;
}
float *h3_upscale_volume(h3_upscale_model *m,const float *input,int time,int height,int width,
    int th,int tw,h3_progress_callback progress,void *opaque,h3_upscale_trace trace,
    void *trace_opaque,char *e,size_t n) {
    if(!m||!input||time<1||time>107||height<1||width<1||th<height||tw<width||
       th>128||tw>128||(th==height&&tw==width)) {fail(e,n,"invalid full-context enlargement (T<=107, target <=128x128 latent)");return NULL;}
    size_t source_spatial=(size_t)time*height*width,target_spatial=(size_t)time*th*tw;
    uint64_t reserve=(uint64_t)target_spatial*512*2*4+(UINT64_C(1)<<30);
    if(!h3_memory_check(reserve,"latent upscaler forward",e,n))return NULL;
    double begin=h3_av_now();h3_gpu *g=m->gpu;float *result=NULL;
    h3_gpu_tensor *input_gpu=NULL,*x=NULL,*a=NULL,*b=NULL,*columns=NULL,*tile=NULL,*embedding=NULL,*temp=NULL,*mod=NULL,*out=NULL;
    uint16_t *host=malloc(source_spatial*24*2);
    if(!host){fail(e,n,"cannot allocate normalized input");return NULL;}
    for(size_t j=0;j<source_spatial;j++)for(size_t c=0;c<24;c++) {
        float value=input[c*source_spatial+j];
        if(!isfinite(value)){fail(e,n,"nonfinite source video");goto done;}
        value=round_bf(round_bf(round_bf(value)-round_bf(up_mean[c]))/round_bf(up_std[c]));
        if(!isfinite(value)){fail(e,n,"source video overflows BF16");goto done;}
        host[j*24+c]=bf(value);
    }
    if(progress&&progress("latent upscale",0,40,opaque)){fail(e,n,"cancelled before transfer");goto done;}
    input_gpu=h3_gpu_tensor_from_bf16(g,host,source_spatial*24);free(host);host=NULL;
    x=h3_gpu_tensor_new_bf16(g,source_spatial*512);
    a=h3_gpu_tensor_new_bf16(g,source_spatial*512);b=h3_gpu_tensor_new_bf16(g,source_spatial*512);
    columns=h3_gpu_tensor_new_bf16(g,(size_t)UP_TILE*512*27);tile=h3_gpu_tensor_new_bf16(g,UP_TILE*512);
    embedding=h3_gpu_tensor_new_bf16(g,64);temp=h3_gpu_tensor_new_bf16(g,64);mod=h3_gpu_tensor_new_bf16(g,1024);
    uint16_t scale=bf(((float)th/(float)height+(float)tw/(float)width)*.5f-1.f);
    h3_gpu_tensor *scale_gpu=h3_gpu_tensor_from_bf16(g,&scale,1);
    if(!input_gpu||!x||!a||!b||!columns||!tile||!embedding||!temp||!mod||!scale_gpu) {
        h3_gpu_tensor_free(scale_gpu);goto gpu_error;
    }
    m->stats.workspace_bytes=(uint64_t)UP_TILE*(512*27+512)*2;
    int ok=h3_gpu_begin(g)&&
        h3_gpu_linear_bf16(g,temp,scale_gpu,weight(m,"embed.0.weight"),weight(m,"embed.0.bias"),1,1,64)&&
        h3_gpu_silu_bf16(g,temp,temp,64)&&
        h3_gpu_linear_bf16(g,embedding,temp,weight(m,"embed.2.weight"),weight(m,"embed.2.bias"),1,64,64)&&h3_gpu_submit(g);
    h3_gpu_tensor_free(scale_gpu);if(!ok)goto gpu_error;
    if(!emit(m,embedding,"embed",64,1,1,1,trace,trace_opaque)||
       !emit(m,input_gpu,"normalized",24,time,height,width,trace,trace_opaque))goto gpu_error;
    h3_upscale_gpu_shape p={.time=(unsigned)time,.height=(unsigned)height,.width=(unsigned)width,.channels=24};
    if(!h3_gpu_begin(g)||!conv(m,x,input_gpu,"conv_in",p,512,3,columns,tile)||!h3_gpu_submit(g))goto gpu_error;
    h3_gpu_tensor_free(input_gpu);input_gpu=NULL;p.channels=512;
    if(!emit(m,x,"conv_in",512,time,height,width,trace,trace_opaque))goto gpu_error;
    for(int half=0;half<2;half++) {
        if(half) {
            out=h3_gpu_tensor_new_bf16(g,target_spatial*512);p.out_height=(unsigned)th;p.out_width=(unsigned)tw;
            if(!out||!h3_gpu_begin(g)||!h3_gpu_upscale(g,out,x,NULL,NULL,H3_UP_RESIZE,p)||!h3_gpu_submit(g))goto gpu_error;
            h3_gpu_tensor_free(x);h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);
            x=out;out=NULL;a=h3_gpu_tensor_new_bf16(g,target_spatial*512);b=h3_gpu_tensor_new_bf16(g,target_spatial*512);
            if (!a || !b)
                goto gpu_error;
            p.height = (unsigned)th;
            p.width = (unsigned)tw;
        }
        for(int block=0;block<18;block++) {
            char prefix[64];snprintf(prefix,sizeof(prefix),"%s_blocks.%d",half?"out":"in",block);
            if(progress&&progress("latent upscale",2+half*18+block,40,opaque)){fail(e,n,"cancelled during transfer");goto done;}
            if(!h3_gpu_begin(g))goto gpu_error;
            if(!run_block(m,x,a,b,embedding,temp,mod,columns,tile,p,half,block))goto gpu_error;
            if(!h3_gpu_submit(g)||!emit(m,x,prefix,512,time,(int)p.height,(int)p.width,trace,trace_opaque))goto gpu_error;
        }
    }
    if(!h3_gpu_begin(g)||!norm(m,a,x,"norm_out",p)||!h3_gpu_submit(g)||
       !emit(m,a,"norm_out",512,time,th,tw,trace,trace_opaque))goto gpu_error;
    out=h3_gpu_tensor_new_bf16(g,target_spatial*24);
    if(!out||!h3_gpu_begin(g)||!h3_gpu_silu_bf16(g,a,a,(uint32_t)(target_spatial*512))||
       !conv(m,out,a,"conv_out",p,24,3,columns,tile)||!h3_gpu_submit(g)||
       !emit(m,out,"conv_out",24,time,th,tw,trace,trace_opaque))goto gpu_error;
    host=malloc(target_spatial*24*2);result=malloc(target_spatial*24*4);
    if(!host||!result||!h3_gpu_tensor_read_bf16(out,host,target_spatial*24))goto gpu_error;
    for(size_t j=0;j<target_spatial;j++)for(size_t c=0;c<24;c++) {
        float v=round_bf(round_bf(fp(host[j*24+c])*round_bf(up_std[c]))+round_bf(up_mean[c]));
        if(!isfinite(v)){fail(e,n,"nonfinite learned output");goto failed;}
        result[c*target_spatial+j]=v;
    }
    if(trace&&trace("output",result,24,time,th,tw,trace_opaque)){fail(e,n,"trace cancelled");goto failed;}
    if(progress&&progress("latent upscale",40,40,opaque)){fail(e,n,"cancelled after transfer");goto failed;}
    m->stats.forward_seconds+=h3_av_now()-begin;
    h3_gpu_stats stats;h3_gpu_get_stats(g,&stats);m->stats.peak_gpu_bytes=stats.peak_live_bytes;
    fprintf(stderr,"h3cli: latent upscale %dx%dx%d -> %dx%dx%d: %.6fs, packing=%llu peak_gpu=%llu\n",
        time,height,width,time,th,tw,h3_av_now()-begin,(unsigned long long)m->stats.workspace_bytes,
        (unsigned long long)m->stats.peak_gpu_bytes);
    goto done;
gpu_error:fail(e,n,h3_gpu_error(g));
failed:free(result);result=NULL;
done:
    h3_gpu_cancel(g);free(host);
    h3_gpu_tensor_free(input_gpu);h3_gpu_tensor_free(x);h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);
    h3_gpu_tensor_free(columns);h3_gpu_tensor_free(tile);h3_gpu_tensor_free(embedding);h3_gpu_tensor_free(temp);
    h3_gpu_tensor_free(mod);h3_gpu_tensor_free(out);return result;
}

float *h3_upscale_probe_layer(h3_upscale_model *m,const char *name,const float *input,
    const float emb[64],int time,int height,int width,char *e,size_t n) {
    if(!m||!name||!input||!emb||time<1||time>27||height<1||height>12||width<1||width>16) {
        fail(e,n,"invalid bounded layer probe");return NULL;
    }
    int half=-1,block=-1,used=0,head=!strcmp(name,"conv_out"),norm_only=!strcmp(name,"norm_out");
    if(sscanf(name,"in_blocks.%d%n",&block,&used)==1&&used==(int)strlen(name))half=0;
    else if(sscanf(name,"out_blocks.%d%n",&block,&used)==1&&used==(int)strlen(name))half=1;
    if(!head&&!norm_only&&(half<0||block<0||block>=18)){fail(e,n,"unknown layer probe");return NULL;}
    size_t spatial=(size_t)time*height*width,target=half==1&&block==0?spatial*4:spatial;
    h3_gpu *g=m->gpu;float *result=NULL;uint16_t *raw=malloc(target*512*2),be[64];
    h3_gpu_tensor *x=NULL,*a=NULL,*b=NULL,*embedding=NULL,*temp=NULL,*mod=NULL,*columns=NULL,*tile=NULL,*resized=NULL;
    if(!raw)goto done;
    for(size_t i=0;i<spatial;i++)for(size_t c=0;c<512;c++)raw[i*512+c]=bf(input[c*spatial+i]);
    for(int i=0;i<64;i++)be[i]=bf(emb[i]);
    x=h3_gpu_tensor_from_bf16(g,raw,spatial*512);a=h3_gpu_tensor_new_bf16(g,target*512);b=h3_gpu_tensor_new_bf16(g,target*512);
    embedding=h3_gpu_tensor_from_bf16(g,be,64);temp=h3_gpu_tensor_new_bf16(g,64);mod=h3_gpu_tensor_new_bf16(g,1024);
    columns=h3_gpu_tensor_new_bf16(g,(size_t)UP_TILE*512*27);tile=h3_gpu_tensor_new_bf16(g,UP_TILE*512);
    if(!x||!a||!b||!embedding||!temp||!mod||!columns||!tile||!h3_gpu_begin(g))goto done;
    h3_upscale_gpu_shape p={.time=(unsigned)time,.height=(unsigned)height,.width=(unsigned)width,.channels=512};
    if(target!=spatial) {
        resized=h3_gpu_tensor_new_bf16(g,target*512);p.out_height=p.height*2;p.out_width=p.width*2;
        if(!resized||!h3_gpu_upscale(g,resized,x,NULL,NULL,H3_UP_RESIZE,p)||!h3_gpu_submit(g))goto done;
        h3_gpu_tensor_free(x);x=resized;resized=NULL;p.height*=2;p.width*=2;
        if(!h3_gpu_begin(g))goto done;
    }
    unsigned channels=head?24:512;h3_gpu_tensor *read=x;
    if(head) {
        if(!h3_gpu_silu_bf16(g,a,x,(uint32_t)(target*512))||!conv(m,b,a,"conv_out",p,24,3,columns,tile))goto done;
        read=b;
    } else if(norm_only) {
        if (!norm(m, a, x, "norm_out", p))
            goto done;
        read = a;
    } else if(!run_block(m,x,a,b,embedding,temp,mod,columns,tile,p,half,block))goto done;
    if(!h3_gpu_submit(g)||!h3_gpu_tensor_read_bf16(read,raw,target*channels))goto done;
    result=malloc(target*channels*4);if(!result)goto done;
    for(size_t i=0;i<target;i++)for(unsigned c=0;c<channels;c++)result[(size_t)c*target+i]=fp(raw[i*channels+c]);
done:
    if (!result)
        fail(e, n, h3_gpu_error(g));
    h3_gpu_cancel(g);
    free(raw);
    h3_gpu_tensor_free(x);h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);h3_gpu_tensor_free(embedding);
    h3_gpu_tensor_free(temp);h3_gpu_tensor_free(mod);h3_gpu_tensor_free(columns);h3_gpu_tensor_free(tile);
    h3_gpu_tensor_free(resized);return result;
}
