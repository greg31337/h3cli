#include "src/upscale/upscale.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include "src/denoise/dit.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int fail(char *e,size_t n,const char *m) {if(e&&n)snprintf(e,n,"upscale plan: %s",m);return 0;}
const h3_upscale_plan_info *h3_upscale_plan_get_info(const h3_upscale_plan *p) {return p?&p->info:NULL;}
void h3_upscale_plan_free(h3_upscale_plan *p) {free(p);}
h3_upscale_plan *h3_upscale_plan_create(const h3_upscale_source *source,char *e,size_t n) {
    if(!source||!source->state||source->state->upscale.stage!=1||
       !h3_sampler_state_validate(source->state,e,n))return NULL;
    const h3_sampler_state *s=source->state;
    if(s->render_width>960||s->render_height>960) {fail(e,n,"2x target exceeds the 1920-axis profile");return NULL;}
    int width=s->render_width*2,height=s->render_height*2;
    int profile=(int64_t)width*height>H3_MAX_PIXELS;
    h3_av_state_info shape;
    if(!h3_av_state_shape_profile(width,height,s->aligned_frames,profile,&shape)) {
        fail(e,n,"2x target exceeds the 2,088,960-pixel profile");return NULL;
    }
    h3_upscale_plan *p=calloc(1,sizeof(*p));if(!p){fail(e,n,"out of memory");return NULL;}
    p->info=(h3_upscale_plan_info){.source_width=s->render_width,.source_height=s->render_height,
        .width=width,.height=height,.frames=s->aligned_frames,.video_t=s->latent_t,.audio_t=s->audio_t,
        .geometry_profile=profile,.video_elements=shape.video_elements,.audio_elements=shape.audio_elements,
        .network_reserve_bytes=(uint64_t)shape.video_t*shape.latent_h*shape.latent_w*512*2*4+(UINT64_C(2)<<30)};
    p->reference_count=s->reference_count;p->keyframe_count=s->upscale.keyframe_count;
    memcpy(p->source_identity,s->loaded_hash,32);
    memcpy(p->keyframes,s->upscale.keyframes,sizeof(p->keyframes));
    for(size_t i=0;i<p->reference_count;i++) {
        h3_layout_ref r=s->references[i];
        /* high/max images and videos keep their stored intrinsic canvases,
         * including stored max-size sources. No original image needs reopening. */
        if(r.kind==H3_LAYOUT_REF_IMAGE&&s->params.reference_image_size==H3_REFERENCE_IMAGE_MATCH) {
            int w,h;
            if(!h3_reference_image_canvas(s->upscale.original_width[i],s->upscale.original_height[i],width,height,0,&w,&h))goto invalid;
            if(w/16<r.latent_w||h/16<r.latent_h||
               ((w/16!=r.latent_w||h/16!=r.latent_h)&&(w/16>128||h/16>128)))goto invalid;
            r.latent_w=w/16;r.latent_h=h/16;
        }
        p->references[i]=r;
        if(r.kind!=H3_LAYOUT_REF_AUDIO)p->video_condition_elements+=(size_t)r.latent_t*r.latent_h*r.latent_w*24;
        p->audio_condition_elements+=(size_t)r.audio_t*64;
    }
    if(!s->ref2va)p->video_condition_elements=p->keyframe_count*(size_t)shape.latent_h*shape.latent_w*24;
    if(p->video_condition_elements>(UINT64_C(1)<<28)||p->audio_condition_elements!=(size_t)s->condition_audio_elements)goto invalid;
    h3_layout layout={0};h3_layout_spec spec={.text_len=(int)s->text.tokens,.latent_t=s->latent_t,
        .latent_h=shape.latent_h,.latent_w=shape.latent_w,.audio_t=s->audio_t,.frame_count=s->aligned_frames,
        .references=p->references,.reference_count=p->reference_count,.keyframes=p->keyframes,.keyframe_count=p->keyframe_count};
    if(!h3_layout_build(&spec,&layout,e,n)){free(p);return NULL;}
    p->info.sequence_rows=layout.seq_len;
    int valid=layout.seq_len<=UINT32_MAX/(5376u*3u)&&layout.img_cond_rows*96==p->video_condition_elements&&
        layout.audio_cond_rows*32==p->audio_condition_elements;
    h3_layout_free(&layout);if(!valid)goto invalid;
    return p;
invalid:fail(e,n,"unsupported reference retarget canvas, count or 32-bit DiT index range");free(p);return NULL;
}
float *h3_upscale_bilinear(const float *input,int t,int h,int w,int th,int tw,char *e,size_t n) {
    if(!input||t<1||t>107||h<1||w<1||th<h||tw<w||th>128||tw>128) {fail(e,n,"invalid bilinear comparison volume");return NULL;}
    size_t spatial=(size_t)t*h*w,target=(size_t)t*th*tw;float *out=malloc(target*24*4);
    if(!out){fail(e,n,"bilinear allocation failed");return NULL;}
    for(size_t i=0;i<spatial*24;i++)if(!isfinite(input[i])){free(out);fail(e,n,"nonfinite bilinear input");return NULL;}
    for(int c=0;c<24;c++)for(int z=0;z<t;z++)for(int y=0;y<th;y++)for(int x=0;x<tw;x++) {
        float xf=fmaxf(0,((float)x+.5f)*(float)w/(float)tw-.5f),yf=fmaxf(0,((float)y+.5f)*(float)h/(float)th-.5f);
        int x0=(int)xf,y0=(int)yf,x1=x0+1<w?x0+1:w-1,y1=y0+1<h?y0+1:h-1;
        float wx=xf-(float)x0,wy=yf-(float)y0;const float *v=input+(size_t)c*spatial+(size_t)z*h*w;
        out[(size_t)c*target+((size_t)z*th+y)*tw+x]=(1-wy)*((1-wx)*v[y0*w+x0]+wx*v[y0*w+x1])+wy*((1-wx)*v[y1*w+x0]+wx*v[y1*w+x1]);
    }
    return out;
}
void h3_upscale_conditions_free(h3_upscale_conditions *c) {
    if(c){free(c->video);free(c->audio);h3_layout_free(&c->layout);memset(c,0,sizeof(*c));}
}
static int retarget_span(const float *input,float *output,int t,int h,int w,int th,int tw,
    h3_upscale_model *model,h3_progress_callback progress,void *opaque,char *e,size_t n) {
    size_t count=(size_t)t*h*w*24,target=(size_t)t*th*tw*24;
    if(th==h&&tw==w){memcpy(output,input,count*4);return 1;}
    float *volume=malloc(count*4);if(!volume)return fail(e,n,"cannot unpack raw reference");
    int ok=h3_dit_unpatchify_video(input,24,t,h,w,volume,count);
    float *larger=ok?(model?h3_upscale_volume(model,volume,t,h,w,th,tw,progress,opaque,NULL,NULL,e,n):
        h3_upscale_bilinear(volume,t,h,w,th,tw,e,n)):NULL;
    free(volume);ok=larger&&h3_dit_patchify_video(larger,24,t,th,tw,output,target);free(larger);return ok;
}
int h3_upscale_retarget(const h3_upscale_source *source,const h3_upscale_plan *p,h3_upscale_model *model,
    h3_upscale_conditions *c,h3_progress_callback progress,void *opaque,char *e,size_t n) {
    if (!c)
        return fail(e, n, "missing destination");
    memset(c, 0, sizeof(*c));
    h3_upscale_plan *check=h3_upscale_plan_create(source,e,n);if(!check)return 0;
    int same=p!=NULL;
#define MATCH(field) do { if(p&&check->field!=p->field)same=0; } while(0)
    MATCH(info.source_width);MATCH(info.source_height);MATCH(info.width);MATCH(info.height);
    MATCH(info.frames);MATCH(info.video_t);MATCH(info.audio_t);MATCH(info.geometry_profile);
    MATCH(info.video_elements);MATCH(info.audio_elements);MATCH(info.sequence_rows);MATCH(info.network_reserve_bytes);
    MATCH(reference_count);MATCH(keyframe_count);MATCH(video_condition_elements);MATCH(audio_condition_elements);
#undef MATCH
    if(p&&(memcmp(check->references,p->references,sizeof(p->references))||memcmp(check->keyframes,p->keyframes,sizeof(p->keyframes))||
           memcmp(check->source_identity,p->source_identity,32)))same=0;
    free(check);
    if(!same)return fail(e,n,"stale or mismatched source plan");
    const h3_sampler_state *s=source->state;
    c->video_elements=p->video_condition_elements;c->audio_elements=p->audio_condition_elements;
    c->video=c->video_elements?malloc(c->video_elements*4):NULL;c->audio=c->audio_elements?malloc(c->audio_elements*4):NULL;
    if((c->video_elements&&!c->video)||(c->audio_elements&&!c->audio)){fail(e,n,"condition allocation failed");goto bad;}
    if(c->audio_elements)memcpy(c->audio,s->condition_audio,c->audio_elements*4);
    size_t from=0,to=0,count=s->ref2va?p->reference_count:p->keyframe_count;
    for(size_t i=0;i<count;i++) {
        if(progress&&progress("retarget conditions",(int)i,(int)count,opaque)){fail(e,n,"cancelled retarget");goto bad;}
        h3_layout_ref old=s->ref2va?s->references[i]:(h3_layout_ref){.kind=H3_LAYOUT_REF_IMAGE,.latent_t=1,.latent_h=s->latent_h,.latent_w=s->latent_w};
        h3_layout_ref next=s->ref2va?p->references[i]:(h3_layout_ref){.kind=H3_LAYOUT_REF_IMAGE,.latent_t=1,.latent_h=p->info.height/16,.latent_w=p->info.width/16};
        if(old.kind==H3_LAYOUT_REF_AUDIO)continue;
        size_t ni=(size_t)old.latent_t*old.latent_h*old.latent_w*24,no=(size_t)next.latent_t*next.latent_h*next.latent_w*24;
        if(ni>s->condition_video_elements-from||no>c->video_elements-to||
           !retarget_span(s->condition_video+from,c->video+to,old.latent_t,old.latent_h,old.latent_w,next.latent_h,next.latent_w,model,progress,opaque,e,n))goto bad;
        from+=ni;to+=no;
    }
    if(from!=s->condition_video_elements||to!=c->video_elements){fail(e,n,"retarget row count mismatch");goto bad;}
    h3_layout_spec spec={.text_len=(int)s->text.tokens,.latent_t=s->latent_t,.latent_h=p->info.height/16,.latent_w=p->info.width/16,
        .audio_t=s->audio_t,.frame_count=s->aligned_frames,.references=p->references,.reference_count=p->reference_count,
        .keyframes=p->keyframes,.keyframe_count=p->keyframe_count};
    if(!h3_layout_build(&spec,&c->layout,e,n))goto bad;
    /* Every target audio row is visible as clean conditioning. No video prefix. */
    c->layout.prefix=(h3_denoise_prefix){0,s->audio_t};c->layout.frozen_audio=1;
    return 1;
bad:h3_upscale_conditions_free(c);return 0;
}
