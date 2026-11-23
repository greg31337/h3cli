/* Export pre-augmentation learned retargeting for an independent Torch oracle. */
#include "src/upscale/upscale.h"
#include "src/upscale/upscale_network.h"
#include "src/sampling/sampler_state.h"
#include "src/denoise/dit.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static char error[1024];
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
static void volume(const char *dir,int index,const char *name,const float *patched,int t,int h,int w) {
    size_t n=(size_t)24*t*h*w;float *v=malloc(n*4);CHECK(v);
    CHECK(h3_dit_unpatchify_video(patched,24,t,h,w,v,n));
    char path[4096];snprintf(path,sizeof(path),"%s/%d-%s.f32",dir,index,name);
    FILE *f=fopen(path,"wbx");CHECK(f&&fwrite(v,4,n,f)==n&&!fclose(f));free(v);
}
int main(int argc,char **argv) {
    CHECK(argc==4);
    h3_upscale_source *source=h3_upscale_source_load(argv[2],error,sizeof(error));CHECK(source);
    const h3_sampler_state *s=source->state;CHECK(s->aligned_frames==22||s->aligned_frames==39);
    h3_upscale_plan *p=h3_upscale_plan_create(source,error,sizeof(error));CHECK(p);
    h3_upscale_model *m=h3_upscale_model_load(argv[1],error,sizeof(error));CHECK(m);
    h3_upscale_conditions c={0};CHECK(h3_upscale_retarget(source,p,m,&c,NULL,NULL,error,sizeof(error)));
    CHECK(c.audio_elements==s->condition_audio_elements);
    if(c.audio_elements)CHECK(!memcmp(c.audio,s->condition_audio,c.audio_elements*4));
    char path[4096];snprintf(path,sizeof(path),"%s/conditions.json",argv[3]);FILE *f=fopen(path,"wx");CHECK(f);
    fprintf(f,"{\"image_size\":%d,\"anchors\":%s,\"cases\":[",s->params.reference_image_size,s->ref2va?"false":"true");
    size_t from=0,to=0,count=s->ref2va?p->reference_count:p->keyframe_count;int written=0;
    for(size_t i=0;i<count;i++) {
        h3_layout_ref a=s->ref2va?s->references[i]:(h3_layout_ref){.kind=H3_LAYOUT_REF_IMAGE,.latent_t=1,.latent_h=s->latent_h,.latent_w=s->latent_w};
        h3_layout_ref b=s->ref2va?p->references[i]:(h3_layout_ref){.kind=H3_LAYOUT_REF_IMAGE,.latent_t=1,.latent_h=p->info.height/16,.latent_w=p->info.width/16};
        if(a.kind==H3_LAYOUT_REF_AUDIO)continue;
        CHECK(a.latent_t<=12&&a.latent_t==b.latent_t);
        volume(argv[3],(int)i,"input",s->condition_video+from,a.latent_t,a.latent_h,a.latent_w);
        volume(argv[3],(int)i,"target",c.video+to,b.latent_t,b.latent_h,b.latent_w);
        fprintf(f,"%s{\"id\":%zu,\"kind\":%d,\"input\":[1,24,%d,%d,%d],\"target\":[1,24,%d,%d,%d]}",
            written++?",":"",i,a.kind,a.latent_t,a.latent_h,a.latent_w,b.latent_t,b.latent_h,b.latent_w);
        from+=(size_t)24*a.latent_t*a.latent_h*a.latent_w;to+=(size_t)24*b.latent_t*b.latent_h*b.latent_w;
    }
    CHECK(from==s->condition_video_elements&&to==c.video_elements);CHECK(fprintf(f,"]}\n")>0&&!fclose(f));
    h3_upscale_conditions_free(&c);h3_upscale_model_free(m);h3_upscale_plan_free(p);h3_upscale_source_free(source);
    puts("PASS raw condition retarget export, order and audio identity");return 0;
}
