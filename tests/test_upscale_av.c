/* AV transfer policy and presentation persistence; no models or GPU. */
#include "src/upscale/upscale.h"
#include "src/sampling/av_state.h"
#include "src/media/delivery.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>

static int checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %s:%d: %s (%s)\n",__FILE__,__LINE__,#x,error);exit(1);}} while(0)

static h3_presentation native(const h3_av_state_info *s) {
    return (h3_presentation){.version=H3_PRESENTATION_VERSION,.av_metadata_identity=1,
        .render_width=s->render_width,.render_height=s->render_height,.width=s->render_width,.height=s->render_height,
        .geometry_profile=s->geometry_profile,.codec_version=H3_NATIVE_CUDA?2:1,.fps=24,.sample_rate=32000};
}
static void policies(void) {
    char error[512]={0};h3_av_state source={0};h3_av_state_info target;
    CHECK(h3_av_state_shape_profile(256,256,56,0,&source.info));
    h3_presentation p=native(&source.info);
    for(int ref=0;ref<=1;ref++)for(int trim=0;trim<=39;trim+=39) {
        p.ref2va=ref;p.trim_frames=trim;p.trim_samples=trim*32000/24;
        CHECK(h3_upscale_av_plan(&source,&p,&target,error,sizeof(error)));
        CHECK(target.render_width==512&&target.render_height==512&&target.frames==56&&target.video_t==17&&target.audio_t==93);
    }
    h3_presentation base=p;
    p.trim_samples++;CHECK(!h3_upscale_av_plan(&source,&p,&target,error,sizeof(error)));p=base;
    p.trim_frames=40;CHECK(!h3_upscale_av_plan(&source,&p,&target,error,sizeof(error)));p=base;
    p.width=p.height=512;CHECK(!h3_upscale_av_plan(&source,&p,&target,error,sizeof(error)));p=base;
    p.codec_version=H3_NATIVE_CUDA?1:2;CHECK(!h3_upscale_av_plan(&source,&p,&target,error,sizeof(error)));p=base;
    memset(p.tiny_sha256,'a',64);CHECK(!h3_upscale_av_plan(&source,&p,&target,error,sizeof(error)));p=base;
    const int shapes[][3]={{960,544,22},{1344,768,39},{1920,1088,22},{256,256,362},{256,256,736}};
    for(size_t i=0;i<sizeof(shapes)/sizeof(*shapes);i++) {
        int w=shapes[i][0],h=shapes[i][1],f=shapes[i][2],profile=h3_canvas_valid(w,h,0)?0:2;
        CHECK(h3_av_state_shape_profile(w,h,f,profile,&source.info));p=native(&source.info);
        int allowed=H3_NATIVE_CUDA||(w<=960&&h<=960&&f<=362);
        CHECK(h3_upscale_av_plan(&source,&p,&target,error,sizeof(error))==allowed);
        if(allowed)CHECK(target.render_width==w*2&&target.render_height==h*2&&target.frames==f);
    }
}
static void persistence(void) {
    char error[512]={0},directory[]="/tmp/h3-upscale-av-XXXXXX",path[256],side[280];
    CHECK(mkdtemp(directory));snprintf(path,sizeof(path),"%s/state.h3av",directory);snprintf(side,sizeof(side),"%s.presentation",path);
    uint8_t signature[32]={1};h3_av_state *state=h3_av_state_new(512,512,56,42,signature);CHECK(state);
    h3_result result={.av_state=state,.presentation=native(&state->info)};
    result.presentation.upscale_recipe=3;result.presentation.trim_frames=39;result.presentation.trim_samples=52000;
    memset(result.presentation.upscale_parent_sha256,'a',64);memset(result.presentation.upscale_artifact_sha256,'b',64);
    CHECK(h3_result_save_av_state(&result,path,error,sizeof(error)));
    h3_presentation read={0};CHECK(h3_presentation_load(path,state,&read,error,sizeof(error))==1);
    CHECK(read.upscale_recipe==3&&read.trim_frames==39&&read.trim_samples==52000&&read.upscale_steps==0);
    h3_av_state_info target;CHECK(!h3_upscale_av_plan(state,&read,&target,error,sizeof(error)));
    result.presentation.upscale_recipe=1;CHECK(!h3_result_save_av_state(&result,path,error,sizeof(error)));
    result.presentation.upscale_recipe=2;CHECK(!h3_result_save_av_state(&result,path,error,sizeof(error)));
    result.presentation.upscale_recipe=3;result.presentation.upscale_steps=2;result.presentation.upscale_sigma=.25f;
    CHECK(!h3_result_save_av_state(&result,path,error,sizeof(error)));
    result.presentation.upscale_steps=0;result.presentation.upscale_sigma=0;result.presentation.trim_samples++;
    CHECK(!h3_result_save_av_state(&result,path,error,sizeof(error)));
    h3_av_state_free(state);CHECK(!unlink(side));CHECK(!unlink(path));CHECK(!rmdir(directory));
}
static void compare(const char *input,const char *output) {
    char error[512]={0};h3_av_state *source=h3_av_state_load(input,error,sizeof(error));CHECK(source);
    h3_av_state *target=h3_av_state_load(output,error,sizeof(error));CHECK(target);
    h3_presentation a,b;CHECK(h3_presentation_load(input,source,&a,error,sizeof(error))==1);
    CHECK(h3_presentation_load(output,target,&b,error,sizeof(error))==1);
    CHECK(target->info.render_width==2*source->info.render_width&&target->info.render_height==2*source->info.render_height);
    CHECK(target->info.frames==source->info.frames&&target->info.seed==source->info.seed);
    CHECK(target->info.audio_elements==source->info.audio_elements);
    CHECK(!memcmp(target->audio,source->audio,source->info.audio_elements*sizeof(float)));
    CHECK(!memcmp(target->info.compatibility,source->info.compatibility,32));
    CHECK(a.ref2va==b.ref2va&&a.trim_frames==b.trim_frames&&a.trim_samples==b.trim_samples&&b.upscale_recipe==3&&b.upscale_steps==0);
    uint8_t parent[32];char hex[65];h3_av_state_fingerprint(source,parent);
    for(int i=0;i<32;i++)snprintf(hex+2*i,3,"%02x",parent[i]);
    CHECK(!strcmp(b.upscale_parent_sha256,hex));CHECK(!strcmp(b.upscale_artifact_sha256,h3_upscale_artifact_sha256()));
    h3_av_state_free(source);h3_av_state_free(target);
}
int main(int argc,char **argv) {
    if(argc==4&&!strcmp(argv[1],"--compare"))compare(argv[2],argv[3]);
    else if(argc==1){policies();persistence();}else return 2;
    printf("ok: %d AV upscale checks\n",checks);return 0;
}
