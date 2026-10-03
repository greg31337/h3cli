/* Geometry, validation and cache identity only; no numerical model oracle. */
#include "src/host.h"
#include "src/internal.h"
#include "src/conditioning/conditioning.h"
#include "src/conditioning/vision_encoder.h"
#include <fenv.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
static unsigned checks;
static char error[512];
#define CHECK(x) do { checks++; if(!(x)){fprintf(stderr,"reference image check %u line %d: %s (%s)\n",checks,__LINE__,#x,error);exit(1);} } while(0)
static void shape(int w,int h,int tw,int th,int mode,int expected_w,int expected_h) {
    h3_reference_image_shape s;
    CHECK(h3_reference_image_resolve(w,h,tw,th,mode,&s,error,sizeof(error)));
    CHECK(s.width==expected_w && s.height==expected_h);
    CHECK(s.patches==(size_t)(expected_w/16)*(size_t)(expected_h/16));
    CHECK(s.tokens*4==s.patches);
}
int main(void) {
    h3_vision_input inputs[128];h3_vision_plan plan;
    const int ns[]={1,2,3,5,9};
    for(size_t k=0;k<sizeof(ns)/sizeof(*ns);k++) {
        int count=ns[k];
        for(int i=0;i<count;i++)inputs[i]=(h3_vision_input){NULL,1,2048,8192};
        CHECK(h3_vision_plan_create(inputs,count,0,0,&plan,error,sizeof(error)));
        CHECK(plan.rows==(unsigned)count*65536);
        CHECK(!h3_vision_plan_create(inputs,count,plan.host_bytes-1,0,&plan,error,sizeof(error)));
        CHECK(strstr(error,"memory budget"));
        CHECK(h3_vision_plan_create(inputs,count,0,0,&plan,error,sizeof(error)));
        CHECK(!h3_vision_plan_create(inputs,count,0,plan.device_bytes-1,&plan,error,sizeof(error)));
    }
    inputs[0]=(h3_vision_input){NULL,1,2048,8224};
    CHECK(!h3_vision_plan_create(inputs,1,0,0,&plan,error,sizeof(error)));
    CHECK(strstr(error,"per-image"));
    inputs[0].frames=2;CHECK(h3_vision_plan_create(inputs,1,0,0,&plan,error,sizeof(error)));
    inputs[0]=(h3_vision_input){NULL,2,INT_MAX-31,INT_MAX-31};
    CHECK(!h3_vision_plan_create(inputs,1,0,0,&plan,error,sizeof(error)));
    CHECK(strstr(error,"index range"));
    CHECK(!h3_vision_plan_create(inputs,129,0,0,&plan,error,sizeof(error)));
    for(int i=0;i<3;i++)inputs[i]=(h3_vision_input){NULL,1,2048,2048};
    CHECK(h3_vision_plan_create(inputs,2,0,0,&plan,error,sizeof(error))&&plan.rows==32768);
    CHECK(h3_vision_plan_create(inputs,3,0,0,&plan,error,sizeof(error))&&plan.rows==49152);
    for(int i=0;i<2;i++)inputs[i].width=2720;
    CHECK(h3_vision_plan_create(inputs,2,0,0,&plan,error,sizeof(error))&&plan.rows==43520);
    const uint32_t rows[]={32764,32768,32772,43520,49152,65536,65540,589824};
    for(size_t i=0;i<sizeof(rows)/sizeof(*rows);i++) {
        uint32_t offset=0,chunks=0;
        while(offset<rows[i]) {
            uint32_t n=rows[i]-offset;if(n>H3_VISION_PATCH_CHUNK_ROWS)n=H3_VISION_PATCH_CHUNK_ROWS;
            CHECK((uint64_t)n*8192*2<=(UINT64_C(512)<<20));
            CHECK((uint64_t)offset*1152+(uint64_t)n*1152<=(uint64_t)rows[i]*1152);
            offset+=n;chunks++;
        }
        CHECK(offset==rows[i]&&chunks==(rows[i]+32767)/32768);
    }
    CHECK(H3_REFERENCE_IMAGE_MATCH==0&&H3_REFERENCE_IMAGE_MAX==1&&H3_REFERENCE_IMAGE_HIGH==2);
    const int cases[][8]={
        {640,480,640,480,2048,1536,2720,2048},
        {1365,1821,480,640,1536,2048,2048,2720},
        {32,32,32,32,2048,2048,2048,2048},
        {4000,1000,1120,288,2048,512,8192,2048}};
    for(size_t i=0;i<sizeof(cases)/sizeof(*cases);i++) {
        const int *c=cases[i];
        shape(c[0],c[1],640,480,H3_REFERENCE_IMAGE_MATCH,c[2],c[3]);
        shape(c[0],c[1],0,0,H3_REFERENCE_IMAGE_HIGH,c[4],c[5]);
        shape(c[0],c[1],0,0,H3_REFERENCE_IMAGE_MAX,c[6],c[7]);
        shape(c[1],c[0],1344,768,H3_REFERENCE_IMAGE_HIGH,c[5],c[4]);
        shape(c[1],c[0],1344,768,H3_REFERENCE_IMAGE_MAX,c[7],c[6]);
    }
    shape(1365,1821,1344,768,H3_REFERENCE_IMAGE_MATCH,864,1184);
    shape(640,480,1024,1024,H3_REFERENCE_IMAGE_MATCH,640,480);
    shape(1000,100,640,480,H3_REFERENCE_IMAGE_MATCH,992,96);
    int saved_round=fegetround();
    const int rounds[]={FE_TONEAREST,FE_DOWNWARD,FE_UPWARD,FE_TOWARDZERO};
    for(size_t i=0;i<sizeof(rounds)/sizeof(*rounds);i++) {
        CHECK(!fesetround(rounds[i]));
        shape(2064,2048,0,0,H3_REFERENCE_IMAGE_MAX,2048,2048);
        shape(2096,2048,0,0,H3_REFERENCE_IMAGE_MAX,2112,2048);
        shape(2048,528,0,0,H3_REFERENCE_IMAGE_HIGH,2048,512);
        shape(2048,560,0,0,H3_REFERENCE_IMAGE_HIGH,2048,576);
        shape(INT_MAX,INT_MAX,0,0,H3_REFERENCE_IMAGE_MAX,2048,2048);
    }
    CHECK(!fesetround(saved_round));
    h3_reference_image_shape s;
    for(int mode=1;mode<=2;mode++) {
        CHECK(!h3_reference_image_resolve(4001,1000,640,480,mode,&s,error,sizeof(error)));
        CHECK(strstr(error,"aspect ratio"));
        CHECK(!h3_reference_image_resolve(1000,4001,640,480,mode,&s,error,sizeof(error)));
        CHECK(!h3_reference_image_resolve(0,32,640,480,mode,&s,error,sizeof(error)));
        CHECK(!h3_reference_image_resolve(-1,32,640,480,mode,&s,error,sizeof(error)));
        CHECK(!h3_reference_image_resolve(INT_MAX,1,640,480,mode,&s,error,sizeof(error)));
    }
    CHECK(!h3_reference_image_resolve(64,64,0,0,0,&s,error,sizeof(error)));
    CHECK(!h3_reference_image_resolve(64,64,640,480,3,&s,error,sizeof(error)));
    CHECK(!h3_reference_image_resolve(64,64,640,480,-1,&s,error,sizeof(error)));
    CHECK(!h3_reference_image_resolve(64,64,640,480,1,NULL,error,sizeof(error)));
    CHECK(!h3_reference_image_resolve(INT_MAX,1,INT_MAX,INT_MAX,0,&s,error,sizeof(error)));
    char path[]="/tmp/h3-reference-image-XXXXXX";int fd=mkstemp(path);CHECK(fd>=0);close(fd);
    h3_reference refs[2]={{H3_REFERENCE_IMAGE,path,NULL,0},{H3_REFERENCE_IMAGE,path,NULL,0}};
    h3_params p=H3_PARAMS_DEFAULT;p.references=refs;p.reference_count=2;
    char *keys[3];
    for(int mode=0;mode<=2;mode++) {
        p.reference_image_size=mode;
        keys[mode]=h3_conditioning_key("test",&p,640,480,1);CHECK(keys[mode]);
        CHECK(h3_conditioning_image_geometry(&p)==(mode?2:0));
        CHECK((strstr(keys[mode],"reference-image-geometry=2")!=NULL)==(mode!=0));
    }
    CHECK(strcmp(keys[0],keys[1])&&strcmp(keys[1],keys[2])&&strcmp(keys[0],keys[2]));
    for(int mode=0;mode<=2;mode++)free(keys[mode]);
    /* The retained engine cache admits by this exact key, before loading its
     * tensors. Exercise alternating request identities on a retained context. */
    h3_ctx retained={0};h3_cache_set_enabled(&retained,1);
    const int modes[]={0,0,2,2,1,1,0};
    for(size_t i=0;i<sizeof(modes)/sizeof(*modes);i++) {
        p.reference_image_size=modes[i];
        char *key=h3_conditioning_key("test",&p,640,480,1);CHECK(key);
        int hit=retained.cache_enabled&&retained.conditioning_key&&!strcmp(retained.conditioning_key,key);
        CHECK(hit==(i>0&&modes[i]==modes[i-1]));
        h3_cache_clear(&retained);CHECK(!retained.conditioning_key);
        retained.conditioning_key=key;
    }
    h3_cache_set_enabled(&retained,0);CHECK(!retained.conditioning_key);
    p.reference_image_size=H3_REFERENCE_IMAGE_MAX;p._arithmetic_recipe=4;
    CHECK(!h3_conditioning_image_geometry(&p));
    p.reference_image_size=H3_REFERENCE_IMAGE_HIGH;CHECK(h3_conditioning_image_geometry(&p)==2);
    p.reference_count=0;CHECK(!h3_conditioning_image_geometry(&p));
    CHECK(!h3_conditioning_identity_matches("image-size=1\n", "image-size=1\nreference-image-geometry=2\n",error,sizeof(error)));
    CHECK(!unlink(path));printf("PASS: %u reference image geometry/cache checks\n",checks);return 0;
}
