#include "src/conditioning/conditioning.h"
#include "src/backend.h"
#include "src/internal.h"
#include "src/denoise/attention.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <unistd.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"conditioning check failed at %d: %s (%s)\n",__LINE__,#x,error);return 1;}checks++;}while(0)
int main(int argc,char **argv){
    char error[512]={0};int checks=0;
    if(argc==3&&!strcmp(argv[1],"load")){
        h3_conditioning *s=h3_conditioning_load(argv[2],error,sizeof(error));
        if(!s){fprintf(stderr,"%s\n",error);return 1;}h3_conditioning_free(s);return 0;
    }
    char path[]="/tmp/h3-conditioning-XXXXXX";int fd=mkstemp(path);CHECK(fd>=0);close(fd);
    h3_conditioning s={0};s.identity="format=1\nprompt-sha256=host-fixture\n";
    s.text.tokens=3;s.text.width=H3_TEXT_HIDDEN_SIZE;s.text.values=calloc(s.text.tokens*s.text.width,2);
    s.text.tags=calloc(s.text.tokens,1);CHECK(s.text.values&&s.text.tags);
    for(size_t i=0;i<s.text.tokens*s.text.width;i++)s.text.values[i]=(uint16_t)(0x3e80+i%31);
    s.text.tags[0]=1;s.text.tags[2]=1;
    float video[96]={0},audio[64]={0};s.video=video;s.audio=audio;s.video_elements=96;s.audio_elements=64;
    h3_layout_ref refs[2]={{H3_LAYOUT_REF_IMAGE,1,2,2,0},{H3_LAYOUT_REF_AUDIO,0,0,0,1}};
    s.references=refs;s.reference_count=2;s.conditioned=1;
    s.prepared.version=1;s.prepared.count=1;s.prepared.tensors[0]=(h3_prepared_tensor){1,3*5376,calloc(3*5376,2)};
    CHECK(s.prepared.tensors[0].values);
    s.has_schedule=1;CHECK(!h3_conditioning_save(&s,path,error,sizeof(error)));s.has_schedule=0;
    s.prepared.count=2;s.prepared.tensors[1]=s.prepared.tensors[0];
    CHECK(!h3_conditioning_save(&s,path,error,sizeof(error)));s.prepared.count=1;
    s.prepared.tensors[1]=(h3_prepared_tensor){0};
    s.prepared.tensors[0].id=999;CHECK(!h3_conditioning_save(&s,path,error,sizeof(error)));s.prepared.tensors[0].id=1;
    CHECK(h3_conditioning_save(&s,path,error,sizeof(error)));
    h3_conditioning *read=h3_conditioning_load(path,error,sizeof(error));CHECK(read);
    CHECK(!memcmp(s.text.values,read->text.values,s.text.tokens*s.text.width*2));
    CHECK(!memcmp(s.text.tags,read->text.tags,3));CHECK(read->reference_count==2);
    CHECK(!memcmp(s.references,read->references,2*sizeof(*refs)));CHECK(read->conditioned==1);
    CHECK(read->prepared.count==1&&read->prepared.tensors[0].elements==3*5376);
    h3_params request=H3_PARAMS_DEFAULT;request.width=64;request.height=32;
    h3_reference request_refs[2]={{H3_REFERENCE_IMAGE,"image",NULL,0},{H3_REFERENCE_AUDIO,"audio",NULL,0}};
    request.references=request_refs;request.reference_count=2;
    CHECK(h3_conditioning_request_matches(read,&request,error,sizeof(error)));
    read->reference_count=1;CHECK(!h3_conditioning_request_matches(read,&request,error,sizeof(error)));read->reference_count=2;
    read->references[0].latent_h=16384;CHECK(!h3_conditioning_request_matches(read,&request,error,sizeof(error)));read->references[0].latent_h=2;
    CHECK(h3_conditioning_identity_matches(s.identity,read->identity,error,sizeof(error)));
    CHECK(!h3_conditioning_identity_matches(s.identity,"format=1\nprompt-sha256=changed\n",error,sizeof(error)));
    CHECK(strstr(error,"prompt-sha256 mismatch"));h3_conditioning_free(read);
    h3_sigma_schedule sigmas;h3_layout layout={0};uint8_t a[32],b[32];
    CHECK(h3_serving_schedule_build(1,&sigmas));CHECK(sigmas.video[0]==1&&sigmas.video[1]==0);
    CHECK(h3_serving_schedule_build(6,&sigmas));h3_conditioning_schedule_key(&sigmas,&layout,0,a);
    sigmas.audio[3]+=0.001f;h3_conditioning_schedule_key(&sigmas,&layout,0,b);CHECK(memcmp(a,b,32));
    request.cuda_attention=H3_ATTENTION_SUBBLOCK;request_refs[0].path=path;
    request_refs[1]=(h3_reference){H3_REFERENCE_IMAGE,"inputs/1.jpg",NULL,0};
    request.reference_count=2;
    char *base=h3_conditioning_key("prompt",&request,640,480,1);CHECK(base);
    char *prepared=h3_prepared_key(base,&request,640,480);CHECK(prepared);
    for(int change=0;change<5;change++) {
        h3_params other=request;h3_reference swapped[2]={request_refs[1],request_refs[0]};
        if(change==0)other.reference_count=1;
        if(change==1)other.references=swapped;
        if(change==2)other.reference_image_size=H3_REFERENCE_IMAGE_MAX;
        if(change==3)other.reference_image_size=H3_REFERENCE_IMAGE_HIGH;
        char *key=h3_conditioning_key("prompt",&other,change==4?672:640,480,1);CHECK(key&&strcmp(base,key));
        char *dit_key=h3_prepared_key(key,&other,640,480);CHECK(dit_key&&strcmp(prepared,dit_key));
        free(key);free(dit_key);
    }
    request.adaptive_cache_max_bytes=UINT64_C(8192)*1048576;
    char *same=h3_conditioning_key("prompt",&request,640,480,1);CHECK(same&&!strcmp(base,same));
    char *same_dit=h3_prepared_key(same,&request,640,480);CHECK(same_dit&&!strcmp(prepared,same_dit));
    free(base);free(prepared);free(same);free(same_dit);
    request.adaptive_cache=1;request_refs[1]=(h3_reference){H3_REFERENCE_VIDEO,path,NULL,0};
    base=h3_conditioning_key("prompt",&request,640,480,1);CHECK(base);
    request_refs[1].include_embedded_audio=1;
    char *embedded=h3_conditioning_key("prompt",&request,640,480,1);CHECK(embedded&&strcmp(base,embedded));
    request_refs[1]=(h3_reference){H3_REFERENCE_VIDEO_AUDIO,path,"inputs/1.jpg",0};
    char *replacement=h3_conditioning_key("prompt",&request,640,480,1);CHECK(replacement&&strcmp(embedded,replacement));
    request_refs[1].audio_path="inputs/2.jpg";
    char *changed=h3_conditioning_key("prompt",&request,640,480,1);CHECK(changed&&strcmp(replacement,changed));
    free(base);free(embedded);free(replacement);free(changed);

    h3_backend backend;h3_attention_mode attention;
    CHECK(h3_backend_parse("metal",&backend)&&backend==H3_BACKEND_METAL);
    CHECK(h3_backend_parse("mpsgraph",&backend)&&backend==H3_BACKEND_MPSGRAPH_REFERENCE);
    CHECK(!h3_backend_parse("auto",&backend));CHECK(h3_attention_mode_parse("dense",&attention));
    CHECK(!h3_backend_preflight((h3_backend_scope){.backend=H3_BACKEND_METAL,.attention=H3_ATTN_SOL},1,error,sizeof(error)));
    h3_backend_scope old=h3_backend_exchange((h3_backend_scope){.backend=H3_BACKEND_METAL,.attention=H3_ATTN_DENSE});
    CHECK(h3_backend_current().backend==H3_BACKEND_METAL);h3_backend_exchange(old);
    setenv("H3_TEST_MAX_EVALUATIONS","6",1);CHECK(h3_test_evaluation_budget(6,error,sizeof(error)));
    CHECK(!h3_test_evaluation_budget(7,error,sizeof(error)));unsetenv("H3_TEST_MAX_EVALUATIONS");
    setenv("H3_TEST_MAX_EVALUATIONS","50",1);CHECK(h3_test_evaluation_budget(50,error,sizeof(error)));
    CHECK(!h3_test_evaluation_budget(51,error,sizeof(error)));unsetenv("H3_TEST_MAX_EVALUATIONS");
    if(argc==3&&!strcmp(argv[1],"fixture")){CHECK(h3_conditioning_save(&s,argv[2],error,sizeof(error)));}
    free(s.text.values);free(s.text.tags);h3_prepared_cache_free(&s.prepared);unlink(path);
    printf("PASS: %d conditioning/backend checks\n",checks);return 0;
}
