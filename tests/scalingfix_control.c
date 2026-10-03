/* Actual Qwen execution counter is injected in a test-only encoder copy. */
#include "src/memory.h"
#include "src/conditioning/text_encoder.h"
#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int entered,callbacks,memory;
static char error[512];
#define CHECK(x) do {if(!(x)){fprintf(stderr,"FAIL %d %s: %s\n",__LINE__,#x,error);exit(1);}}while(0)
void h3_scaling_enter(void){entered++;}
static int progress(int done,int total,void *opaque){(void)total;(void)opaque;callbacks++;if(!done)return 0;if(memory)setenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES","18446744073709551615",1);return !memory;}
int main(void){
    const char *modes[]={"legacy","scaled-q","reference"};
    for(int m=0;m<3;m++){
        setenv("H3_QWEN_GQA_SCALE_MODE",modes[m],1);
        for(memory=0;memory<2;memory++){
            entered=callbacks=0;uint32_t ids[]={151669,9906,151670};h3_text_embedding out;
            int ok=h3_text_encode_bf16("models/MiniMax-H3/FL2VA/text_encoder","src/metal/shaders.metal",ids,3,progress,NULL,&out,error,sizeof(error));
            CHECK(!ok && !out.values && entered==1 && callbacks==2);
            CHECK(strstr(error,memory?"reclaimable physical memory":"cancelled"));unsetenv("H3_TEST_MIN_AVAILABLE_MEMORY_BYTES");h3_text_embedding_free(&out);
            printf("PASS %s %s stopped after exactly one Qwen layer\n",modes[m],memory?"memory guard":"user cancellation");
        }
        h3_sampler_state *s=h3_sampler_state_load("outputs/refvideo-integration-validation/released-final.h3sample",error,sizeof(error));CHECK(s);CHECK(s->text.values && s->text.tokens);
        /* Already-computed conditioning is a stored BF16 payload, independent
         * of the current mode. Full resume still validates build/environment. */
        CHECK(h3_sampler_state_validate(s,error,sizeof(error)));h3_sampler_state_free(s);
        h3_av_state *av=h3_av_state_load("outputs/resume-validation/suite/t2va/oracle.h3av",error,sizeof(error));CHECK(av);h3_av_state_free(av);
        printf("PASS %s loads historical conditioning/checkpoint and AV state\n",modes[m]);
    }
    setenv("H3_QWEN_GQA_SCALE_MODE","bad",1);uint32_t id=9906;h3_text_embedding out;
    CHECK(!h3_text_encode_bf16("missing-weights","src/metal/shaders.metal",&id,1,NULL,NULL,&out,error,sizeof(error)));
    CHECK(strstr(error,"H3_QWEN_GQA_SCALE_MODE"));
    return 0;
}
