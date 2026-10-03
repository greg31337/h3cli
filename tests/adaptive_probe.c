#include "src/gpu.h"
#include "src/denoise/adaptive_cache.h"
#include "src/sampling/bridge.h"
#include "src/sampling/av_state.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s: %s\n",__LINE__,#x,g?h3_gpu_error(g):error);return 1;}}while(0)
static uint16_t bf(float f){uint32_t b;memcpy(&b,&f,4);return (uint16_t)((b+0x7fff+((b>>16)&1))>>16);}
static float fp(uint16_t b){uint32_t u=(uint32_t)b<<16;float f;memcpy(&f,&u,4);return f;}
static int continuation_probe(h3_gpu *g) {
    char error[512];
    for(int context=39;context<=90;context+=51)for(int mode=0;mode<=3;mode++)for(int strength=0;strength<=2;strength++) {
        h3_av_state_info shape;CHECK(h3_av_state_shape(256,256,124,&shape));
        h3_layout_ref refs[]={{H3_LAYOUT_REF_IMAGE,1,128,128,0},{H3_LAYOUT_REF_AUDIO,0,0,0,80}};
        h3_layout_spec spec={65,shape.video_t,shape.latent_h,shape.latent_w,shape.audio_t,124,NULL,0,refs,2};
        h3_layout l={0};CHECK(h3_layout_build(&spec,&l,error,sizeof(error)));CHECK(h3_continuation_context(context,&l.prefix));
        h3_bridge_profile bridge;
        if(mode){CHECK(h3_bridge_profile_build(context,8,(float)strength*.5f,(h3_bridge_profile_type)(mode-1),&bridge,NULL,0,error,sizeof(error)));l.bridge=&bridge;}
        h3_adaptive_regions plan;CHECK(h3_adaptive_regions_build(&l,&plan,error,sizeof(error)));
        unsigned cols=mode==3&&strength==2?137:17;size_t n=l.seq_len*cols;
        uint16_t *zero=calloc(n,2),*anchor=malloc(n*2),*out=malloc(n*2),*got=malloc(n*2);CHECK(zero&&anchor&&out&&got);
        double numerator[23]={0},denominator[23]={0};size_t counts[23]={0};
        for(size_t row=0;row<l.seq_len;row++) {
            unsigned slot=0;
            if(row>=plan.video_start&&row-plan.video_start<plan.video_rows) {
                size_t t=(row-plan.video_start)/plan.spatial;
                if(t>=(size_t)l.prefix.video_prefix_t)slot=1;
                else if(mode&&bridge.class_mask[bridge.video_classes[t]]>0)slot=bridge.video_classes[t]==H3_ROW_GENERATED_VIDEO?12:bridge.video_classes[t]-1;
            } else if(row>=plan.audio_start&&row-plan.audio_start<plan.audio_rows) {
                size_t t=(row-plan.audio_start)%(unsigned)shape.audio_t;
                if(t>=(size_t)l.prefix.audio_prefix_t)slot=2;
                else if(mode&&bridge.class_mask[bridge.audio_classes[t]]>0)slot=bridge.audio_classes[t]==H3_ROW_GENERATED_AUDIO?22:bridge.audio_classes[t]-1;
            }
            for(unsigned col=0;col<cols;col++) {
                size_t i=row*cols+col;anchor[i]=bf(1);
                out[i]=bf(1+(float)slot/32);double diff=fabs((double)fp(out[i])-1);
                numerator[0]+=diff;denominator[0]+=1;counts[0]++;
                if(slot){numerator[slot]+=diff;denominator[slot]+=1;counts[slot]++;}
            }
        }
        h3_gpu_tensor *input=h3_gpu_tensor_from_bf16(g,zero,n),*output=h3_gpu_tensor_from_bf16(g,out,n),
            *old=h3_gpu_tensor_from_bf16(g,anchor,n),*scratch=h3_gpu_tensor_new_f32(g,257*46);CHECK(input&&output&&old&&scratch);
        float scores[23],previous[23],selected;
        for(int repeat=0;repeat<2;repeat++) {
            CHECK(h3_gpu_tensor_write_bf16(input,zero,n));CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_adaptive_continuation_probe(g,input,output,old,scratch,cols,&plan,1,scores));CHECK(h3_gpu_submit(g));
            float maximum=0;
            for(unsigned c=0;c<23;c++) {
                float expected=counts[c]?(float)((numerator[c]/(double)counts[c])/fmax(denominator[c]/(double)counts[c],1e-6)):0;
                CHECK(fabsf(scores[c]-expected)<1e-6f);if(expected>maximum)maximum=expected;
            }
            CHECK(h3_adaptive_regions_score(&plan,scores,&selected)&&fabsf(selected-maximum)<1e-6f);
            if(repeat)CHECK(!memcmp(previous,scores,sizeof(scores)));
            memcpy(previous,scores,sizeof(scores));
        }
        CHECK(h3_gpu_tensor_read_bf16(input,got,n)&&!memcmp(got,out,n*2));
        if(context==90&&mode==1&&strength==1)for(unsigned isolated=1;isolated<23;isolated++)if(plan.counts[isolated]) {
            for(int tiny=0;tiny<2;tiny++) {
                float expected=tiny?fp(bf(2e-7f))/1e-6f:.25f;
                for(size_t row=0;row<l.seq_len;row++) {
                    unsigned slot=0;
                    if(row>=plan.video_start&&row-plan.video_start<plan.video_rows) {
                        size_t t=(row-plan.video_start)/plan.spatial;
                        slot=t>=(size_t)l.prefix.video_prefix_t?1:
                            bridge.class_mask[bridge.video_classes[t]]>0?bridge.video_classes[t]-1:0;
                    } else if(row>=plan.audio_start&&row-plan.audio_start<plan.audio_rows) {
                        size_t t=(row-plan.audio_start)%(unsigned)shape.audio_t;
                        slot=t>=(size_t)l.prefix.audio_prefix_t?2:
                            bridge.class_mask[bridge.audio_classes[t]]>0?bridge.audio_classes[t]-1:0;
                    }
                    for(unsigned col=0;col<cols;col++) {
                        size_t i=row*cols+col;anchor[i]=bf(tiny?0:1);
                        out[i]=bf(tiny?(slot==isolated?2e-7f:0):(slot==isolated?1.25f:1));
                    }
                }
                CHECK(h3_gpu_tensor_write_bf16(old,anchor,n)&&h3_gpu_tensor_write_bf16(output,out,n)&&h3_gpu_tensor_write_bf16(input,zero,n));
                CHECK(h3_gpu_begin(g));CHECK(h3_gpu_adaptive_continuation_probe(g,input,output,old,scratch,cols,&plan,1,scores));CHECK(h3_gpu_submit(g));
                CHECK(fabsf(scores[isolated]-expected)<1e-6f);
                CHECK(scores[0]<scores[isolated]); /* frozen history cannot dilute the decision */
                for(unsigned c=1;c<23;c++)if(c!=isolated)CHECK(scores[c]==0);
                CHECK(h3_adaptive_regions_score(&plan,scores,&selected)&&selected==scores[isolated]);
            }
        }
        plan.audio[0]=24;CHECK(h3_gpu_begin(g));CHECK(!h3_gpu_adaptive_continuation_probe(g,input,output,old,scratch,cols,&plan,1,scores));h3_gpu_cancel(g);
        CHECK(h3_adaptive_regions_build(&l,&plan,error,sizeof(error)));
        out[0]=0x7fc0;CHECK(h3_gpu_tensor_write_bf16(output,out,n));CHECK(h3_gpu_tensor_write_bf16(input,zero,n));CHECK(h3_gpu_begin(g));
        CHECK(!h3_gpu_adaptive_continuation_probe(g,input,output,old,scratch,cols,&plan,0,scores));h3_gpu_cancel(g);
        h3_gpu_tensor_free(input);h3_gpu_tensor_free(output);h3_gpu_tensor_free(old);h3_gpu_tensor_free(scratch);
        free(zero);free(anchor);free(out);free(got);h3_layout_free(&l);
    }
    return 0;
}
int main(void) {
    char error[512];h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));CHECK(g);
    CHECK(!continuation_probe(g));
    const unsigned lengths[]={1,15,16,17,63,64,65,127,129};
    for(size_t c=0;c<sizeof(lengths)/sizeof(*lengths);c++) {
        unsigned rows=lengths[c],cols=5376;size_t n=(size_t)rows*cols;
        uint16_t *input=calloc(n,2),*output=malloc(n*2),*anchor=malloc(n*2),*got=malloc(n*2);CHECK(input&&output&&anchor&&got);
        for(size_t i=0;i<n;i++){input[i]=bf((float)((int)(i%13)-6)/32);output[i]=bf((float)((int)(i%29)-14)/16);anchor[i]=bf((float)((int)(i%17)-8)/16);}
        double diff=0,denom=0;for(size_t i=0;i<n;i++){float p=fp(bf(fp(output[i])-fp(input[i])));diff+=fabs((double)p-fp(anchor[i]));denom+=fabs((double)fp(anchor[i]));}
        float expect=(float)((diff/n)/fmax(denom/n,1e-6));
        h3_gpu_tensor *a=h3_gpu_tensor_from_bf16(g,input,n),*b=h3_gpu_tensor_from_bf16(g,output,n),*d=h3_gpu_tensor_from_bf16(g,anchor,n),*scratch=h3_gpu_tensor_new_f32(g,1542);CHECK(a&&b&&d&&scratch);
        float score[3],previous[3];
        for(int repeat=0;repeat<2;repeat++) {
            CHECK(h3_gpu_tensor_write_bf16(a,input,n));CHECK(h3_gpu_begin(g));
            CHECK(h3_gpu_adaptive_probe(g,a,b,d,scratch,rows,cols,0,rows,0,rows,1,score));CHECK(h3_gpu_submit(g));
            CHECK(fabsf(score[0]-expect)<=2e-4f+2e-5f*fabsf(expect));CHECK(score[0]==score[1]&&score[1]==score[2]);
            if(repeat)CHECK(!memcmp(previous,score,sizeof(score)));memcpy(previous,score,sizeof(score));
        }
        CHECK(h3_gpu_tensor_read_bf16(a,got,n));for(size_t i=0;i<n;i++)CHECK(got[i]==bf(fp(output[i])-fp(input[i])));
        unsigned split=rows/2;
        CHECK(h3_gpu_tensor_write_bf16(a,input,n));CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_adaptive_probe(g,a,b,d,scratch,rows,cols,0,split,split,rows-split,1,score));CHECK(h3_gpu_submit(g));
        for(unsigned region=0;region<2;region++) {
            size_t first=region?(size_t)split*cols:0,last=region?n:(size_t)split*cols;
            double numerator=0,denominator=0;
            for(size_t i=first;i<last;i++){numerator+=fabs((double)fp(bf(fp(output[i])-fp(input[i])))-fp(anchor[i]));denominator+=fabs((double)fp(anchor[i]));}
            float expected=last==first?0:(float)((numerator/(last-first))/fmax(denominator/(last-first),1e-6));
            CHECK(fabsf(score[region+1]-expected)<=2e-4f+2e-5f*fabsf(expected));
        }
        CHECK(h3_gpu_begin(g));CHECK(!h3_gpu_adaptive_probe(g,a,a,d,scratch,rows,cols,0,rows,0,rows,1,score));h3_gpu_cancel(g);
        output[n-1]=0x7fc0;CHECK(h3_gpu_tensor_write_bf16(b,output,n));CHECK(h3_gpu_begin(g));
        CHECK(!h3_gpu_adaptive_probe(g,a,b,d,scratch,rows,cols,0,rows,0,rows,0,score));h3_gpu_cancel(g);
        memset(input,0,n*2);CHECK(h3_gpu_tensor_write_bf16(a,input,n));CHECK(h3_gpu_tensor_write_bf16(b,input,n));CHECK(h3_gpu_tensor_write_bf16(d,input,n));CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_adaptive_probe(g,a,b,d,scratch,rows,cols,0,rows,0,rows,1,score));CHECK(h3_gpu_submit(g));CHECK(score[0]==0);
        for(size_t i=0;i<n;i++)output[i]=bf(0x1p-20f);
        CHECK(h3_gpu_tensor_write_bf16(b,output,n));CHECK(h3_gpu_begin(g));
        CHECK(h3_gpu_adaptive_probe(g,a,b,d,scratch,rows,cols,0,rows,0,rows,1,score));CHECK(h3_gpu_submit(g));
        CHECK(fabsf(score[0]-0x1p-20f/1e-6f)<2e-5f);
        h3_gpu_tensor_free(a);h3_gpu_tensor_free(b);h3_gpu_tensor_free(d);h3_gpu_tensor_free(scratch);free(input);free(output);free(anchor);free(got);
    }
    /* A large unchanged multimodal prefix must not hide either target stream.
     * Audio is one contiguous span containing both generated stereo channels;
     * the preceding 26 reference-audio rows must not enter that span. */
    {
        const unsigned rows=1069,cols=5376,video_start=1052,video_rows=17,audio_start=1026,audio_rows=26;
        size_t n=(size_t)rows*cols;uint16_t *zero=calloc(n,2),*anchor=malloc(n*2),*output=malloc(n*2),*got=malloc(n*2);CHECK(zero&&anchor&&output&&got);
        for(size_t i=0;i<n;i++)anchor[i]=output[i]=bf(1);
        for(size_t i=(size_t)video_start*cols;i<n;i++)output[i]=bf(1.125f);
        for(size_t i=(size_t)audio_start*cols;i<(size_t)(audio_start+13)*cols;i++)output[i]=bf(1.25f);
        for(size_t i=(size_t)(audio_start+13)*cols;i<(size_t)video_start*cols;i++)output[i]=bf(1.5f);
        h3_gpu_tensor *input=h3_gpu_tensor_from_bf16(g,zero,n),*out=h3_gpu_tensor_from_bf16(g,output,n),
            *old=h3_gpu_tensor_from_bf16(g,anchor,n),*scratch=h3_gpu_tensor_new_f32(g,1542);CHECK(input&&out&&old&&scratch);
        float scores[3],selected;
        CHECK(h3_gpu_begin(g));CHECK(h3_gpu_adaptive_probe(g,input,out,old,scratch,rows,cols,video_start,video_rows,audio_start,audio_rows,1,scores));CHECK(h3_gpu_submit(g));
        CHECK(fabsf(scores[0]-(17*.125f+13*.25f+13*.5f)/rows)<1e-6f);
        CHECK(scores[1]==.125f&&scores[2]==.375f);
        CHECK(h3_adaptive_score(3,scores,&selected)&&selected==.375f);
        CHECK(scores[0]<.04f&&selected>.04f);
        for(int region=0;region<4;region++){
            for(size_t i=0;i<n;i++)output[i]=anchor[i];
            unsigned start=region==0?video_start:region==1?audio_start:region==2?audio_start+13:1000;
            unsigned count=region==0?video_rows:region==3?26:13;
            float delta=region==0?.125f:region==1?.25f:region==2?.5f:.75f;
            for(size_t i=(size_t)start*cols;i<(size_t)(start+count)*cols;i++)output[i]=bf(1+delta);
            CHECK(h3_gpu_tensor_write_bf16(input,zero,n));CHECK(h3_gpu_tensor_write_bf16(out,output,n));
            CHECK(h3_gpu_begin(g));CHECK(h3_gpu_adaptive_probe(g,input,out,old,scratch,rows,cols,video_start,video_rows,audio_start,audio_rows,1,scores));CHECK(h3_gpu_submit(g));
            CHECK(fabsf(scores[0]-count*delta/rows)<1e-6f);
            CHECK(scores[1]==(region==0?.125f:0));
            CHECK(scores[2]==(region==1?.125f:region==2?.25f:0));
        }
        /* BF16 suffix reconstruction rounds the sum once. */
        CHECK(h3_gpu_begin(g));CHECK(h3_gpu_add_bf16(g,input,out,old,n));CHECK(h3_gpu_submit(g));
        CHECK(h3_gpu_tensor_read_bf16(input,got,n));
        for(size_t i=0;i<n;i++)CHECK(got[i]==bf(fp(output[i])+fp(anchor[i])));
        h3_gpu_tensor_free(input);h3_gpu_tensor_free(out);h3_gpu_tensor_free(old);h3_gpu_tensor_free(scratch);
        free(zero);free(anchor);free(output);free(got);
    }
    h3_gpu_free(g);puts("PASS adaptive GPU probes, deterministic reductions, BF16 residuals, ragged shapes, zeros and nonfinite recovery");return 0;
}
