/* Model-backed public API policy scopes and prepared-context isolation. */
#include "src/h3.h"
#include "src/denoise/attention.h"
#include "src/sampling/av_state.h"
#include "src/internal.h"
#include "src/denoise/dit.h"
#include "src/gpu.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#define CHECK(x) do{if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,ctx?h3_last_error(ctx):"load");return 1;}}while(0)
int main(int argc,char **argv) {
    if(argc!=6)return 2;
    h3_ctx *ctx=h3_load_dir(argv[1]);CHECK(ctx);h3_cache_set_enabled(ctx,1);
    h3_params p=H3_PARAMS_DEFAULT;
    p.width=p.height=128;p.frames=22;p.steps=2;p.seed=1001;
    p.denoise_reuse=p.core_reuse=1;p.preview_vae=1;p.preview_vae_model=argv[3];
    p.cuda_denoise_quant=2;p.cuda_denoise_quant_cache=argv[4];
    h3_reference ref={H3_REFERENCE_IMAGE,argv[2],NULL,0};p.references=&ref;p.reference_count=1;
    const int modes[]={0,1,2,1};char output[4096];char *first_key=NULL;h3_result *saved=NULL;
    for(unsigned i=0;i<4;i++) {
        p.cuda_attention=modes[i];p.cuda_attention_set=1;
        snprintf(output,sizeof(output),"%s/context-%u.mp4",argv[5],i);p.output_path=output;
        h3_result *r=h3_generate(ctx,"A person smiles and waves in a sunny garden. Birds sing softly.",&p);CHECK(r&&r->status==H3_RESULT_COMPLETE);
        CHECK(h3_attention_current()==0);
        h3_gpu_stats stats;CHECK(ctx->dit&&h3_dit_get_gpu_stats(ctx->dit,&stats));
        CHECK((modes[i]==1)==(stats.sage2_attention_dispatches>0));
        CHECK((modes[i]==2)==(stats.sage3_attention_dispatches>0));
        CHECK(ctx->dit_key);
        if(i==0)CHECK(!strstr(ctx->dit_key,"|attention="));
        if(i==1){saved=r;first_key=strdup(ctx->dit_key);CHECK(first_key);}
        else if(i==3){
            CHECK(!strcmp(first_key,ctx->dit_key));
            CHECK(saved->av_state->info.video_elements==r->av_state->info.video_elements);
            CHECK(!memcmp(saved->av_state->video,r->av_state->video,r->av_state->info.video_elements*sizeof(float)));
            CHECK(!memcmp(saved->av_state->audio,r->av_state->audio,r->av_state->info.audio_elements*sizeof(float)));
            h3_result_free(r);
        }
        else h3_result_free(r);
    }
    h3_result_free(saved);free(first_key);
    for(int mode=1;mode<=2;mode++) {
      for(int feature=0;feature<3;feature++) {
        p.cuda_attention=mode;p.steps=4;p.denoise_reuse=feature==0?2:1;p.core_reuse=feature==1?2:1;p.token_reduction=feature==2;
        snprintf(output,sizeof(output),"%s/context-%d-feature-%d.mp4",argv[5],mode,feature);p.output_path=output;
        h3_result *r=h3_generate(ctx,"A person smiles and waves in a sunny garden. Birds sing softly.",&p);CHECK(r&&r->status==H3_RESULT_COMPLETE);h3_result_free(r);
        CHECK(h3_attention_current()==0);
      }
    }
    h3_cache_clear(ctx);h3_cache_info info;h3_cache_get_info(ctx,&info);CHECK(!info.prepared_dit&&!info.video_decoder);
    h3_free(ctx);puts("PASS alternating attention contexts, scope restoration, reuse/core/reduction and cleanup");return 0;
}
