#include "src/sglang/sglang.h"
#include "src/execution.h"
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
static int checks;
#define CHECK(x) do {checks++;if(!(x)){fprintf(stderr,"FAIL %d: %s (%s)\n",__LINE__,#x,error);return 1;}}while(0)
int main(void){char error[512]={0};h3_params d=H3_PARAMS_DEFAULT,p=d;h3_cuda_policy policy;
 CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));CHECK(policy.active&&policy.base_recipe==4);
 CHECK(!memcmp(&p,&d,sizeof(p)));CHECK(h3_sglang_resolve(&p,"cuda",0)==4);
 for(int a=0;a<4;a++)for(int q=0;q<3;q++){p=d;p.cuda_attention=a;p.cuda_denoise_quant=q;
  CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));CHECK(policy.attention==a&&policy.projection_precision==q);
 }
 p=d;CHECK(h3_cuda_policy_resolve(&p,"cuda",1,&policy,error,sizeof(error)));p.core_reuse=2;CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));
 p=d;p.preview_vae=1;CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error))&&policy.preview);
 p=d;p.still=1;CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error))&&!policy.active);p.cuda_attention=1;CHECK(!h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));
 p=d;CHECK(h3_cuda_policy_resolve(&p,"metal",0,&policy,error,sizeof(error))&&!policy.active);p.cuda_attention=1;CHECK(!h3_cuda_policy_resolve(&p,"metal",0,&policy,error,sizeof(error)));
 p=d;CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&policy,error,sizeof(error)));h3_cuda_policy old=h3_cuda_policy_exchange(policy),inner=policy;inner.attention=2;inner.projection_precision=1;
 h3_cuda_policy saved=h3_cuda_policy_exchange(inner);CHECK(!saved.attention&&!saved.projection_precision);CHECK(h3_cuda_policy_current().attention==2);h3_cuda_policy_exchange(saved);CHECK(!h3_cuda_policy_current().attention);h3_cuda_policy_exchange(old);CHECK(!h3_cuda_policy_current().active);
 printf("PASS %d single-pipeline host policy checks\n",checks);return 0;}
