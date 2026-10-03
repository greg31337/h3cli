#include "src/metal/metal_fp16.h"
#include <math.h>
#include <stdio.h>
#include <stdlib.h>

int main(void) {
    char error[4096]={0};int ok=0;
    h3_gpu *g=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));
    h3_gpu_tensor *q=NULL,*k=NULL,*raw=NULL;
    const unsigned n=33*128;uint16_t *values=malloc(n*3*sizeof(*values));
    if(!g||!values)goto done;
    for(unsigned i=0;i<n;i++)values[i]=0x3f00;
    q=h3_gpu_tensor_from_bf16(g,values,n);k=h3_gpu_tensor_from_bf16(g,values,n);
    for(unsigned i=0;i<n;i++){
        unsigned base=(i/128)*384+i%128;
        values[base]=0x3f00;values[base+128]=0xbf80;values[base+256]=0x4780;
    }
    raw=h3_gpu_tensor_from_bf16(g,values,n*3);
    if(!q||!k||!raw||!h3_gpu_begin(g))goto done;
    for(unsigned stage=1;stage<=3;stage++)if(!h3_gpu_diagnostic_range(g,raw,n,128,384,(stage-1)*128,0,stage))goto done;
    if(!h3_gpu_diagnostic_scores(g,q,k,33,1,1.f/sqrtf(128.f),0,0)||!h3_gpu_submit(g)||!h3_gpu_diagnostic_finish(g,0))goto done;
    for(unsigned i=0;i<n;i++)values[i]=0x3f00;
    values[0]=0x7fc0;
    if(!h3_gpu_tensor_write_bf16(q,values,n)||!h3_gpu_begin(g)||
       !h3_gpu_diagnostic_range(g,q,n,n,n,0,0,4)||
       !h3_gpu_diagnostic_scores(g,q,k,33,1,1.f/sqrtf(128.f),0,0)||!h3_gpu_submit(g))goto done;
    if(h3_gpu_diagnostic_finish(g,1))goto done;
    ok=1;
done:
    if(!ok)fprintf(stderr,"diagnostic GPU test failed: %s\n",g?h3_gpu_error(g):error);
    h3_gpu_tensor_free(q);h3_gpu_tensor_free(k);h3_gpu_tensor_free(raw);free(values);h3_gpu_free(g);
    return ok?0:1;
}
