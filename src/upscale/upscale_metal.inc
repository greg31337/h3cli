/* Included after native_aux; all weights/activations remain native BF16. */
#include "src/upscale/upscale_gpu.h"
int h3_gpu_upscale_linear(h3_gpu *g,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *weight,uint32_t rows,uint32_t inputs,uint32_t outputs) {
    return h3_gpu_linear_bf16(g,out,in,weight,NULL,rows,inputs,outputs);
}
int h3_gpu_upscale(h3_gpu *opaque,h3_gpu_tensor *out,const h3_gpu_tensor *in,
    const h3_gpu_tensor *a,const h3_gpu_tensor *b,h3_upscale_gpu_op op,h3_upscale_gpu_shape p) {
    H3GPU *g=GPU(opaque);size_t n[4];
    if(!h3_upscale_gpu_counts(op,p,n))return 0;
    if(!h3_gpu_require_bf16(g,out,n[0],@"upscale output")||
       !h3_gpu_require_bf16(g,in,n[1],@"upscale input")||
       (n[2]&&!h3_gpu_require_bf16(g,a,n[2],@"upscale weights"))||
       (n[3]&&!h3_gpu_require_bf16(g,b,n[3],@"upscale bias"))||
       ((op==H3_UP_COLUMNS||op==H3_UP_RESIZE)&&out==in))return 0;
    uint32_t kind=(uint32_t)op;
    return native_aux(g,op==H3_UP_NORM?@"h3_up_norm":@"h3_up_point",
        MTLSizeMake(op==H3_UP_NORM?32:(n[0]+255)/256,1,1),MTLSizeMake(256,1,1),
        ^(id<MTLComputeCommandEncoder> e) {
            [e setBuffer:TENSOR(out).buffer offset:0 atIndex:0];
            [e setBuffer:TENSOR(in).buffer offset:0 atIndex:1];
            [e setBuffer:TENSOR(a?a:in).buffer offset:0 atIndex:2];
            [e setBuffer:TENSOR(b?b:in).buffer offset:0 atIndex:3];
            [e setBytes:&p length:sizeof(p) atIndex:4];
            if(op!=H3_UP_NORM)[e setBytes:&kind length:sizeof(kind) atIndex:5];
        });
}
