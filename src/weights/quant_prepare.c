/* Explicit artifact preparation without encoders, denoising or VAE work. */
#include "src/gpu.h"
#include "src/weights/quant.h"
#include "src/execution.h"
#include "src/weights/weights.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv) {
    int mode;char error[512]={0};
    if(argc!=4||!h3_quant_parse(argv[1],&mode)||!mode) {
        fprintf(stderr,"usage: %s fp8|nvfp4 TRANSFORMER_DIRECTORY CACHE_DIRECTORY\n",argv[0]);return 2;
    }
    if(!h3_quant_preflight(mode,error,sizeof(error))){fprintf(stderr,"h3cli: %s\n",error);return 1;}
    setenv("H3_CUDA_WEIGHT_MODE","stream",1);
    h3_weight_store *store=h3_weight_store_open(argv[2],error,sizeof(error));
    if(!store){fprintf(stderr,"h3cli: %s\n",error);return 1;}
    h3_gpu *gpu=h3_gpu_create("src/metal/shaders.metal",error,sizeof(error));int ok=gpu!=NULL;
    if(ok)ok=h3_gpu_quant_configure(gpu,mode,argv[3],1<<20,1<<20,1);
    const char *names[]={"attn.qkv_proj.weight","attn.out_proj.weight","mlp.fc1.weight","mlp.fc2.weight"};
    unsigned dims[][2]={{21504,5376},{5376,7168},{28672,5376},{5376,14336}};
    for(unsigned block=0;ok&&block<50;block++)for(unsigned i=0;ok&&i<4;i++) {
        char name[128];snprintf(name,sizeof(name),"blocks.%u.%s",block,names[i]);
        const h3_st_header *header;const h3_st_tensor *tensor=h3_weight_find(store,name,&header);
        if(!tensor||tensor->dtype!=H3_DTYPE_BF16||tensor->ndim!=2||tensor->shape[0]!=dims[i][0]||tensor->shape[1]!=dims[i][1]) {
            snprintf(error,sizeof(error),"missing/invalid %s",name);ok=0;break;
        }
        h3_gpu_tensor *weight=h3_gpu_quant_load(gpu,header->path,tensor->file_offset,dims[i][0],dims[i][1]);
        ok=weight!=NULL;h3_gpu_tensor_free(weight);
        if(ok)fprintf(stderr,"quantized weights %u/200\n",block*4+i+1);
    }
    if(!ok)fprintf(stderr,"h3cli: %s %s\n",error,gpu?h3_gpu_error(gpu):"");
    h3_gpu_free(gpu);h3_weight_store_free(store);return ok?0:1;
}
