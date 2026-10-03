/* Replay the production timestep embedding without text, denoising or VAE. */
#include "../src/denoise/dit_schedule.c"
int main(int argc,char **argv) {
    if(argc!=5&&(argc!=6||strcmp(argv[5],"--times"))){fprintf(stderr,"usage: cuda_sglang_time TRANSFORMER FEATURES.f32 ROWS OUTPUT.bf16 [--times]\n");return 2;}
    char *end;long parsed=strtol(argv[3],&end,10);
    if(!*argv[3]||*end||parsed<1||parsed>102)return 2;
    unsigned rows=(unsigned)parsed;size_t n=(size_t)rows*TIME_INPUT;
    float *features=calloc(n,4),*times=NULL;FILE*f=fopen(argv[2],"rb");size_t read_count=argc==6?rows:n;
    int ok=features&&f&&fread(features,4,read_count,f)==read_count&&fgetc(f)==EOF;if(f&&fclose(f))ok=0;
    if(!ok){free(features);return 1;}
    for(size_t i=0;i<n;i++)if(!isfinite(features[i])){free(features);return 2;}
    if(argc==6){times=malloc(rows*4);if(!times){free(features);return 1;}memcpy(times,features,rows*4);}
    char error[1024]={0};h3_sglang_exchange(1);
    h3_weight_store*weights=h3_weight_store_open(argv[1],error,sizeof(error));
    h3_gpu*g=weights?h3_gpu_create("src/metal/shaders.metal",error,sizeof(error)):NULL;
    h3_gpu_tensor*t=g?time_embeddings(weights,g,rows,features,times,0,error,sizeof(error)):NULL;
    size_t count=(size_t)rows*H3_DIT_TIME_DIM;uint16_t*result=malloc(count*2);
    ok=t&&result&&h3_gpu_tensor_read_bf16(t,result,count);
    if(ok){f=fopen(argv[4],"wbx");ok=f&&fwrite(result,2,count,f)==count;if(f&&fclose(f))ok=0;}
    if(!ok)fprintf(stderr,"timestep replay failed: %s\n",*error?error:h3_gpu_error(g));
    free(result);free(features);free(times);h3_gpu_tensor_free(t);h3_gpu_free(g);h3_weight_store_free(weights);return ok?0:1;
}
