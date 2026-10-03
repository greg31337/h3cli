#include "src/sglang/sglang.h"
#include "src/vae/audio_vae.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv){
    if (argc != 4)
        return 2;
    h3_sglang_exchange(1);
    char error[1024] = {0};
    FILE *f = fopen(argv[2], "rb");
    if (!f)
        return 1;
    fseek(f, 0, SEEK_END);
    long bytes = ftell(f);
    rewind(f);
    if (bytes <= 0 || bytes % 8 || bytes > 15 * 32000 * 8) {
        fclose(f);
        return 1;
    }
 float *pcm=malloc((size_t)bytes);if(!pcm||fread(pcm,1,(size_t)bytes,f)!=(size_t)bytes){fclose(f);free(pcm);return 1;}fclose(f);
 h3_audio_latent out={0};int ok=h3_audio_vae_encode(argv[1],"src/metal/shaders.metal",pcm,(int)(bytes/8),NULL,NULL,&out,error,sizeof(error));free(pcm);
 if(ok){size_t n=(size_t)out.channels*out.stereo*out.length;f=fopen(argv[3],"wbx");ok=f&&fwrite(out.values,4,n,f)==n;if(f&&fclose(f))ok=0;printf("length=%d\n",out.length);}
 h3_audio_latent_free(&out);if(!ok)fprintf(stderr,"%s\n",error);return ok?0:1;
}
