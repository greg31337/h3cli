#include "src/sglang/sglang.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv){
    if (argc != 3)
        return 2;
    h3_sglang_exchange(1);
    float *pcm = NULL;
    int samples = 0;
    char error[512] = {0};
    if (!h3_sglang_read_soundtrack(argv[1], 124, &pcm, &samples, error, sizeof(error))) {
        fprintf(stderr, "%s\n", error);
        return 1;
    }
 FILE*f=fopen(argv[2],"wbx");int ok=f&&fwrite(pcm,4,(size_t)samples*2,f)==(size_t)samples*2;
 if (f && fclose(f))
     ok = 0;
 free(pcm);
 printf("samples=%d\n", samples);
 return ok ? 0 : 1;
}
