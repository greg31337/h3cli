/* CPU-only preprocessing diagnostic, linked with the ordinary native library. */
#include "src/sglang/sglang.h"
#include <stdio.h>
#include <stdlib.h>
int main(int argc,char **argv){
    if(argc!=7)return 2;
    float *out=NULL;char error[1024]={0};h3_sglang_exchange(1);
    int iw=atoi(argv[2]),ih=atoi(argv[3]),ow=atoi(argv[4]),oh=atoi(argv[5]);
    if(!h3_sglang_read_reference_image(argv[1],iw,ih,ow,oh,&out,error,sizeof(error))){fprintf(stderr,"%s\n",error);return 1;}
    size_t n=(size_t)ow*oh*3;FILE*f=fopen(argv[6],"wbx");int ok=f&&fwrite(out,4,n,f)==n;
    if (f && fclose(f))
        ok = 0;
    free(out);
    return ok ? 0 : 1;
}
