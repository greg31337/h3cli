#include "src/sampling/sampler_state.h"
#include <stdio.h>
int main(int argc,char **argv) {
    if(argc!=2)return 2;
    uint8_t digest[32];char error[512]={0};
    if(!h3_sampler_model_fingerprint_effective(argv[1],NULL,0,digest,error,sizeof(error))) {
        fprintf(stderr,"%s\n",error);return 1;
    }
    for(unsigned i=0;i<32;i++)printf("%02x",digest[i]);putchar('\n');return 0;
}
