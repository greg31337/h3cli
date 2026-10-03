#include "src/sampling/sampler_state.h"
#include "src/sampling/av_state.h"
#include <stdio.h>
#include <string.h>

int main(int argc, char **argv) {
    if (argc != 3) return 2;
    unsigned char model[32], av[32]; char error[512];
    int ref = strcmp(argv[2], "Ref2VA") == 0;
    if (!h3_sampler_model_fingerprint(argv[1],ref,model,error,sizeof(error)) ||
        !h3_av_state_signature(argv[1],ref,av,error,sizeof(error))) {
        fprintf(stderr,"%s\n",error); return 1;
    }
    for (int i=0;i<32;i++) printf("%02x",model[i]);
    printf(" ");
    for (int i=0;i<32;i++) printf("%02x",av[i]);
    printf("\n");
    return 0;
}
