/* Exercise the production full-VAE lifetime checks under the reference policy. */
#include "src/sglang/sglang.h"
#define main existing_lifetime_main
#include "fast_vae_lifetime.c"
#undef main
int main(int argc,char **argv) {
    if(argc!=4||strcmp(argv[3],"legacy"))return 2;
    h3_sglang_exchange(1);
    return existing_lifetime_main(argc,argv);
}
