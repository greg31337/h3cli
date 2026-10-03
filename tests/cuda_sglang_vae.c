/* Reuse the production decoder benchmark's raw RGB sink and tile handling.
 * The parity probe uses the sole current CUDA arithmetic recipe. */
#include "src/sglang/sglang.h"
#define main ordinary_vae_benchmark
#include "fast_vae_bench.c"
#undef main
int main(int argc,char **argv) {
    if(argc!=7||strcmp(argv[3],"default"))die("usage: bin/cuda_sglang_vae WEIGHTS STATE default OUTPUT.f32 REPEATS tile|full");
    h3_sglang_exchange(1);
    return ordinary_vae_benchmark(argc,argv);
}
