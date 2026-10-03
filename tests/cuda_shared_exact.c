/* Shared-base substitutions: exact operators, cache keys and error recovery. */
#define main reference_isolation_main
#include "cuda_sglang_isolation.c"
#undef main
int main(void) {
    h3_params p=H3_PARAMS_DEFAULT;h3_cuda_policy fast;
    CHECK(h3_cuda_policy_resolve(&p,"cuda",0,&fast,error,sizeof(error)));
    h3_cuda_policy old=h3_cuda_policy_exchange(fast);
    uint16_t *a=evaluate(1);free(a);
    host_weight_cache();
    CHECK(!setenv("H3_TEST_EXACT_NO_MMAP","1",1));host_weight_cache();
    CHECK(!unsetenv("H3_TEST_EXACT_NO_MMAP"));
    decoder_failure_recovery();
    h3_cuda_policy_exchange(old);
    puts("PASS: single-pipeline shared operators, mapped/copied host weights, invalidation, teardown and decoder recovery");
    return 0;
}
