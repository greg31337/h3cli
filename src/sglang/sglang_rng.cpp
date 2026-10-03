/* Pinned PyTorch 2.13 contiguous CPU FP32 normal fill, implemented without
 * libtorch. The upstream AVX math helper retains its zlib notice. */
#include "src/sglang/sglang.h"
#include <random>
#include <cmath>
#if defined(__x86_64__)
#pragma GCC push_options
#pragma GCC target("avx2,fma")
#pragma GCC optimize("fp-contract=fast")
#define CPU_CAPABILITY_AVX2
#include "third_party/torch-rng/avx_mathfun.h"
#undef CPU_CAPABILITY_AVX2
static void normal16(float *data) {
    __m256 u1=_mm256_sub_ps(_mm256_set1_ps(1),_mm256_loadu_ps(data));
    __m256 u2=_mm256_loadu_ps(data+8);
    __m256 radius=_mm256_sqrt_ps(_mm256_mul_ps(_mm256_set1_ps(-2),log256_ps(u1)));
    __m256 angle=_mm256_mul_ps(_mm256_set1_ps(6.2831853071795864769f),u2),s,c;
    sincos256_ps(angle,&s,&c);
    _mm256_storeu_ps(data,_mm256_mul_ps(radius,c));
    _mm256_storeu_ps(data+8,_mm256_mul_ps(radius,s));
}
#pragma GCC pop_options
#endif
extern "C" int h3_sglang_normal(uint64_t seed,float *out,size_t count) {
    if(!out||count<16)return 0;
#if defined(__x86_64__)
    if(!__builtin_cpu_supports("avx2")||!__builtin_cpu_supports("fma"))return 0;
    std::mt19937 rng(static_cast<uint32_t>(seed));
    auto uniform=[&](){return static_cast<float>(rng()&0xffffffu)*0x1p-24f;};
    for(size_t i=0;i<count;i++)out[i]=uniform();
    for(size_t i=0;i+15<count;i+=16)normal16(out+i);
    if(count%16){for(size_t i=count-16;i<count;i++)out[i]=uniform();normal16(out+count-16);}
    return 1;
#else
    (void)seed;return 0;
#endif
}
