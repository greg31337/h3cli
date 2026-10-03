# PyTorch CPU normal compatibility

`avx_mathfun.h` comes from the installed PyTorch 2.13.0 source headers on the
RTX PRO 5000 qualification server. Its zlib notice is retained verbatim.
Local changes replace the ATen intrinsics include with `immintrin.h` and make
the constant table's existing double-to-float conversions explicit. The
constant values and arithmetic are unchanged.

`src/sglang_rng.cpp` uses the standard C++ MT19937 engine, low 24-bit uniform
conversion, and the same 16-value Box–Muller traversal and tail refill as
PyTorch's contiguous CPU FP32 normal fill. It requires AVX2/FMA and is selected
only by the explicit SGLang CUDA reference recipe. No PyTorch runtime dependency
is introduced. Validation must compare actual installed-oracle tensors before
claiming compatibility with another PyTorch version or CPU implementation.
