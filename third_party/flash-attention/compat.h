/* h3 host-only error adapter for the pinned inference kernel. */
#pragma once
#include <cuda_runtime.h>
#include <stdexcept>
inline void h3_flash_cuda_check(cudaError_t status) {
    if(status!=cudaSuccess)throw std::runtime_error(cudaGetErrorString(status));
}
#define C10_CUDA_CHECK(call) h3_flash_cuda_check(call)
#define C10_CUDA_KERNEL_LAUNCH_CHECK() h3_flash_cuda_check(cudaGetLastError())
