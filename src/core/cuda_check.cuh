#pragma once

#include <cuda_runtime.h>
#include <stdexcept>
#include <string>

[[noreturn]] inline void throw_cuda_error(const char* label, cudaError_t error) {
    throw std::runtime_error(
        std::string("CUDA error after ") + label + ": " + cudaGetErrorString(error));
}

inline void check_cuda_or_throw(const char* label) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) throw_cuda_error(label, error);
}

inline void synchronize_cuda_or_throw(const char* label) {
    cudaError_t error = cudaDeviceSynchronize();
    if (error != cudaSuccess) throw_cuda_error(label, error);
}

#define LUX_CUDA_CHECK(call) do { \
    cudaError_t err = (call); \
    if (err != cudaSuccess) { \
        throw_cuda_error(#call, err); \
    } \
} while(0)
