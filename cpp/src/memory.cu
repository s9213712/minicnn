#include <cuda_runtime.h>
#include <cstdlib>
#include "cuda_check.h"

extern "C" {
    int cuda_runtime_status() {
        int device_count = 0;
        cudaError_t err = cudaGetDeviceCount(&device_count);
        if (err != cudaSuccess) {
            cudaGetLastError();
            return static_cast<int>(err);
        }
        return 0;
    }

    int cuda_runtime_driver_version() {
        int version = 0;
        cudaError_t err = cudaDriverGetVersion(&version);
        if (err != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return version;
    }

    int cuda_runtime_version() {
        int version = 0;
        cudaError_t err = cudaRuntimeGetVersion(&version);
        if (err != cudaSuccess) {
            cudaGetLastError();
            return 0;
        }
        return version;
    }

    const char* cuda_runtime_status_string(int status) {
        return cudaGetErrorString(static_cast<cudaError_t>(status));
    }

    void* gpu_malloc(size_t size) {
        void* ptr = nullptr;
        CUDA_CHECK(cudaMalloc(&ptr, size));
        return ptr;
    }

    void gpu_free(void* ptr) {
        if (ptr == nullptr) return;
        CUDA_CHECK(cudaFree(ptr));
    }

    void gpu_memcpy_h2d(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice));
    }

    void gpu_memcpy_d2h(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost));
    }

    void gpu_memcpy_d2d(void* dst, const void* src, size_t size) {
        CUDA_CHECK(cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice));
    }

    void gpu_memset(void* dst, int value, size_t size) {
        CUDA_CHECK(cudaMemset(dst, value, size));
    }

    void gpu_synchronize() {
        CUDA_CHECK(cudaDeviceSynchronize());
    }
}
