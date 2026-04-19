#include "cuda_check.h"
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void check_gpu_status() {
    size_t free_bytes = 0;
    size_t total_bytes = 0;
    CUDA_CHECK(cudaMemGetInfo(&free_bytes, &total_bytes));
    size_t used_bytes = total_bytes - free_bytes;
    fprintf(stdout, "%zu,%zu\n", used_bytes, total_bytes);
}
