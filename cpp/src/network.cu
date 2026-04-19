#include "network.h"
#include "tensor.h"
#include "cuda_check.h"
#include <iostream>
#include <cuda_runtime.h>
#include <memory>

extern "C" {
    void apply_maxpool(float* d_input, float* d_output, int n, int c, int h, int w);
    void im2col_forward(float* d_input, float* d_output, int N, int C, int H, int W, int KH, int KW, int outH, int outW);
    void gemm_forward(float* d_A, float* d_B, float* d_C, int M, int N, int K);
}

__global__ void relu_forward_copy_kernel(const float* input, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        output[idx] = fmaxf(0.0f, input[idx]);
    }
}

std::unique_ptr<CudaTensor> ConvLayer::forward(CudaTensor* input) {
    int outH = input->h - kh + 1;
    int outW = input->w - kw + 1;
    auto output = std::make_unique<CudaTensor>(input->n, out_c, outH, outW);

    int col_rows = input->c * kh * kw;
    int col_cols = input->n * outH * outW;
    size_t col_cache_size = (size_t)col_rows * col_cols;
    if (col_cache_size > d_col_cache_size) {
        if (d_col_cache != nullptr) {
            CUDA_CHECK(cudaFree(d_col_cache));
        }
        CUDA_CHECK(cudaMalloc(&d_col_cache, col_cache_size * sizeof(float)));
        d_col_cache_size = col_cache_size;
    }

    // 1. Image -> Column (GPU)
    im2col_forward(input->data, d_col_cache, input->n, input->c, input->h, input->w, kh, kw, outH, outW);

    // 2. GEMM: Output = Weights * Col (GPU)
    gemm_forward(d_weights, d_col_cache, output->data, out_c, col_cols, col_rows);

    return output;
}

std::unique_ptr<CudaTensor> ReLULayer::forward(CudaTensor* input) {
    auto output = std::make_unique<CudaTensor>(input->n, input->c, input->h, input->w);
    int size = input->n * input->c * input->h * input->w;

    int tpb = 256;
    relu_forward_copy_kernel<<<(size + tpb - 1) / tpb, tpb>>>(input->data, output->data, size);
    CUDA_KERNEL_CHECK();

    return output;
}

std::unique_ptr<CudaTensor> MaxPoolLayer::forward(CudaTensor* input) {
    auto output = std::make_unique<CudaTensor>(input->n, input->c, input->h / 2, input->w / 2);
    apply_maxpool(input->data, output->data, input->n, input->c, input->h, input->w);
    return output;
}
