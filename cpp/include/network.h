#pragma once
#include "cuda_check.h"
#include "tensor.h"
#include <cuda_runtime.h>
#include <memory>
#include <vector>
#include <string>

// 層類基類
class Layer {
public:
    virtual ~Layer() {}
    virtual std::unique_ptr<CudaTensor> forward(CudaTensor* input) = 0;
};

// 卷積層
class ConvLayer : public Layer {
    int in_c, out_c, kh, kw;
    float* d_weights;
    float* d_grad_weights;
    float* d_col_cache;
    size_t d_col_cache_size;
public:
    ConvLayer(int in_c, int out_c, int kh, int kw)
        : in_c(in_c), out_c(out_c), kh(kh), kw(kw), d_col_cache(nullptr), d_col_cache_size(0) {
        size_t size = (size_t)out_c * in_c * kh * kw * sizeof(float);
        CUDA_CHECK(cudaMalloc(&d_weights, size));
        CUDA_CHECK(cudaMalloc(&d_grad_weights, size));
        CUDA_CHECK(cudaMemset(d_grad_weights, 0, size));
    }
    ~ConvLayer() {
        if (d_col_cache != nullptr) {
            CUDA_CHECK(cudaFree(d_col_cache));
        }
        CUDA_CHECK(cudaFree(d_weights));
        CUDA_CHECK(cudaFree(d_grad_weights));
    }

    void set_weights(const float* weights) {
        size_t size = (size_t)out_c * in_c * kh * kw * sizeof(float);
        CUDA_CHECK(cudaMemcpy(d_weights, weights, size, cudaMemcpyHostToDevice));
    }

    float* get_weights() { return d_weights; }
    float* get_grad_weights() { return d_grad_weights; }
    void clear_grads() {
        size_t size = (size_t)out_c * in_c * kh * kw * sizeof(float);
        CUDA_CHECK(cudaMemset(d_grad_weights, 0, size));
    }

    std::unique_ptr<CudaTensor> forward(CudaTensor* input) override;
};

// 激活層 (ReLU)
class ReLULayer : public Layer {
public:
    std::unique_ptr<CudaTensor> forward(CudaTensor* input) override;
};

// 池化層 (MaxPool)
class MaxPoolLayer : public Layer {
public:
    std::unique_ptr<CudaTensor> forward(CudaTensor* input) override;
};
