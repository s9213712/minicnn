#include <cuda_runtime.h>
#include <cmath>
#include "cuda_check.h"

// ============== Activation Backward ==============
__global__ void relu_backward_kernel(float* data, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        grad[idx] = (data[idx] > 0) ? grad[idx] : 0.0f;
    }
}

__global__ void relu_backward_inplace_kernel(float* data_grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        data_grad[idx] = (data_grad[idx] > 0) ? data_grad[idx] : 0.0f;
    }
}

__global__ void sigmoid_backward_kernel(float* data, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float sig = 1.0f / (1.0f + expf(-data[idx]));
        grad[idx] = grad[idx] * sig * (1.0f - sig);
    }
}

__global__ void tanh_backward_kernel(float* data, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float t = tanhf(data[idx]);
        grad[idx] = grad[idx] * (1.0f - t * t);
    }
}

__global__ void silu_backward_kernel(float* data, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float sig = 1.0f / (1.0f + expf(-x));
        grad[idx] = grad[idx] * (sig + x * sig * (1.0f - sig));
    }
}

__global__ void gelu_backward_kernel(float* data, float* grad, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float x2 = x * x;
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x2);
        float t = tanhf(inner);
        float sech2 = 1.0f - t * t;
        float inner_grad = 0.7978845608028654f * (1.0f + 3.0f * 0.044715f * x2);
        float deriv = 0.5f * (1.0f + t) + 0.5f * x * sech2 * inner_grad;
        grad[idx] = grad[idx] * deriv;
    }
}

// ============== MaxPool Backward ==============
__global__ void maxpool_backward_kernel(const float* grad_out, const float* input, float* grad_input,
                                        int n, int c, int h, int w) {
    int out_h = h / 2;
    int out_w = w / 2;

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % c;
    int on = idx / (out_w * out_h * c);

    int h_start = oh * 2;
    int w_start = ow * 2;

    // Find which input element was the max
    float max_val = -1e38f;
    int max_h = h_start, max_w = w_start;
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int in_idx = ((on * c + oc) * h + (h_start + i)) * w + (w_start + j);
            if (input[in_idx] > max_val) {
                max_val = input[in_idx];
                max_h = h_start + i;
                max_w = w_start + j;
            }
        }
    }

    // Propagate gradient only to max location
    int grad_in_idx = ((on * c + oc) * h + max_h) * w + max_w;
    int grad_out_idx = idx;
    atomicAdd(&grad_input[grad_in_idx], grad_out[grad_out_idx]);
}

extern "C" {
    void apply_relu_backward(float* data, float* grad, int size) {
        int tpb = 256;
        relu_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, size);
        CUDA_KERNEL_CHECK();
    }

    void sigmoid_backward(float* data, float* grad, int size) {
        int tpb = 256;
        sigmoid_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, size);
        CUDA_KERNEL_CHECK();
    }

    void tanh_backward(float* data, float* grad, int size) {
        int tpb = 256;
        tanh_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, size);
        CUDA_KERNEL_CHECK();
    }

    void silu_backward(float* data, float* grad, int size) {
        int tpb = 256;
        silu_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, size);
        CUDA_KERNEL_CHECK();
    }

    void gelu_backward(float* data, float* grad, int size) {
        int tpb = 256;
        gelu_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(data, grad, size);
        CUDA_KERNEL_CHECK();
    }

    void maxpool_backward(float* d_grad_out, float* d_input, float* d_grad_input, int n, int c, int h, int w) {
        int out_h = h / 2;
        int out_w = w / 2;
        int size = n * c * out_h * out_w;
        int tpb = 256;
        // Zero grad_input before scatter; kernel uses atomicAdd.
        CUDA_CHECK(cudaMemset(d_grad_input, 0, n * c * h * w * sizeof(float)));
        maxpool_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_grad_out, d_input, d_grad_input, n, c, h, w);
        CUDA_KERNEL_CHECK();
    }
}
