#include "cuda_check.h"

// Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Input: (N, C, H, W) -> normalize over (H*W)
// Output: (N, C, H, W)

__device__ float block_reduce_sum(float value, float* shared) {
    int tid = threadIdx.x;
    shared[tid] = value;
    __syncthreads();

    int active = blockDim.x;
    while (active > 1) {
        int half = (active + 1) >> 1;
        if (tid < active - half) {
            shared[tid] += shared[tid + half];
        }
        active = half;
        __syncthreads();
    }
    return shared[0];
}

__global__ void layer_norm_forward_kernel(float* output, const float* input,
                                           const float* gamma, const float* beta,
                                           int N, int C, int H, int W, float eps) {
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;
    int hw = H * W;
    int base = ((n * C + c) * H) * W;

    extern __shared__ float shared[];
    __shared__ float mean;
    __shared__ float inv_std;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        local_sum += input[base + i];
    }
    float sum = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) {
        mean = sum / static_cast<float>(hw);
    }
    __syncthreads();

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        float diff = input[base + i] - mean;
        local_var += diff * diff;
    }
    float var_sum = block_reduce_sum(local_var, shared);
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum / static_cast<float>(hw) + eps);
    }
    __syncthreads();

    float g = gamma[c];
    float b = beta[c];
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        output[base + i] = ((input[base + i] - mean) * inv_std) * g + b;
    }
}

__global__ void layer_norm_backward_kernel(float* grad_input, const float* grad_output,
                                            const float* input, const float* gamma,
                                            int N, int C, int H, int W, float eps) {
    int nc = blockIdx.x;
    int n = nc / C;
    int c = nc % C;
    int hw = H * W;
    int base = ((n * C + c) * H) * W;
    float inv_hw = 1.0f / static_cast<float>(hw);

    extern __shared__ float shared[];
    __shared__ float mean;
    __shared__ float inv_std;
    __shared__ float mean_dy;
    __shared__ float mean_dy_xhat;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        local_sum += input[base + i];
    }
    float sum = block_reduce_sum(local_sum, shared);
    if (threadIdx.x == 0) {
        mean = sum * inv_hw;
    }
    __syncthreads();

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        float diff = input[base + i] - mean;
        local_var += diff * diff;
    }
    float var_sum = block_reduce_sum(local_var, shared);
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum * inv_hw + eps);
    }
    __syncthreads();

    float local_dy = 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        local_dy += grad_output[base + i];
    }
    float dy_sum = block_reduce_sum(local_dy, shared);
    if (threadIdx.x == 0) {
        mean_dy = dy_sum * inv_hw;
    }
    __syncthreads();

    float local_dy_xhat = 0.0f;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        float x_hat = (input[base + i] - mean) * inv_std;
        local_dy_xhat += grad_output[base + i] * x_hat;
    }
    float dy_xhat_sum = block_reduce_sum(local_dy_xhat, shared);
    if (threadIdx.x == 0) {
        mean_dy_xhat = dy_xhat_sum * inv_hw;
    }
    __syncthreads();

    float scale = gamma[c] * inv_std;
    for (int i = threadIdx.x; i < hw; i += blockDim.x) {
        float x_hat = (input[base + i] - mean) * inv_std;
        float dy = grad_output[base + i];
        grad_input[base + i] = scale * (dy - mean_dy - x_hat * mean_dy_xhat);
    }
}

extern "C" void layer_norm_forward(float* d_output, float* d_input,
                                    float* d_gamma, float* d_beta,
                                    int N, int C, int H, int W, float eps) {
    int hw = H * W;
    int tpb = hw < 256 ? hw : 256;
    if (tpb < 1) tpb = 1;
    layer_norm_forward_kernel<<<N * C, tpb, tpb * sizeof(float)>>>(
        d_output, d_input, d_gamma, d_beta, N, C, H, W, eps
    );
    CUDA_KERNEL_CHECK();
}

extern "C" void layer_norm_backward(float* d_grad_input, float* d_grad_output,
                                      float* d_input, float* d_gamma,
                                      int N, int C, int H, int W, float eps) {
    int hw = H * W;
    int tpb = hw < 256 ? hw : 256;
    if (tpb < 1) tpb = 1;
    layer_norm_backward_kernel<<<N * C, tpb, tpb * sizeof(float)>>>(
        d_grad_input, d_grad_output, d_input, d_gamma, N, C, H, W, eps
    );
    CUDA_KERNEL_CHECK();
}
