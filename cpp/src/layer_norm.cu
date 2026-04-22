#include "cuda_check.h"

// Layer Normalization: y = (x - mean) / sqrt(var + eps) * gamma + beta
// Input: (N, C, H, W) -> normalize over (H*W)
// Output: (N, C, H, W)
//
// NOTE: Experimental - not used in the main CIFAR-10 training path.
// The current CUDA training loop uses LeakyReLU activations and does not
// call layer_norm_forward / layer_norm_backward.  This kernel exists as a
// building block for future architectures (e.g. Vision Transformer pre-norm).
// Correctness is verified by test_layer_norm_gradient_matches_pytorch in
// tests/test_layer_norm.py; do not integrate into training without first
// running that test with the actual build.

// Warp-level sum using __shfl_down_sync (no bank conflicts).
__device__ float warp_reduce_sum_ln(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

// Block-level sum: warp shuffle for intra-warp, shared[] for inter-warp stage.
// shared[] must have at least ceil(blockDim.x / 32) elements.
__device__ float block_reduce_sum(float value, float* shared) {
    int tid    = threadIdx.x;
    int lane   = tid & 31;
    int warp_id = tid >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    value = warp_reduce_sum_ln(value);

    if (lane == 0) shared[warp_id] = value;
    __syncthreads();

    // First warp reduces the per-warp results.
    value = (tid < nwarps) ? shared[tid] : 0.0f;
    if (warp_id == 0) value = warp_reduce_sum_ln(value);

    return value;
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

extern "C" void layer_norm_forward(float* d_output, const float* d_input,
                                    const float* d_gamma, const float* d_beta,
                                    int N, int C, int H, int W, float eps) {
    int hw = H * W;
    int tpb = hw < 256 ? hw : 256;
    if (tpb < 1) tpb = 1;
    layer_norm_forward_kernel<<<N * C, tpb, tpb * sizeof(float)>>>(
        d_output, d_input, d_gamma, d_beta, N, C, H, W, eps
    );
    CUDA_KERNEL_CHECK();
}

extern "C" void layer_norm_backward(float* d_grad_input, const float* d_grad_output,
                                      const float* d_input, const float* d_gamma,
                                      int N, int C, int H, int W, float eps) {
    int hw = H * W;
    int tpb = hw < 256 ? hw : 256;
    if (tpb < 1) tpb = 1;
    // extern __shared__ is used only for inter-warp reduction: needs ceil(tpb/32) floats.
    // Static __shared__ scalars in the kernel are separate from this dynamic buffer.
    // Allocate at least tpb floats so the reduction buffer is sufficient even at tpb=1.
    int nwarps = (tpb + 31) / 32;
    size_t shared_bytes = (size_t)(nwarps < tpb ? tpb : nwarps) * sizeof(float);
    layer_norm_backward_kernel<<<N * C, tpb, shared_bytes>>>(
        d_grad_input, d_grad_output, d_input, d_gamma, N, C, H, W, eps
    );
    CUDA_KERNEL_CHECK();
}
