#include "cuda_check.h"
#include <cuda_runtime.h>
#include <cmath>

// ---------------------------------------------------------------------------
// Block-level helpers
// ---------------------------------------------------------------------------

__device__ float bn_warp_reduce_sum(float val) {
    for (int off = 16; off > 0; off >>= 1)
        val += __shfl_down_sync(0xffffffff, val, off);
    return val;
}

// Reduce two values simultaneously: stores per-warp results in shared[0..nwarps-1]
// and shared[nwarps..2*nwarps-1].  shared must have >= 2*nwarps floats.
__device__ void bn_block_reduce2(float& a, float& b, float* shared) {
    int lane    = threadIdx.x & 31;
    int warp_id = threadIdx.x >> 5;
    int nwarps  = (blockDim.x + 31) >> 5;

    a = bn_warp_reduce_sum(a);
    b = bn_warp_reduce_sum(b);

    if (lane == 0) {
        shared[warp_id]          = a;
        shared[nwarps + warp_id] = b;
    }
    __syncthreads();

    a = (threadIdx.x < nwarps) ? shared[threadIdx.x]          : 0.0f;
    b = (threadIdx.x < nwarps) ? shared[nwarps + threadIdx.x] : 0.0f;
    if (warp_id == 0) {
        a = bn_warp_reduce_sum(a);
        b = bn_warp_reduce_sum(b);
    }
    // broadcast from warp 0 lane 0
    a = __shfl_sync(0xffffffff, a, 0);
    b = __shfl_sync(0xffffffff, b, 0);
}

// Single-value block reduce.  shared must have >= nwarps floats.
__device__ float bn_block_reduce1(float val, float* shared) {
    float dummy = 0.0f;
    bn_block_reduce2(val, dummy, shared);
    return val;
}

// ---------------------------------------------------------------------------
// BatchNorm2d train forward
// Input/output: NCHW layout.
// Normalises over the M = N*H*W elements per channel.
// Saves x_hat (normalised activation), batch_mean, batch_inv_std for backward.
// Updates running stats in-place.
// ---------------------------------------------------------------------------
__global__ void bn_train_forward_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    float* __restrict__ x_hat,
    float* __restrict__ batch_mean,
    float* __restrict__ batch_inv_std,
    float* __restrict__ running_mean,
    float* __restrict__ running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int H, int W,
    float eps, float momentum
) {
    int c   = blockIdx.x;
    if (c >= C) return;

    int HW  = H * W;
    int CHW = C * HW;
    int M   = N * HW;

    // --- shared memory layout: [2 * nwarps] floats for dual reduction ---
    extern __shared__ float shared[];

    // Step 1: compute mean
    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int n  = i / HW;
        int hw = i % HW;
        local_sum += x[n * CHW + c * HW + hw];
    }
    float dummy = 0.0f;
    bn_block_reduce2(local_sum, dummy, shared);
    __shared__ float s_mean, s_inv_std;
    if (threadIdx.x == 0) {
        s_mean = local_sum / static_cast<float>(M);
        batch_mean[c] = s_mean;
        running_mean[c] = (1.0f - momentum) * running_mean[c] + momentum * s_mean;
    }
    __syncthreads();

    // Step 2: compute variance
    float local_var = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int n  = i / HW;
        int hw = i % HW;
        float d = x[n * CHW + c * HW + hw] - s_mean;
        local_var += d * d;
    }
    bn_block_reduce2(local_var, dummy, shared);
    if (threadIdx.x == 0) {
        float var = local_var / static_cast<float>(M);
        s_inv_std = rsqrtf(var + eps);
        batch_inv_std[c] = s_inv_std;
        running_var[c] = (1.0f - momentum) * running_var[c] + momentum * var;
    }
    __syncthreads();

    // Step 3: normalise, scale, shift, save x_hat
    float g = gamma[c];
    float b_val = beta[c];
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int n  = i / HW;
        int hw = i % HW;
        int idx = n * CHW + c * HW + hw;
        float xh = (x[idx] - s_mean) * s_inv_std;
        x_hat[idx] = xh;
        y[idx] = g * xh + b_val;
    }
}

// ---------------------------------------------------------------------------
// BatchNorm2d eval forward (uses running stats, no saved intermediates)
// Applied in-place on y (caller passes y == x if in-place is desired).
// ---------------------------------------------------------------------------
__global__ void bn_eval_forward_kernel(
    float* __restrict__ y,
    const float* __restrict__ x,
    const float* __restrict__ running_mean,
    const float* __restrict__ running_var,
    const float* __restrict__ gamma,
    const float* __restrict__ beta,
    int N, int C, int H, int W, float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * H * W;
    if (idx >= total) return;

    // Derive channel index from NCHW flat index
    int HW = H * W;
    int c  = (idx / HW) % C;

    float inv_std = rsqrtf(running_var[c] + eps);
    float xh = (x[idx] - running_mean[c]) * inv_std;
    y[idx] = gamma[c] * xh + beta[c];
}

// ---------------------------------------------------------------------------
// BatchNorm2d backward
// Computes dx (NCHW), dgamma[C], dbeta[C].
// dgamma and dbeta are accumulated (NOT zeroed first); caller must zero them
// before calling if a fresh accumulation is needed.
// ---------------------------------------------------------------------------
__global__ void bn_backward_kernel(
    float* __restrict__ dx,
    float* __restrict__ dgamma,
    float* __restrict__ dbeta,
    const float* __restrict__ dy,
    const float* __restrict__ x_hat,
    const float* __restrict__ gamma,
    const float* __restrict__ inv_std,
    int N, int C, int H, int W
) {
    int c   = blockIdx.x;
    if (c >= C) return;

    int HW  = H * W;
    int CHW = C * HW;
    int M   = N * HW;

    extern __shared__ float shared[];

    // Reduce sum(dy) and sum(dy * x_hat) for this channel simultaneously.
    float local_dy_sum     = 0.0f;
    float local_dy_xhat    = 0.0f;
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int n  = i / HW;
        int hw = i % HW;
        int idx = n * CHW + c * HW + hw;
        float dyi  = dy[idx];
        float xhi  = x_hat[idx];
        local_dy_sum  += dyi;
        local_dy_xhat += dyi * xhi;
    }
    bn_block_reduce2(local_dy_sum, local_dy_xhat, shared);

    __shared__ float s_mean_dy, s_mean_dy_xhat;
    if (threadIdx.x == 0) {
        float fM = static_cast<float>(M);
        dgamma[c] += local_dy_xhat;   // accumulate gradient for gamma
        dbeta[c]  += local_dy_sum;    // accumulate gradient for beta
        s_mean_dy      = local_dy_sum  / fM;
        s_mean_dy_xhat = local_dy_xhat / fM;
    }
    __syncthreads();

    // Compute dx
    float g_inv_std = gamma[c] * inv_std[c];
    for (int i = threadIdx.x; i < M; i += blockDim.x) {
        int n  = i / HW;
        int hw = i % HW;
        int idx = n * CHW + c * HW + hw;
        float xhi = x_hat[idx];
        float dyi = dy[idx];
        dx[idx] = g_inv_std * (dyi - s_mean_dy - xhi * s_mean_dy_xhat);
    }
}

// ---------------------------------------------------------------------------
// C ABI
// ---------------------------------------------------------------------------
extern "C" {

void bn_train_forward(
    float* d_y,
    const float* d_x,
    float* d_x_hat,
    float* d_batch_mean,
    float* d_batch_inv_std,
    float* d_running_mean,
    float* d_running_var,
    const float* d_gamma,
    const float* d_beta,
    int N, int C, int H, int W,
    float eps, float momentum
) {
    int tpb    = 256;
    int nwarps = (tpb + 31) / 32;
    size_t smem = 2 * nwarps * sizeof(float);
    bn_train_forward_kernel<<<C, tpb, smem>>>(
        d_y, d_x, d_x_hat,
        d_batch_mean, d_batch_inv_std,
        d_running_mean, d_running_var,
        d_gamma, d_beta,
        N, C, H, W, eps, momentum
    );
    CUDA_KERNEL_CHECK();
}

void bn_eval_forward(
    float* d_y,
    const float* d_x,
    const float* d_running_mean,
    const float* d_running_var,
    const float* d_gamma,
    const float* d_beta,
    int N, int C, int H, int W, float eps
) {
    int total = N * C * H * W;
    int tpb   = 256;
    int bpg   = (total + tpb - 1) / tpb;
    bn_eval_forward_kernel<<<bpg, tpb>>>(
        d_y, d_x,
        d_running_mean, d_running_var,
        d_gamma, d_beta,
        N, C, H, W, eps
    );
    CUDA_KERNEL_CHECK();
}

void bn_backward(
    float* d_dx,
    float* d_dgamma,
    float* d_dbeta,
    const float* d_dy,
    const float* d_x_hat,
    const float* d_gamma,
    const float* d_inv_std,
    int N, int C, int H, int W
) {
    int tpb    = 256;
    int nwarps = (tpb + 31) / 32;
    size_t smem = 2 * nwarps * sizeof(float);
    bn_backward_kernel<<<C, tpb, smem>>>(
        d_dx, d_dgamma, d_dbeta,
        d_dy, d_x_hat, d_gamma, d_inv_std,
        N, C, H, W
    );
    CUDA_KERNEL_CHECK();
}

} // extern "C"
