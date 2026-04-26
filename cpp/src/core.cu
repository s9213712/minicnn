#include "tensor.h"
#include "cuda_check.h"
#include <cuda_runtime.h>
#ifndef USE_CUBLAS
#define USE_CUBLAS 1
#endif

#if USE_CUBLAS
#include <cublas_v2.h>
#include "cublas_check.h"
#include "cublas_context.h"
#endif
#include <iostream>
#include <vector>

// -----------------------------------------------------------------------------
// CUDA Kernels
// -----------------------------------------------------------------------------

__global__ void relu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = fmaxf(0.0f, data[idx]);
}

__global__ void sigmoid_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = 1.0f / (1.0f + expf(-data[idx]));
}

__global__ void tanh_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) data[idx] = tanhf(data[idx]);
}

__global__ void silu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        data[idx] = x / (1.0f + expf(-x));
    }
}

__global__ void gelu_forward_kernel(float* data, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float x = data[idx];
        float inner = 0.7978845608028654f * (x + 0.044715f * x * x * x);
        data[idx] = 0.5f * x * (1.0f + tanhf(inner));
    }
}

__global__ void add_forward_kernel(const float* a, const float* b, float* output, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) output[idx] = a[idx] + b[idx];
}

__global__ void concat_forward_kernel(
    const float* a,
    const float* b,
    float* output,
    int outer,
    int a_axis,
    int b_axis,
    int inner
) {
    int out_axis = a_axis + b_axis;
    int total = outer * out_axis * inner;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;

    int inner_idx = idx % inner;
    int axis_idx = (idx / inner) % out_axis;
    int outer_idx = idx / (inner * out_axis);
    if (axis_idx < a_axis) {
        int a_idx = (outer_idx * a_axis + axis_idx) * inner + inner_idx;
        output[idx] = a[a_idx];
    } else {
        int b_axis_idx = axis_idx - a_axis;
        int b_idx = (outer_idx * b_axis + b_axis_idx) * inner + inner_idx;
        output[idx] = b[b_idx];
    }
}

__global__ void maxpool_forward_kernel(const float* input, float* output, int n, int c, int h, int w) {
    int out_h = h / 2; int out_w = w / 2;
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = n * c * out_h * out_w;
    if (idx >= total_elements) return;
    int ow = idx % out_w; int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % c; int on = idx / (out_w * out_h * c);
    int in_h_start = oh * 2; int in_w_start = ow * 2;
    float max_val = -1e38f;
    for (int i = 0; i < 2; ++i) {
        for (int j = 0; j < 2; ++j) {
            int in_idx = ((on * c + oc) * h + (in_h_start + i)) * w + (in_w_start + j);
            max_val = fmaxf(max_val, input[in_idx]);
        }
    }
    output[idx] = max_val;
}

__global__ void global_avgpool2d_forward_kernel(const float* input, float* output, int n, int c, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c;
    if (idx >= total) return;
    int channel = idx % c;
    int batch = idx / c;
    int hw = h * w;
    int base = ((batch * c + channel) * h) * w;
    float sum = 0.0f;
    for (int i = 0; i < hw; ++i) {
        sum += input[base + i];
    }
    output[idx] = sum / static_cast<float>(hw);
}

__global__ void global_avgpool2d_backward_kernel(const float* grad_output, float* grad_input, int n, int c, int h, int w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * h * w;
    if (idx >= total) return;
    int hw = h * w;
    int channel = (idx / hw) % c;
    int batch = idx / (c * hw);
    grad_input[idx] = grad_output[batch * c + channel] / static_cast<float>(hw);
}

__global__ void avgpool2d_forward_kernel(
    const float* input,
    float* output,
    int n,
    int c,
    int h,
    int w,
    int out_h,
    int out_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * out_h * out_w;
    if (idx >= total) return;
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int ch = (idx / (out_w * out_h)) % c;
    int batch = idx / (out_w * out_h * c);
    float sum = 0.0f;
    for (int r = 0; r < kh; ++r) {
        int ih = oh * stride_h + r - pad_h;
        if (ih < 0 || ih >= h) continue;
        for (int s = 0; s < kw; ++s) {
            int iw = ow * stride_w + s - pad_w;
            if (iw < 0 || iw >= w) continue;
            sum += input[(batch * c + ch) * h * w + ih * w + iw];
        }
    }
    output[idx] = sum / static_cast<float>(kh * kw);
}

__global__ void avgpool2d_backward_kernel(
    const float* grad_output,
    float* grad_input,
    int n,
    int c,
    int h,
    int w,
    int out_h,
    int out_w,
    int kh,
    int kw,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * c * out_h * out_w;
    if (idx >= total) return;
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int ch = (idx / (out_w * out_h)) % c;
    int batch = idx / (out_w * out_h * c);
    float grad = grad_output[idx] / static_cast<float>(kh * kw);
    for (int r = 0; r < kh; ++r) {
        int ih = oh * stride_h + r - pad_h;
        if (ih < 0 || ih >= h) continue;
        for (int s = 0; s < kw; ++s) {
            int iw = ow * stride_w + s - pad_w;
            if (iw < 0 || iw >= w) continue;
            atomicAdd(&grad_input[(batch * c + ch) * h * w + ih * w + iw], grad);
        }
    }
}

__global__ void depthwise_conv2d_forward_kernel(
    const float* input,
    const float* weight,
    const float* bias,
    float* output,
    int n,
    int c,
    int h,
    int w,
    int out_c,
    int kh,
    int kw,
    int out_h,
    int out_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * out_c * out_h * out_w;
    if (idx >= total) return;
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_c;
    int batch = idx / (out_w * out_h * out_c);
    int multiplier = out_c / c;
    int ic = oc / multiplier;
    float sum = has_bias ? bias[oc] : 0.0f;
    for (int r = 0; r < kh; ++r) {
        int ih = oh * stride_h + r - pad_h;
        if (ih < 0 || ih >= h) continue;
        for (int s = 0; s < kw; ++s) {
            int iw = ow * stride_w + s - pad_w;
            if (iw < 0 || iw >= w) continue;
            int in_idx = ((batch * c + ic) * h + ih) * w + iw;
            int weight_idx = ((oc * 1) * kh + r) * kw + s;
            sum += input[in_idx] * weight[weight_idx];
        }
    }
    output[idx] = sum;
}

__global__ void depthwise_conv2d_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* weight,
    float* grad_input,
    float* grad_weight,
    float* grad_bias,
    int n,
    int c,
    int h,
    int w,
    int out_c,
    int kh,
    int kw,
    int out_h,
    int out_w,
    int stride_h,
    int stride_w,
    int pad_h,
    int pad_w,
    int has_bias
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * out_c * out_h * out_w;
    if (idx >= total) return;
    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int oc = (idx / (out_w * out_h)) % out_c;
    int batch = idx / (out_w * out_h * out_c);
    int multiplier = out_c / c;
    int ic = oc / multiplier;
    float grad = grad_output[idx];
    if (has_bias) {
        atomicAdd(&grad_bias[oc], grad);
    }
    for (int r = 0; r < kh; ++r) {
        int ih = oh * stride_h + r - pad_h;
        if (ih < 0 || ih >= h) continue;
        for (int s = 0; s < kw; ++s) {
            int iw = ow * stride_w + s - pad_w;
            if (iw < 0 || iw >= w) continue;
            int in_idx = ((batch * c + ic) * h + ih) * w + iw;
            int weight_idx = ((oc * 1) * kh + r) * kw + s;
            atomicAdd(&grad_input[in_idx], weight[weight_idx] * grad);
            atomicAdd(&grad_weight[weight_idx], input[in_idx] * grad);
        }
    }
}

__global__ void layernorm2d_forward_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int n,
    int c,
    int h,
    int w,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = n * h * w;
    if (idx >= total) return;
    int spatial = h * w;
    int batch = idx / spatial;
    int hw = idx % spatial;
    float mean = 0.0f;
    for (int ch = 0; ch < c; ++ch) {
        mean += input[(batch * c + ch) * spatial + hw];
    }
    mean /= static_cast<float>(c);
    float var = 0.0f;
    for (int ch = 0; ch < c; ++ch) {
        float diff = input[(batch * c + ch) * spatial + hw] - mean;
        var += diff * diff;
    }
    float inv_std = rsqrtf(var / static_cast<float>(c) + eps);
    for (int ch = 0; ch < c; ++ch) {
        int offset = (batch * c + ch) * spatial + hw;
        output[offset] = ((input[offset] - mean) * inv_std) * gamma[ch] + beta[ch];
    }
}

__device__ float warp_reduce_sum_layernorm_nd(float value) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        value += __shfl_down_sync(0xffffffff, value, offset);
    }
    return value;
}

__device__ float block_reduce_sum_layernorm_nd(float value, float* shared) {
    int tid = threadIdx.x;
    int lane = tid & 31;
    int warp_id = tid >> 5;
    int nwarps = (blockDim.x + 31) >> 5;

    value = warp_reduce_sum_layernorm_nd(value);
    if (lane == 0) {
        shared[warp_id] = value;
    }
    __syncthreads();

    value = (tid < nwarps) ? shared[tid] : 0.0f;
    if (warp_id == 0) {
        value = warp_reduce_sum_layernorm_nd(value);
    }
    return value;
}

__global__ void layernorm_nd_forward_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int rows,
    int features,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int base = row * features;

    extern __shared__ float shared[];
    __shared__ float mean;
    __shared__ float inv_std;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        local_sum += input[base + i];
    }
    float sum = block_reduce_sum_layernorm_nd(local_sum, shared);
    if (threadIdx.x == 0) {
        mean = sum / static_cast<float>(features);
    }
    __syncthreads();

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float diff = input[base + i] - mean;
        local_var += diff * diff;
    }
    float var_sum = block_reduce_sum_layernorm_nd(local_var, shared);
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum / static_cast<float>(features) + eps);
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x_hat = (input[base + i] - mean) * inv_std;
        output[base + i] = x_hat * gamma[i] + beta[i];
    }
}

__global__ void layernorm_nd_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* gamma,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int rows,
    int features,
    float eps
) {
    int row = blockIdx.x;
    if (row >= rows) return;
    int base = row * features;
    float inv_features = 1.0f / static_cast<float>(features);

    extern __shared__ float shared[];
    __shared__ float mean;
    __shared__ float inv_std;
    __shared__ float mean_dxhat;
    __shared__ float mean_dxhat_xhat;

    float local_sum = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        local_sum += input[base + i];
    }
    float sum = block_reduce_sum_layernorm_nd(local_sum, shared);
    if (threadIdx.x == 0) {
        mean = sum * inv_features;
    }
    __syncthreads();

    float local_var = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float diff = input[base + i] - mean;
        local_var += diff * diff;
    }
    float var_sum = block_reduce_sum_layernorm_nd(local_var, shared);
    if (threadIdx.x == 0) {
        inv_std = rsqrtf(var_sum * inv_features + eps);
    }
    __syncthreads();

    float local_dxhat = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        local_dxhat += grad_output[base + i] * gamma[i];
    }
    float dxhat_sum = block_reduce_sum_layernorm_nd(local_dxhat, shared);
    if (threadIdx.x == 0) {
        mean_dxhat = dxhat_sum * inv_features;
    }
    __syncthreads();

    float local_dxhat_xhat = 0.0f;
    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x_hat = (input[base + i] - mean) * inv_std;
        local_dxhat_xhat += grad_output[base + i] * gamma[i] * x_hat;
    }
    float dxhat_xhat_sum = block_reduce_sum_layernorm_nd(local_dxhat_xhat, shared);
    if (threadIdx.x == 0) {
        mean_dxhat_xhat = dxhat_xhat_sum * inv_features;
    }
    __syncthreads();

    for (int i = threadIdx.x; i < features; i += blockDim.x) {
        float x_hat = (input[base + i] - mean) * inv_std;
        float grad_out = grad_output[base + i];
        float dxhat = grad_out * gamma[i];
        grad_input[base + i] = inv_std * (dxhat - mean_dxhat - x_hat * mean_dxhat_xhat);
        atomicAdd(&grad_gamma[i], grad_out * x_hat);
        atomicAdd(&grad_beta[i], grad_out);
    }
}

__global__ void layernorm2d_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* gamma,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int n,
    int c,
    int h,
    int w,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = h * w;
    int total = n * c * spatial;
    if (idx >= total) return;
    int hw = idx % spatial;
    int ch = (idx / spatial) % c;
    int batch = idx / (c * spatial);
    int sample_offset = batch * c * spatial;

    float mean = 0.0f;
    for (int cc = 0; cc < c; ++cc) {
        mean += input[sample_offset + cc * spatial + hw];
    }
    mean /= static_cast<float>(c);

    float var = 0.0f;
    for (int cc = 0; cc < c; ++cc) {
        float diff = input[sample_offset + cc * spatial + hw] - mean;
        var += diff * diff;
    }
    float inv_std = rsqrtf(var / static_cast<float>(c) + eps);

    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (int cc = 0; cc < c; ++cc) {
        int offset = sample_offset + cc * spatial + hw;
        float x_hat = (input[offset] - mean) * inv_std;
        float dxhat = grad_output[offset] * gamma[cc];
        sum_dxhat += dxhat;
        sum_dxhat_xhat += dxhat * x_hat;
    }

    float x_hat_ch = (input[idx] - mean) * inv_std;
    float dxhat_ch = grad_output[idx] * gamma[ch];
    grad_input[idx] = (inv_std / static_cast<float>(c)) * (
        static_cast<float>(c) * dxhat_ch - sum_dxhat - x_hat_ch * sum_dxhat_xhat
    );
    atomicAdd(&grad_gamma[ch], grad_output[idx] * x_hat_ch);
    atomicAdd(&grad_beta[ch], grad_output[idx]);
}

__global__ void groupnorm_forward_kernel(
    const float* input,
    const float* gamma,
    const float* beta,
    float* output,
    int n,
    int c,
    int h,
    int w,
    int num_groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = h * w;
    int total = n * c * spatial;
    if (idx >= total) return;
    int ch = (idx / spatial) % c;
    int batch = idx / (c * spatial);
    int channels_per_group = c / num_groups;
    int group = ch / channels_per_group;
    int channel_start = group * channels_per_group;
    int group_elements = channels_per_group * spatial;
    float mean = 0.0f;
    for (int gc = 0; gc < channels_per_group; ++gc) {
        int cc = channel_start + gc;
        for (int s = 0; s < spatial; ++s) {
            mean += input[(batch * c + cc) * spatial + s];
        }
    }
    mean /= static_cast<float>(group_elements);
    float var = 0.0f;
    for (int gc = 0; gc < channels_per_group; ++gc) {
        int cc = channel_start + gc;
        for (int s = 0; s < spatial; ++s) {
            float diff = input[(batch * c + cc) * spatial + s] - mean;
            var += diff * diff;
        }
    }
    float inv_std = rsqrtf(var / static_cast<float>(group_elements) + eps);
    output[idx] = ((input[idx] - mean) * inv_std) * gamma[ch] + beta[ch];
}

__global__ void groupnorm_backward_kernel(
    const float* grad_output,
    const float* input,
    const float* gamma,
    float* grad_input,
    float* grad_gamma,
    float* grad_beta,
    int n,
    int c,
    int h,
    int w,
    int num_groups,
    float eps
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int spatial = h * w;
    int total = n * c * spatial;
    if (idx >= total) return;
    int ch = (idx / spatial) % c;
    int batch = idx / (c * spatial);
    int channels_per_group = c / num_groups;
    int group = ch / channels_per_group;
    int channel_start = group * channels_per_group;
    int group_elements = channels_per_group * spatial;
    int sample_offset = batch * c * spatial;

    float mean = 0.0f;
    for (int gc = 0; gc < channels_per_group; ++gc) {
        int cc = channel_start + gc;
        for (int s = 0; s < spatial; ++s) {
            mean += input[sample_offset + cc * spatial + s];
        }
    }
    mean /= static_cast<float>(group_elements);

    float var = 0.0f;
    for (int gc = 0; gc < channels_per_group; ++gc) {
        int cc = channel_start + gc;
        for (int s = 0; s < spatial; ++s) {
            float diff = input[sample_offset + cc * spatial + s] - mean;
            var += diff * diff;
        }
    }
    float inv_std = rsqrtf(var / static_cast<float>(group_elements) + eps);

    float sum_dxhat = 0.0f;
    float sum_dxhat_xhat = 0.0f;
    for (int gc = 0; gc < channels_per_group; ++gc) {
        int cc = channel_start + gc;
        for (int s = 0; s < spatial; ++s) {
            int offset = sample_offset + cc * spatial + s;
            float x_hat = (input[offset] - mean) * inv_std;
            float dxhat = grad_output[offset] * gamma[cc];
            sum_dxhat += dxhat;
            sum_dxhat_xhat += dxhat * x_hat;
        }
    }

    float x_hat_ch = (input[idx] - mean) * inv_std;
    float dxhat_ch = grad_output[idx] * gamma[ch];
    grad_input[idx] = (inv_std / static_cast<float>(group_elements)) * (
        static_cast<float>(group_elements) * dxhat_ch - sum_dxhat - x_hat_ch * sum_dxhat_xhat
    );
    atomicAdd(&grad_gamma[ch], grad_output[idx] * x_hat_ch);
    atomicAdd(&grad_beta[ch], grad_output[idx]);
}

__global__ void im2col_kernel(const float* input, float* output, int N, int C, int H, int W, int KH, int KW, int outH, int outW) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total_elements = (C * KH * KW) * (N * outH * outW);
    if (idx >= total_elements) return;
    int row = idx / (N * outH * outW);
    int col = idx % (N * outH * outW);
    int c = row / (KH * KW); int kh = (row / KW) % KH; int kw = row % KW;
    int n = col / (outH * outW); int ow = col % outW; int oh = (col / outW) % outH;
    output[idx] = input[((n * C + c) * H + (oh + kh)) * W + (ow + kw)];
}

#if !USE_CUBLAS
// Tiled shared-memory GEMM: C[M,N] = A[M,K] * B[K,N], row-major.
// TILE_SIZE should be a power of 2; 16 gives a good occupancy/register balance.
#define TILE_SIZE 16
__global__ void gemm_kernel(const float* __restrict__ A,
                             const float* __restrict__ B,
                             float* C_out,
                             int M, int N, int K) {
    __shared__ float tileA[TILE_SIZE][TILE_SIZE];
    __shared__ float tileB[TILE_SIZE][TILE_SIZE];

    int row = blockIdx.y * TILE_SIZE + threadIdx.y;
    int col = blockIdx.x * TILE_SIZE + threadIdx.x;
    float sum = 0.0f;

    for (int t = 0; t * TILE_SIZE < K; ++t) {
        int aCol = t * TILE_SIZE + threadIdx.x;
        int bRow = t * TILE_SIZE + threadIdx.y;
        tileA[threadIdx.y][threadIdx.x] = (row < M && aCol < K) ? A[row * K + aCol] : 0.0f;
        tileB[threadIdx.y][threadIdx.x] = (bRow < K && col < N) ? B[bRow * N + col] : 0.0f;
        __syncthreads();

        for (int k = 0; k < TILE_SIZE; ++k)
            sum += tileA[threadIdx.y][k] * tileB[k][threadIdx.x];
        __syncthreads();
    }

    if (row < M && col < N)
        C_out[row * N + col] = sum;
}
#undef TILE_SIZE
#endif

extern "C" {
    void im2col_forward(float* d_input, float* d_output, int N, int C, int H, int W, int KH, int KW, int outH, int outW) {
        int total_elements = (C * KH * KW) * (N * outH * outW);
        int tpb = 256;
        im2col_kernel<<<(total_elements + tpb - 1) / tpb, tpb>>>(d_input, d_output, N, C, H, W, KH, KW, outH, outW);
        CUDA_KERNEL_CHECK();
    }

    void gemm_forward(float* d_A, float* d_B, float* d_C, int M, int N, int K) {
#if USE_CUBLAS
        cublasHandle_t handle = minicnn_get_cublas_handle();
        const float alpha = 1.0f;
        const float beta = 0.0f;

        // Row-major C[M, N] = A[M, K] * B[K, N].
        // cuBLAS is column-major, so compute C^T[N, M] = B^T[N, K] * A^T[K, M].
        CUBLAS_CHECK(cublasSgemm(
            handle,
            CUBLAS_OP_N,
            CUBLAS_OP_N,
            N,
            M,
            K,
            &alpha,
            d_B,
            N,
            d_A,
            K,
            &beta,
            d_C,
            N
        ));
#else
        dim3 block(16, 16);
        dim3 grid((N + 15) / 16, (M + 15) / 16);
        gemm_kernel<<<grid, block>>>(d_A, d_B, d_C, M, N, K);
        CUDA_KERNEL_CHECK();
#endif
    }

    void apply_relu(float* d_data, int size) {
        int tpb = 256;
        relu_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_data, size);
        CUDA_KERNEL_CHECK();
    }

    void sigmoid_forward(float* d_data, int size) {
        int tpb = 256;
        sigmoid_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_data, size);
        CUDA_KERNEL_CHECK();
    }

    void tanh_forward(float* d_data, int size) {
        int tpb = 256;
        tanh_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_data, size);
        CUDA_KERNEL_CHECK();
    }

    void silu_forward(float* d_data, int size) {
        int tpb = 256;
        silu_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_data, size);
        CUDA_KERNEL_CHECK();
    }

    void gelu_forward(float* d_data, int size) {
        int tpb = 256;
        gelu_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_data, size);
        CUDA_KERNEL_CHECK();
    }

    void add_forward(float* d_a, float* d_b, float* d_output, int size) {
        int tpb = 256;
        add_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_a, d_b, d_output, size);
        CUDA_KERNEL_CHECK();
    }

    void concat_forward(float* d_a, float* d_b, float* d_output, int outer, int a_axis, int b_axis, int inner) {
        int total = outer * (a_axis + b_axis) * inner;
        int tpb = 256;
        concat_forward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(d_a, d_b, d_output, outer, a_axis, b_axis, inner);
        CUDA_KERNEL_CHECK();
    }

    void apply_maxpool(float* d_input, float* d_output, int n, int c, int h, int w) {
        int out_h = h / 2; int out_w = w / 2;
        int size = n * c * out_h * out_w;
        int tpb = 256;
        maxpool_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_input, d_output, n, c, h, w);
        CUDA_KERNEL_CHECK();
    }

    void global_avgpool2d_forward(float* d_input, float* d_output, int n, int c, int h, int w) {
        int size = n * c;
        int tpb = 256;
        global_avgpool2d_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_input, d_output, n, c, h, w);
        CUDA_KERNEL_CHECK();
    }

    void global_avgpool2d_backward(float* d_grad_output, float* d_grad_input, int n, int c, int h, int w) {
        int size = n * c * h * w;
        int tpb = 256;
        global_avgpool2d_backward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(d_grad_output, d_grad_input, n, c, h, w);
        CUDA_KERNEL_CHECK();
    }

    void avgpool2d_forward(
        float* d_input,
        float* d_output,
        int n,
        int c,
        int h,
        int w,
        int out_h,
        int out_w,
        int kh,
        int kw,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w
    ) {
        int size = n * c * out_h * out_w;
        int tpb = 256;
        avgpool2d_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(
            d_input, d_output, n, c, h, w, out_h, out_w,
            kh, kw, stride_h, stride_w, pad_h, pad_w
        );
        CUDA_KERNEL_CHECK();
    }

    void avgpool2d_backward(
        float* d_grad_output,
        float* d_grad_input,
        int n,
        int c,
        int h,
        int w,
        int out_h,
        int out_w,
        int kh,
        int kw,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w
    ) {
        int input_size = n * c * h * w;
        int output_size = n * c * out_h * out_w;
        int tpb = 256;
        cudaMemset(d_grad_input, 0, input_size * sizeof(float));
        CUDA_KERNEL_CHECK();
        avgpool2d_backward_kernel<<<(output_size + tpb - 1) / tpb, tpb>>>(
            d_grad_output, d_grad_input, n, c, h, w, out_h, out_w,
            kh, kw, stride_h, stride_w, pad_h, pad_w
        );
        CUDA_KERNEL_CHECK();
    }

    void depthwise_conv2d_forward(
        float* d_input,
        float* d_weight,
        float* d_bias,
        float* d_output,
        int n,
        int c,
        int h,
        int w,
        int out_c,
        int kh,
        int kw,
        int out_h,
        int out_w,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        int has_bias
    ) {
        int size = n * out_c * out_h * out_w;
        int tpb = 256;
        depthwise_conv2d_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(
            d_input, d_weight, d_bias, d_output,
            n, c, h, w, out_c, kh, kw, out_h, out_w,
            stride_h, stride_w, pad_h, pad_w, has_bias
        );
        CUDA_KERNEL_CHECK();
    }

    void depthwise_conv2d_backward(
        float* d_grad_output,
        float* d_input,
        float* d_weight,
        float* d_grad_input,
        float* d_grad_weight,
        float* d_grad_bias,
        int n,
        int c,
        int h,
        int w,
        int out_c,
        int kh,
        int kw,
        int out_h,
        int out_w,
        int stride_h,
        int stride_w,
        int pad_h,
        int pad_w,
        int has_bias
    ) {
        int input_size = n * c * h * w;
        int weight_size = out_c * kh * kw;
        int bias_size = out_c;
        int output_size = n * out_c * out_h * out_w;
        int tpb = 256;
        cudaMemset(d_grad_input, 0, input_size * sizeof(float));
        CUDA_KERNEL_CHECK();
        cudaMemset(d_grad_weight, 0, weight_size * sizeof(float));
        CUDA_KERNEL_CHECK();
        if (has_bias) {
            cudaMemset(d_grad_bias, 0, bias_size * sizeof(float));
            CUDA_KERNEL_CHECK();
        }
        depthwise_conv2d_backward_kernel<<<(output_size + tpb - 1) / tpb, tpb>>>(
            d_grad_output, d_input, d_weight, d_grad_input, d_grad_weight, d_grad_bias,
            n, c, h, w, out_c, kh, kw, out_h, out_w, stride_h, stride_w, pad_h, pad_w, has_bias
        );
        CUDA_KERNEL_CHECK();
    }

    void layernorm2d_forward(float* d_input, float* d_gamma, float* d_beta, float* d_output, int n, int c, int h, int w, float eps) {
        int size = n * h * w;
        int tpb = 256;
        layernorm2d_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(
            d_input, d_gamma, d_beta, d_output, n, c, h, w, eps
        );
        CUDA_KERNEL_CHECK();
    }

    void layernorm2d_backward(
        float* d_grad_output,
        float* d_input,
        float* d_gamma,
        float* d_grad_input,
        float* d_grad_gamma,
        float* d_grad_beta,
        int n,
        int c,
        int h,
        int w,
        float eps
    ) {
        int total = n * c * h * w;
        int tpb = 256;
        cudaMemset(d_grad_gamma, 0, c * sizeof(float));
        CUDA_KERNEL_CHECK();
        cudaMemset(d_grad_beta, 0, c * sizeof(float));
        CUDA_KERNEL_CHECK();
        layernorm2d_backward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(
            d_grad_output, d_input, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta,
            n, c, h, w, eps
        );
        CUDA_KERNEL_CHECK();
    }

    void layernorm_nd_forward(
        float* d_input,
        float* d_gamma,
        float* d_beta,
        float* d_output,
        int rows,
        int features,
        float eps
    ) {
        int tpb = features < 256 ? features : 256;
        if (tpb < 1) tpb = 1;
        int nwarps = (tpb + 31) / 32;
        size_t shared_bytes = (size_t)(nwarps < tpb ? tpb : nwarps) * sizeof(float);
        layernorm_nd_forward_kernel<<<rows, tpb, shared_bytes>>>(
            d_input, d_gamma, d_beta, d_output, rows, features, eps
        );
        CUDA_KERNEL_CHECK();
    }

    void layernorm_nd_backward(
        float* d_grad_output,
        float* d_input,
        float* d_gamma,
        float* d_grad_input,
        float* d_grad_gamma,
        float* d_grad_beta,
        int rows,
        int features,
        float eps
    ) {
        int tpb = features < 256 ? features : 256;
        if (tpb < 1) tpb = 1;
        int nwarps = (tpb + 31) / 32;
        size_t shared_bytes = (size_t)(nwarps < tpb ? tpb : nwarps) * sizeof(float);
        cudaMemset(d_grad_gamma, 0, features * sizeof(float));
        CUDA_KERNEL_CHECK();
        cudaMemset(d_grad_beta, 0, features * sizeof(float));
        CUDA_KERNEL_CHECK();
        layernorm_nd_backward_kernel<<<rows, tpb, shared_bytes>>>(
            d_grad_output, d_input, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta, rows, features, eps
        );
        CUDA_KERNEL_CHECK();
    }

    void groupnorm_forward(float* d_input, float* d_gamma, float* d_beta, float* d_output, int n, int c, int h, int w, int num_groups, float eps) {
        int size = n * c * h * w;
        int tpb = 256;
        groupnorm_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(
            d_input, d_gamma, d_beta, d_output, n, c, h, w, num_groups, eps
        );
        CUDA_KERNEL_CHECK();
    }

    void groupnorm_backward(
        float* d_grad_output,
        float* d_input,
        float* d_gamma,
        float* d_grad_input,
        float* d_grad_gamma,
        float* d_grad_beta,
        int n,
        int c,
        int h,
        int w,
        int num_groups,
        float eps
    ) {
        int total = n * c * h * w;
        int tpb = 256;
        cudaMemset(d_grad_gamma, 0, c * sizeof(float));
        CUDA_KERNEL_CHECK();
        cudaMemset(d_grad_beta, 0, c * sizeof(float));
        CUDA_KERNEL_CHECK();
        groupnorm_backward_kernel<<<(total + tpb - 1) / tpb, tpb>>>(
            d_grad_output, d_input, d_gamma, d_grad_input, d_grad_gamma, d_grad_beta,
            n, c, h, w, num_groups, eps
        );
        CUDA_KERNEL_CHECK();
    }

    // Layout conversion functions defined in layout_convert.cu
    void nchw_to_cnhw(float* d_input, float* d_output, int N, int C, int H, int W);
    void cnhw_to_nchw(float* d_input, float* d_output, int N, int C, int H, int W);

    // Forward declarations for functions implemented in other .cu files
    void reorganize_forward(float* d_input, float* d_output, int N, int C, int H, int W);
    void reorganize_backward(const float* d_grad_output, float* d_grad_input, int N, int C, int H, int W);
    void layer_norm_forward(float* d_output, const float* d_input, const float* d_gamma, const float* d_beta, int N, int C, int H, int W, float eps);
    void layer_norm_backward(float* d_grad_input, const float* d_grad_output, const float* d_input, const float* d_gamma, int N, int C, int H, int W, float eps);
    void bn_train_forward(float* d_y, const float* d_x, float* d_x_hat, float* d_batch_mean, float* d_batch_inv_std, float* d_running_mean, float* d_running_var, const float* d_gamma, const float* d_beta, int N, int C, int H, int W, float eps, float momentum);
    void bn_eval_forward(float* d_y, const float* d_x, const float* d_running_mean, const float* d_running_var, const float* d_gamma, const float* d_beta, int N, int C, int H, int W, float eps);
    void bn_backward(float* d_dx, float* d_dgamma, float* d_dbeta, const float* d_dy, const float* d_x_hat, const float* d_gamma, const float* d_inv_std, int N, int C, int H, int W);
    void maxpool_forward_store(float* d_output, const float* d_input, int* d_max_idx, int N, int C, int H, int W);
    void maxpool_backward_use_idx(const float* d_grad_out, const int* d_max_idx, float* d_grad_input, int N, int C, int H, int W);
}
