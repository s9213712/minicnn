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

    void layernorm2d_forward(float* d_input, float* d_gamma, float* d_beta, float* d_output, int n, int c, int h, int w, float eps) {
        int size = n * h * w;
        int tpb = 256;
        layernorm2d_forward_kernel<<<(size + tpb - 1) / tpb, tpb>>>(
            d_input, d_gamma, d_beta, d_output, n, c, h, w, eps
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
