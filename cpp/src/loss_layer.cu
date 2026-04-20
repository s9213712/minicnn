#include <cuda_runtime.h>
#include <cmath>
#include "cuda_check.h"

// ============== Softmax CrossEntropy ==============
// One warp (32 threads) per sample.  Requires features <= 1024 (= 32 * 32).
// For CIFAR-10 (10 classes) this is always satisfied.
__global__ void softmax_kernel(const float* input, float* output, int N, int features) {
    int n   = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N || tid >= 32) return;

    const float* row     = input  + n * features;
    float*       out_row = output + n * features;

    // Warp-parallel max
    float thread_max = -1e38f;
    for (int j = tid; j < features; j += 32)
        thread_max = fmaxf(thread_max, row[j]);
    for (int offset = 16; offset > 0; offset >>= 1)
        thread_max = fmaxf(thread_max, __shfl_down_sync(0xffffffff, thread_max, offset));
    float max_val = __shfl_sync(0xffffffff, thread_max, 0);

    // Warp-parallel exp + sum
    float thread_sum = 0.0f;
    for (int j = tid; j < features; j += 32) {
        float e = expf(row[j] - max_val);
        out_row[j] = e;
        thread_sum += e;
    }
    float sum = __shfl_sync(0xffffffff, warp_reduce_sum(thread_sum), 0);

    // Normalize
    for (int j = tid; j < features; j += 32)
        out_row[j] /= sum;
}

__global__ void cross_entropy_backward_kernel(float* grad_out, float* probs, int N, int features) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * features;
    if (idx >= total) return;

    // dL/dsoftmax = softmax - label (one-hot)
    // grad_out contains label (one-hot encoded)
    grad_out[idx] = (probs[idx] - grad_out[idx]);  // label[j=target] = 1, else 0
}

__device__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1) {
        val += __shfl_down_sync(0xffffffff, val, offset);
    }
    return val;
}

__device__ void warp_reduce_max_with_index(float& val, int& index) {
    // Tree reduction: each lane already holds the min-index max within its own elements.
    // On tie, keep the smaller index so the global minimum-index argmax is preserved.
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_val   = __shfl_down_sync(0xffffffff, val,   offset);
        int   other_index = __shfl_down_sync(0xffffffff, index, offset);
        if (other_val > val || (other_val == val && other_index < index)) {
            val   = other_val;
            index = other_index;
        }
    }
}

__global__ void softmax_xent_grad_loss_acc_kernel(
    const float* logits,
    const int* labels,
    float* probs,
    float* grad_logits,
    float* loss_sum,
    int* correct_count,
    int N,
    int features
) {
    int n = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N || tid >= 32) return;

    const float* row = logits + n * features;
    float* prob_row = probs + n * features;
    float* grad_row = grad_logits + n * features;
    int label = labels[n];

    float thread_max = -1e38f;
    int thread_pred = features;  // sentinel: no valid prediction yet
    for (int j = tid; j < features; j += 32) {
        float v = row[j];
        // Keep first (smallest-index) occurrence among equal maxima.
        if (v > thread_max || (v == thread_max && j < thread_pred)) {
            thread_max = v;
            thread_pred = j;
        }
    }
    warp_reduce_max_with_index(thread_max, thread_pred);
    float max_val = __shfl_sync(0xffffffff, thread_max, 0);
    int pred = __shfl_sync(0xffffffff, thread_pred, 0);

    float thread_sum = 0.0f;
    for (int j = tid; j < features; j += 32) {
        float p = expf(row[j] - max_val);
        prob_row[j] = p;
        thread_sum += p;
    }
    float sum = __shfl_sync(0xffffffff, warp_reduce_sum(thread_sum), 0);

    for (int j = tid; j < features; j += 32) {
        float p = prob_row[j] / sum;
        prob_row[j] = p;
        float target = (j == label) ? 1.0f : 0.0f;
        grad_row[j] = (p - target) / static_cast<float>(N);
    }
    __syncwarp();

    if (tid == 0) {
        // Accumulate batch loss as a sum; Python divides by N when reporting mean loss.
        atomicAdd(loss_sum, -logf(prob_row[label] + 1e-10f));
        if (pred == label) {
            atomicAdd(correct_count, 1);
        }
    }
}

__global__ void count_correct_kernel(
    const float* logits,
    const int* labels,
    int* correct_count,
    int N,
    int features
) {
    int n = blockIdx.x;
    if (n >= N || threadIdx.x != 0) return;

    const float* row = logits + n * features;
    int pred = 0;
    float best = row[0];
    for (int j = 1; j < features; ++j) {
        float v = row[j];
        if (v > best) {
            best = v;
            pred = j;
        }
    }
    if (pred == labels[n]) {
        atomicAdd(correct_count, 1);
    }
}

extern "C" {
    void softmax_forward(float* d_input, float* d_output, int N, int features) {
        softmax_kernel<<<N, 32>>>(d_input, d_output, N, features);
        CUDA_KERNEL_CHECK();
    }

    void softmax_backward(float* d_grad_out, float* d_probs, int N, int features) {
        dim3 block(256);
        dim3 grid((N * features + 255) / 256);
        cross_entropy_backward_kernel<<<grid, block>>>(d_grad_out, d_probs, N, features);
        CUDA_KERNEL_CHECK();
    }

    void softmax_xent_grad_loss_acc(
        float* d_logits,
        int* d_labels,
        float* d_probs,
        float* d_grad_logits,
        float* d_loss_sum,
        int* d_correct_count,
        int N,
        int features
    ) {
        softmax_xent_grad_loss_acc_kernel<<<N, 32>>>(
            d_logits, d_labels, d_probs, d_grad_logits, d_loss_sum, d_correct_count, N, features
        );
        CUDA_KERNEL_CHECK();
    }

    void count_correct(float* d_logits, int* d_labels, int* d_correct_count, int N, int features) {
        count_correct_kernel<<<N, 1>>>(d_logits, d_labels, d_correct_count, N, features);
        CUDA_KERNEL_CHECK();
    }
}

// ============== MSE Loss ==============
// Labels are integer class indices; the kernel converts them to one-hot targets internally.
// Gradient: dL/dlogit_j = 2 * (logit_j - target_j) / N
__global__ void mse_fwd_grad_loss_acc_kernel(
    const float* logits,
    const int*   labels,
    float* grad_logits,
    float* loss_sum,
    int*   correct_count,
    int N,
    int features
) {
    int n   = blockIdx.x;
    int tid = threadIdx.x;
    if (n >= N) return;

    const float* row  = logits      + n * features;
    float*       grow = grad_logits + n * features;
    int label = labels[n];

    // Each warp lane covers a strided subset of features.
    float thread_loss = 0.0f;
    int thread_pred   = 0;
    float thread_best = row[0];

    for (int j = tid; j < features; j += 32) {
        float target = (j == label) ? 1.0f : 0.0f;
        float diff   = row[j] - target;
        thread_loss += diff * diff;
        grow[j]      = 2.0f * diff / static_cast<float>(N);
        if (row[j] > thread_best || (j == 0)) {
            thread_best = row[j];
            thread_pred = j;
        }
    }

    float total_loss = __shfl_sync(0xffffffff, warp_reduce_sum(thread_loss), 0);

    // Warp argmax for accuracy (same logic as CE kernel)
    warp_reduce_max_with_index(thread_best, thread_pred);
    int pred = __shfl_sync(0xffffffff, thread_pred, 0);

    if (tid == 0) {
        atomicAdd(loss_sum, total_loss / static_cast<float>(features));
        if (pred == label) atomicAdd(correct_count, 1);
    }
}

// ============== BCEWithLogits Loss ==============
// Single-output binary classification: features must equal 1.
// Numerically stable: L = max(x,0) - x*y + log(1 + exp(-|x|))
// Gradient: sigmoid(x) - y
// Accuracy: predict 1 if logit >= 0, else 0.
__global__ void bce_fwd_grad_loss_acc_kernel(
    const float* logits,
    const int*   labels,
    float* grad_logits,
    float* loss_sum,
    int*   correct_count,
    int N
) {
    int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= N) return;

    float x = logits[n];
    float y = static_cast<float>(labels[n]);  // 0 or 1

    // Numerically stable BCE
    float loss = fmaxf(x, 0.0f) - x * y + logf(1.0f + expf(-fabsf(x)));
    // sigmoid(x)
    float sig  = 1.0f / (1.0f + expf(-x));
    grad_logits[n] = (sig - y) / static_cast<float>(N);

    atomicAdd(loss_sum, loss);
    int pred = (x >= 0.0f) ? 1 : 0;
    if (pred == static_cast<int>(y)) atomicAdd(correct_count, 1);
}

extern "C" {
    void mse_fwd_grad_loss_acc(
        float* d_logits,
        int*   d_labels,
        float* d_grad_logits,
        float* d_loss_sum,
        int*   d_correct_count,
        int N,
        int features
    ) {
        mse_fwd_grad_loss_acc_kernel<<<N, 32>>>(
            d_logits, d_labels, d_grad_logits, d_loss_sum, d_correct_count, N, features
        );
        CUDA_KERNEL_CHECK();
    }

    void bce_fwd_grad_loss_acc(
        float* d_logits,
        int*   d_labels,
        float* d_grad_logits,
        float* d_loss_sum,
        int*   d_correct_count,
        int N
    ) {
        int tpb = 256;
        int bpg = (N + tpb - 1) / tpb;
        bce_fwd_grad_loss_acc_kernel<<<bpg, tpb>>>(
            d_logits, d_labels, d_grad_logits, d_loss_sum, d_correct_count, N
        );
        CUDA_KERNEL_CHECK();
    }
}

// ============== Dense/FC Layer ==============
extern "C" {
    void dense_forward(float* d_input, float* d_weights, float* d_bias, float* d_output, int N, int in_f, int out_f);
}

// ============== GEMM Backward (for FC gradient to weights) ==============
__global__ void gemm_backward_A_kernel(const float* grad_out, const float* B, float* grad_A, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < M && col < K) {
        float sum = 0.0f;
        for (int i = 0; i < N; i++) {
            sum += grad_out[row * N + i] * B[i * K + col];
        }
        grad_A[row * K + col] = sum;
    }
}

__global__ void gemm_backward_B_kernel(const float* A, const float* grad_out, float* grad_B, int M, int N, int K) {
    int row = blockIdx.y * blockDim.y + threadIdx.y;
    int col = blockIdx.x * blockDim.x + threadIdx.x;

    if (row < K && col < N) {
        float sum = 0.0f;
        for (int i = 0; i < M; i++) {
            sum += A[i * K + row] * grad_out[i * N + col];
        }
        grad_B[row * N + col] = sum;
    }
}

extern "C" {
    void gemm_backward(float* d_grad_out, float* d_A, float* d_B, float* d_grad_A, float* d_grad_B, int M, int N, int K) {
        // grad_A = grad_out @ B^T  (M x K)
        dim3 block(16, 16);
        dim3 grid((K + 15) / 16, (M + 15) / 16);
        gemm_backward_A_kernel<<<grid, block>>>(d_grad_out, d_B, d_grad_A, M, N, K);

        // grad_B = A^T @ grad_out  (K x N)
        grid.x = (N + 15) / 16;
        grid.y = (K + 15) / 16;
        gemm_backward_B_kernel<<<grid, block>>>(d_A, d_grad_out, d_grad_B, M, N, K);

        CUDA_KERNEL_CHECK();
    }
}
