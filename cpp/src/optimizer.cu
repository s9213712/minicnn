#include "cuda_check.h"

__global__ void sgd_update_kernel(float* weights, const float* grad, float lr, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        weights[idx] -= lr * grad[idx];
    }
}

__global__ void momentum_update_kernel(
    float* weights,
    const float* grad,
    float* velocity,
    float lr,
    float momentum,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        velocity[idx] = momentum * velocity[idx] - lr * grad[idx];
        weights[idx] += velocity[idx];
    }
}

__global__ void conv_update_fused_kernel(
    float* weights,
    float* grad,
    float* velocity,
    float lr,
    float momentum,
    float weight_decay,
    float clip_val,
    float normalizer,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx] / normalizer + weight_decay * weights[idx];
        g = fmaxf(-clip_val, fminf(clip_val, g));
        velocity[idx] = momentum * velocity[idx] - lr * g;
        weights[idx] += velocity[idx];
    }
}

__global__ void clip_inplace_kernel(float* values, float clip_val, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        values[idx] = fmaxf(-clip_val, fminf(clip_val, values[idx]));
    }
}

// Adam/AdamW fused update kernel.
// bias_corr1 = 1 - beta1^t, bias_corr2 = 1 - beta2^t (computed on host each step).
// weight_decay is applied in decoupled AdamW style (after the adaptive step).
// clip_val <= 0 disables gradient clipping.
__global__ void adam_update_fused_kernel(
    float* weights,
    const float* grad,
    float* m,           // first moment (EMA of gradients)
    float* v,           // second moment (EMA of squared gradients)
    float lr,
    float beta1,
    float beta2,
    float eps,
    float weight_decay,
    float clip_val,
    float normalizer,   // spatial/global grad normalizer (same role as in conv_update_fused)
    float bias_corr1,   // 1 - beta1^t
    float bias_corr2,   // 1 - beta2^t
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx] / normalizer;
    if (clip_val > 0.0f) {
        g = fmaxf(-clip_val, fminf(clip_val, g));
    }

    m[idx] = beta1 * m[idx] + (1.0f - beta1) * g;
    v[idx] = beta2 * v[idx] + (1.0f - beta2) * g * g;

    float m_hat = m[idx] / bias_corr1;
    float v_hat = v[idx] / bias_corr2;

    float update = lr * m_hat / (sqrtf(v_hat) + eps);
    // AdamW decoupled weight decay
    weights[idx] -= update + lr * weight_decay * weights[idx];
}

extern "C" {
    void apply_sgd_update(float* d_weights, float* d_grad, float lr, int size) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        sgd_update_kernel<<<bpg, tpb>>>(d_weights, d_grad, lr, size);
        CUDA_KERNEL_CHECK();
    }

    void apply_momentum_update(
        float* d_weights,
        float* d_grad,
        float* d_velocity,
        float lr,
        float momentum,
        int size
    ) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        momentum_update_kernel<<<bpg, tpb>>>(d_weights, d_grad, d_velocity, lr, momentum, size);
        CUDA_KERNEL_CHECK();
    }

    void conv_update_fused(
        float* d_weights,
        float* d_grad,
        float* d_velocity,
        float lr,
        float momentum,
        float weight_decay,
        float clip_val,
        float normalizer,
        int size
    ) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        conv_update_fused_kernel<<<bpg, tpb>>>(
            d_weights, d_grad, d_velocity,
            lr, momentum, weight_decay, clip_val, normalizer, size
        );
        CUDA_KERNEL_CHECK();
    }

    void clip_inplace(float* d_values, float clip_val, int size) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        clip_inplace_kernel<<<bpg, tpb>>>(d_values, clip_val, size);
        CUDA_KERNEL_CHECK();
    }

    void adam_update_fused(
        float* d_weights,
        const float* d_grad,
        float* d_m,
        float* d_v,
        float lr,
        float beta1,
        float beta2,
        float eps,
        float weight_decay,
        float clip_val,
        float normalizer,
        float bias_corr1,
        float bias_corr2,
        int size
    ) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        adam_update_fused_kernel<<<bpg, tpb>>>(
            d_weights, d_grad, d_m, d_v,
            lr, beta1, beta2, eps,
            weight_decay, clip_val, normalizer,
            bias_corr1, bias_corr2, size
        );
        CUDA_KERNEL_CHECK();
    }
}
