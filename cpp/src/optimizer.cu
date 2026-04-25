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

// SGD fused update kernel.
// Matches minicnn.cuda_native.training.sgd_update:
// - no momentum: decoupled weight decay, then SGD step
// - momentum: coupled weight decay in the gradient, then momentum step
// clip_val <= 0 disables per-element clipping. The public gpu_native runtime
// still gates optimizer.grad_clip_global because global-norm clipping needs a
// separate cross-parameter reduction.
__global__ void sgd_update_fused_kernel(
    float* weights,
    const float* grad,
    float* velocity,
    float lr,
    float momentum,
    float weight_decay,
    float clip_val,
    float normalizer,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx] / normalizer;
    if (clip_val > 0.0f) {
        g = fmaxf(-clip_val, fminf(clip_val, g));
    }
    if (momentum > 0.0f) {
        g += weight_decay * weights[idx];
        velocity[idx] = momentum * velocity[idx] - lr * g;
        weights[idx] += velocity[idx];
    } else {
        float next_weight = weights[idx];
        if (weight_decay > 0.0f) {
            next_weight *= (1.0f - lr * weight_decay);
        }
        weights[idx] = next_weight - lr * g;
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

__global__ void grad_l2_sumsq_kernel(const float* grad, float* sumsq, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        float g = grad[idx];
        atomicAdd(sumsq, g * g);
    }
}

__global__ void scale_inplace_kernel(float* values, float scale, int size) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < size) {
        values[idx] *= scale;
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

// RMSprop fused update kernel.
// weight_decay is coupled into the gradient, matching minicnn.cuda_native.training.rmsprop_update.
// clip_val <= 0 disables gradient clipping.
__global__ void rmsprop_update_fused_kernel(
    float* weights,
    const float* grad,
    float* square_avg,
    float* momentum_buffer,
    float lr,
    float alpha,
    float eps,
    float momentum,
    float weight_decay,
    float clip_val,
    float normalizer,
    int size
) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= size) return;

    float g = grad[idx] / normalizer + weight_decay * weights[idx];
    if (clip_val > 0.0f) {
        g = fmaxf(-clip_val, fminf(clip_val, g));
    }

    square_avg[idx] = alpha * square_avg[idx] + (1.0f - alpha) * g * g;
    float step = g / (sqrtf(square_avg[idx]) + eps);
    if (momentum > 0.0f) {
        momentum_buffer[idx] = momentum * momentum_buffer[idx] + step;
        step = momentum_buffer[idx];
    }
    weights[idx] -= lr * step;
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

    void sgd_update_fused(
        float* d_weights,
        const float* d_grad,
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
        sgd_update_fused_kernel<<<bpg, tpb>>>(
            d_weights, d_grad, d_velocity,
            lr, momentum, weight_decay, clip_val, normalizer, size
        );
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

    void grad_l2_sumsq(const float* d_grad, float* d_sumsq, int size) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        grad_l2_sumsq_kernel<<<bpg, tpb>>>(d_grad, d_sumsq, size);
        CUDA_KERNEL_CHECK();
    }

    void scale_inplace(float* d_values, float scale, int size) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        scale_inplace_kernel<<<bpg, tpb>>>(d_values, scale, size);
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

    void rmsprop_update_fused(
        float* d_weights,
        const float* d_grad,
        float* d_square_avg,
        float* d_momentum_buffer,
        float lr,
        float alpha,
        float eps,
        float momentum,
        float weight_decay,
        float clip_val,
        float normalizer,
        int size
    ) {
        int tpb = 256;
        int bpg = (size + tpb - 1) / tpb;
        rmsprop_update_fused_kernel<<<bpg, tpb>>>(
            d_weights, d_grad, d_square_avg, d_momentum_buffer,
            lr, alpha, eps, momentum,
            weight_decay, clip_val, normalizer, size
        );
        CUDA_KERNEL_CHECK();
    }
}
