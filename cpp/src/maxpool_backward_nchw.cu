#include "cuda_check.h"
#include <cassert>

// MaxPool Backward for (N, C, H, W) layout with 2x2 stride
__global__ void maxpool_backward_nchw_kernel(const float* grad_out, const float* input, float* grad_input,
                                              int N, int C, int in_h, int in_w, int out_h, int out_w) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = N * C * out_h * out_w;
    if (idx >= total) return;

    int ow = idx % out_w;
    int oh = (idx / out_w) % out_h;
    int c = (idx / (out_w * out_h)) % C;
    int n = idx / (out_w * out_h * C);

    int h_start = oh * 2;
    int w_start = ow * 2;

    float max_val = -1e38f;
    int max_h = h_start, max_w = w_start;

    // Find max location in 2x2 window
    for (int i = 0; i < 2; i++) {
        for (int j = 0; j < 2; j++) {
            int in_idx = ((n * C + c) * in_h + (h_start + i)) * in_w + (w_start + j);
            if (input[in_idx] > max_val) {
                max_val = input[in_idx];
                max_h = h_start + i;
                max_w = w_start + j;
            }
        }
    }

    // Propagate gradient to max location
    int grad_in_idx = ((n * C + c) * in_h + max_h) * in_w + max_w;
    atomicAdd(&grad_input[grad_in_idx], grad_out[idx]);
}

extern "C" void maxpool_backward_nchw(float* d_grad_out, float* d_input, float* d_grad_input,
                                      int N, int C, int in_h, int in_w, int out_h, int out_w) {
    assert(in_h == out_h * 2 && in_w == out_w * 2);
    int total = N * C * out_h * out_w;
    int tpb = 256;
    CUDA_CHECK(cudaMemset(d_grad_input, 0, N * C * in_h * in_w * sizeof(float)));
    maxpool_backward_nchw_kernel<<<(total + tpb - 1) / tpb, tpb>>>(
        d_grad_out, d_input, d_grad_input, N, C, in_h, in_w, out_h, out_w
    );
    CUDA_KERNEL_CHECK();
}
