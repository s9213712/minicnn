#include "cublas_context.h"

#if USE_CUBLAS
#include <cstdio>
#include <cstdlib>

static cublasHandle_t g_minicnn_cublas = nullptr;

static void cublas_context_check(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d: %s failed with status %d\n",
                     file, line, expr, static_cast<int>(status));
        std::fflush(stderr);
        std::abort();
    }
}

#define CUBLAS_CONTEXT_CHECK(expr) cublas_context_check((expr), #expr, __FILE__, __LINE__)

cublasHandle_t minicnn_get_cublas_handle() {
    if (!g_minicnn_cublas) {
        CUBLAS_CONTEXT_CHECK(cublasCreate(&g_minicnn_cublas));
    }
    return g_minicnn_cublas;
}
#endif
