#pragma once

#include <cublas_v2.h>
#include <cstdio>
#include <cstdlib>

inline void cublas_check(cublasStatus_t status, const char* expr, const char* file, int line) {
    if (status != CUBLAS_STATUS_SUCCESS) {
        std::fprintf(stderr, "cuBLAS error at %s:%d: %s failed with status %d\n",
                     file, line, expr, static_cast<int>(status));
        std::fflush(stderr);
        std::abort();
    }
}

#define CUBLAS_CHECK(expr) cublas_check((expr), #expr, __FILE__, __LINE__)
