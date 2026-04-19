#pragma once

#ifndef USE_CUBLAS
#define USE_CUBLAS 1
#endif

#if USE_CUBLAS
#include <cublas_v2.h>

cublasHandle_t minicnn_get_cublas_handle();
#endif
