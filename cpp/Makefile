CUDA_HOME ?= /usr/local/cuda
NVCC ?= $(CUDA_HOME)/bin/nvcc
USE_CUBLAS ?= 1
OUTPUT ?= libminimal_cuda_cnn.so
CFLAGS = -O3 -Xcompiler -fPIC -arch=sm_86 -DUSE_CUBLAS=$(USE_CUBLAS)
LDFLAGS = -shared -o $(OUTPUT) -Xlinker -rpath,$(CUDA_HOME)/lib64
EXTRA_LDFLAGS ?=

ifeq ($(USE_CUBLAS),1)
CUBLAS_LDFLAGS = -lcublas
else
CUBLAS_LDFLAGS =
endif

SRCS = \
	src/core.cu \
	src/gpu_monitor.cu \
	src/network.cu \
	src/dense_layer.cu \
	src/backward.cu \
	src/memory.cu \
	src/loss_layer.cu \
	src/conv_backward.cu \
	src/optimizer.cu \
	src/reorganize.cu \
	src/reorganize_backward.cu \
	src/maxpool_backward_nchw.cu \
	src/leaky_relu.cu \
	src/layer_norm.cu \
	src/maxpool_store.cu \
	src/maxpool_backward_use_idx.cu \
	src/layout_convert.cu
INCS = -Iinclude

.PHONY: all cublas handmade variants clean check check-cublas check-handmade check-variants

all:
	$(NVCC) $(CFLAGS) $(INCS) $(SRCS) $(LDFLAGS) $(CUBLAS_LDFLAGS) $(EXTRA_LDFLAGS)

cublas:
	$(MAKE) USE_CUBLAS=1 OUTPUT=libminimal_cuda_cnn_cublas.so all

handmade:
	$(MAKE) USE_CUBLAS=0 OUTPUT=libminimal_cuda_cnn_handmade.so all

variants: cublas handmade

check: all
	@test -f $(OUTPUT)
	@nm -D $(OUTPUT) | grep -q ' gpu_malloc'
	@nm -D $(OUTPUT) | grep -q ' gemm_forward'
	@nm -D $(OUTPUT) | grep -q ' conv_backward_precol'
	@nm -D $(OUTPUT) | grep -q ' softmax_xent_grad_loss_acc'
	@echo "$(OUTPUT) build check passed (USE_CUBLAS=$(USE_CUBLAS))"

check-cublas:
	$(MAKE) USE_CUBLAS=1 OUTPUT=libminimal_cuda_cnn_cublas.so check

check-handmade:
	$(MAKE) USE_CUBLAS=0 OUTPUT=libminimal_cuda_cnn_handmade.so check

check-variants: check-cublas check-handmade

clean:
	rm -f *.so src/*.o
