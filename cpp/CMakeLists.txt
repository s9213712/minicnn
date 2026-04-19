cmake_minimum_required(VERSION 3.20)
project(minimal_cuda_cnn LANGUAGES CXX CUDA)

option(USE_CUBLAS "Use cuBLAS implementation where available" ON)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

file(GLOB MINICNN_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_library(minimal_cuda_cnn SHARED ${MINICNN_SOURCES})
target_include_directories(minimal_cuda_cnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(minimal_cuda_cnn PUBLIC CUDA::cudart)

if(USE_CUBLAS)
  target_compile_definitions(minimal_cuda_cnn PRIVATE USE_CUBLAS=1)
  target_link_libraries(minimal_cuda_cnn PUBLIC CUDA::cublas)
else()
  target_compile_definitions(minimal_cuda_cnn PRIVATE USE_CUBLAS=0)
endif()

set_target_properties(minimal_cuda_cnn PROPERTIES
  OUTPUT_NAME minimal_cuda_cnn
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
)
