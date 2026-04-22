cmake_minimum_required(VERSION 3.20)
set(CMAKE_CUDA_ARCHITECTURES 86 CACHE STRING "CUDA architectures to build, for example 86 for RTX 30 series")
project(minimal_cuda_cnn LANGUAGES CXX CUDA)

option(USE_CUBLAS "Use cuBLAS implementation where available" ON)
set(MINICNN_OUTPUT_NAME "minimal_cuda_cnn" CACHE STRING "Shared library output name without lib prefix or .so suffix")
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CUDA_STANDARD 17)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(CUDAToolkit REQUIRED)

file(GLOB MINICNN_SOURCES CONFIGURE_DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/src/*.cu)
add_library(minimal_cuda_cnn SHARED ${MINICNN_SOURCES})
target_include_directories(minimal_cuda_cnn PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/include)
target_link_libraries(minimal_cuda_cnn PUBLIC CUDA::cudart)
target_compile_definitions(minimal_cuda_cnn PRIVATE $<$<CONFIG:Debug>:MINICNN_DEBUG_SYNC>)

if(USE_CUBLAS)
  target_compile_definitions(minimal_cuda_cnn PRIVATE USE_CUBLAS=1)
  target_link_libraries(minimal_cuda_cnn PUBLIC CUDA::cublas)
else()
  target_compile_definitions(minimal_cuda_cnn PRIVATE USE_CUBLAS=0)
endif()

set_target_properties(minimal_cuda_cnn PROPERTIES
  OUTPUT_NAME ${MINICNN_OUTPUT_NAME}
  LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  ARCHIVE_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
  CUDA_SEPARABLE_COMPILATION ON
)
