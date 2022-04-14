cmake_minimum_required(VERSION 3.8)

project(cuda-wrapper-examples CXX CUDA)

find_library(CUDA_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDART_LIBRARY cudart HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

include_directories(SYSTEM ${CMAKE_SOURCE_DIR})  # path to cuda_wrapper
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CXX_FLAGS "-Wall -pedantic")

add_executable(first first/main.cpp first/kernel.cu)
target_link_libraries(first ${CUDA_LIBRARY} ${CUDART_LIBRARY})
