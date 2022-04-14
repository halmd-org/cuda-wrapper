#include "kernel.hpp"
#include <stdio.h>
#include <iterator>

__global__ void printid(int* vector)
{
    int x = threadIdx.x + blockIdx.x * blockDim.x;
    vector[x] = x;
}

wrapper wrapper::kernel = {printid};