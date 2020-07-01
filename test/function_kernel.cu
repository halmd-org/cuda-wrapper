/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#include <cuda_wrapper/cuda_wrapper.hpp>

__global__ void __kernel_simple() {}

__global__ void __kernel_add(double const *d_a, double const *d_b, double *d_c)
{
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;
    d_c[gid] = d_a[gid] + d_b[gid];
}

cuda::function<void ()> kernel_simple(__kernel_simple);
cuda::function<void (double const *, double const *, double*)> kernel_add(__kernel_add);
