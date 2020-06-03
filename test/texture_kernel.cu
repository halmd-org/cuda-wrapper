/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#include <cuda_wrapper/cuda_wrapper.hpp>

__global__ void __kernel_add(cudaTextureObject_t a, cudaTextureObject_t b, float *d_c)
{
    unsigned int gid = threadIdx.x + blockIdx.x * blockDim.x;
    d_c[gid] = tex1Dfetch<float>(a, gid) + tex1Dfetch<float>(b, gid);
}

cuda::function<void (cudaTextureObject_t, cudaTextureObject_t, float *)> kernel_add(__kernel_add);