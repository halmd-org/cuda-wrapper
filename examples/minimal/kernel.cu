/* examples/minimal/kernel.cu
 *
 * Copyright (C) 2022 Viktor Skoblin
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#include "kernel.hpp"

// the following code is part of the documentation in README.rst
__global__ void get_id(unsigned int* vector)
{
    unsigned int i = static_cast<unsigned int>(threadIdx.x + blockIdx.x * blockDim.x);
    vector[i] = i;
}

wrapper wrapper::kernel = { ::get_id };
// end of usage in README.rst
