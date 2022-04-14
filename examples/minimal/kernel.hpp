/* examples/minimal/kernel.hpp
 *
 * Copyright (C) 2022 Viktor Skoblin
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#include <cuda_wrapper/cuda_wrapper.hpp>

struct wrapper
{
    cuda::function<void (unsigned int*)> get_id;
    static wrapper kernel;
};
