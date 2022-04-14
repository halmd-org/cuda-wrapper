/* examples/minimal/main.cpp
 *
 * Copyright (C) 2022 Viktor Skoblin
 * Copyright (C) 2022 Felix Höfling
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#include "kernel.hpp"

/**
 * This is a minimal example to demonstate the basic usage of cuda_wrapper.
 */

#include <iostream>

int main()
{
    // create CUDA context on device 0
    cuda::device dev;
    dev.set(0);

    // allocate managed memory
    cuda::config dim(30, 128);
    cuda::memory::managed::vector<unsigned int> array(dim.threads());

    // call a CUDA kernel from a different compilation unit
    wrapper::kernel.get_id.configure(dim.grid, dim.block);
    wrapper::kernel.get_id(array);
    cuda::thread::synchronize();

    // verify output
    unsigned int j = 0;
    for (unsigned int i : array) {
        if (i != j) {
            std::cout << i << " ≠ " << j << std::endl;
            return -1;
        }
        ++j;
    }

    return 0;
}
