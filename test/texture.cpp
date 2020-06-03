/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE texture
#include <boost/test/included/unit_test.hpp>

#include <random>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

static const size_t BLOCKS = 4096;
static const size_t THREADS = 128;

extern cuda::function<void (cudaTextureObject_t, cudaTextureObject_t, float *)> kernel_add;

BOOST_AUTO_TEST_CASE(add) {
    cuda::config dim(BLOCKS, THREADS);

    cuda::host::vector<float> h_a(BLOCKS * THREADS);
    cuda::host::vector<float> h_b(h_a.size());
    cuda::host::vector<float> h_c(h_a.size());

    cuda::vector<float> d_a(h_a.size());
    cuda::vector<float> d_b(h_a.size());
    cuda::vector<float> d_c(h_a.size());

    cuda::texture<float> t_a(d_a);
    cuda::texture<float> t_b(d_b);

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<float> rand(0, 1);

    // generate random numbers and copy to device
    std::generate(h_a.begin(), h_a.end(), std::bind(rand, std::ref(gen)));
    cuda::copy(h_a.begin(), h_a.end(), d_a.begin());

    std::generate(h_b.begin(), h_b.end(), std::bind(rand, std::ref(gen)));
    cuda::copy(h_b.begin(), h_b.end(), d_b.begin());

    // configure kernel
    kernel_add.configure(dim.grid, dim.block);
    // launch kernell
    kernel_add(t_a, t_b, d_c);

    // calculate the result on the host
    cuda::host::vector<float> result(h_a.size());
    std::transform(h_a.begin(), h_a.end(), h_b.begin(), result.begin(),
        std::plus<float>());

    // wait for kernel to be finished (if it hasn't already)
    cuda::thread::synchronize();
    // copy back result from device to host
    cuda::copy(d_c.begin(), d_c.end(), h_c.begin());

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), h_c.begin(), h_c.end());
}
