/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE texture
#include <boost/test/unit_test.hpp>

#include <random>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

static const size_t BLOCKS = 4096;
static const size_t THREADS = 128;

extern cuda::function<void (cudaTextureObject_t, cudaTextureObject_t, float *)> kernel_add;

BOOST_AUTO_TEST_CASE(add) {
    cuda::config dim(BLOCKS, THREADS);

    cuda::vector<float> v_a(BLOCKS * THREADS);
    cuda::vector<float> v_b(v_a.size());
    cuda::vector<float> v_c(v_a.size());

    cuda::texture<float> t_a(v_a);
    cuda::texture<float> t_b(v_b);

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<float> rand(0, 1);

    // generate random numbers
    std::generate(v_a.begin(), v_a.end(), std::bind(rand, std::ref(gen)));
    std::generate(v_b.begin(), v_b.end(), std::bind(rand, std::ref(gen)));

    // configure kernel
    kernel_add.configure(dim.grid, dim.block);
    // launch kernell
    kernel_add(t_a, t_b, v_c);

    // calculate the result on the host
    std::vector<float> result(v_a.size());
    std::transform(v_a.begin(), v_a.end(), v_b.begin(), result.begin(), std::plus<float>());

    // wait for kernel to be finished (if it hasn't already)
    cuda::thread::synchronize();

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), v_c.begin(), v_c.end());
}
