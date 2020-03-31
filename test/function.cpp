/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE function
#include <boost/test/included/unit_test.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <cmath>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

static const size_t BLOCKS = 4096;
static const size_t THREADS = 128;

extern cuda::function<void ()> kernel_simple;
extern cuda::function<void (double const *, double const *, double *)> kernel_add;
extern cuda::function<void (double const *, double *)> kernel_sqrt;

BOOST_AUTO_TEST_CASE(simple) {
    cuda::config dim(BLOCKS, THREADS);

    kernel_simple.configure(dim.grid, dim.block);
    kernel_simple();

    BOOST_TEST_MESSAGE("binary version: " << kernel_simple.binary_version());
    BOOST_TEST_MESSAGE("const size bytes: " << kernel_simple.const_size_bytes());
    BOOST_TEST_MESSAGE("local size bytes: " << kernel_simple.local_size_bytes());
    BOOST_TEST_MESSAGE("max threads per block: " << kernel_simple.max_threads_per_block());
    BOOST_TEST_MESSAGE("num regs: " << kernel_simple.num_regs());
    BOOST_TEST_MESSAGE("ptx version: " << kernel_simple.ptx_version());
    BOOST_TEST_MESSAGE("shared size bytes: " << kernel_simple.shared_size_bytes());
    BOOST_TEST_MESSAGE("min_grid_size: " << kernel_simple.min_grid_size());
    BOOST_TEST_MESSAGE("max_block_size: " << kernel_simple.max_block_size());
}

BOOST_AUTO_TEST_CASE(add) {
    cuda::config dim(BLOCKS, THREADS);

    cuda::host::vector<double> h_a(BLOCKS * THREADS);
    cuda::host::vector<double> h_b(h_a.size());
    cuda::host::vector<double> h_c(h_a.size());

    cuda::vector<double> d_a(h_a.size());
    cuda::vector<double> d_b(h_a.size());
    cuda::vector<double> d_c(h_a.size());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate random numbers and copy to device
    std::generate(h_a.begin(), h_a.end(), std::bind(rand, std::ref(gen)));
    cuda::copy(h_a.begin(), h_a.end(), d_a.begin());

    std::generate(h_b.begin(), h_b.end(), std::bind(rand, std::ref(gen)));
    cuda::copy(h_b.begin(), h_b.end(), d_b.begin());

    // configure kernel
    kernel_add.configure(dim.grid, dim.block);
    // launch kernell
    kernel_add(d_a, d_b, d_c);

    // calculate the result on the host
    cuda::host::vector<double> result(h_a.size());
    std::transform(h_a.begin(), h_a.end(), h_b.begin(), result.begin(),
        std::plus<double>());

    // wait for kernel to be finished (if it hasn't already)
    cuda::thread::synchronize();
    // copy back result from device to host
    cuda::copy(d_c.begin(), d_c.end(), h_c.begin());

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), h_c.begin(), h_c.end());
}

BOOST_AUTO_TEST_CASE(test_sqrt)
{
    cuda::config dim(BLOCKS, THREADS);

    cuda::host::vector<double> h_a(BLOCKS * THREADS);
    cuda::host::vector<double> h_b(h_a.size());

    cuda::vector<double> d_a(h_a.size());
    cuda::vector<double> d_b(h_a.size());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate rundom numbers and copy to device
    std::generate(h_a.begin(), h_a.end(), std::bind(rand, std::ref(gen)));
    cuda::copy(h_a.begin(), h_a.end(), d_a.begin());

    // configure kernel
    kernel_sqrt.configure(dim.grid, dim.block);
    // launch kernell
    kernel_sqrt(d_a, d_b);

    // calculate the result on the host
    cuda::host::vector<double> result(h_a.size());
    std::transform(h_a.begin(), h_a.end(), result.begin(), [](const double &a) -> double { return std::sqrt(a); });

    // wait for kernel to be finished (if it hasn't already)
    cuda::thread::synchronize();
    // copy back result from device to host
    cuda::copy(d_b.begin(), d_b.end(), h_b.begin());

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), h_b.begin(), h_b.end());
}
