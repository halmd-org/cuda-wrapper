/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE function
#include <boost/test/unit_test.hpp>

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

    cuda::vector<double> a(BLOCKS * THREADS);
    cuda::vector<double> b(a.size());
    cuda::vector<double> c(a.size());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate random numbers
    std::generate(a.begin(), a.end(), std::bind(rand, std::ref(gen)));
    std::generate(b.begin(), b.end(), std::bind(rand, std::ref(gen)));

    // configure kernel
    kernel_add.configure(dim.grid, dim.block);
    // launch kernell
    kernel_add(a, b, c);

    // calculate the result on the host
    std::vector<double> result(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result.begin(), std::plus<double>());

    // wait for kernel to finish (if it hasn't already)
    cuda::thread::synchronize();

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), c.begin(), c.end());
}

BOOST_AUTO_TEST_CASE(test_sqrt)
{
    cuda::config dim(BLOCKS, THREADS);

    cuda::vector<double> a(BLOCKS * THREADS);
    cuda::vector<double> b(a.size());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate rundom numbers
    std::generate(a.begin(), a.end(), std::bind(rand, std::ref(gen)));

    // configure kernel
    kernel_sqrt.configure(dim.grid, dim.block);
    // launch kernell
    kernel_sqrt(a, b);

    // calculate the result on the host
    std::vector<double> result(a.size());
    std::transform(a.begin(), a.end(), result.begin(), [](const double &a) -> double { return std::sqrt(a); });

    // wait for kernel to finish (if it hasn't already)
    cuda::thread::synchronize();

    BOOST_CHECK_EQUAL_COLLECTIONS(result.begin(), result.end(), b.begin(), b.end());
}
