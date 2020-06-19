/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE event
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <functional>
#include <random>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

static const size_t BLOCKS = 4096;
static const size_t THREADS = 128;

// from function_kernel.cu test
extern cuda::function<void (double const *, double *)> kernel_sqrt;

BOOST_AUTO_TEST_CASE(timing) {
    // create two events (both with default flags)
    cuda::event t1, t2(CU_EVENT_DEFAULT);

    // both events should be finished
    BOOST_CHECK(t1.query() == true);
    BOOST_CHECK(t2.query() == true);

    cuda::config dim(BLOCKS, THREADS);

    cuda::host::vector<double> h(BLOCKS * THREADS);

    cuda::vector<double> d_a(h.size());
    cuda::vector<double> d_b(h.size());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate random numbers and copy to device
    std::generate(h.begin(), h.end(), std::bind(rand, std::ref(gen)));

    // place first event into the default stream
    t1.record();

    // copy the data (synchronized by default)
    cuda::copy(h.begin(), h.end(), d_a.begin());

    // place the second event into the default stream
    t2.record();
    // wait for the second event to be processed by the GPU
    t2.synchronize();

    // compute the delta
    float delta = t2 - t1;

    BOOST_TEST_MESSAGE("copy time: " << delta << "s");
    BOOST_CHECK(delta > 0);

    BOOST_CHECK(t1.query() == true);
    BOOST_CHECK(t2.query() == true);

    // configure kernel
    kernel_sqrt.configure(dim.grid, dim.block);

    // place first event into the default stream
    t1.record();

    // launch kernel
    kernel_sqrt(d_a, d_b);
    // wait for kernel to finish
    cuda::thread::synchronize();

    // place first event into the default stream
    t2.record();
    // wait for the second event to be processed by the GPU
    t2.synchronize();

    // compute the delta
    delta = t2 - t1;

    BOOST_TEST_MESSAGE("execution time: " << delta << "s");
    BOOST_CHECK(delta > 0);
}
