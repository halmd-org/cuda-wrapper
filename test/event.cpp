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

extern cuda::function<void (double const*, double const*, double*)> kernel_add;

BOOST_AUTO_TEST_CASE(timing) {
    using cuda::memory::device::vector;
    namespace host = cuda::memory::host;

    // create two events (both with default flags)
    cuda::event t1, t2(CU_EVENT_DEFAULT);

    // both events should be finished
    BOOST_CHECK(t1.query() == true);
    BOOST_CHECK(t2.query() == true);

    cuda::config dim(BLOCKS, THREADS);

    host::vector<double> h_a(BLOCKS * THREADS);
    host::vector<double> h_b(BLOCKS * THREADS);

    vector<double> d_a(h_a.size());
    vector<double> d_b(h_a.size());
    vector<double> d_c(h_a.size());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate random numbers and copy to device
    std::generate(h_a.begin(), h_a.end(), std::bind(rand, std::ref(gen)));
    std::generate(h_b.begin(), h_b.end(), std::bind(rand, std::ref(gen)));

    // place first event into the default stream
    t1.record();

    // copy the data (synchronized by default)
    cuda::copy(h_a.begin(), h_a.end(), d_a.begin());
    cuda::copy(h_b.begin(), h_b.end(), d_b.begin());

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
    kernel_add.configure(dim.grid, dim.block);

    // place first event into the default stream
    t1.record();

    // launch kernel
    kernel_add(d_a, d_b, d_c);
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
