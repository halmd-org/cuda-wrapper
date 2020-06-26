/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE stream
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <functional>
#include <random>
#include <cmath>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

static const size_t BLOCKS = 4096;
static const size_t THREADS = 128;

// from function_kernel.cu test
extern cuda::function<void (double const *, double const *, double *)> kernel_add;
extern cuda::function<void (double const *, double *)> kernel_sqrt;

/*
 * These tests uses the kernel_add and kernel_sqrt from the function test.
 * Both vectors with input numbers and the results are copied asynchronously.
 * In addition the kernel_add and kernel_sqrt are both launched asynchronously.
 */

BOOST_AUTO_TEST_CASE(normal) {
    // create two streams (both with default flags)
    cuda::stream s1, s2(CU_STREAM_DEFAULT);

    // streams should both be empty
    BOOST_CHECK(s1.query() == true);
    BOOST_CHECK(s2.query() == true);

    cuda::config dim(BLOCKS, THREADS);

    cuda::host::vector<double> h_a(BLOCKS * THREADS);
    cuda::host::vector<double> h_b(h_a.size());
    cuda::host::vector<double> h_c(h_a.size());
    cuda::host::vector<double> h_d(h_a.size());

    cuda::device::vector<double> d_a(h_a.size());
    cuda::device::vector<double> d_b(h_a.size());
    cuda::device::vector<double> d_c(h_a.size());
    cuda::device::vector<double> d_d(h_a.size());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate random numbers
    std::generate(h_a.begin(), h_a.end(), std::bind(rand, std::ref(gen)));
    std::generate(h_b.begin(), h_b.end(), std::bind(rand, std::ref(gen)));

    // copy data from host to device
    // both vectors are copied asynchronously in two different streams
    // copy the first vector in stream s1
    cuda::copy(h_a.begin(), h_a.end(), d_a.begin(), s1);

    // stream s1 should now be busy
    BOOST_CHECK(s1.query() == false);

    // copy the second vector in stream s2
    cuda::copy(h_b.begin(), h_b.end(), d_b.begin(), s2);

    // both streams should be busy
    BOOST_CHECK(s1.query() == false);
    BOOST_CHECK(s2.query() == false);

    // wait for both copies to be complete
    s1.synchronize();
    s2.synchronize();

    // both streams should be empty
    BOOST_CHECK(s1.query() == true);
    BOOST_CHECK(s2.query() == true);

    // configure kernels (with two different streams)
    kernel_add.configure(dim.grid, dim.block, s1);
    kernel_sqrt.configure(dim.grid, dim.block, s2);

    // launch kernel (in stream s1)
    kernel_add(d_a, d_b, d_c);

    // stream s1 should now be busy
    BOOST_CHECK(s1.query() == false);

    // launch kernel (in stream s2)
    kernel_sqrt(d_a, d_d);

    // both streams should now be busy
    BOOST_CHECK(s1.query() == false);
    BOOST_CHECK(s2.query() == false);

    // calculate the results on the host
    std::vector<double> result_add(h_a.size());
    std::transform(h_a.begin(), h_a.end(), h_b.begin(), result_add.begin(), std::plus<double>());

    std::vector<double> result_sqrt(h_a.size());
    std::transform(h_a.begin(), h_a.end(), result_sqrt.begin(), [](const double &a) -> double { return std::sqrt(a); });

    // wait for kernels to finish (if they haven't already)
    s1.synchronize();
    s2.synchronize();

    // both streams should be empty
    BOOST_CHECK(s1.query() == true);
    BOOST_CHECK(s2.query() == true);

    // copy back result from device to host in stream s1
    cuda::copy(d_c.begin(), d_c.end(), h_c.begin(), s1);

    // stream s1 should now be busy
    BOOST_CHECK(s1.query() == false);

    // copy back result from device to host in stream s2
    cuda::copy(d_d.begin(), d_d.end(), h_d.begin(), s2);

    // both streams should now be busy
    BOOST_CHECK(s1.query() == false);
    BOOST_CHECK(s2.query() == false);

    // wait for asynchonous copy to be finished
    s1.synchronize();
    s2.synchronize();

    // both streams should now be empty
    BOOST_CHECK(s1.query() == true);
    BOOST_CHECK(s2.query() == true);

    // check results
    BOOST_CHECK_EQUAL_COLLECTIONS(result_add.begin(), result_add.end(), h_c.begin(), h_c.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(result_sqrt.begin(), result_sqrt.end(), h_d.begin(), h_d.end());
}

BOOST_AUTO_TEST_CASE(attach)
{
    // create two streams
    cuda::stream s1;
    cuda::stream s2;

    // streams should both be empty
    BOOST_CHECK(s1.query() == true);
    BOOST_CHECK(s2.query() == true);

    cuda::config dim(BLOCKS, THREADS);

    // create vectors with managed memory
    cuda::vector<double> a(BLOCKS * THREADS);
    cuda::vector<double> b(a.size());
    cuda::vector<double> c(a.size());
    cuda::vector<double> d(a.size());

    // attach data to stream s1
    // a is not attached becuase it will be accessed by both streams s1 and s2
    s1.attach(b.data());
    s1.attach(c.data());

    // attach data to stream s2
    s2.attach(d.data());

    // create random number generator
    std::default_random_engine gen;
    std::uniform_real_distribution<double> rand(0, 1);

    // generate random numbers
    std::generate(a.begin(), a.end(), std::bind(rand, std::ref(gen)));
    std::generate(b.begin(), b.end(), std::bind(rand, std::ref(gen)));

    // configure kernels (with two different streams)
    kernel_add.configure(dim.grid, dim.block, s1);
    kernel_sqrt.configure(dim.grid, dim.block, s2);

    // launch kernell (in stream s1)
    kernel_add(a, b, c);

    // stream s1 should now be busy
    BOOST_CHECK(s1.query() == false);

    // launch kernel (in stream s2)
    kernel_sqrt(a, d);

    // both streams should now be busy
    BOOST_CHECK(s1.query() == false);
    BOOST_CHECK(s2.query() == false);

    // calculate the results on the host
    std::vector<double> result_add(a.size());
    std::transform(a.begin(), a.end(), b.begin(), result_add.begin(), std::plus<double>());

    std::vector<double> result_sqrt(a.size());
    std::transform(a.begin(), a.end(), result_sqrt.begin(), [](const double &a) -> double { return std::sqrt(a); });

    // wait for kernels to finish (if they haven't already)
    s1.synchronize();
    s2.synchronize();

    // both streams should be empty
    BOOST_CHECK(s1.query() == true);
    BOOST_CHECK(s2.query() == true);

    // check results
    BOOST_CHECK_EQUAL_COLLECTIONS(result_add.begin(), result_add.end(), c.begin(), c.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(result_sqrt.begin(), result_sqrt.end(), d.begin(), d.end());
}
