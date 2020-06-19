/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE host_allocator
#include <boost/test/unit_test.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

#define TEST(type, dataset)                                                    \
    cuda::host::allocator<type> A;                                             \
    BOOST_DATA_TEST_CASE(allocate, dataset, n) {                               \
        type *ptr = A.allocate(n);                                             \
        BOOST_TEST(ptr == A.address(*ptr));                                    \
        A.deallocate(ptr, n);                                                  \
    }

struct x
{
    int a;
    float b;
    double c = 0;

    x(int a, float b) : a(a), b(b) {}

    double d() { return a + b + c; }
    int e(int f, float g) { return f * g; }
};

class y
{
    int a;
    float b;
    double c = 0;

    double d() { return a + b + c; }
    int e(int f, float g) { return f * g; }
};

auto dataset = boost::unit_test::data::make<unsigned int>({
    0, 1, 3, 10, 57, 111, 999, 4321, 10000, 31415, 100000
});

BOOST_AUTO_TEST_SUITE(host_allocator)
    BOOST_AUTO_TEST_SUITE(type_int)
        TEST(int, dataset)
    BOOST_AUTO_TEST_SUITE_END()

    BOOST_AUTO_TEST_SUITE(type_unsigned_long_long)
        TEST(unsigned long long, dataset)
    BOOST_AUTO_TEST_SUITE_END()

    BOOST_AUTO_TEST_SUITE(type_float)
        TEST(float, dataset)
    BOOST_AUTO_TEST_SUITE_END()

    BOOST_AUTO_TEST_SUITE(type_double)
        TEST(double, dataset)
    BOOST_AUTO_TEST_SUITE_END()

    BOOST_AUTO_TEST_SUITE(type_struct)
        TEST(x, dataset)
    BOOST_AUTO_TEST_SUITE_END()

    BOOST_AUTO_TEST_SUITE(type_class)
        TEST(y, dataset)
    BOOST_AUTO_TEST_SUITE_END()
BOOST_AUTO_TEST_SUITE_END()
