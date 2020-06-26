/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE vector
#include <boost/test/unit_test.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

#define TEST_CASE(type, dataset)                                               \
    BOOST_DATA_TEST_CASE(allocate, dataset, n) {                               \
        type v(n);                                                             \
        BOOST_TEST(v.data() == &*v.begin());                                   \
        BOOST_TEST(v.size() == n);                                             \
        BOOST_TEST(v.data() + v.size() == &*v.end());                          \
    }

struct x
{
    int a;
    float b;
    double c = 0;

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

#define TEST(vector)                                                           \
    BOOST_AUTO_TEST_SUITE(type_int)                                            \
        TEST_CASE(vector<int>, dataset)                                        \
    BOOST_AUTO_TEST_SUITE_END()                                                \
                                                                               \
    BOOST_AUTO_TEST_SUITE(type_unsigned_long_long)                             \
        TEST_CASE(vector<unsigned long long>, dataset)                         \
    BOOST_AUTO_TEST_SUITE_END()                                                \
                                                                               \
    BOOST_AUTO_TEST_SUITE(type_float)                                          \
        TEST_CASE(vector<float>, dataset)                                      \
    BOOST_AUTO_TEST_SUITE_END()                                                \
                                                                               \
    BOOST_AUTO_TEST_SUITE(type_double)                                         \
        TEST_CASE(vector<double>, dataset)                                     \
    BOOST_AUTO_TEST_SUITE_END()                                                \
                                                                               \
    BOOST_AUTO_TEST_SUITE(type_struct)                                         \
        TEST_CASE(vector<x>, dataset)                                          \
    BOOST_AUTO_TEST_SUITE_END()                                                \
                                                                               \
    BOOST_AUTO_TEST_SUITE(type_class)                                          \
        TEST_CASE(vector<y>, dataset)                                          \
    BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(managed)
    TEST(cuda::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(device)
    TEST(cuda::device::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(host)
    TEST(cuda::host::vector)
BOOST_AUTO_TEST_SUITE_END()
