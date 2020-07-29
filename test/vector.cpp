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

#define TEST(vector_type)                                                           \
    BOOST_AUTO_TEST_SUITE(type_int)                                                 \
    TEST_CASE(vector_type<int>, dataset)                                            \
    BOOST_AUTO_TEST_SUITE_END()                                                     \
                                                                                    \
    BOOST_AUTO_TEST_SUITE(type_unsigned_long_long)                                  \
    TEST_CASE(vector_type<unsigned long long>, dataset)                             \
    BOOST_AUTO_TEST_SUITE_END()                                                     \
                                                                                    \
    BOOST_AUTO_TEST_SUITE(type_float)                                               \
    TEST_CASE(vector_type<float>, dataset)                                          \
    BOOST_AUTO_TEST_SUITE_END()                                                     \
                                                                                    \
    BOOST_AUTO_TEST_SUITE(type_double)                                              \
    TEST_CASE(vector_type<double>, dataset)                                         \
    BOOST_AUTO_TEST_SUITE_END()                                                     \
                                                                                    \
    BOOST_AUTO_TEST_SUITE(type_struct)                                              \
    TEST_CASE(vector_type<x>, dataset)                                              \
    BOOST_AUTO_TEST_SUITE_END()                                                     \
                                                                                    \
    BOOST_AUTO_TEST_SUITE(type_class)                                               \
    TEST_CASE(vector_type<y>, dataset)                                              \
    BOOST_AUTO_TEST_SUITE_END()                                                     \
                                                                                    \
    BOOST_AUTO_TEST_CASE(constructor_assignemnt) {                                  \
        vector_type<int> v0{1, 2, 3};                                               \
        vector_type<int> v1(v0);                                                    \
        v0 = {4, 5, 6};                                                             \
        v1 = v0;                                                                    \
        vector_type<int> v2(v0.begin(), v0.end());                                  \
                                                                                    \
        cuda::memory::host::vector<int> h0(3);                                      \
        cuda::memory::host::vector<int> h1(3);                                      \
        cuda::memory::host::vector<int> h2(3);                                      \
        cuda::copy(v0.begin(), v0.end(), h0.begin());                               \
        cuda::copy(v1.begin(), v1.end(), h1.begin());                               \
        cuda::copy(v2.begin(), v2.end(), h2.begin());                               \
        BOOST_CHECK_EQUAL_COLLECTIONS(h0.begin(), h0.end(), h1.begin(), h1.end());  \
        BOOST_CHECK_EQUAL_COLLECTIONS(h0.begin(), h0.end(), h2.begin(), h2.end());  \
    }


BOOST_AUTO_TEST_SUITE(managed)
    TEST(cuda::memory::managed::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(device)
    TEST(cuda::memory::device::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(host)
    TEST(cuda::memory::host::vector)
BOOST_AUTO_TEST_SUITE_END()
