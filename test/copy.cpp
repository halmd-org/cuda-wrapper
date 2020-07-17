/*
 * Copyright (C) 2020 Jaslo Ziska
 * Copyright (C) 2010, 2012 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE copy
#include <boost/test/unit_test.hpp>
#include <boost/test/data/monomorphic.hpp>
#include <boost/test/data/test_case.hpp>
#include <boost/numeric/ublas/vector.hpp>

#include <limits>
#include <random>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

std::default_random_engine gen;
std::uniform_int_distribution<int> dist(0, std::numeric_limits<int>::max() - 42);

auto dataset = boost::unit_test::data::make<unsigned int>({
    999, 4321, 10000, 31415, 100000
});

// copy from host to host vector (all iteratos that are convertible to std::random_access_iterator)
#define TEST_HOST_HOST(type_a, type_b)                                                                              \
    BOOST_DATA_TEST_CASE(nonconst_iterator, dataset, n) {                                                           \
        type_a<int> a(n);                                                                                           \
        type_b<int> b(n);                                                                                           \
                                                                                                                    \
        std::generate(a.begin(), a.end(), std::bind(dist, std::ref(gen)));                                          \
        BOOST_CHECK(cuda::copy(a.begin(), a.end(), b.begin()) == b.end());                                          \
        BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());                                      \
                                                                                                                    \
        std::transform(a.begin(), a.end(), b.begin(), [](int& a) -> int { return a + 42; });                        \
        BOOST_CHECK(cuda::copy(b.begin(), b.end(), a.begin()) == a.end());                                          \
        BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());                                      \
    }                                                                                                               \
    BOOST_DATA_TEST_CASE(const_iterator, dataset, n) {                                                              \
        type_a<int> a(n);                                                                                           \
        type_b<int> b(n);                                                                                           \
                                                                                                                    \
        std::generate(a.begin(), a.end(), std::bind(dist, std::ref(gen)));                                          \
        type_a<int> const& a_const(a);                                                                              \
        BOOST_CHECK(cuda::copy(a_const.begin(), a_const.end(), b.begin()) == b.end());                              \
        BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());                                      \
                                                                                                                    \
        std::transform(a_const.begin(), a_const.end(), b.begin(), [](int const& a) -> int { return a + 42; });      \
        type_b<int> const& b_const(b);                                                                              \
        BOOST_CHECK(cuda::copy(b_const.begin(), b_const.end(), a.begin()) == a.end());                              \
        BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());                                      \
    }                                                                                                               \
    BOOST_DATA_TEST_CASE(pointer, dataset, n) {                                                                     \
        type_a<int> a(n);                                                                                           \
        type_b<int> b(n);                                                                                           \
                                                                                                                    \
        std::generate(&*a.begin(), &*a.end(), std::bind(dist, std::ref(gen)));                                      \
        BOOST_CHECK(cuda::copy(a.begin(), a.end(), &*b.begin()) == &*b.end());                                      \
        BOOST_CHECK_EQUAL_COLLECTIONS(&*a.begin(), &*a.end(), b.begin(), b.end());                                  \
                                                                                                                    \
        std::transform(a.begin(), a.end(), &*b.begin(), [](int& a) -> int { return a + 42; });                      \
        BOOST_CHECK(cuda::copy(&*b.begin(), &*b.end(), &*a.begin()) == &*a.end());                                  \
        BOOST_CHECK_EQUAL_COLLECTIONS(&*a.begin(), &*a.end(), &*b.begin(), &*b.end());                              \
    }                                                                                                               \
    BOOST_DATA_TEST_CASE(const_pointer, dataset, n) {                                                               \
        type_a<int> a(n);                                                                                           \
        type_b<int> b(n);                                                                                           \
                                                                                                                    \
        std::generate(&*a.begin(), &*a.end(), std::bind(dist, std::ref(gen)));                                      \
        type_a<int> const& a_const(a);                                                                              \
        BOOST_CHECK(cuda::copy(a_const.begin(), a_const.end(), &*b.begin()) == &*b.end());                          \
        BOOST_CHECK_EQUAL_COLLECTIONS(&*a.begin(), &*a.end(), b.begin(), b.end());                                  \
                                                                                                                    \
        std::transform(a_const.begin(), a_const.end(), &*b.begin(), [](int const& a) -> int { return a + 42; });    \
        type_b<int> const& b_const(b);                                                                              \
        BOOST_CHECK(cuda::copy(&*b_const.begin(), &*b_const.end(), &*a.begin()) == &*a.end());                      \
        BOOST_CHECK_EQUAL_COLLECTIONS(&*a.begin(), &*a.end(), &*b.begin(), &*b.end());                              \
    }                                                                                                               \
    BOOST_DATA_TEST_CASE(asynchronous, dataset, n) {                                                                \
        cuda::stream stream;                                                                                        \
        type_a<int> a(n);                                                                                           \
        type_b<int> b(n);                                                                                           \
                                                                                                                    \
        std::generate(a.begin(), a.end(), std::bind(dist, std::ref(gen)));                                          \
        BOOST_CHECK(cuda::copy(a.begin(), a.end(), b.begin(), stream) == b.end());                                  \
        stream.synchronize();                                                                                       \
        BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());                                      \
                                                                                                                    \
        std::transform(a.begin(), a.end(), b.begin(), [](int& a) -> int { return a + 42; });                        \
        BOOST_CHECK(cuda::copy(b.begin(), b.end(), a.begin(), stream) == a.end());                                  \
        stream.synchronize();                                                                                       \
        BOOST_CHECK_EQUAL_COLLECTIONS(a.begin(), a.end(), b.begin(), b.end());                                      \
    }

// copy from host to device vector and back
#define TEST_DEVICE_HOST(type)                                                                      \
    BOOST_DATA_TEST_CASE(nonconst_iterator, dataset, n) {                                           \
        cuda::memory::device::vector<int> d(n);                                                     \
        type<int> h_a(n);                                                                           \
        type<int> h_b(n);                                                                           \
                                                                                                    \
        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));                      \
        BOOST_CHECK(cuda::copy(h_a.begin(), h_a.end(), d.begin()) == d.end());                      \
        BOOST_CHECK(cuda::copy(d.begin(), d.end(), h_b.begin()) == h_b.end());                      \
        BOOST_CHECK_EQUAL_COLLECTIONS(h_a.begin(), h_a.end(), h_b.begin(), h_b.end());              \
    }                                                                                               \
    BOOST_DATA_TEST_CASE(const_iterator, dataset, n) {                                              \
        cuda::memory::device::vector<int> d(n);                                                     \
        type<int> h_a(n);                                                                           \
        type<int> h_b(n);                                                                           \
                                                                                                    \
        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));                      \
        type<int> const& h_a_const(h_a);                                                            \
        BOOST_CHECK(cuda::copy(h_a_const.begin(), h_a_const.end(), d.begin()) == d.end());          \
                                                                                                    \
        cuda::memory::device::vector<int> const& d_const(d);                                        \
        BOOST_CHECK(cuda::copy(d_const.begin(), d_const.end(), h_b.begin()) == h_b.end());          \
        BOOST_CHECK_EQUAL_COLLECTIONS(h_a.begin(), h_a.end(), h_b.begin(), h_b.end());              \
    }                                                                                               \
    BOOST_DATA_TEST_CASE(pointer, dataset, n) {                                                     \
        cuda::memory::device::vector<int> d(n);                                                     \
        type<int> h_a(n);                                                                           \
        type<int> h_b(n);                                                                           \
                                                                                                    \
        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));                      \
        BOOST_CHECK(cuda::copy(&*h_a.begin(), &*h_a.end(), d.begin()) == d.end());                  \
        BOOST_CHECK(cuda::copy(d.begin(), d.end(), &*h_b.begin()) == &*h_b.end());                  \
        BOOST_CHECK_EQUAL_COLLECTIONS(&*h_a.begin(), &*h_a.end(), &*h_b.begin(), &*h_b.end());      \
    }                                                                                               \
    BOOST_DATA_TEST_CASE(const_pointer, dataset, n) {                                               \
        cuda::memory::device::vector<int> d(n);                                                     \
        type<int> h_a(n);                                                                           \
        type<int> h_b(n);                                                                           \
                                                                                                    \
        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));                      \
        type<int> const& h_a_const(h_a);                                                            \
        BOOST_CHECK(cuda::copy(&*h_a_const.begin(), &*h_a_const.end(), d.begin()) == d.end());      \
                                                                                                    \
        cuda::memory::device::vector<int> const& d_const(d);                                        \
        BOOST_CHECK(cuda::copy(d_const.begin(), d_const.end(), &*h_b.begin()) == &*h_b.end());      \
        BOOST_CHECK_EQUAL_COLLECTIONS(&*h_a.begin(), &*h_a.end(), &*h_b.begin(), &*h_b.end());      \
    }                                                                                               \
    BOOST_DATA_TEST_CASE(asynchronous, dataset, n) {                                                \
        cuda::stream stream;                                                                        \
        cuda::memory::device::vector<int> d(n);                                                     \
        type<int> h_a(n);                                                                           \
        type<int> h_b(n);                                                                           \
                                                                                                    \
        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));                      \
        BOOST_CHECK(cuda::copy(h_a.begin(), h_a.end(), d.begin(), stream) == d.end());              \
        stream.synchronize();                                                                       \
                                                                                                    \
        BOOST_CHECK(cuda::copy(d.begin(), d.end(), h_b.begin(), stream) == h_b.end());              \
        stream.synchronize();                                                                       \
        BOOST_CHECK_EQUAL_COLLECTIONS(h_a.begin(), h_a.end(), h_b.begin(), h_b.end());              \
    }


BOOST_AUTO_TEST_SUITE(host_host)
    TEST_HOST_HOST(cuda::memory::host::vector, cuda::memory::host::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(managed_managed)
    TEST_HOST_HOST(cuda::memory::managed::vector, cuda::memory::managed::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(host_managed)
    TEST_HOST_HOST(cuda::memory::host::vector, cuda::memory::managed::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(std_host)
    TEST_HOST_HOST(std::vector, cuda::memory::host::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(std_managed)
    TEST_HOST_HOST(std::vector, cuda::memory::managed::vector)
BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(device_host)
    TEST_DEVICE_HOST(cuda::memory::host::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(device_managed)
    TEST_DEVICE_HOST(cuda::memory::managed::vector)
BOOST_AUTO_TEST_SUITE_END()

BOOST_AUTO_TEST_SUITE(device_std)
    TEST_DEVICE_HOST(std::vector)
BOOST_AUTO_TEST_SUITE_END()


BOOST_AUTO_TEST_SUITE(device_device)
    BOOST_DATA_TEST_CASE(nonconst_iterator, dataset, n) {
        cuda::memory::host::vector<int> h_a(n);
        cuda::memory::host::vector<int> h_b(n);
        cuda::memory::device::vector<int> d_a(n);
        cuda::memory::device::vector<int> d_b(n);

        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));
        BOOST_CHECK(cuda::copy(h_a.begin(), h_a.end(), d_a.begin()) == d_a.end());
        BOOST_CHECK(cuda::copy(d_a.begin(), d_a.end(), d_b.begin()) == d_b.end());
        BOOST_CHECK(cuda::copy(d_b.begin(), d_b.end(), h_b.begin()) == h_b.end());
        BOOST_CHECK_EQUAL_COLLECTIONS(h_a.begin(), h_a.end(), h_b.begin(), h_b.end());
    }
    BOOST_DATA_TEST_CASE(const_iterator, dataset, n) {
        cuda::memory::host::vector<int> h_a(n);
        cuda::memory::host::vector<int> h_b(n);
        cuda::memory::device::vector<int> d_a(n);
        cuda::memory::device::vector<int> d_b(n);

        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));
        cuda::memory::host::vector<int> const& h_a_const(h_a);
        BOOST_CHECK(cuda::copy(h_a_const.begin(), h_a_const.end(), d_a.begin()) == d_a.end());

        cuda::memory::device::vector<int> const& d_a_const(d_a);
        BOOST_CHECK(cuda::copy(d_a_const.begin(), d_a_const.end(), d_b.begin()) == d_b.end());

        cuda::memory::device::vector<int> const& d_b_const(d_b);
        BOOST_CHECK(cuda::copy(d_b_const.begin(), d_b_const.end(), h_b.begin()) == h_b.end());

        BOOST_CHECK_EQUAL_COLLECTIONS(h_a.begin(), h_a.end(), h_b.begin(), h_b.end());
    }
    BOOST_DATA_TEST_CASE(asynchronous, dataset, n) {
        cuda::stream stream;
        cuda::memory::host::vector<int> h_a(n);
        cuda::memory::host::vector<int> h_b(n);
        cuda::memory::device::vector<int> d_a(n);
        cuda::memory::device::vector<int> d_b(n);

        std::generate(h_a.begin(), h_a.end(), std::bind(dist, std::ref(gen)));
        BOOST_CHECK(cuda::copy(h_a.begin(), h_a.end(), d_a.begin(), stream) == d_a.end());
        stream.synchronize();
        BOOST_CHECK(cuda::copy(d_a.begin(), d_a.end(), d_b.begin(), stream) == d_b.end());
        stream.synchronize();
        BOOST_CHECK(cuda::copy(d_b.begin(), d_b.end(), h_b.begin(), stream) == h_b.end());
        stream.synchronize();

        BOOST_CHECK_EQUAL_COLLECTIONS(h_a.begin(), h_a.end(), h_b.begin(), h_b.end());
    }
BOOST_AUTO_TEST_SUITE_END()

BOOST_DATA_TEST_CASE(memset_iterators, dataset * boost::unit_test::data::make<unsigned char>({0, 0xff, 42}), n, c) {
    std::vector<unsigned char> v(n, c);
    cuda::memory::host::vector<unsigned char> h(n);
    cuda::memory::device::vector<unsigned char> d(n);

    cuda::memset(d.begin(), d.end(), c);
    BOOST_CHECK(cuda::copy(d.begin(), d.end(), h.begin()) == h.end());
    BOOST_CHECK_EQUAL_COLLECTIONS(v.begin(), v.end(), h.begin(), h.end());
}
