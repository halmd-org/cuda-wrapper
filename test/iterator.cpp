/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE iterator
#include <boost/test/unit_test.hpp>

#include <type_traits>

#include <cuda_wrapper/cuda_wrapper.hpp>
#include <cuda_wrapper/iterator_category.hpp>

BOOST_AUTO_TEST_CASE(convertible) {
    // the device iterator tag should not be convertible to std::random_access_iterator_tag
    BOOST_CHECK((std::is_convertible<
        cuda::device_random_access_iterator_tag
      , std::random_access_iterator_tag
    >::value) == false);

    // host iterator tag should be convertible to std::random_access_iterator_tag
    BOOST_CHECK((std::is_convertible<
        cuda::host_random_access_iterator_tag
      , std::random_access_iterator_tag
    >::value) == true);

    // managed memory iterator tag must be convertible to host, device and std::random_access_iterator_tag
    BOOST_CHECK((std::is_convertible<
        cuda::managed_random_access_iterator_tag
      , std::random_access_iterator_tag
    >::value) == true);
    BOOST_CHECK((std::is_convertible<
        cuda::managed_random_access_iterator_tag
      , cuda::host_random_access_iterator_tag
    >::value) == true);
    BOOST_CHECK((std::is_convertible<
        cuda::managed_random_access_iterator_tag
      , cuda::device_random_access_iterator_tag
    >::value) == true);
}

BOOST_AUTO_TEST_CASE(vectors) {
    BOOST_CHECK((std::is_same<
        typename std::iterator_traits<cuda::memory::device::vector<int>::iterator>::iterator_category
      , cuda::device_random_access_iterator_tag
    >::value) == true);

    BOOST_CHECK((std::is_same<
        typename std::iterator_traits<cuda::memory::host::vector<int>::iterator>::iterator_category
      , cuda::host_random_access_iterator_tag
    >::value) == true);

    BOOST_CHECK((std::is_same<
        typename std::iterator_traits<cuda::memory::managed::vector<int>::iterator>::iterator_category
      , cuda::managed_random_access_iterator_tag
    >::value) == true);
}
