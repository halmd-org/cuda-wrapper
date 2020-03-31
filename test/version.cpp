/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE version
#include <boost/test/included/unit_test.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

BOOST_AUTO_TEST_CASE(version)
{
    BOOST_TEST_MESSAGE("driver: " << cuda::driver_version());
    BOOST_TEST_MESSAGE("runtime: " << cuda::runtime_version());
}
