/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE version
#include <boost/test/unit_test.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

BOOST_AUTO_TEST_CASE(version)
{
    int driver = cuda::driver_version();
    BOOST_TEST_MESSAGE("driver: " << driver);

    int runtime = cuda::runtime_version();
    BOOST_TEST_MESSAGE("runtime: " << runtime);

    BOOST_CHECK_MESSAGE(driver >= runtime, "The CUDA driver must not be older than the CUDA runtime library");
}
