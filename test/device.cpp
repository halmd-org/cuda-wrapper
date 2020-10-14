/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE device
#include <boost/test/unit_test.hpp>

#include <cuda_wrapper/cuda_wrapper.hpp>

#define NO_FIXTURE
#include "test.hpp"

BOOST_AUTO_TEST_CASE(device)
{
    cuda::device dev;

    BOOST_CHECK(cuda::device::count() > 0);
    BOOST_TEST_MESSAGE("count: " << dev.count());

    BOOST_CHECK(dev.get() == -1);

    BOOST_CHECK(cuda::device::active(TEST_DEVICE) == false);

    dev.set(TEST_DEVICE);
    BOOST_CHECK(dev.get() == 0);

    BOOST_CHECK(cuda::device::active(TEST_DEVICE) == true);

    dev.set(TEST_DEVICE);
    BOOST_CHECK(dev.get() == 0);

    dev.reset();
    BOOST_CHECK(dev.get() == -1);

    dev.set(TEST_DEVICE);
    BOOST_CHECK(dev.get() == 0);

    dev.remove();
    BOOST_CHECK(dev.get() == -1);

    dev.set(TEST_DEVICE);
    BOOST_CHECK(dev.get() == 0);

    cuda::device::properties prop(dev);

    BOOST_TEST_MESSAGE("device name: " << prop.name());
    BOOST_TEST_MESSAGE("total global mem: " << prop.total_global_mem());
    BOOST_TEST_MESSAGE("shared mem per block: " << prop.shared_mem_per_block());
    BOOST_TEST_MESSAGE("regs per block: " << prop.regs_per_block());
    BOOST_TEST_MESSAGE("warp size: " << prop.warp_size());
    BOOST_TEST_MESSAGE("mem pitch: " << prop.mem_pitch());
    BOOST_TEST_MESSAGE("max threads per block: " << prop.max_threads_per_block());
    dim3 max_threads = prop.max_threads_dim();
    BOOST_TEST_MESSAGE("max threads: dim(" << max_threads.x << ", " << max_threads.y << ", " << max_threads.z << ")");
    dim3 max_grid_size = prop.max_grid_size();
    BOOST_TEST_MESSAGE("max threads: dim(" << max_grid_size.x << ", " << max_grid_size.y << ", "
        << max_grid_size.z << ")");
    BOOST_TEST_MESSAGE("total const mem: " << prop.total_const_mem());
    BOOST_TEST_MESSAGE("major: " << prop.major());
    BOOST_TEST_MESSAGE("minor: " << prop.warp_size());
    BOOST_TEST_MESSAGE("clock_rate: " << prop.clock_rate());
    BOOST_TEST_MESSAGE("texture_alignment: " << prop.texture_alignment());
    BOOST_TEST_MESSAGE("device_overlap: " << prop.device_overlap());
    BOOST_TEST_MESSAGE("multi_processor_count: " << prop.multi_processor_count());
    BOOST_TEST_MESSAGE("max_threads_per_multi_processor: " << prop.max_threads_per_multi_processor());
}
