/*
 * Copyright © 2012 Felix Höfling, Peter Colberg
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#define BOOST_TEST_MODULE host_vector
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>
#include <vector>

#include <cuda_wrapper/cuda_wrapper.hpp>

struct set_map_host {
    set_map_host()
    {
        CUDA_CALL( cudaSetDeviceFlags(cudaDeviceMapHost) );
    }
};

BOOST_GLOBAL_FIXTURE( set_map_host )

/**
 * test mapping of page-locked host memory into device memory
 */
BOOST_AUTO_TEST_CASE( mapped_host_memory )
{
    BOOST_TEST_MESSAGE("allocate and fill page-locked host memory");
    cuda::host::vector<int> h_i(10000);
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );

    BOOST_TEST_MESSAGE("device → device: copy mapped host memory to device-only memory");
    cuda::vector<int> g_i(h_i.size());
    BOOST_CHECK(
        g_i.end() == cuda::copy(h_i.gbegin(), h_i.gend(), g_i.begin())
    );

    BOOST_TEST_MESSAGE("device → host: copy device-only memory to conventional host memory");
    std::vector<int> v_j(g_i.size());
    BOOST_CHECK(
        v_j.end() == cuda::copy(g_i.begin(), g_i.end(), v_j.begin())
    );

    // check data integrity
    BOOST_CHECK_EQUAL_COLLECTIONS(
        v_j.begin(), v_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(v_j.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin(), h_i.end(), v_j.begin(), v_j.end()
    );

    BOOST_TEST_MESSAGE("device → host: copy mapped host memory from device to conventional host memory");
    BOOST_CHECK(
        v_j.end() == cuda::copy(h_i.gbegin(), h_i.gend(), v_j.begin())
    );

    // check data integrity
    BOOST_CHECK_EQUAL_COLLECTIONS(
        v_j.begin(), v_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(v_j.size())
    );
}
