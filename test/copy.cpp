/*
 * Copyright (C) 2010, 2012 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#define BOOST_TEST_MODULE copy
#include <boost/test/unit_test.hpp>

#include <algorithm>
#include <boost/iterator/counting_iterator.hpp>
#include <boost/numeric/ublas/vector.hpp>
#include <vector>

#include <cuda_wrapper/cuda_wrapper.hpp>

#include "test.hpp"

BOOST_AUTO_TEST_SUITE( global_device_memory )

/**
 * test cuda::copy with non-const iterators
 */
BOOST_AUTO_TEST_CASE( nonconst_iterator )
{
    cuda::host::vector<int> h_i(10000);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );

    cuda::host::vector<int> h_j(10000);
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(42)
      , boost::counting_iterator<int>(h_i.size() + 42)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , g_j.begin()) == g_j.end()
    );
    BOOST_CHECK( cuda::copy(
        g_j.begin()
      , g_j.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(42)
      , boost::counting_iterator<int>(h_i.size() + 42)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(43)
      , boost::counting_iterator<int>(h_i.size() + 43)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(43)
      , boost::counting_iterator<int>(h_i.size() + 43)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test cuda::copy with const iterators
 */
BOOST_AUTO_TEST_CASE( const_iterator )
{
    cuda::host::vector<int> h_i(99999);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    cuda::host::vector<int> const& h_i_const(h_i);
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , g_i.begin()) == g_i.end()
    );

    cuda::host::vector<int> h_j(99999);
    cuda::vector<int> const& g_i_const(g_i);
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(44)
      , boost::counting_iterator<int>(h_i.size() + 44)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , g_j.begin()) == g_j.end()
    );
    cuda::vector<int> const& g_j_const(g_j);
    BOOST_CHECK( cuda::copy(
        g_j_const.begin()
      , g_j_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(44)
      , boost::counting_iterator<int>(h_i.size() + 44)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(45)
      , boost::counting_iterator<int>(h_i.size() + 45)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(45)
      , boost::counting_iterator<int>(h_i.size() + 45)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test cuda::copy with std::vector::iterator
 */
BOOST_AUTO_TEST_CASE( std_vector_iterator )
{
    std::vector<int> h_i(999);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );

    std::vector<int> h_j(999);
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(46)
      , boost::counting_iterator<int>(h_i.size() + 46)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , g_j.begin()) == g_j.end()
    );
    BOOST_CHECK( cuda::copy(
        g_j.begin()
      , g_j.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(46)
      , boost::counting_iterator<int>(h_i.size() + 46)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(47)
      , boost::counting_iterator<int>(h_i.size() + 47)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(47)
      , boost::counting_iterator<int>(h_i.size() + 47)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test cuda::copy with std::vector::const_iterator
 */
BOOST_AUTO_TEST_CASE( std_vector_const_iterator )
{
    std::vector<int> h_i(100000);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    std::vector<int> const& h_i_const(h_i);
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , g_i.begin()) == g_i.end()
    );

    std::vector<int> h_j(100000);
    cuda::vector<int> const& g_i_const(g_i);
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(48)
      , boost::counting_iterator<int>(h_i.size() + 48)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , g_j.begin()) == g_j.end()
    );
    cuda::vector<int> const& g_j_const(g_j);
    BOOST_CHECK( cuda::copy(
        g_j_const.begin()
      , g_j_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(48)
      , boost::counting_iterator<int>(h_i.size() + 48)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(49)
      , boost::counting_iterator<int>(h_i.size() + 49)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(49)
      , boost::counting_iterator<int>(h_i.size() + 49)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test cuda::copy with boost::numeric::ublas::vector::iterator
 */
BOOST_AUTO_TEST_CASE( boost_ublas_vector_iterator )
{
    boost::numeric::ublas::vector<int> h_i(999);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );

    boost::numeric::ublas::vector<int> h_j(999);
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(111)
      , boost::counting_iterator<int>(h_i.size() + 111)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , g_j.begin()) == g_j.end()
    );
    BOOST_CHECK( cuda::copy(
        g_j.begin()
      , g_j.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(111)
      , boost::counting_iterator<int>(h_i.size() + 111)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(12931)
      , boost::counting_iterator<int>(h_i.size() + 12931)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(12931)
      , boost::counting_iterator<int>(h_i.size() + 12931)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test cuda::copy with boost::numeric::ublas::vector::const_iterator
 */
BOOST_AUTO_TEST_CASE( boost_ublas_vector_const_iterator )
{
    boost::numeric::ublas::vector<int> h_i(100000);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    boost::numeric::ublas::vector<int> const& h_i_const(h_i);
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , g_i.begin()) == g_i.end()
    );

    boost::numeric::ublas::vector<int> h_j(100000);
    cuda::vector<int> const& g_i_const(g_i);
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(542)
      , boost::counting_iterator<int>(h_i.size() + 542)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , g_j.begin()) == g_j.end()
    );
    cuda::vector<int> const& g_j_const(g_j);
    BOOST_CHECK( cuda::copy(
        g_j_const.begin()
      , g_j_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(542)
      , boost::counting_iterator<int>(h_i.size() + 542)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(783)
      , boost::counting_iterator<int>(h_i.size() + 783)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(783)
      , boost::counting_iterator<int>(h_i.size() + 783)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test cuda::copy with pointer
 */
BOOST_AUTO_TEST_CASE( pointer )
{
    std::vector<int> h_i(100001);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        &*h_i.begin()
      , &*h_i.end()
      , g_i.begin()) == g_i.end()
    );

    std::vector<int> h_j(100001);
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , &*h_j.begin()) == &*h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_j.begin()
      , &*h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_i.begin()
      , &*h_i.end()
      , &*h_j.begin()
      , &*h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(46)
      , boost::counting_iterator<int>(h_i.size() + 46)
      , &*h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        &*h_i.begin()
      , &*h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , g_j.begin()) == g_j.end()
    );
    BOOST_CHECK( cuda::copy(
        g_j.begin()
      , g_j.end()
      , &*h_j.begin()) == &*h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_j.begin()
      , &*h_j.end()
      , boost::counting_iterator<int>(46)
      , boost::counting_iterator<int>(h_i.size() + 46)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_i.begin()
      , &*h_i.end()
      , &*h_j.begin()
      , &*h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(47)
      , boost::counting_iterator<int>(h_i.size() + 47)
      , &*h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        &*h_i.begin()
      , &*h_i.end()
      , &*h_j.begin()) == &*h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_j.begin()
      , &*h_j.end()
      , boost::counting_iterator<int>(47)
      , boost::counting_iterator<int>(h_i.size() + 47)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_i.begin()
      , &*h_i.end()
      , &*h_j.begin()
      , &*h_j.end()
    );
}

/**
 * test cuda::copy with const pointer
 */
BOOST_AUTO_TEST_CASE( const_pointer )
{
    std::vector<int> h_i(424242);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , &*h_i.begin()
    );
    std::vector<int> const& h_i_const(h_i);
    BOOST_CHECK( cuda::copy(
        &*h_i_const.begin()
      , &*h_i_const.end()
      , g_i.begin()) == g_i.end()
    );

    std::vector<int> h_j(424242);
    cuda::vector<int> const& g_i_const(g_i);
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , &*h_j.begin()) == &*h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_j.begin()
      , &*h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_i.begin()
      , &*h_i.end()
      , &*h_j.begin()
      , &*h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(48)
      , boost::counting_iterator<int>(h_i.size() + 48)
      , &*h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        &*h_i.begin()
      , &*h_i.end()
      , g_i.begin()) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , g_j.begin()) == g_j.end()
    );
    cuda::vector<int> const& g_j_const(g_j);
    BOOST_CHECK( cuda::copy(
        g_j_const.begin()
      , g_j_const.end()
      , &*h_j.begin()) == &*h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_j.begin()
      , &*h_j.end()
      , boost::counting_iterator<int>(48)
      , boost::counting_iterator<int>(h_i.size() + 48)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_i.begin()
      , &*h_i.end()
      , &*h_j.begin()
      , &*h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(49)
      , boost::counting_iterator<int>(h_i.size() + 49)
      , &*h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        &*h_i_const.begin()
      , &*h_i_const.end()
      , &*h_j.begin()) == &*h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_j.begin()
      , &*h_j.end()
      , boost::counting_iterator<int>(49)
      , boost::counting_iterator<int>(h_i.size() + 49)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        &*h_i.begin()
      , &*h_i.end()
      , &*h_j.begin()
      , &*h_j.end()
    );
}

/**
 * test asynchronous cuda::copy
 */
BOOST_AUTO_TEST_CASE( asynchronous )
{
    cuda::stream stream;
    cuda::host::vector<int> h_i(10000);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()
      , stream) == g_i.end()
    );

    cuda::host::vector<int> h_j(10000);
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()
      , stream) == h_j.end()
    );
    stream.synchronize();
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(42)
      , boost::counting_iterator<int>(h_i.size() + 42)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()
      , stream) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , g_j.begin()
      , stream) == g_j.end()
    );
    BOOST_CHECK( cuda::copy(
        g_j.begin()
      , g_j.end()
      , h_j.begin()
      , stream) == h_j.end()
    );
    stream.synchronize();
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(42)
      , boost::counting_iterator<int>(h_i.size() + 42)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(43)
      , boost::counting_iterator<int>(h_i.size() + 43)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , stream) == h_j.end()
    );
    stream.synchronize();
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(43)
      , boost::counting_iterator<int>(h_i.size() + 43)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test asynchronous cuda::copy with const iterators
 */
BOOST_AUTO_TEST_CASE( asynchronous_const_iterators )
{
    cuda::stream stream;
    cuda::host::vector<int> h_i(10000);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    cuda::host::vector<int> const& h_i_const(h_i);
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , g_i.begin()
      , stream) == g_i.end()
    );

    cuda::host::vector<int> h_j(10000);
    cuda::vector<int> const& g_i_const(g_i);
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , h_j.begin()
      , stream) == h_j.end()
    );
    stream.synchronize();
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(44)
      , boost::counting_iterator<int>(h_i.size() + 44)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()
      , stream) == g_i.end()
    );
    cuda::vector<int> g_j(h_i.size());
    BOOST_CHECK( cuda::copy(
        g_i_const.begin()
      , g_i_const.end()
      , g_j.begin()
      , stream) == g_j.end()
    );
    cuda::vector<int> const& g_j_const(g_j);
    BOOST_CHECK( cuda::copy(
        g_j_const.begin()
      , g_j_const.end()
      , h_j.begin()
      , stream) == h_j.end()
    );
    stream.synchronize();
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(44)
      , boost::counting_iterator<int>(h_i.size() + 44)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );

    std::copy(
        boost::counting_iterator<int>(45)
      , boost::counting_iterator<int>(h_i.size() + 45)
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i_const.begin()
      , h_i_const.end()
      , h_j.begin()
      , stream) == h_j.end()
    );
    stream.synchronize();
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(45)
      , boost::counting_iterator<int>(h_i.size() + 45)
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_i.begin()
      , h_i.end()
      , h_j.begin()
      , h_j.end()
    );
}

/**
 * test cuda::memset
 */
BOOST_AUTO_TEST_CASE( memset_iterators )
{
    cuda::host::vector<int> h_i(10000);
    cuda::vector<int> g_i(h_i.size());
    std::copy(
        boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
      , h_i.begin()
    );
    BOOST_CHECK( cuda::copy(
        h_i.begin()
      , h_i.end()
      , g_i.begin()) == g_i.end()
    );

    cuda::host::vector<int> h_j(10000);
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL_COLLECTIONS(
        h_j.begin()
      , h_j.end()
      , boost::counting_iterator<int>(0)
      , boost::counting_iterator<int>(h_i.size())
    );

    cuda::memset(
        g_i.begin()
      , g_i.end()
      , 0
    );
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL( std::count(h_j.begin(), h_j.end(), 0), 10000 );

    cuda::memset(
        g_i.end() - 9999
      , g_i.begin() + 9000
      , 0xff
    );
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL( h_j.front(), 0 );
    BOOST_CHECK_EQUAL( std::count(h_j.begin() + 1, h_j.end() - 1000, -1), 8999 );
    BOOST_CHECK_EQUAL( std::count(h_j.end() - 1000, h_j.end(), 0), 1000 );

    cuda::memset(
        g_i.begin()
      , g_i.end()
      , 1 // assigns *byte* value
    );
    BOOST_CHECK( cuda::copy(
        g_i.begin()
      , g_i.end()
      , h_j.begin()) == h_j.end()
    );
    BOOST_CHECK_EQUAL( std::count(h_j.begin(), h_j.end(), 1), 0 );
}

BOOST_AUTO_TEST_SUITE_END() // global_device_memory
