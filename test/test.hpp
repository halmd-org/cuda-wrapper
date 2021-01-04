/*
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef TEST_HPP
#define TEST_HPP

#include <boost/version.hpp>

// macro set by cmake to define which device is used
#ifndef TEST_DEVICE
# warning "TEST_DEVICE was not set, using device 0"
# define TEST_DEVICE 0
#endif

// initialize the device with a Boost fixture
// disabled by defining NO_FIXTURE
#ifndef NO_FIXTURE
struct Fixture
{
    Fixture() { dev.set(TEST_DEVICE); }
    cuda::device dev;
};

# if BOOST_VERSION < 106500
BOOST_GLOBAL_FIXTURE(Fixture);
# else
BOOST_TEST_GLOBAL_CONFIGURATION(Fixture);
# endif
#endif

#endif /* ! TEST_HPP */
