/*
 * Copyright © 2010, 2012 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_WRAPPER_ITERATOR_CATEGORY_HPP
#define CUDA_WRAPPER_ITERATOR_CATEGORY_HPP

#include <iterator>

namespace cuda {

/**
 * Extend STL iterator with a memory tag.
 *
 * This allows to differentiate between device and host memory iterators.
 */

/**
 * Global device memory.
 *
 * This category does not derive from std::random_access_iterator_tag,
 * which avoids breaking host algorithms that do not and should not
 * check that the category of an iterator is not convertible to
 * cuda::device_random_access_iterator_tag.
 */
struct device_random_access_iterator_tag {};

/**
 * Page-locked host memory.
 *
 * This category derives from std::random_access_iterator_tag.
 */
struct host_random_access_iterator_tag : std::random_access_iterator_tag {};

} // namespace cuda

#endif /* CUDA_WRAPPER_ITERATOR_CATEGORY_HPP */
