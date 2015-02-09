/*
 * Copyright © 2010, 2012 Peter Colberg
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

#ifndef CUDA_WRAPPER_ITERATOR_CATEGORY_HPP
#define CUDA_WRAPPER_ITERATOR_CATEGORY_HPP

#include <iterator>
#include <boost/version.hpp>

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

#if BOOST_VERSION >= 105700
#include <boost/type_traits/is_convertible.hpp>
namespace boost {

template<>
struct is_convertible<cuda::device_random_access_iterator_tag, std::random_access_iterator_tag>
  : public true_type {};

}
#endif // BOOST_VERSION >= 105700

#endif /* CUDA_WRAPPER_ITERATOR_CATEGORY_HPP */
