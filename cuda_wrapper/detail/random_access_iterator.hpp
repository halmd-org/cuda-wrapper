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

#ifndef CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP
#define CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <type_traits>

namespace cuda {
namespace detail {

/**
 * Random access iterator with given base iterator and category.
 */
template <typename Iterator, typename Category>
class random_access_iterator
  : public boost::iterator_adaptor<
        random_access_iterator<Iterator, Category>
      , Iterator
      , boost::use_default
      , std::random_access_iterator_tag
    >
{
private:
    struct enabler {};  // a private type avoids misuse

public:
    typedef Category iterator_category;

    random_access_iterator()
      : random_access_iterator::iterator_adaptor_(0) {}

    explicit random_access_iterator(Iterator iterator)
      : random_access_iterator::iterator_adaptor_(iterator) {}

    template <typename OtherIterator>
    random_access_iterator(
        random_access_iterator<OtherIterator, Category> const& other
      , typename std::enable_if<
            std::is_convertible<OtherIterator, Iterator>::value
          , enabler
        >::type = enabler()
    )
      : random_access_iterator::iterator_adaptor_(other.base()) {}
};

} // namespace detail
} // namespace cuda

#endif /* CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP */
