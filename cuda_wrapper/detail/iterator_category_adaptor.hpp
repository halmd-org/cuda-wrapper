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

#ifndef CUDA_WRAPPER_DETAIL_ITERATOR_CATEGORY_ADAPTOR_HPP
#define CUDA_WRAPPER_DETAIL_ITERATOR_CATEGORY_ADAPTOR_HPP

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <type_traits>

namespace cuda {
namespace detail {

/**
 * Iterator category adapter
 */
template <typename Iterator, typename Category>
class iterator_category_adaptor
  : public boost::iterator_adaptor<
        iterator_category_adaptor<Iterator, Category>
      , Iterator
      , boost::use_default
      , Category
    >
{
private:
    struct enabler {};  // a private type avoids misuse

public:
    iterator_category_adaptor()
      : iterator_category_adaptor::iterator_adaptor_(0) {}

    explicit iterator_category_adaptor(Iterator iterator)
      : iterator_category_adaptor::iterator_adaptor_(iterator) {}

    template <typename OtherIterator>
    iterator_category_adaptor(
        iterator_category_adaptor<OtherIterator, Category> const& other
      , typename std::enable_if<
            std::is_convertible<OtherIterator, Iterator>::value
          , enabler
        >::type = enabler()
    )
      : iterator_category_adaptor::iterator_adaptor_(other.base()) {}
};

} // namespace detail
} // namespace cuda

#endif /* CUDA_WRAPPER_DETAIL_ITERATOR_CATEGORY_ADAPTOR_HPP */
