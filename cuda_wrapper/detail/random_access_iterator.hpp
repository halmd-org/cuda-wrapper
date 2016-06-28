/*
 * Copyright © 2010, 2012 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP
#define CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP

#include <boost/iterator/iterator_adaptor.hpp>
#include <boost/iterator/iterator_traits.hpp>
#include <boost/type_traits/is_convertible.hpp>
#include <boost/utility/enable_if.hpp>

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
      , boost::random_access_traversal_tag
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
      , typename boost::enable_if<
            boost::is_convertible<OtherIterator, Iterator>
          , enabler
        >::type = enabler()
    )
      : random_access_iterator::iterator_adaptor_(other.base()) {}
};

} // namespace detail
} // namespace cuda

#endif /* CUDA_WRAPPER_DETAIL_RANDOM_ACCESS_ITERATOR_HPP */
