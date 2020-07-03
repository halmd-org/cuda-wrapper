/*
 * Copyright (C) 2007, 2012 Peter Colberg
 * Copyright (C) 2020       Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_MEMORY_MANAGED_VECTOR_HPP
#define CUDA_MEMORY_MANAGED_VECTOR_HPP

#include <vector>

#include <cuda.h>

#include <cuda_wrapper/detail/random_access_iterator.hpp>
#include <cuda_wrapper/iterator_category.hpp>
#include <cuda_wrapper/memory/managed/allocator.hpp>

namespace cuda {
namespace memory {
namespace managed {

/**
 * CUDA managed memory vector
 */
template <typename T>
class vector : public std::vector<T, allocator<T> >
{
private:
    typedef allocator<T> _Alloc;
    typedef std::vector<T, allocator<T>> _Base;

public:
    typedef typename _Base::size_type size_type;
    typedef typename _Base::value_type value_type;
    typedef typename _Base::pointer pointer;
    typedef typename _Base::const_pointer const_pointer;
    typedef detail::random_access_iterator<typename _Base::iterator, managed_random_access_iterator_tag> iterator;
    typedef detail::random_access_iterator<typename _Base::const_iterator, managed_random_access_iterator_tag> const_iterator;

public:
    /** creates an empty vector */
    vector() : _Base() {}
    /** creates a vector with n elements */
    vector(size_type n, T const& t = T()) : _Base(n, t) {}
    /** creates a vector with a copy of a range */
    template <class InputIterator>
    vector(InputIterator begin, InputIterator end) : _Base(begin, end) {}

    /**
     * returns device pointer to allocated device memory
     */
    operator pointer()
    {
        return _Base::data();
    }

    operator const_pointer() const
    {
        return _Base::data();
    }

    /**
     * Returns host iterator to the first element of the array.
     */
    iterator begin()
    {
        return iterator(_Base::begin());
    }

    /**
     * Returns host iterator to the first element of the array.
     */
    const_iterator begin() const
    {
        return const_iterator(_Base::begin());
    }

    /**
     * Returns host iterator to the element following the last element of the array.
     */
    iterator end()
    {
        return iterator(_Base::end());
    }

    /**
     * Returns host iterator to the element following the last element of the array.
     */
    const_iterator end() const
    {
        return const_iterator(_Base::end());
    }


};

} // namespace managed
} // namespace memory

// make the managed memory vector the default
using memory::managed::vector;
} // namespace cuda

#endif // CUDA_MEMORY_MANAGED_VECTOR_HPP
