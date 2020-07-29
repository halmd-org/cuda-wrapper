/*
 * Copyright (C) 2007, 2012 Peter Colberg
 * Copyright (C) 2020       Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_MEMORY_HOST_VECTOR_HPP
#define CUDA_MEMORY_HOST_VECTOR_HPP

#include <vector>

#include <cuda.h>

#include <cuda_wrapper/detail/random_access_iterator.hpp>
#include <cuda_wrapper/iterator_category.hpp>
#include <cuda_wrapper/memory/host/allocator.hpp>

namespace cuda {
namespace memory {
namespace host {

/**
 * CUDA page-locked host memory vector
 *
 * Custom STL like vector class as described in:
 * https://en.cppreference.com/w/cpp/container/vector
 */
template <typename T>
class vector : public std::vector<T, allocator<T>>
{
private:
    typedef std::vector<T, allocator<T>> _Base;

public:
    typedef typename _Base::value_type value_type;
    typedef typename _Base::allocator_type allocator_type;
    typedef typename _Base::size_type size_type;
    typedef typename _Base::difference_type difference_type;
    typedef typename _Base::reference reference;
    typedef typename _Base::const_reference const_reference;
    typedef typename _Base::pointer pointer;
    typedef typename _Base::const_pointer const_pointer;
    typedef detail::random_access_iterator<typename _Base::iterator, host_random_access_iterator_tag> iterator;
    typedef detail::random_access_iterator<typename _Base::const_iterator, host_random_access_iterator_tag> const_iterator;

    /** use the std::vector constructor and assignment operator */
    using _Base::_Base;
    using _Base::operator=;

    /**
     * Returns host iterator to the first element of the array.
     */
    iterator begin() noexcept
    {
        return iterator(_Base::begin());
    }

    /**
     * Returns host iterator to the first element of the array.
     */
    const_iterator begin() const noexcept
    {
        return const_iterator(_Base::begin());
    }

    /**
     * Returns host iterator to the element following the last element of the array.
     */
    iterator end() noexcept
    {
        return iterator(_Base::end());
    }

    /**
     * Returns host iterator to the element following the last element of the array.
     */
    const_iterator end() const noexcept
    {
        return const_iterator(_Base::end());
    }
};

} // namespace host
} // namespace memory
} // namespace cuda

#endif // CUDA_MEMORY_HOST_VECTOR_HPP
