/*
 * Copyright (C) 2007, 2012 Peter Colberg
 * Copyright (C) 2020       Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_HOST_VECTOR_HPP
#define CUDA_HOST_VECTOR_HPP

#include <vector>

#include <cuda.h>

#include <cuda_wrapper/detail/random_access_iterator.hpp>
#include <cuda_wrapper/host/allocator.hpp>
#include <cuda_wrapper/iterator_category.hpp>

namespace cuda {
namespace host {

/**
 * CUDA page-locked host memory vector
 */
template <typename T>
class vector
  : public std::vector<T, allocator<T> >
{
private:
    typedef allocator<T> _Alloc;
    typedef std::vector<T, allocator<T> > _Base;

public:
    typedef typename _Base::size_type size_type;
    typedef typename _Base::value_type value_type;
    typedef typename _Base::pointer pointer;
    typedef typename _Base::const_pointer const_pointer;
    typedef detail::random_access_iterator<typename _Base::iterator, host_random_access_iterator_tag> iterator;
    typedef detail::random_access_iterator<typename _Base::const_iterator, host_random_access_iterator_tag> const_iterator;
    typedef detail::random_access_iterator<pointer, device_random_access_iterator_tag> device_iterator;
    typedef detail::random_access_iterator<const_pointer, device_random_access_iterator_tag> const_device_iterator;

public:
    /** creates an empty vector */
    vector(_Alloc const& alloc = _Alloc()) : _Base(alloc) {}
    /** creates a vector with n elements */
    vector(size_type n, T const& t = T(), _Alloc const& alloc = _Alloc()) : _Base(n, t, alloc) {}
    /** creates a vector with a copy of a range */
    template <class InputIterator>
    vector(InputIterator begin, InputIterator end, _Alloc const& alloc = _Alloc()) : _Base(begin, end, alloc) {}

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

    /**
     * Returns device iterator to the first element of the array.
     *
     * This is needed for backwards-compatibility with devices of compute
     * capability 1.1 or lower. Devices of compute capability support
     * unified addressing, i.e. a pointer to page-locked host memory
     * is usable both on the device and the host.
     */
    device_iterator gbegin()
    {
        CUdeviceptr p;
        CU_CALL(cuMemHostGetDevicePointer(&p, &*_Base::begin(), 0));
        return device_iterator(static_cast<pointer>(p));
    }

    /**
     * Returns device iterator to the first element of the array.
     *
     * This is needed for backwards-compatibility with devices of compute
     * capability 1.1 or lower. Devices of compute capability support
     * unified addressing, i.e. a pointer to page-locked host memory
     * is usable both on the device and the host.
     */
    const_device_iterator gbegin() const
    {
        CUdeviceptr p;
        CU_CALL(cuMemHostGetDevicePointer(&p, const_cast<pointer>(&*_Base::begin()), 0));
        return const_device_iterator(static_cast<const_pointer>(p));
    }

    /**
     * Returns device iterator to the element following the last element of the array.
     *
     * This is needed for backwards-compatibility with devices of compute
     * capability 1.1 or lower. Devices of compute capability support
     * unified addressing, i.e. a pointer to page-locked host memory
     * is usable both on the device and the host.
     */
    device_iterator gend()
    {
        CUdeviceptr p;
        CU_CALL(cuMemHostGetDevicePointer(&p, &*_Base::begin(), 0));
        return device_iterator(static_cast<pointer>(p) + _Base::size());
    }

    /**
     * Returns device iterator to the element following the last element of the array.
     *
     * This is needed for backwards-compatibility with devices of compute
     * capability 1.1 or lower. Devices of compute capability support
     * unified addressing, i.e. a pointer to page-locked host memory
     * is usable both on the device and the host.
     */
    const_device_iterator gend() const
    {
        CUdeviceptr p;
        CU_CALL(cuMemHostGetDevicePointer(&p, const_cast<pointer>(&*_Base::begin()), 0));
        return const_device_iterator(static_cast<const_pointer>(p) + _Base::size());
    }
};

} // namespace host
} // namespace cuda

#endif /* ! CUDA_HOST_VECTOR_HPP */
