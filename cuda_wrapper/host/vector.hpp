/*
 * Copyright © 2007, 2012 Peter Colberg
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

#ifndef CUDA_WRAPPER_HOST_VECTOR_HPP
#define CUDA_WRAPPER_HOST_VECTOR_HPP

#include <cuda_wrapper/detail/random_access_iterator.hpp>
#include <cuda_wrapper/host/allocator.hpp>
#include <cuda_wrapper/iterator_category.hpp>

#include <vector>

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
#if CUDART_VERSION >= 2020
    typedef detail::random_access_iterator<pointer, device_random_access_iterator_tag> device_iterator;
    typedef detail::random_access_iterator<const_pointer, device_random_access_iterator_tag> const_device_iterator;
#endif

public:
    /** creates an empty vector */
    vector(_Alloc const& alloc = _Alloc(cudaHostAllocMapped)) : _Base(alloc) {}
    /** creates a vector with n elements */
    vector(size_type n, T const& t = T(), _Alloc const& alloc = _Alloc(cudaHostAllocMapped)) : _Base(n, t, alloc) {}
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

#if CUDART_VERSION >= 2020
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
        pointer p = nullptr;
        CUDA_CALL( cudaHostGetDevicePointer(&p, &*_Base::begin(), 0) );
        return device_iterator(p);
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
        const_pointer p = nullptr;
        CUDA_CALL( cudaHostGetDevicePointer(&p, const_cast<pointer>(&*_Base::begin()), 0) );
        return const_device_iterator(p);
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
        pointer p = nullptr;
        CUDA_CALL( cudaHostGetDevicePointer(&p, &*_Base::begin(), 0) );
        return device_iterator(p + _Base::size());
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
        const_pointer p = nullptr;
        CUDA_CALL( cudaHostGetDevicePointer(&p, const_cast<pointer>(&*_Base::begin()), 0) );
        return const_device_iterator(p + _Base::size());
    }
#endif /* CUDART_VERSION >= 2020 */
};

} // namespace host
} // namespace cuda

#endif /* ! CUDA_WRAPPER_HOST_VECTOR_HPP */
