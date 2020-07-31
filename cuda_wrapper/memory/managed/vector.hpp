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
#include <cuda_wrapper/device.hpp>
#include <cuda_wrapper/iterator_category.hpp>
#include <cuda_wrapper/memory/managed/allocator.hpp>
#include <cuda_wrapper/stream.hpp>

namespace cuda {
namespace memory {
namespace managed {

/**
 * CUDA managed memory vector
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
    typedef detail::random_access_iterator<typename _Base::iterator, managed_random_access_iterator_tag> iterator;
    typedef detail::random_access_iterator<typename _Base::const_iterator, managed_random_access_iterator_tag> const_iterator;
    typedef detail::random_access_iterator<pointer, device_random_access_iterator_tag> device_iterator;
    typedef detail::random_access_iterator<const_pointer, device_random_access_iterator_tag> const_device_iterator;

    /** use the std::vector constructor and assignment operator */
    using _Base::_Base;
    using _Base::operator=;

    /**
     * Advise about the usage pattern of the vector
     */
    void advise(CUmem_advise advice, ::cuda::device const& device)
    {
        CU_CALL(cuMemAdvise(reinterpret_cast<CUdeviceptr>(_Base::data()), _Base::capacity(), advice, device.data()));
    }
    /**
     * Advise about the usage pattern of the vector, overloaded for usage with CU_DEVICE_CPU
     */
    void advise(CUmem_advise advice, CUdevice device)
    {
        CU_CALL(cuMemAdvise(reinterpret_cast<CUdeviceptr>(_Base::data()), _Base::capacity(), advice, device));
    }

    /**
     * Prefetch the vector to the specified device
     */
    void prefetch_async(::cuda::device const& device, stream const& stream)
    {
        CU_CALL(cuMemPrefetchAsync(
            reinterpret_cast<CUdeviceptr>(_Base::data()), _Base::capacity(), device.data(), stream.data()));
    }
    /**
     * Prefetch the vector to the specified device, overloaded for usage with CU_DEVICE_CPU
     */
    void prefetch_async(CUdevice device, stream const& stream)
    {
        CU_CALL(cuMemPrefetchAsync(
            reinterpret_cast<CUdeviceptr>(_Base::data()), _Base::capacity(), device, stream.data()));
    }

    /**
     * Returns device pointer to allocated device memory
     */
    operator pointer() noexcept
    {
        return _Base::data();
    }

    /**
     * Returns const device pointer to allocated device memory
     */
    operator const_pointer() const noexcept
    {
        return _Base::data();
    }

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

    device_iterator device_begin() noexcept
    {
        return device_iterator(&*_Base::begin());
    }

    const_device_iterator device_begin() const noexcept
    {
        return const_device_iterator(&*_Base::begin());
    }

    device_iterator device_end() noexcept
    {
        return device_iterator(&*_Base::end());
    }

    const_device_iterator device_end() const noexcept
    {
        return const_device_iterator(&*_Base::end());
    }
};

} // namespace managed
} // namespace memory

// make the managed memory vector the default
using memory::managed::vector;
} // namespace cuda

#endif // CUDA_MEMORY_MANAGED_VECTOR_HPP
