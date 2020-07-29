/* Allocator that wraps cuMemAlloc -*- C++ -*-
 *
 * Copyright (C) 2016 Felix Höfling
 * Copyright (C) 2008 Peter Colberg
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_MEMORY_DEVICE_ALLOCATOR_HPP
#define CUDA_MEMORY_DEVICE_ALLOCATOR_HPP

#include <limits>
#include <new>

#include <cuda.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {
namespace memory {
namespace device {

using std::size_t;
using std::ptrdiff_t;

/**
 * Allocator that wraps cuMemAlloc
 *
 * The implementation of a custom allocator class for the STL as described
 * here: https://en.cppreference.com/w/cpp/memory/allocator
 *
 * The same pattern is used in ext/malloc_allocator.h of the GNU Standard C++
 * Library, which wraps "C" malloc.
 */
template <typename T>
struct allocator {
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T value_type;

    template <typename U> struct rebind { typedef allocator<U> other; };

    allocator() noexcept {}
    allocator(allocator const&) noexcept {}

    template <typename U>
    allocator(allocator<U> const&) noexcept {}

    ~allocator() {}

    pointer address(reference x) const noexcept { return &x; }
    const_pointer address(const_reference x) const noexcept { return &x; }

    pointer allocate(size_type s, void const* = 0)
    {
        CUdeviceptr p;

        if (s == 0) {
            return NULL;
        }
        if (__builtin_expect(s > this->max_size(), false)) {
            throw std::bad_alloc();
        }

        CU_CALL(cuMemAlloc(&p, s * sizeof(T)));

        return reinterpret_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type)
    {
        if (p != NULL) {
            cuMemFree(reinterpret_cast<CUdeviceptr>(p));
        }
    }

    size_type max_size() const noexcept
    {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    template<typename U, typename... Args>
    void construct(U* p, Args&&... args)
    {
        ::new((void *)p) U(std::forward<Args>(args)...);
    }

    template<class U>
    void destroy(U* p)
    {
        p->~U();
    }
};

template<typename T, typename U>
inline bool operator==(allocator<T> const&, allocator<U> const&) noexcept
{
    return true;
}

template<typename T, typename U>
inline bool operator!=(allocator<T> const&, allocator<U> const&) noexcept
{
    return false;
}

} // namespace device
} // namespace memory
} // namespace cuda

#endif // CUDA_MEMORY_DEVICE_ALLOCATOR_HPP
