/* Allocator that wraps cuMemHostAlloc -*- C++ -*-
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

#ifndef CUDA_MEMORY_HOST_ALLOCATOR_HPP
#define CUDA_MEMORY_HOST_ALLOCATOR_HPP

#include <limits>
#include <new>

#include <cuda.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {
namespace memory {
namespace host {

using std::size_t;
using std::ptrdiff_t;

/**
 * Allocator that wraps cuMemHostAlloc
 *
 * The implementation of a custom allocator class for the STL as described
 * here: https://en.cppreference.com/w/cpp/memory/allocator
 *
 * The same pattern is used in ext/malloc_allocator.h of the GNU Standard C++
 * Library, which wraps "C" malloc.
 */
template <typename T>
class allocator {
public:
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef T& reference;
    typedef T const& const_reference;
    typedef T value_type;

    template <typename U> struct rebind { typedef allocator<U> other; };

    allocator(unsigned int flags = 0) noexcept : flags_(flags) {}
    allocator(allocator const& alloc) noexcept : flags_(alloc.flags_) {}

    template<typename U>
    allocator(allocator<U> const& alloc) noexcept : flags_(alloc.flags_) {}

    ~allocator() {}

    pointer address(reference x) const noexcept { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const* = 0)
    {
        void* p;

        if (__builtin_expect(s > this->max_size(), false)) {
            throw std::bad_alloc();
        }

        CU_CALL(cuMemHostAlloc(&p, s * sizeof(T), flags_));

        return reinterpret_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type)
    {
        cuMemFreeHost(reinterpret_cast<void*>(p));
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

private:
    unsigned int flags_;
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

} // namespace host
} // namespace memory
} // namespace cuda

#endif // CUDA_MEMORY_HOST_ALLOCATOR_HPP
