/* Allocator that wraps cudaMallocHost -*- C++ -*-
 *
 * Copyright (C) 2016 Felix Höfling
 * Copyright (C) 2008 Peter Colberg
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

#ifndef CUDA_HOST_ALLOCATOR_HPP
#define CUDA_HOST_ALLOCATOR_HPP

#include <bits/functexcept.h>
#include <cstdlib>
#include <cuda_runtime.h>
#include <limits>
#include <new>

#include <cuda_wrapper/error.hpp>

namespace cuda {
namespace host {

using std::size_t;
using std::ptrdiff_t;

/*
 * The implementation of a custom allocator class for the STL is described
 * here and there:
 * http://www.codeproject.com/Articles/4795/C-Standard-Allocator-An-Introduction-and-Implement
 * http://stackoverflow.com/a/11417774
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
    typedef const T* const_pointer;
    typedef T& reference;
    typedef const T& const_reference;
    typedef T value_type;

    template <typename U> struct rebind { typedef allocator<U> other; };

    allocator(unsigned int flags = cudaHostAllocDefault) throw() : _flags(flags) {}

    allocator(const allocator& alloc) throw() : _flags(alloc._flags) {}

    template<typename U>
    allocator(const allocator<U>& alloc) throw() : _flags(alloc._flags) {}

    pointer address(reference x) const { return &x; }
    const_pointer address(const_reference x) const { return &x; }

    pointer allocate(size_type s, void const* = 0)
    {
        void* p;

        if (__builtin_expect(s > this->max_size(), false))
        {
            throw std::bad_alloc();
        }

        CUDA_CALL(cudaHostAlloc(&p, s * sizeof(T), _flags));

        return reinterpret_cast<pointer>(p);
    }

    void deallocate(pointer p, size_type) throw() // no no-throw guarantee (p may be a null pointer)
    {
        cudaFreeHost(reinterpret_cast<void *>(p));
    }

    size_type max_size() const throw()
    {
        return std::numeric_limits<size_t>::max() / sizeof(T);
    }

    void construct(pointer p, const T& val)
    {
        ::new((void *)p) T(val);
    }

    void destroy(pointer p)
    {
        p->~T();
    }

private:
    unsigned int _flags;
};

template<typename T>
inline bool operator==(const allocator<T>&, const allocator<T>&)
{
    return true;
}

template<typename T>
inline bool operator!=(const allocator<T>&, const allocator<T>&)
{
    return false;
}

} // namespace host
} // namespace cuda

#endif
