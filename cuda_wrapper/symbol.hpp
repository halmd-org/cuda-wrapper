/* CUDA device symbol
 *
 * Copyright (C) 2007  Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_SYMBOL_HPP
#define CUDA_SYMBOL_HPP

#include <cuda_runtime.h>

#ifndef __CUDACC__
# include <cuda_wrapper/error.hpp>
#endif

namespace cuda {

/**
 * CUDA device symbol constant
 */
template <typename T>
class symbol
{
public:
    typedef T value_type;
    typedef size_t size_type;

public:
    /**
     * initialize device symbol constant
     */
    symbol(value_type const& symbol)
      : ptr_(&symbol)
    {}

    /**
     * return element count of device symbol
     */
    size_type size() const
    {
        return 1;
    }

    /**
     * returns device pointer to device symbol
     */
    value_type const* data() const
    {
        return ptr_;
    }

private:
    /** device symbol pointer */
    value_type const* ptr_;
};


/**
 * CUDA device symbol vector
 */
template <typename T>
class symbol<T[]>
{
public:
    typedef T value_type;
    typedef size_t size_type;

public:
    /**
     * initialize device symbol vector
     */
    template <typename Array>
    symbol(Array const& array)
      : ptr_(array)
      , size_(sizeof(array) / sizeof(value_type))
    {}

    /**
     * return element count of device symbol
     */
    size_type size() const
    {
        return size_;
    }

    /**
     * returns device pointer to device symbol
     */
    value_type const* data() const
    {
        return ptr_;
    }

private:
    /** device symbol pointer */
    value_type const* ptr_;
    /** array size */
    size_type size_;
};

} // namespace cuda

#endif /* ! CUDA_SYMBOL_HPP */
