/*
 * Copyright (C) 2010  Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_TRAITS_HPP
#define CUDA_TRAITS_HPP

#include <cstddef>
#include <cuda.h>

namespace cuda {

#if (CUDA_VERSION >= 3020)
typedef std::size_t size_type;
#else
typedef unsigned int size_type;
#endif

} // namespace cuda

#endif /* ! CUDA_TRAITS_HPP */
