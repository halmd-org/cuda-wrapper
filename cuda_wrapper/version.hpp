/*
 * Copyright © 2009, 2012 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_WRAPPER_VERSION_HPP
#define CUDA_WRAPPER_VERSION_HPP

#include <cuda_runtime.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {

#if CUDART_VERSION >= 2020

/**
 * Returns version number of CUDA driver library.
 */
inline int driver_version()
{
    int version;
    CUDA_CALL(cudaDriverGetVersion(&version));
    return version;
}

/**
 * Returns version number of CUDA runtime library.
 */
inline int runtime_version()
{
    int version;
    CUDA_CALL(cudaRuntimeGetVersion(&version));
    return version;
}

#endif /* CUDART_VERSION >= 2020 */

} // namespace cuda

#endif /* ! CUDA_WRAPPER_VERSION_HPP */
