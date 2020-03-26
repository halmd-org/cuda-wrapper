/*
 * Copyright (C) 2009, 2012 Peter Colberg
 * Copyright (C) 2020       Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_WRAPPER_VERSION_HPP
#define CUDA_WRAPPER_VERSION_HPP

#include <cuda.h>
#include <cuda_runtime.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {

/**
 * Returns latest version of CUDA supported by the driver.
 */
inline int driver_version()
{
    int version;
    CU_CALL(cuDriverGetVersion(&version));
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

} // namespace cuda

#endif /* ! CUDA_WRAPPER_VERSION_HPP */
