/*
 * Copyright © 2009, 2012 Peter Colberg
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
