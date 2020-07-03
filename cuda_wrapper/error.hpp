/* cuda_wrapper/error.hpp
 *
 * Copyright (C) 2007 Peter Colberg
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

/*
 * CUDA runtime error checking
 */

#ifndef CUDA_ERROR_HPP
#define CUDA_ERROR_HPP

#include <exception>

#include <cuda.h>
#include <cuda_runtime.h>

#define CUDA_ERROR(err) throw cuda::error(err)

#define CUDA_CALL(x)                                                           \
    do {                                                                       \
        if (static_cast<cudaError_t>(x) != cudaSuccess) {                      \
            CUDA_ERROR(x);                                                     \
        }                                                                      \
    } while(0)

#define CU_ERROR(res) throw cuda::error(res)

#define CU_CALL(x)                                                             \
    do {                                                                       \
        if (static_cast<CUresult>(x) != CUDA_SUCCESS) {                        \
            CU_ERROR(x);                                                       \
        }                                                                      \
    } while(0)


namespace cuda {

    /**
 * CUDA error handling
 */
class error : public std::exception
{
public:
    // CUDA runtime or CUDA driver error
    const int err;
    const bool runtime;

    error(cudaError_t err) : err(err), runtime(true) {}
    error(CUresult err): err(err), runtime(false) {}

    /**
     * returns a message string for either CUDA runtime or CUDA driver error
     */
    char const* what() const throw()
    {
        if (runtime) {
            return cudaGetErrorString(static_cast<cudaError_t>(err));
        } else {
            char const* str;
            cuGetErrorString(static_cast<CUresult>(err), &str);
            return str;
        }
    }
};

} // namespace cuda

#endif /* ! CUDA_ERROR_HPP */
