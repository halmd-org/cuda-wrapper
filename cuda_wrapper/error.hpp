/* cuda_wrapper/error.hpp
 *
 * Copyright (C) 2007  Peter Colberg
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

#include <cuda_runtime.h>
#include <exception>


#define CUDA_ERROR(err) throw cuda::error(err)

#define CUDA_CALL(x)                                                        \
    do {                                                                \
        cudaError_t err;                                                \
        if (cudaSuccess != (err = x)) {                                        \
            CUDA_ERROR(err);                                                \
        }                                                                \
    } while(0)


namespace cuda {

/*
 * CUDA error handling
 */
class error : public std::exception
{
public:
    /* CUDA error */
    const cudaError_t err;

    error(cudaError_t err): err(err)
    {
    }

    /*
     * returns a message string for the CUDA error
     */
    const char* what() const throw()
    {
        return cudaGetErrorString(err);
    }
};

} // namespace cuda

#endif /* ! CUDA_ERROR_HPP */
