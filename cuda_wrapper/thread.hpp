/* cuda_wrapper/thread.hpp
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
 * CUDA thread management
 */

#ifndef CUDA_THREAD_HPP
#define CUDA_THREAD_HPP

#include <cuda.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {
namespace thread {

/*
 * blocks until the device has completed all preceding requested tasks
 */
inline void synchronize()
{
    CU_CALL(cuCtxSynchronize());
}

} // namespace thread
} // namespace cuda

#endif /* ! CUDA_THREAD_HPP */
