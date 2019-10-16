/* cuda_wrapper/thread.hpp
 *
 * Copyright (C) 2007  Peter Colberg
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

#include <cuda_runtime.h>
#ifndef NDEBUG
# include <cuda_profiler_api.h>
#endif

#include <cuda_wrapper/error.hpp>

namespace cuda {

class thread
{
public:
    /*
     * blocks until the device has completed all preceding requested tasks
     */
    static void synchronize()
    {
        CUDA_CALL(cudaThreadSynchronize());
    }

    /*
     * cleans up all runtime-related resources associated with calling thread
     */
    static void exit()
    {
#ifndef NDEBUG
        CUDA_CALL(cudaProfilerStop());      // flush profiling buffers
#endif
        CUDA_CALL(cudaThreadExit());
    }
};

} // namespace cuda

#endif /* ! CUDA_THREAD_HPP */
