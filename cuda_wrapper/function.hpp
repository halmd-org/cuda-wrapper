/* CUDA device function execution
 *
 * Copyright (C) 2020 Jaslo Ziska
 * Copyright (C) 2007 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_FUNCTION_HPP
#define CUDA_FUNCTION_HPP

#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
# include <cuda_wrapper/error.hpp>
# include <cuda_wrapper/stream.hpp>
#endif

namespace cuda {

/*
 * CUDA execution configuration
 */
struct config
{
public:
    /* grid dimensions */
    dim3 grid;
    /* block dimensions */
    dim3 block;
    /* FIXME store useful numbers (no. of threads per grid/block) */

    config() {}

    config(dim3 grid, dim3 block) : grid(grid), block(block)
    {
        /* FIXME store useful numbers (no. of threads per grid/block) */
    }

    size_t threads() const
    {
        return grid.y * grid.x * block.z * block.y * block.x;
    }

    size_t blocks_per_grid() const
    {
        return grid.y * grid.x;
    }

    size_t threads_per_block() const
    {
        return block.z * block.y * block.x;
    }
};

template <typename... Args>
class function;

/**
 * CUDA kernel execution wrapper for n-ary device function
 */
template <typename... Args>
class function<void (Args...)>
{
private:
    typedef void (*T)(Args...);

    const void *f_;

    dim3 grid_ = 0;
    dim3 block_ = 0;
    size_t shared_mem_ = 0;
    cudaStream_t stream_ = 0;

    int max_block_size_ = 0;
    int min_grid_size_ = 0;

    cudaFuncAttributes attr_;
public:
    function(T f) : f_(reinterpret_cast<const void *>(f)) {
#ifndef __CUDA_ARCH__
        CUDA_CALL(cudaFuncGetAttributes(&attr_, f_));
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSize(&min_grid_size_,
            &max_block_size_, f));
#endif
    }

    template <typename SMemSizeFun>
    function(T f, SMemSizeFun fun) : f_(reinterpret_cast<const void *>(f))
    {
#ifndef __CUDA_ARCH__
        CUDA_CALL(cudaFuncGetAttributes(&attr_, f_));
        CUDA_CALL(cudaOccupancyMaxPotentialBlockSizeVariableSMem(&min_grid_size_,
            &max_block_size_, f, fun));
#endif
    }

#ifndef __CUDACC__
    /**
     * configure execution parameters
     */
    void configure(dim3 const& grid, dim3 const& block, size_t shared_mem = 0)
    {
        grid_ = grid;
        block_ = block;
        shared_mem_ = shared_mem;
        stream_ = 0;
    }

    void configure(dim3 const& grid, dim3 const& block, stream& stream)
    {
        grid_ = grid;
        block_ = block;
        shared_mem_ = 0;
        stream_ = stream.data();
    }

    void configure(dim3 const& grid, dim3 const& block, size_t shared_mem,
        stream& stream)
    {
        grid_ = grid;
        block_ = block;
        shared_mem_ = shared_mem;
        stream_ = stream.data();
    }


    /**
     * execute kernel
     */
    void operator()(Args... args) const
    {
        void *p[] = {static_cast<void *>(&args)...};
        CUDA_CALL(cudaLaunchKernel(f_, grid_, block_, p, shared_mem_, stream_));
    }

    unsigned int binary_version() const
    {
        return attr_.binaryVersion;
    }

    size_t const_size_bytes() const
    {
        return attr_.constSizeBytes;
    }

    size_t local_size_bytes() const
    {
        return attr_.localSizeBytes;
    }

    unsigned int max_threads_per_block() const
    {
        return attr_.maxThreadsPerBlock;
    }

    unsigned int num_regs() const
    {
        return attr_.numRegs;
    }

    unsigned int ptx_version() const
    {
        return attr_.ptxVersion;
    }

    size_t shared_size_bytes() const
    {
        return attr_.sharedSizeBytes;
    }

    int max_block_size() const
    {
        return max_block_size_;
    }

    int min_grid_size() const
    {
        return min_grid_size_;
    }
#endif // __CUDACC__
};

} // namespace cuda

#endif // CUDA_FUNCTION_HPP
