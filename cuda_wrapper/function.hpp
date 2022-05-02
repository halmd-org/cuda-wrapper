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

#include <functional>

#include <cuda_runtime.h>

#ifndef __CUDA_ARCH__
# include <cuda_wrapper/error.hpp>
# include <cuda_wrapper/stream.hpp>
#endif

namespace cuda {

/**
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

    const void* f_;
    std::function<void (int*, int*)> occupancy_;

    dim3 grid_ = 0;
    dim3 block_ = 0;
    size_t shared_mem_ = 0;
    cudaStream_t stream_ = 0;

    int min_grid_size_ = -1;
    int max_block_size_ = -1;

    cudaFuncAttributes attr_ = {};

    cudaFuncAttributes* attributes()
    {
        if (attr_.binaryVersion == 0) {
            CUDA_CALL(cudaFuncGetAttributes(&attr_, f_));
        }
        return &attr_;
    }

public:
    function(T f) : f_(reinterpret_cast<void const*>(f))
    {
#ifdef __CUDACC__
        occupancy_ = std::function<void (int*, int*)>([&](int* min_grid_size, int* max_block_size) {
            cudaOccupancyMaxPotentialBlockSize(min_grid_size, max_block_size, f_);
        });
#endif
    }

    template <typename SMemSizeFunc>
    function(T f, SMemSizeFunc smem_size_func) : f_(reinterpret_cast<void const*>(f))
    {
#ifdef __CUDACC__
        occupancy_ = std::function<void (int*, int*)>([&, smem_size_func](int* min_grid_size, int* max_block_size) {
            cudaOccupancyMaxPotentialBlockSizeVariableSMem(min_grid_size, max_block_size, f_, smem_size_func);
        });
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

    void configure(dim3 const& grid, dim3 const& block, size_t shared_mem, stream& stream)
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
        // use additional nullptr element so the array can't become zero sized when the number of argument is zero
        void* p[] = {nullptr, static_cast<void*>(&args)...};
        CUDA_CALL(cudaLaunchKernel(f_, grid_, block_, p + 1, shared_mem_, stream_));
    }

    /**
     * binary architecture version for which the kernel was compiled
     */
    unsigned int binary_version()
    {
        return attributes()->binaryVersion;
    }

    /**
     * size of constant memory required by the kernel
     */
    size_t const_size_bytes()
    {
        return attributes()->constSizeBytes;
    }

    /**
     * size of local memory used by each thread
     */
    size_t local_size_bytes()
    {
        return attributes()->localSizeBytes;
    }

    /**
     * the maximum number of threads per block
     */
    unsigned int max_threads_per_block()
    {
        return attributes()->maxThreadsPerBlock;
    }

    /**
     * number of registers used by each thread
     */
    unsigned int num_regs()
    {
        return attributes()->numRegs;
    }

    /**
     * ptx architecture for which the kernerl was compiled
     */
    unsigned int ptx_version()
    {
        return attributes()->ptxVersion;
    }

    /**
     * size of shared memory per block
     */
    size_t shared_size_bytes()
    {
        return attributes()->sharedSizeBytes;
    }

    /**
     * grid size that achieves maximum occupancy
     */
    int min_grid_size()
    {
        if (min_grid_size_ < 0) {
            occupancy_(&min_grid_size_, &max_block_size_);
        }
        return min_grid_size_;
    }

    /**
     * block size that achieves maximum occupancy
     */
    int max_block_size()
    {
        if (max_block_size_ < 0) {
            occupancy_(&min_grid_size_, &max_block_size_);
        }
        return max_block_size_;
    }
#endif // ! __CUDACC__
};

} // namespace cuda

#endif // CUDA_FUNCTION_HPP
