/* cuda_wrapper/device.hpp
 *
 * Copyright (C) 2020 Jaslo Ziska
 * Copyright (C) 2007 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_DEVICE_HPP
#define CUDA_DEVICE_HPP

#include <string>

#include <cuda.h>
#include <cuda_runtime.h> // for dim3
#ifndef NDEBUG
# include <cudaProfiler.h>
#endif

#include <cuda_wrapper/error.hpp>

// workaround for warnings with GCC 7
#ifdef minor
# undef minor
#endif
#ifdef major
# undef major
#endif

namespace cuda {

/**
 * CUDA device management
 */
class device
{
private:
    int num_ = -1;
    CUdevice dev_;

public:
    device()
    {
        CU_CALL(cuInit(0));
    }

    ~device() throw() // no-throw guarantee
    {
        // just like remove, but non-throwing
        if (num_ >= 0) {
            cuCtxPopCurrent(NULL);
            cuDevicePrimaryCtxRelease(dev_);
        }
    }

    /**
     * returns number of devices available for execution
     */
    static int count()
    {
        int count;
        CU_CALL(cuDeviceGetCount(&count));
        return count;
    }

    /**
     * returns true if device is busy, false otherwise
     */
    static bool active(int num)
    {
        CUdevice dev;
        unsigned int flags;
        int active;

        CU_CALL(cuDeviceGet(&dev, num));
        CU_CALL(cuDevicePrimaryCtxGetState(dev, &flags, &active));
        return active;
    }

    /**
     * set device on which the active host thread executes device code
     */
    void set(int num)
    {
        // return immediately if device is the same as before
        if (num == num_)
            return;
        // remove old device (if necessary)
        remove();

        CUcontext ctx;
        CU_CALL(cuDeviceGet(&dev_, num));
        CU_CALL(cuDevicePrimaryCtxRetain(&ctx, dev_));
        CU_CALL(cuCtxPushCurrent(ctx));

        num_ = num;
    }

    /*
     * remove the current device
     */
    void remove()
    {
        if (num_ >= 0) {
            CU_CALL(cuCtxPopCurrent(NULL));
            CU_CALL(cuDevicePrimaryCtxRelease(dev_));
            num_ = -1;
        }
    }

    /**
     * get device on which the active host thread executes device code
     */
    int get() const
    {
        return num_;
    }

    /*
     * cleans up all runtime-related resources associated with calling thread
     */
    void reset()
    {
#ifndef NDEBUG
        CU_CALL(cuProfilerStop()); // flush profiling buffers
#endif
        // only call remove if device was set before
        if (num_ >= 0) {
            CU_CALL(cuDevicePrimaryCtxReset(dev_));
            num_ = -1;
        }
    }

    /**
     * CUDA device properties
     */
    class properties
    {
    private:
        /*
         * retrieve the information of attribute attr
         */
        int get_attribute(CUdevice_attribute attr) const
        {
            int num;
            CU_CALL(cuDeviceGetAttribute(&num, attr, dev));
            return num;
        }

        const static size_t BUFFER_NAME_LENGTH = 100;
        CUdevice dev;

    public:
        /**
         * retrieve properties of given device
         */
        properties(int ordinal)
        {
            CU_CALL(cuDeviceGet(&dev, ordinal));
        }

        /**
         * ASCII string identifying the device
         */
        std::string name() const
        {
            char name[BUFFER_NAME_LENGTH];
            CU_CALL(cuDeviceGetName(name, BUFFER_NAME_LENGTH, dev));
            return name;
        }

        /**
         * total amount of global memory available on the device in bytes
         */
        size_t total_global_mem() const
        {
            size_t num;
            CU_CALL(cuDeviceTotalMem(&num, dev));
            return num;
        }

        /**
         * total amount of shared memory available per block in bytes
         */
        size_t shared_mem_per_block() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_MAX_SHARED_MEMORY_PER_BLOCK);
        }

        /**
         * total number of registers available per block
         */
        size_t regs_per_block() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_MAX_REGISTERS_PER_BLOCK);
        }

        /**
         * wrap size
         */
        size_t warp_size() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_WARP_SIZE);
        }

        /**
         * maximum allowed memory allocation pitch
         */
        size_t mem_pitch() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_MAX_PITCH);
        }

        /**
         * maximum number of threads per block
         */
        unsigned int max_threads_per_block() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_BLOCK);
        }

        /**
         * maximum sizes of each dimension of a block
         */
        dim3 max_threads_dim() const
        {
            int x = get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_X);
            int y = get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Y);
            int z = get_attribute(CU_DEVICE_ATTRIBUTE_MAX_BLOCK_DIM_Z);
            return dim3(x, y, z);
        }

        /**
         * maximum sizes of each dimension of a grid
         */
        dim3 max_grid_size() const
        {
            int x = get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_X);
            int y = get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Y);
            int z = get_attribute(CU_DEVICE_ATTRIBUTE_MAX_GRID_DIM_Z);
            return dim3(x, y, z);
        }

        /**
         * total amount of constant memory available on the device in bytes
         */
        size_t total_const_mem() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_TOTAL_CONSTANT_MEMORY);
        }

        /**
         * major revision number of device's compute capatibility
         */
        unsigned int major() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MAJOR);
        }

        /**
         * minor revision number of device's compute capatibility
         */
        unsigned int minor() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_COMPUTE_CAPABILITY_MINOR);
        }

        /**
         * clock frequency in kHz
         */
        unsigned int clock_rate() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_CLOCK_RATE);
        }

        /**
         * texture alignment requirement
         */
        size_t texture_alignment() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_TEXTURE_ALIGNMENT);
        }

        /**
         * asynchronous kernel and memory operations capability
         */
        int device_overlap() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_GPU_OVERLAP);
        }

        /**
         * number of multiprocessors
         */
        int multi_processor_count() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_MULTIPROCESSOR_COUNT);
        }

        /**
         * maximum resident threads per multiprocessor
         */
        size_t max_threads_per_multi_processor() const
        {
            return get_attribute(CU_DEVICE_ATTRIBUTE_MAX_THREADS_PER_MULTIPROCESSOR);
        }
    };

    // forward declerations to make cuda::device:: usable like a namespace
    template <typename T>
    class vector;

    template <typename T>
    struct allocator;
};

} // namespace cuda

#endif /* ! CUDA_DEVICE_HPP */
