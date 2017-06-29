/* CUDA device function execution
 *
 * Copyright (C) 2007  Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#if !BOOST_PP_IS_ITERATING

    #ifndef CUDA_FUNCTION_HPP
    #define CUDA_FUNCTION_HPP

    #include <boost/preprocessor/iteration/iterate.hpp>
    #include <boost/preprocessor/repetition/enum_params.hpp>
    #include <boost/preprocessor/repetition/enum_binary_params.hpp>
    #include <boost/preprocessor/repetition/repeat.hpp>
    #include <cuda_runtime.h>

    #ifndef __CUDACC__
    # include <cstddef>
    # include <cuda_wrapper/error.hpp>
    # include <cuda_wrapper/stream.hpp>
    #endif

    /* maximum number of arguments passed to device functions */
    #ifndef CUDA_FUNCTION_MAX_ARGS
    #define CUDA_FUNCTION_MAX_ARGS 16
    #endif

    namespace cuda
    {

    /*
     * CUDA execution configuration
     */
    class config
    {
    public:
        /* grid dimensions */
        dim3 grid;
        /* block dimensions */
        dim3 block;
        /* FIXME store useful numbers (no. of threads per grid/block) */

        config()
        {
        }

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

    #ifndef __CUDACC__

    /**
     * configure execution parameters
     */
    __inline__ void configure(dim3 const& grid, dim3 const& block, size_t shared_mem = 0)
    {
        CUDA_CALL(cudaConfigureCall(grid, block, shared_mem, 0));
    }

    #if (CUDART_VERSION >= 1010)

    /**
     * configure execution parameters
     */
    __inline__ void configure(dim3 const& grid, dim3 const& block, stream& stream)
    {
        CUDA_CALL(cudaConfigureCall(grid, block, 0, stream.data()));
    }

    /**
     * configure execution parameters
     */
    __inline__ void configure(dim3 const& grid, dim3 const& block, size_t shared_mem, stream& stream)
    {
        CUDA_CALL(cudaConfigureCall(grid, block, shared_mem, stream.data()));
    }

    #endif /* CUDART_VERSION >= 1010 */

    #endif /* ! __CUDACC__ */

    template <typename T>
    class function;

    } // namespace cuda

    #define BOOST_PP_FILENAME_1 <cuda_wrapper/function.hpp>
    #define BOOST_PP_ITERATION_LIMITS (1, CUDA_FUNCTION_MAX_ARGS)
    #include BOOST_PP_ITERATE()

    #endif /* ! CUDA_FUNCTION_HPP */

#elif BOOST_PP_ITERATION_DEPTH() == 1

    #define CUDA_FUNCTION_ARGS BOOST_PP_FRAME_ITERATION(1)

    namespace cuda
    {

    /**
     * CUDA kernel execution wrapper for n-ary device function
     */
    template <BOOST_PP_ENUM_PARAMS(CUDA_FUNCTION_ARGS, typename T)>
    class function<void (BOOST_PP_ENUM_PARAMS(CUDA_FUNCTION_ARGS, T))>
    {
    public:
        typedef void T (BOOST_PP_ENUM_PARAMS(CUDA_FUNCTION_ARGS, T));

    public:
        function(T* f) : f_(reinterpret_cast<char const*>(f)), maxBlockSize_(0), minGridSize_(0)
        {
    #if defined(__CUDACC__) && CUDART_VERSION >= 6500
            cudaOccupancyMaxPotentialBlockSize(&minGridSize_, &maxBlockSize_, *f);
    #endif
        }
        template<typename SMemSizeFun>
        function(T* f, SMemSizeFun fun) : f_(reinterpret_cast<const char*>(f)), maxBlockSize_(0), minGridSize_(0)
        {
    #if defined(__CUDACC__) && CUDART_VERSION >= 6500
            cudaOccupancyMaxPotentialBlockSizeVariableSMem(&minGridSize_, &maxBlockSize_, *f, fun);
    #endif
        }

    #ifndef __CUDACC__

        /**
         * execute kernel
         */
        void operator()(BOOST_PP_ENUM_BINARY_PARAMS(CUDA_FUNCTION_ARGS, T, x)) const
        {
            // properly align CUDA device function arguments
            struct offset
            {
                #define DECL_ARG(z, n, arg) T##n arg##n;
                BOOST_PP_REPEAT(CUDA_FUNCTION_ARGS, DECL_ARG, x)
                #undef DECL_ARG
            };
            offset* __offset = 0;
            // push aligned arguments onto CUDA execution stack
            #define DECL_ARG(z, n, offset) CUDA_CALL(cudaSetupArgument(&x##n, sizeof(T##n), reinterpret_cast<size_t>(&offset->x##n)));
            BOOST_PP_REPEAT(CUDA_FUNCTION_ARGS, DECL_ARG, __offset)
            #undef DECL_ARG
            // launch CUDA device function
            CUDA_CALL(cudaLaunch(f_));
        }

    #if CUDART_VERSION >= 4010

        unsigned int binary_version() const
        {
            cudaFuncAttributes attr;
            CUDA_CALL( cudaFuncGetAttributes(&attr, f_) );
            return attr.binaryVersion;
        }

        std::size_t const_size_bytes() const
        {
            cudaFuncAttributes attr;
            CUDA_CALL( cudaFuncGetAttributes(&attr, f_) );
            return attr.constSizeBytes;
        }

        std::size_t local_size_bytes() const
        {
            cudaFuncAttributes attr;
            CUDA_CALL( cudaFuncGetAttributes(&attr, f_) );
            return attr.localSizeBytes;
        }

        unsigned int max_threads_per_block() const
        {
            cudaFuncAttributes attr;
            CUDA_CALL( cudaFuncGetAttributes(&attr, f_) );
            return attr.maxThreadsPerBlock;
        }

        unsigned int num_regs() const
        {
            cudaFuncAttributes attr;
            CUDA_CALL( cudaFuncGetAttributes(&attr, f_) );
            return attr.numRegs;
        }

        unsigned int ptx_version() const
        {
            cudaFuncAttributes attr;
            CUDA_CALL( cudaFuncGetAttributes(&attr, f_) );
            return attr.ptxVersion;
        }

        std::size_t shared_size_bytes() const
        {
            cudaFuncAttributes attr;
            CUDA_CALL( cudaFuncGetAttributes(&attr, f_) );
            return attr.sharedSizeBytes;
        }

        int max_block_size() const
        {
            return maxBlockSize_;
        }

        int min_grid_size() const
        {
            return minGridSize_;
        }

    #endif /* CUDART_VERSION >= 4010 */

    #endif /* ! __CUDACC__ */

    private:
        char const* f_;
        int maxBlockSize_;
        int minGridSize_;
    };

    } // namespace cuda

    #undef CUDA_FUNCTION_ARGS

#endif /* !BOOST_PP_IS_ITERATING */
