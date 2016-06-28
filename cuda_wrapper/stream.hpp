/* cuda_wrapper/stream.hpp
 *
 * Copyright (C) 2007  Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <cuda_runtime.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {

#if (CUDART_VERSION >= 1010)

/**
 * CUDA stream wrapper class
 */
class stream
{
private:
    struct container : boost::noncopyable
    {
        /**
         * creates a stream
         */
        container()
        {
            CUDA_CALL(cudaStreamCreate(&m_stream));
        }

        /**
         * destroys the stream
         */
        ~container() throw() // no-throw guarantee
        {
            cudaStreamDestroy(m_stream);
        }

        cudaStream_t m_stream;
    };

public:
    /**
     * creates a stream
     */
    stream() : m_stream(new container) {}

    /**
     * blocks until the device has completed all operations in the stream
     */
    void synchronize()
    {
        CUDA_CALL(cudaStreamSynchronize(m_stream->m_stream));
    }

    /**
     * checks if the device has completed all operations in the stream
     *
     * WARNING: this function will not detect kernel launch failures
     */
    bool query()
    {
        cudaError_t err = cudaStreamQuery(m_stream->m_stream);
        if (cudaSuccess == err)
            return true;
        else if (cudaErrorNotReady == err)
            return false;
        CUDA_ERROR(err);
    }

    /**
     * returns stream
     */
    cudaStream_t data() const
    {
        return m_stream->m_stream;
    }

private:
    boost::shared_ptr<container> m_stream;
};

#endif /* CUDART_VERSION >= 1010 */

} // namespace cuda

#endif /* ! CUDA_STREAM_HPP */
