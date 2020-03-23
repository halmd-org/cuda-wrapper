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

#include <cuda.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {

#if (CUDA_VERSION >= 1010)

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
            CU_CALL(cuStreamCreate(&m_stream, CU_STREAM_DEFAULT));
        }

        /**
         * destroys the stream
         */
        ~container() throw() // no-throw guarantee
        {
            cuStreamDestroy(m_stream);
        }

        CUstream m_stream;
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
        CU_CALL(cuStreamSynchronize(m_stream->m_stream));
    }

    /**
     * checks if the device has completed all operations in the stream
     *
     * WARNING: this function will not detect kernel launch failures
     */
    bool query()
    {
        CUresult res = cuStreamQuery(m_stream->m_stream);
        if (res == CUDA_SUCCESS)
            return true;
        else if (res == CUDA_ERROR_NOT_READY)
            return false;
        else
            CU_ERROR(res);
    }

    /**
     * returns stream
     */
    CUstream data() const
    {
        return m_stream->m_stream;
    }

private:
    boost::shared_ptr<container> m_stream;
};

#endif /* CUDA_VERSION >= 1010 */

} // namespace cuda

#endif /* ! CUDA_STREAM_HPP */
