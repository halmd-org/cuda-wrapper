/* cuda_wrapper/stream.hpp
 *
 * Copyright (C) 2007 Peter Colberg
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_STREAM_HPP
#define CUDA_STREAM_HPP

#include <memory>

#include <cuda.h>

#include <cuda_wrapper/error.hpp>

namespace cuda {

/**
 * CUDA stream wrapper class
 */
class stream
{
private:
    class container
    {
    public:
        /**
         * make the class noncopyable by deleting the copy and assignment operator
         */
        container(const container&) = delete;
        container& operator=(const container&) = delete;

        /**
         * creates a stream
         */
        container(unsigned int flags)
        {
            CU_CALL(cuStreamCreate(&m_stream, flags));
        }

        /**
         * destroys the stream
         */
        ~container()
        {
            cuStreamDestroy(m_stream);
        }

        CUstream m_stream;
    };

public:
    /**
     * creates a stream with given flags
     */
    stream(unsigned int flags = CU_STREAM_DEFAULT)
        : m_stream(new container(flags)) {}

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
     * attach memory to a stream asynchronously
     */
    template <typename T>
    void attach(T *ptr, size_t s = 0, unsigned int flags = CU_MEM_ATTACH_SINGLE)
    {
        CU_CALL(cuStreamAttachMemAsync(m_stream->m_stream, reinterpret_cast<CUdeviceptr>(ptr), s * sizeof(T), flags));
    }

    /**
     * returns stream
     */
    CUstream data() const
    {
        return m_stream->m_stream;
    }

private:
    std::shared_ptr<container> m_stream;
};

} // namespace cuda

#endif /* ! CUDA_STREAM_HPP */
