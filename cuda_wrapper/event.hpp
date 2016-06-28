/* cuda_wrapper/event.hpp
 *
 * Copyright (C) 2007  Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_EVENT_HPP
#define CUDA_EVENT_HPP

#include <boost/shared_ptr.hpp>
#include <boost/utility.hpp>
#include <cuda_runtime.h>

#include <cuda_wrapper/error.hpp>
#include <cuda_wrapper/stream.hpp>

namespace cuda {

#if (CUDART_VERSION >= 1010)

/**
 * CUDA event wrapper class
 */
class event
{
private:
    struct container : boost::noncopyable
    {
        /**
         * creates an event
         */
        container()
        {
            CUDA_CALL(cudaEventCreate(&m_event));
        }

#if (CUDART_VERSION >= 2020)
        /**
         * creates an event with given flags
         */
        container(int flags)
        {
            CUDA_CALL(cudaEventCreateWithFlags(&m_event, flags));
        }
#endif

        /**
         * destroys the event
         */
        ~container() throw() // no-throw guarantee
        {
            cudaEventDestroy(m_event);
        }

        cudaEvent_t m_event;
    };

public:
    /**
     * creates an event
     */
    event() : m_event(new container) {}

#if (CUDART_VERSION >= 2020)
    /**
     * creates an event with given flags
     */
    event(int flags) : m_event(new container(flags)) {}
#endif

    /**
     * records an event
     *
     * after all preceding operations in the CUDA context have been completed
     */
    void record()
    {
        CUDA_CALL(cudaEventRecord(m_event->m_event, 0));
    }

    /**
     * records an event
     *
     * after all preceding operations in the stream have been completed
     */
    void record(const stream& stream)
    {
        CUDA_CALL(cudaEventRecord(m_event->m_event, stream.data()));
    }

    /**
     * blocks until the event has actually been recorded
     */
    void synchronize()
    {
        CUDA_CALL(cudaEventSynchronize(m_event->m_event));
    }

    /**
     * checks if the event has actually been recorded
     *
     * WARNING: this function will not detect kernel launch failures
     */
    bool query()
    {
        cudaError_t err = cudaEventQuery(m_event->m_event);
        if (cudaSuccess == err)
            return true;
        else if (cudaErrorNotReady == err)
            return false;
        CUDA_ERROR(err);
    }

    /**
     * computes the elapsed time between two events
     *
     * (in seconds with a resolution of around 0.5 microseconds)
     */
    float operator-(const event &start)
    {
        float time;
        CUDA_CALL(cudaEventElapsedTime(&time, start.m_event->m_event, m_event->m_event));
        return (1.e-3f * time);
    }

    /**
     * returns event
     */
    cudaEvent_t data() const
    {
        return m_event->m_event;
    }

private:
    boost::shared_ptr<container> m_event;
};

#endif /* CUDART_VERSION >= 1010 */

} // namespace cuda

#endif /* ! CUDA_EVENT_HPP */
