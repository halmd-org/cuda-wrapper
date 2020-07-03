/* cuda_wrapper/event.hpp
 *
 * Copyright (C) 2007 Peter Colberg
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_EVENT_HPP
#define CUDA_EVENT_HPP

#include <memory>

#include <cuda.h>

#include <cuda_wrapper/error.hpp>
#include <cuda_wrapper/stream.hpp>

namespace cuda {

/**
 * CUDA event wrapper class
 */
class event
{
private:
    struct container
    {
        /**
         * make the class noncopyable by deleting the copy and assignment operator
         */
        container(const container&) = delete;
        container& operator=(const container&) = delete;

        /**
         * creates an event
         */
        container(unsigned int flags)
        {
            CU_CALL(cuEventCreate(&m_event, flags));
        }

        /**
         * destroys the event
         */
        ~container()
        {
            cuEventDestroy(m_event);
        }

        CUevent m_event;
    };

public:
    /**
     * creates an event with given flags
     */
    event(unsigned int flags = CU_EVENT_DEFAULT)
        : m_event(new container(flags)) {}

    /**
     * records an event
     *
     * after all preceding operations in the CUDA context have been completed
     */
    void record()
    {
        CU_CALL(cuEventRecord(m_event->m_event, 0));
    }

    /**
     * records an event
     *
     * after all preceding operations in the stream have been completed
     */
    void record(const stream& stream)
    {
        CU_CALL(cuEventRecord(m_event->m_event, stream.data()));
    }

    /**
     * blocks until the event has actually been recorded
     */
    void synchronize()
    {
        CU_CALL(cuEventSynchronize(m_event->m_event));
    }

    /**
     * checks if the event has actually been recorded
     *
     * WARNING: this function will not detect kernel launch failures
     */
    bool query()
    {
        CUresult res = cuEventQuery(m_event->m_event);
        if (res == CUDA_SUCCESS)
            return true;
        else if (res == CUDA_ERROR_NOT_READY)
            return false;
        CU_ERROR(res);
    }

    /**
     * computes the elapsed time between two events
     *
     * (in seconds with a resolution of around 0.5 microseconds)
     */
    float operator-(const event &start)
    {
        float time;
        CU_CALL(cuEventElapsedTime(&time, start.m_event->m_event, m_event->m_event));
        return (1.e-3f * time);
    }

    /**
     * returns event
     */
    CUevent data() const
    {
        return m_event->m_event;
    }

private:
    std::shared_ptr<container> m_event;
};

} // namespace cuda

#endif /* ! CUDA_EVENT_HPP */
