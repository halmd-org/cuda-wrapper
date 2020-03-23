/* CUDA global device memory vector
 *
 * Copyright (C) 2007 Peter Colberg
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_VECTOR_HPP
#define CUDA_VECTOR_HPP

#include <memory>

#include <cuda_wrapper/detail/random_access_iterator.hpp>
#include <cuda_wrapper/iterator_category.hpp>
#include <cuda_wrapper/allocator.hpp>

namespace cuda {

/**
 * CUDA global device memory vector
 */
template <typename T>
class vector
{
public:
    typedef vector<T> vector_type;
    typedef T value_type;
    typedef T* pointer;
    typedef T const* const_pointer;
    typedef detail::random_access_iterator<T*, device_random_access_iterator_tag> iterator;
    typedef detail::random_access_iterator<T const*, device_random_access_iterator_tag> const_iterator;
    typedef size_t size_type;

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
         * allocate global device memory
         */
        container(size_type size) : m_size(size), m_ptr(NULL)
        {
            m_ptr = allocator<value_type>().allocate(m_size);
        }

        /**
         * free global device memory
         */
        ~container() throw()
        {
            allocator<value_type>().deallocate(m_ptr, m_size);
        }

        size_type size() const
        {
            return m_size;
        }

        operator pointer()
        {
            return m_ptr;
        }

        operator const_pointer() const
        {
            return m_ptr;
        }

    private:
        size_type m_size;
        pointer m_ptr;
    };

public:
    /**
     * initialize device vector of given size
     */
    vector(size_type size = 0) : m_mem(new container(size)), m_size(size) {}

    /**
     * returns element count of device vector
     */
    size_type size() const
    {
        return m_size;
    }

    /**
     * returns capacity
     */
    size_type capacity() const
    {
        return m_mem->size();
    }

    /**
     * resize element count of device vector
     */
    void resize(size_type size)
    {
        this->reserve(size);
        m_size = size;
    }

    /**
     * allocate sufficient memory for specified number of elements
     */
    void reserve(size_type size)
    {
        if (size > m_mem->size()) {
            m_mem.reset();
            m_mem.reset(new container(size));
        }
    }

    /**
     * swap device memory with vector
     */
    void swap(vector_type& v)
    {
        m_mem.swap(v.m_mem);
        std::swap(m_size, v.m_size);
    }

    /**
     * returns device pointer to allocated device memory
     */
    pointer data()
    {
        return *m_mem;
    }

    operator pointer()
    {
        return *m_mem;
    }

    const_pointer data() const
    {
        return *m_mem;
    }

    operator const_pointer() const
    {
        return *m_mem;
    }

    iterator begin()
    {
        return iterator(*m_mem);
    }

    const_iterator begin() const
    {
        return const_iterator(*m_mem);
    }

    iterator end()
    {
        return iterator(*m_mem + m_size);
    }

    const_iterator end() const
    {
        return const_iterator(*m_mem + m_size);
    }

private:
    std::shared_ptr<container> m_mem;
    size_type m_size;
};

} // namespace cuda

#endif /* CUDA_VECTOR_HPP */
