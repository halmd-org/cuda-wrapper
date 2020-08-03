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

#ifndef CUDA_MEMORY_DEVICE_VECTOR_HPP
#define CUDA_MEMORY_DEVICE_VECTOR_HPP

#include <memory>
#include <initializer_list>

#include <cuda_wrapper/copy.hpp>
#include <cuda_wrapper/detail/random_access_iterator.hpp>
#include <cuda_wrapper/iterator_category.hpp>
#include <cuda_wrapper/memory/device/allocator.hpp>

namespace cuda {
namespace memory {
namespace device {

/**
 * CUDA global device memory vector
 */
template <typename T>
class vector
{
public:
    typedef T value_type;
    typedef allocator<T> allocator_type;
    typedef size_t size_type;
    typedef ptrdiff_t difference_type;
    typedef value_type& reference;
    typedef value_type const& const_reference;
    typedef value_type* pointer;
    typedef value_type const* const_pointer;
    typedef detail::random_access_iterator<pointer, device_random_access_iterator_tag> iterator;
    typedef detail::random_access_iterator<const_pointer, device_random_access_iterator_tag> const_iterator;

private:
    class container
    {
    public:
        /**
         * make the class noncopyable by deleting the copy and assignment operator
         */
        container(container const&) = delete;
        container& operator=(container const&) = delete;

        /**
         * allocate global device memory
         */
        container(size_type size) : m_size(size), m_ptr(NULL)
        {
            m_ptr = allocator_type().allocate(m_size);
        }

        /**
         * free global device memory
         */
        ~container()
        {
            allocator_type().deallocate(m_ptr, m_size);
        }

        size_type size() const noexcept
        {
            return m_size;
        }

        operator pointer() noexcept
        {
            return m_ptr;
        }

        operator const_pointer() const noexcept
        {
            return m_ptr;
        }

    private:
        size_type m_size;
        pointer m_ptr;
    };

public:
    /** create an empty vector */
    vector() : m_size(0), m_mem(new container(m_size)) {}
    /** create a vector with n elements */
    explicit vector(size_type n) : m_size(n), m_mem(new container(m_size)) {}
    /** creates a vector with a copy of a range */
    template <class InputIterator>
    vector(InputIterator begin, InputIterator end) : m_size(end - begin), m_mem(new container(m_size))
    {
        copy(begin, end, this->begin());
    }

    /** copy constructor */
    vector(vector const& other) : m_size(other.m_size), m_mem(new container(m_size))
    {
        copy(other.begin(), other.end(), begin());
    }

    /** move constructor */
    vector(vector&& other) noexcept : vector()
    {
        swap(*this, other);
    }

    /** construct vector from initializer list */
    vector(std::initializer_list<T> list) : m_size(list.size()), m_mem(new container(m_size))
    {
        copy(list.begin(), list.end(), this->begin());
    }

    /** destructor */
    ~vector()
    {
        delete m_mem;
    }

    /** copy assignment */
    vector& operator=(vector other)
    {
        swap(*this, other);
        return *this;
    }

    /** move assignment */
    vector& operator=(vector&& other)
    {
        delete m_mem;

        m_size = other.m_size;
        m_mem = other.m_mem;

        other.m_size = 0;
        other.m_mem = nullptr;
        return *this;
    }

    /** initializer list assignment */
    vector& operator=(std::initializer_list<T> list)
    {
        delete m_mem;

        m_size = list.size();
        m_mem = new container(m_size);

        copy(list.begin(), list.end(), this->begin());
        return *this;
    }

    /**
     * returns the device allocator
     */
    allocator_type get_allocator() const noexcept
    {
        return allocator_type();
    }

    /**
     * checks whether the container is empty
     */
    bool empty() const noexcept
    {
        return this->begin() == this->end();
    }

    /**
     * returns element count of device vector
     */
    size_type size() const noexcept
    {
        return m_size;
    }

    /**
     * returns the maximum possible number of elements
     */
    size_type max_size() const noexcept
    {
        return allocator_type().max_size();
    }

    /**
     * returns capacity
     */
    size_type capacity() const noexcept
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
            delete m_mem;
            m_mem = new container(size);
        }
    }

    /**
     * swap device memory with vector
     */
    friend void swap(vector& first, vector& second) noexcept
    {
        using std::swap;

        swap(first.m_mem, second.m_mem);
        swap(first.m_size, second.m_size);
    }

    /**
     * returns device pointer to allocated device memory
     */
    pointer data() noexcept
    {
        return *m_mem;
    }

    operator pointer() noexcept
    {
        return *m_mem;
    }

    const_pointer data() const noexcept
    {
        return *m_mem;
    }

    operator const_pointer() const noexcept
    {
        return *m_mem;
    }

    iterator begin() noexcept
    {
        return iterator(*m_mem);
    }

    const_iterator begin() const noexcept
    {
        return const_iterator(*m_mem);
    }

    iterator end() noexcept
    {
        return iterator(*m_mem + m_size);
    }

    const_iterator end() const noexcept
    {
        return const_iterator(*m_mem + m_size);
    }

private:
    size_type m_size;
    container* m_mem;
};

} // namespace device
} // namespace memory
} // namespace cuda

#endif // CUDA_MEMORY_DEVICE_VECTOR_HPP
