/*
 * Copyright (C) 2007, 2012 Peter Colberg
 * Copyright (C) 2020       Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_MEMORY_MANAGED_VECTOR_HPP
#define CUDA_MEMORY_MANAGED_VECTOR_HPP

#include <vector>

#include <cuda.h>

#include <cuda_wrapper/detail/random_access_iterator.hpp>
#include <cuda_wrapper/device.hpp>
#include <cuda_wrapper/iterator_category.hpp>
#include <cuda_wrapper/memory/managed/allocator.hpp>
#include <cuda_wrapper/stream.hpp>

namespace cuda {
namespace memory {
namespace managed {

/**
 * CUDA managed memory vector
 *
 * Custom STL like vector class as described in: `https://en.cppreference.com/w/cpp/container/vector`
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
    typedef detail::random_access_iterator<pointer, managed_random_access_iterator_tag> iterator;
    typedef detail::random_access_iterator<const_pointer, managed_random_access_iterator_tag> const_iterator;

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
        container(size_type size) : size_(size), ptr_(NULL)
        {
            ptr_ = allocator_type().allocate(size_);
        }

        /**
         * free global device memory
         */
        ~container()
        {
            allocator_type().deallocate(ptr_, size_);
        }

        size_type size() const noexcept
        {
            return size_;
        }

        operator pointer() noexcept
        {
            return ptr_;
        }

        operator const_pointer() const noexcept
        {
            return ptr_;
        }

    private:
        size_type size_;
        pointer ptr_;
    };

    size_type size_;
    container* mem_;

public:
    /** create an empty vector */
    vector() : size_(0), mem_(new container(size_)) {}

    /** create vector with `count` copies of `value` */
    vector(size_type count, const T& value) : size_(count), mem_(new container(size_))
    {
        for (size_t i = 0; i < size_; ++i) {
            (*mem_)[i] = value;
        }
    }

    /** create a vector with `count` elements */
    explicit vector(size_type count) : size_(count), mem_(new container(size_)) {}

    /** creates a vector with a copy of a range */
    template <class InputIterator, typename = typename std::enable_if<std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category, std::random_access_iterator_tag>::value
    >::type>
    vector(InputIterator begin, InputIterator end) : size_(end - begin), mem_(new container(size_))
    {
        ::cuda::copy(begin, end, this->begin());
    }

    /** copy constructor */
    vector(vector const& other) : size_(other.size_), mem_(new container(size_))
    {
        ::cuda::copy(other.begin(), other.end(), begin());
    }

    /** move constructor */
    vector(vector&& other) noexcept : vector()
    {
        swap(*this, other);
    }

    /** construct vector from initializer list */
    vector(std::initializer_list<T> list) : size_(list.size()), mem_(new container(size_))
    {
        ::cuda::copy(list.begin(), list.end(), this->begin());
    }

    /** destructor */
    ~vector()
    {
        delete mem_;
    }

    /** copy and move assignment
     *
     *  We use the copy-and-swap idiom and ensure that the operations are
     *  self-safe and that the move assignment doesn't leak resources.
     */
    vector& operator=(vector other)
    {
        vector temp;
        swap(temp, other);             // 'other' is empty now
        swap(*this, temp);
        return *this;
    }

    /** initializer list assignment */
    vector& operator=(std::initializer_list<T> list)
    {
        delete mem_;

        size_ = list.size();
        mem_ = new container(size_);

        ::cuda::copy(list.begin(), list.end(), this->begin());
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
        return size_;
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
        return mem_->size();
    }

    /**
     * resize element count of device vector
     */
    void resize(size_type size)
    {
        this->reserve(size);
        size_ = size;
    }

    /**
     * allocate sufficient memory for specified number of elements
     */
    void reserve(size_type size)
    {
        if (size > mem_->size()) {
            delete mem_;
            mem_ = new container(size);
        }
    }

    /**
     * swap device memory with vector
     */
    friend void swap(vector& first, vector& second) noexcept
    {
        using std::swap;

        swap(first.mem_, second.mem_);
        swap(first.size_, second.size_);
    }

    reference front()
    {
        return **mem_;
    }

    const_reference front() const
    {
        return **mem_;
    }

    reference back()
    {
        return (*mem_)[mem_->size_];
    }

    const_reference back() const
    {
        return (*mem_)[mem_->size_];
    }

    /**
     * returns device pointer to allocated device memory
     */
    pointer data() noexcept
    {
        return *mem_;
    }

    operator pointer() noexcept
    {
        return *mem_;
    }

    const_pointer data() const noexcept
    {
        return *mem_;
    }

    operator const_pointer() const noexcept
    {
        return *mem_;
    }

    iterator begin() noexcept
    {
        return iterator(*mem_);
    }

    const_iterator begin() const noexcept
    {
        return const_iterator(*mem_);
    }

    iterator end() noexcept
    {
        return iterator(*mem_ + size_);
    }

    const_iterator end() const noexcept
    {
        return const_iterator(*mem_ + size_);
    }

#if CUDA_VERSION >= 8000
    /**
     * Advise about the usage pattern of the vector
     */
    void advise(CUmem_advise advice, ::cuda::device const& device)
    {
        CU_CALL(cuMemAdvise(reinterpret_cast<CUdeviceptr>(data()), size_, advice, device.data()));
    }
    /**
     * Advise about the usage pattern of the vector, overloaded for usage with cuda::device::CPU
     */
    void advise(CUmem_advise advice, ::cuda::device::cpu)
    {
        CU_CALL(cuMemAdvise(reinterpret_cast<CUdeviceptr>(data()), size_, advice, CU_DEVICE_CPU));
    }

    /**
     * Prefetch the vector to the specified device
     */
    void prefetch_async(::cuda::device const& device) const
    {
        CU_CALL(cuMemPrefetchAsync(
            reinterpret_cast<CUdeviceptr>(data()), size_, device.data(), 0));
    }
    /**
     * Prefetch the vector to the specified device
     */
    void prefetch_async(::cuda::device const& device, stream const& stream) const
    {
        CU_CALL(cuMemPrefetchAsync(reinterpret_cast<CUdeviceptr>(data()), size_, device.data(), stream.data()));
    }

    /**
     * Prefetch the vector to the specified device, overloaded for usage with cuda::device::CPU
     */
    void prefetch_async(::cuda::device::cpu) const
    {
        CU_CALL(cuMemPrefetchAsync(reinterpret_cast<CUdeviceptr>(data()), size_, CU_DEVICE_CPU, 0));
    }
    /**
     * Prefetch the vector to the specified device, overloaded for usage with cuda::device::CPU
     */
    void prefetch_async(::cuda::device::cpu, stream const& stream) const
    {
        CU_CALL(cuMemPrefetchAsync(reinterpret_cast<CUdeviceptr>(data()), size_, CU_DEVICE_CPU, stream.data()));
    }
#endif
};

} // namespace managed
} // namespace memory

// make the managed memory vector the default
using memory::managed::vector;
} // namespace cuda

#endif // CUDA_MEMORY_MANAGED_VECTOR_HPP
