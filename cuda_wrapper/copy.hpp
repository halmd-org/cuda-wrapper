/*
 * Copyright © 2008-2010, 2012 Peter Colberg
 *
 * This file is part of HALMD.
 *
 * HALMD is free software: you can redistribute it and/or modify
 * it under the terms of the GNU General Public License as published by
 * the Free Software Foundation, either version 3 of the License, or
 * (at your option) any later version.
 *
 * This program is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU General Public License for more details.
 *
 * You should have received a copy of the GNU General Public License
 * along with this program.  If not, see <http://www.gnu.org/licenses/>.
 */

#ifndef CUDA_WRAPPER_COPY_HPP
#define CUDA_WRAPPER_COPY_HPP

#include <cuda_runtime.h>
#include <iterator>
#include <type_traits>

#include <cuda_wrapper/iterator_category.hpp>
#include <cuda_wrapper/stream.hpp>

namespace cuda {

/**
 * Copy from host memory area to device memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value
    && !std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpy(
        &*result
      , &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyHostToDevice
    ) );
    return result + size;
}

/**
 * Copy from device memory area to host memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value
    && !std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpy(
        &*result
      , &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyDeviceToHost
    ) );
    return result + size;
}

/**
 * Copy from device memory area to device memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpy(
        &*result
      , &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyDeviceToDevice
    ) );
    return result + size;
}

/**
 * Copy from host memory area to host memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value
    && !std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , std::random_access_iterator_tag
    >::value
    && !std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpy(
        &*result
      , &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyHostToHost
    ) );
    return result + size;
}

#if (CUDART_VERSION >= 1010)

/**
 * Asynchronous copy from host memory area to device memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , host_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result, stream& stream)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpyAsync(
        &*result, &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyHostToDevice, stream.data()
    ) );
    return result + size;
}

/**
 * Asynchronous copy from device memory area to host memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , host_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result, stream& stream)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpyAsync(
        &*result
      , &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyDeviceToHost
      , stream.data()
    ) );
    return result + size;
}

/**
 * Asynchronous copy from device memory area to device memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result, stream& stream)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpyAsync(
        &*result
      , &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyDeviceToDevice
      , stream.data()
    ) );
    return result + size;
}

/**
 * Asynchronous copy from host memory area to host memory area.
 */
template <typename InputIterator, typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<InputIterator>::iterator_category
      , host_random_access_iterator_tag
    >::value
    && std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , host_random_access_iterator_tag
    >::value
    && std::is_same<
        typename std::iterator_traits<InputIterator>::value_type
      , typename std::iterator_traits<OutputIterator>::value_type
    >::value
  , OutputIterator>::type copy(InputIterator first, InputIterator last, OutputIterator result, stream& stream)
{
    typename std::iterator_traits<InputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemcpyAsync(
        &*result
      , &*first
      , size * sizeof(typename std::iterator_traits<InputIterator>::value_type)
      , cudaMemcpyHostToHost
      , stream.data()
    ) );
    return result + size;
}

#endif /* (CUDART_VERSION >= 1010) */

/**
 * Fill device memory area with constant byte value.
 */
template <typename OutputIterator>
inline typename std::enable_if<
    std::is_convertible<
        typename std::iterator_traits<OutputIterator>::iterator_category
      , device_random_access_iterator_tag
    >::value
  , void>::type memset(OutputIterator first, OutputIterator last, unsigned char value)
{
    typename std::iterator_traits<OutputIterator>::difference_type size = last - first;
    CUDA_CALL( cudaMemset(
        &*first
      , value
      , size * sizeof(typename std::iterator_traits<OutputIterator>::value_type)
    ) );
}

} // namespace cuda

#endif /* ! CUDA_WRAPPER_COPY_HPP */
