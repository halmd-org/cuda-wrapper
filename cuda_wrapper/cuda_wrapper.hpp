/* cuda_wrapper/cuda_wrapper.hpp
 *
 * Copyright (C) 2013 Felix Höfling
 * Copyright (C) 2007 Peter Colberg
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

/*
 * CUDA runtime API wrapper classes
 */

#ifndef CUDA_WRAPPER_HPP
#define CUDA_WRAPPER_HPP

/* Disable warning for CUDA 5.5 headers emitted by Clang ≤ 3.3:
 *   /usr/local/cuda-5.5/include/cuda_runtime.h:225:33: warning: function
 *   'cudaMallocHost' is not needed and will not be emitted [-Wunneeded-internal-declaration]
 *
 * The CUDA runtime API version cannot be checked at this stage since the macro
 * CUDART_VERSION is not yet defined.
 */
#if (defined(__clang__) && __clang_major__ == 3 && __clang_minor__ <= 3)
# pragma GCC diagnostic push
# pragma GCC diagnostic ignored "-Wunneeded-internal-declaration"
# include <cuda_runtime.h>
# pragma GCC diagnostic pop
#else
# include <cuda_runtime.h>
#endif

/*
 * C++ wrappers requiring runtime functionality (e.g. exceptions)
 */

#ifndef __CUDACC__
# include <cuda_wrapper/allocator.hpp>
# include <cuda_wrapper/copy.hpp>
# include <cuda_wrapper/deprecated/copy.hpp>
# include <cuda_wrapper/device.hpp>
# include <cuda_wrapper/error.hpp>
# include <cuda_wrapper/event.hpp>
# include <cuda_wrapper/host/allocator.hpp>
# include <cuda_wrapper/host/vector.hpp>
# include <cuda_wrapper/stream.hpp>
# include <cuda_wrapper/thread.hpp>
# include <cuda_wrapper/vector.hpp>
# include <cuda_wrapper/version.hpp>
#endif /* ! __CUDACC__ */

/*
 * C++ wrappers *not* requiring runtime functionality
 */

#include <cuda_wrapper/function.hpp>
#include <cuda_wrapper/symbol.hpp>
#include <cuda_wrapper/texture.hpp>

#endif /* ! CUDA_WRAPPER_HPP */
