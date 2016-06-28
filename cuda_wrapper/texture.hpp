/*
 * Copyright © 2013      Felix Höfling
 * Copyright © 2007-2012 Peter Colberg
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#ifndef CUDA_TEXTURE_HPP
#define CUDA_TEXTURE_HPP

#include <cuda_runtime.h>

#ifndef __CUDACC__
# include <cuda_wrapper/error.hpp>
# include <cuda_wrapper/vector.hpp>
#endif

/*
 * CUDA texture management
 */

namespace cuda {

template <typename T, int dim = 1, cudaTextureReadMode mode = cudaReadModeElementType>
class texture
{
public:
#ifdef __CUDACC__
    /**
     * type-safe constructor for CUDA host code
     */
    texture(::texture<T, dim, mode>& tex) : ptr_(&tex), desc_(tex.channelDesc) {}

    /**
     * variant constructor for CUDA host code
     *
     * For variant textures we need to override the channel desciptor.
     */
    texture(::texture<void, dim, mode>& tex) : ptr_(&tex), desc_(cudaCreateChannelDesc<T>()) {}
#else /* ! __CUDACC__ */
    /**
     * bind CUDA texture to device memory array
     */
    void bind(cuda::vector<T> const& array) const
    {
        ptr_->channelDesc = desc_;
        CUDA_CALL(cudaBindTexture(NULL, ptr_, array.data(), &desc_));
    }

    /**
     * unbind CUDA texture
     */
    void unbind() const
    {
        CUDA_CALL(cudaUnbindTexture(ptr_));
    }
#endif /* ! __CUDACC__ */

private:
#ifndef __CUDACC__
    texture() : ptr_(NULL), desc_() {}
#endif

    textureReference* ptr_;
    cudaChannelFormatDesc const desc_;
};

} // namespace cuda

#endif /* ! CUDA_TEXTURE_HPP */
