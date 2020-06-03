/* cuda_wrapper/texture.hpp
 *
 * Copyright (C) 2020 Jaslo Ziska
 *
 * This file is part of cuda-wrapper.
 *
 * This software may be modified and distributed under the terms of the
 * 3-clause BSD license.  See accompanying file LICENSE for details.
 */

#include <cuda_runtime.h>

#include <cuda_wrapper/error.hpp>
#include <cuda_wrapper/vector.hpp>

#ifndef CUDA_TEXTURE_HPP
#define CUDA_TEXTURE_HPP

namespace cuda {

template <typename T, size_t dim = 1, cudaTextureReadMode mode = cudaReadModeElementType>
class texture
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
         * create a texture object
         */
        container(const cuda::vector<T>& vector) : data_(vector.data())
        {
            cudaResourceDesc resource_desc = {};
            resource_desc.resType = cudaResourceTypeLinear;
            resource_desc.res.linear.devPtr = const_cast<T*>(data_);
            resource_desc.res.linear.desc = cudaCreateChannelDesc<T>();
            resource_desc.res.linear.sizeInBytes = vector.capacity() * sizeof(T);

            cudaTextureDesc texture_desc = {};
            texture_desc.readMode = mode;

            CUDA_CALL(cudaCreateTextureObject(&texture_, &resource_desc,
                &texture_desc, NULL));
        }

        /**
         * destroy the texture object
         */
        ~container() throw() // no-throw guarantee
        {
            cudaDestroyTextureObject(texture_);
        }

        cudaTextureObject_t texture_;
        // save the shared pointer to the device memory so it can't be freed
        T const* data_;
    };

    std::shared_ptr<container> texture_;

public:
    texture(const cuda::vector<T>& vector) : texture_(new container(vector)) {}

    inline operator cudaTextureObject_t() const
    {
        return texture_->texture_;
    }
};

} // namespace cuda

#endif // CUDA_TEXTURE_HPP
