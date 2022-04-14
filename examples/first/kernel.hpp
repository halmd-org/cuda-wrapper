#include "cuda_wrapper/cuda_wrapper.hpp"

struct wrapper
{
    cuda::function<void (int*)> func;
    static wrapper kernel;
};

