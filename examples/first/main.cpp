#include "kernel.hpp"
#include <iostream>
#include <vector>

int main()
{
    cuda::device dev;
    dev.set(0);
    cuda::memory::managed::vector<int> array(10);
    //cuda::thread::synchronize();
    //size_t size = 128 * sizeof(int);
    //int* array;
    //cudaMallocManaged(&array, size);
    cuda::config dim(64, 128);
    wrapper::kernel.func.configure(dim.grid, dim.block);
    wrapper::kernel.func(array);
    cuda::thread::synchronize();
    auto begin = array.begin();
    auto end = array.end();
    for (auto i = begin; i < end; i++)
        std::cout << *i << std::endl;
    return 0;
}