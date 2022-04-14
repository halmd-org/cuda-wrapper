CXX = nvcc
CXXFLAGS = -std=c++17 -I. -I./cuda-wrapper/ -l cuda

all:
	$(CXX) $(CXXFLAGS) -o example main.cpp kernel.cu
	
clean: 
	rm -f example

run:
	./example
