set(Boost_VERSION ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION})

set(TEST_DEVICE 0 CACHE STRING "The device used for the tests")
set(TEST_CUDA_MEMCHECK ON CACHE BOOL
  "Set to OFF to disable CUDA memory checks while running the tests.")

# FindCUDA uses the static cuda runtime by default, we need to set this variable
# before find_package is called
set(CUDA_USE_STATIC_CUDA_RUNTIME OFF CACHE BOOL
  "Use the static version of the CUDA runtime library if available")
find_package(CUDA REQUIRED COMPONENTS cuda cudart)

if(TEST_CUDA_MEMCHECK)
  find_program(CUDA_MEMCHECK cuda-memcheck)
  if(CUDA_MEMCHECK)
    set(CUDA_MEMCHECK_COMMAND ${CUDA_MEMCHECK} --error-exitcode 1)
  else()
    message(WARNING
      "cuda-memcheck was not found. No memory checks will be performed.")
    set(CUDA_MEMCHECK_COMMAND)
  endif()
else()
  set(CUDA_MEMCHECK_COMMAND)
endif()

include_directories(${CMAKE_SOURCE_DIR})
include_directories(SYSTEM ${CUDA_INCLUDE_DIRS})
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

set(CUDA_NVCC_FLAGS "-std=c++11 -Xcompiler -fPIC")
set(CMAKE_CXX_FLAGS "-Wall -std=c++11 -fPIC")

# create a test that only runs on the host
# usage: create_test(NAME FILES...)
macro(create_test NAME)
  add_executable(test_cuda_wrapper_${NAME} ${ARGN})
  target_compile_definitions(test_cuda_wrapper_${NAME}
    PRIVATE TEST_DEVICE=${TEST_DEVICE}
    PRIVATE BOOST_TEST_DYN_LINK
  )
  target_link_libraries(test_cuda_wrapper_${NAME}
    ${CUDA_CUDA_LIBRARY}
    ${CUDA_LIBRARIES}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )
  add_test(cuda-wrapper/${NAME}
    test_cuda_wrapper_${NAME} --log_level=test_suite
  )
endmacro()

# create a test that runs on the host and device
# usage: create_gpu_test(NAME [NO_MEMCHECK] FILES...)
macro(create_gpu_test NAME NM_OR_FILE)
  if ("${NM_OR_FILE}" STREQUAL "NO_MEMCHECK")
    cuda_add_executable(test_cuda_wrapper_${NAME} ${ARGN})
  else()
    cuda_add_executable(test_cuda_wrapper_${NAME} ${NM_OR_FILE} ${ARGN})
  endif()

  target_compile_definitions(test_cuda_wrapper_${NAME}
    PRIVATE TEST_DEVICE=${TEST_DEVICE}
    PRIVATE BOOST_TEST_DYN_LINK
  )
  target_link_libraries(test_cuda_wrapper_${NAME}
    ${CUDA_CUDA_LIBRARY}
    ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  )
  if ("${NM_OR_FILE}" STREQUAL "NO_MEMCHECK")
    add_test(cuda-wrapper/${NAME}
      test_cuda_wrapper_${NAME} --log_level=test_suite)
  else()
    add_test(cuda-wrapper/${NAME}
      ${CUDA_MEMCHECK_COMMAND} test_cuda_wrapper_${NAME} --log_level=test_suite)
  endif()
endmacro()

create_test(version version.cpp)
create_test(device device.cpp)
create_test(copy copy.cpp)
create_gpu_test(function function.cpp function_kernel.cu)
create_gpu_test(event event.cpp function_kernel.cu)
create_gpu_test(texture texture.cpp texture_kernel.cu)
# this test somehow doesn't work with cuda-memcheck
create_gpu_test(stream NO_MEMCHECK stream.cpp function_kernel.cu)

# tests which use data driven test cases don't work with Boost < 1.59.0
if(NOT ${Boost_VERSION} VERSION_LESS 1.59.0)
  create_test(allocator allocator.cpp)
  create_test(host_allocator host_allocator.cpp)
  create_test(vector vector.cpp)
  create_test(host_vector host_vector.cpp)
endif()
