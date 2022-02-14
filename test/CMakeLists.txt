set(Boost_VERSION ${Boost_MAJOR_VERSION}.${Boost_MINOR_VERSION}.${Boost_SUBMINOR_VERSION})

set(TEST_DEVICE 0 CACHE STRING "The device used for the tests")
set(TEST_CUDA_MEMCHECK ON CACHE BOOL
  "Set to OFF to disable CUDA memory checks while running the tests.")

enable_language(CUDA)
find_library(CUDA_LIBRARY cuda HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
find_library(CUDART_LIBRARY cudart HINTS ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})

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
include_directories(SYSTEM ${CMAKE_CUDA_TOOLKIT_INCLUDE_DIRECTORIES})
include_directories(SYSTEM ${Boost_INCLUDE_DIR})

set(CMAKE_POSITION_INDEPENDENT_CODE ON)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)
set(CMAKE_CUDA_STANDARD 14)
set(CMAKE_CUDA_ARCHITECTURES OFF)
set(CMAKE_CXX_FLAGS "-Wall -pedantic")

# create a test that runs on the host and device
# usage: create_gpu_test(NAME [NO_MEMCHECK] FILES...)
macro(create_test NAME NM_OR_FILE)
  if ("${NM_OR_FILE}" STREQUAL "NO_MEMCHECK")
    add_executable(test_cuda_wrapper_${NAME} ${ARGN})
  else()
    add_executable(test_cuda_wrapper_${NAME} ${NM_OR_FILE} ${ARGN})
  endif()

  target_compile_definitions(test_cuda_wrapper_${NAME}
    PRIVATE TEST_DEVICE=${TEST_DEVICE}
    PRIVATE BOOST_TEST_DYN_LINK
  )
  target_link_libraries(test_cuda_wrapper_${NAME}
    ${CUDA_LIBRARY}
    ${CUDART_LIBRARY}
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

create_test(version NO_MEMCHECK version.cpp)
create_test(device NO_MEMCHECK device.cpp)
create_test(iterator NO_MEMCHECK iterator.cpp)
create_test(function function.cpp function_kernel.cu)
create_test(event event.cpp event_kernel.cu)
create_test(texture texture.cpp texture_kernel.cu)
# this test somehow doesn't work with cuda-memcheck
create_test(stream NO_MEMCHECK stream.cpp stream_kernel.cu)

# tests which use data driven test cases don't work with Boost < 1.59.0
if(NOT ${Boost_VERSION} VERSION_LESS 1.59.0)
  create_test(allocator NO_MEMCHECK allocator.cpp)
  create_test(vector NO_MEMCHECK vector.cpp)
  create_test(copy NO_MEMCHECK copy.cpp)
endif()
