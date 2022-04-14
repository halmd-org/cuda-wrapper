cmake_minimum_required(VERSION 3.8)

# set cmake policies to avoid warnings in newer cmake versions.
if(POLICY CMP0053)
  cmake_policy(SET CMP0053 NEW)
endif()
if(POLICY CMP0026)
  cmake_policy(SET CMP0026 NEW)
endif()
if(POLICY CMP0058)
  cmake_policy(SET CMP0058 NEW)
endif()

project(cuda-wrapper CXX)

include(GNUInstallDirs)
install(DIRECTORY cuda_wrapper DESTINATION ${CMAKE_INSTALL_INCLUDEDIR})

find_package(Boost 1.59.0 COMPONENTS unit_test_framework)
if(NOT Boost_FOUND)
  message(WARNING "Boost was not found, unit tests are disabled")
else()
  enable_testing()
  add_subdirectory(test)
endif()

add_subdirectory(examples)
