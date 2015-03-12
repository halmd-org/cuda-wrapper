#
# Define dynamic Boost Test linking flag if needed.
#
# We support both the static and dynamic library variants.
#
# http://www.boost.org/doc/libs/release/libs/test/doc/html/utf/user-guide/usage-variants.html
#
get_filename_component(
  suffix "${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}" EXT
)
if(suffix STREQUAL "${CMAKE_SHARED_LIBRARY_SUFFIX}")
  get_directory_property(
    flags COMPILE_DEFINITIONS
  )
  list(APPEND flags
    "BOOST_TEST_DYN_LINK"
  )
  set_directory_properties(PROPERTIES
    COMPILE_DEFINITIONS "${flags}"
  )
endif(suffix STREQUAL "${CMAKE_SHARED_LIBRARY_SUFFIX}")

add_executable(test_cuda_wrapper_copy
  copy.cpp
)
target_link_libraries(test_cuda_wrapper_copy
  ${CUDA_LIBRARIES}
  ${Boost_UNIT_TEST_FRAMEWORK_LIBRARY}
  pthread
)
add_test(cuda_wrapper/copy
  test_cuda_wrapper_copy --log_level=test_suite
)
