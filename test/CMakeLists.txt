cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(lib_nn_tests)

add_subdirectory(unit_test)

# integration only supports native builds (fetches third-party test deps)
if(BUILD_NATIVE)
  add_subdirectory(integration)
endif()
