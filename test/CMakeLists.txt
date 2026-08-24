cmake_minimum_required(VERSION 3.21)

project(lib_nn_tests LANGUAGES C CXX ASM)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

option(ENABLE_SANITIZERS "Enable sanitizers for native tests" OFF)

if(ENABLE_SANITIZERS AND NOT CMAKE_SYSTEM_NAME STREQUAL "XS3A" AND NOT CMAKE_SYSTEM_NAME STREQUAL "VX4A")
  add_compile_options(-fsanitize=address,undefined -fno-omit-frame-pointer)
  add_link_options(-fsanitize=address,undefined)
endif()

add_subdirectory("${CMAKE_CURRENT_LIST_DIR}/../lib_nn"
                 "${CMAKE_CURRENT_BINARY_DIR}/lib_nn")
add_subdirectory(shared)
add_subdirectory(unit_test)

if(NOT CMAKE_SYSTEM_NAME STREQUAL "XS3A")
  add_subdirectory(gtests)
endif()
