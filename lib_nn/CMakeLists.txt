option(BUILD_DDR_FIX "Build XCore binary with DDR fix enabled" OFF)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE LIB_NN_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
     "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp")

if(CMAKE_SYSTEM_NAME STREQUAL "XS3A" OR CMAKE_SYSTEM_NAME STREQUAL "VX4A")
  file(GLOB_RECURSE LIB_NN_ASM_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/src/*.S")
  list(APPEND LIB_NN_SOURCES ${LIB_NN_ASM_SOURCES})
endif()

add_library(lib_nn STATIC ${LIB_NN_SOURCES})

target_include_directories(lib_nn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/api")

target_compile_options(
  lib_nn PRIVATE -g -O3 -Wno-unused-variable -Wno-missing-braces)

if(CMAKE_SYSTEM_NAME STREQUAL "XS3A")
  target_compile_options(lib_nn PRIVATE -Wno-xcore-fptrgroup)
  if(BUILD_DDR_FIX)
    target_compile_definitions(lib_nn PRIVATE USE_DDR_FIX)
  endif()
elseif(NOT CMAKE_SYSTEM_NAME STREQUAL "VX4A")
  target_compile_definitions(lib_nn PRIVATE NN_USE_REF)
endif()
