cmake_minimum_required(VERSION 3.12.4) #TODO: This is just my cmake version

project(xmos-lib_nn)       #TODO: Is this reasonable?

set( XCORE_ARCH xs3a )

enable_language( C CXX ASM )

# Set our C++ version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_compile_options( -Os -g -MMD )

if( ${CMAKE_SYSTEM_NAME} STREQUAL "XCORE" )
  add_compile_options( -march=${XCORE_ARCH} )
endif()


add_subdirectory( lib_nn )
add_subdirectory( test/etc/Unity )

add_subdirectory( test/unit_test )