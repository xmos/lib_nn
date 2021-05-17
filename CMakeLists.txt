cmake_minimum_required( VERSION 3.10.0 ) #TODO: This is just my cmake version

project(xmos-lib_nn)       #TODO: Is this reasonable?

set( XCORE_ARCH xs3a )

enable_language( C CXX ASM )

# Set our C++ version
set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS ON)

add_compile_options( -Os -g -MMD
                     -Wno-format )

if( ${CMAKE_SYSTEM_NAME} STREQUAL "XCORE" )
  add_compile_options( -march=${XCORE_ARCH} )
else()
  add_compile_options( -DNN_USE_REF=1 )
endif()


add_subdirectory( lib_nn )
add_subdirectory( test/etc/Unity )

set( BUILD_GMOCK OFF )
add_subdirectory( test/deps/googletest )

add_subdirectory( test/unit_test )
add_subdirectory( test/gtests )