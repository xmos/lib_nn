cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)

project(unit_test)
set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_INCLUDES src src/etc)
set(APP_DEPENDENT_MODULES "lib_nn" "lib_unity(v2.7.0)")

if(NOT BUILD_NATIVE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS")
        if(NOT DEFINED APP_HW_TARGET)
            set(APP_HW_TARGET XK-EVK-XU316)
        endif()
    endif()
endif()

set(APP_COMPILER_FLAGS -Os -g)
if(BUILD_NATIVE)
    list(APPEND APP_COMPILER_FLAGS -DTEST_BUILD_NATIVE)
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS")
    list(APPEND APP_COMPILER_FLAGS -report -fcmdline-buffer-bytes=1024)
endif()

XMOS_REGISTER_APP()
