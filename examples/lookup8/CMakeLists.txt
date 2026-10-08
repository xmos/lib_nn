cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)
project(lookup8)

if(NOT BUILD_NATIVE)
    if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS")
        if(NOT DEFINED APP_HW_TARGET)
            set(APP_HW_TARGET XK-EVK-XU316)
        endif()
    endif()
endif()

set(APP_COMPILER_FLAGS -Os -g)
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS")
	list(APPEND APP_COMPILER_FLAGS -report)
endif()

set(APP_DEPENDENT_MODULES  "lib_nn")

set(APP_INCLUDES           src)

set(XMOS_SANDBOX_DIR       ${CMAKE_CURRENT_LIST_DIR}/../../..)

XMOS_REGISTER_APP()
