cmake_minimum_required(VERSION 3.21)
include($ENV{XMOS_CMAKE_PATH}/xcommon.cmake)

project(unit_test)
set(XMOS_SANDBOX_DIR ${CMAKE_CURRENT_LIST_DIR}/../../..)

set(APP_INCLUDES src src/etc)
set(APP_DEPENDENT_MODULES "lib_nn" "lib_unity(v2.7.0)")
set(APP_C_SRCS
  src/main.c
  src/test_add_elementwise.c
  src/test_add_int16.c
  src/test_bsign8.c
  src/test_dequantize_int16.c
  src/test_expand_8_to_16.c
  src/test_lookup8.c
  src/test_mul_elementwise.c
  src/test_multiply_int16.c
  src/test_pad_3_to_4.c
  src/test_quantize_int16.c
  src/test_softmax.c
  src/test_vpu.c
  src/etc/helpers.c
  src/etc/tst_common.c
)
set(APP_CXX_SRCS
  src/etc/Rand.cpp
  src/test_AggregateFns.cpp
  src/test_MemCpyFns.cpp
  src/test_OutputTransforms.cpp
  src/test_maxpool.cpp
)

if(NOT DEFINED APP_HW_TARGET AND NOT BUILD_NATIVE)
    set(APP_HW_TARGET XK-EVK-XU316)
endif()

set(APP_COMPILER_FLAGS -Os -g)
if(BUILD_NATIVE)
    list(APPEND APP_COMPILER_FLAGS -DTEST_BUILD_NATIVE)
else()
    list(APPEND APP_COMPILER_FLAGS -report -fcmdline-buffer-bytes=1024)
endif()

XMOS_REGISTER_APP()
