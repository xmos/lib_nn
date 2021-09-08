cmake_minimum_required(VERSION 3.14)

enable_language( C CXX ASM )

project(xmos-lib_nn)

# Build flags
#set(BUILD_FLAGS "-g -DTF_LITE_DISABLE_X86_NEON -O3 -DNN_USE_REF")
set(BUILD_FLAGS "-g")
list(APPEND BUILD_FLAGS "-DTF_LITE_DISABLE_X86_NEON" )
list(APPEND BUILD_FLAGS "-O3" )
list(APPEND BUILD_FLAGS "-DNN_USE_REF" )
set(BUILD_FLAGS_XCORE "-march=xs3a")
list(APPEND   BUILD_FLAGS ${BUILD_FLAGS_${CMAKE_SYSTEM_NAME}} )
list(JOIN BUILD_FLAGS " " BUILD_FLAGS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS ${BUILD_FLAGS})

set(CMAKE_C_FLAGS_X86 "-std=c99")
#set(CMAKE_C_FLAGS_XCORE "-march=xs3a")

set(CMAKE_C_FLAGS "${BUILD_FLAGS}")
list(APPEND   CMAKE_C_FLAGS ${CMAKE_C_FLAGS_${CMAKE_SYSTEM_NAME}} )
#list(APPEND   CMAKE_CXX_FLAGS ${CMAKE_CXX_FLAGS_${CMAKE_SYSTEM_NAME}} )

list(JOIN CMAKE_CXX_FLAGS " " CMAKE_CXX_FLAGS)
list(JOIN CMAKE_C_FLAGS " " CMAKE_C_FLAGS)
list(APPEND   COMPILE_FLAGS_XCORE  -march=xs3a   )
list(APPEND   COMPILE_FLAGS ${COMPILE_FLAGS_${CMAKE_SYSTEM_NAME}} )


# Target name
set(LIB_NAME lib_nn)

include(lib_nn/lib_nn.cmake)

# Deliverable is a static library
add_library(${LIB_NAME} STATIC ${LIB_NN_SOURCES})

target_include_directories(${LIB_NAME} PUBLIC ${LIB_NN_INCLUDE_DIR})

set_target_properties(
  ${LIB_NAME}
  PROPERTIES PREFIX ""
             OUTPUT_NAME ${LIB_NAME}
             SUFFIX ".a")

target_compile_options(${LIB_NAME} PRIVATE ${COMPILE_FLAGS})

target_compile_options(${LIB_NAME} PRIVATE "$<$<CONFIG:DEBUG>:-DDEBUGG=1>")

set(LIB_NN_TESTS_SHARED_INCLUDE_DIR
    "${CMAKE_CURRENT_SOURCE_DIR}/test/shared/include")

add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/shared)
add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/unit_test)
#add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/gtests)
