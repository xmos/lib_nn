cmake_minimum_required(VERSION 3.14)

enable_language( C CXX ASM )

project(xmos-lib_nn)

option(BUILD_DDR_FIX "Build XCore binary with DDR fix enabled" OFF)

# Build flags
set(BUILD_FLAGS "-g")
list(APPEND BUILD_FLAGS "-DTF_LITE_DISABLE_X86_NEON" )
list(APPEND BUILD_FLAGS "-O3" )

set(BUILD_FLAGS_Darwin "-DNN_USE_REF")
set(BUILD_FLAGS_Linux "-DNN_USE_REF")

list(APPEND BUILD_FLAGS_Darwin "-Wall" )
list(APPEND BUILD_FLAGS_Darwin "-fsanitize=undefined" )
list(APPEND BUILD_FLAGS_Darwin "-fsanitize=integer" )
list(APPEND BUILD_FLAGS_Darwin "-fsanitize=implicit-conversion" )
list(APPEND BUILD_FLAGS_Darwin "-fsanitize=address" )
list(APPEND BUILD_FLAGS_Darwin "-fsanitize-recover=address" )

set(BUILD_FLAGS_XCORE "-march=xs3a")
if(BUILD_DDR_FIX)
  list(APPEND BUILD_FLAGS_XCORE "-DUSE_DDR_FIX")
endif(BUILD_DDR_FIX)
    
list(APPEND   BUILD_FLAGS ${BUILD_FLAGS_${CMAKE_SYSTEM_NAME}} )
# list(JOIN BUILD_FLAGS " " BUILD_FLAGS)

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_FLAGS ${BUILD_FLAGS})

set(CMAKE_C_FLAGS_X86 "-std=c99")

set(CMAKE_C_FLAGS "${BUILD_FLAGS}")
list(APPEND   CMAKE_C_FLAGS ${CMAKE_C_FLAGS_${CMAKE_SYSTEM_NAME}} )

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

option(BUILD_GTESTS "Builds the googletests test suite" OFF)

if(BUILD_GTESTS)
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/test/gtests)
endif()