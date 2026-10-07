option(BUILD_DDR_FIX "Build XCore binary with DDR fix enabled" OFF)

set(LIB_NN_VPU_SYMMETRIC_SATURATION "" CACHE STRING
  "Whether the target VPU saturates to a symmetric signed range (0 or 1)")
set(LIB_NN_VPU_VLMACC_PRODUCT_LSB_DROPS "" CACHE STRING
  "Number of product LSBs dropped by the target VPU VLMACC operation")
set(LIB_NN_VPU_VLMUL_SHIFT_OFFSET "" CACHE STRING
  "Value subtracted from element width to obtain the target VPU VLMUL shift")

set(CMAKE_CXX_STANDARD 11)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

file(GLOB_RECURSE LIB_NN_SOURCES CONFIGURE_DEPENDS
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.c"
  "${CMAKE_CURRENT_SOURCE_DIR}/src/*.cpp"
)

if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS3A" OR
   CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_VX4" OR
   CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv32")
  file(GLOB_RECURSE LIB_NN_ASM_SOURCES CONFIGURE_DEPENDS
       "${CMAKE_CURRENT_SOURCE_DIR}/src/*.S")
  list(APPEND LIB_NN_SOURCES ${LIB_NN_ASM_SOURCES})
endif()

add_library(lib_nn STATIC ${LIB_NN_SOURCES})

target_include_directories(lib_nn PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}/api")

set(_lib_nn_vpu_capabilities
    LIB_NN_VPU_SYMMETRIC_SATURATION
    LIB_NN_VPU_VLMACC_PRODUCT_LSB_DROPS
    LIB_NN_VPU_VLMUL_SHIFT_OFFSET)
set(_lib_nn_vpu_capability_count 0)
foreach(_capability IN LISTS _lib_nn_vpu_capabilities)
  if(NOT "${${_capability}}" STREQUAL "")
    math(EXPR _lib_nn_vpu_capability_count "${_lib_nn_vpu_capability_count} + 1")
  endif()
endforeach()

if(_lib_nn_vpu_capability_count GREATER 0 AND
   NOT _lib_nn_vpu_capability_count EQUAL 3)
  message(FATAL_ERROR "Set all LIB_NN_VPU_* capabilities or none of them")
elseif(_lib_nn_vpu_capability_count EQUAL 3)
  target_compile_definitions(lib_nn PUBLIC
    VPU_SYMMETRIC_SATURATION=${LIB_NN_VPU_SYMMETRIC_SATURATION}
    VPU_VLMACC_PRODUCT_LSB_DROPS=${LIB_NN_VPU_VLMACC_PRODUCT_LSB_DROPS}
    VPU_VLMUL_SHIFT_OFFSET=${LIB_NN_VPU_VLMUL_SHIFT_OFFSET})
endif()

if(BUILD_DDR_FIX)
  target_compile_definitions(lib_nn PRIVATE USE_DDR_FIX)
endif()

# -------- target specific ----------
message(STATUS "Building lib_nn for CMAKE_SYSTEM_PROCESSOR : ${CMAKE_SYSTEM_PROCESSOR}")

# xs3
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS3A")
  target_compile_options(lib_nn PRIVATE -O3 -Wall -Wno-xcore-fptrgroup -Werror -Wextra)
# vx4
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_VX4" OR CMAKE_SYSTEM_PROCESSOR STREQUAL "riscv32")
  target_compile_options(lib_nn PRIVATE -O3 -Wall -Wno-fptrgroup -Werror -Wextra)
  target_compile_options(lib_nn PRIVATE -ffunction-sections -fdata-sections)
  target_link_options(lib_nn PRIVATE -Wl,--gc-sections)
# Native MSVC
elseif(MSVC) 
  target_compile_options(lib_nn PRIVATE /O2)
  target_compile_definitions(lib_nn PRIVATE NN_USE_REF)
# Native GCC/Clang
else()
  target_compile_options(lib_nn PRIVATE -O3 -Wall -Werror -Wextra)
  target_compile_definitions(lib_nn PRIVATE NN_USE_REF)
  target_link_libraries(lib_nn PRIVATE m)
endif()

# Supress warning on native build gcc < 9 on vpu_sim.c : GetAccumulator
if(WIN32 AND CMAKE_C_COMPILER_ID STREQUAL "GNU"
   AND CMAKE_C_COMPILER_VERSION VERSION_LESS 9)
  set_source_files_properties(src/c/vpu_sim.c PROPERTIES COMPILE_OPTIONS "-Wno-array-bounds")
endif()
