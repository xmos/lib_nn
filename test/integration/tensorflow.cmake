if(NOT DEFINED XMOS_SANDBOX_DIR)
  message(FATAL_ERROR "XMOS_SANDBOX_DIR must be defined before including tensorflow.cmake")
endif()

# tflite reference kernels (conv/depthwise_conv/transpose_conv) are header-only;
# their only third-party transitive dependency is gemmlowp's fixedpoint/fixedpoint.h
add_library(tflite_reference_kernels INTERFACE)
target_include_directories(tflite_reference_kernels INTERFACE
  ${XMOS_SANDBOX_DIR}/tensorflow
  ${XMOS_SANDBOX_DIR}/gemmlowp
)

# RuntimeShape and MultiplyByQuantizedMultiplier are declared in common.h/
# runtime_shape.h but only defined out-of-line in these two .cc files
set(TFLITE_REFERENCE_KERNEL_SRCS
  ${XMOS_SANDBOX_DIR}/tensorflow/tensorflow/lite/kernels/internal/common.cc
  ${XMOS_SANDBOX_DIR}/tensorflow/tensorflow/lite/kernels/internal/runtime_shape.cc
)

# Link TFLITE with the targets
foreach(target ${APP_BUILD_TARGETS})
    message(STATUS "Linking ${target} with tflite_reference_kernels")
    target_link_libraries(${target} PRIVATE tflite_reference_kernels)
    target_sources(${target} PRIVATE ${TFLITE_REFERENCE_KERNEL_SRCS})
endforeach()

