cmake_minimum_required(VERSION 3.21)

project(custom_cmake_build C ASM)

# Add lib_nn and dependencies
set(SANDBOX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")
add_subdirectory("${SANDBOX_DIR}/lib_nn/lib_nn" "${CMAKE_BINARY_DIR}/lib_nn")

# Add executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")
set(TARGET_NAME add_tensor)
set(COMMON_OPTIONS -Os -g -Wall)

add_executable(${TARGET_NAME})
target_sources(${TARGET_NAME} PRIVATE src/main.c)
target_include_directories(${TARGET_NAME} PRIVATE src)
target_link_libraries(${TARGET_NAME} PRIVATE lib_nn)

# xs3
if(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_XS3A")
  target_compile_options(${TARGET_NAME} PRIVATE -target=XK-EVK-XU316 -report ${COMMON_OPTIONS})
  target_link_options(${TARGET_NAME} PRIVATE -target=XK-EVK-XU316)
# vx4
elseif(CMAKE_SYSTEM_PROCESSOR STREQUAL "XCORE_VX4")
  target_compile_options(${TARGET_NAME} PRIVATE -target=XK-EVK-XU416 ${COMMON_OPTIONS})
  target_link_options(${TARGET_NAME} PRIVATE -target=XK-EVK-XU416)
else()
  target_compile_options(${TARGET_NAME} PRIVATE ${COMMON_OPTIONS})
endif()
