cmake_minimum_required(VERSION 3.21)

project(custom_cmake_build C ASM)

## Add lib_nn and dependencies
set(SANDBOX_DIR "${CMAKE_CURRENT_LIST_DIR}/../../..")

add_subdirectory(
    "${SANDBOX_DIR}/lib_nn/lib_nn"
    "${CMAKE_BINARY_DIR}/lib_nn"
)

### Add executable
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY "${CMAKE_CURRENT_LIST_DIR}/bin")

add_executable(add_tensor)

target_sources(add_tensor PRIVATE
    src/main.c
)

target_include_directories(add_tensor PRIVATE
    src
)

target_link_libraries(add_tensor PRIVATE
    lib_nn
)

set(COMMON_OPTIONS
    -Os
    -g
    -Wall
)

if(CMAKE_SYSTEM_NAME STREQUAL "XCORE_XS3A")
    target_compile_options(add_tensor PRIVATE
        -target=XK-EVK-XU316
        ${COMMON_OPTIONS}
    )
    target_link_options(add_tensor PRIVATE
        -target=XK-EVK-XU316
        -report
    )
else()
    target_compile_options(add_tensor PRIVATE ${COMMON_OPTIONS})
endif()
