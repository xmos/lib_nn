include(${CMAKE_CURRENT_LIST_DIR}/../etc/Unity/Unity.cmake)

file(GLOB_RECURSE UNIT_TEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_LIST_DIR}/src/*.c"
     "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

add_executable(unit_test ${UNIT_TEST_SOURCES})

target_include_directories(unit_test PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
target_compile_options(unit_test PRIVATE -g -O3 -Wno-unused-variable)
target_link_libraries(unit_test PRIVATE test_shared)

add_unity_to_target(unit_test CONFIG)

if(CMAKE_SYSTEM_NAME STREQUAL "XS3A" OR CMAKE_SYSTEM_NAME STREQUAL "VX4A")
  set(XSCOPE_CONFIG "${CMAKE_CURRENT_LIST_DIR}/config.xscope")
  target_link_options(
    unit_test
    PRIVATE "-target=${XCORE_TARGET}" -report -fcmdline-buffer-bytes=1024
            ${XSCOPE_CONFIG})
  set_property(TARGET unit_test PROPERTY LINK_DEPENDS ${XSCOPE_CONFIG})
endif()
