add_subdirectory(${CMAKE_CURRENT_LIST_DIR}/../deps/googletest
                 ${CMAKE_CURRENT_BINARY_DIR}/googletest
                 EXCLUDE_FROM_ALL)

file(GLOB_RECURSE GTEST_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_LIST_DIR}/src/*.c"
     "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

add_executable(gtests ${GTEST_SOURCES})

target_include_directories(gtests PRIVATE "${CMAKE_CURRENT_LIST_DIR}/src")
target_compile_options(gtests PRIVATE -g -O3 -Wno-unused-variable)
target_link_libraries(gtests PRIVATE test_shared gtest)

if(CMAKE_SYSTEM_NAME STREQUAL "XS3A" OR CMAKE_SYSTEM_NAME STREQUAL "VX4A")
  target_link_options(
    gtests PRIVATE "-target=${XCORE_TARGET}" -report
                   -fcmdline-buffer-bytes=1024)
endif()
