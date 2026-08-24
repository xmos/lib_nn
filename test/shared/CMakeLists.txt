file(GLOB_RECURSE SHARED_SOURCES CONFIGURE_DEPENDS
     "${CMAKE_CURRENT_LIST_DIR}/src/*.c"
     "${CMAKE_CURRENT_LIST_DIR}/src/*.cpp")

add_library(test_shared ${SHARED_SOURCES})

target_include_directories(
  test_shared
  PUBLIC "${CMAKE_CURRENT_LIST_DIR}/include"
  PRIVATE
    "${CMAKE_CURRENT_LIST_DIR}/src"
    "${CMAKE_CURRENT_LIST_DIR}/../deps/compute-engine"
    "${CMAKE_CURRENT_LIST_DIR}/../deps/flatbuffers/include"
    "${CMAKE_CURRENT_LIST_DIR}/../deps/gemmlowp"
    "${CMAKE_CURRENT_LIST_DIR}/../deps/ruy"
    "${CMAKE_CURRENT_LIST_DIR}/../deps/tensorflow")

target_compile_options(test_shared PRIVATE -g -O3)
target_link_libraries(test_shared PUBLIC lib_nn)
