cmake_minimum_required(VERSION 3.14)

set(APP_NAME unit_test)

include(${CMAKE_CURRENT_LIST_DIR}/../etc/Unity/Unity.cmake)

# Depends on libraries
list(APPEND DEP_LIBS_XCORE "")

list(APPEND DEP_LIBS lib_nn test_shared ${DEP_LIBS_${CMAKE_SYSTEM_NAME}})

list(APPEND DEP_LIBS ${DEP_LIBS_${CMAKE_SYSTEM_NAME}})

# Sources
file(GLOB_RECURSE SOURCES_C src/*.c)
file(GLOB_RECURSE SOURCES_CPP src/*.cpp)

list(APPEND SOURCES ${SOURCES_C})
list(APPEND SOURCES ${SOURCES_CPP})
list(APPEND SOURCES ${SOURCES_${CMAKE_SYSTEM_NAME}})

# set( XSCOPE_CONFIG config.xscope )
get_filename_component(XSCOPE_CONFIG config.xscope ABSOLUTE)

# Compile flags
unset(COMPILE_FLAGS)
list(APPEND COMPILE_FLAGS -Wno-unused-variable)
list(APPEND COMPILE_FLAGS ${COMPILE_FLAGS_${CMAKE_SYSTEM_NAME}})

# Linker flags
unset(LINKER_FLAGS)
list(APPEND LINKER_FLAGS "")

unset(LINKER_FLAGS_XCORE)
list(APPEND LINKER_FLAGS_XCORE "-target=${XCORE_TARGET}")
list(APPEND LINKER_FLAGS_XCORE "-report")
list(APPEND LINKER_FLAGS_XCORE "-fcmdline-buffer-bytes=1024")
list(APPEND LINKER_FLAGS_XCORE "${XSCOPE_CONFIG}")

list(APPEND LINKER_FLAGS ${LINKER_FLAGS_${CMAKE_SYSTEM_NAME}})

#

add_executable(${APP_NAME} ${SOURCES})

target_include_directories(
  ${APP_NAME}
  PRIVATE src
  PRIVATE ${LIB_NN_INCLUDE_DIR}
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../deps/compute-engine
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../deps/tensorflow
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../deps/ruy
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../deps/gemmlowp
  PRIVATE ${CMAKE_CURRENT_LIST_DIR}/../deps/flatbuffers/include
  PRIVATE ${LIB_NN_ALT_INCLUDE_DIR}
  PRIVATE ${LIB_NN_TESTS_SHARED_INCLUDE_DIR})

target_link_libraries(${APP_NAME} ${DEP_LIBS})

target_compile_options(${APP_NAME} PRIVATE ${COMPILE_FLAGS})

# (because otherwise the set_target_properties command fails)
string(REPLACE ";" " " LINKER_FLAGS_STR "${LINKER_FLAGS}")
set_target_properties(${APP_NAME} PROPERTIES LINK_FLAGS "${LINKER_FLAGS_STR}")

if(XCORE)
  set_target_properties(${APP_NAME} PROPERTIES SUFFIX ".xe" LINK_DEPENDS
                                                            ${XSCOPE_CONFIG})
endif()

add_unity_to_target(${APP_NAME} CONFIG)
