cmake_minimum_required(VERSION 3.14)

set(LIB_NAME test_shared)

include(${CMAKE_CURRENT_LIST_DIR}/../../../ai_deployment_framework.cmake)

file(GLOB_RECURSE SHARED_SOURCES_C ${CMAKE_CURRENT_LIST_DIR}/*.c)
file(GLOB_RECURSE SHARED_SOURCES_CPP ${CMAKE_CURRENT_LIST_DIR}/*.cpp)
file(GLOB_RECURSE SHARED_SOURCES_XC ${CMAKE_CURRENT_LIST_DIR}/*.xc)

list(APPEND INCLUDE_DIRS ${CMAKE_CURRENT_LIST_DIR}/include
     ${CMAKE_CURRENT_LIST_DIR}/src)

# ##############################################################################

# cmake doesn't recognize XC files. Tell it to treat them as C files
set_source_files_properties(${SHARED_SOURCES_XC} PROPERTIES LANGUAGE C)

list(APPEND SHARED_SOURCES ${SHARED_SOURCES_C})
list(APPEND SHARED_SOURCES ${SHARED_SOURCES_CPP})

list(APPEND SHARED_SOURCES ${SHARED_SOURCES_${CMAKE_SYSTEM_NAME}})

#list(APPEND   COMPILE_FLAGS_XCORE  -march=xs3a   )
#list(APPEND   COMPILE_FLAGS ${COMPILE_FLAGS_${CMAKE_SYSTEM_NAME}} )
#list(JOIN COMPILE_FLAGS " " COMPILE_FLAGS)

#set(BUILD_FLAGS_XCORE "-march=xs3a")
#list(APPEND   BUILD_FLAGS ${BUILD_FLAGS_${CMAKE_SYSTEM_NAME}} )
#list(JOIN BUILD_FLAGS " " BUILD_FLAGS)

# Add library called "test_shared"
add_library(${LIB_NAME} ${SHARED_SOURCES})

target_include_directories(
  ${LIB_NAME}
  PRIVATE ${INCLUDE_DIRS}
  PRIVATE ${LIB_NN_INCLUDE_DIR}
  PRIVATE ${LIB_NN_ALT_INCLUDE_DIR}
  PRIVATE ${TENSORFLOW_INCLUDE_DIR}
  PRIVATE ${GEMMLOWP_INCLUDE_DIR})

target_link_libraries(${LIB_NAME} lib_nn)
