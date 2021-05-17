
# Needs to be different than "unit_test" because that's a different app
set( APP_NAME  gtest_tests )

file( GLOB_RECURSE C_SOURCES src/*.c )
file( GLOB_RECURSE CPP_SOURCES src/*.cpp )

file( GLOB_RECURSE SHARED_C ../shared/src/*.c )
file( GLOB_RECURSE SHARED_CPP ../shared/src/*.cpp )

set( XSCOPE_CONFIG config.xscope )

set( SOURCES "" )
list( APPEND SOURCES ${C_SOURCES} )
list( APPEND SOURCES ${CPP_SOURCES} )
list( APPEND SOURCES ${SHARED_C} )
list( APPEND SOURCES ${SHARED_CPP} )

if( ${CMAKE_SYSTEM_NAME} STREQUAL "XCORE" )
  list( APPEND SOURCES ${XSCOPE_CONFIG} )
endif()

add_executable( ${APP_NAME} ${SOURCES} )

target_link_libraries( ${APP_NAME}    lib_nn gtest)

target_include_directories( ${APP_NAME} PUBLIC 
                                ../shared/include
                                ../deps/compute-engine
                                ../deps/flatbuffers/include
                                ../deps/ruy
                                ../deps/tensorflow
                                ../deps/gemmlowp )
                          
target_compile_options( ${APP_NAME} PRIVATE
                          -Wno-unused-variable
                          -Wno-missing-braces )

if ( ${CMAKE_SYSTEM_NAME} STREQUAL "XCORE" )

  set_target_properties( ${APP_NAME} PROPERTIES
                            SUFFIX ".xe"
                            LINK_FLAGS "-target=XCORE-AI-EXPLORER -report" )

  target_compile_options( ${APP_NAME} PRIVATE
                            -Wno-xcore-fptrgroup )
endif()
