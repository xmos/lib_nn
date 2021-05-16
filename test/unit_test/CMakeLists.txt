

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

add_executable( unit_test ${SOURCES} )

target_link_libraries( unit_test    lib_nn Unity)

target_include_directories( unit_test PUBLIC 
                                ../shared/include
                                ../deps/compute-engine
                                ../deps/flatbuffers/include
                                ../deps/ruy
                                ../deps/tensorflow
                                ../deps/gemmlowp )
                          
target_compile_options( unit_test PRIVATE
                          -Wno-unused-variable
                          -Wno-missing-braces )

if ( ${CMAKE_SYSTEM_NAME} STREQUAL "XCORE" )

  set_target_properties( unit_test PROPERTIES
                            SUFFIX ".xe"
                            LINK_FLAGS "-target=XCORE-AI-EXPLORER -report" )

  target_compile_options( lib_nn PRIVATE
                            -Wno-xcore-fptrgroup )
endif()
