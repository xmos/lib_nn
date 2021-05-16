

file( GLOB_RECURSE C_SOURCES src/*.c )
file( GLOB_RECURSE CPP_SOURCES src/*.cpp )
file( GLOB_RECURSE ASM_SOURCES src/*.S )

set_source_files_properties( ${ASM_SOURCES} PROPERTIES
                              LANGUAGE ASM )

set( SOURCES "" )
list( APPEND SOURCES ${C_SOURCES} )
list( APPEND SOURCES ${CPP_SOURCES} )

if( ${CMAKE_SYSTEM_NAME} STREQUAL "XCORE" )
  list( APPEND SOURCES ${ASM_SOURCES} )
endif() 


ADD_LIBRARY( lib_nn STATIC ${SOURCES} )

# Add api/ and src/ as include dirs for targets using this library
target_include_directories( lib_nn PUBLIC 
                                api
                                src )

# static lib name is lib_nn.a
set_target_properties(lib_nn PROPERTIES 
                      PREFIX ""
                      OUTPUT_NAME "lib_nn"
                      SUFFIX ".a" )

# compile flags
target_compile_options( lib_nn PRIVATE
                        -Wno-xcore-fptrgroup
                        -Wno-unused-variable
                        -Wno-missing-braces )