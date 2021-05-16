
set( UNITY_PATH ../../deps/Unity )

ADD_LIBRARY( Unity STATIC ${UNITY_PATH}/src/unity.c )

target_include_directories( Unity PUBLIC   
                                ${UNITY_PATH}/src )

set_target_properties(Unity PROPERTIES 
                 PREFIX ""
                 OUTPUT_NAME "Unity.a"
                 SUFFIX ".a" )

target_compile_options( Unity PRIVATE
                   -Wno-unused-variable
                   -Wno-missing-braces )

                   
if( ${CMAKE_SYSTEM_NAME} STREQUAL "XCORE" )
  target_compile_options( lib_nn PRIVATE
                            -Wno-xcore-fptrgroup )
endif()