
## Target name
set( LIB_NAME   lib_nn  )

## Export lib directory as variable
set( LIB_NN_PATH ${CMAKE_CURRENT_SOURCE_DIR} PARENT_SCOPE )

message(STATUS "lib_nn CMakeList -----------------------------------------------------------------------------------")
include(lib_nn.cmake)

## Deliverable is a static library
add_library( ${LIB_NAME}  STATIC      ${LIB_NN_SOURCES} )

target_include_directories( ${LIB_NAME} PUBLIC   ${LIB_NN_INCLUDE_DIRS}  )

set_target_properties( ${LIB_NAME} PROPERTIES   PREFIX      ""
                                                OUTPUT_NAME ${LIB_NAME}
                                                SUFFIX      ".a"        )

target_compile_options( ${LIB_NAME} PRIVATE ${COMPILE_FLAGS} )

target_compile_options( ${LIB_NAME} PRIVATE "$<$<CONFIG:DEBUG>:-DDEBUGG=1>" )


