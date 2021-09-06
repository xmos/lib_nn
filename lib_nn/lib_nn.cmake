## Source files
file( GLOB_RECURSE    LIB_NN_C_SOURCES       "src/*.c" )
file( GLOB_RECURSE    LIB_NN_CPP_SOURCES     "src/*.cpp" )
file( GLOB_RECURSE    LIB_NN_ASM_SOURCES     "src/*.S"   )

# file( GLOB_RECURSE    LIB_NN_C_SOURCES_REF "src/arch/ref/*.c" )
message(STATUS "lib_nn.cmake -----------------------------------------------------------------------------------")
## Compile flags for all platforms
unset(LIB_NN_COMPILE_FLAGS)
list( APPEND  LIB_NN_COMPILE_FLAGS     -Wno-unused-variable -Wno-missing-braces)

## Platform-specific compile flags can go here
unset(LIB_NN_COMPILE_FLAGS_XCORE)
list( APPEND  LIB_NN_COMPILE_FLAGS_XCORE   -Wno-xcore-fptrgroup)

## set LIB_NN_INCLUDES & LIB_NN_SOURCES
# set( LIB_NN_INCLUDES     api  )

set(LIB_NN_INCLUDE_DIRS
  "api"
  "src"
)

unset(LIB_NN_SOURCES)
list( APPEND  LIB_NN_SOURCES   ${LIB_NN_C_SOURCES}    )
list( APPEND  LIB_NN_SOURCES   ${LIB_NN_CPP_SOURCES}  )

unset(LIB_NN_SOURCES_XCORE)
list( APPEND  LIB_NN_SOURCES_XCORE   ${LIB_NN_ASM_SOURCES} )

# unset(LIB_NN_SOURCES_Linux) 
# list( APPEND  LIB_NN_SOURCES_Linux   ${LIB_NN_C_SOURCES_REF} )

## Combine platform-agnostic and platform-specific variables..
list( APPEND  LIB_NN_COMPILE_FLAGS    ${LIB_NN_COMPILE_FLAGS_${CMAKE_SYSTEM_NAME}} )

list( APPEND  LIB_NN_SOURCES  ${LIB_NN_SOURCES_${CMAKE_SYSTEM_NAME}}        )

## cmake doesn't recognize .S files as assembly by default
set_source_files_properties( ${LIB_NN_ASM_SOURCES} PROPERTIES LANGUAGE ASM )

foreach(COMPILE_FLAG ${LIB_NN_COMPILE_FLAGS})
  set_source_files_properties( ${LIB_NN_SOURCES} PROPERTIES COMPILE_FLAGS ${COMPILE_FLAG}) 
endforeach()
