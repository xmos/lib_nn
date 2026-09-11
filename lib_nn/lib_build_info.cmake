set(LIB_NAME lib_nn)
set(LIB_VERSION 0.6.0)
set(LIB_INCLUDES api)
set(LIB_DEPENDENT_MODULES "")

# compiler flags based on target
#TODO add on native -fsanitize=address,undefined -fno-omit-frame-pointer
set(COMPILER_GLAGS -O3 -Wall)
if(APP_BUILD_ARCH STREQUAL "xs3a") # xs3
    set(LIB_COMPILER_FLAGS ${COMPILER_GLAGS} -Wno-xcore-fptrgroup -Werror -Wextra)
elseif(APP_BUILD_ARCH STREQUAL "vx4b") # vx4
    set(LIB_COMPILER_FLAGS ${COMPILER_GLAGS} -Wno-fptrgroup -Werror -Wextra)
else() # native
    set(LIB_COMPILER_FLAGS ${COMPILER_GLAGS} -DNN_USE_REF -Werror -Wextra) 
endif()

XMOS_REGISTER_MODULE()

foreach(target ${APP_BUILD_TARGETS})
  if(APP_BUILD_ARCH STREQUAL "vx4b")
    target_compile_options(${target} PRIVATE -ffunction-sections -fdata-sections)
    target_link_options(${target} PRIVATE -Wl,--gc-sections)
  endif()
  if(BUILD_NATIVE AND (NOT CMAKE_CXX_COMPILER_ID STREQUAL "MSVC"))
    target_link_libraries(${target} PRIVATE m)
  endif()

endforeach()
