set(LIB_NAME lib_nn)
set(LIB_VERSION 0.6.0)
set(LIB_INCLUDES api)
set(LIB_DEPENDENT_MODULES "")

# compiler flags based on target
if(APP_BUILD_ARCH STREQUAL "xs3a") # xs3
    set(LIB_COMPILER_FLAGS -O3 -Wno-xcore-fptrgroup)
elseif(APP_BUILD_ARCH STREQUAL "vx4b") # vx4
    set(LIB_COMPILER_FLAGS -O3 -Wno-fptrgroup)
else() # native
    set(LIB_COMPILER_FLAGS -O3 -DNN_USE_REF) #TODO add -fsanitize=address,undefined -fno-omit-frame-pointer
endif()

XMOS_REGISTER_MODULE()
