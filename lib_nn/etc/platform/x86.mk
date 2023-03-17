

PLATFORM_NAME = x86

PLATFORM_FLAGS_DEFAULT := 
PLATFORM_INCLUDES :=

ifeq ($(OS),Windows_NT)
  ifeq ($(findstring windows32,$(shell uname -s)),windows32)
    PLATFORM_EXE_SUFFIX = .a
  else
    PLATFORM_EXE_SUFFIX = .exe
  endif
else
  PLATFORM_EXE_SUFFIX = 
endif

PLATFORM_FLAGS := $(PLATFORM_FLAGS_DEFAULT) -DTF_LITE_DISABLE_X86_NEON
# PLATFORM_FLAGS := $(PLATFORM_FLAGS_DEFAULT) -DTF_LITE_DISABLE_X86_NEON -Wall -fsanitize=undefined -fsanitize=integer -fsanitize=implicit-conversion -fsanitize=address -fsanitize-recover=address


CC := cc
XCC := c++
CXX := c++

AR_FLAGS := -r
CC_FLAGS  := -g -O3
XCC_FLAGS := -g -O3
CXX_FLAGS := -g -O3 -std=c++11

LD_FLAGS  := -L/usr/local/lib -lm -lstdc++

ifeq ($(SANITIZE),true)
  CC:=clang
  CC_FLAGS  := $(CC_FLAGS) -fsanitize=address -fsanitize-recover=address
  LD_FLAGS  := -lasan $(LD_FLAGS) # NOTE: -lasan must be first
endif

ifeq ($(MEMORY_SAFE),true)
  CC_FLAGS  := $(CC_FLAGS) -DMEMORY_SAFE
endif
