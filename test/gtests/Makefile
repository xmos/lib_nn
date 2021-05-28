


PLATFORM ?= xcore

ifndef XMOS_TOOL_PATH
  $(info Compiling for x86.)
  PLATFORM = x86
endif

VERBOSE ?= 

PLATFORM_MF = ../../lib_nn/etc/platform/$(strip $(PLATFORM)).mk
COMMON_MF = ../../lib_nn/etc/common.mk
include $(PLATFORM_MF)
include $(COMMON_MF)

ifneq ($(VERBOSE),$(EMPTY_STR))
  $(info Building for platform: $(PLATFORM) )
endif

help:
	$(info *************************************************************************************)
	$(info *             make targets                                                          *)
	$(info *                                                                                   *)
	$(info *   help:      Display this message                                                 *)
	$(info *   clean:     Clean the build directory                                            *)
	$(info *   build:     Build the unit tests                                                 *)
	$(info *                                                                                   *)
	$(info *************************************************************************************)


APP_NAME := unit_test

TARGET_DEVICE = XCORE-AI-EXPLORER

NN_PATH := ../../lib_nn

DEPS_PATH := ./../deps
SHARED_PATH := ./../shared

GTEST_PATH := $(abspath $(DEPS_PATH)/googletest)
COMPUTE_ENGINE_PATH := $(DEPS_PATH)/compute-engine
FLAT_BUFFERS_PATH := $(DEPS_PATH)/flatbuffers
RUY_PATH := $(DEPS_PATH)/ruy
TENSORFLOW_PATH := $(DEPS_PATH)/tensorflow
GEMMLOWP_PATH := $(DEPS_PATH)/gemmlowp

BUILD_DIR := .build
BIN_DIR := bin
EXE_DIR   := $(BIN_DIR)/$(PLATFORM)
OBJ_DIR   := $(BUILD_DIR)/$(PLATFORM)
LIB_DIR   := $(OBJ_DIR)/lib
EMPTY_STR :=
LIB_NN_FILE := $(LIB_DIR)/lib_nn.a

ifneq ($(VERBOSE),$(EMPTY_STR))
  $(info XSCOPE_CONFIG: $(XSCOPE_CONFIG) )
  $(info NN_PATH: $(NN_PATH) )
  $(info LIB_NN_FILE: $(LIB_NN_FILE) )
  $(info BUILD_DIR: $(BUILD_DIR) )
  $(info OBJ_DIR: $(OBJ_DIR) )
endif

INCLUDES := $(NN_PATH)/api $(NN_PATH)/src $(COMPUTE_ENGINE_PATH) $(FLAT_BUFFERS_PATH)/include
INCLUDES += $(RUY_PATH) $(TENSORFLOW_PATH) $(GEMMLOWP_PATH) $(GTEST_PATH)/googletest/include
INCLUDES += $(SHARED_PATH)/include
SOURCE_DIRS := src $(SHARED_PATH)/src
SOURCE_FILE_EXTENSIONS := c cpp cc

SOURCE_FILES := 

ifneq ($(VERBOSE),$(EMPTY_STR))
  $(info SOURCE_FILE_EXTENTIONS: $(SOURCE_FILE_EXTENSIONS) )
  $(info INCLUDES: $(INCLUDES) )
  $(info SOURCE_DIRS: $(SOURCE_DIRS) )
endif

ifeq ($(strip $(PLATFORM)),$(strip xcore))
  PLATFORM_FLAGS += -target=$(TARGET_DEVICE)
endif

USE_REF = 0

#######################################################
# SOURCE FILE SEARCH
#######################################################

# Recursively search within SOURCE_DIRS for files with extensions from SOURCE_FILE_EXTENSIONS
SOURCE_FILES += $(strip $(foreach src_dir,$(SOURCE_DIRS),\
                        $(call rwildcard,./$(src_dir),$(SOURCE_FILE_EXTENSIONS:%=*.%))))


ifneq ($(VERBOSE),$(EMPTY_STR))
  $(info Library source files:)
  $(foreach f,$(SOURCE_FILES), $(info $f) )
  $(info )
endif


#######################################################
# COMPONENT OBJECT FILES
#######################################################

OBJECT_FILES := $(patsubst %, $(OBJ_DIR)/%.o, $(SOURCE_FILES:./%=%))

# Set object file prerequisites
$(OBJECT_FILES) : $(OBJ_DIR)/%.o: %


ifneq ($(VERBOSE),$(EMPTY_STR))
  $(info $(APP_NAME) object files:)
  $(foreach f,$(OBJECT_FILES), $(info $f) )
  $(info )
endif

#########
## Recipe-scoped variables for building objects.
#########

# OBJ_FILE_TYPE
# The source file's file type
$(eval $(foreach ext,$(SOURCE_FILE_EXTENSIONS),   \
           $(filter %.$(ext).o,$(OBJECT_FILES)): OBJ_FILE_TYPE = $(ext)$(newline)))

# OBJ_TOOL
# Maps from file extension to the tool type (not necessarily 1-to-1 mapping with
# file extension). This simplifies some of the code below.
$(OBJECT_FILES): OBJ_TOOL = $(MAP_COMP_$(OBJ_FILE_TYPE))

# OBJ_COMPILER: Compilation program for this object
$(OBJECT_FILES): OBJ_COMPILER = $($(OBJ_TOOL))

# $(1) - Tool
# $(2) - File extension
tf_combo_str = $(1)_$(2) $(1) $(2)
flags_combo_str = GLOBAL_FLAGS PLATFORM_FLAGS $(patsubst %,%_FLAGS,$(tf_combo_str))
includes_combo_str = INCLUDES PLATFORM_INCLUDES $(patsubst %,%_INCLUDES,$(tf_combo_str))

$(OBJECT_FILES): OBJ_FLAGS = $(strip $(foreach grp,$(call flags_combo_str,$(OBJ_TOOL),$(OBJ_FILE_TYPE)),$($(grp))))
$(OBJECT_FILES): OBJ_INCLUDES = $(strip $(foreach grp,$(call includes_combo_str,$(OBJ_TOOL),$(OBJ_FILE_TYPE)),$($(grp))))

###
# make target for each object file.
#
$(OBJECT_FILES):
	$(info [$(APP_NAME)] Compiling $<)
	@$(OBJ_COMPILER) $(OBJ_FLAGS) $(addprefix -I,$(OBJ_INCLUDES)) -o $@ -c $<

###
# If the -MMD flag is used when compiling, the .d files will contain additional header 
# file prerequisites for each object file. Otherwise it won't know to recompile if only
# header files have changed, for example.
-include $(OBJECT_FILES:%.o=%.d)

#######################################################
# GOOGLETEST TARGETS
#######################################################

GTEST_LIB := $(LIB_DIR)/libgtest.a 
GTEST_MAIN_LIB := $(LIB_DIR)/libgtest_main.a
GMOCK_LIB := $(LIB_DIR)/libgmock.a
GMOCK_MAIN_LIB := $(LIB_DIR)/libgmock_main.a

$(GTEST_LIB): $(GTEST_PATH)/build/lib/libgtest.a
	@cp $^ $@
$(GTEST_MAIN_LIB): $(GTEST_PATH)/build/lib/libgtest_main.a
	@cp $^ $@
$(GMOCK_LIB): $(GTEST_PATH)/build/lib/libgmock.a
	@cp $^ $@
$(GMOCK_MAIN_LIB): $(GTEST_PATH)/build/lib/libgmock_main.a
	@cp $^ $@

$(GTEST_PATH)/build/lib/libgtest.a: $(GTEST_PATH)/build/Makefile
	@$(MAKE) -C $(GTEST_PATH)/build all
$(GTEST_PATH)/build/lib/libgtest_main.a: $(GTEST_PATH)/build/Makefile
	@$(MAKE) -C $(GTEST_PATH)/build all
$(GTEST_PATH)/build/lib/libgmock.a: $(GTEST_PATH)/build/Makefile
	@$(MAKE) -C $(GTEST_PATH)/build all
$(GTEST_PATH)/build/lib/libgmock_main.a: $(GTEST_PATH)/build/Makefile
	@$(MAKE) -C $(GTEST_PATH)/build all

#######################################################
# LIBRARY TARGETS
#######################################################

# Libraries are built using a recursive make call.

LIB_MAKE_OPTS := VERBOSE=$(VERBOSE) BUILD_DIR=$(abspath $(BUILD_DIR)/lib_nn) LIB_DIR=$(abspath $(LIB_DIR)) \
                 PLATFORM=$(PLATFORM) TARGET_DEVICE=$(TARGET_DEVICE)

DEP_MAKE_OPTS := VERBOSE=$(VERBOSE) OBJ_DIR=$(abspath $(OBJ_DIR)) PLATFORM=$(PLATFORM) \
                 TARGET_DEVICE=$(TARGET_DEVICE)

force_look:
	@true

$(LIB_NN_FILE): force_look
	@$(MAKE) -C $(NN_PATH) $(abspath $@ ) $(LIB_MAKE_OPTS)

ALL_STATIC_LIBS += $(LIB_NN_FILE) $(DEPENDENCY_LIBS) 
ALL_STATIC_LIBS += $(GTEST_LIB) $(GTEST_MAIN_LIB)

#######################################################
# HOUSEKEEPING
#######################################################

# Annoying problem when doing parallel build is directory creation can fail if two threads both try to do it.
# To solve that, make all files in the build directory dependent on a sibling "marker" file, the recipe for which
# is just the creation of that directory and file.
$(eval  $(foreach bfile,$(OBJECT_FILES),       \
            $(bfile): | $(dir $(bfile)).marker $(newline)))
			
$(eval  $(foreach bfile,$(ALL_STATIC_LIBS),       \
            $(bfile): | $(dir $(bfile)).marker $(newline)))

$(BUILD_DIR)/%.marker:
	$(info Creating dir: $(dir $@))
	$(call mkdir_cmd,$@)
	@touch $@



#######################################################
# APPLICATION TARGETS
#######################################################

#
# Application executable files
APP_EXE_FILE = $(EXE_DIR)/$(APP_NAME)$(PLATFORM_EXE_SUFFIX)

$(APP_EXE_FILE): $(OBJECT_FILES) $(ALL_STATIC_LIBS) $(XSCOPE_CONFIG)

$(APP_EXE_FILE): REQUIRED_LIBRARIES = $(ALL_STATIC_LIBS)


$(APP_EXE_FILE):
	$(call mkdir_cmd,$@)
	$(info Linking binary $@)
	@$(XCC) $(LDFLAGS)                      \
		$(APP_FLAGS)                        \
		$(PLATFORM_FLAGS)                   \
		$(OBJECT_FILES)                     \
		$(XSCOPE_CONFIG)					\
		-o $@                               \
		$(REQUIRED_LIBRARIES) -pthread
		

# #######################################################
# # OTHER TARGETS
# #######################################################

.PHONY: help all build clean xcore ref

all: build

compile: $(OBJECT_FILES)

build: $(APP_EXE_FILE)

clean:
	$(info Cleaning project...)
	rm -rf $(BUILD_DIR)
