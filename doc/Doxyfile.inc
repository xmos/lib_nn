# This file provides overrides to the Doxyfile configuration

PROJECT_NAME           = lib_nn
PROJECT_BRIEF          = "Neural network library"

INPUT                  = ../lib_nn/api

PREDEFINED             = C_API= ACTIVATION_FUNCTION= __DOXYGEN__=1
PREDEFINED            += MEM_FN_GROUP= AGG_FN_GROUP= OT_FN_GROUP= MEMCPY_FPTRGROUP= MACC_FPTR=

SORT_MEMBER_DOCS        = YES
SORT_BRIEF_DOCS         = YES
TYPEDEF_HIDES_STRUCT    = YES
ENABLE_PREPROCESSING    = YES
MACRO_EXPANSION         = YES
