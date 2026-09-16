# This file provides overrides to the Doxyfile configuration

PROJECT_NAME           = lib_nn
PROJECT_BRIEF          = "Neural network library"

INPUT                  = ../lib_nn/api

PREDEFINED             = C_API= EXTERN_C= ACTIVATION_FUNCTION= __DOXYGEN__=1

SORT_MEMBER_DOCS        = YES
SORT_BRIEF_DOCS         = YES
TYPEDEF_HIDES_STRUCT    = YES
ENABLE_PREPROCESSING    = YES
MACRO_EXPANSION         = YES
