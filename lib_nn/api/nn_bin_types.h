// Copyright 2020 XMOS LIMITED. This Software is subject to the terms of the 
// XMOS Public License: Version 1
#ifndef NN_BNN_TYPES_H
#define NN_BNN_TYPES_H

#include "nn_api.h"

#include <stdint.h>

C_API typedef int8_t bnn_bool_t;
C_API typedef int32_t bnn_b32_t;

C_API typedef struct bnn_b256_t {
  bnn_b32_t d[8]; 
} bnn_b256_t;

#endif //NN_BNN_TYPES_H