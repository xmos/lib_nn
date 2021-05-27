// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_BNN_TYPES_H
#define NN_BNN_TYPES_H
#include <stdint.h>

typedef int8_t bnn_bool_t;
typedef int32_t bnn_b32_t;

typedef struct bnn_b256_t {
  bnn_b32_t d[8];
} bnn_b256_t;

#endif  // NN_BNN_TYPES_H