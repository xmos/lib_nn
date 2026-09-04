// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_ADD_ELEMENTWISE_H_
#define NN_ADD_ELEMENTWISE_H_

#include <stdint.h>
#include "nn_api.h"

typedef struct {
  int16_t m1[16];
  int16_t m2[16];
  int16_t shift[16];
  int16_t bias_hi[16];
  int16_t bias_lo[16];
} nn_add_params_t;

void add_elementwise(int8_t Y[], const int8_t X1[], const int8_t X2[],
                     nn_add_params_t *p, const int elm_start,
                     const int elm_count);

#endif
