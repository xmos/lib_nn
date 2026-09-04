// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_MUL_H_
#define NN_MUL_H_

#include <stdint.h>
#include "nn_api.h"

typedef struct nn_mul_params_t {
  int8_t in1_zero_point;
  int8_t in2_zero_point;
  int16_t bias;
  int16_t scalar;
  int16_t vlashr_shr;
} nn_mul_params_t;

void mul_boggle(nn_mul_params_t *params, double in1Scale, double in2Scale,
                double outputScale, int8_t in1ZeroPoint, int8_t in2ZeroPoint,
                int8_t outputZeroPoint);
void mul_elementwise(const int8_t *in1_data, const int8_t *in2_data,
                     int element_count, nn_mul_params_t *params,
                     int8_t *out_data);

#endif
