// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#ifndef NN_MEAN_H_
#define NN_MEAN_H_

#include <stdint.h>

void mean_int8(const int8_t *input, int8_t *output, int start_dim_size,
               int mean_dim_size, int end_dim_size, float in_zero_point,
               float out_zero_point, float scale_mul);
void mean_int16(const int16_t *input, int16_t *output, int start_dim_size,
                int mean_dim_size, int end_dim_size, float scale_mul);

#endif
