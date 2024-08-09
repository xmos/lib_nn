// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "nn_operator.h"
#include <math.h>
#include <stdint.h>

// scale_mul is in_scale / out_scale
void mean_int8(const int8_t *input, int8_t *output, const int start_dim_size,
               const int mean_dim_size, const int end_dim_size,
               const float in_zero_point, const float out_zero_point,
               const float scale_mul) {

  const int32_t start = -in_zero_point * mean_dim_size;
  for (int i = 0; i < start_dim_size; ++i) {
    const int i_mul = i * mean_dim_size * end_dim_size;
    for (int k = 0; k < end_dim_size; ++k) {
      int32_t accumulator = start;
      for (int j = 0; j < mean_dim_size; ++j) {
        const int index = i_mul + j * end_dim_size + k;
        accumulator += input[index];
      }

      // Calculate the mean and apply quantization
      float quantized_value = (float)accumulator * scale_mul + out_zero_point;

      // Clamp the quantized value to int8 range
      if (quantized_value > 127.0f)
        quantized_value = 127.0f;
      else if (quantized_value < -128.0f)
        quantized_value = -128.0f;

      int out_index = i * end_dim_size + k;
      output[out_index] = (int8_t)(roundf(quantized_value));
    }
  }
}
