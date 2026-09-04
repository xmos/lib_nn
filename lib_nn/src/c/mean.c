// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "nn_operator.h"
#include <math.h>
#include <stdint.h>

/**
 * VPU optimized int8 mean calculation
 * Only support mean axis be the last axis, which end_dim_size = 1
 * mean_dim_size needs to be divided by 4
 */
extern void mean_int8_asm(
  const int8_t *input,
  int8_t *output,
  const int start_dim_size,
  const int mean_dim_size,
  const int8_t *vpu_buffer, // 64 byte vpu buffer
  const float in_zero_point_sum,
  const float out_zero_point,
  const float scale_mul);

// scale_mul is in_scale / out_scale
void mean_int8(const int8_t *input, int8_t *output, const int start_dim_size,
               const int mean_dim_size, const int end_dim_size,
               const float in_zero_point, const float out_zero_point,
               const float scale_mul) {

#if !defined(NN_USE_REF) && defined(__XS3A__)
  if ((end_dim_size == 1) && (mean_dim_size % 4 == 0)) {
    int8_t vpu_buffer[64];
    float in_zero_point_sum = (float)((int32_t)(in_zero_point*mean_dim_size));  // rounding it to keep the same performance as ref
    mean_int8_asm(
      input, output, start_dim_size, mean_dim_size, 
      vpu_buffer, 
      in_zero_point_sum, out_zero_point, scale_mul);
    return;
  }
#endif

  const int32_t start = -in_zero_point * mean_dim_size;
  for (int i = 0; i < start_dim_size; ++i) {
    const int i_mul = i * mean_dim_size * end_dim_size;
    for (int k = 0; k < end_dim_size; ++k) {
      int32_t accumulator = start;
// This is to avoid the for loop being badly misaligned
#ifdef __xcore__
      asm volatile(".align 16");
#endif
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

void mean_int16(const int16_t *input, int16_t *output, const int start_dim_size,
                const int mean_dim_size, const int end_dim_size,
                const float scale_mul) {

  for (int i = 0; i < start_dim_size; ++i) {
    const int i_mul = i * mean_dim_size * end_dim_size;
    for (int k = 0; k < end_dim_size; ++k) {
      int32_t accumulator = 0;
// This is to avoid the for loop being badly misaligned
#ifdef __xcore__
      asm volatile(".align 16");
#endif
      for (int j = 0; j < mean_dim_size; ++j) {
        const int index = i_mul + j * end_dim_size + k;
        accumulator += input[index];
      }

      // Calculate the mean and apply quantization
      float quantized_value = (float)accumulator * scale_mul;

      // Clamp the quantized value to int16 range
      if (quantized_value > 32767.0f)
        quantized_value = 32767.0f;
      else if (quantized_value < -32768.0f)
        quantized_value = -32768.0f;

      int out_index = i * end_dim_size + k;
      output[out_index] = (int16_t)(roundf(quantized_value));
    }
  }
}
