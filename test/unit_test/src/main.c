// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdio.h>

#include "unity.h"

#define CALL(F)                                                                \
  do {                                                                         \
    void F();                                                                  \
    F();                                                                       \
  } while (0)

int main(void) {
  UNITY_BEGIN();

  CALL(test_softmax);
  CALL(test_conv2d_binary_regression);
  CALL(test_transposeconv2d_regression);

  CALL(test_conv2d_regression);
  CALL(test_output_transforms);
  CALL(test_aggregate_fns);
  CALL(test_conv2d_dw_regression);
  CALL(test_mem_cpy_fns);

  CALL(test_vpu_memcpy);
  CALL(test_vpu_memset);

  CALL(test_bsign_8);
  CALL(test_pad);
  CALL(test_3_to_4);
  CALL(test_maxpool);
  CALL(test_add_elementwise);
  CALL(test_mul_elementwise);

  CALL(test_expand_8_to_16);

  CALL(test_output_transform_16);
  CALL(test_multiply_int16);
  CALL(test_dequantize_int16);
  CALL(test_quantize_int16);
  CALL(test_add_int16);

  return UNITY_END();
}
