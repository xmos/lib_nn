#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "nn_operator.h"
#include "tst_common.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

#define LENGTH (16)

static void test_softmax_case0() {
  PRINTF("%s...\n", __func__);
  int8_t WORD_ALIGNED Y[LENGTH];
  int8_t WORD_ALIGNED X[LENGTH];

  int8_t Y_expected[LENGTH];

  for (int i = 0; i < LENGTH; i++) {
    X[i] = i;
  }
  float lut[256];
  const int8_t zero_point = -128;
  const float scale = 0.00390625;
  softmax_ref(Y_expected, X, zero_point, scale, LENGTH);
  generateExpLUT(zero_point, scale, lut);
  float sum;
  exp_sum(&sum, X, lut, 0, LENGTH);
  const float inv_sum = 1.0f / sum;
  exp_div(Y, X, lut, inv_sum, 0, LENGTH);
  TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LENGTH);
}
