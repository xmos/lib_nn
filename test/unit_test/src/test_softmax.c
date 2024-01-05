#include <stdio.h>
#include <string.h>
#include <assert.h>
#include <math.h>

#include "nn_operator.h"
#include "tst_common.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

void generateExpLUT(float zero_point, float scale, float *lut) {
  for (int i = 0; i < 256; i++) {
    int8_t quantized_val = i;
    float real_val = (quantized_val - zero_point) * scale;
    lut[i] = expf(real_val);
  }
}

// Reference implementation: as accurate as possible
// Round to int before casting
// Minus max float value to avoid numerical instability:
// exp(arr) / sum(exp(arr)) = exp(arr + C) / sum(exp(arr + C))
void softmax_ref(int8_t *Y, const int8_t *X, const float zero_point,
                 const float scale, const int length) {
  int8_t max_val = X[0];
  for (int i = 1; i < length; i++) {
    max_val = X[i] > max_val ? X[i] : max_val;
  }
  const float max_val_f = ((float)max_val - zero_point) * scale;
  float sum = 0;
  for (int i = 0; i < length; i++) {
    sum += expf(((float)X[i] - zero_point) * scale - max_val_f);
  }
  for (int i = 0; i < length; i++) {
    const float real_val =
        (expf(((float)X[i] - zero_point) * scale - max_val_f) / sum);
    Y[i] = (int8_t)(roundf(real_val * 255.0 - 128.0));
  }
}

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
