// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdlib.h>
#include <math.h>

#include "nn_operator.h"
#include "tst_common.h"
#include "unity_fixture.h"

#define LENGTH (16)

TEST_GROUP_RUNNER(group_softmax) {
  RUN_TEST_CASE(group_softmax, high_level);
  RUN_TEST_CASE(group_softmax, inv_sum);
  RUN_TEST_CASE(group_softmax, exp_lut);
  RUN_TEST_CASE(group_softmax, exp_sum);
  RUN_TEST_CASE(group_softmax, exp_div);
  RUN_TEST_CASE(group_softmax, single);
  RUN_TEST_CASE(group_softmax, high_level_random);
}

TEST_GROUP(group_softmax);
TEST_SETUP(group_softmax)    { srand(563456); }
TEST_TEAR_DOWN(group_softmax) {}

TEST(group_softmax, high_level) {
  int8_t WORD_ALIGNED Y[LENGTH];
  int8_t WORD_ALIGNED X[LENGTH];
  const int8_t Y_expected[LENGTH] = {
      -112, -112, -112, -112, -112, -112, -112, -112,
      -112, -112, -112, -112, -112, -112, -112, -112};

  for (int i = 0; i < LENGTH; i++) {
    X[i] = i;
  }

  const int8_t zero_point = -128;
  const float scale = 0.00390625;
  softmax(Y, X, zero_point, scale, LENGTH);
  TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LENGTH);
}

TEST(group_softmax, inv_sum) {
  const float expected = 1.0f;
  float sums[5] = {11.11, 22.22, 33.33, 44.44, 144.9};
  float inv_sum;
  softmax_calculate_inv_sum(&inv_sum, sums);
  TEST_ASSERT_EQUAL_FLOAT(expected, inv_sum);
}

TEST(group_softmax, exp_lut) {
  float lut[256];
  softmax_generate_exp_lut(128, 1.0f, lut);
  TEST_ASSERT_EQUAL_FLOAT(1.0f, lut[128]);
}

TEST(group_softmax, exp_sum) {
  int8_t X[4] = {-2, -1, 0, 1};
  float lut[256];
  float sum;
  for (unsigned i = 0; i < 256; i++) lut[i] = 1.0f;

  softmax_exp_sum(&sum, X, lut, 1, 2);

  TEST_ASSERT_EQUAL_FLOAT(2.0f, sum);
}

TEST(group_softmax, exp_div) {
  int8_t X[4] = {-2, -1, 0, 1};
  int8_t Y[4] = {0};
  float lut[256];
  for (unsigned i = 0; i < 256; i++) lut[i] = 1.0f;

  softmax_exp_div(Y, X, lut, 64.0f, 0, 4);

  const int8_t expected[4] = {-64, -64, -64, -64};
  TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, 4);
}

TEST(group_softmax, single) {
  int8_t X[4] = {-2, -1, 0, 1};
  int8_t Y[4];
  float lut[256];
  for (unsigned i = 0; i < 256; i++) lut[i] = 1.0f;

  softmax_single(Y, X, lut, 4);

  const int8_t expected[4] = {-64, -64, -64, -64};
  TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, 4);
}

TEST(group_softmax, high_level_random) {
  int8_t WORD_ALIGNED output[LENGTH];
  int8_t WORD_ALIGNED input[LENGTH];
  int8_t expected[LENGTH];
  const int8_t zero_point = -128;
  const float scale = 0.00390625f;
  double sum = 0.0;

  for (int i = 0; i < LENGTH; i++) {
    input[i] = pseudo_rand_int8();
    sum += exp(((double)input[i] - zero_point) * scale);
  }

  softmax(output, input, zero_point, scale, LENGTH);

  for (int i = 0; i < LENGTH; i++) {
    int value = (int)round(exp(((double)input[i] - zero_point) * scale) /
                            sum * 256.0) - 128;
    if (value > 127) value = 127;
    if (value < -128) value = -128;
    expected[i] = value;
  }

  TEST_ASSERT_EQUAL_INT8_ARRAY(expected, output, LENGTH);
}
