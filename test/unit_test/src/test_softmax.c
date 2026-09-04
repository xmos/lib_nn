// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdlib.h>
#include <math.h>

#include "nn_operator.h"
#include "tst_common.h"
#include "unity_fixture.h"

#define LENGTH (16)

TEST_GROUP_RUNNER(group_softmax) {
  RUN_TEST_CASE(group_softmax, case0);
}

TEST_GROUP(group_softmax);
TEST_SETUP(group_softmax)    { srand(563456); }
TEST_TEAR_DOWN(group_softmax) {}

TEST(group_softmax, case0) {
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
