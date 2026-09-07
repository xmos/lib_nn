// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdint.h>

#include "nn_operator.h"
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(group_argmax16);
TEST_SETUP(group_argmax16) {}
TEST_TEAR_DOWN(group_argmax16) {}
TEST_GROUP_RUNNER(group_argmax16) {
  RUN_TEST_CASE(group_argmax16, test_simple);
  RUN_TEST_CASE(group_argmax16, test_random);
}

TEST(group_argmax16, test_simple) {
  // find the largest value and keep the first index on a tie
  int16_t input[] = {4, -7, 12, 12, 3};
  int32_t output = -1;
  argmax_16(&output, input, 5);
  TEST_ASSERT_EQUAL_INT32(2, output);
}

TEST(group_argmax16, test_random) {
  // check random vectors against a simple reference implementation
  int16_t input[64];
  int32_t output;
  int32_t expected;

  for (unsigned run = 0; run < 10; ++run) {
    const int32_t count = 1 + (pseudo_rand_uint32() % 64);

    for (int32_t i = 0; i < count; ++i) {
      input[i] = pseudo_rand_int16();
    }

    expected = 0;
    for (int32_t i = 1; i < count; ++i) {
      if (input[i] > input[expected]) {
        expected = i;
      }
    }
    argmax_16(&output, input, count);
    TEST_ASSERT_EQUAL_INT32(expected, output);
  }
}
