// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_operator.h"

#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

#define BUFFER_SIZE 128
#define GUARD_VALUE 0xAA

static uint8_t lut[256];
static uint8_t inputs[BUFFER_SIZE] __attribute__((aligned(4)));
static uint8_t outputs[BUFFER_SIZE] __attribute__((aligned(4)));

TEST_GROUP(group_lookup8);
TEST_SETUP(group_lookup8) {
  for (unsigned i = 0; i < 256; i++) {
    unsigned doubled = i * 2;
    lut[i] = (doubled > 255) ? 255 : (uint8_t)doubled;
  }
  for (unsigned i = 0; i < BUFFER_SIZE; i++) {
    inputs[i] = (uint8_t)(i * i);
  }
}
TEST_TEAR_DOWN(group_lookup8) {}
TEST_GROUP_RUNNER(group_lookup8) {
  RUN_TEST_CASE(group_lookup8, test_lookup8);
// Sweeping every (elm_start, elm_count) pair is slow under simulation, so only run it natively.
#ifdef TEST_BUILD_NATIVE
  RUN_TEST_CASE(group_lookup8, test_lookup8_full);
#endif // TEST_BUILD_NATIVE
}

static void impl_test_lookup8(const unsigned max_elm_start,
                               const unsigned max_elm_count) {
  for (unsigned elm_start = 0; elm_start <= max_elm_start; elm_start++) {
    for (unsigned elm_count = 0; elm_count <= max_elm_count; elm_count++) {
      memset(outputs, GUARD_VALUE, BUFFER_SIZE);

      lookup8(outputs, inputs, lut, elm_start, elm_count);

      for (unsigned i = 0; i < BUFFER_SIZE; i++) {
        if (i < elm_start || i >= elm_start + elm_count) {
          TEST_ASSERT_EQUAL(GUARD_VALUE, outputs[i]);
        } else {
          TEST_ASSERT_EQUAL(lut[inputs[i]], outputs[i]);
        }
      }
    }
  }
}

TEST(group_lookup8, test_lookup8) {
  impl_test_lookup8(4, 16);
}

#ifdef TEST_BUILD_NATIVE
TEST(group_lookup8, test_lookup8_full) {
  impl_test_lookup8(8, 64);
}
#endif // TEST_BUILD_NATIVE
