// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"
#include "xs3_vpu.h"

#define LHS_ROW_SIZE 128
#define CHANNEL_SIZE 256
#define RHS_COL_SIZE 128
#define GUARD_VALUE 0xAA

TEST_GROUP(group_matmul);
TEST_SETUP(group_matmul) { srand(563456); }
TEST_TEAR_DOWN(group_matmul) {}
TEST_GROUP_RUNNER(group_matmul) {
  RUN_TEST_CASE(group_matmul, test_matmul);
#ifdef TEST_BUILD_NATIVE
  RUN_TEST_CASE(group_matmul, test_matmul_full);
#endif // TEST_BUILD_NATIVE
}

static void impl_test_matmul(const unsigned lhs_row,
                             const unsigned channel,
                             const unsigned rhs_col,
                             const unsigned out_offset) {
#if defined(__VX4A__) || defined(__VX4B__)
  TEST_IGNORE_MESSAGE("mat_mul_real_int8 not implemented on VX4");
#endif

  double lhsScale = 1. / 128.;
  double rhsScale = 1. / 128.;
  double outputScale = 1. / 128.;
  int8_t lhsZeroPoint = 0;
  int8_t rhsZeroPoint = 0;
  int8_t outputZeroPoint = 0;
  int8_t WORD_ALIGNED lhs[LHS_ROW_SIZE * CHANNEL_SIZE];
  int8_t WORD_ALIGNED rhs[RHS_COL_SIZE * CHANNEL_SIZE]; // matmul requires rhs to be in column-major order
  int8_t WORD_ALIGNED out[LHS_ROW_SIZE * RHS_COL_SIZE];
  int8_t expected[LHS_ROW_SIZE * RHS_COL_SIZE];
  int8_t WORD_ALIGNED vpu_buf0[32 * 2];
  int8_t WORD_ALIGNED vpu_buf1[32 * 2];
  int8_t WORD_ALIGNED vpu_buf2[32 * 2];
  pseudo_rand_bytes((char*)lhs, LHS_ROW_SIZE * CHANNEL_SIZE);
  pseudo_rand_bytes((char*)rhs, CHANNEL_SIZE * RHS_COL_SIZE);

  nn_mat_mul_real_params_t params = {
      .lhs_zp = lhsZeroPoint,
      .rhs_zp = rhsZeroPoint,
      .in_zp_sum = channel * lhsZeroPoint * rhsZeroPoint,
      .out_zp = outputZeroPoint,
      .scale = lhsScale * rhsScale / outputScale,
      .lhs_row_size = lhs_row,
      .channel_size = channel,
      .rhs_col_size = rhs_col};

  memset(out, GUARD_VALUE, LHS_ROW_SIZE * RHS_COL_SIZE);

  // Calculate the expected output
  for (int l = 0; l < lhs_row; l++) {
    for (int r = 0; r < rhs_col; r++) {
      float acc = 0;
      for (int ch = 0; ch < channel; ch++) {
        float lhs_val = lhsScale * ((float)lhs[l * channel + ch] - lhsZeroPoint);
        float rhs_val = rhsScale * ((float)rhs[r * channel + ch] - rhsZeroPoint);
        acc += lhs_val * rhs_val;
      }
      // Qunatize float to int8 and clamp to int8 range
      acc = (acc / outputScale) + outputZeroPoint;
      if (acc > 127.0f)
        acc = 127.0f;
      else if (acc < -128.0f)
        acc = -128.0f;
      expected[l * rhs_col + r] = (int8_t)(roundf(acc));
    }
  }

  mat_mul_real_int8(
      &params,
      vpu_buf0, vpu_buf1, vpu_buf2,
      lhs, rhs, out + out_offset);

  TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out+out_offset, lhs_row * rhs_col);
  for (unsigned i = 0; i < LHS_ROW_SIZE * RHS_COL_SIZE; i++) {
    if (i < out_offset || i > out_offset + lhs_row * rhs_col) {
      TEST_ASSERT_EQUAL(GUARD_VALUE, *((uint8_t*)(&out[i])));
    }
  }
}

TEST(group_matmul, test_matmul) {
  // Unaligned matrix test
  impl_test_matmul(65, 48, 72, 0);
  impl_test_matmul(65, 48, 72, 2);
  // // Small matrix test
  impl_test_matmul(8, 8, 8, 0);
  impl_test_matmul(8, 8, 8, 1);
}

#ifdef TEST_BUILD_NATIVE
TEST(group_matmul, test_matmul_full) {
  // Full matrix test
  impl_test_matmul(128, 256, 125, 0);
  // Unaligned matrix test
  impl_test_matmul(65, 248, 72, 0);
  impl_test_matmul(65, 248, 72, 2);
  impl_test_matmul(65, 248, 72, 5);
  impl_test_matmul(65, 248, 72, 8);
  impl_test_matmul(65, 248, 72, 16);
  // Small matrix test
  impl_test_matmul(8, 8, 8, 0);
  impl_test_matmul(8, 8, 8, 1);
  impl_test_matmul(8, 8, 8, 3);
  impl_test_matmul(8, 8, 8, 5);
  impl_test_matmul(8, 8, 8, 15);
}
#endif // TEST_BUILD_NATIVE