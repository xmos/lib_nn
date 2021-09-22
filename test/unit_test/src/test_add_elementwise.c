// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "nn_op_helper.h"
#include "nn_operator.h"
#include "tst_common.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "unity.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

#ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
#define CONFIG_SYMMETRIC_SATURATION_add_elementwise \
  CONFIG_SYMMETRIC_SATURATION_GLOBAL
#else
#ifndef CONFIG_SYMMETRIC_SATURATION_add_elementwise
#define CONFIG_SYMMETRIC_SATURATION_add_elementwise (0)
#endif
#endif

#if CONFIG_SYMMETRIC_SATURATION_add_elementwise
#define NEG_SAT_VAL (-127)
#else
#define NEG_SAT_VAL (-128)
#endif

char msg_buff[200];

#define LENGTH (16)

// Keep this real simple.
static void test_add_elementwise_case0() {
  PRINTF("%s...\n", __func__);

  int8_t WORD_ALIGNED Y[LENGTH];
  int8_t WORD_ALIGNED X1[LENGTH];
  int8_t WORD_ALIGNED X2[LENGTH];

  int8_t Y_expected[LENGTH];

  for (int i = 0; i < LENGTH; i++) {
    X1[i] = X2[i] = i;
  }

  nn_add_params_t params = {{{0, 0x0001}, {0, 0x0001}}, {0, 1}};

  for (int i = 0; i < LENGTH; i++) Y_expected[i] = i;

  add_elementwise(Y, X1, X2, &params, 0, LENGTH);
  TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LENGTH);
}
#undef LENGTH

#define LENGTH (128)

// Keep this real simple.
static void test_add_elementwise_case1() {
  PRINTF("%s...\n", __func__);

  int8_t WORD_ALIGNED Y[LENGTH];
  int8_t WORD_ALIGNED X1[LENGTH];
  int8_t WORD_ALIGNED X2[LENGTH];

  int8_t Y_expected[LENGTH];

  for (int i = 0; i < LENGTH; i++) {
    X1[i] = X2[i] = i;
  }

  nn_add_params_t params = {{{-8, 0x0001}, {-7, 0x0002}}, {-0x00008000, 8}};

  /*
      y[i] = 1*(((x1[i] << 8) + 2*(x2[i] << 7)) + bias) >> 8

      y[i] = (i * 2^8 + i * 2^8 + bias) >> 8
           = (2 * i * 2^8 + bias) >> 8
           = i * 2^1 + (bias>>8)
           = 2*i + (bias>>8)

      bias =  -128 * 2^8
      y[i] = 2*i - 0x8000>>8 = 2*i - 128

      y[0] = 2*0 - 128 = -128
      y[127] = 2 * 127 - 128 = 126
  */

  for (int i = 0; i < LENGTH; i++) Y_expected[i] = 2 * i - 128;

  unsigned start = 0;

  {                       // 0 <= i < 16
    unsigned count = 16;  // One full vector
    memset(Y, 0xCC, sizeof(Y));
    add_elementwise(Y, X1, X2, &params, start, count);
    TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
    TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count],
                                LENGTH - (start + count));
    start += count;
  }

  {                      // 16 <= i < 20
    unsigned count = 4;  // Less than one vector
    memset(Y, 0xCC, sizeof(Y));
    add_elementwise(Y, X1, X2, &params, start, count);
    TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
    TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
    TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count],
                                LENGTH - (start + count));
    start += count;
  }

  {                       // 20 <= i < 52
    unsigned count = 32;  // Two full vectors
    memset(Y, 0xCC, sizeof(Y));
    add_elementwise(Y, X1, X2, &params, start, count);
    TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
    TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
    TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count],
                                LENGTH - (start + count));
    start += count;
  }

  {                       // 52 <= i < 128
    unsigned count = 76;  // 4 vectors and change.
    memset(Y, 0xCC, sizeof(Y));
    add_elementwise(Y, X1, X2, &params, start, count);
    TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
    TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
  }
}
#undef LENGTH

void test_add_elementwise() {
  srand(563456);

  UNITY_SET_FILE();

  RUN_TEST(test_add_elementwise_case0);
  RUN_TEST(test_add_elementwise_case1);
}