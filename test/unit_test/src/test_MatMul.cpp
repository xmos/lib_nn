// Copyright 2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <algorithm>

#include "AggregateFn.hpp"

extern "C" {
#include "unity.h"
#include "unity_fixture.h"
}

using namespace nn;

extern "C" {

// Exercises public matrix-multiply APIs on native and device targets, checking
// the expected accumulator values written to the VPURingBuffer.

TEST_GROUP(group_mat_mul);
TEST_SETUP(group_mat_mul) {}
TEST_TEAR_DOWN(group_mat_mul) {}
TEST_GROUP_RUNNER(group_mat_mul) {
  RUN_TEST_CASE(group_mat_mul, Test_GenericInt8);
  RUN_TEST_CASE(group_mat_mul, Test_GenericBinary);
  RUN_TEST_CASE(group_mat_mul, Test_DirectInt8);
  RUN_TEST_CASE(group_mat_mul, Test_DirectBinary);
  RUN_TEST_CASE(group_mat_mul, Test_DirectInt16);
  RUN_TEST_CASE(group_mat_mul, Test_DirectInt16x8);
  RUN_TEST_CASE(group_mat_mul, Test_DepthwiseInt8);
  RUN_TEST_CASE(group_mat_mul, Test_DepthwiseInt16);
}

constexpr int chn_n = VPU_INT16_EPV;  // Output channel count.
constexpr int elm_n = XS3_VPU_VREG_WIDTH_BYTES;  // Elements per channel.
constexpr int ocg = 0;  // Output channel group.
constexpr int dir_wgt_n = chn_n * elm_n;
constexpr int gen_wgt_n = dir_wgt_n;
constexpr int dw_wgt_n = chn_n;
constexpr int i16_wgt_n = dir_wgt_n + VPU_INT16_EPV;
const WindowGeometry win(1, 1, 1, 1, 1, 1);
const ImageGeometry img_i8(1, 1, elm_n);
const ImageGeometry img_i16(1, 1, elm_n, 16);
int8_t inp_i8[elm_n] WORD_ALIGNED;
int8_t wgt_i8[gen_wgt_n] WORD_ALIGNED;
int16_t inp_i16[elm_n] WORD_ALIGNED;
int16_t wgt_i16[i16_wgt_n] WORD_ALIGNED;
VPURingBuffer A WORD_ALIGNED;

static int32_t accumulator_value(const VPURingBuffer &accumulator, int channel) {
  int32_t value;
  ((int16_t *)&value)[0] = accumulator.vR[channel];
  ((int16_t *)&value)[1] = accumulator.vD[channel];
  return value;
}

static void assert_accumulator_value(const VPURingBuffer &accumulator, int32_t expected) {
  for (int channel = 0; channel < VPU_INT16_EPV; ++channel) {
    TEST_ASSERT_EQUAL_INT32(expected, accumulator_value(accumulator, channel));
  }
}

TEST(group_mat_mul, Test_GenericInt8) {
  MatMulInt8 gen(chn_n, elm_n);
  mat_mul_generic_params_t prm = gen.getParams();
  std::fill_n(inp_i8, elm_n, 1);
  std::fill_n(wgt_i8, gen_wgt_n, 1);

  mat_mul_generic_int8(&prm, &A, inp_i8, ocg, wgt_i8);
  assert_accumulator_value(A, elm_n);
}

TEST(group_mat_mul, Test_GenericBinary) {
  MatMulBinary gen(chn_n, elm_n);
  mat_mul_generic_params_t prm = gen.getParams();
  std::fill_n(inp_i8, elm_n, -1);
  std::fill_n(wgt_i8, gen_wgt_n, -1);

  mat_mul_generic_binary(&prm, &A, inp_i8, ocg, wgt_i8);
  assert_accumulator_value(A, elm_n * 4);
}

TEST(group_mat_mul, Test_DirectInt8) {
  MatMulDirectFn dir(img_i8, win, elm_n);
  mat_mul_direct_params_t prm = dir.getParams();
  std::fill_n(inp_i8, elm_n, 1);
  std::fill_n(wgt_i8, dir_wgt_n, 1);

  mat_mul_direct_int8(&prm, &A, inp_i8, ocg, wgt_i8);
  assert_accumulator_value(A, elm_n);
}

TEST(group_mat_mul, Test_DirectBinary) {
  MatMulBinaryDirectFn dir(img_i8, win, elm_n);
  mat_mul_direct_params_t prm = dir.getParams();
  std::fill_n(inp_i8, elm_n, -1);
  std::fill_n(wgt_i8, dir_wgt_n, -1);

  mat_mul_direct_binary(&prm, &A, inp_i8, ocg, wgt_i8);
  assert_accumulator_value(A, elm_n * 4);
}

TEST(group_mat_mul, Test_DirectInt16) {
#if defined(__VX4B__)
  TEST_IGNORE_MESSAGE("mat_mul_direct_int16 assembly faults on VX4");
#else
  MatMulDirectFn dir(img_i16, win, elm_n);
  mat_mul_direct_params_t prm = dir.getParams();
  std::fill_n(inp_i16, elm_n, 1);
  std::fill_n(wgt_i16, i16_wgt_n, 0);
  std::fill_n(wgt_i16, dir_wgt_n, 1);

  mat_mul_direct_int16(&prm, &A, inp_i16, ocg, wgt_i16);
  assert_accumulator_value(A, elm_n);
#endif
}

TEST(group_mat_mul, Test_DirectInt16x8) {
#if defined(__XS3A__)
  TEST_IGNORE_MESSAGE("mat_mul_direct_int16x8 assembly is not implemented on XS3");
#endif
  MatMulDirectFn dir(img_i16, win, elm_n);
  mat_mul_direct_params_t prm = dir.getParams();
  std::fill_n(inp_i16, elm_n, 1);
  std::fill_n(wgt_i8, dir_wgt_n, 1);

  mat_mul_direct_int16x8(&prm, &A, inp_i16, ocg, wgt_i8);
  assert_accumulator_value(A, elm_n);
}

TEST(group_mat_mul, Test_DepthwiseInt8) {
  MatMulDirectFn_DW dir(img_i8, win);
  mat_mul_dw_direct_params_t prm = dir.getParams();
  std::fill_n(inp_i8, chn_n, 1);
  std::fill_n(wgt_i8, dw_wgt_n, 1);

  mat_mul_dw_direct(&prm, &A, inp_i8, ocg, wgt_i8);
  assert_accumulator_value(A, 1);
}

TEST(group_mat_mul, Test_DepthwiseInt16) {
  MatMulDirectFn_DW dir(img_i16, win);
  mat_mul_dw_direct_params_t prm = dir.getParams();
  std::fill_n(inp_i16, chn_n, 1);
  std::fill_n(wgt_i16, dw_wgt_n, 1);

  mat_mul_dw_direct_int16(&prm, &A, inp_i16, ocg, wgt_i16);
  assert_accumulator_value(A, 1);
}

}  // extern "C"
