// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>

extern "C" {
#include "unity.h"
#include "unity_fixture.h"
}

// Copied unchanged from OutputTransformFn.cpp to test the current conversion.
template <class T>
static int64_t float_to_int(T f, int e) {
  return (int64_t)std::rint(ldexp(f, e));
}

template <class T>
static int16_t float_to_int16(T f, int e) {
  const auto v = std::rint(std::ldexp(f, e));
  if (v >= INT16_MAX) return INT16_MAX;
  if (v <= INT16_MIN) return INT16_MIN;
  return static_cast<int16_t>(v);
}

extern "C" {

TEST_GROUP(group_float_to_int16);
TEST_SETUP(group_float_to_int16) {}
TEST_TEAR_DOWN(group_float_to_int16) {}
TEST_GROUP_RUNNER(group_float_to_int16) {
  RUN_TEST_CASE(group_float_to_int16, scaling);
  RUN_TEST_CASE(group_float_to_int16, rounding);
  RUN_TEST_CASE(group_float_to_int16, saturation);
  RUN_TEST_CASE(group_float_to_int16, positive_overflow);
  RUN_TEST_CASE(group_float_to_int16, negative_overflow);
  RUN_TEST_CASE(group_float_to_int16, positive_infinity);
  RUN_TEST_CASE(group_float_to_int16, negative_infinity);
}

TEST(group_float_to_int16, scaling) {
  TEST_ASSERT_EQUAL_INT16(0, float_to_int16(0.0f, 0));
  TEST_ASSERT_EQUAL_INT16(16384, float_to_int16(0.5f, 15));
  TEST_ASSERT_EQUAL_INT16(-16384, float_to_int16(-0.5, 15));
  TEST_ASSERT_EQUAL_INT16(3, float_to_int16(12.0, -2));
}

TEST(group_float_to_int16, rounding) {
  TEST_ASSERT_EQUAL_INT16(1, float_to_int16(1.25, 0));
  TEST_ASSERT_EQUAL_INT16(-1, float_to_int16(-1.25, 0));
  TEST_ASSERT_EQUAL_INT16(2, float_to_int16(2.5, 0));
  TEST_ASSERT_EQUAL_INT16(4, float_to_int16(3.5, 0));
  TEST_ASSERT_EQUAL_INT16(-2, float_to_int16(-2.5, 0));
  TEST_ASSERT_EQUAL_INT16(-4, float_to_int16(-3.5, 0));
}

TEST(group_float_to_int16, saturation) {
  TEST_ASSERT_EQUAL_INT16(INT16_MAX, float_to_int16(32767.0, 0));
  TEST_ASSERT_EQUAL_INT16(INT16_MIN, float_to_int16(-32768.0, 0));
  TEST_ASSERT_EQUAL_INT16(INT16_MAX, float_to_int16(32768.0, 0));
  TEST_ASSERT_EQUAL_INT16(INT16_MIN, float_to_int16(-32769.0, 0));
}

TEST(group_float_to_int16, positive_overflow) {
  TEST_ASSERT_EQUAL_INT16(INT16_MAX, float_to_int16(1.0, 63));
  TEST_ASSERT_EQUAL_INT16(
      INT16_MAX, float_to_int16(std::numeric_limits<double>::max(), 0));
}

TEST(group_float_to_int16, negative_overflow) {
  TEST_ASSERT_EQUAL_INT16(INT16_MIN, float_to_int16(-1.0, 64));
  TEST_ASSERT_EQUAL_INT16(
      INT16_MIN, float_to_int16(-std::numeric_limits<double>::max(), 0));
}

TEST(group_float_to_int16, positive_infinity) {
  TEST_ASSERT_EQUAL_INT16(
      INT16_MAX, float_to_int16(std::numeric_limits<double>::infinity(), 0));
}

TEST(group_float_to_int16, negative_infinity) {
  TEST_ASSERT_EQUAL_INT16(
      INT16_MIN, float_to_int16(-std::numeric_limits<double>::infinity(), 0));
}

}  // extern "C"
