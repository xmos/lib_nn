// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.


#include <algorithm>
#include <cassert>
#include <cstdint>
#include <cstring>
#include <iostream>
#include <memory>
#include <vector>

#include "Rand.hpp"
#include "RandExtensions.hpp"
#include "RefOps.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "geom/util.hpp"
#include "nn_types.h"

extern "C" {
#include "unity.h"
#include "unity_fixture.h"
}

using namespace nn;
using namespace nn::test;

extern "C" {

TEST_GROUP(group_Conv2dDenseReference);
TEST_SETUP(group_Conv2dDenseReference) {}
TEST_TEAR_DOWN(group_Conv2dDenseReference) {}
TEST_GROUP_RUNNER(group_Conv2dDenseReference) {
  RUN_TEST_CASE(group_Conv2dDenseReference, NoPadding);
  RUN_TEST_CASE(group_Conv2dDenseReference, WithPadding);
}

static void CheckNoPadding(Filter2dGeometry geom) {
  auto weights =
      std::vector<int8_t>(geom.window.shape.ElementCount() * geom.output.depth);
  auto bias = std::vector<int32_t>(geom.output.depth);
  auto eff_mult = std::vector<float>(geom.output.depth);
  auto input = std::vector<int8_t>(geom.input.ElementCount());
  auto expected = std::vector<int8_t>(geom.output.ElementCount());

  int8_t input_zero = -5;
  int8_t output_zero = 5;

  memset(&weights[0], 1, sizeof(int8_t) * weights.size());
  memset(&input[0], 1, sizeof(int8_t) * input.size());

  int32_t acc32 = geom.window.shape.ElementCount() *
                  (int32_t(weights[0]) * (int32_t(input[0]) - input_zero));

  int32_t tarGetAccu32 = 32;
  int8_t tarGetAccu8 = 16;

  for (int k = 0; k < geom.output.depth; k++) {
    bias[k] = tarGetAccu32 - acc32;
    eff_mult[k] = float(tarGetAccu8) / float(tarGetAccu32);
  }

  memset(&expected[0], int8_t(tarGetAccu8 + output_zero),
         sizeof(int8_t) * expected.size());

  auto output = nn::test::ops::ref::Conv2dDenseReference(
      geom, &input[0], &weights[0], &bias[0], &eff_mult[0], input_zero,
      output_zero);

  TEST_ASSERT_EQUAL(expected.size(), output.size());
  for (int i = 0; i < expected.size(); i++) {
    TEST_ASSERT_EQUAL(expected[i], output[i]);
  }
}

TEST(group_Conv2dDenseReference, NoPadding) {
  CheckNoPadding(Filter2dGeometry(ImageGeometry(1, 1, 2), ImageGeometry(1, 1, 2),
                                  WindowGeometry(1, 1, 2)));
  CheckNoPadding(Filter2dGeometry(ImageGeometry(2, 2, 2), ImageGeometry(2, 2, 2),
                                  WindowGeometry(1, 1, 2)));
  CheckNoPadding(Filter2dGeometry(ImageGeometry(2, 2, 2), ImageGeometry(1, 1, 2),
                                  WindowGeometry(2, 2, 2)));

  auto iterA = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(
      100, SimpleFilter(false, false));
  for (auto geom : iterA) {
    CheckNoPadding(geom);
  }
}

/**
 *
 *
 *
 */

TEST(group_Conv2dDenseReference, WithPadding) {
  auto iterB = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(
      100, SimpleFilter(false, true), 763577);
  for (auto geom : iterB) {

  auto weights =
      std::vector<int8_t>(geom.window.shape.ElementCount() * geom.output.depth);
  auto bias = std::vector<int32_t>(geom.output.depth);
  auto eff_mult = std::vector<float>(geom.output.depth);
  auto input = std::vector<int8_t>(geom.input.ElementCount());
  auto expected = std::vector<int8_t>(geom.output.ElementCount());

  int8_t input_zero = -5;
  int8_t output_zero = 5;

  memset(&weights[0], 1, sizeof(int8_t) * weights.size());
  memset(&input[0], 1, sizeof(int8_t) * input.size());

  int32_t tarGetAccu32 = 32;
  int8_t tarGetAccu8 = 16;

  auto pix_acc =
      geom.input.depth * int32_t(weights[0]) * (int32_t(input[0]) - input_zero);

  for (int k = 0; k < geom.output.depth; k++) {
    bias[k] = tarGetAccu32 - geom.window.shape.PixelCount() * pix_acc;
    eff_mult[k] = float(tarGetAccu8) / float(tarGetAccu32);
  }

  for (int row = 0; row < geom.output.height; row++) {
    for (int col = 0; col < geom.output.width; col++) {
      auto non_pad_pixels =
          geom.GetWindow(row, col, 0)
              .Fold<int, int8_t>(
                  &input[0],
                  [](const nn::ImageVect &, const nn::ImageVect &, int acc,
                     int8_t, bool is_pad) { return acc + (is_pad ? 0 : 1); },
                  0);
      auto patch_pix = non_pad_pixels / geom.window.shape.depth;

      int32_t acc32 = patch_pix * pix_acc;

      for (int cout = 0; cout < geom.output.depth; cout++) {
        const auto output_index = geom.output.Index(row, col, cout);
        auto r = (((acc32 + bias[cout]) * eff_mult[cout]) + output_zero);
        expected[output_index] = round_int8(r);
      }
    }
  }

  auto output = nn::test::ops::ref::Conv2dDenseReference(
      geom, &input[0], &weights[0], &bias[0], &eff_mult[0], input_zero,
      output_zero);

  TEST_ASSERT_EQUAL(expected.size(), output.size());
  for (int i = 0; i < expected.size(); i++) {
    TEST_ASSERT_EQUAL(expected[i], output[i]);
  }
  }
}

}  // extern "C"
