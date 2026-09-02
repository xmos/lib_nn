// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.


#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "Rand.hpp"
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

TEST_GROUP(group_Conv2dDenseBinaryReference);
TEST_SETUP(group_Conv2dDenseBinaryReference) {}
TEST_TEAR_DOWN(group_Conv2dDenseBinaryReference) {}
TEST_GROUP_RUNNER(group_Conv2dDenseBinaryReference) {
  RUN_TEST_CASE(group_Conv2dDenseBinaryReference, BinaryOutNoPadding);
  RUN_TEST_CASE(group_Conv2dDenseBinaryReference, IntOutNoPadding);
}

const int bnn_elements_per_word = 32;

static const Filter2dGeometry basic_geometries[] = {
    Filter2dGeometry(ImageGeometry(1, 1, 32), ImageGeometry(1, 1, 32),
                      WindowGeometry(1, 1, 32)),
    Filter2dGeometry(ImageGeometry(2, 2, 32), ImageGeometry(2, 2, 32),
                      WindowGeometry(1, 1, 32)),
    Filter2dGeometry(ImageGeometry(2, 2, 32), ImageGeometry(1, 1, 32),
                      WindowGeometry(2, 2, 32)),
    Filter2dGeometry(ImageGeometry(2, 2, 256), ImageGeometry(1, 1, 32),
                      WindowGeometry(2, 2, 256)),
    Filter2dGeometry(ImageGeometry(2, 2, 32), ImageGeometry(1, 1, 256),
                      WindowGeometry(2, 2, 32)),
    Filter2dGeometry(ImageGeometry(2, 2, 64), ImageGeometry(1, 1, 64),
                      WindowGeometry(2, 2, 64)),
    Filter2dGeometry(ImageGeometry(8, 8, 32), ImageGeometry(3, 3, 32),
                      WindowGeometry(6, 6, 32)),
    Filter2dGeometry(ImageGeometry(8, 8, 32), ImageGeometry(8, 8, 32),
                      WindowGeometry(1, 1, 32)),
};

static void CheckBinaryOutNoPadding(Filter2dGeometry geom) {
  int packed_weight_word_count = geom.window.shape.ElementCount() /
                                 bnn_elements_per_word * geom.output.depth;

  auto packed_filter = std::vector<int32_t>(packed_weight_word_count, 0);

  int packed_input_word_count =
      geom.input.ElementCount() / bnn_elements_per_word;
  auto packed_input = std::vector<int32_t>(packed_input_word_count, 0);

  int packed_output_word_count =
      geom.output.ElementCount() / bnn_elements_per_word;
  auto expected_packed_output =
      std::vector<int32_t>(packed_output_word_count, 0);

  auto thresholds = std::vector<int32_t>(geom.output.depth, 0);

  auto output = nn::test::ops::ref::Conv2dBNNBinaryOutReference(
      geom, packed_input.data(), packed_filter.data(), thresholds.data());

  TEST_ASSERT_EQUAL(expected_packed_output.size(), output.size());
  for (int i = 0; i < expected_packed_output.size(); i++) {
    TEST_ASSERT_EQUAL(expected_packed_output[i], output[i]);
  }
}

TEST(group_Conv2dDenseBinaryReference, BinaryOutNoPadding) {
  for (auto &geom : basic_geometries) {
    CheckBinaryOutNoPadding(geom);
  }
}

static void CheckIntOutNoPadding(Filter2dGeometry geom) {
  int receptive_volume = geom.window.shape.ElementCount();
  int packed_weight_word_count =
      (receptive_volume * geom.output.depth) / bnn_elements_per_word;

  auto packed_filter = std::vector<int32_t>(packed_weight_word_count, ~0);

  int packed_input_word_count =
      geom.input.ElementCount() / bnn_elements_per_word;
  auto packed_input = std::vector<int32_t>(packed_input_word_count, 0);

  int val = 24;  // not special - just a target for the scaled accumulator
  auto post_activation_multiplier =
      std::vector<float>(geom.output.depth, (float)val / receptive_volume);
  auto post_activation_bias = std::vector<float>(geom.output.depth, 0.);

  const int clamp_min = INT32_MIN;
  const int clamp_max = INT32_MAX;

  auto output = nn::test::ops::ref::Conv2dBNNIntOutReference(
      geom, packed_input.data(), packed_filter.data(),
      post_activation_multiplier.data(), post_activation_bias.data(), clamp_min,
      clamp_max);

  //[asj] The 2 is due to the random shift left in the output transform
  auto expected = std::vector<int8_t>(output.size(), val * 2);

  TEST_ASSERT_EQUAL(expected.size(), output.size());
  for (int i = 0; i < expected.size(); i++) {
    TEST_ASSERT_EQUAL(expected[i], output[i]);
  }
}

TEST(group_Conv2dDenseBinaryReference, IntOutNoPadding) {
  for (auto &geom : basic_geometries) {
    CheckIntOutNoPadding(geom);
  }
}

}  // extern "C"
