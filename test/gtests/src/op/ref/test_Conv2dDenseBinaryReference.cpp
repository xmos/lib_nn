

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
#include "gtest/gtest.h"
#include "nn_types.h"
#include "ref_tests.hpp"

using namespace nn;
using namespace nn::test;

const int bnn_elements_per_word = 32;

class BNNConv2dDenseBinaryReferenceTestA
    : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(BNNConv2dDenseBinaryReferenceTestA, NoPadding) {
  auto geom = GetParam();

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
  ASSERT_EQ(output, expected_packed_output);
}

INSTANTIATE_TEST_SUITE_P(
    Basic, BNNConv2dDenseBinaryReferenceTestA,
    ::testing::Values(
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
                         WindowGeometry(1, 1, 32))));

// static auto iterA = nn::test::ParamedRandIter<Filter2dGeometry,
// SimpleFilter>(
//     100, SimpleFilter(false, false));
// INSTANTIATE_TEST_SUITE_P(Random, BNNConv2dDenseBinaryReferenceTestA,
//                          ::testing::ValuesIn(iterA.begin(), iterA.end()));

class BNNConv2dDenseIntReferenceTestA
    : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(BNNConv2dDenseIntReferenceTestA, NoPadding) {
  auto geom = GetParam();

  int receptive_volume = geom.window.shape.ElementCount();
  int packed_weight_word_count =
      (receptive_volume * geom.output.depth) / bnn_elements_per_word;

  auto packed_filter = std::vector<int32_t>(packed_weight_word_count, ~0);

  int packed_input_word_count =
      geom.input.ElementCount() / bnn_elements_per_word;
  auto packed_input = std::vector<int32_t>(packed_input_word_count, 0);

  int packed_output_word_count = geom.output.ElementCount();
  auto expected_packed_output =
      std::vector<int8_t>(packed_output_word_count, 0);

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

  ASSERT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(
    Basic, BNNConv2dDenseIntReferenceTestA,
    ::testing::Values(
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
                         WindowGeometry(1, 1, 32))));

// static auto iterD = nn::test::ParamedRandIter<Filter2dGeometry,
// SimpleFilter>(
//     100, SimpleFilter(false, false));
// INSTANTIATE_TEST_SUITE_P(Random, BNNConv2dDenseIntReferenceTestA,
//                          ::testing::ValuesIn(iterD.begin(), iterD.end()));
