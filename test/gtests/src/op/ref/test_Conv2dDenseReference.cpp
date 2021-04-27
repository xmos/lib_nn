

#include "nn_types.h"
#include "geom/util.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "RefOps.hpp"
#include "Rand.hpp"
#include "ref_tests.hpp"
#include "gtest/gtest.h"

#include "Conv2d.hpp"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>

using namespace nn;
using namespace nn::test;

static auto rng = Rand(69);

/**
 * 
 * 
 * 
 */
class Conv2dDenseReferenceTestA : public ::testing::TestWithParam<Filter2dGeometry>
{
};

TEST_P(Conv2dDenseReferenceTestA, NoPadding)
{
  auto geom = GetParam();

  auto weights = std::vector<int8_t>(geom.window.shape.imageElements() * geom.output.depth);
  auto bias = std::vector<int32_t>(geom.output.depth);
  auto eff_mult = std::vector<float>(geom.output.depth);
  auto input = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  int8_t input_zero = -5;
  int8_t output_zero = 5;

  memset(&weights[0], 1, sizeof(int8_t) * weights.size());
  memset(&input[0], 1, sizeof(int8_t) * input.size());

  int32_t acc32 = geom.window.shape.imageElements() * (int32_t(weights[0]) * (int32_t(input[0]) - input_zero));

  int32_t target_acc32 = 32;
  int8_t target_acc8 = 16;

  for (int k = 0; k < geom.output.depth; k++)
  {
    bias[k] = target_acc32 - acc32;
    eff_mult[k] = float(target_acc8) / float(target_acc32);
  }

  memset(&expected[0], int8_t(target_acc8 + output_zero), sizeof(int8_t) * expected.size());

  auto output = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                         input_zero, output_zero);

  ASSERT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(Basic, Conv2dDenseReferenceTestA, ::testing::Values(Filter2dGeometry(ImageGeometry(1, 1, 2), ImageGeometry(1, 1, 2), WindowGeometry(1, 1, 2)), Filter2dGeometry(ImageGeometry(2, 2, 2), ImageGeometry(2, 2, 2), WindowGeometry(1, 1, 2)), Filter2dGeometry(ImageGeometry(2, 2, 2), ImageGeometry(1, 1, 2), WindowGeometry(2, 2, 2))));

static auto iterA = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(false, false));
INSTANTIATE_TEST_SUITE_P(Random, Conv2dDenseReferenceTestA, ::testing::ValuesIn(iterA.begin(), iterA.end()));

/**
 * 
 * 
 * 
 */

class Conv2dDenseReferenceTestB : public ::testing::TestWithParam<Filter2dGeometry>
{
};

TEST_P(Conv2dDenseReferenceTestB, WithPadding)
{
  auto geom = GetParam();

  auto weights = std::vector<int8_t>(geom.window.shape.imageElements() * geom.output.depth);
  auto bias = std::vector<int32_t>(geom.output.depth);
  auto eff_mult = std::vector<float>(geom.output.depth);
  auto input = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  int8_t input_zero = -5;
  int8_t output_zero = 5;

  memset(&weights[0], 1, sizeof(int8_t) * weights.size());
  memset(&input[0], 1, sizeof(int8_t) * input.size());

  int32_t target_acc32 = 32;
  int8_t target_acc8 = 16;

  auto pix_acc = geom.input.depth * int32_t(weights[0]) * (int32_t(input[0]) - input_zero);

  for (int k = 0; k < geom.output.depth; k++)
  {
    bias[k] = target_acc32 - geom.window.shape.imagePixels() * pix_acc;
    eff_mult[k] = float(target_acc8) / float(target_acc32);
  }

  auto init_pad = geom.ModelPadding();
  for (int row = 0; row < geom.output.height; row++)
  {
    for (int col = 0; col < geom.output.width; col++)
    {

      // The behavior here depends on how many pixels are outside the input image.
      auto pad = init_pad;
      pad.top -= row * geom.window.stride.row;
      pad.left -= col * geom.window.stride.col;
      pad.bottom += row * geom.window.stride.row;
      pad.right += col * geom.window.stride.col;
      pad.MakeUnsigned();

      auto patch_rows = int(geom.window.shape.height) - pad.top - pad.bottom;
      auto patch_cols = int(geom.window.shape.width) - pad.left - pad.right;

      auto patch_pix = patch_rows * patch_cols;

      int32_t acc32 = patch_pix * pix_acc;

      for (int cout = 0; cout < geom.output.depth; cout++)
      {
        const auto output_index = geom.output.Index(row, col, cout);
        auto r = (((acc32 + bias[cout]) * eff_mult[cout]) + output_zero);
        expected[output_index] = round_int8(r);
      }
    }
  }

  auto output = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                         input_zero, output_zero);

  ASSERT_EQ(output, expected);
}

static auto iterB = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(false, true), 763577);
INSTANTIATE_TEST_SUITE_P(Random, Conv2dDenseReferenceTestB, ::testing::ValuesIn(iterB.begin(), iterB.end()));