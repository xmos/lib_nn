

#include "nn_types.h"
#include "geom/util.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "RefOps.hpp"
#include "Rand.hpp"
#include "ref_tests.hpp"
#include "MaxPool2d.hpp"

#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>

/*
  This is just a sanity check for MaxPoolReference(). It's not meant to thoroughly vet it, because if we're going to
  do that, we have no need for it.
*/

using namespace nn;
using namespace nn::test;

static auto rng = Rand();

class MaxPoolReferenceTest : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(MaxPoolReferenceTest, SanityCheck)
{
  auto geom = GetParam();

  auto input = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  for(int k = 0; k < input.size(); k++)
    input[k] = rng.rand<int8_t>();

  for(int xan = 0; xan < geom.output.depth; xan++){
    for(int row = 0; row < geom.output.height; row++){
      for(int col = 0; col < geom.output.width; col++){

        auto out_index = geom.output.Index(row, col, xan);
        
        auto loc = geom.GetWindow(row, col, xan);
        auto max_val = loc.Fold<int8_t,int8_t>(&input[0], 
                                               MaxPool2d::FoldFunc<int8_t>, 
                                               std::numeric_limits<int8_t>::min(), 
                                               std::numeric_limits<int8_t>::min());

        expected[out_index] = max_val;
      }
    }
  }

  auto output = nn::test::ops::ref::MaxPoolReference(geom, &input[0]);

  ASSERT_EQ(output, expected);
}

static auto iterA = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(300, SimpleFilter(true, false, 32));
static auto iterB = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(300, SimpleFilter(true, true, 32));

INSTANTIATE_TEST_SUITE_P(Unpadded, MaxPoolReferenceTest, ::testing::ValuesIn(iterA.begin(), iterA.end()));
INSTANTIATE_TEST_SUITE_P(Padded, MaxPoolReferenceTest, ::testing::ValuesIn(iterB.begin(), iterB.end()));

