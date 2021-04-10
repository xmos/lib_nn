

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "RefOps.hpp"
#include "Rand.hpp"
#include "ref_tests.hpp"

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

class MaxPoolReferenceTestA : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(MaxPoolReferenceTestA, NoPadding)
{
  auto geom = GetParam();

  auto input = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  auto in_cov = geom.input.getAddressCovector<int8_t>();
  auto out_cov = geom.output.getAddressCovector<int8_t>();

  for(int xan = 0; xan < geom.output.depth; xan++){
    for(int row = 0; row < geom.output.height; row++){
      for(int col = 0; col < geom.output.width; col++){

        int8_t* Y = out_cov.resolve(&expected[0], row, col, xan);
        
        *Y = std::numeric_limits<int8_t>::min();

        for(int kr = 0; kr < geom.window.shape.height; kr++){
          for(int kc = 0; kc < geom.window.shape.width; kc++){

            auto xr = row * int(geom.window.shape.height) + kr;
            auto xc = col * int(geom.window.shape.width ) + kc;

            int8_t* X = in_cov.resolve(&input[0], xr, xc, xan);

            *X = rng.rand<int8_t>();
            *Y = std::max<int8_t>(*Y, *X);
          }
        }
      }
    }
  }

  auto output = nn::test::ops::ref::MaxPoolReference(geom, &input[0]);

  ASSERT_EQ(output, expected);
}

static auto iterA = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(true, false, 32));
INSTANTIATE_TEST_SUITE_P(, MaxPoolReferenceTestA, ::testing::ValuesIn(iterA.begin(), iterA.end()));






class MaxPoolReferenceTestB : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(MaxPoolReferenceTestB, WithPadding)
{
  auto geom = GetParam();

  auto input = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  auto in_cov = geom.input.getAddressCovector<int8_t>();
  auto out_cov = geom.output.getAddressCovector<int8_t>();

  for(int xan = 0; xan < geom.output.depth; xan++){
    for(int row = 0; row < geom.output.height; row++){
      for(int col = 0; col < geom.output.width; col++){

        int8_t* Y = out_cov.resolve(&expected[0], row, col, xan);
        
        *Y = std::numeric_limits<int8_t>::min();

        for(int kr = 0; kr < geom.window.shape.height; kr++){
          for(int kc = 0; kc < geom.window.shape.width; kc++){

            auto xr = geom.window.start.row + row * int(geom.window.shape.height) + kr;
            auto xc = geom.window.start.col + col * int(geom.window.shape.width ) + kc;

            if(xr < 0 || xr >= geom.input.height) continue;
            if(xc < 0 || xc >= geom.input.width ) continue;

            int8_t* X = in_cov.resolve(&input[0], xr, xc, xan);

            *X = rng.rand<int8_t>();
            *Y = std::max<int8_t>(*Y, *X);
          }
        }
      }
    }
  }

  auto output = nn::test::ops::ref::MaxPoolReference(geom, &input[0]);

  ASSERT_EQ(output, expected);

}

static auto iterB = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(true, true, 32));
INSTANTIATE_TEST_SUITE_P(, MaxPoolReferenceTestB, ::testing::ValuesIn(iterB.begin(), iterB.end()));