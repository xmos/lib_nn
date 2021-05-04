

#include "nn_types.h"
#include "geom/util.hpp"
#include "geom/Filter2dGeometry.hpp"
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

#include "FilterGeometryIterHelper.hpp"

/*
  This is just a sanity check for AveragePoolReference(). It's not meant to thoroughly vet it, because if we're going to
  do that, we have no need for it.
*/

using namespace nn::test;

static int32_t AddElements(const ImageVect&, const ImageVect&, int32_t acc32, int8_t elm, bool)
{
  return acc32 + elm;
}

static int CountPixels(const ImageVect&, const ImageVect&, int pix, int8_t, bool is_pad)
{
  return is_pad? (pix) : (pix+1);
}

static auto rng = Rand();

static nn::ff::FilterGeometryIterator filter_sets[] = {
  test::unpadded::SimpleDepthwise({1,8}, {1,4}, {4,66}),
};

TEST(AveragePoolReference_Test, SanityCheck)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets){
    filter_set.Reset();
    for(auto geom : filter_set){

    auto input = std::vector<int8_t>(geom.input.imageElements());
    auto expected = std::vector<int8_t>(geom.output.imageElements());

    auto win_pixy = geom.window.shape.imagePixels();

    for(int k = 0; k < input.size(); k++)
      input[k] = rng.rand<int8_t>();

    for(int xan = 0; xan < geom.output.depth; xan++){
      for(int row = 0; row < geom.output.height; row++){
        for(int col = 0; col < geom.output.width; col++){

          auto loc = geom.GetWindow(row, col, xan);

          const auto sum = loc.Fold<int32_t,int8_t>(&input[0], AddElements, 0, 0);
          const auto pix = loc.Fold<int,int8_t>(&input[0], CountPixels, 0, 0);

          const auto out_index = geom.output.Index(row, col, xan);

          expected[out_index] = round_int8(sum / float(pix));
        }
      }
    }

    auto output = nn::test::ops::ref::AveragePoolReference(geom, &input[0]);

    ASSERT_EQ(output, expected);
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}