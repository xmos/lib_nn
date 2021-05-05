#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "geom/Filter2dGeometry.hpp"
#include "Rand.hpp"
#include "../op/ref/ref_tests.hpp"

#include "FilterGeometryIterHelper.hpp"

using namespace nn;

// class Filter2dGeometryTest : public ::testing::TestWithParam<Filter2dGeometry> {};



static nn::ff::FilterGeometryIterator filter_sets[] = {
  test::unpadded::AllUnpadded( nn::Filter2dGeometry({0,0,36}, {3,3,36}, {{4,4,0}, {2,2}, {2,3}, {2,3}}), false, 1),
  test::padded::AllPadded( nn::Filter2dGeometry({0,0,0}, {3,3,36}, {{4,4,0}, {2,2}, {2,3}, {2,3}} ), {3, 3, 3, 3}, false, 1),
  test::unpadded::AllUnpadded( nn::Filter2dGeometry({0,0,36}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}}), true, 1),
  test::padded::AllPadded( nn::Filter2dGeometry({0,0,0}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}} ), {3, 3, 3, 3}, true, 1),
};


TEST(Filter2dGeometry_Test, ModelIsDepthwise)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_EQ(filter.ModelIsDepthwise(), filter.window.stride.channel == 1);
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


TEST(Filter2dGeometry_Test, ModelPadding)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {
      // nn::Filter2dGeometry filter = {{1,2,1,1}, {1,1,1,1}, {{1,1,1},{0,1},{1,1,1},{1,2}}}; {
      // if(total_iter % 1024 == 0)
      //   std::cout << "iters... " << total_iter << "\n";

      auto pad_init_signed = filter.ModelPadding(true, true);
      auto pad_init_unsigned = filter.ModelPadding(true, false);

      auto first_loc = filter.GetWindow({0,0,0});
      auto last_loc = filter.GetWindow({filter.output.height-1, filter.output.width-1, 0});

#define FAIL_MSG  "iter: " << total_iter << std::endl                     \
                  << "filter: " << filter << std::endl                    \
                  << "pad_init_signed: " << pad_init_signed << std::endl  \
                  << "pad_init_unsigned: " << pad_init_unsigned     

      auto first_pad = first_loc.SignedPadding();
      auto last_pad = last_loc.SignedPadding();

      ASSERT_EQ(pad_init_signed.top,  first_pad.top)      << FAIL_MSG;
      ASSERT_EQ(pad_init_signed.left, first_pad.left)     << FAIL_MSG;
      ASSERT_EQ(pad_init_signed.bottom, last_pad.bottom)  << FAIL_MSG;
      ASSERT_EQ(pad_init_signed.right,  last_pad.right)   << FAIL_MSG;

      first_pad.MakeUnsigned();
      last_pad.MakeUnsigned();

      ASSERT_EQ(pad_init_unsigned.top,  first_pad.top)      << FAIL_MSG;
      ASSERT_EQ(pad_init_unsigned.left, first_pad.left)     << FAIL_MSG;
      ASSERT_EQ(pad_init_unsigned.bottom, last_pad.bottom)  << FAIL_MSG;
      ASSERT_EQ(pad_init_unsigned.right,  last_pad.right)   << FAIL_MSG;
#undef FAIL_MSG
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}
