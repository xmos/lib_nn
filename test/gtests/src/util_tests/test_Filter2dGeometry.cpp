#include <iostream>
#include <tuple>
#include <vector>

#include "../op/ref/ref_tests.hpp"
#include "FilterGeometryIterHelper.hpp"
#include "Rand.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "gtest/gtest.h"

using namespace nn;

static const bool PRINT_CASE_COUNTS = true;

static nn::ff::FilterGeometryIterator filter_sets[] = {

    // Dense
    test::unpadded::AllUnpadded(
        nn::Filter2dGeometry({0, 0, 12}, {3, 3, 12},
                             {{4, 4, 0}, {2, 2}, {2, 3}, {2, 3}}),
        false, 1),

    test::padded::AllPadded(
        nn::Filter2dGeometry({0, 0, 12}, {3, 3, 12},
                             {{4, 4, 0}, {2, 2}, {2, 3}, {2, 3}}),
        {2, 2, 2, 2}, false, 1),

    // Depthwise
    test::unpadded::AllUnpadded(
        nn::Filter2dGeometry({0, 0, 0}, {3, 3, 12},
                             {{4, 4, 1}, {2, 2}, {2, 3}, {2, 3}}),
        true, 1),

    test::padded::AllPadded(
        nn::Filter2dGeometry({0, 0, 0}, {3, 3, 12},
                             {{4, 4, 1}, {2, 2}, {2, 3}, {2, 3}}),
        {2, 2, 2, 2}, true, 1),
};

/////////////////////////////////////////////////////////////////////////
//
//
TEST(Filter2dGeometry_Test, IsDepthwise) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      ASSERT_EQ(filter.IsDepthwise(), filter.window.stride.channel == 1);
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(Filter2dGeometry_Test, GetWindow) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      for (int row = 0; row < filter.output.height; row += 2) {
        for (int col = 0; col < filter.output.width; col += 2) {
          for (int chan = 0; chan < filter.output.depth; chan += 5) {
            auto v = nn::ImageVect(row, col, chan);
            auto loc1 = filter.GetWindow(row, col, chan);
            auto loc2 = filter.GetWindow(v);

            ASSERT_EQ(v, loc1.output_coords);
            ASSERT_EQ(v, loc2.output_coords);

            ASSERT_EQ(filter, loc1.filter);
            ASSERT_EQ(filter, loc2.filter);
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(Filter2dGeometry_Test, Padding) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      padding_t padding = filter.Padding();

      padding_t exp_padding;

      exp_padding.top = -filter.window.start.row;
      exp_padding.left = -filter.window.start.col;

      auto loc = filter.GetWindow(filter.output.height - 1,
                                  filter.output.width - 1, 0);
      auto last_x = loc.InputCoords(filter.window.shape.height - 1,
                                    filter.window.shape.width - 1, 0);

      exp_padding.bottom = last_x.row - (filter.input.height - 1);
      exp_padding.right = last_x.col - (filter.input.width - 1);

      exp_padding.MakeUnsigned();

      ASSERT_EQ(exp_padding, padding) << "Filter: " << filter;
    }
  }
}
