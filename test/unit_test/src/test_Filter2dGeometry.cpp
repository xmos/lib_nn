// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "FilterGeometryIterHelper.hpp"
#include "Rand.hpp"
#include "geom/Filter2dGeometry.hpp"

extern "C" {
#include "unity.h"
#include "unity_fixture.h"
}

using namespace nn;

// VX4's reduced libc++ runtime lacks typeinfo for shared_ptr's internal
// control block used by FilterGeometryIterator's polymorphic frame stack.
#if !defined(__VX4A__) && !defined(__VX4B__)

extern "C" {

TEST_GROUP(group_Filter2dGeometry);
TEST_SETUP(group_Filter2dGeometry) {}
TEST_TEAR_DOWN(group_Filter2dGeometry) {}
TEST_GROUP_RUNNER(group_Filter2dGeometry) {
  RUN_TEST_CASE(group_Filter2dGeometry, IsDepthwise);
  RUN_TEST_CASE(group_Filter2dGeometry, GetWindow);
  RUN_TEST_CASE(group_Filter2dGeometry, Padding);
}

static nn::ff::FilterGeometryIterator filter_sets[] = {

    // Dense
    test::unpadded::AllUnpadded(
        nn::Filter2dGeometry(
            nn::ImageGeometry{0, 0, 12}, nn::ImageGeometry{3, 3, 12},
            nn::WindowGeometry(std::array<int, 3>{{4, 4, 0}},
                               std::array<int, 2>{{2, 2}},
                               std::array<int, 3>{{2, 3, 0}},
                               std::array<int, 2>{{2, 3}})),
        false, 1),

    test::padded::AllPadded(
        nn::Filter2dGeometry(
            nn::ImageGeometry{0, 0, 12}, nn::ImageGeometry{3, 3, 12},
            nn::WindowGeometry(std::array<int, 3>{{4, 4, 0}},
                               std::array<int, 2>{{2, 2}},
                               std::array<int, 3>{{2, 3, 0}},
                               std::array<int, 2>{{2, 3}})),
        {2, 2, 2, 2}, false, 1),

    // Depthwise
    test::unpadded::AllUnpadded(
        nn::Filter2dGeometry(
            nn::ImageGeometry{0, 0, 0}, nn::ImageGeometry{3, 3, 12},
            nn::WindowGeometry(std::array<int, 3>{{4, 4, 1}},
                               std::array<int, 2>{{2, 2}},
                               std::array<int, 3>{{2, 3, 0}},
                               std::array<int, 2>{{2, 3}})),
        true, 1),

    test::padded::AllPadded(
        nn::Filter2dGeometry(
            nn::ImageGeometry{0, 0, 0}, nn::ImageGeometry{3, 3, 12},
            nn::WindowGeometry(std::array<int, 3>{{4, 4, 1}},
                               std::array<int, 2>{{2, 2}},
                               std::array<int, 3>{{2, 3, 0}},
                               std::array<int, 2>{{2, 3}})),
        {2, 2, 2, 2}, true, 1),
};

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_Filter2dGeometry, IsDepthwise) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      TEST_ASSERT_EQUAL(filter.IsDepthwise(), filter.window.stride.channel == 1);
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_Filter2dGeometry, GetWindow) {
  for (auto filter_set : filter_sets) {
    filter_set.Reset();
    for (auto filter : filter_set) {
      for (int row = 0; row < filter.output.height; row += 2) {
        for (int col = 0; col < filter.output.width; col += 2) {
          for (int chan = 0; chan < filter.output.depth; chan += 5) {
            auto v = nn::ImageVect(row, col, chan);
            auto loc1 = filter.GetWindow(row, col, chan);
            auto loc2 = filter.GetWindow(v);

            TEST_ASSERT_TRUE(v == loc1.output_coords);
            TEST_ASSERT_TRUE(v == loc2.output_coords);

            TEST_ASSERT_TRUE(filter == loc1.filter);
            TEST_ASSERT_TRUE(filter == loc2.filter);
          }
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_Filter2dGeometry, Padding) {
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

      TEST_ASSERT_TRUE(exp_padding == padding);
    }
  }
}

}  // extern "C"

#endif  // !__VX4A__ && !__VX4B__
