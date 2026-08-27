// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "Rand.hpp"
#include "geom/WindowGeometry.hpp"
#include "unity.h"
#include "unity_fixture.h"

using namespace nn;

TEST_GROUP(group_WindowGeometry);
TEST_SETUP(group_WindowGeometry) {}
TEST_TEAR_DOWN(group_WindowGeometry) {}
TEST_GROUP_RUNNER(group_WindowGeometry) {
  RUN_TEST_CASE(group_WindowGeometry, Constructor);
  RUN_TEST_CASE(group_WindowGeometry, UsesDilation);
  RUN_TEST_CASE(group_WindowGeometry, WindowOffset);
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_WindowGeometry, Constructor) {
  for (int iter = 0; iter < 100; iter++) {
    auto rng = nn::test::Rand(iter);

    auto height = rng.rand<unsigned>(1, 20);
    auto width = rng.rand<unsigned>(1, 20);
    auto depth = rng.rand<unsigned>(1, 100);
    auto cbytes = rng.rand<unsigned>(1, 4);

    auto start_row = rng.rand<int>(1 - height, 0);
    auto start_col = rng.rand<int>(1 - width, 0);

    auto stride_row = rng.rand<int>(1, height);
    auto stride_col = rng.rand<int>(1, width);
    auto stride_chan = rng.rand<int>(0, 1);

    auto dil_row = rng.rand<int>(1, 3);
    auto dil_col = rng.rand<int>(1, 3);

    auto window =
        WindowGeometry(height, width, depth, start_row, start_col, stride_row,
                       stride_col, stride_chan, dil_row, dil_col, cbytes);

    TEST_ASSERT_EQUAL(window.shape.height, height);
    TEST_ASSERT_EQUAL(window.shape.width, width);
    TEST_ASSERT_EQUAL(window.shape.depth, depth);
    TEST_ASSERT_EQUAL(window.shape.element_bits, cbytes * CHAR_BIT);

    TEST_ASSERT_EQUAL(window.start.row, start_row);
    TEST_ASSERT_EQUAL(window.start.col, start_col);

    TEST_ASSERT_EQUAL(window.stride.row, stride_row);
    TEST_ASSERT_EQUAL(window.stride.col, stride_col);
    TEST_ASSERT_EQUAL(window.stride.channel, stride_chan);

    TEST_ASSERT_EQUAL(window.dilation.row, dil_row);
    TEST_ASSERT_EQUAL(window.dilation.col, dil_col);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_WindowGeometry, UsesDilation) {
  auto r = nn::test::Rand();

  const auto max_height = 4;
  const auto max_width = 4;
  const auto max_depth = 4;

  for (int K_h = 1; K_h < max_height; K_h++) {
    for (int K_w = 1; K_w < max_width; K_w++) {
      for (int K_d = 1; K_d < max_depth; K_d++) {
        for (int i = 0; i < 10; ++i) {
          auto win = WindowGeometry(K_h, K_w, K_d, r.rand(1 - K_h, K_h - 1),
                                    r.rand(1 - K_w, K_w - 1), r.rand(1, 3),
                                    r.rand(1, 3), r.rand(0, 1), r.rand(1, 3),
                                    r.rand(1, 3));

          auto is_using_dilation =
              (win.dilation.row != 1) || (win.dilation.col != 1);

          TEST_ASSERT_EQUAL(is_using_dilation, win.UsesDilation());
        }
      }
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_WindowGeometry, WindowOffset) {
  auto r = nn::test::Rand();

  const auto max_height = 4;
  const auto max_width = 4;
  const auto max_depth = 4;

  for (int K_h = 1; K_h < max_height; K_h++) {
    for (int K_w = 1; K_w < max_width; K_w++) {
      for (int K_d = 1; K_d < max_depth; K_d++) {
        for (int i = 0; i < 10; ++i) {
          auto win = WindowGeometry(K_h, K_w, K_d, r.rand(1 - K_h, K_h - 1),
                                    r.rand(1 - K_w, K_w - 1), r.rand(1, 3),
                                    r.rand(1, 3), r.rand(0, 1), r.rand(1, 3),
                                    r.rand(1, 3));

          ImageVect exp_v(win.start.row, win.start.col, 0);

          for (int i = 0; i < 10; ++i) {
            for (int j = 0; j < 10; ++j) {
              for (int k = 0; k < 10; ++k) {
                auto out_v = ImageVect(i, j, k);
                auto res_v = win.WindowOffset(out_v);
                TEST_ASSERT_EQUAL(exp_v.row, res_v.row);
                TEST_ASSERT_EQUAL(exp_v.col, res_v.col);
                TEST_ASSERT_EQUAL(exp_v.channel, res_v.channel);
                exp_v.channel += win.stride.channel;
              }
              exp_v.col += win.stride.col;
              exp_v.channel = 0;
            }
            exp_v.row += win.stride.row;
            exp_v.col = win.start.col;
          }
        }
      }
    }
  }
}
