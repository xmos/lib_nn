// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "Rand.hpp"
#include "geom/ImageGeometry.hpp"
#include "geom/util.hpp"

extern "C" {
#include "unity.h"
#include "unity_fixture.h"
}

using namespace nn;
using namespace nn::test;

extern "C" {

TEST_GROUP(group_ImageRegion);
TEST_SETUP(group_ImageRegion) {}
TEST_TEAR_DOWN(group_ImageRegion) {}
TEST_GROUP_RUNNER(group_ImageRegion) {
  RUN_TEST_CASE(group_ImageRegion, Constructor);
  RUN_TEST_CASE(group_ImageRegion, StartVect);
  RUN_TEST_CASE(group_ImageRegion, EndVect);
  RUN_TEST_CASE(group_ImageRegion, Within);
  RUN_TEST_CASE(group_ImageRegion, Counts);
  RUN_TEST_CASE(group_ImageRegion, ChannelOutputGroups);
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageRegion, Constructor) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(1278);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const ImageGeometry image = {rng.rand<int>(10, 100), rng.rand<int>(10, 100),
                                 rng.rand<int>(10, 100)};

    const auto row = rng.rand<int>(0, image.height - 1);
    const auto col = rng.rand<int>(0, image.width - 1);
    const auto chan = rng.rand<int>(0, image.depth - 1);

    const auto height = rng.rand<int>(1, image.height - row);
    const auto width = rng.rand<int>(1, image.width - col);
    const auto depth = rng.rand<int>(1, image.depth - chan);

    ImageRegion regionA(std::array<int, 3>{{row, col, chan}},
                       std::array<int, 3>{{height, width, depth}});
    ImageRegion regionB(std::array<int, 3>{{row, col, chan}},
                       std::array<int, 3>{{height, width, depth}});

    TEST_ASSERT_EQUAL(row, regionA.start.row);
    TEST_ASSERT_EQUAL(col, regionA.start.col);
    TEST_ASSERT_EQUAL(chan, regionA.start.channel);

    TEST_ASSERT_EQUAL(height, regionA.shape.height);
    TEST_ASSERT_EQUAL(width, regionA.shape.width);
    TEST_ASSERT_EQUAL(depth, regionA.shape.depth);

    TEST_ASSERT_EQUAL(row, regionB.start.row);
    TEST_ASSERT_EQUAL(col, regionB.start.col);
    TEST_ASSERT_EQUAL(chan, regionB.start.channel);

    TEST_ASSERT_EQUAL(height, regionB.shape.height);
    TEST_ASSERT_EQUAL(width, regionB.shape.width);
    TEST_ASSERT_EQUAL(depth, regionB.shape.depth);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageRegion, StartVect) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(8383);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const ImageGeometry image = {rng.rand<int>(10, 100), rng.rand<int>(10, 100),
                                 rng.rand<int>(10, 100)};

    const ImageVect start(rng.rand<int>(0, image.height - 1),
                          rng.rand<int>(0, image.width - 1),
                          rng.rand<int>(0, image.depth - 1));

    const ImageVect shape(rng.rand<int>(1, image.height - start.row),
                          rng.rand<int>(1, image.width - start.col),
                          rng.rand<int>(1, image.depth - start.channel));

    ImageRegion region(std::array<int, 3>{{start.row, start.col, start.channel}},
                       std::array<int, 3>{{shape.row, shape.col, shape.channel}});

    auto startVect = region.StartVect();

    TEST_ASSERT_TRUE(start == startVect);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageRegion, EndVect) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(123555);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const ImageGeometry image = {rng.rand<int>(10, 100), rng.rand<int>(10, 100),
                                 rng.rand<int>(10, 100)};

    const ImageVect start(rng.rand<int>(0, image.height - 1),
                          rng.rand<int>(0, image.width - 1),
                          rng.rand<int>(0, image.depth - 1));

    const ImageVect end(rng.rand<int>(start.row, image.height - 1),
                        rng.rand<int>(start.col, image.width - 1),
                        rng.rand<int>(start.channel, image.depth - 1));
    const ImageVect end_inclusive = end;
    const ImageVect end_exclusive = end.add(1, 1, 1);

    ImageRegion region(std::array<int, 3>{{start.row, start.col, start.channel}},
                       std::array<int, 3>{{end.row - start.row + 1,
                                          end.col - start.col + 1,
                                          end.channel - start.channel + 1}});

    auto endVect_inclusive_true = region.EndVect(true);
    auto endVect_inclusive_false = region.EndVect(false);

    TEST_ASSERT_TRUE(endVect_inclusive_true == end_inclusive);
    TEST_ASSERT_TRUE(endVect_inclusive_false == end_exclusive);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageRegion, Within) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(7684);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const ImageGeometry image = {rng.rand<int>(10, 100), rng.rand<int>(10, 100),
                                 rng.rand<int>(10, 100)};

    const ImageVect start(rng.rand<int>(0, image.height - 1),
                          rng.rand<int>(0, image.width - 1),
                          rng.rand<int>(0, image.depth - 1));

    const ImageVect shape(rng.rand<int>(1, image.height - start.row),
                          rng.rand<int>(1, image.width - start.col),
                          rng.rand<int>(1, image.depth - start.channel));

    ImageRegion region(std::array<int, 3>{{start.row, start.col, start.channel}},
                       std::array<int, 3>{{shape.row, shape.col, shape.channel}});

    int row = rng.rand<int>(0, image.height - 1);
    int col = rng.rand<int>(0, image.width - 1);
    int chan = rng.rand<int>(0, image.depth - 1);

    const auto end = region.EndVect(false);

    bool expected = true;
    expected = expected && (row >= region.start.row);
    expected = expected && (col >= region.start.col);
    expected = expected && (chan >= region.start.channel);
    expected = expected && (row < end.row);
    expected = expected && (col < end.col);
    expected = expected && (chan < end.channel);

    TEST_ASSERT_EQUAL(expected, region.Within(row, col, chan));
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageRegion, Counts) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(7684);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const ImageGeometry image = {rng.rand<int>(10, 100), rng.rand<int>(10, 100),
                                 rng.rand<int>(10, 100)};

    const ImageVect start(rng.rand<int>(0, image.height - 1),
                          rng.rand<int>(0, image.width - 1),
                          rng.rand<int>(0, image.depth - 1));

    const ImageVect shape(rng.rand<int>(1, image.height - start.row),
                          rng.rand<int>(1, image.width - start.col),
                          rng.rand<int>(1, image.depth - start.channel));

    ImageRegion region(std::array<int, 3>{{start.row, start.col, start.channel}},
                       std::array<int, 3>{{shape.row, shape.col, shape.channel}});

    const auto pixel_count = shape.row * shape.col;
    const auto element_count = pixel_count * shape.channel;

    TEST_ASSERT_EQUAL(pixel_count, region.PixelCount());
    TEST_ASSERT_EQUAL(element_count, region.ElementCount());
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageRegion, ChannelOutputGroups) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(7684);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const ImageGeometry image = {rng.rand<int>(10, 100), rng.rand<int>(10, 100),
                                 rng.rand<int>(10, 1000)};

    const ImageVect start(rng.rand<int>(0, image.height - 1),
                          rng.rand<int>(0, image.width - 1),
                          rng.rand<int>(0, image.depth - 1));

    const ImageVect shape(rng.rand<int>(1, image.height - start.row),
                          rng.rand<int>(1, image.width - start.col),
                          rng.rand<int>(1, image.depth - start.channel));

    ImageRegion region(std::array<int, 3>{{start.row, start.col, start.channel}},
                       std::array<int, 3>{{shape.row, shape.col, shape.channel}});

    const auto chans_per_group = rng.rand<int>(1, 32);

    const auto expected_count =
        (region.shape.depth + chans_per_group - 1) / chans_per_group;

    TEST_ASSERT_EQUAL(expected_count, region.ChannelOutputGroups(chans_per_group));
  }
}

}  // extern "C"
