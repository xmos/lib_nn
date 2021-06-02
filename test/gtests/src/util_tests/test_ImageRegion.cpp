
#include <iostream>
#include <tuple>
#include <vector>

#include "Rand.hpp"
#include "geom/ImageGeometry.hpp"
#include "geom/util.hpp"
#include "gtest/gtest.h"

using namespace nn;
using namespace nn::test;

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageRegion_Test, Constructor) {
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

    ImageRegion regionA(row, col, chan, height, width, depth);
    ImageRegion regionB({row, col, chan}, {height, width, depth});

    ASSERT_EQ(row, regionA.start.row);
    ASSERT_EQ(col, regionA.start.col);
    ASSERT_EQ(chan, regionA.start.channel);

    ASSERT_EQ(height, regionA.shape.height);
    ASSERT_EQ(width, regionA.shape.width);
    ASSERT_EQ(depth, regionA.shape.depth);

    ASSERT_EQ(row, regionB.start.row);
    ASSERT_EQ(col, regionB.start.col);
    ASSERT_EQ(chan, regionB.start.channel);

    ASSERT_EQ(height, regionB.shape.height);
    ASSERT_EQ(width, regionB.shape.width);
    ASSERT_EQ(depth, regionB.shape.depth);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageRegion_Test, StartVect) {
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

    ImageRegion region({start.row, start.col, start.channel},
                       {shape.row, shape.col, shape.channel});

    auto startVect = region.StartVect();

    ASSERT_EQ(start, startVect);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageRegion_Test, EndVect) {
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

    ImageRegion region({start.row, start.col, start.channel},
                       {end.row - start.row + 1, end.col - start.col + 1,
                        end.channel - start.channel + 1});

    auto endVect_inclusive_true = region.EndVect(true);
    auto endVect_inclusive_false = region.EndVect(false);

    ASSERT_EQ(endVect_inclusive_true, end_inclusive);
    ASSERT_EQ(endVect_inclusive_false, end_exclusive);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageRegion_Test, Within) {
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

    ImageRegion region({start.row, start.col, start.channel},
                       {shape.row, shape.col, shape.channel});

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

    ASSERT_EQ(expected, region.Within(row, col, chan))
        << "Region: " << region << "\n"
        << "Test coords: " << ImageVect(row, col, chan);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageRegion_Test, Counts) {
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

    ImageRegion region({start.row, start.col, start.channel},
                       {shape.row, shape.col, shape.channel});

    const auto pixel_count = shape.row * shape.col;
    const auto element_count = pixel_count * shape.channel;

    ASSERT_EQ(pixel_count, region.PixelCount());
    ASSERT_EQ(element_count, region.ElementCount());
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageRegion_Test, ChannelOutputGroups) {
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

    ImageRegion region({start.row, start.col, start.channel},
                       {shape.row, shape.col, shape.channel});

    const auto chans_per_group = rng.rand<int>(1, 32);

    const auto expected_count =
        (region.shape.depth + chans_per_group - 1) / chans_per_group;

    EXPECT_EQ(expected_count, region.ChannelOutputGroups(chans_per_group));
  }
}
