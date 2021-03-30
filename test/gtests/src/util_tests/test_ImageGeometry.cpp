
#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "../src/cpp/filt2d/geom/ImageGeometry.hpp"
#include "Rand.hpp"

using namespace nn;

class ImageGeometryTest : public ::testing::TestWithParam<ImageGeometry> {};


TEST_P(ImageGeometryTest, pixelElements)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth, img.pixelElements());
}

TEST_P(ImageGeometryTest, rowElements)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth * img.width, img.rowElements());
}

TEST_P(ImageGeometryTest, colElements)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth * img.height, img.colElements());
}

TEST_P(ImageGeometryTest, imageElements)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth * img.width * img.height, img.imageElements());
}

TEST_P(ImageGeometryTest, pixelBytes)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth * img.channel_depth, img.pixelBytes());
}

TEST_P(ImageGeometryTest, rowBytes)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth * img.width * img.channel_depth, img.rowBytes());
}

TEST_P(ImageGeometryTest, colBytes)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth * img.height * img.channel_depth, img.colBytes());
}

TEST_P(ImageGeometryTest, imageBytes)
{
  auto img = GetParam();

  ASSERT_EQ(img.depth * img.width * img.height * img.channel_depth, img.imageBytes());
}



TEST_P(ImageGeometryTest, getStride)
{
  auto img = GetParam();

  // imageBytes() so it is deterministic but not the same for every case
  auto rng = nn::test::Rand(img.imageBytes());

  for(int k = 0; k < 10; ++k){

    auto rows1 = rng.rand<unsigned>(0, img.height-1);
    auto cols1 = rng.rand<unsigned>(0, img.width -1);
    auto xans1 = rng.rand<unsigned>(0, img.depth -1);
    auto vect1 = ImageVect(rows1, cols1, xans1);

    auto rows2 = rng.rand<unsigned>(0, img.height-1);
    auto cols2 = rng.rand<unsigned>(0, img.width -1);
    auto xans2 = rng.rand<unsigned>(0, img.depth -1);
    auto vect2 = ImageVect(rows2, cols2, xans2);

    auto stride = img.getStride(rows1, cols1, xans1);

    ASSERT_EQ(stride, (xans1 + cols1 * img.depth + rows1 * img.depth * img.width) * img.channel_depth);

    stride = img.getStride(vect1);

    ASSERT_EQ(stride, (xans1 + cols1 * img.depth + rows1 * img.depth * img.width) * img.channel_depth);

    stride = img.getStride( vect1, vect2 );

    auto delta = vect2 - vect1;

    ASSERT_EQ(stride, (delta.channel + delta.col * img.depth + delta.row * img.depth * img.width) * img.channel_depth);
  }
}

TEST_P(ImageGeometryTest, getAddressCovector)
{
  auto img = GetParam();

  // imageBytes() so it is deterministic but not the same for every case
  auto rng = nn::test::Rand(img.imageBytes()+1);

  auto cov = img.getAddressCovector<int8_t>();

  for(int k = 0; k < 10; ++k){
    auto rows = rng.rand<unsigned>(0, img.height-1);
    auto cols = rng.rand<unsigned>(0, img.width -1);
    auto xans = rng.rand<unsigned>(0, img.depth -1);

    ASSERT_EQ(cov.chan_bytes, img.channel_depth);
    ASSERT_EQ(cov.col_bytes, img.depth * img.channel_depth);
    ASSERT_EQ(cov.row_bytes, img.depth * img.width * img.channel_depth);

  }
}

static std::vector<ImageGeometry> TestGeometries()
{
  const auto max_height = 8;
  const auto max_width = 8;
  const auto max_depth = 8;
  const auto max_chan_depth_log2 = 2;

  auto res = std::vector<ImageGeometry>( );

  for(int h = 1; h < max_height; ++h)
    for(int w = 1; w < max_width; ++w)
      for(int d = 1; d < max_depth; ++d)
        for(int x = 0; x < max_chan_depth_log2; ++x)
          res.push_back(ImageGeometry(h,w,d, (1 << x) ));
  
  return res;
}


INSTANTIATE_TEST_SUITE_P(, ImageGeometryTest, ::testing::ValuesIn( TestGeometries() )); 


