
#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "../src/cpp/filt2d/geom/ImageGeometry.hpp"
#include "Rand.hpp"

using namespace nn;


/**
 * Generates a set of ImageGeometry objects used as parameters for tests.
 * It's templated because the GetElement() method is templated, and I don't know how to write
 * the GetElement() test in a way that works for more than one template type (and isn't ugly)
 */
template <typename T>
static std::vector<ImageGeometry> TestGeometries()
{
  const auto max_height = 8;
  const auto max_width = 8;
  const auto max_depth = 8;

  auto res = std::vector<ImageGeometry>( );

  for(int h = 1; h < max_height; ++h)
    for(int w = 1; w < max_width; ++w)
      for(int d = 1; d < max_depth; ++d)
        res.push_back(ImageGeometry(h,w,d, sizeof(T) ));
  
  return res;
}

/**
 * Test Classes
 *
 * (I wouldn't need to specify more than one class here, except that when running parameterized tests, ALL parameters
 *  are run against ALL tests of that class. Unfortunate.)
 */

/** ImageGeometryTest is run against all three types */
class ImageGeometryTest : public ::testing::TestWithParam<ImageGeometry> {};
class ImageGeometryTest_int8 : public ImageGeometryTest {};
class ImageGeometryTest_int16 : public ImageGeometryTest {};
class ImageGeometryTest_int32 : public ImageGeometryTest {};

/**
 * Tests
 */

TEST_P(ImageGeometryTest, imagePixels)
{
  auto img = GetParam();
  ASSERT_EQ(img.width * img.height, img.imagePixels());
}


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






TEST_P(ImageGeometryTest, IsWithinImage)
{
  auto img = GetParam();

  for(int row = 0; row < img.height; ++row){
    for(int col = 0; col < img.width; ++col){
      for(int chan = 0; chan < img.depth; ++chan){
        EXPECT_TRUE( img.IsWithinImage( ImageVect(row, col, chan) ) );
        EXPECT_TRUE( img.IsWithinImage( row, col, chan ) );
      }
    }
  }

  for(int row = -2; row <= 2; ++row){
    for(int col = -2; col <= 2; ++col){
      for(int chan = -2; chan <= 2; ++chan){
        if(row==0 && col==0 && chan==0) continue;

        int xr = row + ((row <= 0)? 0 : (int(img.height)-1));
        int xc = col + ((col <= 0)? 0 : (int(img.width)-1));
        int xx = chan + ((chan <= 0)? 0 : (int(img.depth)-1));

        EXPECT_FALSE( img.IsWithinImage( ImageVect(xr, xc, xx) ) );
        EXPECT_FALSE( img.IsWithinImage( xr, xc, xx ) );
      }
    }
  }
}


TEST_P(ImageGeometryTest_int8, Element)
{
  using T_elm = int8_t;
  auto img = GetParam();
  auto input = std::vector<T_elm>( img.imageElements() );
  int k = 0;
  for(int row = 0; row < img.height; ++row){
    for(int col = 0; col < img.width; ++col){
      for(int chan = 0; chan < img.depth; ++chan){

#define FAIL_MSG  "v = " << nn::ImageVect(row, col, chan)

        T_elm& refA = img.Element<T_elm>(&input[0], row, col, chan);
        T_elm& refB = input[k++];
        ASSERT_EQ(&(refA), &(refB))                   << FAIL_MSG;
        refA = T_elm(13*row + 7*col + chan);
        ASSERT_EQ(refB, T_elm(13*row + 7*col + chan)) << FAIL_MSG;

#undef FAIL_MSG
      }
    }
  }
}

TEST_P(ImageGeometryTest_int16, Element)
{
  using T_elm = int16_t;
  auto img = GetParam();
  auto input = std::vector<T_elm>( img.imageElements() );
  int k = 0;
  for(int row = 0; row < img.height; ++row){
    for(int col = 0; col < img.width; ++col){
      for(int chan = 0; chan < img.depth; ++chan){
        T_elm& refA = img.Element<T_elm>(&input[0], row, col, chan);
        T_elm& refB = input[k++];
        ASSERT_EQ(&(refA), &(refB));
        refA = T_elm(13*row + 7*col + chan);
        ASSERT_EQ(refB, T_elm(13*row + 7*col + chan));
      }
    }
  }
}

TEST_P(ImageGeometryTest_int32, Element)
{
  using T_elm = int32_t;
  auto img = GetParam();
  auto input = std::vector<T_elm>( img.imageElements() );
  int k = 0;
  for(int row = 0; row < img.height; ++row){
    for(int col = 0; col < img.width; ++col){
      for(int chan = 0; chan < img.depth; ++chan){
        T_elm& refA = img.Element<T_elm>(&input[0], row, col, chan);
        T_elm& refB = input[k++];
        ASSERT_EQ(&(refA), &(refB));
        refA = T_elm(13*row + 7*col + chan);
        ASSERT_EQ(refB, T_elm(13*row + 7*col + chan));
      }
    }
  }
}



TEST_P(ImageGeometryTest_int8, Get)
{
  using T_elm = int8_t;
  auto img = GetParam();
  auto input = std::vector<T_elm>( img.imageElements() );
  int k = 0;
  for(int row = -1; row <= img.height; ++row){
    for(int col = -1; col <= img.width; ++col){
      for(int xan = -1; xan <= img.depth; ++xan){
        ImageVect v(row,col,xan);
#define FAIL_MSG  "v = " << v << " | k = " << k
        if(img.IsWithinImage(v)){
          T_elm& elm = input[k++];
          elm = -23;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -23)             << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -23) << FAIL_MSG;
          elm = 77;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, 99), 77)               << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, 99), 77)   << FAIL_MSG;
        } else {
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -52)             << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -52) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v,  99),  99)             << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan,  99),  99) << FAIL_MSG;
        }
#undef FAIL_MSG
      }
    }
  }  
}



TEST_P(ImageGeometryTest_int16, Get)
{
  using T_elm = int16_t;
  auto img = GetParam();
  auto input = std::vector<T_elm>( img.imageElements() );
  int k = 0;
  for(int row = -1; row <= img.height; ++row){
    for(int col = -1; col <= img.width; ++col){
      for(int xan = -1; xan <= img.depth; ++xan){
        ImageVect v(row,col,xan);
#define FAIL_MSG  "v = " << v << " | k = " << k
        if(img.IsWithinImage(v)){
          T_elm& elm = input[k++];
          elm = -23;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -23) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -23) << FAIL_MSG;
          elm = 77;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, 99), 77) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, 99), 77) << FAIL_MSG;
        } else {
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -52) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -52) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v,  99),  99) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan,  99),  99) << FAIL_MSG;
        }
#undef FAIL_MSG
      }
    }
  }  
}



TEST_P(ImageGeometryTest_int32, Get)
{
  using T_elm = int32_t;
  auto img = GetParam();
  auto input = std::vector<T_elm>( img.imageElements() );
  int k = 0;
  for(int row = -1; row <= img.height; ++row){
    for(int col = -1; col <= img.width; ++col){
      for(int xan = -1; xan <= img.depth; ++xan){
        ImageVect v(row,col,xan);
#define FAIL_MSG  "v = " << v << " | k = " << k
        if(img.IsWithinImage(v)){
          T_elm& elm = input[k++];
          elm = -23;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -23) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -23) << FAIL_MSG;
          elm = 77;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, 99), 77) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, 99), 77) << FAIL_MSG;
        } else {
          ASSERT_EQ(img.Get<T_elm>(&input[0], v, -52), -52) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan, -52), -52) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], v,  99),  99) << FAIL_MSG;
          ASSERT_EQ(img.Get<T_elm>(&input[0], row, col, xan,  99),  99) << FAIL_MSG;
        }
#undef FAIL_MSG
      }
    }
  }  
}


INSTANTIATE_TEST_SUITE_P(int8,  ImageGeometryTest, ::testing::ValuesIn( TestGeometries<int8_t>() )); 
INSTANTIATE_TEST_SUITE_P(int16, ImageGeometryTest, ::testing::ValuesIn( TestGeometries<int16_t>() )); 
INSTANTIATE_TEST_SUITE_P(int32, ImageGeometryTest, ::testing::ValuesIn( TestGeometries<int32_t>() )); 

INSTANTIATE_TEST_SUITE_P(,  ImageGeometryTest_int8,  ::testing::ValuesIn( TestGeometries<int8_t>() )); 
INSTANTIATE_TEST_SUITE_P(, ImageGeometryTest_int16, ::testing::ValuesIn( TestGeometries<int16_t>() )); 
INSTANTIATE_TEST_SUITE_P(, ImageGeometryTest_int32, ::testing::ValuesIn( TestGeometries<int32_t>() )); 

