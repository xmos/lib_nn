
#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "../src/cpp/filt2d/util/AddressCovector.hpp"

using namespace nn;

class AddressCovectorBaseTest : public ::testing::TestWithParam<std::tuple<int,int,int>> {};


TEST_P(AddressCovectorBaseTest, Constructor)
{
  const auto width  = std::get<1>(GetParam());
  const auto depth  = std::get<2>(GetParam());
  
  auto cov = AddressCovectorBase( width * depth * sizeof(int32_t), 
                                  depth * sizeof(int32_t), 
                                  sizeof(int32_t));

  // There isn't really much to test here.
  EXPECT_EQ(cov.row_bytes,  width * depth * sizeof(int32_t));
  EXPECT_EQ(cov.col_bytes,  depth * sizeof(int32_t));
  EXPECT_EQ(cov.chan_bytes, sizeof(int32_t));
  EXPECT_EQ(cov.zero, 0);
}

TEST_P(AddressCovectorBaseTest, DotRowColChannel)
{
  const auto height = std::get<0>(GetParam());
  const auto width  = std::get<1>(GetParam());
  const auto depth  = std::get<2>(GetParam());
  
  auto cov = AddressCovectorBase( width * depth * sizeof(int32_t), 
                                  depth * sizeof(int32_t), 
                                  sizeof(int32_t));

  for(auto row = 0; row < height; ++row) {
    for(auto col = 0; col < width; ++col) {
      for(auto chan = 0; chan < depth; ++chan) {
        EXPECT_EQ(cov.dot(row,col,chan), sizeof(int32_t) * (depth * ((row * width) + col) + chan)  );
      }
    }
  }
}

TEST_P(AddressCovectorBaseTest, DotImageVect)
{
  const auto height = std::get<0>(GetParam());
  const auto width  = std::get<1>(GetParam());
  const auto depth  = std::get<2>(GetParam());
  
  auto cov = AddressCovectorBase( width * depth * sizeof(int32_t), 
                                  depth * sizeof(int32_t), 
                                  sizeof(int32_t));

  for(auto row = 0; row < height; ++row) {
    for(auto col = 0; col < width; ++col) {
      for(auto chan = 0; chan < depth; ++chan) {
        EXPECT_EQ(cov.dot(ImageVect(row, col, chan)), sizeof(int32_t) * (depth * ((row * width) + col) + chan)  );
      }
    }
  }
}


INSTANTIATE_TEST_SUITE_P(, AddressCovectorBaseTest, ::testing::Combine(
                                                        ::testing::Range(1, 10),    // Height  
                                                        ::testing::Range(1, 10),    // Width
                                                        ::testing::Range(1, 20) ));  // Depth



class AddressCovectorTest : public ::testing::TestWithParam<std::tuple<int,int,int>> {

  private:

    std::vector<int32_t> img;

  protected:

    int height;
    int width;
    int depth;

    virtual void SetUp() override 
    {
      height = std::get<0>(GetParam());
      width  = std::get<1>(GetParam());
      depth  = std::get<2>(GetParam());

      img.resize(height * width * depth);
    }

    virtual int32_t& ImageElement(int row, int col, int channel)
    {
      return this->img[row * (width * depth) + col * depth + channel];
    }

};


TEST_P(AddressCovectorTest, Constructor_A)
{
  auto cov = AddressCovector<int32_t>(width * depth * sizeof(int32_t),
                                      depth * sizeof(int32_t),
                                      sizeof(int32_t)  );

  // There isn't really much to test here.
  EXPECT_EQ(cov.row_bytes,  (int16_t) (((uint64_t)&ImageElement(1,0,0)) - ((uint64_t)&ImageElement(0,0,0)) )  );
  EXPECT_EQ(cov.col_bytes,  (int16_t) (((uint64_t)&ImageElement(0,1,0)) - ((uint64_t)&ImageElement(0,0,0)) )  );
  EXPECT_EQ(cov.chan_bytes, (int16_t) (((uint64_t)&ImageElement(0,0,1)) - ((uint64_t)&ImageElement(0,0,0)) )  );
  EXPECT_EQ(cov.zero, 0);
}

TEST_P(AddressCovectorTest, Constructor_B)
{
  auto cov = AddressCovector<int32_t>(width, depth);

  // There isn't really much to test here.
  EXPECT_EQ(cov.row_bytes,  (int16_t) (((uint64_t)&ImageElement(1,0,0)) - ((uint64_t)&ImageElement(0,0,0)) )  );
  EXPECT_EQ(cov.col_bytes,  (int16_t) (((uint64_t)&ImageElement(0,1,0)) - ((uint64_t)&ImageElement(0,0,0)) )  );
  EXPECT_EQ(cov.chan_bytes, (int16_t) (((uint64_t)&ImageElement(0,0,1)) - ((uint64_t)&ImageElement(0,0,0)) )  );
  EXPECT_EQ(cov.zero, 0);
}


TEST_P(AddressCovectorTest, resolve_A)
{
  auto cov = AddressCovector<int32_t>(width, depth);

  for(int r = 0; r < height; ++r){
    for(int c = 0; c < width; ++c){
      for(int k = 0; k < depth; ++k){
        ImageElement(r,c,k) = (r + 1)*(c + 7)*(k + 13);
      }
    }
  }

  int32_t* base = &ImageElement(0,0,0);

  
  for(int r = 0; r < height; ++r){
    for(int c = 0; c < width; ++c){
      for(int k = 0; k < depth; ++k){
        ASSERT_EQ(cov.resolve(base, r,c,k), &ImageElement(r,c,k));
        ASSERT_EQ(cov.resolve(base, r,c,k)[0], (r + 1)*(c + 7)*(k + 13));
      }
    }
  }
}


TEST_P(AddressCovectorTest, resolve_B)
{
  auto cov = AddressCovector<int32_t>(width, depth);

  for(int r = 0; r < height; ++r){
    for(int c = 0; c < width; ++c){
      for(int k = 0; k < depth; ++k){
        ImageElement(r,c,k) = (r + 1)*(c + 7)*(k + 13);
      }
    }
  }

  int32_t* base = &ImageElement(0,0,0);

  
  for(int r = 0; r < height; ++r){
    for(int c = 0; c < width; ++c){
      for(int k = 0; k < depth; ++k){
        ASSERT_EQ(cov.resolve(base, ImageVect(r,c,k)), &ImageElement(r,c,k));
        ASSERT_EQ(cov.resolve(base, ImageVect(r,c,k))[0], (r + 1)*(c + 7)*(k + 13));
      }
    }
  }
}

INSTANTIATE_TEST_SUITE_P(, AddressCovectorTest, ::testing::Combine(
                                                        ::testing::Range(1, 10),    // Height  
                                                        ::testing::Range(1, 10),    // Width
                                                        ::testing::Range(1, 20) ));  // Depth