
#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "../src/cpp/filt2d/geom/util.hpp"
#include "Rand.hpp"

using namespace nn::filt2d;
using namespace nn::test;

class ImageVect_1Vect_Test : public ::testing::TestWithParam<std::tuple<int,int,int>> {};

TEST_P(ImageVect_1Vect_Test, Constructor)
{
  const auto row = std::get<0>(GetParam());
  const auto col = std::get<1>(GetParam());
  const auto xan = std::get<2>(GetParam());
  
  auto vect = ImageVect(row, col, xan);

  ASSERT_EQ(vect.row, row);
  ASSERT_EQ(vect.col, col);
  ASSERT_EQ(vect.channel, xan);
}
  
static auto params_1vect = RandRangeTupleIter<int,int,int>(1000, {-1000, 1000}, 
                                                                 {-1000, 1000}, 
                                                                 {-1000, 1000} );

INSTANTIATE_TEST_SUITE_P(, ImageVect_1Vect_Test, ::testing::ValuesIn( params_1vect.begin(), params_1vect.end()));






class ImageVect_2Vect_Test : public ::testing::TestWithParam<std::tuple<
                                                               std::tuple<int,int,int>,
                                                               std::tuple<int,int,int>>> {};

TEST_P(ImageVect_2Vect_Test, add)
{
  const auto row1 = std::get<0>(std::get<0>(GetParam()));
  const auto col1 = std::get<1>(std::get<0>(GetParam()));
  const auto xan1 = std::get<2>(std::get<0>(GetParam()));
  const auto row2 = std::get<0>(std::get<1>(GetParam()));
  const auto col2 = std::get<1>(std::get<1>(GetParam()));
  const auto xan2 = std::get<2>(std::get<1>(GetParam()));


  auto vect1 = ImageVect(row1, col1, xan1);

  auto vect2 = vect1.add(row2, col2, xan2);
  
  ASSERT_EQ(vect2.row, row1+row2);
  ASSERT_EQ(vect2.col, col1+col2);
  ASSERT_EQ(vect2.channel, xan1+xan2);

}

TEST_P(ImageVect_2Vect_Test, sub)
{
  const auto row1 = std::get<0>(std::get<0>(GetParam()));
  const auto col1 = std::get<1>(std::get<0>(GetParam()));
  const auto xan1 = std::get<2>(std::get<0>(GetParam()));
  const auto row2 = std::get<0>(std::get<1>(GetParam()));
  const auto col2 = std::get<1>(std::get<1>(GetParam()));
  const auto xan2 = std::get<2>(std::get<1>(GetParam()));


  auto vect1 = ImageVect(row1, col1, xan1);

  auto vect2 = vect1.sub(row2, col2, xan2);
  
  ASSERT_EQ(vect2.row,     row1-row2);
  ASSERT_EQ(vect2.col,     col1-col2);
  ASSERT_EQ(vect2.channel, xan1-xan2);

}

TEST_P(ImageVect_2Vect_Test, add_operator)
{
  const auto row1 = std::get<0>(std::get<0>(GetParam()));
  const auto col1 = std::get<1>(std::get<0>(GetParam()));
  const auto xan1 = std::get<2>(std::get<0>(GetParam()));
  const auto row2 = std::get<0>(std::get<1>(GetParam()));
  const auto col2 = std::get<1>(std::get<1>(GetParam()));
  const auto xan2 = std::get<2>(std::get<1>(GetParam()));


  auto vect1 = ImageVect(row1, col1, xan1);
  auto vect2 = ImageVect(row2, col2, xan2);

  auto vect3 = vect1 + vect2;
  
  ASSERT_EQ(vect3.row,     row1+row2);
  ASSERT_EQ(vect3.col,     col1+col2);
  ASSERT_EQ(vect3.channel, xan1+xan2);
}

TEST_P(ImageVect_2Vect_Test, sub_operator)
{
  const auto row1 = std::get<0>(std::get<0>(GetParam()));
  const auto col1 = std::get<1>(std::get<0>(GetParam()));
  const auto xan1 = std::get<2>(std::get<0>(GetParam()));
  const auto row2 = std::get<0>(std::get<1>(GetParam()));
  const auto col2 = std::get<1>(std::get<1>(GetParam()));
  const auto xan2 = std::get<2>(std::get<1>(GetParam()));


  auto vect1 = ImageVect(row1, col1, xan1);
  auto vect2 = ImageVect(row2, col2, xan2);

  auto vect3 = vect1 - vect2;
  
  ASSERT_EQ(vect3.row,     row1-row2);
  ASSERT_EQ(vect3.col,     col1-col2);
  ASSERT_EQ(vect3.channel, xan1-xan2);
}

static auto params_2vect = RandRangeIter< std::tuple<std::tuple<int,int,int>,
                                                     std::tuple<int,int,int>> >(1000, 
                                                  {{-1000, -1000, -1000}, {-1000, -1000, -1000}}, 
                                                  {{ 1000,  1000,  1000}, { 1000,  1000,  1000}} );

INSTANTIATE_TEST_SUITE_P(, ImageVect_2Vect_Test, ::testing::ValuesIn( params_2vect.begin(), params_2vect.end()));



class ImageVect_2VectB_Test : public ::testing::TestWithParam<std::tuple<int,int,int,int,int,int>> {};


TEST_P(ImageVect_2VectB_Test, eq_neq_operator)
{
  const auto row1 = std::get<0>(GetParam());
  const auto col1 = std::get<1>(GetParam());
  const auto xan1 = std::get<2>(GetParam());
  const auto row2 = std::get<3>(GetParam());
  const auto col2 = std::get<4>(GetParam());
  const auto xan2 = std::get<5>(GetParam());


  auto vect1 = ImageVect(row1, col1, xan1);
  auto vect2 = ImageVect(row2, col2, xan2);

  bool should_eq = row1 == row2 && col1 == col2 && xan1 == xan2;

  ASSERT_EQ(vect1 == vect2, should_eq);
  ASSERT_EQ(vect1 != vect2, !should_eq);
}

INSTANTIATE_TEST_SUITE_P(, ImageVect_2VectB_Test, ::testing::Combine(
                                                      ::testing::Range<int>(-2, 2),
                                                      ::testing::Range<int>(-2, 2),
                                                      ::testing::Range<int>(-2, 2),
                                                      ::testing::Range<int>(-2, 2),
                                                      ::testing::Range<int>(-2, 2),
                                                      ::testing::Range<int>(-2, 2) ));

