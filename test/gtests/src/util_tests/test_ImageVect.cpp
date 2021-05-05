
#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "geom/util.hpp"
#include "Rand.hpp"

using namespace nn;
using namespace nn::test;

TEST(ImageVect_Test, Constructor)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    const auto row = rng.rand<int>(-1000, 1000);
    const auto col = rng.rand<int>(-1000, 1000);
    const auto xan = rng.rand<int>(-1000, 1000);
  
    auto vect = ImageVect(row, col, xan);

    ASSERT_EQ(vect.row, row);
    ASSERT_EQ(vect.col, col);
    ASSERT_EQ(vect.channel, xan);
  }
}



TEST(ImageVect_Test, add)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    const auto row1 = rng.rand<int>(-1000, 1000);
    const auto col1 = rng.rand<int>(-1000, 1000);
    const auto xan1 = rng.rand<int>(-1000, 1000);
    const auto row2 = rng.rand<int>(-1000, 1000);
    const auto col2 = rng.rand<int>(-1000, 1000);
    const auto xan2 = rng.rand<int>(-1000, 1000);

    auto vect1 = ImageVect(row1, col1, xan1);

    auto vect2 = vect1.add(row2, col2, xan2);
    
    ASSERT_EQ(vect2.row, row1+row2);
    ASSERT_EQ(vect2.col, col1+col2);
    ASSERT_EQ(vect2.channel, xan1+xan2);
  }
}

TEST(ImageVect_Test, sub)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    const auto row1 = rng.rand<int>(-1000, 1000);
    const auto col1 = rng.rand<int>(-1000, 1000);
    const auto xan1 = rng.rand<int>(-1000, 1000);
    const auto row2 = rng.rand<int>(-1000, 1000);
    const auto col2 = rng.rand<int>(-1000, 1000);
    const auto xan2 = rng.rand<int>(-1000, 1000);

    auto vect1 = ImageVect(row1, col1, xan1);

    auto vect2 = vect1.sub(row2, col2, xan2);
    
    ASSERT_EQ(vect2.row,     row1-row2);
    ASSERT_EQ(vect2.col,     col1-col2);
    ASSERT_EQ(vect2.channel, xan1-xan2);
  }
}

TEST(ImageVect_Test, add_operator)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    const auto row1 = rng.rand<int>(-1000, 1000);
    const auto col1 = rng.rand<int>(-1000, 1000);
    const auto xan1 = rng.rand<int>(-1000, 1000);
    const auto row2 = rng.rand<int>(-1000, 1000);
    const auto col2 = rng.rand<int>(-1000, 1000);
    const auto xan2 = rng.rand<int>(-1000, 1000);

    auto vect1 = ImageVect(row1, col1, xan1);
    auto vect2 = ImageVect(row2, col2, xan2);

    auto vect3 = vect1 + vect2;
    
    ASSERT_EQ(vect3.row,     row1+row2);
    ASSERT_EQ(vect3.col,     col1+col2);
    ASSERT_EQ(vect3.channel, xan1+xan2);
  }
}

TEST(ImageVect_Test, sub_operator)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    const auto row1 = rng.rand<int>(-1000, 1000);
    const auto col1 = rng.rand<int>(-1000, 1000);
    const auto xan1 = rng.rand<int>(-1000, 1000);
    const auto row2 = rng.rand<int>(-1000, 1000);
    const auto col2 = rng.rand<int>(-1000, 1000);
    const auto xan2 = rng.rand<int>(-1000, 1000);

    auto vect1 = ImageVect(row1, col1, xan1);
    auto vect2 = ImageVect(row2, col2, xan2);

    auto vect3 = vect1 - vect2;
    
    ASSERT_EQ(vect3.row,     row1-row2);
    ASSERT_EQ(vect3.col,     col1-col2);
    ASSERT_EQ(vect3.channel, xan1-xan2);

  }
}


TEST(ImageVect_Test, eq_neq_operator)
{
  for(int row1 = -2; row1 <= 2; row1++){
    for(int col1 = -2; col1 <= 2; col1++){
      for(int xan1 = -2; xan1 <= 2; xan1++){
        for(int row2 = -2; row2 <= 2; row2++){
          for(int col2 = -2; col2 <= 2; col2++){
            for(int xan2 = -2; xan2 <= 2; xan2++){

              auto vect1 = ImageVect(row1, col1, xan1);
              auto vect2 = ImageVect(row2, col2, xan2);

              bool should_eq = row1 == row2 && col1 == col2 && xan1 == xan2;

              ASSERT_EQ(vect1 == vect2, should_eq);
              ASSERT_EQ(vect1 != vect2, !should_eq);

            }
          }
        }
      }
    }
  }
}

