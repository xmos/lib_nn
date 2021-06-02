
#include <iostream>
#include <tuple>
#include <vector>

#include "Rand.hpp"
#include "geom/util.hpp"
#include "gtest/gtest.h"

using namespace nn;
using namespace nn::test;

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageVect_Test, Constructor) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const auto row = rng.rand<int>(-1000, 1000);
    const auto col = rng.rand<int>(-1000, 1000);
    const auto xan = rng.rand<int>(-1000, 1000);

    auto vect1 = ImageVect(row, col, xan);
    ImageVect vect2 = {row, col, xan};

    ASSERT_EQ(vect1.row, row);
    ASSERT_EQ(vect1.col, col);
    ASSERT_EQ(vect1.channel, xan);

    ASSERT_EQ(vect2.row, row);
    ASSERT_EQ(vect2.col, col);
    ASSERT_EQ(vect2.channel, xan);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageVect_Test, addition) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const auto rowA = rng.rand<int>(-1000, 1000);
    const auto colA = rng.rand<int>(-1000, 1000);
    const auto xanA = rng.rand<int>(-1000, 1000);

    const auto rowB = rng.rand<int>(-1000, 1000);
    const auto colB = rng.rand<int>(-1000, 1000);
    const auto xanB = rng.rand<int>(-1000, 1000);

    ImageVect vectA = {rowA, colA, xanA};
    ImageVect vectB = {rowB, colB, xanB};

    {
      auto sum_vect = vectA.add(rowB, colB, xanB);
      ASSERT_EQ(sum_vect.row, rowA + rowB);
      ASSERT_EQ(sum_vect.col, colA + colB);
      ASSERT_EQ(sum_vect.channel, xanA + xanB);
    }
    {
      auto sum_vect = vectA + vectB;
      ASSERT_EQ(sum_vect.row, rowA + rowB);
      ASSERT_EQ(sum_vect.col, colA + colB);
      ASSERT_EQ(sum_vect.channel, xanA + xanB);
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageVect_Test, subtraction) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const auto rowA = rng.rand<int>(-1000, 1000);
    const auto colA = rng.rand<int>(-1000, 1000);
    const auto xanA = rng.rand<int>(-1000, 1000);

    const auto rowB = rng.rand<int>(-1000, 1000);
    const auto colB = rng.rand<int>(-1000, 1000);
    const auto xanB = rng.rand<int>(-1000, 1000);

    ImageVect vectA = {rowA, colA, xanA};
    ImageVect vectB = {rowB, colB, xanB};

    {
      auto sum_vect = vectA.sub(rowB, colB, xanB);
      ASSERT_EQ(sum_vect.row, rowA - rowB);
      ASSERT_EQ(sum_vect.col, colA - colB);
      ASSERT_EQ(sum_vect.channel, xanA - xanB);
    }
    {
      auto sum_vect = vectA - vectB;
      ASSERT_EQ(sum_vect.row, rowA - rowB);
      ASSERT_EQ(sum_vect.col, colA - colB);
      ASSERT_EQ(sum_vect.channel, xanA - xanB);
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(ImageVect_Test, equality) {
  for (int row1 = -2; row1 <= 2; row1++) {
    for (int col1 = -2; col1 <= 2; col1++) {
      for (int xan1 = -2; xan1 <= 2; xan1++) {
        for (int row2 = -2; row2 <= 2; row2++) {
          for (int col2 = -2; col2 <= 2; col2++) {
            for (int xan2 = -2; xan2 <= 2; xan2++) {
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
