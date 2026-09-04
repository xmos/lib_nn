// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include "Rand.hpp"
#include "geom/util.hpp"

extern "C" {
#include "unity.h"
#include "unity_fixture.h"
}

using namespace nn;
using namespace nn::test;

extern "C" {

TEST_GROUP(group_ImageVect);
TEST_SETUP(group_ImageVect) {}
TEST_TEAR_DOWN(group_ImageVect) {}
TEST_GROUP_RUNNER(group_ImageVect) {
  RUN_TEST_CASE(group_ImageVect, Constructor);
  RUN_TEST_CASE(group_ImageVect, addition);
  RUN_TEST_CASE(group_ImageVect, subtraction);
  RUN_TEST_CASE(group_ImageVect, equality);
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageVect, Constructor) {
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(4563456);

  for (int iter = 0; iter < ITER_COUNT; iter++) {
    const auto row = rng.rand<int>(-1000, 1000);
    const auto col = rng.rand<int>(-1000, 1000);
    const auto xan = rng.rand<int>(-1000, 1000);

    auto vect1 = ImageVect(row, col, xan);
    ImageVect vect2 = {row, col, xan};

    TEST_ASSERT_EQUAL(vect1.row, row);
    TEST_ASSERT_EQUAL(vect1.col, col);
    TEST_ASSERT_EQUAL(vect1.channel, xan);

    TEST_ASSERT_EQUAL(vect2.row, row);
    TEST_ASSERT_EQUAL(vect2.col, col);
    TEST_ASSERT_EQUAL(vect2.channel, xan);
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageVect, addition) {
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
      TEST_ASSERT_EQUAL(sum_vect.row, rowA + rowB);
      TEST_ASSERT_EQUAL(sum_vect.col, colA + colB);
      TEST_ASSERT_EQUAL(sum_vect.channel, xanA + xanB);
    }
    {
      auto sum_vect = vectA + vectB;
      TEST_ASSERT_EQUAL(sum_vect.row, rowA + rowB);
      TEST_ASSERT_EQUAL(sum_vect.col, colA + colB);
      TEST_ASSERT_EQUAL(sum_vect.channel, xanA + xanB);
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageVect, subtraction) {
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
      TEST_ASSERT_EQUAL(sum_vect.row, rowA - rowB);
      TEST_ASSERT_EQUAL(sum_vect.col, colA - colB);
      TEST_ASSERT_EQUAL(sum_vect.channel, xanA - xanB);
    }
    {
      auto sum_vect = vectA - vectB;
      TEST_ASSERT_EQUAL(sum_vect.row, rowA - rowB);
      TEST_ASSERT_EQUAL(sum_vect.col, colA - colB);
      TEST_ASSERT_EQUAL(sum_vect.channel, xanA - xanB);
    }
  }
}

/////////////////////////////////////////////////////////////////////////
//
//
TEST(group_ImageVect, equality) {
  for (int row1 = -2; row1 <= 2; row1++) {
    for (int col1 = -2; col1 <= 2; col1++) {
      for (int xan1 = -2; xan1 <= 2; xan1++) {
        for (int row2 = -2; row2 <= 2; row2++) {
          for (int col2 = -2; col2 <= 2; col2++) {
            for (int xan2 = -2; xan2 <= 2; xan2++) {
              auto vect1 = ImageVect(row1, col1, xan1);
              auto vect2 = ImageVect(row2, col2, xan2);

              bool should_eq = row1 == row2 && col1 == col2 && xan1 == xan2;

              TEST_ASSERT_EQUAL(vect1 == vect2, should_eq);
              TEST_ASSERT_EQUAL(vect1 != vect2, !should_eq);
            }
          }
        }
      }
    }
  }
}

}  // extern "C"
