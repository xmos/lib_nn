
#include "../src/filt2d/f2d.hpp"

#include "gtest/gtest.h"

#include <iostream>
#include <cstring>
#include <cstdlib>
#include <ctime>


int main(int argc, char **argv) {
  ::testing::InitGoogleTest(&argc, argv);

  srand(time(NULL));

  return RUN_ALL_TESTS();
}