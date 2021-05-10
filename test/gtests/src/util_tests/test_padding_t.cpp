
#include <iostream>
#include <vector>
#include <tuple>

#include "gtest/gtest.h"

#include "geom/util.hpp"
#include "Rand.hpp"

using namespace nn;
using namespace nn::test;






/////////////////////////////////////////////////////////////////////////
//
//
TEST(padding_t_Test, MakeUnsigned)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(6456);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    padding_t pad;
    pad.top    = rng.rand<int16_t>();
    pad.left   = rng.rand<int16_t>();
    pad.bottom = rng.rand<int16_t>();
    pad.right  = rng.rand<int16_t>();

    padding_t expected;
    expected.top    = (pad.top    <= 0)? 0 : pad.top;
    expected.left   = (pad.left   <= 0)? 0 : pad.left;
    expected.bottom = (pad.bottom <= 0)? 0 : pad.bottom;
    expected.right  = (pad.right  <= 0)? 0 : pad.right;

    pad.MakeUnsigned();

    ASSERT_EQ(expected.top,    pad.top)    << "pad: " << pad;
    ASSERT_EQ(expected.left,   pad.left)   << "pad: " << pad;
    ASSERT_EQ(expected.bottom, pad.bottom) << "pad: " << pad;
    ASSERT_EQ(expected.right,  pad.right)  << "pad: " << pad;
  }
}


/////////////////////////////////////////////////////////////////////////
//
//
TEST(padding_t_Test, HasPadding)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(6456);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    padding_t pad;
    pad.top    = rng.rand<int16_t>(-10, 10);
    pad.left   = rng.rand<int16_t>(-10, 10);
    pad.bottom = rng.rand<int16_t>(-10, 10);
    pad.right  = rng.rand<int16_t>(-10, 10);

    bool has_pad = false;
    has_pad = has_pad || pad.top    > 0;
    has_pad = has_pad || pad.left   > 0;
    has_pad = has_pad || pad.bottom > 0;
    has_pad = has_pad || pad.right  > 0;

    ASSERT_EQ(has_pad, pad.HasPadding());

  }
}



/////////////////////////////////////////////////////////////////////////
//
//
TEST(padding_t_Test, equality)
{
  constexpr int ITER_COUNT = 1000;

  auto rng = Rand(898754);

  for(int iter = 0; iter < ITER_COUNT; iter++){

    bool are_equal = rng.rand<bool>();

    padding_t padA, padB;
    padA.top    = rng.rand<int16_t>(-10, 10);
    padA.left   = rng.rand<int16_t>(-10, 10);
    padA.bottom = rng.rand<int16_t>(-10, 10);
    padA.right  = rng.rand<int16_t>(-10, 10);

    if(are_equal) {
      padB = padA;
    } else {
      padB.top    = rng.rand<int16_t>(-10, 10);
      padB.left   = rng.rand<int16_t>(-10, 10);
      padB.bottom = rng.rand<int16_t>(-10, 10);
      padB.right  = rng.rand<int16_t>(-10, 10);
    }
    
    ASSERT_EQ(are_equal,  padA == padB);
    ASSERT_EQ(!are_equal, padA != padB);
  }
}


