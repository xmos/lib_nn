

#include "nn_types.h"
#include "geom/util.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "RefOps.hpp"
#include "Rand.hpp"
#include "ref_tests.hpp"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>


using namespace nn;
using namespace nn::test;

/**
 * 
 * 
 * 
 */
class FullyConnectedReferenceTest : public ::testing::TestWithParam<std::tuple<int, int>> {};

TEST_P(FullyConnectedReferenceTest, SimpleTest)
{
  const auto N_chans_in  = std::get<0>(GetParam());
  const auto N_chans_out = std::get<1>(GetParam());

  auto rng = Rand(N_chans_in * N_chans_out);

  auto weights  = std::vector<int8_t>( N_chans_out * N_chans_in );
  auto bias     = std::vector<int32_t>( N_chans_out );
  auto input    = std::vector<int8_t>( N_chans_in );
  auto expected = std::vector<int8_t>( N_chans_out );

  for(int k = 0; k < weights.size(); k++){
    weights[k] = rng.rand<int8_t>();
  }

  for(int k = 0; k < input.size(); k++){
    input[k] = rng.rand<int8_t>();
  }

  int8_t input_zero  = rng.rand<int8_t>(-10, 10);

  int64_t most_acc_min = std::numeric_limits<int64_t>::max();
  int64_t most_acc_max = std::numeric_limits<int64_t>::min();

  for(int k = 0; k < N_chans_out; k++){

    int64_t acc_min = 0;
    int64_t acc_max = 0;

    int offset = N_chans_in * k;

    for(int i = 0; i < N_chans_in; i++){

      int32_t v_min = std::numeric_limits<int8_t>::min();
      int32_t v_max = std::numeric_limits<int8_t>::max();

      if(weights[offset + i] < 0){
        auto tmp = v_max;
        v_max = v_min;
        v_min = tmp;
      }

      v_min -= input_zero;
      v_max -= input_zero;

      acc_min += v_min * int32_t(weights[offset + i]);
      acc_max += v_max * int32_t(weights[offset + i]);
    }

    bias[k] = rng.rand<int32_t>(-100, 100);

    acc_min += bias[k];
    acc_max += bias[k];

    assert( acc_min >= std::numeric_limits<int32_t>::min() );
    assert( acc_max <= std::numeric_limits<int32_t>::max() );

    most_acc_min = std::min<int64_t>(most_acc_min, acc_min);
    most_acc_max = std::max<int64_t>(most_acc_max, acc_max);
  }

  int64_t acc_span = most_acc_max - most_acc_min;

  float output_mult = ldexpf(1, 8) / acc_span;

  int8_t output_zero = round_int8( std::numeric_limits<int8_t>::min() - (most_acc_min * output_mult) );

  for(int k = 0; k < N_chans_out; k++){

    int32_t acc = bias[k];

    int offset = N_chans_in * k;

    for(int i = 0; i < N_chans_in; i++){
      acc += (int32_t(input[i]) - input_zero) * int32_t( weights[offset + i] );
    }

    expected[k] = round_int8( acc * output_mult + output_zero );
  }


  auto output = nn::test::ops::ref::FullyConnectedReference(N_chans_in, N_chans_out, &input[0], &weights[0], 
                                                            &bias[0], output_mult, input_zero, output_zero);

  for(int k = 0; k < expected.size(); k++){
    // Very difficult to avoid conditions where slight differences round to different values
    ASSERT_NEAR(expected[k], output[k], 1);
  }

  // ASSERT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(Targeted, FullyConnectedReferenceTest, ::testing::Values( std::make_tuple(2, 3) ));

static auto iterA = nn::test::RandRangeTupleIter<int,int>(100, { 1, 1 }, { 20, 20 }, 5745 );
INSTANTIATE_TEST_SUITE_P(Random, FullyConnectedReferenceTest, ::testing::ValuesIn(iterA.begin(), iterA.end()));

