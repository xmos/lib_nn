
#include <iostream>
#include <vector>
#include <tuple>
#include <sstream>
#include <cassert>
#include <limits>

#include "gtest/gtest.h"

#include "AggregateFn.hpp"
#include "Rand.hpp"
#include "VpuHelpers.hpp"




class MaxPoolPatchFnParamsTest : public ::testing::TestWithParam<std::tuple<int,int>> {};
class MaxPoolPatchFnTest : public ::testing::TestWithParam<std::tuple<int,int>> {};


static std::vector<int8_t> run_op(MaxPoolPatchFn::Params* params)
{
  auto input = std::vector<int8_t>( params->pixel_count * MaxPoolPatchFn::ChannelsPerOutputGroup );
  
  for(int c = 0; c < MaxPoolPatchFn::ChannelsPerOutputGroup; c++)
    for(int p = 0; p < params->pixel_count; p++)
      input[p * MaxPoolPatchFn::ChannelsPerOutputGroup + c] = c;
  
  auto mp = MaxPoolPatchFn(params);
  vpu_ring_buffer_t acc;
  mp.aggregate_fn( &acc, &input[0], 0 );

  auto res = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
  std::memcpy( &res[0], acc.vR, MaxPoolPatchFn::ChannelsPerOutputGroup );
  return res;
}


////////////////////
TEST_P(MaxPoolPatchFnParamsTest, ConstructorA)
{
  const int height = std::get<0>(GetParam());
  const int width  = std::get<1>(GetParam());
  const auto pixel_count = height * width;

  MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( pixel_count );

  auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
  for(int i = 0; i < exp.size(); i++) exp[i] = i;

  auto res = run_op(&params);

  ASSERT_EQ(exp, res);
  
}

////////////////////
TEST_P(MaxPoolPatchFnParamsTest, ConstructorB)
{
  const int height = std::get<0>(GetParam());
  const int width  = std::get<1>(GetParam());
  const auto pixel_count = height * width;

  auto window = nn::WindowGeometry( height, width, VPU_INT8_EPV, 0, 0, 1, 1, 1);
  MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( window );

  auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
  for(int i = 0; i < exp.size(); i++) exp[i] = i;

  auto res = run_op(&params);

  ASSERT_EQ(exp, res);
}

////////////////////
TEST_P(MaxPoolPatchFnParamsTest, ConstructorC)
{
  const int height = std::get<0>(GetParam());
  const int width  = std::get<1>(GetParam());
  const auto pixel_count = height * width;

  auto stream = std::stringstream();
  stream.write(reinterpret_cast<const char*>(&pixel_count), sizeof(pixel_count));

  MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( stream );

  auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
  for(int i = 0; i < exp.size(); i++) exp[i] = i;

  auto res = run_op(&params);

  ASSERT_EQ(exp, res);
  
}


////////////////////
TEST_P(MaxPoolPatchFnParamsTest, Serialize)
{
  const int height = std::get<0>(GetParam());
  const int width  = std::get<1>(GetParam());
  const auto pixel_count = height * width;

  MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( pixel_count );

  auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
  for(int i = 0; i < exp.size(); i++) exp[i] = i;

  auto stream = std::stringstream();
  params.Serialize( stream );

  // Check whether it works by deserializing it and running it.
  params = MaxPoolPatchFn::Params( stream );

  auto res = run_op(&params);

  ASSERT_EQ(exp, res);
}


////////////////////
TEST_P(MaxPoolPatchFnTest, aggregate_fn)
{
  const int height = std::get<0>(GetParam());
  const int width  = std::get<1>(GetParam());
  const auto pixel_count = height * width;

  auto rand = nn::test::Rand( pixel_count * 34523 );

  MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( pixel_count );

  for(int i = 0; i < 1000; i++){


    auto patch = std::vector<int8_t>( pixel_count * MaxPoolPatchFn::ChannelsPerOutputGroup );
    int8_t exp[ MaxPoolPatchFn::ChannelsPerOutputGroup ];
    std::memset(exp, 0x80, sizeof(exp));

    for(int chan = 0; chan < MaxPoolPatchFn::ChannelsPerOutputGroup; chan++){
      for(int pix = 0; pix < pixel_count; pix++){
        int8_t v = rand.rand<int8_t>();
        patch[pix * MaxPoolPatchFn::ChannelsPerOutputGroup + chan] = v;
        exp[chan] = std::max<int8_t>(exp[chan], v);
      }
    }

    vpu_ring_buffer_t acc;
    const auto* output = reinterpret_cast<const int8_t*>( acc.vR );
    auto op = MaxPoolPatchFn(&params);
    op.aggregate_fn( &acc, &patch[0], 0);

    for(int chan = 0; chan < MaxPoolPatchFn::ChannelsPerOutputGroup; chan++){
      ASSERT_EQ(exp[chan], output[chan]);
    }
  }
}


INSTANTIATE_TEST_SUITE_P(, MaxPoolPatchFnParamsTest,  ::testing::Combine(
                                                          ::testing::Range(1, 6, 1),
                                                          ::testing::Range(1, 6, 1)) );
INSTANTIATE_TEST_SUITE_P(, MaxPoolPatchFnTest, ::testing::Combine(
                                                  ::testing::Range(1, 6, 1),
                                                  ::testing::Range(1, 6, 1)) );






class maxpool_patch_ref_test : public ::testing::TestWithParam<int> {};

TEST_P(maxpool_patch_ref_test, Test)
{
  const auto pixel_count = GetParam();
  auto rand = nn::test::Rand(4555 * pixel_count);

  auto patch = std::vector<int8_t>( pixel_count * VPU_INT8_EPV );

  int8_t exp[VPU_INT8_EPV];

  for(int iter = 0; iter < 1000; iter++){

    std::memset(exp, 0x80, sizeof(exp));

    for(int pix = 0; pix < pixel_count; pix++){
      for(int chan = 0; chan < VPU_INT8_EPV; chan++){
        auto v = rand.rand<int8_t>();
        patch[pix * VPU_INT8_EPV + chan] = v;
        exp[chan] = std::max<int8_t>( exp[chan], v );
      }
    }

    vpu_ring_buffer_t acc;
    int8_t* out = reinterpret_cast<int8_t*>(&acc);

    maxpool_patch_ref(&acc, &patch[0], pixel_count);
      
    for(int i = 0; i < VPU_INT8_EPV; i++){
      ASSERT_EQ(exp[i], out[i])
        << "Failure Details..\n"
        << "  i = " << i << "\n";
    }
  }
}

INSTANTIATE_TEST_SUITE_P(, maxpool_patch_ref_test, ::testing::Range(1, 20));