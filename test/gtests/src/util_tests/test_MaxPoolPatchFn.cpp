
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

#include "FilterGeometryIterHelper.hpp"


using namespace nn;





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



static nn::ff::FilterGeometryIterator filter_sets[] = {
  test::unpadded::AllUnpadded( nn::Filter2dGeometry({0,0,36}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}}), true, 4),
  test::padded::AllPadded( nn::Filter2dGeometry({0,0,0}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}} ), {3, 3, 3, 3}, true, 4),
};


////////////////////
TEST(MaxPoolPatchFn_Test, ConstructorA)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      const int height = filter.window.shape.height;
      const int width  = filter.window.shape.width;
      const auto pixel_count = filter.window.shape.imagePixels();

      MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( pixel_count );

      auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
      for(int i = 0; i < exp.size(); i++) exp[i] = i;

      auto res = run_op(&params);

      ASSERT_EQ(exp, res);
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}

////////////////////
TEST(MaxPoolPatchFn_Test, ConstructorB)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      const int height = filter.window.shape.height;
      const int width  = filter.window.shape.width;
      const auto pixel_count = filter.window.shape.imagePixels();

      auto window = nn::WindowGeometry( height, width, VPU_INT8_EPV, 0, 0, 1, 1, 1);
      MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( window );

      auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
      for(int i = 0; i < exp.size(); i++) exp[i] = i;

      auto res = run_op(&params);

      ASSERT_EQ(exp, res);
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}

////////////////////
TEST(MaxPoolPatchFn_Test, ConstructorC)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      const int height = filter.window.shape.height;
      const int width  = filter.window.shape.width;
      const auto pixel_count = filter.window.shape.imagePixels();

      auto stream = std::stringstream();
      stream.write(reinterpret_cast<const char*>(&pixel_count), sizeof(pixel_count));

      MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( stream );

      auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
      for(int i = 0; i < exp.size(); i++) exp[i] = i;

      auto res = run_op(&params);

      ASSERT_EQ(exp, res);
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


////////////////////
TEST(MaxPoolPatchFn_Test, Serialize)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      const int height = filter.window.shape.height;
      const int width  = filter.window.shape.width;
      const auto pixel_count = filter.window.shape.imagePixels();

      MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( pixel_count );

      auto exp = std::vector<int8_t>( MaxPoolPatchFn::ChannelsPerOutputGroup );
      for(int i = 0; i < exp.size(); i++) exp[i] = i;

      auto stream = std::stringstream();
      params.Serialize( stream );

      // Check whether it works by deserializing it and running it.
      params = MaxPoolPatchFn::Params( stream );

      auto res = run_op(&params);

      ASSERT_EQ(exp, res);
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


////////////////////
TEST(MaxPoolPatchFn_Test, aggregate_fn)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      const int height = filter.window.shape.height;
      const int width  = filter.window.shape.width;
      const auto pixel_count = filter.window.shape.imagePixels();

      auto rand = nn::test::Rand( pixel_count * 34523 );

      MaxPoolPatchFn::Params params = MaxPoolPatchFn::Params( pixel_count );

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

      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}





class maxpool_patch_ref_test : public ::testing::TestWithParam<int> {};

TEST(MaxPoolPatchFn_Test, maxpool_patch_ref_test)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      const int height = filter.window.shape.height;
      const int width  = filter.window.shape.width;
      const auto pixel_count = filter.window.shape.imagePixels();

      auto rand = nn::test::Rand(4555 * pixel_count);

      auto patch = std::vector<int8_t>( pixel_count * VPU_INT8_EPV );

      int8_t exp[VPU_INT8_EPV];


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
        
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}
