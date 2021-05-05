
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
#include "MaxPool2d.hpp"
#include "geom/WindowLocation.hpp"

#include "FilterGeometryIterHelper.hpp"


using namespace nn;


static std::vector<int8_t> run_op(MaxPoolDirectValidFn::Params* params,
                                  const nn::Filter2dGeometry& filter)
{
  auto input = std::vector<int8_t>( filter.input.imageElements() );
  auto mp = MaxPoolDirectValidFn(params);
  vpu_ring_buffer_t acc;

  const auto output_chans = std::min<int>( MaxPoolDirectValidFn::ChannelsPerOutputGroup, 
                                           filter.output.depth );

  {
    int k = 0;
    for(int row = 0; row < filter.input.height; row++)
      for(int col = 0; col < filter.input.width; col++)
        for(int chan = 0; chan < filter.input.depth; chan++)
          input[k++] = chan;
  }
  
  mp.aggregate_fn( &acc, &input[0], 0 );

  auto res = std::vector<int8_t>( output_chans );
  std::memcpy( &res[0], acc.vR, sizeof(int8_t) * output_chans );

  return res;
}


static nn::ff::FilterGeometryIterator filter_sets[] = {
  test::unpadded::AllUnpadded( nn::Filter2dGeometry({0,0,36}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}}), true, 4)
};


////////////////////
TEST(MaxPoolDirectValidFn_Test, ConstructorA)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      auto rand = nn::test::Rand(filter.input.imageBytes() * 6632);

      maxpool_direct_valid_params mp_params;
      mp_params.rows = filter.window.shape.height;
      mp_params.cols = filter.window.shape.width;
      mp_params.col_stride = filter.window.dilation.col * filter.input.pixelBytes();
      mp_params.row_stride = (filter.window.dilation.row * filter.input.rowBytes())
                            - (filter.window.shape.width * mp_params.col_stride);

      auto params = MaxPoolDirectValidFn::Params(mp_params);

      auto res = run_op( &params, filter );

      for(int i = 0; i < res.size(); i++)
        ASSERT_EQ(i, res[i]);

      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


////////////////////
TEST(MaxPoolDirectValidFn_Test, ConstructorB)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      auto rand = nn::test::Rand(filter.input.imageBytes() * 6632);

      auto params = MaxPoolDirectValidFn::Params(filter.input, filter.window);

      auto res = run_op( &params, filter );

      for(int i = 0; i < res.size(); i++)
        ASSERT_EQ(i, res[i]);

      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


////////////////////
TEST(MaxPoolDirectValidFn_Test, Serialization)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      auto rand = nn::test::Rand(filter.input.imageBytes() * 6632);

      MaxPoolDirectValidFn::Params params = MaxPoolDirectValidFn::Params(filter.input, filter.window);

      auto stream = std::stringstream();

      params.Serialize(stream);

      params = MaxPoolDirectValidFn::Params( stream );

      auto res = run_op( &params, filter );

      for(int i = 0; i < res.size(); i++)
        ASSERT_EQ(i, res[i]);

      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}





////////////////////
TEST(MaxPoolDirectValidFn_Test, aggregate_fn)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      const auto window_pixels = filter.window.shape.imagePixels();

      auto rand = nn::test::Rand(43242 * window_pixels );

      auto params = MaxPoolDirectValidFn::Params( filter.input, filter.window );
      auto op = MaxPoolDirectValidFn( &params );

      auto input_img  = std::vector<int8_t>( filter.input.imageElements() );
      auto in_start = filter.input.Index({filter.window.start.row, filter.window.start.col, 0});
      auto exp = std::vector<int8_t>( MaxPoolDirectValidFn::ChannelsPerOutputGroup );

      rand.rand_bytes( &input_img[0], sizeof(int8_t) * input_img.size() );

      std::memset(&exp[0], 0x80, sizeof(int8_t) * exp.size());

      const auto out_channels = std::min<int>(filter.output.depth, VPU_INT8_EPV);

      for(int out_chan = 0; out_chan < out_channels; out_chan++){
        auto loc = filter.GetWindow( 0, 0, out_chan );

        for(int row = 0; row < filter.window.shape.height; row++){
          for(int col = 0; col < filter.window.shape.width; col++){
            for(int xan = 0; xan < filter.window.shape.depth; xan++){
              exp[out_chan] = std::max<int8_t>( loc.GetInput<int8_t>(&input_img[0], row, col, xan, 0), 
                                                                    exp[out_chan] );
            }
          }
        }
      }

      vpu_ring_buffer_t acc;
      int8_t* out = reinterpret_cast<int8_t*>(&acc);

      maxpool_direct_valid_ref(&acc, &input_img[in_start], &params.mp_params);
        
      for(int i = 0; i < out_channels; i++){
        ASSERT_EQ(exp[i], out[i])
          << "Failure Details..\n"
          << "  i = " << i << "\n";
      }
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}



////////////////////
TEST(MaxPoolDirectValidFn_Test, maxpool_direct_valid_ref_test)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      const auto window_pixels = filter.window.shape.imagePixels();

      auto rand = nn::test::Rand(4322 * window_pixels );

      auto params = MaxPoolDirectValidFn::Params( filter.input, filter.window );

      int8_t exp[VPU_INT8_EPV];

      auto input_img = std::vector<int8_t>( filter.input.imageElements() );
      auto in_start = filter.input.Index({filter.window.start.row, filter.window.start.col, 0});

      rand.rand_bytes( &input_img[0], sizeof(int8_t) * input_img.size() );

      std::memset(exp, 0x80, sizeof(exp));

      const auto out_channels = std::min<int>(filter.output.depth, VPU_INT8_EPV);

      for(int out_chan = 0; out_chan < out_channels; out_chan++){
        auto loc = filter.GetWindow( 0, 0, out_chan );

        for(int row = 0; row < filter.window.shape.height; row++){
          for(int col = 0; col < filter.window.shape.width; col++){
            for(int xan = 0; xan < filter.window.shape.depth; xan++){
              exp[out_chan] = std::max<int8_t>( loc.GetInput<int8_t>(&input_img[0], row, col, xan, 0), 
                                                                    exp[out_chan] );
            }
          }
        }
      }

      vpu_ring_buffer_t acc;
      int8_t* out = reinterpret_cast<int8_t*>(&acc);

      maxpool_direct_valid_ref(&acc, &input_img[in_start], &params.mp_params);
        
      for(int i = 0; i < out_channels; i++){
        ASSERT_EQ(exp[i], out[i])
          << "Failure Details..\n"
          << "  i = " << i << "\n";
      }
      
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


