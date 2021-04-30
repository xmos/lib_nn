
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


using namespace nn;

class MaxPoolDirectValidFnParamsTest : public ::testing::TestWithParam<nn::Filter2dGeometry> {};
class MaxPoolDirectValidFnTest : public ::testing::TestWithParam<nn::Filter2dGeometry> {};


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

/**
 * Update the filter to the next param
 */
static bool UpdateParam(nn::Filter2dGeometry& filter)
{
  auto& input = filter.input;
  auto& output = filter.output;
  auto& window = filter.window;

  constexpr int X_h_MAX = 8;
  constexpr int X_w_MAX = 8;
  constexpr int X_d_MAX = 66;

  constexpr int K_h_MAX = 4;
  constexpr int K_w_MAX = 4;

  constexpr int K_dil_h_MAX = 3;
  constexpr int K_dil_w_MAX = 3;

#define UPDATE(FIELD, INIT, INCR, CONDITION)    do {                  \
                                                  FIELD += (INCR);    \
                                                  if( (CONDITION) )   \
                                                    return true;      \
                                                  FIELD = (INIT);     \
                                                } while(0)

  UPDATE(window.dilation.col, 1, 1, 
    ( (window.dilation.col <= K_dil_w_MAX) && (((window.shape.width-1)*window.dilation.col) < input.width) ) );
  UPDATE(window.dilation.row, 1, 1,
    ( (window.dilation.row <= K_dil_h_MAX) && (((window.shape.height-1)*window.dilation.row) < input.height) ) );
  UPDATE(window.shape.width, 1, 1, window.shape.width <= input.width && window.shape.width <= K_w_MAX);
  UPDATE(window.shape.height, 1, 1, window.shape.height <= input.height && window.shape.height <= K_h_MAX);
  UPDATE(input.depth, 4, 4, input.depth <= X_d_MAX);
  UPDATE(input.width, 1, 1, input.width <= X_w_MAX);
  UPDATE(input.height, 1, 1, input.height <= X_h_MAX);

  return false;

#undef UPDATE
}

/**
 * Generates the sequence of test parameters
 */
static bool NextParam(nn::Filter2dGeometry& filter)
{
  if(!UpdateParam(filter)) return false;

  // Fix it up to make sure it stays valid.
  filter.output.depth = filter.input.depth;

  return true;
}


////////////////////
TEST(MaxPoolDirectValidFn_Test, ConstructorA)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

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

  } while( NextParam(filter) );
}


////////////////////
TEST(MaxPoolDirectValidFn_Test, ConstructorB)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {
    ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    auto rand = nn::test::Rand(filter.input.imageBytes() * 6632);

    auto params = MaxPoolDirectValidFn::Params(filter.input, filter.window);

    auto res = run_op( &params, filter );

    for(int i = 0; i < res.size(); i++)
      ASSERT_EQ(i, res[i]);

  } while( NextParam(filter) );
}


////////////////////
TEST(MaxPoolDirectValidFn_Test, Serialization)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {
    ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    auto rand = nn::test::Rand(filter.input.imageBytes() * 6632);

    MaxPoolDirectValidFn::Params params = MaxPoolDirectValidFn::Params(filter.input, filter.window);

    auto stream = std::stringstream();

    params.Serialize(stream);

    params = MaxPoolDirectValidFn::Params( stream );

    auto res = run_op( &params, filter );

    for(int i = 0; i < res.size(); i++)
      ASSERT_EQ(i, res[i]);

  } while( NextParam(filter) );
}





////////////////////
TEST(MaxPoolDirectValidFn_Test, aggregate_fn)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {
    ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    const auto window_pixels = filter.window.shape.imagePixels();

    auto rand = nn::test::Rand(43242 * window_pixels );

    auto params = MaxPoolDirectValidFn::Params( filter.input, filter.window );
    auto op = MaxPoolDirectValidFn( &params );

    auto input_img  = std::vector<int8_t>( filter.input.imageElements() );
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

    maxpool_direct_valid_ref(&acc, &input_img[0], &params.mp_params);
      
    for(int i = 0; i < out_channels; i++){
      ASSERT_EQ(exp[i], out[i])
        << "Failure Details..\n"
        << "  i = " << i << "\n";
    }


  } while( NextParam(filter) );
}



////////////////////
TEST(MaxPoolDirectValidFn_Test, maxpool_direct_valid_ref_test)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {
    ASSERT_TRUE( nn::MaxPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    const auto window_pixels = filter.window.shape.imagePixels();

    auto rand = nn::test::Rand(4322 * window_pixels );

    auto params = MaxPoolDirectValidFn::Params( filter.input, filter.window );

    int8_t exp[VPU_INT8_EPV];

    auto input_img = std::vector<int8_t>( filter.input.imageElements() );

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

    maxpool_direct_valid_ref(&acc, &input_img[0], &params.mp_params);
      
    for(int i = 0; i < out_channels; i++){
      ASSERT_EQ(exp[i], out[i])
        << "Failure Details..\n"
        << "  i = " << i << "\n";
    }

  } while( NextParam(filter) );
}

TEST_P(MaxPoolDirectValidFnTest, maxpool_direct_valid_ref_test)
{
  const auto filter = GetParam();
}


