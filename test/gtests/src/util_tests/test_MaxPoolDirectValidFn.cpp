
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



TEST_P(MaxPoolDirectValidFnParamsTest, ConstructorA)
{
  const auto filter = GetParam();

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

}

TEST_P(MaxPoolDirectValidFnParamsTest, ConstructorB)
{
  const auto filter = GetParam();

  auto rand = nn::test::Rand(filter.input.imageBytes() * 6632);

  auto params = MaxPoolDirectValidFn::Params(filter.input, filter.window);

  auto res = run_op( &params, filter );

  for(int i = 0; i < res.size(); i++)
    ASSERT_EQ(i, res[i]);

}

TEST_P(MaxPoolDirectValidFnParamsTest, Serialization)
{
  const auto filter = GetParam();

  auto rand = nn::test::Rand(filter.input.imageBytes() * 6632);

  MaxPoolDirectValidFn::Params params = MaxPoolDirectValidFn::Params(filter.input, filter.window);

  auto stream = std::stringstream();

  params.Serialize(stream);

  params = MaxPoolDirectValidFn::Params( stream );

  auto res = run_op( &params, filter );

  for(int i = 0; i < res.size(); i++)
    ASSERT_EQ(i, res[i]);

}





TEST_P(MaxPoolDirectValidFnTest, aggregate_fn)
{
  const auto filter = GetParam();

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
}



TEST_P(MaxPoolDirectValidFnTest, maxpool_direct_valid_ref_test)
{
  const auto filter = GetParam();

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
}





static std::vector<nn::Filter2dGeometry> GenerateParams()
{
  auto vec = std::vector<nn::Filter2dGeometry>();

  for(int X_h = 1; X_h <= 16; X_h += 4){
    for(int X_w = 1; X_w <= 16; X_w += 4){
      for(int X_d = 4; X_d <= 64; X_d += 4){
        auto input_img = nn::ImageGeometry(X_h, X_w, X_d);
        auto output_img = nn::ImageGeometry(1, 1, X_d);

        for(int K_h = 1; K_h <= X_h && K_h <= 4; K_h++){
          for(int K_w = 1; K_w <= X_w && K_w <= 4; K_w++){
            for(int dil_row = 1; (dil_row <= 3) && (((K_h-1)*dil_row) < X_h); dil_row++){
              for(int dil_col = 1; (dil_col <= 3) && (((K_w-1)*dil_col) < X_w); dil_col++){
                auto window = nn::WindowGeometry( K_h, K_w, 1,    0, 0,   1, 1, 1,   dil_row, dil_col);
                vec.push_back( nn::Filter2dGeometry(input_img, output_img, window));
              }
            }

          }
        }

      }
    }
  }

  return vec;
}

static auto filter_iter = ::testing::ValuesIn( GenerateParams() );

INSTANTIATE_TEST_SUITE_P(, MaxPoolDirectValidFnParamsTest, filter_iter);
INSTANTIATE_TEST_SUITE_P(, MaxPoolDirectValidFnTest, filter_iter);