
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

#include "../src/cpp/filt2d/geom/WindowLocation.hpp"




class AvgPoolDirectValidFnTest : public ::testing::TestWithParam<nn::Filter2dGeometry> {};


static vpu_ring_buffer_t run_op(AvgPoolDirectValidFn::Params* params,
                                  const nn::Filter2dGeometry& filter)
{
  auto input = std::vector<int8_t>( filter.input.imageElements() );
  
  for(int r = 0; r < filter.input.height; r++)
    for(int c = 0; c < filter.input.width; c++)
      for(int x = 0; x < filter.input.depth; x++)
        filter.input.Element<int8_t>(&input[0], r, c, x) = x;
      

  auto ap = AvgPoolDirectValidFn(params);
  vpu_ring_buffer_t acc = {{0}};
  ap.aggregate_fn( &acc, &input[0], 0 );

    for(int k = filter.input.depth; k < AvgPoolDirectValidFn::ChannelsPerOutputGroup; k++)
      acc.set_acc(k, 0);

  return acc;
}


////////////////////
TEST_P(AvgPoolDirectValidFnTest, ConstructorA)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  avgpool_direct_valid_params ap_params;
  ap_params.rows = filter.window.shape.height;
  ap_params.cols = filter.window.shape.width;
  ap_params.col_stride = filter.window.dilation.col * filter.input.pixelBytes();
  ap_params.row_stride = (filter.window.dilation.row * filter.input.rowBytes())
                         - (filter.window.shape.width * ap_params.col_stride);
  std::memset(ap_params.scale, 12, sizeof(ap_params.scale));

  AvgPoolDirectValidFn::Params params = AvgPoolDirectValidFn::Params(ap_params );

  const auto bleh = std::min<int>( filter.input.depth, AvgPoolDirectValidFn::ChannelsPerOutputGroup );

  vpu_ring_buffer_t expected = {{0}};
  for(int i = 0; i < bleh; i++) expected.set_acc(i, pixel_count * i * params.ap_params.scale[0]);

  auto res = run_op(&params, filter);

  ASSERT_EQ(expected, res);
  
}

////////////////////
TEST_P(AvgPoolDirectValidFnTest, ConstructorB)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  AvgPoolDirectValidFn::Params params = AvgPoolDirectValidFn::Params( filter, 13 );

  const auto bleh = std::min<int>( filter.input.depth, AvgPoolDirectValidFn::ChannelsPerOutputGroup );

  vpu_ring_buffer_t expected = {{0}};
  for(int i = 0; i < bleh; i++) expected.set_acc(i, pixel_count * i * params.ap_params.scale[0]);

  auto res = run_op(&params, filter);

  ASSERT_EQ(expected, res);
}

////////////////////
TEST_P(AvgPoolDirectValidFnTest, Serialization)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  AvgPoolDirectValidFn::Params params = AvgPoolDirectValidFn::Params( filter, 13 );

  // Serialize and Deserialize the params
  auto stream = std::stringstream();
  params.Serialize(stream);
  params = AvgPoolDirectValidFn::Params(stream);

  const auto bleh = std::min<int>( filter.input.depth, AvgPoolDirectValidFn::ChannelsPerOutputGroup );

  vpu_ring_buffer_t expected = {{0}};
  for(int i = 0; i < bleh; i++) expected.set_acc(i, pixel_count * i * params.ap_params.scale[0]);

  auto res = run_op(&params, filter);

  ASSERT_EQ(expected, res);
}



////////////////////
TEST_P(AvgPoolDirectValidFnTest, aggregate_fn)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  auto rand = nn::test::Rand( pixel_count * 87989 );

  auto input_img = std::vector<int8_t>( filter.input.imageElements() );

  for(int iter = 0; iter < 30; iter++){
    
    int32_t scale = rand.rand<int8_t>();
    AvgPoolDirectValidFn::Params params = AvgPoolDirectValidFn::Params( filter, int8_t(scale) );

    vpu_ring_buffer_t expected = {{0}};

    for(int i = 0; i < input_img.size(); i++)
      input_img[i] = rand.rand<int8_t>();

    // Using this lambda in a fold operation on the window locations
    auto get_expected = [=](const nn::ImageVect& filter_coords, 
                            const nn::ImageVect& input_coords,
                            const int32_t acc,
                            const int8_t value, 
                            const bool is_padding) -> int32_t {
      return acc + (int32_t(value) * scale);
    };

    for(int chan = 0; chan < filter.output.depth && chan < AvgPoolDirectValidFn::ChannelsPerOutputGroup; chan++){
      nn::WindowLocation loc = filter.GetWindow( nn::ImageVect(0, 0, chan) );
      expected.set_acc(chan, loc.Fold<int32_t, int8_t>(&input_img[0], get_expected, 0));
    }

    vpu_ring_buffer_t acc;
    auto op = AvgPoolDirectValidFn(&params);
    op.aggregate_fn( &acc, &input_img[0], 0);

    // If input channels < VPU_INT8_ACC_PERIOD, the last few accumulators will have junk in them
    for(int k = filter.input.depth; k < AvgPoolDirectValidFn::ChannelsPerOutputGroup; k++)
      acc.set_acc(k, 0);

    ASSERT_EQ(expected, acc) << "iter = " << iter;
  }
}

////////////////////
TEST_P(AvgPoolDirectValidFnTest, avgpool_direct_valid_ref)
{
  const auto filter = GetParam();

  auto rand = nn::test::Rand(8876 * filter.input.imageBytes() );

  auto input_img = std::vector<int8_t>( filter.input.imageElements() );

  for(int iter = 0; iter < 30; iter++){

    const auto scale = rand.rand<int8_t>(1, INT8_MAX);

    auto ap_params = AvgPoolDirectValidFn::Params(filter, scale).ap_params;

    vpu_ring_buffer_t exp_acc = {{0}};

    for(int i = 0; i < input_img.size(); i++)
      input_img[i] = rand.rand<int8_t>();

    // Using this lambda in a fold operation on the window locations
    auto get_expected = [&](const nn::ImageVect& filter_coords, 
                            const nn::ImageVect& input_coords,
                            const int32_t acc,
                            const int8_t value, 
                            const bool is_padding) -> int32_t {
      return acc + (int32_t(value) * scale);
    };

    for(int chan = 0; chan < filter.output.depth && chan < AvgPoolDirectValidFn::ChannelsPerOutputGroup; chan++){
      nn::WindowLocation loc = filter.GetWindow( nn::ImageVect(0, 0, chan) );
      exp_acc.set_acc(chan, loc.Fold<int32_t, int8_t>(&input_img[0], get_expected, 0));
    }

    vpu_ring_buffer_t acc = {{0}};
    avgpool_direct_valid_ref(&acc, &input_img[0], &ap_params);

    // If input channels < VPU_INT8_ACC_PERIOD, the last few accumulators will have junk in them
    for(int k = filter.input.depth; k < AvgPoolDirectValidFn::ChannelsPerOutputGroup; k++)
      acc.set_acc(k, 0);
    
    ASSERT_EQ(exp_acc, acc) << "Failure details...\n"
      << "\t| iter = " << iter;
  }
}






static std::vector<nn::Filter2dGeometry> GenerateParams()
{
  auto vec = std::vector<nn::Filter2dGeometry>();

  for(int X_h = 1; X_h <= 16; X_h += 4){
    for(int X_w = 1; X_w <= 16; X_w += 4){
      for(int X_d = 4; X_d <= 36; X_d += 4){
        auto input_img = nn::ImageGeometry(X_h, X_w, X_d);
        auto output_img = nn::ImageGeometry(1, 1, X_d);

        for(int K_h = 1; K_h <= X_h && K_h <= 3; K_h++){
          for(int K_w = 1; K_w <= X_w && K_w <= 3; K_w++){
            for(int dil_row = 1; (dil_row <= 3) && (((K_h-1)*dil_row) < X_h); dil_row++){
              for(int dil_col = 1; (dil_col <= 3) && (((K_w-1)*dil_col) < X_w); dil_col++){
                auto window = nn::WindowGeometry( K_h, K_w, 1,    0, 0,   1, 1, 1,   dil_row, dil_col);
                vec.push_back( nn::Filter2dGeometry(input_img, output_img, window));
  } } } } } } }

  return vec;
}

static auto filter_iter = ::testing::ValuesIn( GenerateParams() );

INSTANTIATE_TEST_SUITE_P(, AvgPoolDirectValidFnTest, filter_iter);
