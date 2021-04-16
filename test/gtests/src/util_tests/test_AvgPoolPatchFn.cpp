
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

using namespace nn;

class AvgPoolPatchFnTest : public ::testing::TestWithParam<nn::Filter2dGeometry> {};


static vpu_ring_buffer_t run_op(AvgPoolPatchFn::Params* params,
                                  const nn::Filter2dGeometry& filter)
{
  auto input = std::vector<int8_t>( params->ap_params.pixels * AvgPoolPatchFn::ChannelsPerOutputGroup );
  
  for(int c = 0; c < AvgPoolPatchFn::ChannelsPerOutputGroup; c++)
    for(int p = 0; p < params->ap_params.pixels; p++)
      input[p * AvgPoolPatchFn::ChannelsPerOutputGroup + c] = c;
  
  auto ap = AvgPoolPatchFn(params);
  vpu_ring_buffer_t acc;
  ap.aggregate_fn( &acc, &input[0], 0 );

  return acc;
}


////////////////////
TEST_P(AvgPoolPatchFnTest, ConstructorA)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  avgpool_patch_params ap_params;
  ap_params.pixels = pixel_count;
  std::memset(ap_params.scale, 10, sizeof(ap_params.scale));

  AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( ap_params );

  vpu_ring_buffer_t expected;
  for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 10);

  auto res = run_op(&params, filter);

  ASSERT_EQ(expected, res);
  
}

////////////////////
TEST_P(AvgPoolPatchFnTest, ConstructorB)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( filter.window, 11 );

  vpu_ring_buffer_t expected;
  for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 11);

  auto res = run_op(&params, filter);

  ASSERT_EQ(expected, res);
}

////////////////////
TEST_P(AvgPoolPatchFnTest, Serialization)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( filter.window, 11 );

  auto stream = std::stringstream();

  params.Serialize(stream);

  params = AvgPoolPatchFn::Params(stream);

  vpu_ring_buffer_t expected;
  for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 11);

  auto res = run_op(&params, filter);

  ASSERT_EQ(expected, res);
}



////////////////////
TEST_P(AvgPoolPatchFnTest, aggregate_fn)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  auto rand = nn::test::Rand( pixel_count * 87989 );

  for(int iter = 0; iter < 30; iter++){
    
    int32_t scale = rand.rand<int8_t>();
    AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( filter.window, int8_t(scale) );

    auto patch = std::vector<int8_t>( pixel_count * AvgPoolPatchFn::ChannelsPerOutputGroup );

    vpu_ring_buffer_t expected;
    std::memset(&expected, 0, sizeof(expected));

    {
      int k = 0;
      for(int pix = 0; pix < pixel_count; pix++){
        for(int chan = 0; chan < AvgPoolPatchFn::ChannelsPerOutputGroup; chan++){
          patch[k] = rand.rand<int8_t>();
          expected.add_acc(chan, patch[k] * scale);
          k++;
        }
      }
    }

    vpu_ring_buffer_t acc;
    auto op = AvgPoolPatchFn(&params);
    op.aggregate_fn( &acc, &patch[0], 0);

    ASSERT_EQ(expected, acc) << "iter = " << iter;
  }
}

////////////////////
TEST_P(AvgPoolPatchFnTest, avgpool_patch_ref)
{
  const auto filter = GetParam();
  const auto pixel_count = filter.window.shape.imagePixels();

  auto rand = nn::test::Rand(8876 * pixel_count);

  auto patch = std::vector<int8_t>( pixel_count * VPU_INT8_ACC_PERIOD );

  for(int iter = 0; iter < 30; iter++){

    const auto scale = rand.rand<int8_t>(1, INT8_MAX);

    avgpool_patch_params ap_params;
    ap_params.pixels = pixel_count;
    std::memset(ap_params.scale, scale, sizeof(ap_params.scale));

    vpu_ring_buffer_t exp_acc;
    std::memset(&exp_acc, 0, sizeof(exp_acc));

    {
      int k = 0;
      for(int pix = 0; pix < pixel_count; pix++){
        for(int chan = 0; chan < VPU_INT8_ACC_PERIOD; chan++){
          patch[k] = rand.rand<int8_t>();
          int32_t p = int32_t(patch[k]) * int32_t(ap_params.scale[chan]);
          exp_acc.add_acc(chan, p);
          k++;
        }
      }
    }

    vpu_ring_buffer_t acc;
    avgpool_patch_ref(&acc, &patch[0], &ap_params);
      
    ASSERT_EQ(exp_acc, acc) << "iter = " << iter;
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

        for(int K_h = 1; K_h <= X_h && K_h <= 4; K_h++){
          for(int K_w = 1; K_w <= X_w && K_w <= 4; K_w++){
            for(int dil_row = 1; (dil_row <= 3) && (((K_h-1)*dil_row) < X_h); dil_row++){
              for(int dil_col = 1; (dil_col <= 3) && (((K_w-1)*dil_col) < X_w); dil_col++){
                auto window = nn::WindowGeometry( K_h, K_w, 1,    0, 0,   1, 1, 1,   dil_row, dil_col);
                vec.push_back( nn::Filter2dGeometry(input_img, output_img, window));
  } } } } } } }

  return vec;
}

static auto filter_iter = ::testing::ValuesIn( GenerateParams() );

INSTANTIATE_TEST_SUITE_P(, AvgPoolPatchFnTest, filter_iter);
