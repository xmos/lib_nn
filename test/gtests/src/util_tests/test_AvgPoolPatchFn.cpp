
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

#include "AvgPool2d.hpp"

using namespace nn;


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
  constexpr int X_d_MAX = 36;

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
TEST(AvgPoolPatchFn_Test, ConstructorA)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    const auto pixel_count = filter.window.shape.imagePixels();

    avgpool_patch_params ap_params;
    ap_params.pixels = pixel_count;
    std::memset(ap_params.scale, 10, sizeof(ap_params.scale));

    AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( ap_params );

    vpu_ring_buffer_t expected;
    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 10);

    auto res = run_op(&params, filter);

    ASSERT_EQ(expected, res) << "Filter geometry: " << filter;

  } while( NextParam(filter) );
}

////////////////////
TEST(AvgPoolPatchFn_Test, ConstructorB)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    const auto pixel_count = filter.window.shape.imagePixels();

    AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( filter.window, 11 );

    vpu_ring_buffer_t expected;
    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 11);

    auto res = run_op(&params, filter);

    ASSERT_EQ(expected, res) << "Filter geometry: " << filter;

  } while( NextParam(filter) );

  
}


////////////////////
TEST(AvgPoolPatchFn_Test, Serialization)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;


    const auto pixel_count = filter.window.shape.imagePixels();

    AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( filter.window, 11 );

    auto stream = std::stringstream();

    params.Serialize(stream);

    params = AvgPoolPatchFn::Params(stream);

    vpu_ring_buffer_t expected;
    for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 11);

    auto res = run_op(&params, filter);

    ASSERT_EQ(expected, res) << "Filter geometry: " << filter;

  } while( NextParam(filter) );

  
}


////////////////////
TEST(AvgPoolPatchFn_Test, aggregate_fn)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;


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

      ASSERT_EQ(expected, acc) 
        << "Filter geometry: " << filter
        << "iter: " << iter;
    }

  } while( NextParam(filter) );

  
}
