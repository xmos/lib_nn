
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
#include "AvgPool2d.hpp"


using namespace nn;

/**
 * Run the aggregate_fn() and return the result
 */
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
TEST(AvgPoolDirectValidFn_Test, ConstructorA)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

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

    ASSERT_EQ(expected, res) << "Filter geometry: " << filter;

  } while( NextParam(filter) );
}






////////////////////
TEST(AvgPoolDirectValidFn_Test, ConstructorB)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    const auto pixel_count = filter.window.shape.imagePixels();

    AvgPoolDirectValidFn::Params params = AvgPoolDirectValidFn::Params( filter, 13 );

    const auto bleh = std::min<int>( filter.input.depth, AvgPoolDirectValidFn::ChannelsPerOutputGroup );

    vpu_ring_buffer_t expected = {{0}};
    for(int i = 0; i < bleh; i++) expected.set_acc(i, pixel_count * i * params.ap_params.scale[0]);

    auto res = run_op(&params, filter);

    ASSERT_EQ(expected, res) << "Filter geometry: " << filter;

  } while( NextParam(filter) );
}







////////////////////
TEST(AvgPoolDirectValidFn_Test, Serialization)
{
  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

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

    ASSERT_EQ(expected, res) << "Filter geometry: " << filter;

  } while( NextParam(filter) );
}







////////////////////
TEST(AvgPoolDirectValidFn_Test, aggregate_fn)
{
  constexpr int ITER_COUNT = 10;

  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

    const auto pixel_count = filter.window.shape.imagePixels();

    auto rand = nn::test::Rand( pixel_count * 87989 );

    auto input_img = std::vector<int8_t>( filter.input.imageElements() );

    for(int iter = 0; iter < ITER_COUNT; iter++){
      
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

      ASSERT_EQ(expected, acc) 
        << "Filter geometry: " << filter
        << "iter: " << iter;
    }

  } while( NextParam(filter) );
}







////////////////////
TEST(AvgPoolDirectValidFn_Test, avgpool_direct_valid_ref)
{
  constexpr int ITER_COUNT = 10;

  nn::Filter2dGeometry filter(nn::ImageGeometry(1, 1, 4),
                              nn::ImageGeometry(1, 1, 4),
                              nn::WindowGeometry(1, 1, 1,   0, 0,   1, 1, 1,   1, 1)); 
  do {

    ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;




    auto rand = nn::test::Rand(8876 * filter.input.imageBytes() );

    auto input_img = std::vector<int8_t>( filter.input.imageElements() );

    for(int iter = 0; iter < ITER_COUNT; iter++){

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
        << "\t| Filter geometry: " << filter
        << "\t| iter: " << iter;
    }

  } while( NextParam(filter) );
}

