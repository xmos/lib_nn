
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

#include "FilterGeometryIterHelper.hpp"


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


static nn::ff::FilterGeometryIterator filter_sets[] = {
  test::unpadded::AllUnpadded( nn::Filter2dGeometry({0,0,36}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}}), true, 4)
};

////////////////////
TEST(AvgPoolDirectValidFn_Test, ConstructorA)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

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

      ASSERT_EQ(expected, res)
        << "Filter geometry: " << filter << std::endl
        << "iter: " << total_iter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}







////////////////////
TEST(AvgPoolDirectValidFn_Test, ConstructorB)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {


      ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      const auto pixel_count = filter.window.shape.imagePixels();

      AvgPoolDirectValidFn::Params params = AvgPoolDirectValidFn::Params( filter, 13 );

      const auto bleh = std::min<int>( filter.input.depth, AvgPoolDirectValidFn::ChannelsPerOutputGroup );

      vpu_ring_buffer_t expected = {{0}};
      for(int i = 0; i < bleh; i++) expected.set_acc(i, pixel_count * i * params.ap_params.scale[0]);

      auto res = run_op(&params, filter);

      ASSERT_EQ(expected, res)
        << "Filter geometry: " << filter << std::endl
        << "iter: " << total_iter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}








////////////////////
TEST(AvgPoolDirectValidFn_Test, Serialization)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {


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

      ASSERT_EQ(expected, res)
        << "Filter geometry: " << filter << std::endl
        << "iter: " << total_iter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}








////////////////////
TEST(AvgPoolDirectValidFn_Test, aggregate_fn)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {
      // auto filter = nn::Filter2dGeometry( {1,2,4,1}, {1,1,4,1}, {{1,1,1},{0,1},{1,1,1},{1,1}} ); {

      ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      const auto pixel_count = filter.window.shape.imagePixels();

      auto rand = nn::test::Rand( pixel_count * 87989 );

      auto input_img = std::vector<int8_t>( filter.input.imageElements() );
        
      int32_t scale = rand.rand<int8_t>();
      AvgPoolDirectValidFn::Params params = AvgPoolDirectValidFn::Params( filter, int8_t(scale) );

      auto in_start = filter.input.Index({filter.window.start.row, filter.window.start.col, 0});

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
      op.aggregate_fn( &acc, &input_img[in_start], 0);

      // If input channels < VPU_INT8_ACC_PERIOD, the last few accumulators will have junk in them
      for(int k = filter.input.depth; k < AvgPoolDirectValidFn::ChannelsPerOutputGroup; k++)
        acc.set_acc(k, 0);

      ASSERT_EQ(expected, acc) 
        << "Filter geometry: " << filter << std::endl
        << "iter: " << total_iter << std::endl
        << "scale: " << int(scale) << std::endl;

      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}








////////////////////
TEST(AvgPoolDirectValidFn_Test, avgpool_direct_valid_ref)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {
      // auto filter = nn::Filter2dGeometry( {1,2,4,1}, {1,1,4,1}, {{1,1,1},{0,1},{1,1,1},{1,1}} ); {

      ASSERT_TRUE( nn::AvgPool2d_Valid::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      auto rand = nn::test::Rand(8876 * filter.input.imageBytes() );

      auto input_img = std::vector<int8_t>( filter.input.imageElements() );
      auto in_start = filter.input.Index({filter.window.start.row, filter.window.start.col, 0});

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
      avgpool_direct_valid_ref(&acc, &input_img[in_start], &ap_params);

      // If input channels < VPU_INT8_ACC_PERIOD, the last few accumulators will have junk in them
      for(int k = filter.input.depth; k < AvgPoolDirectValidFn::ChannelsPerOutputGroup; k++)
        acc.set_acc(k, 0);
      
      ASSERT_EQ(exp_acc, acc) << "Failure details...\n"
        << "Filter geometry: " << filter << std::endl
        << "iter: " << total_iter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


