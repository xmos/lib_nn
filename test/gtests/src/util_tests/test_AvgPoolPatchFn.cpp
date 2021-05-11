
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

#include "FilterGeometryIterHelper.hpp"

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



static nn::ff::FilterGeometryIterator filter_sets[] = {
  test::unpadded::AllUnpadded( nn::Filter2dGeometry({0,0,36}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}}), true, 4),
  test::padded::AllPadded( nn::Filter2dGeometry({0,0,0}, {3,3,36}, {{4,4,1}, {2,2}, {2,3}, {2,3}} ), {3, 3, 3, 3}, true, 4),
};


////////////////////
TEST(AvgPoolPatchFn_Test, ConstructorA)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      const auto pixel_count = filter.window.shape.PixelCount();

      avgpool_patch_params ap_params;
      ap_params.pixels = pixel_count;
      std::memset(ap_params.scale, 10, sizeof(ap_params.scale));

      AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( ap_params );

      vpu_ring_buffer_t expected;
      for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 10);

      auto res = run_op(&params, filter);

      ASSERT_EQ(expected, res) << "Filter geometry: " << filter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}

////////////////////
TEST(AvgPoolPatchFn_Test, ConstructorB)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;

      const auto pixel_count = filter.window.shape.PixelCount();

      AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( filter.window, 11 );

      vpu_ring_buffer_t expected;
      for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 11);

      auto res = run_op(&params, filter);

      ASSERT_EQ(expected, res) << "Filter geometry: " << filter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


////////////////////
TEST(AvgPoolPatchFn_Test, Serialization)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;


      const auto pixel_count = filter.window.shape.PixelCount();

      AvgPoolPatchFn::Params params = AvgPoolPatchFn::Params( filter.window, 11 );

      auto stream = std::stringstream();

      params.Serialize(stream);

      params = AvgPoolPatchFn::Params(stream);

      vpu_ring_buffer_t expected;
      for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++) expected.set_acc(i, pixel_count * i * 11);

      auto res = run_op(&params, filter);

      ASSERT_EQ(expected, res) << "Filter geometry: " << filter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}


////////////////////
TEST(AvgPoolPatchFn_Test, aggregate_fn)
{
  int total_iter = 0;

  for(auto filter_set : filter_sets) {
    filter_set.Reset();
    for(auto filter : filter_set) {

      ASSERT_TRUE( nn::AvgPool2d_Generic::SupportsGeometry( filter ) ) << "Filter geometry not supported: " << filter;


      const auto pixel_count = filter.window.shape.PixelCount();

      auto rand = nn::test::Rand( pixel_count * 87989 );
        
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
        << "iter: " << total_iter;
      total_iter++;
    }
  }

  std::cout << "Count: " << total_iter << std::endl;
}
