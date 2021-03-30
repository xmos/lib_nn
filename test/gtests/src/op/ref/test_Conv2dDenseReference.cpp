

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "RefOps.hpp"
#include "Rand.hpp"
#include "ref_tests.hpp"

#include "gtest/gtest.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>


using namespace nn;
using namespace nn::test;

static auto rng = Rand(64563);

// static void GenerateParams(Filter2dGeometry& geom,
//                            int8_t kernel[],
//                            int32_t bias[],
//                            float eff_out_mult[],
//                            int8_t& input_zero_point,
//                            int8_t& output_zero_point)
// {

//   // kernel coefficients and input zero_point can just be whatever

//   input_zero_point = rng.rand<int8_t>();

//   for(int k = 0; k < geom.window.windowElements(); k++){
//     kernel[k] = rng.rand<int8_t>();
//   }

//   auto win_cov = AddressCovector<int8_t>(geom.window.shape.width * geom.window.shape.depth, 
//                                          geom.window.shape.depth, 1);

//   int32_t min_out_zero = std::numeric_limits<int32_t>::max();
//   int32_t max_out_zero = std::numeric_limits<int32_t>::min();

//   for(int cout = 0; cout << geom.output.depth; cout++) {

//     int64_t filt_min = 0;
//     int64_t filt_max = 0;

//     for(int row = 0; row < geom.window.shape.height; row++){
//       for(int col = 0; col < geom.window.shape.width; col++){
//         for(int xan = 0; xan < geom.input.depth; xan++){

//           int32_t w = win_cov.resolve(kernel, row, col, xan)[0]; 

//           int32_t min_in = std::numeric_limits<int8_t>::min();
//           int32_t max_in = std::numeric_limits<int8_t>::max();

//           if(w < 0){
//             auto tmp = min_in;
//             min_in = max_in;
//             max_in = tmp;
//           }

//           min_in -= input_zero_point;
//           max_in -= input_zero_point;

//           filt_min += w * min_in;
//           filt_max += w * max_in;
//         }
//       }
//     }

//     // The filt_min and filt_max values imply our bounds of allowable biases.
//     int64_t bias_min = std::numeric_limits<int32_t>::min() - filt_min;
//     int64_t bias_max = std::numeric_limits<int32_t>::max() - filt_max;

//     assert(bias_min >= int64_t( std::numeric_limits<int32_t>::min() ) );
//     assert(bias_max <= int64_t( std::numeric_limits<int32_t>::max() ) );

//     bias[cout] = rng.rand<int32_t>(int32_t(bias_min), int32_t(bias_max));

//     filt_min += bias[cout];
//     filt_max += bias[cout];

//     int32_t filt_span = filt_max - filt_min;

//     eff_out_mult[cout] = ldexpf(1, 8) / filt_span;

//     int32_t out_zero = std::numeric_limits<int8_t>::min() - (filt_min * eff_out_mult[cout]);

//     min_out_zero = std::min<int32_t>(min_out_zero, out_zero);
//     max_out_zero = std::max<int32_t>(max_out_zero, out_zero);

//   }

//   assert(min_out_zero >= std::numeric_limits<int8_t>::min());
//   assert(max_out_zero >= std::numeric_limits<int8_t>::max());

//   output_zero_point = rng.rand<int8_t>(min_out_zero, max_out_zero);

// }


/**
 * 
 * 
 * 
 */
class Conv2dDenseReferenceTestA : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(Conv2dDenseReferenceTestA, NoPadding)
{
  auto geom = GetParam();

  auto weights  = std::vector<int8_t>( geom.window.windowElements() * geom.output.depth );
  auto bias     = std::vector<int32_t>( geom.output.depth );
  auto eff_mult = std::vector<float>( geom.output.depth );
  auto input    = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  int8_t input_zero  = -5;
  int8_t output_zero =  5;

  memset( &weights[0], 1, sizeof(int8_t) * weights.size() );
  memset( &input[0],   1, sizeof(int8_t) * input.size()   );

  int32_t acc32 = geom.window.windowElements() * ( int32_t( weights[0] ) * ( int32_t( input[0] ) - input_zero ) );

  int32_t target_acc32 = 32;
  int8_t target_acc8 = 16;

  for(int k = 0; k < geom.output.depth; k++){
    bias[k] = target_acc32 - acc32;
    eff_mult[k] = float(target_acc8) / float(target_acc32);
  }

  memset( &expected[0], int8_t(target_acc8 + output_zero), sizeof(int8_t) * expected.size() );

  auto output = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                         input_zero, output_zero);

  ASSERT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(Basic, Conv2dDenseReferenceTestA, ::testing::Values( Filter2dGeometry(
                                                                            ImageGeometry(1,1,2),
                                                                            ImageGeometry(1,1,2),
                                                                            WindowGeometry(1,1,2)),
                                                                         Filter2dGeometry(
                                                                            ImageGeometry(2,2,2),
                                                                            ImageGeometry(2,2,2),
                                                                            WindowGeometry(1,1,2)),
                                                                         Filter2dGeometry(
                                                                            ImageGeometry(2,2,2),
                                                                            ImageGeometry(1,1,2),
                                                                            WindowGeometry(2,2,2)) 
));



static auto iterA = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(false, false));
INSTANTIATE_TEST_SUITE_P(Random, Conv2dDenseReferenceTestA, ::testing::ValuesIn(iterA.begin(), iterA.end()));



/**
 * 
 * 
 * 
 */

class Conv2dDenseReferenceTestB : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(Conv2dDenseReferenceTestB, WithPadding)
{
  auto geom = GetParam();

  auto out_cov = geom.output.getAddressCovector<int8_t>();

  auto weights  = std::vector<int8_t>( geom.window.windowElements() * geom.output.depth );
  auto bias     = std::vector<int32_t>( geom.output.depth );
  auto eff_mult = std::vector<float>( geom.output.depth );
  auto input    = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  int8_t input_zero  = -5;
  int8_t output_zero =  5;

  memset( &weights[0], 1, sizeof(int8_t) * weights.size() );
  memset( &input[0],   1, sizeof(int8_t) * input.size()   );
  
  int32_t target_acc32 = 32;
  int8_t target_acc8   = 16;

  auto pix_acc = geom.input.depth * int32_t(weights[0]) * (int32_t(input[0]) - input_zero);

  for(int k = 0; k < geom.output.depth; k++){
    bias[k] = target_acc32 - geom.window.windowPixels() * pix_acc;
    eff_mult[k] = float(target_acc8) / float(target_acc32);
  }

  auto init_pad = geom.ModelPadding();
  for(int row = 0; row < geom.output.height; row++){
    for(int col = 0; col < geom.output.width; col++){

      auto Y = out_cov.resolve(&expected[0], row, col, 0);

      // The behavior here depends on how many pixels are outside the input image.
      auto pad = init_pad;
      pad.top    -= row * geom.window.stride.row;
      pad.left   -= col * geom.window.stride.col;
      pad.bottom += row * geom.window.stride.row;
      pad.right  += col * geom.window.stride.col;
      pad.MakeUnsigned();

      auto patch_rows = int(geom.window.shape.height) - pad.top  - pad.bottom;
      auto patch_cols = int(geom.window.shape.width ) - pad.left - pad.right;

      auto patch_pix = patch_rows * patch_cols;

      int32_t acc32 = patch_pix * pix_acc;
      
      for(int cout = 0; cout < geom.output.depth; cout++){
        auto r = (((acc32 + bias[cout]) * eff_mult[cout]) + output_zero);
        Y[cout] = round_int8( r );
      }
    }
  }



  auto output = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                         input_zero, output_zero);

  ASSERT_EQ(output, expected);

}

static auto iterB = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(false, true), 763577);
INSTANTIATE_TEST_SUITE_P(Random, Conv2dDenseReferenceTestB, ::testing::ValuesIn(iterB.begin(), iterB.end()));