

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "RefOps.hpp"
#include "Rand.hpp"
#include "ref_tests.hpp"
#include "gtest/gtest.h"

#include "Conv2d.hpp"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>
#include <cassert>

using namespace nn;
using namespace nn::test;

static auto rng = Rand(69);

class Conv2dDenseReferenceRegression : public ::testing::Test
{
};

//TODO binary tests for ImToColValid
TEST_F(Conv2dDenseReferenceRegression, BasicTest)
{

  // rng.setSeed(1598715445);
  for (int x_height = 1; x_height <= 8; ++x_height)
  {
    for (int x_width = 1; x_width <= 8; ++x_width)
    {
      for (int x_channels = 1; x_channels <= 16; x_channels += 1)
      {
        for (int k_height = 1; k_height <= x_height; ++k_height)
        {
          for (int k_width = 1; k_width <= x_width; ++k_width)
          {
            for (int k_depth = 1; k_depth <= x_channels; k_depth += 1)
            {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation)
              {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation)
                {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride)
                  {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride)
                    {
                      for (int top_pad = 0; top_pad <= 1; ++top_pad)
                      {
                        for (int left_pad = 0; left_pad <= 1; ++left_pad)
                        {
                          for (int right_pad = 0; right_pad <= 1; ++right_pad)
                          {
                            for (int bottom_pad = 0; bottom_pad <= 1; ++bottom_pad)
                            {
                              padding_t padding = {(int16_t)top_pad, (int16_t)left_pad, (int16_t)bottom_pad, (int16_t)right_pad};

                              int output_height = CONV2D_OUTPUT_LENGTH(x_height + padding.top + padding.bottom, k_height, k_v_dilation, k_v_stride);
                              int output_width = CONV2D_OUTPUT_LENGTH(x_width + padding.left + padding.right, k_width, k_h_dilation, k_h_stride);

                              if (output_height <= 0 || output_width <= 0)
                                continue;

                              int test_seed = rng.getSeed();

                              // std::cout << " x_height: " << x_height << " x_width: " << x_width << " x_channels: " << x_channels << " k_height: " << k_height << " k_width: " << k_width << " k_depth: " << k_depth << " k_h_dilation: " << k_h_dilation << " k_v_dilation: " << k_v_dilation << " k_h_stride: " << k_h_stride << " k_v_stride: " << k_v_stride << " top_pad: " << top_pad << " left_pad: " << left_pad << " bottom_pad: " << bottom_pad << " right_pad: " << right_pad << std::endl;

                              //TODO add this to the filter geom

                              //here output_height + width muct match the allocated memory for y
                              ImageGeometry Y(output_height, output_width, k_depth);

                              ImageGeometry X(x_height, x_width, x_channels);

                              WindowGeometry K(k_height, k_width, k_depth,
                                               -padding.top, -padding.left,
                                               k_v_stride, k_h_stride, 1,
                                               k_v_dilation, k_h_dilation);

                              Filter2dGeometry geom(X, Y, K);

                              auto weights = std::vector<int8_t>(geom.window.shape.height * geom.window.shape.width * geom.output.depth * geom.input.depth + 32, 0);
                              auto bias = std::vector<int32_t>(geom.output.depth);
                              auto eff_mult = std::vector<float>(geom.output.depth);
                              auto input = std::vector<int8_t>(geom.input.height * geom.input.width * geom.input.depth, 0);

                              for (int idx = 0; idx < weights.size(); ++idx)
                                weights[idx] = rng.rand<int8_t>();

                              for (int idx = 0; idx < input.size(); ++idx)
                                input[idx] = rng.rand<int8_t>();

                              int32_t input_zero = rng.rand<int8_t>() % 4;
                              int32_t output_zero = rng.rand<int8_t>() % 32;

                              for (int ch = 0; ch < geom.output.depth; ch++)
                              {
                                while (eff_mult[ch] < (1.0 / 256))
                                  eff_mult[ch] = rng.rand<float>();
                                bias[ch] = rng.rand<int8_t>() % 512;
                              }
                              float eff_mult_scalar = 2.0;
                              for (int ch = 0; ch < geom.output.depth; ch++)
                              {
                                eff_mult[ch] *= eff_mult_scalar;
                              }

                              auto expected = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                                                       input_zero, output_zero);

                              std::vector<int32_t> coefs_sums(k_depth, 0);
                              for (int ko = 0; ko < K.shape.depth; ko++)
                              {
                                for (int kh = 0; kh < K.shape.height; kh++)
                                {
                                  for (int kw = 0; kw < K.shape.width; kw++)
                                  {
                                    for (int ki = 0; ki < X.depth; ki++)
                                    {
                                      int idx = ko * (K.shape.height * K.shape.width * X.depth) + kh * (K.shape.width * X.depth) + kw * X.depth + ki;
                                      coefs_sums[ko] += (int32_t)weights[idx];
                                    }
                                  }
                                }
                              }

                              int8_t padding_val = input_zero;
                              int input_ch_per_output = x_channels; //TODO remove!!

                              ImToColPadded::Params im_to_col_params(X, K, padding, input_ch_per_output, padding_val);
                              ImToColPadded memcpy(&im_to_col_params);

                              int input_bytes = geom.getReceptiveVolumeBytes();

                              int scratch_bytes = MatMulInt8::get_scratch_size(input_bytes) + 32;

                              std::vector<int8_t> T(scratch_bytes, 0);

                              int8_t kernel_pad_val = rng.rand<int8_t>();

                              std::array<int, 4> shape = {k_depth, k_height, k_width, x_channels};
                              Conv2dReorderedWeights rw =
                                  MatMulInt8::reorder_kernel_weights((int8_t *)weights.data(), shape, 8, kernel_pad_val);

                              int accu_modifier[k_depth];

                              const int vpu_bytes = 32; //TODO replace

                              //TODO make this into an int8 specific function
                              for (int i = 0; i < k_depth; i++)
                              {
                                int idx = rw.final_vpu_load_addresses[i];

                                int s = 0;
                                int channel_overlap_start = input_bytes % vpu_bytes;

                                if (channel_overlap_start)
                                {
                                  for (int j = channel_overlap_start; j < vpu_bytes; j++)
                                  {
                                    s += (int)(rw.weights[idx + j]) * T[scratch_bytes - vpu_bytes + j];
                                  }
                                }
                                accu_modifier[i] = s;
                              }

                              MatMulInt8::Params p(k_depth, input_bytes, rw.weights.data());
                              MatMulInt8 aggregator(&p);

                              std::vector<double> f_biases(k_depth, 0);
                              std::vector<double> f_multipliers(k_depth, 0);
                              std::vector<int32_t> accu_min(k_depth, 0);
                              std::vector<int32_t> accu_max(k_depth, 0);

                              for (int ko = 0; ko < K.shape.depth; ko++)
                              {
                                int32_t max_accu_sum = 0;
                                int32_t min_accu_sum = 0;

                                for (int kh = 0; kh < K.shape.height; kh++)
                                {
                                  for (int kw = 0; kw < K.shape.width; kw++)
                                  {
                                    for (int ki = 0; ki < X.depth; ki++)
                                    {
                                      int idx = ko * (K.shape.height * K.shape.width * X.depth) + kh * (K.shape.width * X.depth) + kw * X.depth + ki;

                                      int32_t coef = (int32_t)weights[idx];

                                      if (coef > 0)
                                      {
                                        max_accu_sum += coef * (int32_t)INT8_MAX;
                                        min_accu_sum += coef * (int32_t)INT8_MIN;
                                      }
                                      else
                                      {
                                        max_accu_sum += coef * (int32_t)INT8_MIN;
                                        min_accu_sum += coef * (int32_t)INT8_MAX;
                                      }
                                    }
                                  }
                                }

                                f_biases[ko] = (bias[ko] - input_zero * coefs_sums[ko]) * eff_mult[ko] + output_zero;
                                f_multipliers[ko] = eff_mult[ko];

                                accu_min[ko] = min_accu_sum;
                                accu_max[ko] = max_accu_sum;
                              }

                              QuantisationParams qp = OutputTransformFnInt8::quantise_activation(f_multipliers, f_biases, accu_min, accu_max);

                              OT_int8::Params ot_params((int32_t)k_depth, &qp.otv, qp.biases.data(),
                                                        qp.multipliers.data(), (int16_t *)accu_modifier);

                              OT_int8 ot(&ot_params);
                              auto ir = ImageRegion(0, 0, 0,
                                                    Y.height,
                                                    Y.width, Y.depth);

                              AbstractKernel<Filter2D>::Params akp(Y, ir);

                              Conv2dPaddedInDirect conv2d(
                                  &akp,
                                  &memcpy,
                                  &aggregator,
                                  &ot, &T[0]);

                              auto output = std::vector<int8_t>(Y.height * Y.width * Y.depth);

                              conv2d.execute(&output[0], &input[0]);

                              for (int yh = 0; yh < Y.height; yh++)
                              {
                                for (int yw = 0; yw < Y.width; yw++)
                                {
                                  for (int yd = 0; yd < Y.depth; yd++)
                                  {
                                    int idx = yh * (Y.width * Y.depth) + yw * Y.depth + yd;
                                    // std::cout << "tflite: " << (int)expected[idx] << " xcore: " << (int)output[idx] << std::endl;
                                    EXPECT_NEAR((int)expected[idx], (int)output[idx], 1) << "tflite: " << (int)expected[idx] << " xcore: " << (int)output[idx] << " test seed: " << test_seed << " eff_mult[yd] : " << eff_mult[yd] << " x_height : " << x_height << " x_width : " << x_width << " x_channels : " << x_channels << " k_height : " << k_height << " k_width : " << k_width << " k_depth : " << k_depth << " k_h_dilation : " << k_h_dilation << " k_v_dilation : " << k_v_dilation << " k_h_stride : " << k_h_stride << " k_v_stride : " << k_v_stride << " top_pad : " << top_pad << " left_pad : " << left_pad << " right_pad : " << right_pad << " bottom_pad : " << bottom_pad << std::endl;
                                  }
                                }
                              }
                            }
                          }
                        }
                      }
                    }
                  }
                }
              }
            }
          }
        }
      }
    }
  }
}

#if 0
class Conv2dDenseReferenceRegression : public ::testing::TestWithParam<Filter2dGeometry> {};

TEST_P(Conv2dDenseReferenceRegression, NoPadding)
{
  auto geom = GetParam();

  int imageElements = geom.window.shape.imageElements();
  
  auto weights  = std::vector<int8_t>( imageElements* geom.output.depth );
  auto bias     = std::vector<int32_t>( geom.output.depth );
  auto eff_mult = std::vector<float>( geom.output.depth );
  auto input    = std::vector<int8_t>(imageElements);
  auto expected = std::vector<int8_t>(imageElements);

  int8_t input_zero  = -5;
  int8_t output_zero =  5;

  // for (int output_ch = 0; output_ch < geom.output.depth; ++output_ch){
  //   bias[output_ch] = 0;
  //   eff_mult[output_ch] = 0;
    
  //   for(int e=0;e<imageElements;++e){
  //     weights[output_ch*imageElements + e] = 0;
  //     input[output_ch*imageElements + e] = 0;
  //     expected[output_ch*imageElements + e] = 0;
  //   }
  // }


  memset( &weights[0], 1, sizeof(int8_t) * weights.size() );

  memset( &input[0],   1, sizeof(int8_t) * input.size()   );

  int32_t acc32 = geom.window.shape.imageElements() 
                  * ( int32_t( weights[0] ) * ( int32_t( input[0] ) - input_zero ) );

  int32_t target_acc32 = 32;
  int8_t target_acc8 = 16;

  for(int k = 0; k < geom.output.depth; k++){
    bias[k] = target_acc32 - acc32;
    eff_mult[k] = float(target_acc8) / float(target_acc32);
  }

  memset( &expected[0], int8_t(target_acc8 + output_zero), sizeof(int8_t) * expected.size() );

  auto output = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                         input_zero, output_zero);

                   
  ImageGeometry &X = geom.input;

  WindowGeometry &K = geom.window;
  int input_ch_per_output = 1;

  
  int8_t pad_val = 0x55;
  padding_t padding = {(int16_t)1, (int16_t)1, (int16_t)1, (int16_t)1}; 

  ImToColPadded::Params im2col_params(X, K, padding, input_ch_per_output, pad_val);
  ImToColPadded im2col(&im2col_params);

  // nn::Conv2dPaddedInDirect conv2d(&p, &i, &mm, &o);

  int output_channel_count = geom.output.depth;

  int input_bytes = X.pixelBytes();

  std::array<int, 4> shape = {output_channel_count, K.shape.height, K.shape.width, input_bytes};

  Conv2dReorderedWeights rw = 
    MatMulInt8::reorder_kernel_weights( (int8_t* )&weights[0], shape, 8, pad_val) ;
        
  MatMulInt8::Params p(output_channel_count, input_bytes, rw.weights.data());
  MatMulInt8 mm(&p);


  // conv2d.execute(&input[0] );


  // for (auto i : output){
  //   std::cout << (int)i << std::endl;
  // }
  ASSERT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(Basic, Conv2dDenseReferenceRegression, ::testing::Values( Filter2dGeometry(
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

#endif

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
class Conv2dDenseReferenceTestA : public ::testing::TestWithParam<Filter2dGeometry>
{
};

TEST_P(Conv2dDenseReferenceTestA, NoPadding)
{
  auto geom = GetParam();

  auto weights = std::vector<int8_t>(geom.window.shape.imageElements() * geom.output.depth);
  auto bias = std::vector<int32_t>(geom.output.depth);
  auto eff_mult = std::vector<float>(geom.output.depth);
  auto input = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  int8_t input_zero = -5;
  int8_t output_zero = 5;

  memset(&weights[0], 1, sizeof(int8_t) * weights.size());
  memset(&input[0], 1, sizeof(int8_t) * input.size());

  int32_t acc32 = geom.window.shape.imageElements() * (int32_t(weights[0]) * (int32_t(input[0]) - input_zero));

  int32_t target_acc32 = 32;
  int8_t target_acc8 = 16;

  for (int k = 0; k < geom.output.depth; k++)
  {
    bias[k] = target_acc32 - acc32;
    eff_mult[k] = float(target_acc8) / float(target_acc32);
  }

  memset(&expected[0], int8_t(target_acc8 + output_zero), sizeof(int8_t) * expected.size());

  auto output = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                         input_zero, output_zero);

  ASSERT_EQ(output, expected);
}

INSTANTIATE_TEST_SUITE_P(Basic, Conv2dDenseReferenceTestA, ::testing::Values(Filter2dGeometry(ImageGeometry(1, 1, 2), ImageGeometry(1, 1, 2), WindowGeometry(1, 1, 2)), Filter2dGeometry(ImageGeometry(2, 2, 2), ImageGeometry(2, 2, 2), WindowGeometry(1, 1, 2)), Filter2dGeometry(ImageGeometry(2, 2, 2), ImageGeometry(1, 1, 2), WindowGeometry(2, 2, 2))));

static auto iterA = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(false, false));
INSTANTIATE_TEST_SUITE_P(Random, Conv2dDenseReferenceTestA, ::testing::ValuesIn(iterA.begin(), iterA.end()));

/**
 * 
 * 
 * 
 */

class Conv2dDenseReferenceTestB : public ::testing::TestWithParam<Filter2dGeometry>
{
};

TEST_P(Conv2dDenseReferenceTestB, WithPadding)
{
  auto geom = GetParam();

  auto out_cov = geom.output.getAddressCovector<int8_t>();

  auto weights = std::vector<int8_t>(geom.window.shape.imageElements() * geom.output.depth);
  auto bias = std::vector<int32_t>(geom.output.depth);
  auto eff_mult = std::vector<float>(geom.output.depth);
  auto input = std::vector<int8_t>(geom.input.imageElements());
  auto expected = std::vector<int8_t>(geom.output.imageElements());

  int8_t input_zero = -5;
  int8_t output_zero = 5;

  memset(&weights[0], 1, sizeof(int8_t) * weights.size());
  memset(&input[0], 1, sizeof(int8_t) * input.size());

  int32_t target_acc32 = 32;
  int8_t target_acc8 = 16;

  auto pix_acc = geom.input.depth * int32_t(weights[0]) * (int32_t(input[0]) - input_zero);

  for (int k = 0; k < geom.output.depth; k++)
  {
    bias[k] = target_acc32 - geom.window.shape.imagePixels() * pix_acc;
    eff_mult[k] = float(target_acc8) / float(target_acc32);
  }

  auto init_pad = geom.ModelPadding();
  for (int row = 0; row < geom.output.height; row++)
  {
    for (int col = 0; col < geom.output.width; col++)
    {

      auto Y = out_cov.resolve(&expected[0], row, col, 0);

      // The behavior here depends on how many pixels are outside the input image.
      auto pad = init_pad;
      pad.top -= row * geom.window.stride.row;
      pad.left -= col * geom.window.stride.col;
      pad.bottom += row * geom.window.stride.row;
      pad.right += col * geom.window.stride.col;
      pad.MakeUnsigned();

      auto patch_rows = int(geom.window.shape.height) - pad.top - pad.bottom;
      auto patch_cols = int(geom.window.shape.width) - pad.left - pad.right;

      auto patch_pix = patch_rows * patch_cols;

      int32_t acc32 = patch_pix * pix_acc;

      for (int cout = 0; cout < geom.output.depth; cout++)
      {
        auto r = (((acc32 + bias[cout]) * eff_mult[cout]) + output_zero);
        Y[cout] = round_int8(r);
      }
    }
  }

  auto output = nn::test::ops::ref::Conv2dDenseReference(geom, &input[0], &weights[0], &bias[0], &eff_mult[0],
                                                         input_zero, output_zero);

  ASSERT_EQ(output, expected);
}

static auto iterB = nn::test::ParamedRandIter<Filter2dGeometry, SimpleFilter>(100, SimpleFilter(false, true), 763577);
INSTANTIATE_TEST_SUITE_P(Random, Conv2dDenseReferenceTestB, ::testing::ValuesIn(iterB.begin(), iterB.end()));