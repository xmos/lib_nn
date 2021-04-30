

#include "nn_types.h"
#include "geom/util.hpp"
#include "geom/Filter2dGeometry.hpp"
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

                              AbstractKernel::Params akp(Y, ir);

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
