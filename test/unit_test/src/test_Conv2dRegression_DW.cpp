#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <valarray>
#include <vector>

#include "Conv2d.hpp"
#include "Rand.hpp"
#include "RefOps.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "geom/util.hpp"
#include "nn_types.h"

extern "C" {
#include "tst_common.h"
#include "unity.h"
}

using namespace nn;
using namespace nn::test;

static auto rng = Rand(69);

const int max_k_channels = 32;

struct DWKernelStimulus {
  std::vector<int8_t> weights;
  std::vector<int32_t> bias;
  std::vector<float> eff_mult;
  std::vector<int8_t> input;
  int input_zero_point;
  int output_zero_point;

  DWKernelStimulus(Filter2dGeometry &geom)
      : weights(geom.window.shape.height * geom.window.shape.width *
                    geom.output.depth,
                0),
        bias(geom.output.depth),
        eff_mult(geom.output.depth),
        input(geom.input.height * geom.input.width * geom.input.depth + 16 +
                  (16 - (geom.input.depth %
                         16)),  // these extra bytes are to account for input
                                // over read by VLDC
              0),
        input_zero_point(0),
        output_zero_point(0){};
};

DWKernelStimulus create_simple_stimulus_dw(Filter2dGeometry &geom) {
  DWKernelStimulus ks(geom);

  for (int idx = 0; idx < ks.weights.size(); ++idx)
    ks.weights[idx] = rng.rand<int8_t>();

  for (int idx = 0; idx < ks.input.size(); ++idx)
    ks.input[idx] = rng.rand<int8_t>();

  ks.input_zero_point = rng.rand<int8_t>() % 8;
  ks.output_zero_point = rng.rand<int8_t>() % 8;

  for (int ch = 0; ch < geom.output.depth; ch++) {
    while (ks.eff_mult[ch] < (1.0 / 2)) ks.eff_mult[ch] = rng.rand<float>();
    int n = (1 << 14);
    ks.bias[ch] = (rng.rand<int32_t>() % n) - (rng.rand<int32_t>() % n);
  }
  float eff_mult_scalar = 1.0 / 256;
  for (int ch = 0; ch < geom.output.depth; ch++) {
    ks.eff_mult[ch] *= eff_mult_scalar;
  }
  return ks;
}

void test_Conv2dValidDirectDWRegression() {
  /*
  This is an array of counters to record that all outputs are getting hit
  */
  int output_count[256] = {0};

  for (int x_height = 1; x_height <= 3; ++x_height) {
    for (int x_width = 1; x_width <= 3; ++x_width) {
      for (int x_channels = 4; x_channels <= 32 + 4; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride) {
                    for (int top_pad = 0; top_pad <= 0; ++top_pad) {
                      for (int left_pad = 0; left_pad <= 0; ++left_pad) {
                        for (int right_pad = 0; right_pad <= 0; ++right_pad) {
                          for (int bottom_pad = 0; bottom_pad <= 0;
                               ++bottom_pad) {
                            padding_t padding = {
                                (int16_t)top_pad, (int16_t)left_pad,
                                (int16_t)bottom_pad, (int16_t)right_pad};

                            int output_height = CONV2D_OUTPUT_LENGTH(
                                x_height + padding.top + padding.bottom,
                                k_height, k_v_dilation, k_v_stride);
                            int output_width = CONV2D_OUTPUT_LENGTH(
                                x_width + padding.left + padding.right, k_width,
                                k_h_dilation, k_h_stride);

                            if (output_height <= 0 || output_width <= 0)
                              continue;

                            int test_seed = rng.getSeed();

                            // std::cout << " x_height:" << x_height
                            //           << " x_width:" << x_width
                            //           << " x_channels:" << x_channels
                            //           << " k_height:" << k_height
                            //           << " k_width:" << k_width
                            //           << " k_h_dilation:" << k_h_dilation
                            //           << " k_v_dilation:" << k_v_dilation
                            //           << " k_h_stride:" << k_h_stride
                            //           << " k_v_stride:" << k_v_stride
                            //           << " top_pad:" << top_pad
                            //           << " left_pad:" << left_pad
                            //           << " right_pad:" << right_pad
                            //           << " bottom_pad:" << bottom_pad
                            //           << std::endl;

                            // here output_height + width muct match the
                            // allocated memory for y
                            ImageGeometry Y(output_height, output_width,
                                            x_channels);

                            ImageGeometry X(x_height, x_width, x_channels);

                            WindowGeometry K(k_height, k_width, x_channels,
                                             -padding.top, -padding.left,
                                             k_v_stride, k_h_stride, 1,
                                             k_v_dilation, k_h_dilation);

                            Filter2dGeometry geom(X, Y, K);

                            DWKernelStimulus ks =
                                create_simple_stimulus_dw(geom);

                            auto &weights = ks.weights;
                            auto &bias = ks.bias;
                            auto &eff_mult = ks.eff_mult;
                            auto &input = ks.input;

                            auto expected =
                                nn::test::ops::ref::Conv2dDepthwiseReference(
                                    geom, input.data(), weights.data(),
                                    bias.data(), eff_mult.data(),
                                    ks.input_zero_point, ks.output_zero_point);

                            DerefInputFn::Params im_to_col_params(X, K);
                            DerefInputFn memcpy(&im_to_col_params);

                            int8_t kernel_pad_val = rng.rand<int8_t>();

                            std::array<int, 4> shape = {
                                {1, k_height, k_width, x_channels}};

                            Conv2dReorderedWeights rw =
                                MatMulDirectFn_DW::reorder_kernel_weights(
                                    (int8_t *)weights.data(), shape,
                                    kernel_pad_val);

                            MatMulDirectFn_DW::Params p(X, K);
                            MatMulDirectFn_DW aggregator(&p);
                            aggregator.setWeights(rw.weights.data());

                            MulsAndBias mul_and_biases = OutputTransformFnInt8::
                                canonicalise_mul_and_bias_dw(
                                    eff_mult, bias, weights, shape,
                                    ks.input_zero_point, ks.output_zero_point,
                                    x_channels);

                            QuantisationParams qp =
                                OutputTransformFnInt8::quantise_activation(
                                    mul_and_biases);

                            auto serialised_multipliers_and_biases =
                                OutputTransformFn::serialise_memory(
                                    qp.multipliers, qp.biases);

                            // pad q.biases and  q.multipliers to a multiple
                            // of VPU_INT16_EPV this is to work around array
                            // over reads
                            int16_t pad_val =
                                rng.rand<int16_t>();  // this is arbitrary
                            OutputTransformFn::pad_final_access(
                                serialised_multipliers_and_biases,
                                VPU_INT16_EPV, pad_val);

                            OT_int8::Params ot_params((int32_t)x_channels,
                                                      qp.initial_shr,
                                                      qp.final_shr);

                            OT_int8 ot(&ot_params);
                            ot.setMultipliersAndBiases(
                                serialised_multipliers_and_biases.data());

                            auto ir = ImageRegion(0, 0, 0, Y.height, Y.width,
                                                  Y.depth);

                            Filter2D_DW::Params akp(Y, ir, VPU_INT8_ACC_PERIOD);

                            Conv2dDepthwiseValidDirect conv2d(&memcpy,
                                                              &aggregator, &ot);

                            auto output = std::vector<int8_t>(
                                Y.height * Y.width * Y.depth);

                            nn::execute(&output[0], &input[0], &conv2d, &akp);

                            for (int yh = 0; yh < Y.height; yh++) {
                              for (int yw = 0; yw < Y.width; yw++) {
                                for (int yd = 0; yd < Y.depth; yd++) {
                                  int idx = yh * (Y.width * Y.depth) +
                                            yw * Y.depth + yd;

                                  TEST_ASSERT_INT32_WITHIN(
                                      1, (int)expected[idx], (int)output[idx]);
                                  output_count[(int)expected[idx] - INT8_MIN]++;
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
  std::valarray<int> oc_(output_count, 256);
  float dynamic_range = (float)oc_.max() / (float)oc_.min();
  TEST_ASSERT_FLOAT_WITHIN(128.0, 0, dynamic_range);
}

void test_Conv2dPaddedIndirectDWRegression() {
  /*
  This is an array of counters to record that all outputs are getting hit
  */
  int output_count[256] = {0};

  for (int x_height = 1; x_height <= 3; ++x_height) {
    for (int x_width = 1; x_width <= 3; ++x_width) {
      for (int x_channels = 4; x_channels <= 32 + 4; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_h_dilation = 1; k_h_dilation <= 3; ++k_h_dilation) {
              for (int k_v_dilation = 1; k_v_dilation <= 3; ++k_v_dilation) {
                for (int k_h_stride = 1; k_h_stride <= 3; ++k_h_stride) {
                  for (int k_v_stride = 1; k_v_stride <= 3; ++k_v_stride) {
                    for (int top_pad = 0; top_pad <= 1; ++top_pad) {
                      for (int left_pad = 0; left_pad <= 1; ++left_pad) {
                        for (int right_pad = 0; right_pad <= 1; ++right_pad) {
                          for (int bottom_pad = 0; bottom_pad <= 1;
                               ++bottom_pad) {
                            padding_t padding = {
                                (int16_t)top_pad, (int16_t)left_pad,
                                (int16_t)bottom_pad, (int16_t)right_pad};

                            int output_height = CONV2D_OUTPUT_LENGTH(
                                x_height + padding.top + padding.bottom,
                                k_height, k_v_dilation, k_v_stride);
                            int output_width = CONV2D_OUTPUT_LENGTH(
                                x_width + padding.left + padding.right, k_width,
                                k_h_dilation, k_h_stride);

                            if (output_height <= 0 || output_width <= 0)
                              continue;

                            int test_seed = rng.getSeed();

                            // std::cout << " x_height:" << x_height
                            //           << " x_width:" << x_width
                            //           << " x_channels:" << x_channels
                            //           << " k_height:" << k_height
                            //           << " k_width:" << k_width
                            //           << " k_h_dilation:" << k_h_dilation
                            //           << " k_v_dilation:" << k_v_dilation
                            //           << " k_h_stride:" << k_h_stride
                            //           << " k_v_stride:" << k_v_stride
                            //           << " top_pad:" << top_pad
                            //           << " left_pad:" << left_pad
                            //           << " right_pad:" << right_pad
                            //           << " bottom_pad:" << bottom_pad
                            //           << std::endl;

                            // here output_height + width muct match the
                            // allocated memory for y
                            ImageGeometry Y(output_height, output_width,
                                            x_channels);

                            ImageGeometry X(x_height, x_width, x_channels);

                            WindowGeometry K(k_height, k_width, x_channels,
                                             -padding.top, -padding.left,
                                             k_v_stride, k_h_stride, 1,
                                             k_v_dilation, k_h_dilation);

                            Filter2dGeometry geom(X, Y, K);

                            DWKernelStimulus ks =
                                create_simple_stimulus_dw(geom);

                            auto &weights = ks.weights;
                            auto &bias = ks.bias;
                            auto &eff_mult = ks.eff_mult;
                            auto &input = ks.input;

                            auto expected =
                                nn::test::ops::ref::Conv2dDepthwiseReference(
                                    geom, input.data(), weights.data(),
                                    bias.data(), eff_mult.data(),
                                    ks.input_zero_point, ks.output_zero_point);

                            ImToColPadded::Params im_to_col_params(
                                X, K, padding, 16, ks.input_zero_point);

                            ImToColPadded memcpy(&im_to_col_params);

                            std::array<int, 4> weights_shape = {
                                {1, k_height, k_width, x_channels}};

                            int scratch_bytes =
                                MatMulDirectFn_DW::get_scratch_mem_bytes(
                                    weights_shape);

                            std::vector<int8_t> T(scratch_bytes, 0);

                            // Result should be uneffected by kernel pad value
                            int8_t kernel_pad_val = rng.rand<int8_t>();

                            Conv2dReorderedWeights rw =
                                MatMulDirectFn_DW::reorder_kernel_weights(
                                    (int8_t *)weights.data(), weights_shape,
                                    kernel_pad_val);

                            MatMulDirectFn_DW::Params p(K);
                            MatMulDirectFn_DW aggregator(&p);
                            aggregator.setWeights(rw.weights.data());

                            MulsAndBias mul_and_biases = OutputTransformFnInt8::
                                canonicalise_mul_and_bias_dw(
                                    eff_mult, bias, weights, weights_shape,
                                    ks.input_zero_point, ks.output_zero_point,
                                    x_channels);

                            QuantisationParams qp =
                                OutputTransformFnInt8::quantise_activation(
                                    mul_and_biases);

                            auto serialised_multipliers_and_biases =
                                OutputTransformFn::serialise_memory(
                                    qp.multipliers, qp.biases);

                            // pad q.biases and  q.multipliers to a multiple
                            // of VPU_INT16_EPV this is to work around array
                            // over reads
                            int16_t pad_val =
                                rng.rand<int16_t>();  // this is arbitrary
                            OutputTransformFn::pad_final_access(
                                serialised_multipliers_and_biases,
                                VPU_INT16_EPV, pad_val);
                            OT_int8::Params ot_params((int32_t)x_channels,
                                                      qp.initial_shr,
                                                      qp.final_shr);

                            OT_int8 ot(&ot_params);
                            ot.setMultipliersAndBiases(
                                serialised_multipliers_and_biases.data());

                            auto ir = ImageRegion(0, 0, 0, Y.height, Y.width,
                                                  Y.depth);

                            Filter2D_DW::Params akp(Y, ir, VPU_INT8_ACC_PERIOD);

                            Conv2dDepthwisePaddedIndirect conv2d(
                                &memcpy, &aggregator, &ot);

                            auto output = std::vector<int8_t>(
                                Y.height * Y.width * Y.depth);

                            nn::execute(&output[0], &input[0], &conv2d, &akp, &T[0]);

                            for (int yh = 0; yh < Y.height; yh++) {
                              for (int yw = 0; yw < Y.width; yw++) {
                                for (int yd = 0; yd < Y.depth; yd++) {
                                  int idx = yh * (Y.width * Y.depth) +
                                            yw * Y.depth + yd;
                                  int delta =
                                      (int)expected[idx] - (int)output[idx];
                                  if (delta < 0) delta = -delta;

                                  TEST_ASSERT_INT32_WITHIN(
                                      1, (int)expected[idx], (int)output[idx]);

                                  output_count[(int)expected[idx] - INT8_MIN]++;
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

  std::valarray<int> oc_(output_count, 256);
  float dynamic_range = (float)oc_.max() / (float)oc_.min();
  TEST_ASSERT_FLOAT_WITHIN(128.0, 0, dynamic_range);
}

extern "C" void test_conv2d_dw_regression();
void test_conv2d_dw_regression() {
  UNITY_SET_FILE();
  RUN_TEST(test_Conv2dValidDirectDWRegression);
  RUN_TEST(test_Conv2dPaddedIndirectDWRegression);
}
