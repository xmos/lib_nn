

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "Conv2d.hpp"
#include "Rand.hpp"
#include "RefOps.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "geom/util.hpp"
#include "gtest/gtest.h"
#include "nn_types.h"
#include "ref_tests.hpp"

using namespace nn;
using namespace nn::test;

static auto rng = Rand(69);

class Conv2dPaddedIndirectRegression : public ::testing::Test {};

struct KernelStimulus {
  std::vector<int8_t> weights;
  std::vector<int32_t> bias;
  std::vector<float> eff_mult;
  std::vector<int8_t> input;
  int input_zero_point;
  int output_zero_point;

  KernelStimulus(Filter2dGeometry &geom)
      : weights(geom.window.shape.height * geom.window.shape.width *
                    geom.output.depth * geom.input.depth,
                0),
        bias(geom.output.depth),
        eff_mult(geom.output.depth),
        input(geom.input.height * geom.input.width * geom.input.depth, 0),
        input_zero_point(0),
        output_zero_point(0){};
};

KernelStimulus create_simple_stimulus(Filter2dGeometry &geom) {
  KernelStimulus ks(geom);

  for (int idx = 0; idx < ks.weights.size(); ++idx)
    ks.weights[idx] = rng.rand<int8_t>();

  for (int idx = 0; idx < ks.input.size(); ++idx)
    ks.input[idx] = rng.rand<int8_t>();

  ks.input_zero_point = rng.rand<int8_t>() % 4;
  ks.output_zero_point = rng.rand<int8_t>() % 32;

  for (int ch = 0; ch < geom.output.depth; ch++) {
    while (ks.eff_mult[ch] < (1.0 / 256)) ks.eff_mult[ch] = rng.rand<float>();
    ks.bias[ch] = rng.rand<int8_t>() % 512;
  }
  float eff_mult_scalar = 2.0;
  for (int ch = 0; ch < geom.output.depth; ch++) {
    ks.eff_mult[ch] *= eff_mult_scalar;
  }
  return ks;
}

TEST_F(Conv2dPaddedIndirectRegression, BasicTest) {
  for (int x_height = 1; x_height <= 2; ++x_height) {
    for (int x_width = 1; x_width <= 2; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 4; k_depth <= x_channels; k_depth += 4) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
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
                                  x_width + padding.left + padding.right,
                                  k_width, k_h_dilation, k_h_stride);

                              if (output_height <= 0 || output_width <= 0)
                                continue;

                              int test_seed = rng.getSeed();

                              // here output_height + width muct match the
                              // allocated memory for y
                              ImageGeometry Y(output_height, output_width,
                                              k_depth);

                              ImageGeometry X(x_height, x_width, x_channels);

                              WindowGeometry K(k_height, k_width, k_depth,
                                               -padding.top, -padding.left,
                                               k_v_stride, k_h_stride, 1,
                                               k_v_dilation, k_h_dilation);

                              Filter2dGeometry geom(X, Y, K);

                              KernelStimulus ks = create_simple_stimulus(geom);
                              auto &weights = ks.weights;
                              auto &bias = ks.bias;
                              auto &eff_mult = ks.eff_mult;
                              auto &input = ks.input;

                              auto expected =
                                  nn::test::ops::ref::Conv2dDenseReference(
                                      geom, input.data(), weights.data(),
                                      bias.data(), eff_mult.data(),
                                      ks.input_zero_point,
                                      ks.output_zero_point);

                              ImToColPadded::Params im_to_col_params(
                                  X, K, padding, x_channels,
                                  ks.input_zero_point);
                              ImToColPadded memcpy(&im_to_col_params);

                              int input_bytes = geom.window.shape.height *
                                                geom.window.shape.width *
                                                geom.input.depth;
                              int scratch_bytes =
                                  MatMulInt8::get_scratch_mem_bytes(
                                      input_bytes) +
                                  32;
                              std::vector<int8_t> T(scratch_bytes, 0);

                              int8_t kernel_pad_val =
                                  rng.rand<int8_t>();  // This can be anything,
                                                       // 0 in pratice.
                              std::array<int, 4> shape = {k_depth, k_height,
                                                          k_width, x_channels};
                              Conv2dReorderedWeights rw =
                                  MatMulInt8::reorder_kernel_weights(
                                      (int8_t *)weights.data(), shape, 8,
                                      kernel_pad_val);

                              MatMulInt8::Params p(k_depth, input_bytes,
                                                   rw.weights.data());
                              MatMulInt8 aggregator(&p);

                              OutputTransformFnInt8::CanonicalMulAndBias
                                  canonical_values = OutputTransformFnInt8::
                                      canonicalise_mul_and_bias(
                                          eff_mult, bias, weights,
                                          ks.input_zero_point,
                                          ks.output_zero_point, k_depth);

                              QuantisationParams qp =
                                  OutputTransformFnInt8::quantise_activation(
                                      canonical_values.f_multipliers,
                                      canonical_values.f_biases,
                                      canonical_values.accu_min,
                                      canonical_values.accu_max);
                              OT_int8::Params ot_params(
                                  (int32_t)k_depth, &qp.otv, qp.biases.data(),
                                  qp.multipliers.data());

                              OT_int8 ot(&ot_params);
                              auto ir = ImageRegion(0, 0, 0, Y.height, Y.width,
                                                    Y.depth);

                              Filter2D::Params akp(Y, ir, VPU_INT8_ACC_PERIOD);

                              Conv2dPaddedInDirect conv2d(
                                  &akp, &memcpy, &aggregator, &ot, &T[0]);

                              // auto output = std::vector<int8_t>(
                              //     Y.height * Y.width * Y.depth);
                              alignas(4)
                                  int8_t output[Y.height * Y.width * Y.depth];

                              conv2d.execute(&output[0], &input[0]);

                              for (int yh = 0; yh < Y.height; yh++) {
                                for (int yw = 0; yw < Y.width; yw++) {
                                  for (int yd = 0; yd < Y.depth; yd++) {
                                    int idx = yh * (Y.width * Y.depth) +
                                              yw * Y.depth + yd;
                                    // std::cout << "tflite: " <<
                                    // (int)expected[idx] << " xcore: " <<
                                    // (int)output[idx] << std::endl;
                                    EXPECT_NEAR((int)expected[idx],
                                                (int)output[idx], 1)
                                        << "tflite: " << (int)expected[idx]
                                        << " xcore: " << (int)output[idx]
                                        << " test seed: " << test_seed
                                        << " eff_mult[yd] : " << eff_mult[yd]
                                        << " x_height : " << x_height
                                        << " x_width : " << x_width
                                        << " x_channels : " << x_channels
                                        << " k_height : " << k_height
                                        << " k_width : " << k_width
                                        << " k_depth : " << k_depth
                                        << " k_h_dilation : " << k_h_dilation
                                        << " k_v_dilation : " << k_v_dilation
                                        << " k_h_stride : " << k_h_stride
                                        << " k_v_stride : " << k_v_stride
                                        << " top_pad : " << top_pad
                                        << " left_pad : " << left_pad
                                        << " right_pad : " << right_pad
                                        << " bottom_pad : " << bottom_pad
                                        << std::endl;
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

class Conv2dValidIndirectRegression : public ::testing::Test {};

TEST_F(Conv2dValidIndirectRegression, BasicTest) {
  for (int x_height = 1; x_height <= 5; ++x_height) {
    for (int x_width = 1; x_width <= 5; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 4; k_depth <= x_channels; k_depth += 4) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
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
                                  x_width + padding.left + padding.right,
                                  k_width, k_h_dilation, k_h_stride);

                              if (output_height <= 0 || output_width <= 0)
                                continue;

                              int test_seed = rng.getSeed();

                              // here output_height + width muct match the
                              // allocated memory for y
                              ImageGeometry Y(output_height, output_width,
                                              k_depth);

                              ImageGeometry X(x_height, x_width, x_channels);

                              WindowGeometry K(k_height, k_width, k_depth,
                                               -padding.top, -padding.left,
                                               k_v_stride, k_h_stride, 1,
                                               k_v_dilation, k_h_dilation);

                              Filter2dGeometry geom(X, Y, K);

                              KernelStimulus ks = create_simple_stimulus(geom);
                              auto &weights = ks.weights;
                              auto &bias = ks.bias;
                              auto &eff_mult = ks.eff_mult;
                              auto &input = ks.input;

                              auto expected =
                                  nn::test::ops::ref::Conv2dDenseReference(
                                      geom, input.data(), weights.data(),
                                      bias.data(), eff_mult.data(),
                                      ks.input_zero_point,
                                      ks.output_zero_point);

                              ImToColValid::Params im_to_col_params(X, K,
                                                                    x_channels);
                              ImToColValid memcpy(&im_to_col_params);

                              int overread_bytes = memcpy.get_overread_bytes();
                              input.resize(input.size() +
                                               overread_bytes / sizeof(int8_t),
                                           0);

                              int input_bytes = geom.window.shape.height *
                                                geom.window.shape.width *
                                                geom.input.depth;

                              int scratch_bytes =
                                  MatMulInt8::get_scratch_mem_bytes(
                                      input_bytes) +
                                  32;

                              std::vector<int8_t> T(scratch_bytes, 0);

                              int8_t kernel_pad_val = rng.rand<int8_t>();

                              std::array<int, 4> shape = {k_depth, k_height,
                                                          k_width, x_channels};
                              Conv2dReorderedWeights rw =
                                  MatMulInt8::reorder_kernel_weights(
                                      (int8_t *)weights.data(), shape, 8,
                                      kernel_pad_val);

                              MatMulInt8::Params p(k_depth, input_bytes,
                                                   rw.weights.data());
                              MatMulInt8 aggregator(&p);

                              OutputTransformFnInt8::CanonicalMulAndBias
                                  canonical_values = OutputTransformFnInt8::
                                      canonicalise_mul_and_bias(
                                          eff_mult, bias, weights,
                                          ks.input_zero_point,
                                          ks.output_zero_point, k_depth);

                              QuantisationParams qp =
                                  OutputTransformFnInt8::quantise_activation(
                                      canonical_values.f_multipliers,
                                      canonical_values.f_biases,
                                      canonical_values.accu_min,
                                      canonical_values.accu_max);
                              OT_int8::Params ot_params(
                                  (int32_t)k_depth, &qp.otv, qp.biases.data(),
                                  qp.multipliers.data());

                              OT_int8 ot(&ot_params);
                              auto ir = ImageRegion(0, 0, 0, Y.height, Y.width,
                                                    Y.depth);

                              Filter2D::Params akp(Y, ir, VPU_INT8_ACC_PERIOD);

                              Conv2dValidIndirect conv2d(
                                  &akp, &memcpy, &aggregator, &ot, &T[0]);

                              auto output = std::vector<int8_t>(
                                  Y.height * Y.width * Y.depth);

                              conv2d.execute(&output[0], &input[0]);

                              for (int yh = 0; yh < Y.height; yh++) {
                                for (int yw = 0; yw < Y.width; yw++) {
                                  for (int yd = 0; yd < Y.depth; yd++) {
                                    int idx = yh * (Y.width * Y.depth) +
                                              yw * Y.depth + yd;
                                    // std::cout << "tflite: " <<
                                    // (int)expected[idx] << " xcore: " <<
                                    // (int)output[idx] << std::endl;
                                    EXPECT_NEAR((int)expected[idx],
                                                (int)output[idx], 1)
                                        << "tflite: " << (int)expected[idx]
                                        << " xcore: " << (int)output[idx]
                                        << " test seed: " << test_seed
                                        << " eff_mult[yd] : " << eff_mult[yd]
                                        << " x_height : " << x_height
                                        << " x_width : " << x_width
                                        << " x_channels : " << x_channels
                                        << " k_height : " << k_height
                                        << " k_width : " << k_width
                                        << " k_depth : " << k_depth
                                        << " k_h_dilation : " << k_h_dilation
                                        << " k_v_dilation : " << k_v_dilation
                                        << " k_h_stride : " << k_h_stride
                                        << " k_v_stride : " << k_v_stride
                                        << " top_pad : " << top_pad
                                        << " left_pad : " << left_pad
                                        << " right_pad : " << right_pad
                                        << " bottom_pad : " << bottom_pad
                                        << std::endl;
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

class Conv2dValidDirectRegression : public ::testing::Test {};

TEST_F(Conv2dValidDirectRegression, BasicTest) {
  for (int x_height = 1; x_height <= 5; ++x_height) {
    for (int x_width = 1; x_width <= 5; ++x_width) {
      for (int x_channels = 32; x_channels <= 64; x_channels += 32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 16; k_depth <= x_channels; k_depth += 16) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
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
                                  x_width + padding.left + padding.right,
                                  k_width, k_h_dilation, k_h_stride);

                              if (output_height <= 0 || output_width <= 0)
                                continue;

                              int test_seed = rng.getSeed();

                              // here output_height + width muct match the
                              // allocated memory for y
                              ImageGeometry Y(output_height, output_width,
                                              k_depth);

                              ImageGeometry X(x_height, x_width, x_channels);

                              WindowGeometry K(k_height, k_width, k_depth,
                                               -padding.top, -padding.left,
                                               k_v_stride, k_h_stride, 1,
                                               k_v_dilation, k_h_dilation);

                              Filter2dGeometry geom(X, Y, K);

                              KernelStimulus ks = create_simple_stimulus(geom);

                              auto &weights = ks.weights;
                              auto &bias = ks.bias;
                              auto &eff_mult = ks.eff_mult;
                              auto &input = ks.input;

                              auto expected =
                                  nn::test::ops::ref::Conv2dDenseReference(
                                      geom, input.data(), weights.data(),
                                      bias.data(), eff_mult.data(),
                                      ks.input_zero_point,
                                      ks.output_zero_point);

                              DerefInputFn::Params im_to_col_params(X, K);
                              DerefInputFn memcpy(&im_to_col_params);

                              int input_bytes = geom.window.shape.height *
                                                geom.window.shape.width *
                                                geom.input.depth;
                              int scratch_bytes =
                                  MatMulInt8::get_scratch_mem_bytes(
                                      input_bytes) +
                                  32;  //[asj] FIXME

                              std::vector<int8_t> T(scratch_bytes, 0);

                              int8_t kernel_pad_val = rng.rand<int8_t>();

                              std::array<int, 4> shape = {k_depth, k_height,
                                                          k_width, x_channels};
                              Conv2dReorderedWeights rw =
                                  MatMulInt8::reorder_kernel_weights(
                                      (int8_t *)weights.data(), shape, 8,
                                      kernel_pad_val);

                              MatMulDirectFn::Params p(X, K, x_channels,
                                                       rw.weights.data());
                              MatMulDirectFn aggregator(&p);

                              OutputTransformFnInt8::CanonicalMulAndBias
                                  canonical_values = OutputTransformFnInt8::
                                      canonicalise_mul_and_bias(
                                          eff_mult, bias, weights,
                                          ks.input_zero_point,
                                          ks.output_zero_point, k_depth);

                              QuantisationParams qp =
                                  OutputTransformFnInt8::quantise_activation(
                                      canonical_values.f_multipliers,
                                      canonical_values.f_biases,
                                      canonical_values.accu_min,
                                      canonical_values.accu_max);
                              OT_int8::Params ot_params(
                                  (int32_t)k_depth, &qp.otv, qp.biases.data(),
                                  qp.multipliers.data());
                              OT_int8 ot(&ot_params);
                              auto ir = ImageRegion(0, 0, 0, Y.height, Y.width,
                                                    Y.depth);

                              Filter2D::Params akp(Y, ir, VPU_INT8_ACC_PERIOD);

                              Conv2dValidDirect conv2d(&akp, &memcpy,
                                                       &aggregator, &ot);

                              auto output = std::vector<int8_t>(
                                  Y.height * Y.width * Y.depth);

                              conv2d.execute(&output[0], &input[0]);

                              for (int yh = 0; yh < Y.height; yh++) {
                                for (int yw = 0; yw < Y.width; yw++) {
                                  for (int yd = 0; yd < Y.depth; yd++) {
                                    int idx = yh * (Y.width * Y.depth) +
                                              yw * Y.depth + yd;
                                    // std::cout << "tflite: " <<
                                    // (int)expected[idx] << " xcore: " <<
                                    // (int)output[idx] << std::endl;
                                    EXPECT_NEAR((int)expected[idx],
                                                (int)output[idx], 1)
                                        << "tflite: " << (int)expected[idx]
                                        << " xcore: " << (int)output[idx]
                                        << " test seed: " << test_seed
                                        << " eff_mult[yd] : " << eff_mult[yd]
                                        << " x_height : " << x_height
                                        << " x_width : " << x_width
                                        << " x_channels : " << x_channels
                                        << " k_height : " << k_height
                                        << " k_width : " << k_width
                                        << " k_depth : " << k_depth
                                        << " k_h_dilation : " << k_h_dilation
                                        << " k_v_dilation : " << k_v_dilation
                                        << " k_h_stride : " << k_h_stride
                                        << " k_v_stride : " << k_v_stride
                                        << " top_pad : " << top_pad
                                        << " left_pad : " << left_pad
                                        << " right_pad : " << right_pad
                                        << " bottom_pad : " << bottom_pad
                                        << std::endl;
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
