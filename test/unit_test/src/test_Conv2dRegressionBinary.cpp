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
#include "nn_types.h"

extern "C" {
#include "tst_common.h"
#include "unity.h"
}

using namespace nn;
using namespace nn::test;

static auto rng = Rand(69);

const int itt_count = 4;

// int32_t is the smallest word size that the BNNs can input or output.
// There is no point trying to support smaller until the reference code does
// too.
const int chans_per_int32 = sizeof(int32_t) * CHAR_BIT;

struct BinaryKernelStimulus {
  std::vector<int32_t> weights;
  std::vector<float> bias;
  std::vector<float> eff_mult;
  std::vector<int32_t> thresholds;
  std::vector<int32_t> input;
  int32_t clamp_high;
  int32_t clamp_low;

  BinaryKernelStimulus(Filter2dGeometry &geom)
      : weights(geom.window.shape.height * geom.window.shape.width *
                    geom.output.depth * geom.input.depth / chans_per_int32,
                0),
        bias(geom.output.depth),
        eff_mult(geom.output.depth),
        thresholds(geom.output.depth),

        // We add 8 to the input to allow over reading, if it effects the result
        // then it's a bug.
        input(geom.input.height * geom.input.width * geom.input.depth /
                      chans_per_int32 +
                  8,
              0),
        clamp_high(0),
        clamp_low(0){};
};

BinaryKernelStimulus create_simple_binary_stimulus(Filter2dGeometry &geom) {
  BinaryKernelStimulus ks(geom);

  for (int idx = 0; idx < ks.weights.size(); ++idx)
    ks.weights[idx] = rng.rand<int32_t>();

  for (int idx = 0; idx < ks.input.size(); ++idx)
    ks.input[idx] = rng.rand<int32_t>();

  int receptive_volume =
      geom.input.depth * geom.window.shape.height * geom.window.shape.width;

  // Note this is the supported range - outside this range is unsupported.
  ks.clamp_high = receptive_volume;  // specifically, lower than this
  ks.clamp_low = 0;                  // and higher than this
  // as other values are meaningless

  // accu = [0, receptive_volume], map that to [-128, 127]*scalar, where scalar
  // [0.5, 2]
  for (int ch = 0; ch < geom.output.depth; ch++) {
    ks.eff_mult[ch] =
        rng.rand<float>(0.5, 2.0) * 256. / (float)receptive_volume;
    ks.bias[ch] = (rng.rand<int32_t>() % 130) - 65;
    ks.thresholds[ch] = rng.rand<int32_t>(0, receptive_volume);
  }
  return ks;
}

void test_Conv2dValidIndirectBinaryRegression() {
  for (int x_height = 1; x_height <= 2; ++x_height) {
    for (int x_width = 1; x_width <= 2; ++x_width) {
      for (int x_channels = chans_per_int32; x_channels <= chans_per_int32 * 3;
           x_channels += chans_per_int32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = chans_per_int32; k_depth <= chans_per_int32 * 10;
                 k_depth += chans_per_int32) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
                      // Note padding disabled
                      for (int top_pad = 0; top_pad <= 0; ++top_pad) {
                        for (int left_pad = 0; left_pad <= 0; ++left_pad) {
                          for (int right_pad = 0; right_pad <= 0; ++right_pad) {
                            for (int bottom_pad = 0; bottom_pad <= 0;
                                 ++bottom_pad) {
                              for (int itt = 0; itt < itt_count; ++itt) {
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

                                // std::cerr << "x_height :" << x_height
                                //   << " x_width :" << x_width
                                //   << " x_channels :" << x_channels
                                //   << " k_height :" << k_height
                                //   << " k_width :" << k_width
                                //   << " k_depth :" << k_depth
                                //   << " k_h_dilation :" << k_h_dilation
                                //   << " k_v_dilation :" << k_v_dilation
                                //   << " k_h_stride :" << k_h_stride
                                //   << " k_v_stride :" << k_v_stride
                                //   << " output_height :" << output_height
                                //   << " output_width :" << output_width
                                //   << " itt :" << itt <<std::endl;

                                int test_seed = rng.getSeed();

                                // here output_height + width muct match the
                                // allocated memory for y
                                ImageGeometry Y(output_height, output_width,
                                                k_depth, 1);

                                ImageGeometry X(x_height, x_width, x_channels,
                                                1);

                                WindowGeometry K(k_height, k_width, k_depth,
                                                 -padding.top, -padding.left,
                                                 k_v_stride, k_h_stride, 1,
                                                 k_v_dilation, k_h_dilation);

                                Filter2dGeometry geom(X, Y, K);

                                BinaryKernelStimulus ks =
                                    create_simple_binary_stimulus(geom);
                                auto &weights = ks.weights;
                                auto &input = ks.input;
                                auto &thresholds = ks.thresholds;

                                auto expected = nn::test::ops::ref::
                                    Conv2dBNNBinaryOutReference(
                                        geom, input.data(), weights.data(),
                                        thresholds.data());

                                // this is the value that would be inserted if
                                // we were padding
                                int im_to_col_pad_val = 0;
                                ImToColValid::Params im_to_col_params(
                                    X, K, x_channels);

                                ImToColValid memcpy(&im_to_col_params);
                                const int elements_per_byte = 8;
                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth /
                                                  elements_per_byte;
                                int scratch_bytes =
                                    MatMulInt8::get_scratch_mem_bytes(
                                        input_bytes) +
                                    32;

                                std::vector<int8_t> T(scratch_bytes, 0);

                                int8_t kernel_pad_val =
                                    rng.rand<int8_t>();  // This can be
                                                         // anything, 0 in
                                                         // pratice.
                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};

                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 1,
                                        kernel_pad_val);

                                MatMulBinary::Params p(k_depth, input_bytes);
                                MatMulBinary aggregator(&p);

                                aggregator.setWeights(rw.weights.data());

                                // adjust the thresholds from xorpopcount space
                                // to xcore space
                                auto adjusted_thresholds =
                                    OT_binary::adjust_thresholds(
                                        thresholds, x_channels, K, rw);

                                assert(adjusted_thresholds.size() > 0);

                                // pad qp.adjusted_thresholds to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                threshold_t pad_val =
                                    rng.rand<threshold_t>();  // this is
                                                              // arbitrary
                                OutputTransformFn::pad_final_access(
                                    adjusted_thresholds, VPU_INT16_EPV,
                                    pad_val);

                                OT_binary ot;
                                ot.setThresholds(adjusted_thresholds.data());
                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                Filter2D::Params akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);

                                BNNConv2dValidIndirectBinary conv2d(
                                    &memcpy, &aggregator, &ot);
                                alignas(4)
                                    int32_t output[Y.height * Y.width *
                                                   Y.depth / chans_per_int32];
                                std::memset(output, 0x55, sizeof(output));

                                nn::execute((int8_t *)&output[0],
                                            (int8_t *)&input[0],
                                            &conv2d, &akp, &T[0]);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0;
                                         yd < Y.depth / chans_per_int32; yd++) {
                                      int idx = yh * (Y.width * Y.depth /
                                                      chans_per_int32) +
                                                yw * Y.depth / chans_per_int32 +
                                                yd;

                                      TEST_ASSERT_EQUAL_HEX32(
                                          (int)expected[idx], (int)output[idx]);
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
}

void test_Conv2dValidDirectBinaryRegression() {
  for (int x_height = 1; x_height <= 4; ++x_height) {
    for (int x_width = 1; x_width <= 4; ++x_width) {
      for (int x_channels = 256; x_channels <= 256 * 2; x_channels += 256) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 256; k_depth <= 256 * 2; k_depth += 256) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
                      for (int top_pad = 0; top_pad <= 0; ++top_pad) {
                        for (int left_pad = 0; left_pad <= 0; ++left_pad) {
                          for (int right_pad = 0; right_pad <= 0; ++right_pad) {
                            for (int bottom_pad = 0; bottom_pad <= 0;
                                 ++bottom_pad) {
                              for (int itt = 0; itt < itt_count; ++itt) {
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

                                // std::cerr << "x_height :" << x_height
                                //   << " x_width :" << x_width
                                //   << " x_channels :" << x_channels
                                //   << " k_height :" << k_height
                                //   << " k_width :" << k_width
                                //   << " k_depth :" << k_depth
                                //   << " k_h_dilation :" << k_h_dilation
                                //   << " k_v_dilation :" << k_v_dilation
                                //   << " k_h_stride :" << k_h_stride
                                //   << " k_v_stride :" << k_v_stride
                                //   << " output_height :" << output_height
                                //   << " output_width :" << output_width
                                //   << " itt :" << itt <<std::endl;

                                // here output_height + width muct match the
                                // allocated memory for y
                                ImageGeometry Y(output_height, output_width,
                                                k_depth, 1);

                                ImageGeometry X(x_height, x_width, x_channels,
                                                1);

                                WindowGeometry K(k_height, k_width, k_depth,
                                                 -padding.top, -padding.left,
                                                 k_v_stride, k_h_stride, 1,
                                                 k_v_dilation, k_h_dilation);

                                Filter2dGeometry geom(X, Y, K);

                                BinaryKernelStimulus ks =
                                    create_simple_binary_stimulus(geom);
                                auto &weights = ks.weights;
                                auto &input = ks.input;
                                auto &thresholds = ks.thresholds;

                                auto expected = nn::test::ops::ref::
                                    Conv2dBNNBinaryOutReference(
                                        geom, input.data(), weights.data(),
                                        thresholds.data());

                                DerefInputFn::Params im_to_col_params(X, K);
                                DerefInputFn memcpy(&im_to_col_params);

                                int overread_bytes =
                                    memcpy.get_overread_bytes();
                                input.resize(input.size() + overread_bytes /
                                                                sizeof(int8_t),
                                             0);

                                const int elements_per_byte = 8;
                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth /
                                                  elements_per_byte;

                                int8_t kernel_pad_val = rng.rand<int8_t>();

                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};
                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 1,
                                        kernel_pad_val);

                                MatMulBinaryDirectFn::Params p(X, K,
                                                               x_channels);
                                MatMulBinaryDirectFn aggregator(&p);
                                aggregator.setWeights(rw.weights.data());

                                // adjust the thresholds from xorpopcount space
                                // to xcore space
                                auto adjusted_thresholds =
                                    OT_binary::adjust_thresholds(
                                        thresholds, x_channels, K, rw);

                                assert(adjusted_thresholds.size() > 0);

                                // pad q.biases and  q.multipliers to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary

                                OutputTransformFn::pad_final_access(
                                    adjusted_thresholds, VPU_INT16_EPV,
                                    pad_val);

                                OT_binary ot;

                                ot.setThresholds(adjusted_thresholds.data());
                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                Filter2D::Params akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);

                                BNNConv2dValidDirectBinary conv2d(
                                    &memcpy, &aggregator, &ot);

                                alignas(4)
                                    int32_t output[Y.height * Y.width *
                                                   Y.depth / chans_per_int32];
                                std::memset(output, 0x55, sizeof(output));
                                
                                nn::execute((int8_t *)&output[0],
                                            (int8_t *)&input[0],
                                            &conv2d, &akp);
                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0;
                                         yd < Y.depth / chans_per_int32; yd++) {
                                      int idx = yh * (Y.width * Y.depth /
                                                      chans_per_int32) +
                                                yw * Y.depth / chans_per_int32 +
                                                yd;

                                      // printf("%d %08x %08x\n", idx,
                                      // expected[idx], output[idx]);

                                      TEST_ASSERT_EQUAL_HEX32(
                                          (int)expected[idx], (int)output[idx]);
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
}

void test_Conv2dValidIndirectInt8Regression() {
  for (int x_height = 1; x_height <= 2; ++x_height) {
    for (int x_width = 1; x_width <= 2; ++x_width) {
      for (int x_channels = chans_per_int32; x_channels <= chans_per_int32 * 3;
           x_channels += chans_per_int32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = chans_per_int32; k_depth <= chans_per_int32 * 10;
                 k_depth += chans_per_int32) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
                      for (int top_pad = 0; top_pad <= 0; ++top_pad) {
                        for (int left_pad = 0; left_pad <= 0; ++left_pad) {
                          for (int right_pad = 0; right_pad <= 0; ++right_pad) {
                            for (int bottom_pad = 0; bottom_pad <= 0;
                                 ++bottom_pad) {
                              for (int itt = 0; itt < itt_count; ++itt) {
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

                                // std::cerr << "x_height :" << x_height
                                //   << " x_width :" << x_width
                                //   << " x_channels :" << x_channels
                                //   << " k_height :" << k_height
                                //   << " k_width :" << k_width
                                //   << " k_depth :" << k_depth
                                //   << " k_h_dilation :" << k_h_dilation
                                //   << " k_v_dilation :" << k_v_dilation
                                //   << " k_h_stride :" << k_h_stride
                                //   << " k_v_stride :" << k_v_stride
                                //   << " output_height :" << output_height
                                //   << " output_width :" << output_width
                                //   << " itt :" << itt <<std::endl;

                                int test_seed = rng.getSeed();

                                // here output_height + width muct match the
                                // allocated memory for y
                                ImageGeometry Y(output_height, output_width,
                                                k_depth);

                                ImageGeometry X(x_height, x_width, x_channels,
                                                1);

                                WindowGeometry K(k_height, k_width, k_depth,
                                                 -padding.top, -padding.left,
                                                 k_v_stride, k_h_stride, 1,
                                                 k_v_dilation, k_h_dilation);

                                Filter2dGeometry geom(X, Y, K);

                                BinaryKernelStimulus ks =
                                    create_simple_binary_stimulus(geom);
                                auto &weights = ks.weights;
                                auto &input = ks.input;
                                auto &post_activation_multiplier = ks.eff_mult;
                                auto &post_activation_bias = ks.bias;
                                auto clamp_low = ks.clamp_low;
                                auto clamp_high = ks.clamp_high;

                                auto expected = nn::test::ops::ref::
                                    Conv2dBNNIntOutReference(
                                        geom, input.data(), weights.data(),
                                        post_activation_multiplier.data(),
                                        post_activation_bias.data(), clamp_low,
                                        clamp_high);

                                // this is the value that would be inserted if
                                // we were padding
                                int im_to_col_pad_val = 0;
                                ImToColValid::Params im_to_col_params(
                                    X, K, x_channels);

                                ImToColValid memcpy(&im_to_col_params);
                                const int elements_per_byte = 8;
                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth /
                                                  elements_per_byte;
                                int scratch_bytes =
                                    MatMulInt8::get_scratch_mem_bytes(
                                        input_bytes) +
                                    32;

                                std::vector<int8_t> T(scratch_bytes, 0);

                                int8_t kernel_pad_val =
                                    rng.rand<int8_t>();  // This can be
                                                         // anything, 0 in
                                                         // pratice.
                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};

                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 1,
                                        kernel_pad_val);

                                MatMulBinary::Params p(k_depth, input_bytes);
                                MatMulBinary aggregator(&p);

                                aggregator.setWeights(rw.weights.data());
                                // adjust the thresholds from xorpopcount space
                                // to xcore space

                                int receptive_volume =
                                    k_height * k_width * x_channels;

                                MulsAndBias mul_and_biases =
                                    OT_int8_clamped::canonicalise_mul_and_bias(
                                        post_activation_multiplier,
                                        post_activation_bias, receptive_volume,
                                        clamp_low, clamp_high, k_depth);

                                auto accu_overlaps =
                                    OT_int8_clamped::get_accumulator_overlaps(
                                        receptive_volume, k_depth, rw);

                                QuantisationParams qp =
                                    OutputTransformFnInt8::quantise_activation(
                                        mul_and_biases);

                                auto serialised_offsets_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        accu_overlaps, qp.multipliers,
                                        qp.biases);

                                // pad qp.adjusted_thresholds to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                threshold_t pad_val =
                                    rng.rand<threshold_t>();  // this is
                                                              // arbitrary
                                OutputTransformFn::pad_final_access(
                                    serialised_offsets_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8_clamped::Params ot_params(
                                    (int32_t)k_depth, qp.initial_shr,
                                    qp.final_shr);

                                OT_int8_clamped ot(&ot_params);
                                assert(serialised_offsets_multipliers_and_biases
                                           .size() > 0);
                                ot.setOffsetsMultipliersAndBiases(
                                    serialised_offsets_multipliers_and_biases
                                        .data());

                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                Filter2D::Params akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);

                                BNNConv2dValidIndirectInt8 conv2d(
                                    &memcpy, &aggregator, &ot);
                                alignas(4)
                                    int8_t output[Y.height * Y.width * Y.depth];
                                std::memset(output, 0x55, sizeof(output));

                                nn::execute(output, (int8_t *)&input[0],
                                            &conv2d, &akp,
                                            &T[0]);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      TEST_ASSERT_INT8_WITHIN(
                                          1, (int8_t)expected[idx],
                                          (int8_t)output[idx]);
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
}

void test_Conv2dValidDirectInt8Regression() {
  for (int x_height = 1; x_height <= 4; ++x_height) {
    for (int x_width = 1; x_width <= 4; ++x_width) {
      for (int x_channels = 256; x_channels <= 256 * 2; x_channels += 256) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 256; k_depth <= 256 * 2; k_depth += 256) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
                      for (int top_pad = 0; top_pad <= 0; ++top_pad) {
                        for (int left_pad = 0; left_pad <= 0; ++left_pad) {
                          for (int right_pad = 0; right_pad <= 0; ++right_pad) {
                            for (int bottom_pad = 0; bottom_pad <= 0;
                                 ++bottom_pad) {
                              for (int itt = 0; itt < itt_count; ++itt) {
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

                                // std::cerr << "x_height :" << x_height
                                //   << " x_width :" << x_width
                                //   << " x_channels :" << x_channels
                                //   << " k_height :" << k_height
                                //   << " k_width :" << k_width
                                //   << " k_depth :" << k_depth
                                //   << " k_h_dilation :" << k_h_dilation
                                //   << " k_v_dilation :" << k_v_dilation
                                //   << " k_h_stride :" << k_h_stride
                                //   << " k_v_stride :" << k_v_stride
                                //   << " output_height :" << output_height
                                //   << " output_width :" << output_width
                                //   << " itt :" << itt <<std::endl;

                                // here output_height + width muct match the
                                // allocated memory for y
                                ImageGeometry Y(output_height, output_width,
                                                k_depth, 8);

                                ImageGeometry X(x_height, x_width, x_channels,
                                                1);

                                WindowGeometry K(k_height, k_width, k_depth,
                                                 -padding.top, -padding.left,
                                                 k_v_stride, k_h_stride, 1,
                                                 k_v_dilation, k_h_dilation);

                                Filter2dGeometry geom(X, Y, K);

                                BinaryKernelStimulus ks =
                                    create_simple_binary_stimulus(geom);
                                auto &weights = ks.weights;
                                auto &input = ks.input;
                                auto &post_activation_multiplier = ks.eff_mult;
                                auto &post_activation_bias = ks.bias;
                                auto clamp_low = ks.clamp_low;
                                auto clamp_high = ks.clamp_high;

                                auto expected = nn::test::ops::ref::
                                    Conv2dBNNIntOutReference(
                                        geom, input.data(), weights.data(),
                                        post_activation_multiplier.data(),
                                        post_activation_bias.data(), clamp_low,
                                        clamp_high);

                                DerefInputFn::Params im_to_col_params(X, K);
                                DerefInputFn memcpy(&im_to_col_params);

                                int overread_bytes =
                                    memcpy.get_overread_bytes();
                                input.resize(input.size() + overread_bytes /
                                                                sizeof(int8_t),
                                             0);

                                const int elements_per_byte = 8;
                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth /
                                                  elements_per_byte;

                                int8_t kernel_pad_val = rng.rand<int8_t>();

                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};
                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 1,
                                        kernel_pad_val);

                                MatMulBinaryDirectFn::Params p(X, K,
                                                               x_channels);
                                MatMulBinaryDirectFn aggregator(&p);
                                aggregator.setWeights(rw.weights.data());

                                int receptive_volume =
                                    k_height * k_width * x_channels;

                                MulsAndBias mul_and_biases =
                                    OT_int8_clamped::canonicalise_mul_and_bias(
                                        post_activation_multiplier,
                                        post_activation_bias, receptive_volume,
                                        clamp_low, clamp_high, k_depth);

                                auto accu_overlaps =
                                    OT_int8_clamped::get_accumulator_overlaps(
                                        receptive_volume, k_depth, rw);

                                QuantisationParams qp =
                                    OutputTransformFnInt8::quantise_activation(
                                        mul_and_biases);

                                auto serialised_offsets_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        accu_overlaps, qp.multipliers,
                                        qp.biases);

                                // pad q.biases and  q.multipliers to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary

                                OutputTransformFn::pad_final_access(
                                    serialised_offsets_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8_clamped::Params ot_params(
                                    (int32_t)k_depth, qp.initial_shr,
                                    qp.final_shr);

                                OT_int8_clamped ot(&ot_params);

                                assert(serialised_offsets_multipliers_and_biases
                                           .size() > 0);
                                ot.setOffsetsMultipliersAndBiases(
                                    serialised_offsets_multipliers_and_biases
                                        .data());

                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                Filter2D::Params akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);

                                BNNConv2dValidDirectInt8 conv2d(
                                    &memcpy, &aggregator, &ot);

                                alignas(4)
                                    int8_t output[Y.height * Y.width * Y.depth];
                                std::memset(output, 0x55, sizeof(output));

                                nn::execute((int8_t *)output,
                                            (int8_t *)input.data(),
                                            &conv2d, &akp);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      TEST_ASSERT_INT8_WITHIN(
                                          1, (int8_t)expected[idx],
                                          (int8_t)output[idx]);
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
}

extern "C" void test_conv2d_binary_regression();
void test_conv2d_binary_regression() {
  UNITY_SET_FILE();
  RUN_TEST(test_Conv2dValidIndirectBinaryRegression);
  RUN_TEST(test_Conv2dValidDirectBinaryRegression);
  RUN_TEST(test_Conv2dValidIndirectInt8Regression);
  RUN_TEST(test_Conv2dValidDirectInt8Regression);
}
