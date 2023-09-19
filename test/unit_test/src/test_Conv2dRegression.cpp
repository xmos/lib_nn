#include <algorithm>
#include <cassert>
#include <cstdint>
#include <iostream>
#include <memory>
#include <vector>

#include "AbstractKernel.hpp"
#include "AggregateFn.hpp"
#include "MemCpyFn.hpp"
#include "OutputTransformFn.hpp"
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
const int itt_count = 32;

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
    ks.eff_mult[ch] = rng.rand<float>(1.0 / (256 * 256), 1.0 / 256);
    ks.bias[ch] = rng.rand<int8_t>() % 512;
  }
  float eff_mult_scalar = 2.0;
  for (int ch = 0; ch < geom.output.depth; ch++) {
    ks.eff_mult[ch] *= eff_mult_scalar;
  }
  return ks;
}

void test_Conv2dPaddedIndirectRegression() {
  for (int x_height = 1; x_height <= 2; ++x_height) {
    for (int x_width = 1; x_width <= 2; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 4; k_depth <= max_k_channels; k_depth += 4) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
                      for (int top_pad = 0; top_pad <= 1; ++top_pad) {
                        for (int left_pad = 0; left_pad <= 1; ++left_pad) {
                          for (int right_pad = 0; right_pad <= 1; ++right_pad) {
                            for (int bottom_pad = 0; bottom_pad <= 1;
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

                                KernelStimulus ks =
                                    create_simple_stimulus(geom);
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

                                ImToColPadded memcpy(
                                    X, K, padding, x_channels,
                                    ks.input_zero_point);

                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth;
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
                                        (int8_t *)weights.data(), shape, 8,
                                        kernel_pad_val);

                                MatMulInt8 aggregator(k_depth, input_bytes);

                                assert(eff_mult.size() > 0);

                                MulsAndBias mul_and_biases =
                                    OutputTransformFnInt8::
                                        canonicalise_mul_and_bias(
                                            eff_mult, bias, weights,
                                            ks.input_zero_point,
                                            ks.output_zero_point, k_depth);

                                auto quantizer =
                                    OutputTransformFnInt8_Group::Quantizer();
                                OutputTransformFnInt8_Group::QuantisationParams
                                    qp = quantizer.quantise_activation(
                                        mul_and_biases, false);

                                assert(qp.multipliers.size() > 0);
                                assert(qp.biases.size() > 0);

                                auto serialised_offsets_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        qp.multipliers, qp.biases);

                                // pad qp.multipliers_and_biases to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary
                                OutputTransformFn::pad_final_access(
                                    serialised_offsets_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8 ot((int32_t)k_depth,
                                                          qp.initial_shr,
                                                          qp.final_shr);
                                assert(qp.multipliers.size() > 0);
                                assert(qp.biases.size() > 0);

                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                AbstractKernel akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);
                                alignas(4)
                                    int8_t output[Y.height * Y.width * Y.depth];
                             
                                abstract_kernel_params_t a = akp.getParams();
                                memcpyfn_imtocol_padded_params_t m = memcpy.getParams();
                                mat_mul_generic_params_t agg = aggregator.getParams();
                                otfn_int8_params_t o = ot.getParams();
                                conv_params_t params;
                                params.memcopy_fn = (MemFnType)memcpyfn_imtocol_padded;
                                params.aggregate_fn = (AggFnType)mat_mul_generic_int8;
                                params.output_transform_fn = (OtFnType)otfn_int8;
                                params.mem_p = &m;
                                params.agg_p = &agg;
                                params.ot_p = &o;
                                nn::execute(&output[0], &input[0], &params,
                                            &a, rw.weights.data(), serialised_offsets_multipliers_and_biases.data(), /*isConv=*/true, &T[0]);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;

                                      TEST_ASSERT_INT32_WITHIN(
                                          1, (int)expected[idx],
                                          (int)output[idx]);
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

void test_Conv2dPaddedIndirectRegression_channelwise() {
  for (int x_height = 1; x_height <= 2; ++x_height) {
    for (int x_width = 1; x_width <= 2; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 4; k_depth <= max_k_channels; k_depth += 4) {
              for (int k_h_dilation = 1; k_h_dilation <= 2; ++k_h_dilation) {
                for (int k_v_dilation = 1; k_v_dilation <= 2; ++k_v_dilation) {
                  for (int k_h_stride = 1; k_h_stride <= 2; ++k_h_stride) {
                    for (int k_v_stride = 1; k_v_stride <= 2; ++k_v_stride) {
                      for (int top_pad = 0; top_pad <= 1; ++top_pad) {
                        for (int left_pad = 0; left_pad <= 1; ++left_pad) {
                          for (int right_pad = 0; right_pad <= 1; ++right_pad) {
                            for (int bottom_pad = 0; bottom_pad <= 1;
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

                                KernelStimulus ks =
                                    create_simple_stimulus(geom);
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

                                ImToColPadded memcpy(
                                    X, K, padding, x_channels,
                                    ks.input_zero_point);

                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth;
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
                                        (int8_t *)weights.data(), shape, 8,
                                        kernel_pad_val);

                                MatMulInt8 aggregator(k_depth, input_bytes);

                                assert(eff_mult.size() > 0);

                                MulsAndBias mul_and_biases =
                                    OutputTransformFnInt8::
                                        canonicalise_mul_and_bias(
                                            eff_mult, bias, weights,
                                            ks.input_zero_point,
                                            ks.output_zero_point, k_depth);

                                auto quantizer =
                                    OutputTransformFnInt8_Channelwise::
                                        Quantizer();
                                OutputTransformFnInt8_Channelwise::
                                    QuantisationParams qp =
                                        quantizer.quantise_activation(
                                            mul_and_biases, false);

                                assert(qp.initial_shifts.size() > 0);
                                assert(qp.multipliers.size() > 0);
                                assert(qp.biases.size() > 0);

                                auto serialised_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        qp.initial_shifts, qp.multipliers,
                                        qp.biases);

                                // pad qp.multipliers_and_biases to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary
                                OutputTransformFn::pad_final_access(
                                    serialised_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8_channelwise ot(
                                    (int32_t)k_depth, qp.final_shr);
                                assert(qp.multipliers.size() > 0);
                                assert(qp.biases.size() > 0);

                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                AbstractKernel akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);
                                alignas(4)
                                    int8_t output[Y.height * Y.width * Y.depth];

                                abstract_kernel_params_t a = akp.getParams();
                                memcpyfn_imtocol_padded_params_t m = memcpy.getParams();
                                mat_mul_generic_params_t agg = aggregator.getParams();
                                otfn_int8_channelwise_params_t o = ot.getParams();
                                conv_params_t params;
                                params.memcopy_fn = (MemFnType)memcpyfn_imtocol_padded;
                                params.aggregate_fn = (AggFnType)mat_mul_generic_int8;
                                params.output_transform_fn = (OtFnType)otfn_int8_channelwise;
                                params.mem_p = &m;
                                params.agg_p = &agg;
                                params.ot_p = &o;
                                nn::execute(&output[0], &input[0], &params,
                                            &a, rw.weights.data(), serialised_multipliers_and_biases.data(), /*isConv=*/true, &T[0]);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      if (qp.final_shr <= 0)
                                        TEST_ASSERT_INT32_WITHIN(
                                            1, (int)expected[idx],
                                            (int)output[idx]);
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

void test_Conv2dValidIndirectRegression() {
  for (int x_height = 1; x_height <= 5; ++x_height) {
    for (int x_width = 1; x_width <= 5; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 4; k_depth <= max_k_channels; k_depth += 4) {
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

                                KernelStimulus ks =
                                    create_simple_stimulus(geom);
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

                                ImToColValid memcpy(
                                    X, K, x_channels);

                                int overread_bytes =
                                    memcpy.get_overread_bytes();
                                input.resize(input.size() + overread_bytes /
                                                                sizeof(int8_t),
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

                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};
                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 8,
                                        kernel_pad_val);

                                MatMulInt8 aggregator(k_depth, input_bytes);

                                MulsAndBias mul_and_biases =
                                    OutputTransformFnInt8::
                                        canonicalise_mul_and_bias(
                                            eff_mult, bias, weights,
                                            ks.input_zero_point,
                                            ks.output_zero_point, k_depth);

                                auto quantizer =
                                    OutputTransformFnInt8_Group::Quantizer();
                                OutputTransformFnInt8_Group::QuantisationParams
                                    qp = quantizer.quantise_activation(
                                        mul_and_biases, false);

                                auto serialised_offsets_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        qp.multipliers, qp.biases);
                                // pad q.biases and  q.multipliers to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary
                                OutputTransformFn::pad_final_access(
                                    serialised_offsets_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8 ot((int32_t)k_depth,
                                                          qp.initial_shr,
                                                          qp.final_shr);
                                assert(serialised_offsets_multipliers_and_biases
                                           .size() > 0);
                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                AbstractKernel akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);

                                auto output = std::vector<int8_t>(
                                    Y.height * Y.width * Y.depth);

                                abstract_kernel_params_t a = akp.getParams();
                                memcpyfn_imtocol_valid_params_t m = memcpy.getParams();
                                mat_mul_generic_params_t agg = aggregator.getParams();
                                otfn_int8_params_t o = ot.getParams();
                                conv_params_t params;
                                params.memcopy_fn = (MemFnType)memcpyfn_imtocol_valid;
                                params.aggregate_fn = (AggFnType)mat_mul_generic_int8;
                                params.output_transform_fn = (OtFnType)otfn_int8;
                                params.mem_p = &m;
                                params.agg_p = &agg;
                                params.ot_p = &o;
                                nn::execute(&output[0], &input[0], &params,
                                            &a, rw.weights.data(), serialised_offsets_multipliers_and_biases.data(), /*isConv=*/true, &T[0]);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      // std::cout << "tflite: " <<
                                      // (int)expected[idx] << " xcore: " <<
                                      // (int)output[idx] << std::endl;

                                      TEST_ASSERT_INT32_WITHIN(
                                          1, (int)expected[idx],
                                          (int)output[idx]);
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

void test_Conv2dValidIndirectRegression_channelwise() {
  for (int x_height = 1; x_height <= 5; ++x_height) {
    for (int x_width = 1; x_width <= 5; ++x_width) {
      for (int x_channels = 4; x_channels <= 16; x_channels += 4) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 4; k_depth <= max_k_channels; k_depth += 4) {
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

                                KernelStimulus ks =
                                    create_simple_stimulus(geom);
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

                                ImToColValid memcpy(
                                    X, K, x_channels);

                                int overread_bytes =
                                    memcpy.get_overread_bytes();
                                input.resize(input.size() + overread_bytes /
                                                                sizeof(int8_t),
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

                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};
                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 8,
                                        kernel_pad_val);

                                MatMulInt8 aggregator(k_depth, input_bytes);

                                MulsAndBias mul_and_biases =
                                    OutputTransformFnInt8::
                                        canonicalise_mul_and_bias(
                                            eff_mult, bias, weights,
                                            ks.input_zero_point,
                                            ks.output_zero_point, k_depth);

                                auto quantizer =
                                    OutputTransformFnInt8_Channelwise::
                                        Quantizer();
                                OutputTransformFnInt8_Channelwise::
                                    QuantisationParams qp =
                                        quantizer.quantise_activation(
                                            mul_and_biases, false);

                                auto serialised_offsets_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        qp.initial_shifts, qp.multipliers,
                                        qp.biases);
                                // pad q.biases and  q.multipliers to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary
                                OutputTransformFn::pad_final_access(
                                    serialised_offsets_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8_channelwise ot(
                                    (int32_t)k_depth, qp.final_shr);
                                assert(serialised_offsets_multipliers_and_biases
                                           .size() > 0);
                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                AbstractKernel akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);
                                auto output = std::vector<int8_t>(
                                    Y.height * Y.width * Y.depth);

                                abstract_kernel_params_t a = akp.getParams();
                                memcpyfn_imtocol_valid_params_t m = memcpy.getParams();
                                mat_mul_generic_params_t agg = aggregator.getParams();
                                otfn_int8_channelwise_params_t o = ot.getParams();
                                conv_params_t params;
                                params.memcopy_fn = (MemFnType)memcpyfn_imtocol_valid;
                                params.aggregate_fn = (AggFnType)mat_mul_generic_int8;
                                params.output_transform_fn = (OtFnType)otfn_int8_channelwise;
                                params.mem_p = &m;
                                params.agg_p = &agg;
                                params.ot_p = &o;
                                nn::execute(&output[0], &input[0], &params,
                                            &a, rw.weights.data(), serialised_offsets_multipliers_and_biases.data(), /*isConv=*/true, &T[0]);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      // std::cout << "tflite: " <<
                                      // (int)expected[idx] << " xcore: " <<
                                      // (int)output[idx] << std::endl;
                                      if (qp.final_shr <= 0)
                                        TEST_ASSERT_INT32_WITHIN(
                                            1, (int)expected[idx],
                                            (int)output[idx]);
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

void test_Conv2dValidDirectRegression() {
  for (int x_height = 1; x_height <= 3; ++x_height) {
    for (int x_width = 1; x_width <= 3; ++x_width) {
      for (int x_channels = 32; x_channels <= 64; x_channels += 32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 16; k_depth <= max_k_channels; k_depth += 16) {
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

                                KernelStimulus ks =
                                    create_simple_stimulus(geom);

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

                                DerefInputFn memcpy(X,K);
                                memcpyfn_deref_params_t m = memcpy.getParams();

                                conv_params_t params;
                                params.memcopy_fn = (MemFnType)memcpyfn_deref;
                                params.aggregate_fn = (AggFnType)mat_mul_direct_int8;
                                params.output_transform_fn = (OtFnType)otfn_int8;

                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth;
                                int scratch_bytes =
                                    MatMulInt8::get_scratch_mem_bytes(
                                        input_bytes) +
                                    32;  //[asj] FIXME

                                std::vector<int8_t> T(scratch_bytes, 0);

                                int8_t kernel_pad_val = rng.rand<int8_t>();

                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};
                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 8,
                                        kernel_pad_val);

                                MatMulDirectFn aggregator(X, K, x_channels);
                                mat_mul_direct_params_t agg = aggregator.getParams();

                                MulsAndBias mul_and_biases =
                                    OutputTransformFnInt8::
                                        canonicalise_mul_and_bias(
                                            eff_mult, bias, weights,
                                            ks.input_zero_point,
                                            ks.output_zero_point, k_depth);

                                auto quantizer =
                                    OutputTransformFnInt8_Group::Quantizer();
                                OutputTransformFnInt8_Group::QuantisationParams
                                    qp = quantizer.quantise_activation(
                                        mul_and_biases, false);

                                auto serialised_offsets_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        qp.multipliers, qp.biases);
                                // pad q.biases and  q.multipliers to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary
                                OutputTransformFn::pad_final_access(
                                    serialised_offsets_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8 ot((int32_t)k_depth, qp.initial_shr, qp.final_shr);
                                otfn_int8_params_t o = ot.getParams();

                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                AbstractKernel akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);
                                abstract_kernel_params_t a = akp.getParams();
                                
                                params.mem_p = &m;
                                params.agg_p = &agg;
                                params.ot_p = &o;

                                auto output = std::vector<int8_t>(
                                    Y.height * Y.width * Y.depth);

                                nn::execute(&output[0], &input[0], &params,
                                            &a, rw.weights.data(), serialised_offsets_multipliers_and_biases
                                        .data(), /*isConv=*/true);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      TEST_ASSERT_INT32_WITHIN(
                                          1, (int)expected[idx],
                                          (int)output[idx]);
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

void test_Conv2dValidDirectRegression_channelwise() {
  for (int x_height = 1; x_height <= 3; ++x_height) {
    for (int x_width = 1; x_width <= 3; ++x_width) {
      for (int x_channels = 32; x_channels <= 64; x_channels += 32) {
        for (int k_height = 1; k_height <= x_height; ++k_height) {
          for (int k_width = 1; k_width <= x_width; ++k_width) {
            for (int k_depth = 16; k_depth <= max_k_channels; k_depth += 16) {
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

                                KernelStimulus ks =
                                    create_simple_stimulus(geom);

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

                                DerefInputFn memcpy(X,K);

                                int input_bytes = geom.window.shape.height *
                                                  geom.window.shape.width *
                                                  geom.input.depth;
                                int scratch_bytes =
                                    MatMulInt8::get_scratch_mem_bytes(
                                        input_bytes) +
                                    32;  //[asj] FIXME

                                std::vector<int8_t> T(scratch_bytes, 0);

                                int8_t kernel_pad_val = rng.rand<int8_t>();

                                std::array<int, 4> shape = {
                                    {k_depth, k_height, k_width, x_channels}};
                                Conv2dReorderedWeights rw =
                                    MatMulInt8::reorder_kernel_weights(
                                        (int8_t *)weights.data(), shape, 8,
                                        kernel_pad_val);

                                MatMulDirectFn aggregator(X, K, x_channels);

                                MulsAndBias mul_and_biases =
                                    OutputTransformFnInt8::
                                        canonicalise_mul_and_bias(
                                            eff_mult, bias, weights,
                                            ks.input_zero_point,
                                            ks.output_zero_point, k_depth);

                                auto quantizer =
                                    OutputTransformFnInt8_Channelwise::
                                        Quantizer();
                                OutputTransformFnInt8_Channelwise::
                                    QuantisationParams qp =
                                        quantizer.quantise_activation(
                                            mul_and_biases, false);

                                auto serialised_offsets_multipliers_and_biases =
                                    OutputTransformFn::serialise_memory(
                                        qp.initial_shifts, qp.multipliers,
                                        qp.biases);
                                // pad q.biases and  q.multipliers to a multiple
                                // of VPU_INT16_EPV this is to work around array
                                // over reads
                                int16_t pad_val =
                                    rng.rand<int16_t>();  // this is arbitrary
                                OutputTransformFn::pad_final_access(
                                    serialised_offsets_multipliers_and_biases,
                                    VPU_INT16_EPV, pad_val);

                                OT_int8_channelwise ot(
                                    (int32_t)k_depth, qp.final_shr);
                                assert(serialised_offsets_multipliers_and_biases
                                           .size() > 0);

                                auto ir = ImageRegion(0, 0, 0, Y.height,
                                                      Y.width, Y.depth);

                                AbstractKernel akp(Y, ir,
                                                     VPU_INT8_ACC_PERIOD);

                                auto output = std::vector<int8_t>(
                                    Y.height * Y.width * Y.depth);

                                abstract_kernel_params_t a = akp.getParams();
                                memcpyfn_deref_params_t m = memcpy.getParams();
                                mat_mul_direct_params_t agg = aggregator.getParams();
                                otfn_int8_channelwise_params_t o = ot.getParams();
                                conv_params_t params;
                                params.memcopy_fn = (MemFnType)memcpyfn_deref;
                                params.aggregate_fn = (AggFnType)mat_mul_direct_int8;
                                params.output_transform_fn = (OtFnType)otfn_int8_channelwise;
                                params.mem_p = &m;
                                params.agg_p = &agg;
                                params.ot_p = &o;
                                nn::execute(&output[0], &input[0], &params,
                                            &a, rw.weights.data(), serialised_offsets_multipliers_and_biases.data(), /*isConv=*/true);

                                for (int yh = 0; yh < Y.height; yh++) {
                                  for (int yw = 0; yw < Y.width; yw++) {
                                    for (int yd = 0; yd < Y.depth; yd++) {
                                      int idx = yh * (Y.width * Y.depth) +
                                                yw * Y.depth + yd;
                                      if (qp.final_shr <= 0)
                                        TEST_ASSERT_INT32_WITHIN(
                                            1, (int)expected[idx],
                                            (int)output[idx]);
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

extern "C" void test_conv2d_regression();
void test_conv2d_regression() {
  UNITY_SET_FILE();
  RUN_TEST(test_Conv2dPaddedIndirectRegression);
  RUN_TEST(test_Conv2dPaddedIndirectRegression_channelwise);
  RUN_TEST(test_Conv2dValidDirectRegression);
  RUN_TEST(test_Conv2dValidIndirectRegression_channelwise);
  RUN_TEST(test_Conv2dValidIndirectRegression);
  RUN_TEST(test_Conv2dValidIndirectRegression_channelwise);
}
