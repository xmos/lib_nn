#include "conv2d_utils.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdlib>
#include <iostream>
#include <vector>

#include "xs3_vpu.h"

using namespace nn;

struct Conv2dChannelParams {
  int16_t shift1;
  int16_t scale;
  int16_t offset_scale;
  int16_t offset;
  int16_t shift2;
};

static void compute_rshift_scale(int16_t &rshift, int16_t &scale,
                                 const float &multiplier) {
  rshift = 16;
  float fscale = 0x7FFF;

  if (multiplier != 0.0f) {
    rshift = int16_t(-ceilf(log2f(multiplier))) + 1;
    fscale = roundf(ldexpf(multiplier, 14 + rshift));
  }

  if (fscale == ldexpf(1, 15)) {
    rshift -= 1;
    fscale /= 2;
  }

  constexpr int SHIFT_ADJUSTMENT = 7;

  // "We are using 16-bits instead of 8 so we need to adjust the shift"

  rshift -= SHIFT_ADJUSTMENT;
  scale = int16_t(fscale);
}

static Conv2dChannelParams computeConv2dChannelParams(
    const float multiplier, const int32_t output_zero_point) {
  // Conv2dChannelParams params;

  int16_t rshift, scale;
  compute_rshift_scale(rshift, scale, multiplier);

  constexpr int SHIFT_ADJUSTMENT = 7;
  constexpr int OUTPUT_BITS = 8;
  constexpr int MAX_POST_SHIFT = 22 + SHIFT_ADJUSTMENT - OUTPUT_BITS;

  int16_t shift_pre = std::max<int16_t>(rshift, 0);
  int16_t shift_post = MAX_POST_SHIFT + std::min<int16_t>(rshift, 0);

  assert(shift_post >= 0);  // TODO: Not ideal.

  float raw_offset = ldexpf(output_zero_point, shift_post + (OUTPUT_BITS - 8));

  int16_t offset_scale = int16_t(roundf(sqrtf(fabsf(raw_offset))));

  int16_t offset = 0;

  if (offset_scale != 0) offset = int16_t(roundf(raw_offset / offset_scale));

  return Conv2dChannelParams{.shift1 = shift_pre,
                             .scale = scale,
                             .offset_scale = offset_scale,
                             .offset = offset,
                             .shift2 = shift_post};
}

static int64_t sumConv2dWeights_Deep(const Filter2dGeometry &filter,
                                     const int8_t *kernel_weights,
                                     const unsigned output_channel) {
  // Each output channel has a full convolution window's worth of elements
  const unsigned weight_count = filter.window.shape.ElementCount();

  // Channel cout's weights start cout windows into the array
  const int8_t *cout_weights = kernel_weights + weight_count * output_channel;

  // Sum of weights
  int64_t acc = 0;

  for (int i = 0; i < weight_count; ++i) acc += cout_weights[i];

  return acc;
}

static int64_t sumConv2dWeights_Depthwise(const Filter2dGeometry &filter,
                                          const int8_t *kernel_weights,
                                          const unsigned output_channel) {
  // Each output channel has a full convolution window's worth of pixels (not
  // elements, because channels don't interact with a depthwise convolution)
  // const unsigned weight_count = filter.window.shape.PixelCount();

  // In a depthwise convolution, the output channel corresponds to the last axis
  // of the weight vector, not the first
  int64_t acc = 0;

  const auto weight_count = filter.window.shape.ElementCount();

  // Sum the weights
  for (int k = output_channel; k < weight_count; k += filter.output.depth)
    acc += kernel_weights[k];

  return acc;
}

static int64_t sumConv2dWeights(const Filter2dGeometry &filter,
                                const int8_t *kernel_weights,
                                const unsigned output_channel,
                                const bool is_depthwise) {
  return is_depthwise
             ? sumConv2dWeights_Depthwise(filter, kernel_weights,
                                          output_channel)
             : sumConv2dWeights_Deep(filter, kernel_weights, output_channel);
}

/**
 *  For non-depthwise Conv2d's the biases should be the original (e.g. TFlite)
 * biases plus the input zero-point multiplied by the sum of weights for that
 * output channel.
 *
 *  For depthwise Conv2d's the biases should be the original (e.g. TFlite)
 * biases plus the input zero-point multiplied by the sum of weights for that
 * input/output channel.
 *
 * TODO: Having an assertion in here if we saturate our numerical limits doesn't
 * seem ideal. Not sure what else to do.
 */
static int32_t computeConv2dBias(const Filter2dGeometry &filter,
                                 const int32_t bias_in[],
                                 const int8_t *kernel_weights,
                                 const int32_t input_zero_point,
                                 const unsigned output_channel,
                                 const bool is_depthwise) {
  // Each output channel has a full convolution window's worth of elements
  // const unsigned weight_count = filter.window.shape.ElementCount();

  // Sum of weights
  int64_t acc =
      sumConv2dWeights(filter, kernel_weights, output_channel, is_depthwise);

  // Multiply sum of weights by input zero-point
  acc *= input_zero_point;

  // Add in the original bias
  acc += bias_in[output_channel];

  // Make sure the accumulator is within int32_t range
  assert(acc >= ((int64_t)std::numeric_limits<int32_t>::min()));
  assert(acc <= ((int64_t)std::numeric_limits<int32_t>::max()));

  return int32_t(acc);
}

std::vector<vpu_split_acc32_t> conv2d::util::TfLiteConverter::ConvertBiases(
    const Filter2dGeometry &filter, const int8_t kernel_weights[],
    const int32_t biases_in[], const int32_t input_zero_point,
    const bool is_depthwise) {
  // kernel weights do not change.

  const unsigned cog_count = (filter.output.depth + VPU_INT8_ACC_PERIOD - 1) >>
                             VPU_INT8_ACC_PERIOD_LOG2;

  std::vector<vpu_split_acc32_t> biases_out(cog_count);

  for (unsigned cout = 0; cout < filter.output.depth; ++cout) {
    const unsigned cog = cout >> VPU_INT8_ACC_PERIOD_LOG2;
    const unsigned coff = cout - (cog << VPU_INT8_ACC_PERIOD_LOG2);

    int32_t bias = computeConv2dBias(filter, biases_in, kernel_weights,
                                     input_zero_point, cout, is_depthwise);

    int16_t bias_high = int16_t(bias >> 16);
    uint16_t bias_low = uint16_t(bias & 0xFFFF);

    biases_out[cog].high[coff] = bias_high;
    biases_out[cog].low[coff] = bias_low;
  }

  return biases_out;
}

std::vector<nn_acc32_to_int8_params_t>
conv2d::util::TfLiteConverter::ConvertOutputParams(
    const Filter2dGeometry &filter, const float effective_output_multiplier[],
    const int32_t output_zero_point) {
  // kernel weights do not change.

  const unsigned cog_count = (filter.output.depth + VPU_INT8_ACC_PERIOD - 1) >>
                             VPU_INT8_ACC_PERIOD_LOG2;

  std::vector<nn_acc32_to_int8_params_t> params_out(cog_count);

  for (unsigned cout = 0; cout < filter.output.depth; ++cout) {
    const unsigned cog = cout >> VPU_INT8_ACC_PERIOD_LOG2;
    const unsigned coff = cout - (cog << VPU_INT8_ACC_PERIOD_LOG2);

    Conv2dChannelParams params = computeConv2dChannelParams(
        effective_output_multiplier[cout], output_zero_point);

    params_out[cog].shift1[coff] = params.shift1;
    params_out[cog].scale[coff] = params.scale;
    params_out[cog].offset_scale[coff] = params.offset_scale;
    params_out[cog].offset[coff] = params.offset;
    params_out[cog].shift2[coff] = params.shift2;
  }

  return params_out;
}

void conv2d::util::TfLiteConverter::QuantizeEffectiveOutputMultiplier(
    int32_t &mantissa, int32_t &exponent, const double effective_multiplier) {
  if (effective_multiplier == 0.0f) {
    mantissa = exponent = 0;
    return;
  }

  const double x = std::frexp(effective_multiplier, &((int &)exponent));
  int64_t y = static_cast<int64_t>(round(std::ldexp(x, 31)));

  assert(y <= std::ldexp(1, 31));

  if (y == std::ldexp(1, 31)) {
    y /= 2;
    ++exponent;
  }

  if (exponent < -31) {
    exponent = 0;
    y = 0;
  }
  mantissa = static_cast<int32_t>(y);
}
