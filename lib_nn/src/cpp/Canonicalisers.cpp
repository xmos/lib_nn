#include "OutputTransformFn.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <tuple>

using namespace nn;
// using namespace nn::OutputTransformFn;

OutputTransformFn::MulsAndBias OutputTransformFn::canonicaliseConv2DDepthwise(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, const std::array<int, 4> &shape,
      int input_zero_point, int output_zero_point, int output_channels) {
    OutputTransformFn::MulsAndBias canonical_values;

    assert(shape[0] == 1);

    int elements_per_channel = weights.size() / output_channels;
    assert((weights.size() % output_channels) == 0);

    for (int out_chan = 0; out_chan < output_channels; out_chan++) {
      int32_t max_accu_sum = 0;
      int32_t min_accu_sum = 0;

      int32_t coefs_sum = 0;

      for (int e = 0; e < elements_per_channel; e++) {
        int idx = out_chan + output_channels * e;

        int32_t coef = (int32_t)weights[idx];
        coefs_sum += coef;

        if (coef > 0) {
          max_accu_sum += coef * (int32_t)INT8_MAX;
          min_accu_sum += coef * (int32_t)INT8_MIN;
        } else {
          max_accu_sum += coef * (int32_t)INT8_MIN;
          min_accu_sum += coef * (int32_t)INT8_MAX;
        }
      }

      double canonical_bias =
          (bias[out_chan] - input_zero_point * coefs_sum) * eff_mult[out_chan] +
          output_zero_point;

      OutputTransformFn::ActivationParams a(canonical_bias, eff_mult[out_chan],
                                            min_accu_sum, max_accu_sum);
      canonical_values.push_back(a);
    }
    return canonical_values;
  }

OutputTransformFn::MulsAndBias OutputTransformFn::canonicaliseConv2D(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, int input_zero_point,
      int output_zero_point, int output_channel_count) {
    OutputTransformFn::MulsAndBias canonical_values;

    int elements_per_channel = weights.size() / output_channel_count;
    assert((weights.size() % output_channel_count) == 0);

    for (int out_chan = 0; out_chan < output_channel_count; out_chan++) {
      int32_t max_accu_sum = 0;
      int32_t min_accu_sum = 0;

      int32_t coefs_sum = 0;
      for (int e = 0; e < elements_per_channel; e++) {
        int idx = out_chan * elements_per_channel + e;
        int32_t coef = (int32_t)weights[idx];
        coefs_sum += coef;

        if (coef > 0) {
          max_accu_sum += coef * (int32_t)INT8_MAX;
          min_accu_sum += coef * (int32_t)INT8_MIN;
        } else {
          max_accu_sum += coef * (int32_t)INT8_MIN;
          min_accu_sum += coef * (int32_t)INT8_MAX;
        }
      }

      double canonical_bias =
          (bias[out_chan] - input_zero_point * coefs_sum) * eff_mult[out_chan] +
          output_zero_point;

      OutputTransformFn::ActivationParams a(canonical_bias, eff_mult[out_chan],
                                            min_accu_sum, max_accu_sum);
      canonical_values.push_back(a);
    }
    return canonical_values;
  }

OutputTransformFn::MulsAndBias OutputTransformFn::canonicaliseConv2DClamped(
      const std::vector<float> &post_activation_multiplier,
      const std::vector<float> &post_activation_bias, int receptive_volume,
      int32_t clamp_low, int32_t clamp_high, int output_channel_count) 
{
    OutputTransformFn::MulsAndBias canonical_values;

    int32_t xcore_clamp_low = clamp_low - receptive_volume / 2;
    int32_t xcore_clamp_high = clamp_high - receptive_volume / 2;

    /*
        Larq assumes xor-popcount is used for the aggregation but the xcore uses
        sum(xor * 2 - 1)/2. Over a receptive volume this means
        sum(xor * 2 - 1)/2 = xor-popcount - receptive_field/2
        or
        xor-popcount = sum(xor * 2 - 1)/2 + receptive_field/2

        We are implementing:

                std::int32_t x = accum << 1;
                x = std::max<std::int32_t>(std::min<std::int32_t>(x, clamp_max),
       clamp_min);
                // The linear transformation is done in float
                float y =
                    static_cast<float>(x) * multiplier[out_channel] +
       bias[out_channel];
                // And then we round back to int32 and clamp to the int8 range
                return saturate(round(y));

        Which is why we have a spurious x3 below.
    */
    for (int out_chan = 0; out_chan < output_channel_count; out_chan++) {
      float xcore_multiplier = -post_activation_multiplier[out_chan] * 2;

      // Here we add on M*R/2 to account for the way xcore computes the macc as
      // opposed to xor-popcount.
      float xcore_bias =
          post_activation_bias[out_chan] +
          post_activation_multiplier[out_chan] * receptive_volume;

      // This is considering the perspective of the accumulator after a VPOS has
      // been performed on it
      OutputTransformFn::ActivationParams a(xcore_bias, xcore_multiplier,
                                            xcore_clamp_low, xcore_clamp_high);

      canonical_values.push_back(a);
    }
    return canonical_values;
}