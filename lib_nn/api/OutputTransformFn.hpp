#ifndef LIB_NN_OUTPUT_TRANSFORM_FN_H_
#define LIB_NN_OUTPUT_TRANSFORM_FN_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "AggregateFn.hpp"
#include "Utils.hpp"
#include "geom/WindowGeometry.hpp"
#include "vpu.hpp"

#if defined(_MSC_VER) && !defined(__clang__)
#include <intrin.h>
#define __builtin_popcount __popcnt
#endif

namespace nn {

/**
 * Interface implemented by all output transform handlers.
 *
 * Contains a single function, output_transform_fn() which converts a set of
 * 32-bit accumulators into a set of 8-bit outputs and writes them to the
 * specified output image. The 32 bit accumulators will be passed to this class
 * as a verbatim copy of the VPU ring buffer after the aggregate function has
 * been performed. The ring buffer will then be transformed into an output
 * number space and written to the output tensor (Y).
 */
class OutputTransformFn {
 public:
  class ActivationParams {
   public:
    /*
    The originals exist for further optimisation in the future
    */
    double original_bias;
    double original_multiplier;
    int32_t original_accu_min_val;
    int32_t original_accu_max_val;
    int8_t original_output_max_val;
    int8_t original_output_min_val;

    double bias;
    double multiplier;
    int32_t accu_min_val;
    int32_t accu_max_val;
    int8_t output_max_val;
    int8_t output_min_val;

    ActivationParams(double bias, double multiplier, int32_t accu_min_val,
                     int32_t accu_max_val, bool verbose = false)
        : original_bias(bias),
          original_multiplier(multiplier),
          original_accu_min_val(accu_min_val),
          original_accu_max_val(accu_max_val),
          original_output_max_val(INT8_MAX),
          original_output_min_val(INT8_MIN),
          bias(bias),
          multiplier(multiplier),
          accu_min_val(accu_min_val),
          accu_max_val(accu_max_val),
          output_max_val(INT8_MAX),
          output_min_val(INT8_MIN) {
      backprop_output_clamps_to_accu_limits(verbose);
      assert(accu_min_val <= accu_max_val);
    }

   private:
    void backprop_output_clamps_to_accu_limits(bool verbose = false, bool debug = false);
  };

  static void layerwise_stats(std::vector<OutputTransformFn::ActivationParams> &canonical_values){
      double max_mul_log2 = -1e10;
      double min_mul_log2 = -max_mul_log2;
      double max_bias_log2 = -1e10;
      double min_bias_log2 = -max_bias_log2;

      for (auto act : canonical_values){
        if (act.bias != 0.0){
          double bias_log2 = std::log2(std::abs(act.bias));
          max_bias_log2 = std::max(max_bias_log2, bias_log2);
          min_bias_log2 = std::min(min_bias_log2, bias_log2);
        }
        if (act.multiplier != 0.0){
          double mul_log2 = std::log2(std::abs(act.multiplier));
          max_mul_log2 = std::max(max_mul_log2, mul_log2);
          min_mul_log2 = std::min(min_mul_log2, mul_log2);
        }
      }
      printf("mul dr: %.2f bias dr: %.2f\n",max_mul_log2 - min_mul_log2, max_bias_log2-min_bias_log2);
  }
  /**
   * Pads a vector to a boundary with a pad value
   */
  template <class T>
  static void pad(std::vector<T> &vec, int pad_boundary, T pad_val) {
    vec.resize(
        vec.size() + (pad_boundary - vec.size() % pad_boundary) % pad_boundary,
        pad_val);
  }
  /**
   * Pads a vector such that the final access of the vector has access_count
   * elements with a pad value. This is intended to be used with MulsAndBias
   * to pad the final access to a VPU word boundary.
   */
  template <class T>
  static void pad_final_access(std::vector<T> &vec, int access_count,
                               T pad_val) {
    // divide by two as there is always a mul for each bias
    int l = (access_count - (vec.size() / 2) % access_count) % access_count;
    vec.resize(vec.size() + l, pad_val);
  }

  template <class T>
  static std::vector<T> serialise_memory(
      std::vector<T> &first_array, std::vector<T> &second_array,
      int elements_per_group = VPU_INT16_EPV) {
    std::vector<T> serialised_memory;

    assert(first_array.size() == second_array.size());

    int output_channel_groups = first_array.size() / elements_per_group;

    for (int ocg = 0; ocg < output_channel_groups; ++ocg) {
      for (int ch = ocg * elements_per_group;
           ch < (ocg + 1) * elements_per_group; ++ch) {
        serialised_memory.push_back(first_array[ch]);
      }

      for (int ch = ocg * elements_per_group;
           ch < (ocg + 1) * elements_per_group; ++ch) {
        serialised_memory.push_back(second_array[ch]);
      }
    }

    for (int ch = output_channel_groups * elements_per_group;
         ch < first_array.size(); ++ch) {
      serialised_memory.push_back(first_array[ch]);
    }
    for (int ch = output_channel_groups * elements_per_group;
         ch < first_array.size(); ++ch) {
      serialised_memory.push_back(second_array[ch]);
    }
    return serialised_memory;
  }

  template <class T>
  static std::vector<T> serialise_memory(
      std::vector<T> &first_array, std::vector<T> &second_array,
      std::vector<T> &third_array, int elements_per_group = VPU_INT16_EPV) {
    std::vector<T> serialised_memory;

    assert(first_array.size() == second_array.size());
    assert(first_array.size() == third_array.size());

    int output_channel_groups = first_array.size() / elements_per_group;

    // output_channel_groups, first_array.size());
    for (int ocg = 0; ocg < output_channel_groups; ++ocg) {
      for (int ch = ocg * elements_per_group;
           ch < (ocg + 1) * elements_per_group; ++ch) {
        serialised_memory.push_back(first_array[ch]);
      }

      for (int ch = ocg * elements_per_group;
           ch < (ocg + 1) * elements_per_group; ++ch) {
        serialised_memory.push_back(second_array[ch]);
      }

      for (int ch = ocg * elements_per_group;
           ch < (ocg + 1) * elements_per_group; ++ch) {
        serialised_memory.push_back(third_array[ch]);
      }
    }

    for (int ch = output_channel_groups * elements_per_group;
         ch < first_array.size(); ++ch) {
      serialised_memory.push_back(first_array[ch]);
    }
    for (int ch = output_channel_groups * elements_per_group;
         ch < second_array.size(); ++ch) {
      serialised_memory.push_back(second_array[ch]);
    }
    for (int ch = output_channel_groups * elements_per_group;
         ch < third_array.size(); ++ch) {
      serialised_memory.push_back(third_array[ch]);
    }
    return serialised_memory;
  }

  template <class activationT>
  static int get_max_exponent(activationT f) {
    int e;
    std::frexp(f, &e);
    return e;
  }

  template <class activationT>
  static int get_max_exponent(std::vector<activationT> &arr) {
    int m = INT32_MIN;
    for (auto f : arr) m = std::max(m, get_max_exponent(f));
    return m;
  }
};

typedef std::vector<OutputTransformFn::ActivationParams> MulsAndBias;



class OutputTransformFnInt8 : public OutputTransformFn {
 public:
  struct QuantisationParams {
    /**
     * The amount to shift all the 32 bit accumulators right by to reduce them
     * to 16 bit scalars. This will be non-negative. It is used to control the
     * VLSAT. Note, some may saturate in the 16 bit conversion as the output
     * clamp may have been back propagated.
     */
    int16_t initial_shr;

    /**
     * The amount to shift all the 16 bit biased and scaled accumulators right
     * by to reduce them to 8 bit scalars. It is used to control the VLASHR.
     * Also the result may be bigger than the int8 range as clamping is expected
     * to follow.
     */
    int16_t final_shr;

    /**
     * The mutipliers and biases are interleaved into a single array. They are
     * arranged as channel groups of 16 multipliers and 16 biases until the
     * final group of N multipliers and N biases where N is the remaining number
     * of channels after all the full channel groups.
     */
    std::vector<int16_t> multipliers;
    std::vector<int16_t> biases;
  };

  static MulsAndBias canonicalise_mul_and_bias_dw(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, const std::array<int, 4> &shape,
      int input_zero_point, int output_zero_point, int output_channels, bool verbose = false) {
    MulsAndBias canonical_values;

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
                                            min_accu_sum, max_accu_sum, verbose);
      canonical_values.push_back(a);
    }
    return canonical_values;
  }

  static MulsAndBias canonicalise_mul_and_bias(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, int input_zero_point,
      int output_zero_point, int output_channel_count, bool verbose = false) {
    MulsAndBias canonical_values;

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
                                            min_accu_sum, max_accu_sum, verbose);
      canonical_values.push_back(a);
    }
    return canonical_values;
  }

  static int32_t sat(int64_t a, int bits) {
    int64_t max_val = (1LL << bits) - 1;
    int64_t min_val = -(1LL << bits);

    if (a > max_val) return (int32_t)max_val;

    if (a < min_val) return (int32_t)min_val;

    return a;
  }

  static int32_t shr(int32_t val, int shr_amount, int bits = 16) {
    if (shr_amount > 0) {
      return sat(((int64_t)val + (1LL << (shr_amount - 1))) >> shr_amount,
                 bits);
    } else {
      return sat((int64_t)val << (-shr_amount), bits);
    }
  }

  static int32_t add(int32_t a, int32_t b, int bits = 16) {
    return sat((int64_t)a + (int64_t)b, bits);
  }

  static int32_t mul(int32_t a, int32_t b, int bits = 16) {
    int64_t prod = (int64_t)a * (int64_t)b;
    prod = prod + (1LL << (14 - 1));
    return sat(prod >> 14, bits);
  }

  /**
   * Calculate the maximum average error between the reference and quantised
   * implementations of the output transform over each channel. The average is
   * defined over the range of non-saturating accumulators, i.e. accumulators
   * that do not reach a saturating output in the int8 space. The result is the
   * maximum average for all of the channels.
   */
  template <class QParams>
  static double get_quant_error(MulsAndBias &mul_and_bias, QParams &qp,
                                bool use_high_precision = false) {
    if (use_high_precision) {
      double max_avg_abs_error = 0.0;

      for (int idx = 0; idx < mul_and_bias.size(); ++idx) {
        int64_t abs_error_sum = 0;

        for (int accu = mul_and_bias[idx].accu_min_val;
             accu <= mul_and_bias[idx].accu_max_val; ++accu) {
          int32_t t = shr(accu, qp.initial_shr);  // vlsat
          t = mul(t, qp.multipliers[idx]);        // vlmul
          t = add(t, qp.biases[idx]);             // vladd
          t = shr(t, qp.final_shr);               // vlashr
          t = sat(shr(t, 8), 8);                  // vdepth8

          double v = (double)accu * mul_and_bias[idx].multiplier +
                     mul_and_bias[idx].bias;

          int expected = (int)std::round(v);

          expected = std::min((int)expected, (int)INT8_MAX);
          expected = std::max((int)expected, (int)INT8_MIN);

          abs_error_sum += std::abs(expected - t);
        }

        int64_t interesting_accumulators =
            mul_and_bias[idx].accu_max_val - mul_and_bias[idx].accu_min_val + 1;

        if (interesting_accumulators > 0) {
          double avg_abs_error =
              (double)abs_error_sum / (double)interesting_accumulators;
          max_avg_abs_error = std::max(max_avg_abs_error, avg_abs_error);
        }
      }
      return max_avg_abs_error;
    } else {
      // final_shr | number of decimal places | error
      //-8          0                          2   = 1*2^1
      //-7          1                          1   = 1*2^0
      //-6          2                          0.5 = 1*2^-1
      //-5          3                          2^-2
      //-4          4                          2^-3
      //-3          5                          2^-4
      //-2          6                          2^-5
      //-1          7                          2^-6
      // 0           8                          2^-7
      return std::ldexp(1, -(qp.final_shr + 7));
    }
  }
};

// Class containing quantisation strategy for the groupwise output transforms
class OutputTransformFnInt8_Group : public OutputTransformFnInt8 {
 public:
  class Quantizer {
   public:
    /**
     * @brief This translates from the representation of:
     *    output[ch] = min(max((accu[ch] * multipler[ch]) + bias[ch]),
     * INT8_MIN), INT8_MAX) to a form that can be efficiently implemented on the
     * VPU.
     * @param activationParams Set of multpliers and biases for each channel.
     * @return QuantisationParams
     */
    QuantisationParams quantise_activation(MulsAndBias &activationParams,
                                           bool verbose);

   private:
    std::tuple<int, int> solve_for_constraints(MulsAndBias &activationParams,
                                               int vlmul_shr,
                                               bool verbose = false);
  };
};

// Class containing quantisation strategy for channelwise output transforms
class OutputTransformFnInt8_Channelwise : public OutputTransformFnInt8 {
 public:
  struct QuantisationParams {
    /**
     * The amount to shift all the 32 bit accumulators right by to reduce them
     * to 16 bit scalars. This will be non-negative. It is used to control the
     * VLSAT. Note, some may saturate in the 16 bit conversion as the output
     * clamp may have been back propagated.
     */
    int16_t initial_shr;

    /**
     * The amount to shift all the 16 bit biased and scaled accumulators right
     * by to reduce them to 8 bit scalars. It is used to control the VLASHR.
     * Also the result may be bigger than the int8 range as clamping is expected
     * to follow.
     */
    int16_t final_shr;

    /**
     * The initial_shifts, mutipliers and biases are interleaved into a single
     * array. They are arranged as channel groups of 16 initial shifts, 16
     * multipliers and 16 biases until the final group of N initial shifts, N
     * multipliers and N biases where N is the remaining number of channels
     * after all the full channel groups.
     */
    std::vector<int16_t> initial_shifts;
    std::vector<int16_t> multipliers;
    std::vector<int16_t> biases;
  };

  /**
   * Calculate the maximum average error between the reference and quantised
   * implementations of the output transform over each channel. The average is
   * defined over the range of non-saturating accumulators, i.e. accumulators
   * that do not reach a saturating output in the int8 space. The result is the
   * maximum average for all of the channels.
   */
  template <class QParams>
  static double get_quant_error(MulsAndBias &mul_and_bias, QParams &qp,
                                bool use_high_precision = false) {
    if (use_high_precision) {
      double max_avg_abs_error = 0.0;

      for (int idx = 0; idx < mul_and_bias.size(); ++idx) {
        int64_t abs_error_sum = 0;

        for (int accu = mul_and_bias[idx].accu_min_val;
             accu <= mul_and_bias[idx].accu_max_val; ++accu) {
          int32_t t = shr(accu, qp.initial_shifts[idx]);  // vlsat
          t = mul(t, qp.multipliers[idx]);        // vlmul
          t = add(t, qp.biases[idx]);             // vladd
          t = shr(t, qp.final_shr);               // vlashr
          t = sat(shr(t, 8), 8);                  // vdepth8

          double v = (double)accu * mul_and_bias[idx].multiplier +
                     mul_and_bias[idx].bias;

          int expected = (int)std::round(v);

          expected = std::min((int)expected, (int)INT8_MAX);
          expected = std::max((int)expected, (int)INT8_MIN);

          abs_error_sum += std::abs(expected - t);
        }

        int64_t interesting_accumulators =
            mul_and_bias[idx].accu_max_val - mul_and_bias[idx].accu_min_val + 1;

        if (interesting_accumulators > 0) {
          double avg_abs_error =
              (double)abs_error_sum / (double)interesting_accumulators;
          max_avg_abs_error = std::max(max_avg_abs_error, avg_abs_error);
        }
      }
      return max_avg_abs_error;
    } else {
      // final_shr | number of decimal places | error
      //-8          0                          2   = 1*2^1
      //-7          1                          1   = 1*2^0
      //-6          2                          0.5 = 1*2^-1
      //-5          3                          2^-2
      //-4          4                          2^-3
      //-3          5                          2^-4
      //-2          6                          2^-5
      //-1          7                          2^-6
      // 0           8                          2^-7
      double abs_error_sum = mul_and_bias.size()*std::ldexp(1, -(qp.final_shr + 7));
      // for (int idx = 0; idx < mul_and_bias.size(); ++idx) {
      //   abs_error_sum -= std::ldexp(1, -(qp.initial_shifts[idx] +  7));
      // }
      return abs_error_sum/mul_and_bias.size();
    }
  }

  class Quantizer {
   public:
    /**
     * @brief This translates from the representation of:
     *    output[ch] = min(max((accu[ch] * multipler[ch]) + bias[ch]),
     * INT8_MIN), INT8_MAX) to a form that can be efficiently implemented on the
     * VPU. This version calculates an initial shift for each channel in
     * addition, increasing memory requirements but improves performance for
     * cases where channels have diverse dynamic ranges.
     * @param activationParams Set of initial shifts, multpliers and biases for
     * each channel.
     * @return QuantisationParams
     */
    QuantisationParams quantise_activation(MulsAndBias &activationParams,
                                           bool verbose);

   private:
    std::tuple<std::vector<int>, std::vector<int>> solve_for_constraints(
        MulsAndBias &activationParams, int vlmul_shr, bool verbose = false);
  };
};

struct otfn_int8_params_t{
    int32_t output_slice_channel_count;
    int16_t initial_shift;
    int16_t final_shr;
};

/**
 * @brief Output Transform class to converting 32 bit accumulators to an 8 bit
 * output space on a groupwise quantisation scheme.
 *
 */
class OT_int8 : public OutputTransformFnInt8_Group {
  private:
  otfn_int8_params_t p;
 public:
  OT_int8(int32_t output_slice_channel_count, int16_t initial_shift,
           int16_t final_shr) : p{output_slice_channel_count, initial_shift, final_shr} {}
  otfn_int8_params_t getParams() {return p;};
};

int8_t *otfn_int8(const otfn_int8_params_t *params, int8_t *Y,
                                 VPURingBuffer *A, int32_t output_channel_group,
                                 int16_t *multipliers_and_biases);


struct otfn_int8_channelwise_params_t{
    int32_t output_slice_channel_count;
    int16_t final_shr;
};

/**
 * @brief Output Transform class to converting 32 bit accumulators to an 8 bit
 * output space using floating point arithmetic (per channel).
 *
 */
class OT_int8_channelwise : public OutputTransformFnInt8_Channelwise {
  private:
  otfn_int8_channelwise_params_t p;

 public:
  OT_int8_channelwise(int32_t output_slice_channel_count, int16_t final_shr):
    p{output_slice_channel_count, final_shr} {}
  otfn_int8_channelwise_params_t getParams() {return p;};
};

int8_t *otfn_int8_channelwise(const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
                                                 int32_t output_channel_group, int16_t *multipliers_and_biases);
int8_t *otfn_int8_maxpool(const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
                                                 int32_t output_channel_group, int16_t *multipliers_and_biases);


struct otfn_int8_clamped_params_t{
    int32_t output_slice_channel_count;
    int16_t initial_shift;
    int16_t final_shr;
};

/**
 * @brief Output Transform class to converting 32 bit accumulators to an 8 bit
 * output space.
 *
 */
class OT_int8_clamped : public OutputTransformFnInt8_Group {
 private:
  otfn_int8_clamped_params_t p;
 public:
  OT_int8_clamped(int32_t output_slice_channel_count, int16_t initial_shift,
           int16_t final_shr)
        : p{output_slice_channel_count,
          initial_shift,
          final_shr} {}

  otfn_int8_clamped_params_t getParams() {return p;};

  static MulsAndBias canonicalise_mul_and_bias(
      const std::vector<float> &post_activation_multiplier,
      const std::vector<float> &post_activation_bias, int receptive_volume,
      int32_t clamp_low, int32_t clamp_high, int output_channel_count) {
    MulsAndBias canonical_values;

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
  static std::vector<int16_t> get_accumulator_overlaps(
      int receptive_volume, int output_channel_count,
      Conv2dReorderedWeights &reordered_weights) {
    int receptive_bytes = receptive_volume / CHAR_BIT;

    const int vpu_vector_byte_count = VPU_INT8_EPV;

    int final_load_bytes = (receptive_bytes % vpu_vector_byte_count);
    if (final_load_bytes == 0) final_load_bytes = vpu_vector_byte_count;

    std::vector<int16_t> accumulator_overlaps;
    for (int out_chan = 0; out_chan < output_channel_count; out_chan++) {
      int final_vpu_load_address =
          reordered_weights.final_vpu_load_addresses[out_chan];
      int8_t padding_byte = 0;
      int acc = 0;
      for (int i = final_load_bytes; i < vpu_vector_byte_count; i++) {
        int8_t b = reordered_weights.weights[final_vpu_load_address + i];

        int8_t v = (padding_byte ^ b);
        int t = ((2 * __builtin_popcount((~v) & 0xff) - CHAR_BIT) / 2);
        acc += t;
      }
      accumulator_overlaps.push_back(-acc);
    }
    return accumulator_overlaps;
  }
};

int8_t *otfn_int8_clamped(const otfn_int8_clamped_params_t *params, int8_t *Y, VPURingBuffer *A,
                                             int32_t output_channel_group, int16_t *offsets_multipliers_and_biases);

/**
 * @brief Output Transform class to converting 32 bit accumulators to a 1 bit
 * output space.
 *
 */

typedef int16_t threshold_t;
class OT_binary : public OutputTransformFn {
 
 public:
  OT_binary(){};

  static std::vector<threshold_t> adjust_thresholds(
      const std::vector<int32_t> &thresholds, int input_channels,
      const WindowGeometry &K, Conv2dReorderedWeights &reordered_weights) {
    std::vector<threshold_t> adjusted_thresholds(thresholds.size());

    const int vpu_vector_byte_count = VPU_INT8_EPV;

    // Larq assumes xor-popcount is used for the aggregation but the xcore uses
    // sum(xor * 2 - 1)/2. Over a receptive volume this means
    // sum(xor * 2 - 1)/2 = xor-popcount - receptive_field/2
    // or
    // xor-popcount = sum(xor * 2 - 1)/2 + receptive_field/2
    int receptive_field = input_channels * K.shape.width * K.shape.height;
    int receptive_bytes = receptive_field / CHAR_BIT;

    // the number of useful bytes loaded on the final load of the kernel
    int final_load_bytes = (receptive_bytes % vpu_vector_byte_count);
    if (final_load_bytes == 0) final_load_bytes = vpu_vector_byte_count;

    assert(final_load_bytes > 0);
    for (int ch = 0; ch < thresholds.size(); ++ch) {
      int final_vpu_load_address =
          reordered_weights.final_vpu_load_addresses[ch];
      int8_t padding_byte = 0;
      int acc = 0;
      for (int i = final_load_bytes; i < vpu_vector_byte_count; i++) {
        int8_t b = reordered_weights.weights[final_vpu_load_address + i];

        int8_t v = (padding_byte ^ b);
        int t = ((2 * __builtin_popcount((~v) & 0xff) - CHAR_BIT) / 2);
        acc += t;
      }

      adjusted_thresholds[ch] = thresholds[ch] - receptive_field / 2 - acc;
    }
    return adjusted_thresholds;
  }
};

int8_t *otfn_binary(void *p, int8_t *Y, VPURingBuffer *A,
                                       int32_t output_channel_group, int16_t *thresholds);

}  // namespace nn
#endif  // LIB_NN_OUTPUT_TRANSFORM_FN_H_
