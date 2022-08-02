#ifndef LIB_NN_OUTPUT_TRANSFORM_FN_H_
#define LIB_NN_OUTPUT_TRANSFORM_FN_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "AggregateFn.hpp"
#include "Serialisable.hpp"
#include "Utils.hpp"
#include "geom/WindowGeometry.hpp"
#include "vpu.hpp"

#ifdef _MSC_VER
#  include <intrin.h>
#  define __builtin_popcount __popcnt
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

    double bias;
    double multiplier;
    int32_t accu_min_val;
    int32_t accu_max_val;

    ActivationParams(double bias, double multiplier, int32_t accu_min_val,
                     int32_t accu_max_val)
        : original_bias(bias),
          original_multiplier(multiplier),
          original_accu_min_val(accu_min_val),
          original_accu_max_val(accu_max_val),
          bias(bias),
          multiplier(multiplier),
          accu_min_val(accu_min_val),
          accu_max_val(accu_max_val) {
      assert(accu_min_val <= accu_max_val);
    }
  };

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

    // printf("output_channel_groups %d first_array.size():%d\n",
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

  /**
   * @brief The method that will translate the accumulator into the output
   * number space.
   *
   * @param Y Pointer to output tensor.
   * @param A Pointer to a copy of the VPU ring buffer. This is where the output
   * of an aggregate fn will be stored.
   * @param output_channel_group Denotes which channel group will be computed.
   * @return int8_t*
   */
  virtual int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                                      int32_t output_channel_group) = 0;
};

typedef std::vector<OutputTransformFn::ActivationParams> MulsAndBias;

struct QuantisationParams {
  /**
   * The amount to shift all the 32 bit accumulators right by to reduce them to
   * 16 bit scalars. This will be non-negative. It is used to control the VLSAT.
   * Note, some may saturate in the 16 bit conversion as the output clamp may
   * have been back propagated.
   */
  int16_t initial_shr;

  /**
   * The amount to shift all the 16 bit biased and scaled accumulators right by
   * to reduce them to 8 bit scalars. It is used to control the VLASHR. Also the
   * result may be bigger than the int8 range as clamping is expected to follow.
   */
  int16_t final_shr;

  /**
   * The mutipliers and biases are interleaved into a single array. They are
   * arranged as channel groups of 16 multipliers and 16 biases until the final
   * group of N multipliers and N biases where N is the remaining number of
   * channels after all the full channel groups.
   */
  std::vector<int16_t> multipliers;
  std::vector<int16_t> biases;
};

class OutputTransformFnInt8 : public OutputTransformFn {
 public:
  static MulsAndBias canonicalise_mul_and_bias_dw(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, const std::array<int, 4> &shape,
      int input_zero_point, int output_zero_point, int output_channels) {
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
                                            min_accu_sum, max_accu_sum);
      canonical_values.push_back(a);
    }
    return canonical_values;
  }

  static MulsAndBias canonicalise_mul_and_bias(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, int input_zero_point,
      int output_zero_point, int output_channel_count) {
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
                                            min_accu_sum, max_accu_sum);
      canonical_values.push_back(a);
    }
    return canonical_values;
  }

  static int32_t sat(int64_t a, int bits){
    int64_t max_val = (1LL << bits)-1;
    int64_t min_val = -(1LL << bits);

    if(a > max_val)
      return (int32_t)max_val;

    if(a < min_val)
      return (int32_t)min_val;

    return a;
  }

  static int32_t shr(int32_t val, int shr_amount, int bits = 16 ){
    if (shr_amount > 0){
      return sat(((int64_t)val + (1LL << (shr_amount - 1))) >> shr_amount, bits);
    } else {
      return sat((int64_t)val << (-shr_amount), bits);
    }
  }

  static int32_t add(int32_t a, int32_t b, int bits = 16 ){
      return sat((int64_t)a + (int64_t)b, bits);
  }

  static int32_t mul(int32_t a, int32_t b, int bits = 16 ){
      int64_t prod = (int64_t)a * (int64_t)b;
      prod = prod + (1LL << (14-1));
      return sat(prod>>14, bits);
  }

  /**
   * Calculate the maximum average error between the reference and quantised implementations of the output 
   * transform over each channel. The average is defined over the range of non-saturating accumulators, 
   * i.e. accumulators that do not reach a saturating output in the int8 space. 
   * The result is the maximum average for all of the channels. 
   */
  static double get_quant_error(MulsAndBias &mul_and_bias, QuantisationParams &qp, bool use_high_precision = false){
    
    if (use_high_precision){

      double max_avg_abs_error = 0.0;

      for (int idx=0; idx < mul_and_bias.size(); ++idx){
        
        int64_t abs_error_sum = 0;

        for(int accu = mul_and_bias[idx].accu_min_val; accu <= mul_and_bias[idx].accu_max_val; ++accu){

          int32_t t = shr(accu, qp.initial_shr); //vlsat
          t = mul(t, qp.multipliers[idx]);       //vlmul
          t = add(t, qp.biases[idx]);            //vladd
          t = shr(t, qp.final_shr);              //vlashr
          t = sat(shr(t, 8), 8);                 //vdepth8
          
          double v = (double)accu * mul_and_bias[idx].multiplier + mul_and_bias[idx].bias;

          int expected = (int)std::round(v);

          expected = std::min((int)expected, (int)INT8_MAX);
          expected = std::max((int)expected, (int)INT8_MIN);

          abs_error_sum += std::abs(expected - t);
        }

        int64_t interesting_accumulators =  mul_and_bias[idx].accu_max_val - mul_and_bias[idx].accu_min_val + 1;
        
        if (interesting_accumulators > 0){
          double avg_abs_error = (double)abs_error_sum / (double)interesting_accumulators;
          max_avg_abs_error = std::max(max_avg_abs_error, avg_abs_error);
        }
      }
      return max_avg_abs_error;
    } else {
      //final_shr | number of decimal places | error
      //-8          0                          2   = 1*2^1
      //-7          1                          1   = 1*2^0
      //-6          2                          0.5 = 1*2^-1
      //-5          3                          2^-2
      //-4          4                          2^-3
      //-3          5                          2^-4
      //-2          6                          2^-5
      //-1          7                          2^-6
      //0           8                          2^-7
      return std::ldexp(1, -(qp.final_shr + 7));
    }
  }


  /**
   * @brief This translates from the representation of:
   *    output[ch] = min(max((accu[ch] * multipler[ch]) + bias[ch]), INT8_MIN),
   * INT8_MAX) to a form that can be efficiently implemented on the VPU. The
   * accu_min and accu_max allow the quantisiation logic to achieve maximum
   * resolution on the quantised representations of the multipler and bias.
   *
   * @param output_transform_multiplier Vector of the multipier for each
   * channel.
   * @param output_transform_bias Vector of the bias for each channel.
   * @param accu_min Vector of the minimum possible accumulator for each
   * channel.
   * @param accu_max Vector of the maximum possible accumulator for each
   * channel.
   * @return QuantisationParams
   */
  static QuantisationParams quantise_activation(MulsAndBias &activation_params,
                                                bool verbose = false);
};

/**
 * @brief Output Transform class to converting 32 bit accumulators to an 8 bit
 * output space.
 *
 */
class OT_int8 : public OutputTransformFnInt8 {
 public:
  class Params : public Serialisable {
   public:
    int32_t output_slice_channel_count;
    int16_t initial_shift;
    int16_t final_shr;

   public:
    /**
     * @brief Construct a new Params object
     *
     * @param output_slice_channel_count The count of output channels to be
     * computed by this parameter set.
     */
    Params(int32_t output_slice_channel_count, int16_t initial_shift,
           int16_t final_shr)
        : output_slice_channel_count(output_slice_channel_count),
          initial_shift(initial_shift),
          final_shr(final_shr) {}
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;
  int16_t *multipliers_and_biases;

 public:
  OT_int8(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  void setMultipliersAndBiases(int16_t *m) {
    multipliers_and_biases = m;
    assert(m != nullptr);
    assert(is_aligned(multipliers_and_biases, 4));
  }
};

/**
 * @brief Output Transform class to converting 32 bit accumulators to an 8 bit
 * output space.
 *
 */
class OT_int8_clamped : public OutputTransformFnInt8 {
 public:
  class Params : public Serialisable {
   public:
    int32_t output_slice_channel_count;
    int16_t initial_shift;
    int16_t final_shr;

   public:
    /**
     * @brief Construct a new Params object
     *
     * @param output_slice_channel_count The count of output channels to be
     * computed by this parameter set.
     */
    Params(int32_t output_slice_channel_count, int16_t initial_shift,
           int16_t final_shr)
        : output_slice_channel_count(output_slice_channel_count),
          initial_shift(initial_shift),
          final_shr(final_shr) {}
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;
  int16_t *offsets_multipliers_and_biases;

 public:
  OT_int8_clamped(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  void setOffsetsMultipliersAndBiases(int16_t *m) {
    offsets_multipliers_and_biases = m;
    assert(m != nullptr);
    assert(is_aligned(offsets_multipliers_and_biases, 4));
  }

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

/**
 * @brief Output Transform class to converting 32 bit accumulators to a 1 bit
 * output space.
 *
 */

typedef int16_t threshold_t;
class OT_binary : public OutputTransformFn {
 private:
  threshold_t *thresholds;

 public:
  OT_binary(){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  static std::vector<threshold_t> adjust_thresholds(
      const std::vector<int32_t> &thresholds, int input_channels, const WindowGeometry &K,
      Conv2dReorderedWeights &reordered_weights) {
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

  void setThresholds(threshold_t *th) {
    thresholds = th;
    assert(th != nullptr);
    assert(is_aligned(thresholds, 4));
  }
};

}  // namespace nn
#endif  // LIB_NN_OUTPUT_TRANSFORM_FN_H_
