#ifndef LIB_NN_OUTPUT_TRANSFORM_FN_H_
#define LIB_NN_OUTPUT_TRANSFORM_FN_H_

#include <cmath>
#include <cstdint>
#include <limits>
#include <vector>

#include "Serialisable.hpp"
#include "Utils.hpp"
#include "geom/WindowGeometry.hpp"
#include "vpu.hpp"
#include "AggregateFn.hpp"

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
  std::vector<int16_t> multipliers_and_biases;
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
      int output_zero_point, int output_channels) {
    MulsAndBias canonical_values;

    int elements_per_channel = weights.size() / output_channels;
    assert((weights.size() % output_channels) == 0);

    for (int out_chan = 0; out_chan < output_channels; out_chan++) {
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
    int16_t clamp_near;
    int16_t clamp_far_0;
    int16_t clamp_far_1;

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
  OT_int8_clamped(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  void setMultipliersAndBiases(int16_t *m) {
    multipliers_and_biases = m;
    assert(m != nullptr);
    assert(is_aligned(multipliers_and_biases, 4));
  }
};


/**
 * @brief Output Transform class to converting 32 bit accumulators to a 1 bit
 * output space.
 *
 */

typedef int16_t threshold_t;
class OT_binary : public OutputTransformFn {

 public:
  class Params : public Serialisable {
   public:
    int32_t output_slice_channel_count;

   public:
    /**
     * @brief Construct a new Params object
     *
     * @param output_slice_channel_count The count of output channels to be
     * computed by this parameter set.
     */
    Params(int32_t output_slice_channel_count)
        : output_slice_channel_count(output_slice_channel_count) {}
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;
  threshold_t *thresholds;

 public:
  OT_binary(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  static std::vector<threshold_t> 
    adjust_thresholds(std::vector<int32_t>& thresholds, int input_channels, WindowGeometry &K, Conv2dReorderedWeights& reordered_weights){
      // std::cerr << "thresholds.size(): " <<thresholds.size()<< std::endl; 
      std::vector<threshold_t> adjusted_thresholds(thresholds.size());

      // Larq assumes xor-popcount is used for the aggregation but the xcore uses
      // sum(xor * 2 - 1)/2. Over a receptive volume this means 
      // sum(xor * 2 - 1)/2 = xor-popcount - receptive_field/2
      // or
      // xor-popcount = sum(xor * 2 - 1)/2 + receptive_field/2

      // std::cerr << K << std::endl;
      int receptive_field = input_channels * K.shape.width * K.shape.height;

      // printf("receptive_field: %d\n", receptive_field);
      int receptive_bytes = receptive_field/8;
      
      // std::cerr << "receptive_bytes: " <<receptive_bytes<< " " << K.shape <<std::endl;

      //the number of useful bytes loaded on the final load of the kernel
      int final_load_bytes = (receptive_bytes%32);
      if (final_load_bytes == 0)
        final_load_bytes = 32;

      //the number of useless bytes loaded on the final load of the kernel
      // int final_overload_bytes = 32 - final_load_bytes;
      // std::cerr << "final_load_bytes: " <<final_load_bytes <<std::endl; 
      // std::cerr << "final_overload_bytes: " <<final_overload_bytes <<std::endl; 

      assert(final_load_bytes > 0);
      for (int ch=0;ch<thresholds.size();++ch){

        int final_vpu_load_address = reordered_weights.final_vpu_load_addresses[ch];
        int8_t padding_byte = 0;
        int acc = 0;
        for (int i = final_load_bytes; i < 32; i++) {
          int8_t b = reordered_weights.weights[final_vpu_load_address+i];
          
          int8_t v = (padding_byte ^ b);
          int t= ((2 * __builtin_popcount((~v)&0xff) - CHAR_BIT) / 2);
          // printf("%2x %2x -> %d\n", b, v, t);
          acc += t;
        }
        // printf("ch %d final_vpu_load_addresses: %d  acc: %d\n", ch, final_vpu_load_address, acc);

        adjusted_thresholds[ch] = thresholds[ch] - receptive_field/2 - acc;
        // printf("%d %d -> %d\n",ch, thresholds[ch], adjusted_thresholds[ch]);
      }
      return adjusted_thresholds;
    }

  void setThresholds(threshold_t *th) {
    thresholds = th;
    assert(th != nullptr);
    assert(is_aligned(thresholds, 4));
  }
};









/**
 * This output transform assumes the int8_t channel data is in vR[] of the
 * accumulator.
 */
class DirectWriteOutputTransform
    : public OutputTransformFn,
      public ChannelParallelComponent<VPU_INT8_EPV_LOG2> {
 public:
  /**
   * Configuration parameters for DirectWriteOutputTransform
   */
  struct Params {
    /**
     * The number of channels in the filter's output image.
     *
     * This is required to determine the number of channels to write when
     * output_transform_fn() is called.
     */
    int32_t output_img_channels;

    /**
     * Create a DirectWriteOutputTransform::Params for an output image with the
     * specified number of channels.
     */
    Params(const int image_channels);

    /**
     * Create a DirectWriteOutputTransform::Params corresponding to a particular
     * output image geometry.
     */
    Params(const ImageGeometry &output_geometry);
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  const Params *params;

 public:
  /**
   * Construct a DirectWriteOutputTransform using the specified params.
   */
  DirectWriteOutputTransform(const Params *params);

  /**
   * Apply this output transform to the provided accumulators and write them to
   * the provided output image.
   */
  virtual int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *acc,
                                      int32_t output_channel_group) override;
};














/**
 * This output transform applies a per-channel, rounding, saturating right-shift
 * to the 32-bit accumulators to get 8-bit results.
 */
class ShiftInt8OutputTransform
    : public OutputTransformFn,
      public ChannelParallelComponent<VPU_INT8_ACC_PERIOD_LOG2> {
 public:
  /**
   * Configuration parameters for ShiftInt8OutputTransform
   */
  struct Params {
    /**
     * The number of channels in the filter's output image.
     *
     * This is required to determine the number of channels to write when
     * output_transform_fn() is called.
     */
    int32_t output_img_channels;

    /**
     * The per-output-channel arithmetic right-shifts to be applied to the
     * accumulators.
     */
    int16_t shifts[VPU_INT8_ACC_PERIOD];

    /**
     */
    Params() {}

    /**
     * Create a ShiftInt8OutputTransform::Params
     */
    Params(const int output_image_channels, const int16_t shift);

    /**
     * Create a ShiftInt8OutputTransform::Params
     */
    Params(const ImageGeometry &output_image, const int16_t shift);
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  const Params *params;

 public:
  /**
   * Construct a ShiftInt8OutputTransform using the specified params.
   */
  ShiftInt8OutputTransform(const Params *params);

  /**
   * Apply this output transform to the provided accumulators and write them to
   * the provided output image.
   */
  virtual int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *acc,
                                      int32_t output_channel_group) override;
};

/**
 *  xcore implementation of shift_int8_output_transform_ref()
 */
C_API
void shift_int8_output_transform_xcore(int8_t *output, const VPURingBuffer *acc,
                                       const int16_t *right_shifts,
                                       const int channel_count);

/**
 *  Portable implementation of shift_int8_output_transform_xcore()
 */
C_API
void shift_int8_output_transform_ref(int8_t *output, const VPURingBuffer *acc,
                                     const int16_t *right_shifts,
                                     const int channel_count);

}  // namespace nn
#endif  // LIB_NN_OUTPUT_TRANSFORM_FN_H_
