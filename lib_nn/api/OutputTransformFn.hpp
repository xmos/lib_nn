#ifndef LIB_NN_OUTPUT_TRANSFORM_FN_H_
#define LIB_NN_OUTPUT_TRANSFORM_FN_H_

#include <cstdint>
#include <vector>

#include "Utils.hpp"
#include "geom/ImageGeometry.hpp"
#include "vpu.hpp"

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

/**
 * These are in a protected order(internally)
 */
struct OutputTransformValues {
  int16_t bias_multipler[VPU_INT16_EPV];  // The 16 bit bias will be multiplied
                                          // by this to make it a 32 bit value
  int16_t final_shr[VPU_INT16_EPV];
  int16_t accu_shr[VPU_INT16_EPV];  // for the vlsat
  int32_t accu_shl;                 // for the vlashr
};

/**
 * @brief When the output transform requires the raw accumulator value to be
 * clamped between a min and a max then this structure is used to provide that
 * functionality. The clamping is implemented by taking the accumulator (32 bit)
 * and after shifting it into a 16 bit space
 *    - adding a scalar, near, to the accumulator,
 *    - subtracting near from the accumulator,
 *    - adding a scalar, far_0, to the accumulator,
 *    - adding a scalar, far_1, to the accumulator,
 *    - subtracting near far_1 the accumulator,
 *    - subtracting near far_0 the accumulator,
 * Of these three values near represents the scalar that moves (and returns) the
 * accumulator to it's nearest saturation point, i.e. INT16_MAX, and far_0 +
 * far_1 moves (and returns) the accumulator to the farthest saturation point.
 */
struct OutputTransformValuesClamping : OutputTransformValues {
  int16_t clamp_near[VPU_INT16_EPV];
  int16_t clamp_far_0[VPU_INT16_EPV];
  int16_t clamp_far_1[VPU_INT16_EPV];
};

struct QuantisationParams {
  OutputTransformValues otv;
  std::vector<int16_t> biases;
  std::vector<int16_t> multipliers;
};

class OutputTransformFnInt8 : public OutputTransformFn {
 public:
  struct CanonicalMulAndBias {
    std::vector<double> f_biases;
    std::vector<double> f_multipliers;
    std::vector<int32_t> accu_min;
    std::vector<int32_t> accu_max;

    CanonicalMulAndBias(int output_channels)
        : f_biases(output_channels, 0),
          f_multipliers(output_channels, 0),
          accu_min(output_channels, 0),
          accu_max(output_channels, 0){};
  };

  static CanonicalMulAndBias canonicalise_mul_and_bias(
      std::vector<float> &eff_mult, std::vector<int32_t> &bias,
      std::vector<int8_t> &weights, int input_zero_point, int output_zero_point,
      int output_channels) {
    CanonicalMulAndBias canonical_values(output_channels);

    int elements_per_channel = weights.size() / output_channels;
    assert((weights.size() % output_channels) == 0);

    for (int out_chan = 0; out_chan < output_channels; out_chan++) {
      int32_t max_accu_sum = 0;
      int32_t min_accu_sum = 0;

      int32_t coefs_sum = 0;

      for (int e = 0; e < elements_per_channel; e++) {
        int32_t coef = (int32_t)weights[out_chan * elements_per_channel + e];
        coefs_sum += coef;

        if (coef > 0) {
          max_accu_sum += coef * (int32_t)INT8_MAX;
          min_accu_sum += coef * (int32_t)INT8_MIN;
        } else {
          max_accu_sum += coef * (int32_t)INT8_MIN;
          min_accu_sum += coef * (int32_t)INT8_MAX;
        }
      }

      canonical_values.f_biases[out_chan] =
          (bias[out_chan] - input_zero_point * coefs_sum) * eff_mult[out_chan] +
          output_zero_point;
      canonical_values.f_multipliers[out_chan] = eff_mult[out_chan];

      canonical_values.accu_min[out_chan] = min_accu_sum;
      canonical_values.accu_max[out_chan] = max_accu_sum;
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
  static QuantisationParams quantise_activation(
      std::vector<double> &output_transform_multiplier,
      std::vector<double> &output_transform_bias,
      std::vector<int32_t> &accu_min, std::vector<int32_t> &accu_max);
};

/**
 * @brief Output Transform class to converting 32 bit accumulators to an 8 bit
 * output space.
 *
 */
class OT_int8 : public OutputTransformFnInt8 {
 public:
  struct Params {
    int32_t output_slice_channel_count;
    OutputTransformValues *otv;
    int16_t *biases;
    int16_t *multipliers;

    /**
     * @brief Construct a new Params object
     *
     * @param output_slice_channel_count The count of output channels to be
     * computed by this parameter set.
     * @param otv Pointer to struct defining how the VPU will implement the the
     * output transform.
     * @param biases Pointer to the quantised biases.
     * @param multipliers Pointer to the quantised multipliers.
     */
    Params(int32_t output_slice_channel_count, OutputTransformValues *otv,
           int16_t *biases, int16_t *multipliers)
        : output_slice_channel_count(output_slice_channel_count),
          otv(otv),
          biases(biases),
          multipliers(multipliers) {}
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;

 public:
  OT_int8(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);
};

class OTBinary_int8 : public OutputTransformFnInt8 {
 public:
  class Params {
   public:
    int32_t output_slice_channel_count;  // TODO push into base class
    OutputTransformValuesClamping *otv;
    int16_t *biases;         //[output_slice_channel_count];
    int16_t *multipliers;    //[output_slice_channel_count];
    int16_t *accu_modifier;  //[output_slice_channel_count];

    /**
     * @brief Construct a new Params object
     *
     * @param output_slice_channel_count The count of output channels to be
     * computed by this parameter set.
     * @param otv Pointer to struct defining how the VPU will implement the the
     * output transform.
     * @param biases Pointer to the quantised biases.
     * @param multipliers Pointer to the quantised multipliers.
     * @param accu_modifier Pointer to a per channel accumulator modifier. This
     * adjusts each channel by a fixed amount to compensate for channel overlap,
     * allowing for dense weight packing.
     */
    Params(int32_t output_slice_channel_count,
           OutputTransformValuesClamping *otv, int16_t *biases,
           int16_t *multipliers, int16_t *accu_modifier);
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;

 public:
  OTBinary_int8(Params *params) : params(params){};

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
  static QuantisationParams quantise_activation(
      std::vector<double> &output_transform_multiplier,
      std::vector<double> &output_transform_bias,
      std::vector<int32_t> &accu_min, std::vector<int32_t> &accu_max);

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);
};

class OTBinary_bin : public OutputTransformFn {
  int16_t *thresholds;

 public:
  OTBinary_bin(int16_t *thresholds);

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);
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
    Params(const nn::ImageGeometry &output_geometry);

    /**
     * Deserialize a DirectWriteOutputTransform::Params from a stream.
     *
     * The data to be deserialized should come from a previous call to
     * DirectWriteOutputTransform::Params::Serialize(). The serialized data
     * format is considered to be opaque.
     */
    Params(std::istream &stream);

    /**
     * Serialize a DirectWriteOutputTransform::Params into a stream.
     *
     * The serialized object can be recovered later using the appropriate stream
     * constructor.
     */
    void Serialize(std::ostream &stream) const;
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  const Params *params;

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
    Params(const nn::ImageGeometry &output_image, const int16_t shift);

    /**
     * Deseriaized a ShiftInt8OutputTransform::Params from a byte stream.
     *
     * The data in the stream should come from a prior call to
     * ShiftInt8OutputTransform::Params::Serialize().
     */
    Params(std::istream &stream);

    /**
     * Serialize this ShiftInt8OutputTransform::Params into a byte stream.
     *
     * Note: This does not serialize the shift values.
     */
    void Serialize(std::ostream &stream) const;
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  const Params *params;

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
