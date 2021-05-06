#pragma once

#include <vector>

#include "Utils.hpp"
#include "xs3_vpu.h"
#include "vpu.hpp"

#include "geom/ImageGeometry.hpp"
namespace nn
{

  /**
 * Interface implemented by all output transform handlers.
 * 
 * Contains a single function, output_transform_fn() which converts a set of 32-bit accumulators
 * into a set of 8-bit outputs and writes them to the specified output image.
 */
  class OutputTransformFn
  {
  public:
    virtual int8_t *output_transform_fn(int8_t *Y,
                                        vpu_ring_buffer_t *A,
                                        int32_t output_channel_group) = 0;
  };

  /**
 * these are in a protected order(internally)
 */
  struct OutputTransformValues
  {
    int16_t bias_multipler[VPU_INT16_EPV];
    int16_t final_shr[VPU_INT16_EPV];
    int16_t accu_shr[VPU_INT16_EPV]; //for the vlsat
    int32_t accu_shl;                //for the vlashr
  };

  struct OutputTransformValuesClamping : OutputTransformValues
  {
    int16_t clamp_near[VPU_INT16_EPV];
    int16_t clamp_far_0[VPU_INT16_EPV];
    int16_t clamp_far_1[VPU_INT16_EPV];
  };

  /**
 * 
 */
  struct QuantisationParams
  {
    OutputTransformValues otv;
    std::vector<int16_t> biases;
    std::vector<int16_t> multipliers;
  };

  class OutputTransformFnInt8 : public OutputTransformFn
  {

  public:
    struct CanonicalMulAndBias
    {
      std::vector<double> f_biases;
      std::vector<double> f_multipliers;
      std::vector<int32_t> accu_min;
      std::vector<int32_t> accu_max;
      CanonicalMulAndBias(int output_channels) : f_biases(output_channels, 0),
                                                 f_multipliers(output_channels, 0),
                                                 accu_min(output_channels, 0),
                                                 accu_max(output_channels, 0){};
    };

    static CanonicalMulAndBias canonicalise_mul_and_bias(std::vector<float> &eff_mult,
                                                         std::vector<int32_t> &bias, std::vector<int8_t> &weights,
                                                         int input_zero_point, int output_zero_point, int output_channels)
    {

      CanonicalMulAndBias canonical_values(output_channels);

      int elements_per_channel = weights.size() / output_channels;
      assert((weights.size() % output_channels) == 0);

      for (int out_chan = 0; out_chan < output_channels; out_chan++)
      {
        int32_t max_accu_sum = 0;
        int32_t min_accu_sum = 0;

        int32_t coefs_sum = 0;

        for (int e = 0; e < elements_per_channel; e++)
        {
          int32_t coef = (int32_t)weights[out_chan * elements_per_channel + e];
          coefs_sum += coef;

          if (coef > 0)
          {
            max_accu_sum += coef * (int32_t)INT8_MAX;
            min_accu_sum += coef * (int32_t)INT8_MIN;
          }
          else
          {
            max_accu_sum += coef * (int32_t)INT8_MIN;
            min_accu_sum += coef * (int32_t)INT8_MAX;
          }
        }

        canonical_values.f_biases[out_chan] = (bias[out_chan] - input_zero_point * coefs_sum) * eff_mult[out_chan] + output_zero_point;
        canonical_values.f_multipliers[out_chan] = eff_mult[out_chan];

        canonical_values.accu_min[out_chan] = min_accu_sum;
        canonical_values.accu_max[out_chan] = max_accu_sum;
      }
      return canonical_values;
    }

    static QuantisationParams quantise_activation(std::vector<double> &output_transform_multiplier,
                                                  std::vector<double> &output_transform_bias,
                                                  std::vector<int32_t> &accu_min,
                                                  std::vector<int32_t> &accu_max);
  };

  /**
 * 
 */
  class OT_int8 : public OutputTransformFnInt8
  {
  public:
    struct Params
    {
      int32_t output_slice_channel_count; //TODO push into base class
      OutputTransformValues *otv;
      int16_t *biases;      //[output_slice_channel_count];
      int16_t *multipliers; //[output_slice_channel_count];

      Params(int32_t output_slice_channel_count, OutputTransformValues *otv,
             int16_t *biases, int16_t *multipliers) : output_slice_channel_count(output_slice_channel_count),
                                                      otv(otv),
                                                      biases(biases),
                                                      multipliers(multipliers) {}
    };

  private:
    Params *params;

  public:
    OT_int8(Params *params) : params(params){};

    int8_t *output_transform_fn(int8_t *Y, vpu_ring_buffer_t *A, int32_t output_channel_group);
  };

  class OTBinary_int8 : public OutputTransformFnInt8
  {
  public:
    class Params
    {
    public:
      int32_t output_slice_channel_count; //TODO push into base class
      OutputTransformValuesClamping *otv;
      int16_t *biases;        //[output_slice_channel_count];
      int16_t *multipliers;   //[output_slice_channel_count];
      int16_t *accu_modifier; //[output_slice_channel_count];

      Params(int32_t output_slice_channel_count, OutputTransformValuesClamping *otv,
             int16_t *biases, int16_t *multipliers, int16_t *accu_modifier);
    };

    Params *params;

  public:
    OTBinary_int8(Params *params) : params(params){};

    static QuantisationParams quantise_activation(std::vector<double> &output_transform_multiplier,
                                                  std::vector<double> &output_transform_bias,
                                                  std::vector<int32_t> &accu_min,
                                                  std::vector<int32_t> &accu_max);

    int8_t *output_transform_fn(int8_t *Y, vpu_ring_buffer_t *A, int32_t output_channel_group);
  };

  class OTBinary_bin : public OutputTransformFn
  {

    int16_t *thresholds;

  public:
    OTBinary_bin(int16_t *thresholds);

    int8_t *output_transform_fn(int8_t *Y, vpu_ring_buffer_t *A, int32_t output_channel_group);
  };

  /**
 * This output transform assumes the int8_t channel data is in vR[] of the accumulator.
 */
  class DirectWriteOutputTransform : public OutputTransformFn,
                                     public ChannelParallelComponent<VPU_INT8_EPV_LOG2>
  {
  public:
    /**
     * Configuration parameters for DirectWriteOutputTransform
     */
    struct Params
    {
      /**
       * The number of channels in the filter's output image.
       * 
       * This is required to determine the number of channels to write when
       * output_transform_fn() is called.
       */
      int32_t output_img_channels;

      /**
       * Create a DirectWriteOutputTransform::Params for an output image with the specified
       * number of channels.
       */
      Params(const int image_channels);

      /**
       * Create a DirectWriteOutputTransform::Params corresponding to a particular output image geometry.
       */
      Params(const nn::ImageGeometry &output_geometry);

      /**
       * Deserialize a DirectWriteOutputTransform::Params from a stream.
       * 
       * The data to be deserialized should come from a previous call to 
       * DirectWriteOutputTransform::Params::Serialize(). The serialized data format
       * is considered to be opaque.
       */
      Params(std::istream &stream);

      /**
       * Serialize a DirectWriteOutputTransform::Params into a stream.
       * 
       * The serialized object can be recovered later using the appropriate stream constructor.
       */
      void Serialize(std::ostream &stream) const;
    };

    /**
     * Parameters required by this output transform handler.
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
    virtual int8_t *output_transform_fn(int8_t *Y,
                                        vpu_ring_buffer_t *acc,
                                        int32_t output_channel_group) override;
  };

  /**
 * This output transform applies a per-channel, rounding, saturating right-shift to the 32-bit
 * accumulators to get 8-bit results.
 */
  class ShiftInt8OutputTransform : public OutputTransformFn,
                                   public ChannelParallelComponent<VPU_INT8_ACC_PERIOD_LOG2>
  {
  public:
    /**
     * Configuration parameters for ShiftInt8OutputTransform
     */
    struct Params
    {
      /**
       * The number of channels in the filter's output image.
       * 
       * This is required to determine the number of channels to write when
       * output_transform_fn() is called.
       */
      int32_t output_img_channels;

      /**
       * The per-output-channel arithmetic right-shifts to be applied to the accumulators.
       */
      int16_t shifts[VPU_INT8_ACC_PERIOD];

      /**
       */
      Params() {}

      /**
       * Create a ShiftInt8OutputTransform::Params
       */
      Params(const int output_image_channels,
             const int16_t shift);

      /**
       * Create a ShiftInt8OutputTransform::Params
       */
      Params(const nn::ImageGeometry &output_image,
             const int16_t shift);

      /**
       * Deseriaized a ShiftInt8OutputTransform::Params from a byte stream.
       * 
       * The data in the stream should come from a prior call to ShiftInt8OutputTransform::Params::Serialize().
       */
      Params(std::istream &stream);

      /**
       * Serialize this ShiftInt8OutputTransform::Params into a byte stream.
       * 
       * Note: This does not serialize the shift values.
       */
      void Serialize(std::ostream &stream) const;
    };

    /**
     * Parameters required by this output transform handler.
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
    virtual int8_t *output_transform_fn(int8_t *Y,
                                        vpu_ring_buffer_t *acc,
                                        int32_t output_channel_group) override;
  };

  /**
 *  xcore implementation of shift_int8_output_transform_ref()
 */
  C_API 
  void shift_int8_output_transform_xcore(
      int8_t *output,
      const vpu_ring_buffer_t *acc,
      const int16_t *right_shifts,
      const int channel_count);

  /**
 *  Portable implementation of shift_int8_output_transform_xcore()
 */
  C_API 
  void shift_int8_output_transform_ref(
      int8_t *output,
      const vpu_ring_buffer_t *acc,
      const int16_t *right_shifts,
      const int channel_count);

}
