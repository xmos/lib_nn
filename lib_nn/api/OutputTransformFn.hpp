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

//TODO maybe the output type should be templated
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

  private:
    /*
    This is used to complete the ActivationParams attributes 
    */
    void backprop_output_clamps_to_accu_limits(
      int64_t output_high, int64_t output_low, bool verbose = false);

  public:
    ActivationParams(double bias, double multiplier, int32_t accu_min_val,
                    int32_t accu_max_val,
                    int64_t output_high = INT8_MAX,
                    int64_t output_low = INT8_MIN 
       ): original_bias(bias),
          original_multiplier(multiplier),
          original_accu_min_val(accu_min_val),
          original_accu_max_val(accu_max_val),

          // These are going to be adjusted to account for the output clamp  
          bias(bias),     
          multiplier(multiplier),
          accu_min_val(accu_min_val),
          accu_max_val(accu_max_val) {
      
      //This asseumes a linear mapping between output and input, this might not 
      //always be true in the future.
      //TODO write tests for this
      backprop_output_clamps_to_accu_limits(output_high, output_low);

      assert(accu_min_val <= accu_max_val);
      assert(original_accu_min_val <= original_accu_max_val);
    }
  };

  typedef std::vector<OutputTransformFn::ActivationParams> MulsAndBias;

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

  static MulsAndBias canonicaliseConv2D(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, int input_zero_point,
      int output_zero_point, int output_channel_count) ;

  static MulsAndBias canonicaliseConv2DDepthwise(
      const std::vector<float> &eff_mult, const std::vector<int32_t> &bias,
      const std::vector<int8_t> &weights, const std::array<int, 4> &shape,
      int input_zero_point, int output_zero_point, int output_channels);

  static MulsAndBias canonicaliseConv2DClamped(
      const std::vector<float> &post_activation_multiplier,
      const std::vector<float> &post_activation_bias, int receptive_volume,
      int32_t clamp_low, int32_t clamp_high, int output_channel_count);

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

/*
  Actual output transforms beneath here. These require a layout of the 
  quantisation strat in a way that is efficient for the hardware. Also,
  these dont allocate memory.
*/

class OTPerGroup : public OutputTransformFn {
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

    Params()
        : output_slice_channel_count(0),
          initial_shift(0),
          final_shr(0) {}
  };
 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;
  int16_t *multipliers_and_biases;

 public:
  OTPerGroup(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);


  static void layout_for_hw(Params * params, std::vector<int16_t> &data){

  }

  void setMultipliersAndBiases(int16_t *m) {
    multipliers_and_biases = m;
    assert(m != nullptr);
    assert(is_aligned(multipliers_and_biases, 4));
  }
};

class OTPerChannel : public OutputTransformFn {
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
        : output_slice_channel_count(output_slice_channel_count){};

  };
 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;
  int16_t *multipliers_and_biases;

 public:
  OTPerChannel(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  void setMultipliersAndBiases(int16_t *m) {
    multipliers_and_biases = m;
    assert(m != nullptr);
    assert(is_aligned(multipliers_and_biases, 4));
  }
};

class OT32BitBias : public OutputTransformFn {
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
        : output_slice_channel_count(output_slice_channel_count){};

  };
 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;
  int16_t *multipliers_and_biases;

 public:
  OT32BitBias(Params *params) : params(params){};

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  void setMultipliersAndBiases(int16_t *m) {
    multipliers_and_biases = m;
    assert(m != nullptr);
    assert(is_aligned(multipliers_and_biases, 4));
  }
};

class OTPerGroupClamped : public OutputTransformFn {
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

    Params()
        : output_slice_channel_count(0),
          initial_shift(0),
          final_shr(0) {}
  };

 private:
  /**
   * @brief This describes the channels over which this class will perform its
   * operation(OutputTransform) and how each channel will transformed.
   */
  Params *params;
  int16_t *offsets_multipliers_and_biases;

 public:
  OTPerGroupClamped(Params *params) : params(params){};

  static void layout_for_hw(Params * params, std::vector<int16_t> &data){

  }

  int8_t *output_transform_fn(int8_t *Y, VPURingBuffer *A,
                              int32_t output_channel_group);

  void setOffsetsMultipliersAndBiases(int16_t *m) {
    offsets_multipliers_and_biases = m;
    assert(m != nullptr);
    assert(is_aligned(offsets_multipliers_and_biases, 4));
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

typedef int16_t threshold_t;
class OTBinary : public OutputTransformFn {
 private:
  threshold_t *thresholds;

 public:
  OTBinary(){};

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


/*
  Quantisation strategys - these convert the MulsAndBias into a quantised 
  representation.
*/

class QantisationStrategy {
  protected:
  OutputTransformFn::MulsAndBias &activation_params;
  
  public:
  virtual double get_quant_error(bool use_high_precision = false) = 0;

  QantisationStrategy(OutputTransformFn::MulsAndBias &activation_params):activation_params(activation_params){};
};

class QuantisationPerGroupStrategy : public QantisationStrategy {

  std::vector<int16_t> multipliers;
  std::vector<int16_t> biases;
  int16_t initial_shr;
  int16_t final_shr;

public:
  QuantisationPerGroupStrategy(OutputTransformFn::MulsAndBias &activation_params, bool verbose = false);

  double get_quant_error(bool use_high_precision);

  // void layout_for_hw(OTPerGroup::Params * params, std::vector<int16_t> &data){
      // QuantisationParams qp =
        //     OutputTransformFnInt8::quantise_activation(
        //         mul_and_biases);

        // assert(qp.multipliers.size() > 0);
        // assert(qp.biases.size() > 0);

        // auto serialised_offsets_multipliers_and_biases =
        //     OutputTransformFn::serialise_memory(
        //         qp.multipliers, qp.biases);

        // // pad qp.multipliers_and_biases to a multiple
        // // of VPU_INT16_EPV this is to work around array
        // // over reads
        // int16_t pad_val =
        //     rng.rand<int16_t>();  // this is arbitrary
        // OutputTransformFn::pad_final_access(
        //     serialised_offsets_multipliers_and_biases,
        //     VPU_INT16_EPV, pad_val);
  // };

};

class QuantisationPerChannelStrategy : public QantisationStrategy {

  std::vector<int16_t> multipliers;
  std::vector<int16_t> biases;
  std::vector<int16_t> initial_shr;
  std::vector<int16_t> final_shr;

public:
  QuantisationPerChannelStrategy(OutputTransformFn::MulsAndBias &activation_params, bool verbose = false);

  double get_quant_error(bool use_high_precision){ return 0.0;}
  
  void layout_for_hw(OTPerChannel::Params * params, std::vector<int16_t> &data){

  };
};

class QuantisationParams32BitBiasStratergy : public QantisationStrategy {

  std::vector<int16_t> multipliers;
  std::vector<int32_t> biases;
  int32_t initial_shr;
  int32_t final_shr;

public:
  double get_quant_error(bool use_high_precision = false);

  QuantisationParams32BitBiasStratergy(OutputTransformFn::MulsAndBias &activation_params, bool verbose);
  
  void layout_for_hw(OT32BitBias::Params * params, std::vector<int16_t> &data){

  };
};

}  // namespace nn
#endif  // LIB_NN_OUTPUT_TRANSFORM_FN_H_
