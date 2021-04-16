#include <cstdint>
#include <cstring>
#include <vector>
#include "xs3_vpu.h"
#include "vpu.hpp"


namespace nn {
namespace filt2d {

/**
 * Interface implemented by all output transform handlers.
 * 
 * Contains a single function, output_transform_fn() which converts a set of 32-bit accumulators
 * into a set of 8-bit outputs and writes them to the specified output image.
 */
class OutputTransformFn {
  public:
    virtual int8_t * output_transform_fn(int8_t * Y, 
                                         vpu_ring_buffer_t * A, 
                                         int32_t output_channel_group) = 0;
};

/**
 * these are in a protected order(internally)
 */
typedef struct output_transform_values_t {
    int16_t clamp_near[VPU_INT16_EPV];
    int16_t clamp_far_0[VPU_INT16_EPV];
    int16_t clamp_far_1[VPU_INT16_EPV];
    int16_t bias_multipler[VPU_INT16_EPV];
    int16_t final_shr[VPU_INT16_EPV];
    int16_t accu_shr[VPU_INT16_EPV]; //for the vlsat
    int32_t accu_shl;                //for the vlashr
} output_transform_values_t;

/**
 * 
 */
struct QuantisationParams {
    output_transform_values_t otv;
    std::vector<int16_t> biases; 
    std::vector<int16_t> multipliers;
};

class OutputTransformFnInt8 : public OutputTransformFn {
  public:

    static QuantisationParams quantise_activation(std::vector<double> & output_transform_multiplier, 
      std::vector<double> & output_transform_bias, 
      std::vector<int32_t> & accu_min,
      std::vector<int32_t> & accu_max );
};

/**
 * 
 */
class OT_int8 : public OutputTransformFnInt8 
{

  public:
  class Params {
    public:
      int32_t output_slice_channel_count; //TODO push into base class
      output_transform_values_t * otv;
      int16_t * biases;//[output_slice_channel_count];
      int16_t * multipliers;//[output_slice_channel_count];
      int16_t * accu_modifier;//[output_slice_channel_count];

      Params(int32_t output_slice_channel_count, output_transform_values_t * otv, 
        int16_t * biases, int16_t * multipliers, int16_t * accu_modifier):
        output_slice_channel_count(output_slice_channel_count),
        otv(otv),
        biases(biases),
        multipliers(multipliers),
        accu_modifier(accu_modifier){}


      // void foo(
      //   const int output_ch_count,
      //   const int elements_per_channel,
      //   const int8_t kernel_weights[],
      //   const int32_t biases[],
      //   const float effective_output_multiplier[],
      //   const int8_t input_zero_point,
      //   const int8_t output_zero_point,
      //   std::vector<int8_t> & boggled_kernel_weights,
      //   std::vector<int8_t> & boggled_biases,
      //   std::vector<int8_t> & boggled_effective_output_multiplier,
      //   output_transform_values_t & boggled_otv);
  };

  private:
  Params * params;
  public:
    OT_int8(Params * params):params(params){};
    

    int8_t * output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group);
};

class OTBinary_int8 : public OutputTransformFnInt8 
{
  public:
  class Params {
    public:
      int32_t output_slice_channel_count; //TODO push into base class
      output_transform_values_t * otv;
      int16_t * biases;//[output_slice_channel_count];
      int16_t * multipliers;//[output_slice_channel_count];
      int16_t * accu_modifier;//[output_slice_channel_count];

      Params(int32_t output_slice_channel_count, output_transform_values_t * otv, 
        int16_t * biases, int16_t * multipliers, int16_t * accu_modifier);
  };

  Params * params;
  public:
    OTBinary_int8(Params * params):params(params){};

    static QuantisationParams quantise_activation(std::vector<double> & output_transform_multiplier, 
      std::vector<double> & output_transform_bias, 
      std::vector<int32_t> & accu_min,
      std::vector<int32_t> & accu_max );

    int8_t * output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group);
};

class OTBinary_bin : public OutputTransformFn{

  int16_t * thresholds;
  public:
    OTBinary_bin(int16_t * thresholds);

    int8_t * output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group);
};

}
}