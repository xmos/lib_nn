#include <cstdint>
#include <cstring>
#include <vector>
#include "xs3_vpu.h"
#include "vpu.hpp"

class OutputTransformFn {
  public:
    virtual int8_t * output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group) = 0;
};

//these are in a protected order(internally)
typedef struct output_transform_values_t {
    int16_t clamp_near[VPU_INT16_EPV];
    int16_t clamp_far_0[VPU_INT16_EPV];
    int16_t clamp_far_1[VPU_INT16_EPV];
    int16_t bias_multipler[VPU_INT16_EPV];
    int16_t final_shr[VPU_INT16_EPV];
    int16_t accu_shr[VPU_INT16_EPV]; //for the vlsat
    int32_t accu_shl;                //for the vlashr
} output_transform_values_t;

struct QuantisationParams {
    output_transform_values_t otv;
    std::vector<int16_t> biases; 
    std::vector<int16_t> multipliers;
};

class OTBinary_int8 : public OutputTransformFn 
{

  int32_t output_slice_channel_count;
  output_transform_values_t * otv;
  int16_t * biases;//[output_slice_channel_count];
  int16_t * multipliers;//[output_slice_channel_count];
  int16_t * accu_modifier;//[output_slice_channel_count];

  public:
    OTBinary_int8(int32_t output_slice_channel_count, output_transform_values_t * otv, 
      int16_t * biases, int16_t * multipliers, int16_t * accu_modifier);

    
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
