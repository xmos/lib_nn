#ifndef LIB_NN_CONV2D_UTILS_HPP_
#define LIB_NN_CONV2D_UTILS_HPP_

#include <cstdint>
#include <vector>

#include "geom/Filter2dGeometry.hpp"
#include "geom/util.hpp"

C_API typedef struct {
  int16_t high[16];
  uint16_t low[16];
} vpu_split_acc32_t;

C_API typedef struct {
  uint16_t shift1[16];
  int16_t scale[16];
  int16_t offset_scale[16];
  int16_t offset[16];
  uint16_t shift2[16];
} nn_acc32_to_int8_params_t;

namespace nn {
namespace conv2d {
namespace util {

class TfLiteConverter {
 public:
  static std::vector<vpu_split_acc32_t> ConvertBiases(
      const Filter2dGeometry& filter, const int8_t kernel_weights[],
      const int32_t biases_in[], const int32_t input_zero_point,
      const bool is_depthwise);

  static std::vector<nn_acc32_to_int8_params_t> ConvertOutputParams(
      const Filter2dGeometry& filter, const float effective_output_multiplier[],
      const int32_t output_zero_point);

  static void QuantizeEffectiveOutputMultiplier(int32_t& quantized_multiplier,
                                                int32_t& shift,
                                                const double double_multiplier);
};

}  // namespace util
}  // namespace conv2d
}  // namespace nn

#endif  // LIB_NN_CONV2D_UTILS_HPP_
