#pragma once

#include "../misc.hpp"
#include "../geom/Filter2dGeometry.hpp"
#include "../OutputTransformers.hpp"

#include <cstdint>
#include <vector>



namespace nn {
  namespace filt2d {
    namespace conv2d {
      namespace util {
  

class TfLiteConverter {

  public:

    static std::vector<vpu_split_acc32_t> ConvertBiases(
        const geom::Filter2dGeometry& filter,
        const int8_t kernel_weights[],
        const int32_t biases_in[],
        const int32_t input_zero_point,
        const bool is_depthwise);
        
    static std::vector<nn_acc32_to_int8_params_t> ConvertOutputParams(
        const geom::Filter2dGeometry& filter,
        const float effective_output_multiplier[],
        const int32_t output_zero_point);

    static void QuantizeEffectiveOutputMultiplier(
        int32_t& quantized_multiplier,
        int32_t& shift,
        const double double_multiplier);


};


}}}}