#pragma once

#include <vector>
#include <cstdint>

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "../src/cpp/filt2d/OutputTransformers.hpp"

namespace nn {
  namespace test {
    namespace ops {

std::vector<int8_t> Conv2dDeepFilter_Valid(
    nn::Filter2dGeometry& filter_geometry,
    const int8_t input_img[], 
    const int8_t kernel_weights[],
    const int32_t ref_biases[],
    const float ref_eff_out_multiplier[],
    const int8_t input_zero_point,
    const int8_t output_zero_point);

}}}