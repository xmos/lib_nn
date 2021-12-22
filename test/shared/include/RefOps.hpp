#pragma once

#include <cstdint>
#include <vector>

#include "geom/Filter2dGeometry.hpp"

namespace nn {
namespace test {
namespace ops {
namespace ref {

std::vector<int32_t> Conv2dBNNBinaryOutReference(
    const Filter2dGeometry& filter_geometry, const int32_t* packed_input_data,
    const int32_t* packed_filter_data, const int32_t* thresholds);

std::vector<int8_t> Conv2dBNNIntOutReference(
    const Filter2dGeometry& filter_geometry, const int32_t* packed_input_data,
    const int32_t* packed_filter_data, const float* post_activation_multiplier,
    const float* post_activation_bias, const int32_t clamp_min,
    const int32_t clamp_max);

std::vector<int8_t> Conv2dDenseReference(
    const nn::Filter2dGeometry& filter_geometry, const int8_t input_img[],
    const int8_t kernel_weights[], const int32_t biases[],
    const float effective_output_multiplier[], const int8_t input_zero_point,
    const int8_t output_zero_point);

std::vector<int8_t> Conv2dDepthwiseReference(
    const nn::Filter2dGeometry& filter_geometry, const int8_t input_img[],
    const int8_t kernel_weights[], const int32_t biases[],
    const float effective_output_multiplier[], const int8_t input_zero_point,
    const int8_t output_zero_point);

std::vector<int8_t> MaxPoolReference(
    const nn::Filter2dGeometry& filter_geometry, const int8_t input_img[]);

std::vector<int8_t> AveragePoolReference(
    const nn::Filter2dGeometry& filter_geometry, const int8_t input_img[]);

std::vector<int8_t> FullyConnectedReference(
    const int N_input_elements, const int N_output_elements,
    const int8_t input[], const int8_t weights[], const int32_t biases[],
    const float output_multiplier, const int8_t input_zero_point,
    const int8_t output_zero_point);

struct ElementwiseParams {
  struct {
    int8_t zero_point;
    float multiplier;
  } input[2], output;
};

std::vector<int8_t> AddElementwiseReference(
    const nn::ImageGeometry& image_geometry, const int8_t input_img0[],
    const int8_t input_img1[], const ElementwiseParams& params);

}  // namespace ref
}  // namespace ops
}  // namespace test
}  // namespace nn