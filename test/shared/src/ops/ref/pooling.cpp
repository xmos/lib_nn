
#include "RefOps.hpp"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/pooling.h"

using namespace nn;
using namespace nn::test::ops::ref;




std::vector<int8_t> nn::test::ops::ref::MaxPoolReference(
    const nn::Filter2dGeometry& filter_geometry,
    const int8_t input_img[])
{

  tflite::PoolParams params;

  auto padding = filter_geometry.Padding();

  params.activation = tflite::FusedActivationFunctionType::kNone;
  params.padding_type = tflite::PaddingType::kNone;
  params.padding_values.height = padding.top;
  params.padding_values.width = padding.left;
  params.padding_values.height_offset = padding.bottom;
  params.padding_values.width_offset = padding.right;
  params.stride_height = filter_geometry.window.stride.row;
  params.stride_width = filter_geometry.window.stride.col;
  params.filter_height = filter_geometry.window.shape.height;
  params.filter_width = filter_geometry.window.shape.width;
  params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  tflite::RuntimeShape input_shape = 
      { 1, (int) filter_geometry.input.height, (int) filter_geometry.input.width, (int) filter_geometry.input.depth };
  tflite::RuntimeShape output_shape = 
      { 1, (int) filter_geometry.output.height, (int) filter_geometry.output.width, (int) filter_geometry.output.depth };


  auto result = std::vector<int8_t>(filter_geometry.output.imageElements());

  tflite::reference_integer_ops::MaxPool(params, input_shape, input_img, output_shape, &result[0]);

  return result;
}

std::vector<int8_t> nn::test::ops::ref::AveragePoolReference(
    const nn::Filter2dGeometry& filter_geometry,
    const int8_t input_img[])
{

  tflite::PoolParams params;

  auto padding = filter_geometry.Padding();

  params.activation = tflite::FusedActivationFunctionType::kNone;
  params.padding_type = tflite::PaddingType::kNone;
  params.padding_values.height = padding.top;
  params.padding_values.width = padding.left;
  params.padding_values.height_offset = padding.bottom;
  params.padding_values.width_offset = padding.right;
  params.stride_height = filter_geometry.window.stride.row;
  params.stride_width = filter_geometry.window.stride.col;
  params.filter_height = filter_geometry.window.shape.height;
  params.filter_width = filter_geometry.window.shape.width;
  params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  tflite::RuntimeShape input_shape = 
      { 1, (int) filter_geometry.input.height, (int) filter_geometry.input.width, (int) filter_geometry.input.depth };
  tflite::RuntimeShape output_shape = 
      { 1, (int) filter_geometry.output.height, (int) filter_geometry.output.width, (int) filter_geometry.output.depth };


  auto result = std::vector<int8_t>(filter_geometry.output.imageElements());

  tflite::reference_integer_ops::AveragePool(params, input_shape, input_img, output_shape, &result[0]);

  return result;
}