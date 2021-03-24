
#include "RefOps.hpp"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/fully_connected.h"

#include "../src/cpp/filt2d/util/conv2d_utils.hpp"

using namespace nn::filt2d;
using namespace nn::filt2d::geom;
using namespace nn::test::ops::ref;


std::vector<int8_t> nn::test::ops::ref::FullyConnectedReference(
    const int N_input_elements,
    const int N_output_elements,
    const int8_t input_img[],
    const int8_t weights[],
    const int32_t biases[],
    const float output_multiplier,
    const int8_t input_zero_point,
    const int8_t output_zero_point)
{
  
  tflite::FullyConnectedParams params;

  params.input_offset = -input_zero_point;
  params.weights_offset = 0;
  params.output_offset = output_zero_point;
  
  params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  int32_t shift;
  nn::filt2d::conv2d::util::TfLiteConverter::QuantizeEffectiveOutputMultiplier(
                                                params.output_multiplier, shift,
                                                output_multiplier);
  params.output_shift = shift;

  // Note: The TFlite code does not look at input_shape or bias_shape
  tflite::RuntimeShape input_shape = { 1, N_input_elements };
  tflite::RuntimeShape output_shape = {1, N_output_elements };
  tflite::RuntimeShape weights_shape = { N_output_elements, N_input_elements };
  tflite::RuntimeShape bias_shape = { N_output_elements };

  auto result = std::vector<int8_t>(N_output_elements);

  tflite::reference_integer_ops::FullyConnected(params, 
                                                input_shape, input_img, 
                                                weights_shape, weights,
                                                bias_shape, biases,
                                                output_shape, &result[0]);

  return result;
}