
#include "RefOps.hpp"

#include "tensorflow/lite/kernels/internal/reference/integer_ops/add.h"

using namespace nn::filt2d;
using namespace nn::filt2d::geom;
using namespace nn::test::ops::ref;


static inline void Quantize(
                    int32_t& mantissa,
                    int& exponent,
                    const float float_mult)
{
  if(float_mult == 0.0f){
    mantissa = 0;
    exponent = 0;
    return;
  }

  double x = ldexp(frexp(float_mult, &exponent), 31);

  if( x >= ldexp(1, 31) ) {
    x /= 2;
    exponent++;
  }

  mantissa = int64_t(  round(x)  );
}

std::vector<int8_t> nn::test::ops::ref::AddElementwiseReference(
    const nn::filt2d::geom::ImageGeometry& image_geometry,
    const int8_t input_img0[],
    const int8_t input_img1[],
    const ElementwiseParams& params)
{
  // float_value = (int_value - zero_point) * multiplier
  // int_value   = float_value / multiplier + zero_point
  
  tflite::ArithmeticParams tf_params;

  tf_params.left_shift = 23;

  tf_params.input1_offset = -params.input[0].zero_point;
  Quantize(tf_params.input1_multiplier, tf_params.input1_shift, params.input[0].multiplier);
  
  tf_params.input2_offset = -params.input[1].zero_point;
  Quantize(tf_params.input2_multiplier, tf_params.input2_shift, params.input[1].multiplier);

  tf_params.output_offset = params.output.zero_point;
  Quantize(tf_params.output_multiplier, tf_params.output_shift, 1.0f / params.output.multiplier);

  tf_params.output_shift -= tf_params.left_shift;

  tf_params.quantized_activation_min = std::numeric_limits<int8_t>::min();
  tf_params.quantized_activation_max = std::numeric_limits<int8_t>::max();

  auto result = std::vector<int8_t>( image_geometry.imageElements() );

  tflite::reference_integer_ops::AddElementwise(result.size(), tf_params,
                                                input_img0, input_img1, &result[0]);

  return result;
}