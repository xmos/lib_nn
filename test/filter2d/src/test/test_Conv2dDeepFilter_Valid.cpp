

#include "nn_types.h"
#include "../src/cpp/filt2d/misc.hpp"
#include "../src/cpp/filt2d/util/conv2d_utils.hpp"
#include "../src/cpp/filt2d/geom/Filter2dGeometry.hpp"
#include "../src/cpp/filt2d/Conv2dDeepFilter.hpp"
#include "../src/cpp/filt2d/util/RectRange.hpp"

#include "gtest/gtest.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

#include <cstdint>
#include <memory>
#include <algorithm>
#include <vector>
#include <iostream>

using namespace nn::filt2d;
using namespace nn::filt2d::geom;
using namespace nn::filt2d::op;




struct ExtraRefParams {
  struct { int32_t zero_point; } input, filter;
  struct {
    int32_t zero_point;
    int32_t* multiplier;
    int32_t* shift;
    struct { int32_t min, max; } activation;
  } output;
};

std::vector<int8_t> Conv2dDeepReference(
    geom::Filter2dGeometry& geom,
    const int8_t* input_data, const int8_t* filter_data,
    const int32_t* bias_data, const ExtraRefParams& ref_params)
{
  tflite::ConvParams op_params;

  auto pad_initial = geom.ModelPadding(true);
  auto pad_final   = geom.ModelPadding(false);
  op_params.padding_type = tflite::PaddingType::kSame;
  op_params.padding_values.height = pad_initial.top;
  op_params.padding_values.width = pad_initial.left;
  op_params.padding_values.height_offset = pad_final.bottom;
  op_params.padding_values.width_offset = pad_final.right;

  op_params.stride_height = geom.window.stride.row;
  op_params.stride_width  = geom.window.stride.col;
  op_params.dilation_height_factor = geom.window.dilation.row;
  op_params.dilation_width_factor  = geom.window.dilation.col;

  op_params.input_offset = -ref_params.input.zero_point;
  op_params.output_offset = ref_params.output.zero_point;
  op_params.weights_offset = -ref_params.filter.zero_point;

  // When per-channel quantization is used, these values are ignored. Needs the arrays of multipliers and shifts
  // op_params.output_multiplier = ref_params.output.multiplier;
  // op_params.output_shift = ref_params.output.shift;

  op_params.quantized_activation_min = ref_params.output.activation.min;
  op_params.quantized_activation_max = ref_params.output.activation.max;

  struct {
    tflite::RuntimeShape input, output, bias, filter, im2col;
  } shape = {
      .input = {  1, (int) geom.input.height,  (int) geom.input.width,  (int) geom.input.depth },
      .output = { 1, (int) geom.output.height, (int) geom.output.width, (int) geom.output.depth },
      .bias = { (int) geom.output.depth },
      .filter = { (int) geom.output.depth, (int) geom.window.shape.height, 
                  (int) geom.window.shape.width, (int) geom.input.depth },
      .im2col = { (int) geom.window.windowElements() },
  };

  auto output_data = std::vector<int8_t>( geom.output.imageElements() );

  tflite::reference_integer_ops::ConvPerChannel(op_params, 
                              ref_params.output.multiplier, ref_params.output.shift,
                              shape.input, input_data,
                              shape.filter, filter_data,
                              shape.bias, bias_data,
                              shape.output, &output_data[0]);

  return output_data;
}


std::vector<int8_t> Conv2dDeepLibNN(
    geom::Filter2dGeometry& geom,
    const int8_t* input_data, 
    const int8_t* filter_data,
    const vpu_split_acc32_t biases[], 
    const nn_acc32_to_int8_params_t ot_params[])
{
  assert(Conv2dDeepFilter_Valid::SupportsGeometry(geom));

  
  auto xcore_output = std::vector<int8_t>(geom.output.imageElements());
  auto patch_mem = std::vector<int8_t>(geom.window.windowBytes() + 32);

  memset(&xcore_output[0], 0xCC, xcore_output.size());
  memset(&patch_mem[0], 0, patch_mem.size());

  
  auto filter = nn::filt2d::op::Conv2dDeepFilter_Valid(
                      input_data, &xcore_output[0],
                      geom, biases, filter_data,
                      ot_params, false);


  filter.execute(&patch_mem[0]);

  return xcore_output;
}



static void printFilter(geom::Filter2dGeometry filter)
{

  std::cout << "Input Image:  { " << filter.input.height << ", " << filter.input.width << ", " 
            << filter.input.depth << " }" << std::endl;
  std::cout << "Output Image: { " << filter.output.height << ", " << filter.output.width << ", " 
            << filter.output.depth << " }" << std::endl;

  std::cout << "Window: " << "Shape { " << filter.window.shape.height << ", " 
                                        << filter.window.shape.width  << ", " 
                                        << filter.window.shape.depth << " }" << std::endl;
  std::cout << "        " << "Start { " << filter.window.start.row << ", "
                                        << filter.window.start.col << " }" << std::endl;
  std::cout << "        " << "Stride { " << filter.window.stride.row << ", "
                                         << filter.window.stride.col << ", "
                                         << filter.window.stride.channel << " }" << std::endl;
  std::cout << "        " << "Dilation { " << filter.window.dilation.row << ", "
                                           << filter.window.dilation.col << " }" << std::endl;

  std::cout << std::endl;


}




TEST(Conv2dDeepFilter_Valid,BasicTest)
{

  unsigned count = 0;

  for(auto filt_geom: Conv2dDeepFilter_Valid::GetGeometryIterator()){

  
    auto input = std::vector<int8_t>(filt_geom.input.imageBytes());
    auto kernel = std::vector<int8_t>(filt_geom.window.windowBytes() * filt_geom.output.depth);
    auto ref_biases = std::vector<int32_t>(filt_geom.output.depth);
    auto effective_output_multiplier = std::vector<float>(filt_geom.output.depth);
    auto ref_output_multiplier = std::vector<int32_t>(filt_geom.output.depth);
    auto ref_output_shift = std::vector<int32_t>(filt_geom.output.depth);

    auto window_elements = filt_geom.window.windowElements();

    memset(&input[0],  1, input.size());
    memset(&kernel[0], 1, kernel.size());
    // memset(&ref_biases[0], 0, ref_biases.size());

    for(int i = 0; i < filt_geom.output.depth; ++i){
      ref_biases[i] = -window_elements + 27;
      effective_output_multiplier[i] = 1.0f;
      nn::filt2d::conv2d::util::TfLiteConverter::QuantizeEffectiveOutputMultiplier(
                                                  ref_output_multiplier[i], 
                                                  ref_output_shift[i],
                                                  effective_output_multiplier[i] );
    }

    ExtraRefParams ref_params;
    ref_params.input.zero_point = 0;
    ref_params.filter.zero_point = 0;
    ref_params.output.zero_point = 0;
    ref_params.output.multiplier = &ref_output_multiplier[0];
    ref_params.output.shift = &ref_output_shift[0];
    ref_params.output.activation.min = std::numeric_limits<int8_t>::min();
    ref_params.output.activation.max = std::numeric_limits<int8_t>::max();

    auto ref_output = Conv2dDeepReference(filt_geom, &input[0], &kernel[0], 
                                          &ref_biases[0], ref_params);

    auto xcore_biases = 
        nn::filt2d::conv2d::util::TfLiteConverter::ConvertBiases(
                                                      filt_geom, &kernel[0], 
                                                      &ref_biases[0], ref_params.input.zero_point, 
                                                      false);

    auto xcore_output_params = 
        nn::filt2d::conv2d::util::TfLiteConverter::ConvertOutputParams(
                                                      filt_geom, &effective_output_multiplier[0],
                                                      ref_params.output.zero_point);

    auto nn_output = Conv2dDeepLibNN(filt_geom, &input[0], &kernel[0], 
                                    &xcore_biases[0], &xcore_output_params[0]);


    int8_t expected_out = 27;

    ASSERT_EQ(ref_output.size(), filt_geom.output.imageElements());
    ASSERT_EQ(nn_output.size(), filt_geom.output.imageElements());

    for(int i = 0; i < filt_geom.output.imageElements(); ++i){
      EXPECT_EQ(nn_output[i], expected_out);
      EXPECT_EQ(ref_output[i], nn_output[i]);
      
      if(this->HasFailure()){
        std::cout << "Test failed with geometry:" << std::endl;
        printFilter(filt_geom);

        FAIL();
      }
    }

    count++;
  }

  std::cout << "Tested " << count << " geometries." << std::endl;
}



// TEST(Conv2dDeepFilter_Valid,SpecificTest)
// {

//   auto filt_geom = Filter2dGeometry(
//                       ImageGeometry(1, 1, 36),
//                       ImageGeometry(1, 1, 4),
//                       WindowGeometry(1, 1, 36, 0, 0, 1, 1, 0, 1, 1));

//   auto input = std::vector<int8_t>(filt_geom.input.imageBytes());
//   auto kernel = std::vector<int8_t>(filt_geom.window.windowBytes() * filt_geom.output.depth);
//   auto ref_biases = std::vector<int32_t>(filt_geom.output.depth);
//   auto effective_output_multiplier = std::vector<float>(filt_geom.output.depth);
//   auto ref_output_multiplier = std::vector<int32_t>(filt_geom.output.depth);
//   auto ref_output_shift = std::vector<int32_t>(filt_geom.output.depth);

//   memset(&input[0],  1, input.size());
//   memset(&kernel[0], 1, kernel.size());
//   memset(&ref_biases[0], 0, ref_biases.size());

//   for(int i = 0; i < filt_geom.output.depth; ++i){
//     effective_output_multiplier[i] = 1.0f;
//     nn::filt2d::conv2d::util::TfLiteConverter::QuantizeEffectiveOutputMultiplier(
//                                                 ref_output_multiplier[i], 
//                                                 ref_output_shift[i],
//                                                 effective_output_multiplier[i] );
//   }

//   ExtraRefParams ref_params;
//   ref_params.input.zero_point = 0;
//   ref_params.filter.zero_point = 0;
//   ref_params.output.zero_point = 0;
//   ref_params.output.multiplier = &ref_output_multiplier[0];
//   ref_params.output.shift = &ref_output_shift[0];
//   ref_params.output.activation.min = std::numeric_limits<int8_t>::min();
//   ref_params.output.activation.max = std::numeric_limits<int8_t>::max();

//   auto ref_output = Conv2dDeepReference(filt_geom, &input[0], &kernel[0], 
//                                         &ref_biases[0], ref_params);

//   // std::cout << "Ref output: [ ";
//   // for(int i = 0; i < 16; i++)
//   //   std::cout << static_cast<int>(ref_output[i]) << ", ";
//   // std::cout << "]" << std::endl;

//   auto xcore_biases = 
//       nn::filt2d::conv2d::util::TfLiteConverter::ConvertBiases(
//                                                     filt_geom, &kernel[0], 
//                                                     &ref_biases[0], ref_params.input.zero_point, 
//                                                     false);

//   auto xcore_output_params = 
//       nn::filt2d::conv2d::util::TfLiteConverter::ConvertOutputParams(
//                                                     filt_geom, &effective_output_multiplier[0],
//                                                     ref_params.output.zero_point);

//   auto nn_output = Conv2dDeepLibNN(filt_geom, &input[0], &kernel[0], 
//                                   &xcore_biases[0], &xcore_output_params[0]);

  
//   // std::cout << "xCore output: [ ";
//   // for(int i = 0; i < 16; i++)
//   //   std::cout << static_cast<int>(nn_output[i]) << ", ";
//   // std::cout << "]" << std::endl;

//   int8_t expected_out = filt_geom.window.windowElements();

//   ASSERT_EQ(ref_output.size(), filt_geom.output.imageElements());
//   ASSERT_EQ(nn_output.size(), filt_geom.output.imageElements());

//   for(int i = 0; i < filt_geom.output.imageElements(); ++i){
//     EXPECT_EQ(nn_output[i], expected_out);
//     EXPECT_EQ(ref_output[i], nn_output[i]);
    
//     if(this->HasFailure()){
//       std::cout << "Test failed with geometry:" << std::endl;
//       printFilter(filt_geom);

//       FAIL();
//     }
//   }

// }
