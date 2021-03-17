
#include "NNOps.hpp"

#include <memory>
#include <cassert>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"

#include "../src/cpp/filt2d/Conv2dDeepFilter.hpp"
#include "../src/cpp/filt2d/util/conv2d_utils.hpp"

using namespace nn::filt2d;
using namespace nn::filt2d::geom;
using namespace nn::test::ops;

using FilterClass = nn::filt2d::op::Conv2dDeepFilter_Valid;

std::vector<int8_t> nn::test::ops::Conv2dDeepFilter_Valid(
    Filter2dGeometry& filter_geometry,
    const int8_t input_img[], 
    const int8_t kernel_weights[],
    const int32_t ref_biases[], 
    const float ref_eff_out_multiplier[],
    const int8_t input_zero_point,
    const int8_t output_zero_point)
{
  // Need ref_params.input.zero_point to compute biases
  assert(FilterClass::SupportsGeometry(filter_geometry));

  const auto cog_count = FilterClass::CogCount(filter_geometry.output.depth);

  auto xcore_biases = nn::filt2d::conv2d::util::TfLiteConverter::ConvertBiases(
                                      filter_geometry, &kernel_weights[0],
                                      &ref_biases[0], input_zero_point,
                                      FilterClass::IsDepthwise);

  auto ot_params = nn::filt2d::conv2d::util::TfLiteConverter::ConvertOutputParams(
                                      filter_geometry, ref_eff_out_multiplier, output_zero_point);

  auto xcore_output = std::vector<int8_t>(filter_geometry.output.imageElements());
  auto patch_mem = std::vector<int8_t>(filter_geometry.window.windowBytes() + 32);

  memset(&xcore_output[0], 0xCC, xcore_output.size());
  memset(&patch_mem[0], 0, patch_mem.size());
  
  auto filter = op::Conv2dDeepFilter_Valid(
                      input_img, &xcore_output[0],
                      filter_geometry, &xcore_biases[0], kernel_weights,
                      &ot_params[0]);


  filter.execute(&patch_mem[0]);

  return xcore_output;
}