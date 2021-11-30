
// #include "RefOps.hpp"
// #include "conv2d_utils.hpp"
// #include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
// #include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"

#include "larq_compute_engine/core/bconv2d/output_transform.h"
#include "larq_compute_engine/core/bitpacking/bitpack.h"
#include "larq_compute_engine/core/types.h"
#include "nn_operator.h"

using namespace tflite;

#include "larq_compute_engine/core/bconv2d/reference.h"

#include "Conv2d.hpp"
#include "Rand.hpp"
#include "RefOps.hpp"
#include "geom/Filter2dGeometry.hpp"
#include "geom/util.hpp"
#include "nn_types.h"

using namespace nn;
using namespace nn::test::ops::ref;

namespace compute_engine {

namespace ce = compute_engine;

namespace core {

extern "C" void larq_ref_bsign(int8_t* input, int32_t* output,
                               size_t inputLength, int32_t zero_point) {
  bitpacking::bitpack_array<std::int8_t>(input, inputLength, output,
                                         zero_point);
}

using bconv2d::OutputTransform;

// Fill in the OutputTransform values for float and/or int8 outputs
template <typename DstScalar>
void GetOutputTransform(OutputTransform<DstScalar>& output_transform,
                        int32_t output_transform_clamp_min,
                        int32_t output_transform_clamp_max,
                        const float* output_transform_multiplier,
                        const float* output_transform_bias) {
  static_assert(std::is_same<DstScalar, std::int8_t>::value, "");
  output_transform.clamp_min = output_transform_clamp_min;
  output_transform.clamp_max = output_transform_clamp_max;
  output_transform.multiplier = output_transform_multiplier;
  output_transform.bias = output_transform_bias;
}

// Fill in the OutputTransform values for bitpacked outputs
void GetOutputTransform(OutputTransform<core::TBitpacked>& output_transform,
                        const int32_t* thresholds) {
  output_transform.thresholds = thresholds;
}
}  // namespace core


template <typename DstScalar>
std::vector<DstScalar> LarqConv2dBinaryReference(
    const Filter2dGeometry& filter_geometry, 
    const int32_t* packed_input_data,
    const int32_t* packed_filter_data,
    const int channels_per_output_word,
    const ce::core::OutputTransform<DstScalar>& output_transform
    ) {

  const int batches = 1;
  const int channels_per_word = 32;

  compute_engine::core::bconv2d::BConv2DParams params;

  params.filter_width = filter_geometry.window.shape.width;
  params.filter_height = filter_geometry.window.shape.height;
  params.channels_in = filter_geometry.input.depth / channels_per_word;
  params.channels_out = filter_geometry.output.depth / channels_per_word;
  params.groups = 1;
  
  params.stride_height = filter_geometry.window.stride.row;;
  params.stride_width = filter_geometry.window.stride.col;

  params.dilation_height_factor = filter_geometry.window.dilation.row;
  params.dilation_width_factor = filter_geometry.window.dilation.col;

  //TODO this section was previously unsupported
  params.padding_type = TfLitePadding::kTfLitePaddingValid;;//TfLitePadding

  auto padding = filter_geometry.Padding();

  params.padding_values.height = padding.top;
  params.padding_values.height_offset = padding.bottom;
  params.padding_values.width = padding.left;
  params.padding_values.width_offset = padding.right;

  assert(padding.top == 0);
  assert(padding.bottom == 0);
  assert(padding.left == 0);
  assert(padding.right == 0);

  params.pad_value = 1;// Must be 0 or 1

  struct {
    tflite::RuntimeShape input, output, filter;
  } shape = {
      .input = {batches, (int)filter_geometry.input.height,
                (int)filter_geometry.input.width,
                filter_geometry.input.depth / channels_per_word},
      .output = {batches, (int)filter_geometry.output.height,
                 (int)filter_geometry.output.width,
                 filter_geometry.output.depth / channels_per_output_word},
      .filter = {(int)filter_geometry.output.depth,
                 (int)filter_geometry.window.shape.height,
                 (int)filter_geometry.window.shape.width,
                 filter_geometry.input.depth/ channels_per_word},
  };

  auto output_data = std::vector<DstScalar>(filter_geometry.output.ElementCount());

  compute_engine::core::bconv2d::BConv2DReference<std::uint32_t, DstScalar>(
      &params, 
      shape.input, 
      packed_input_data, 
      shape.filter, 
      packed_filter_data, 
      output_transform, 
      shape.output,
      output_data.data()
      );

  return output_data;
}

std::vector<int8_t> Conv2dBNNIntOutReference(
    const Filter2dGeometry& filter_geometry, 
    const int32_t* packed_input_data,
    const int32_t* packed_filter_data,
    int8_t* packed_output_data,
    const float* post_activation_multiplier, 
    const float* post_activation_bias,
    const int clamp_min, const int clamp_max
    ) 
{
  ce::core::OutputTransform<std::int8_t> output_transform;
  ce::core::GetOutputTransform(output_transform, clamp_min, clamp_max,
                               post_activation_multiplier,
                               post_activation_bias);

  const unsigned channels_per_output_word = 1;

  return LarqConv2dBinaryReference<std::int8_t>(filter_geometry, packed_input_data, 
    packed_filter_data, channels_per_output_word, output_transform);
}

std::vector<int32_t> Conv2dBNNBinaryOutReference(
    const Filter2dGeometry& filter_geometry, 
    const int32_t* packed_input_data,
    const int32_t* packed_filter_data,
    int32_t* packed_output_data,
    const int32_t* thresholds
    ) 
{
  ce::core::OutputTransform<std::int32_t> output_transform;
  ce::core::GetOutputTransform(output_transform, thresholds);

  const unsigned channels_per_output_word = 32;

  return LarqConv2dBinaryReference<std::int32_t>(filter_geometry, packed_input_data, 
    packed_filter_data, channels_per_output_word, output_transform);
}

}