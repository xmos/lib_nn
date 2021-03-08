
#include "OutputTransformers.hpp"


void nn::filt2d::Int8OutputTransformHandler::transform(
      int8_t * output,
      vpu_split_acc32_t const& accumulator,
      ImageVect const& output_coords,
      unsigned const channels_out)
{
  const unsigned cog = output_coords.channel >> 4;
  const auto* params = &this->config.ot_params[cog];

  if(this->config.symmetric){
    conv2d_output_transform_symmetric_int8(output, &accumulator, params, channels_out);
  } else {
    conv2d_output_transform_asymmetric_int8(output, &accumulator, params, channels_out);
  }
}