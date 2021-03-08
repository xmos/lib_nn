
#include "AggregationHandlers.hpp"

#include "util.h"


using namespace nn::filt2d;

template <>
vpu_split_acc32_t Conv2dDeepPatchAggregator<int8_t,int8_t,vpu_split_acc32_t>::aggregate(
        int8_t const* input_img,
        ImageVect const& output_coords,
        unsigned const channels_out)
{
  unsigned const cog = output_coords.channel >> 4;

  int8_t const* kernel = advancePointer(this->config.kernel_tensor, 
                                        cog * config.kernel_block_bytes);

  vpu_split_acc32_t accs = config.biases[cog];
  
  conv2d_aggregate_deep_patch_int8(&accs, input_img, kernel, &config.agg_params, channels_out);

  return accs;
}