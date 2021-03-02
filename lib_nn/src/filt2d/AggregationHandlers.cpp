
#include "AggregationHandlers.hpp"

#include "util.h"


using namespace nn::filt2d;

template <>
vpu_split_acc32_t Conv2dDeepPatchAggregator<int8_t,int8_t,vpu_split_acc32_t,16>::aggregate(
        int8_t const* input_img,
        ImageVect const& output_coords,
        unsigned const channels_out)
{
  unsigned const cog = output_coords.channel >> 16;

  int8_t const* kernel = advancePointer<const int8_t>(this->m_kernel_tensor, cog * this->m_kernel_block_bytes);

  vpu_split_acc32_t accs = this->m_biases[cog];
  
  conv2d_aggregate_deep_patch_int8(&accs, input_img, kernel, &this->m_agg_params, channels_out);

  return accs;
}