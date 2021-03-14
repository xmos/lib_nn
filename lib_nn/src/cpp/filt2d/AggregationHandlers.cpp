
#include "AggregationHandlers.hpp"
#include "xs3_vpu.h"
#include "vpu_sim.h"
#include "misc.hpp"
#include "../src/asm/asm_constants.h"


using namespace nn::filt2d;

template<>
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




EXTERN_C void conv2d_aggregate_deep_patch_int8(
  vpu_split_acc32_t* accumulators,
  const int8_t* patch,
  const int8_t* kernel,
  const conv2d_aggregate_deep_patch_int8_params_t* params,
  const channel_count_t out_chans)
{
  const mem_stride_t K_cout_stride = params->K_cout_stride;
  const mem_stride_t K_cig_stride = out_chans * K_cout_stride + VPU_INT8_EPV;
  const mem_stride_t acc_offset = 2*(VPU_INT8_ACC_PERIOD - out_chans);

  unsigned cigs =    (params->patch_bytes + VPU_INT8_EPV - 1) >> VPU_INT8_EPV_LOG2;

  // Start with the final channel.
  kernel += K_cig_stride - K_cout_stride - VPU_INT8_EPV;

  nn::VPU vpu;

  vpu.vsetc(MODE_S8);
  vpu.vldd(advancePointer(accumulators->high, -acc_offset));
  vpu.vldr(advancePointer(accumulators->low,  -acc_offset));

  for(; cigs > 0; cigs--){
    vpu.vldc(patch);
    patch += VPU_INT8_EPV;

    for(int k = 0; k < VPU_INT8_ACC_PERIOD; k++){
      vpu.vlmaccr(kernel);
      kernel -= K_cout_stride;
    }

    kernel += K_cig_stride;
  }

  for(int k = VPU_INT8_ACC_PERIOD - out_chans; k > 0; k--)
    vpu.vlmaccr(vpu_vect_zero);
  
  vpu.vstd(accumulators->high);
  vpu.vstr(accumulators->low);

}
