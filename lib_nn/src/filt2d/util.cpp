
#include "util.h"

#include "../cpp/vpu_sim.hpp"
#include "../asm/asm_constants.h"

#include <cstdio>

template <typename T>
static inline T* offsetPointer(T* orig, int32_t offset_bytes)
{
  return (T*) (((char*)orig)+offset_bytes);
}

extern const uint32_t vpu_vect_zero[8];


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
  vpu.vldd(offsetPointer(accumulators->high, -acc_offset));
  vpu.vldr(offsetPointer(accumulators->low,  -acc_offset));

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



EXTERN_C void conv2d_output_transform_symmetric_int8(
      int8_t * output,
      const vpu_split_acc32_t* accumulator,
      const nn_acc32_to_int8_params_t* params,
      unsigned const channels_out)
{
    nn::VPU vpu;
    vpu_vector_t vec_tmp;

    vpu.vsetc(MODE_S16);

    vpu.vldd(accumulator->high);
    vpu.vldr(accumulator->low);

    vpu.vlsat(params->shift1);
    vpu.vstr(&vec_tmp);
    vpu.vldc(params->scale);
    vpu.vclrdr();
    vpu.vlmacc(&vec_tmp);
    vpu.vldc(params->offset_scale);
    vpu.vlmacc(params->offset);

    vpu.vsetc(MODE_S8);

    vpu.vlsat(params->shift2);

    unsigned mask = (1<<channels_out)-1;
    vpu.vstrpv(output, mask);

}


EXTERN_C void conv2d_output_transform_asymmetric_int8(
      int8_t * output,
      const vpu_split_acc32_t* accumulator,
      const nn_acc32_to_int8_params_t* params,
      unsigned const channels_out)
{
    const uint32_t chan_out_mask = (1<<channels_out)-1;
    nn::VPU vpu;
    vpu_vector_t vec_tmp;

    vpu.vldr(vpu_vects.vec_0x80);
    vpu.vstrpv(output, chan_out_mask);

    vpu.vsetc(MODE_S16);

    vpu.vldd(accumulator->high);
    vpu.vldr(accumulator->low);

    vpu.vlsat(params->shift1);

    vpu.vstr(&vec_tmp);
    vpu.vldc(params->scale);
    vpu.vclrdr();
    vpu.vlmacc(&vec_tmp);
    vpu.vldc(params->offset_scale);
    vpu.vlmacc(params->offset);

    vpu.vlsat(params->shift2);

    vpu.vstr(&vec_tmp);
    vpu.vladd(vpu_vects.vec_0x007F);
    vpu.vdepth1();
    uint32_t mask = chan_out_mask & (~vpu.vR().s32[0]);

    vpu.vlashr(&vec_tmp, -8);
    vpu.vdepth8();
    vpu.vstrpv(output, mask);
}