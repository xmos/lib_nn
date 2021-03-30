
#include "OutputTransformers.hpp"

#include "nn_types.h"
#include "misc.hpp"
#include "xs3_vpu.h"
#include "vpu_sim.h"
#include "../src/asm/asm_constants.h"

using namespace nn;

void Int8OutputTransformHandler::transform(
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


