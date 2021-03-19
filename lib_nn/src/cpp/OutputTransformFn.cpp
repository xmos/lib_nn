#include "OutputTransformFn.hpp"
#include <algorithm>

extern "C" {
  #include "vpu_sim.h"
}

static int64_t saturate_non_sym(
    const int64_t input,
    const unsigned bits)
{
    const int64_t max_val = (((int64_t)1)<<(bits-1))-1;
    const int64_t min_val = -max_val - 1;
    
    return (input > max_val)?  max_val : (input < min_val)? min_val : input;
}

// This is an implementation of VDEPTH8 where the rounding is asymetric
// The acutal asm implements the following but in a more convoluted way 
// in order to work around the rounds issue.
static void VDEPTH8_FIXED(xs3_vpu* vpu){

    vpu_vector_t vec_tmp;
    memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
    memset(&(vpu->vR), 0, sizeof(vpu_vector_t));
    
    for(int i = 0; i < VPU_INT16_EPV; i++){
        int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
        vpu->vR.s8[i] = saturate_non_sym(elm >> 8, 8);
    }
}

OTBinary_int8::OTBinary_int8(){

}

int8_t * OTBinary_int8::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group)
{

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  int16_t* cur_post_activation_bias = biases + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_accu_modifier = accu_modifier + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_post_activation_mul = multipliers + output_channel_group * VPU_INT16_EPV;

  VSETC(vpu, MODE_S8);

  memset(&temp_mem, 0, sizeof(temp_mem));

  //Reduce the accumulator to 16 bits
  VLSAT(vpu, otv->accu_shr);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, otv->accu_shl);

  //Subtract the channel overlap
  VLADD(vpu, cur_accu_modifier);

  VLSUB(vpu, otv->clamp_near);
  VLSUB(vpu, otv->clamp_near);
  VLSUB(vpu, otv->clamp_far_0);
  VLSUB(vpu, otv->clamp_far_1);
  VLSUB(vpu, otv->clamp_far_1);
  VLSUB(vpu, otv->clamp_far_0);

  //Save the 16 bit accumulator, A, to scratch
  VSTR(vpu, &temp_mem);

  //Clear the ring buffer
  VCLRDR(vpu);

  //Multiply the channel-wise bias by the bias multiplier to make it 32 bit per channel
  VLDC(vpu, cur_post_activation_bias);
  VLMACC(vpu, otv->bias_multipler);

  //Multiply A by the post_activation_mul and accumulate it to the bias
  VLDC(vpu, &temp_mem);
  VLMACC(vpu, cur_post_activation_mul);

  //Reduce the accumulator to 16 bits
  VLSAT(vpu, otv->final_shr);

  VDEPTH8_FIXED(vpu);
  
  //we need to know how many we are processing
  int output_count = std::min(output_slice_channel_count - output_channel_group * VPU_INT16_EPV, (int)VPU_INT16_EPV);
  
  int mask = (1<<output_count)-1;

  VSTRPV(vpu, Y, mask);
  Y += output_count;

  return Y;
}

OTBinary_bin::OTBinary_bin(){

}

int8_t * OTBinary_bin::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group)
{
  //this is declared on the stack so that the asm can put the memory
  // in the constant pool
  int16_t zero_mem[32] = {0};

  int16_t * cur_thresholds = thresholds + output_channel_group * VPU_INT16_EPV;
  
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  VSETC(vpu, MODE_S8);
  
  VLDR(vpu, &A->vD);
  VLDD(vpu, &A->vR);

  VLSAT(vpu, &zero_mem);
  VLSAT(vpu, cur_thresholds);
  VDEPTH1(vpu);

  //Do a 16 bit store here
  int16_t * Y16 = (int16_t*)Y;
  Y16[0] = vpu->vR.s16[0];
  Y16 += 1;

  return (int8_t*)Y16;
}