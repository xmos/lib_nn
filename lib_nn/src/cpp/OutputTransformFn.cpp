#include "OutputTransformFn.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <cassert>

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

static int clrsb(int x){
  #if defined(__XS3A__)
  for (unsigned i=0;i<32;i++){
    int y = (x<<i)>>i;
    if (y != x)
      return (i-1);
  }
  return 32;
  #else
  return __builtin_clrsb(x);
  #endif
}

// This puts upper and lower limits on the range of A
// A must reduce the vpu accumulator to 16 bit
// A must not remove all the imformation from the vpu accumulator
static void get_bounds_on_A(int* min_A, int* max_A, int32_t vpu_min_accu,
                            int32_t vpu_max_accu) {
  int32_t max_out = std::max(vpu_min_accu, vpu_max_accu);
  int32_t min_out = std::min(vpu_min_accu, vpu_max_accu);

  int rsb = std::min(clrsb(max_out), clrsb(min_out));

  *max_A = rsb - 16;
  *min_A = *max_A - 16 + 1;
}

// This puts upper and lower limits on the range of Exp
// Exp will be applied to each of the values
// Exp must not saturate and of the values
// Exp must not leave all results as zero
static void get_bounds_on_Exp(int* min_Exp, int* max_Exp, std::vector<float>& values,
                              int bound_width) {
                                
  int max_exponent = std::numeric_limits<int>::min();
  for (float v : values) {
    int e;
    std::frexp(v, &e);
    max_exponent = std::max(max_exponent, e);
  }

  *min_Exp = -max_exponent - 1;
  *max_Exp = *min_Exp + bound_width;
}


std::tuple<int, int, int> solve_constraint(
    std::vector<float> & vpu_output_transform_multiplier,
    std::vector<float> & vpu_output_transform_bias, 

    int32_t vpu_min_accu,
    int32_t vpu_max_accu)
{
  int min_A, max_A;
  int min_B, max_B;
  int min_M, max_M;

  get_bounds_on_A(&min_A, &max_A, vpu_min_accu, vpu_max_accu);

  get_bounds_on_Exp(&min_M, &max_M, vpu_output_transform_multiplier, 16);

  //This is 30 as we cannot make a 32 bit bias with a shr of 14
  get_bounds_on_Exp(&min_B, &max_B, vpu_output_transform_bias, 16 + 14);

  // we also know that A + M = B;
  // Subtract one to ensure the addition is fine (one from A*M, B is already 30 bit at most)
  max_B = std::min(max_A + max_M - 1, max_B);
    
  // printf("min_B:%d max_B:%d\n", min_B, max_B);

  for (int A = max_A; A >= min_A; A--) {
    for (int M = max_M; M >= min_M; M--) {
      // We can squeeze a little more out of the arith by modelling
      // max_Product = max_A * max_M
      // this way we wouldnt need to subtract 2 from max_B

      int B = A + M; 

      if ((B >= min_B) && (B <= max_B)) {
        return std::make_tuple(B, A, M);
      }
    }
  }
  assert(0);
}

template<class T>
void pad(std::vector<T> &vec, int pad_boundary, T pad_val){
  vec.resize(vec.size() + (pad_boundary - vec.size() % pad_boundary) % pad_boundary, pad_val);
}

/*
  This is intended to handle 
*/
QuantisationParams& OTBinary_int8::quantise_activation(
    std::vector<float> & output_transform_multiplier,
    std::vector<float> & output_transform_bias, 
    int32_t accu_min,
    int32_t accu_max)
{

  assert (output_transform_multiplier.size() == output_transform_bias.size());

  QuantisationParams q ;

  // TODO convert to the vpu space

  int B, A, M;
  std::tie(B, A, M) = solve_constraint(output_transform_multiplier, output_transform_bias, accu_min, accu_max);   

  int min_16_bit_B, max_16_bit_B;

  get_bounds_on_Exp(&min_16_bit_B, &max_16_bit_B, output_transform_bias, 16);

  int16_t bias_multipler = 1 << std::max(0, B - max_16_bit_B);
  int adjusted_B = std::min(B, max_16_bit_B);

  std::fill_n(q.otv.bias_multipler, sizeof q.otv.bias_multipler / sizeof bias_multipler, bias_multipler);

  // The -8 is here to leave the result in a 16 bit form so that the quantisation to 8 bit 
  // can deal with the asymertic rounding.
  int16_t final_shr = B - 8; 

  assert(final_shr >= 0);

  std::fill_n(q.otv.final_shr, sizeof q.otv.final_shr / sizeof final_shr, final_shr);

  for (float f : output_transform_multiplier){
    int32_t pa_mul = (int32_t)round(ldexp(f, M));
    assert(clrsb(pa_mul) >= 16); // make sure there is no overflow
    q.multipliers.push_back((int16_t)pa_mul);
  }

  for (float f : output_transform_bias){
    int32_t pa_bias = (int32_t)round(ldexp(f, adjusted_B));

    // assert(clrsb(pa_bias) - 16 >= 0); // make sure there is no overflow
    pa_bias = std::min(INT16_MAX, pa_bias); //TODO think about this
    q.biases.push_back((int16_t)pa_bias);
  }

  //TODO think about who should do this
  //pad q.biases and  q.multipliers to a multiple of VPU_INT16_EPV
  int16_t pad_val = 0; //this is arbitrary
  pad(q.biases, VPU_INT16_EPV, pad_val);
  pad(q.multipliers, (int)VPU_INT16_EPV, pad_val);

  //todo check that post_activation_bias_q * adjusted_B is ldexp(post_activation_bias, B)

  int accu_shr = -A;

  if (accu_shr > 0){
    //use a vlsat
    std::fill_n(q.otv.accu_shr, sizeof q.otv.accu_shr / sizeof accu_shr, accu_shr);
    q.otv.accu_shl = 0;
  } else {
    //use a vashr
    std::fill_n(q.otv.accu_shr, sizeof q.otv.accu_shr / sizeof accu_shr, 0);
    q.otv.accu_shl = accu_shr;
  }

  int16_t no_clamp = 0;
  std::fill_n(q.otv.clamp_near, sizeof q.otv.clamp_near / sizeof no_clamp, no_clamp);
  std::fill_n(q.otv.clamp_far_0, sizeof q.otv.clamp_far_0 / sizeof no_clamp, no_clamp);
  std::fill_n(q.otv.clamp_far_1, sizeof q.otv.clamp_far_1 / sizeof no_clamp, no_clamp);


  // for (unsigned ch = 0; ch < chans_out; ch++){
  //   if (chan_overlaps){
  //     quantised_accu_modifier[ch] = ashr(chan_overlaps[ch], accu_shr);
  //   } else {
  //     quantised_accu_modifier[ch] = 0;
  //   }
  // }

  // int32_t vpu_clamp_min; //TODO
  // int32_t vpu_clamp_max;

  // float min_shifted_accu = ldexp(vpu_clamp_min, - accu_shr);
  // float max_shifted_accu = ldexp(vpu_clamp_max, - accu_shr);

  // int32_t low_clamp_limit = -INT16_MAX * vpu_multipler;
  // int32_t high_clamp_limit = INT16_MAX * vpu_multipler;

  // int32_t t_low_clamp_offset  = (int32_t)((float)low_clamp_limit - min_shifted_accu); //round?
  // int32_t t_high_clamp_offset = (int32_t)((float)high_clamp_limit - max_shifted_accu);

  // int32_t t_clamp_near = t_low_clamp_offset, t_clamp_far_0 = t_high_clamp_offset;
  // if (abs(t_clamp_near) >= abs(t_clamp_far_0)) {
  //   t_clamp_near = t_high_clamp_offset;
  //   t_clamp_far_0 = t_low_clamp_offset;
  // }
  // int32_t t_clamp_far_1 = t_clamp_far_0 / 2;
  // t_clamp_far_0 -= t_clamp_far_1;

  // *clamp_near = -t_clamp_near;
  // *clamp_far_0 = -t_clamp_far_0;
  // *clamp_far_1 = t_clamp_far_1;

  return q;
}

OTBinary_int8::OTBinary_int8(int32_t output_slice_channel_count, output_transform_values_t * otv, 
  int16_t * biases, int16_t * multipliers, int16_t * accu_modifier):
  output_slice_channel_count(output_slice_channel_count), 
  otv(otv),
  biases(biases),
  multipliers(multipliers),
  accu_modifier(accu_modifier)
{

}

int8_t * OTBinary_int8::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group)
{

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  int16_t* cur_post_activation_bias = biases + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_accu_modifier = accu_modifier + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_post_activation_mul = multipliers + output_channel_group * VPU_INT16_EPV;

  VSETC(vpu, MODE_S8);

  vpu_vector_t temp_mem;
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