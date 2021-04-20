#include "OutputTransformFn.hpp"
#include <algorithm>
#include <cmath>
#include <tuple>
#include <cassert>
#include <string>
#include <stdio.h>

#include <iostream>

#include "xs3_vpu.h"
#include "vpu_sim.h"
#include "../src/asm/asm_constants.h"


using namespace nn;

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
#if __has_builtin(__builtin_clrsb)
    return __builtin_clrsb(x);
#else
  for (unsigned i=0;i<32;i++){
    int y = (x<<i)>>i;
    if (y != x)
      return (i-1);
  }
  return 32;
#endif
}

static int clrsbll(long long x){
#if __has_builtin(__builtin_clrsbll)
  return __builtin_clrsbll(x);
#else
  for (unsigned i=0;i<64;i++){
    int y = (x<<i)>>i;
    if (y != x)
      return (i-1);
  }
  return 64;
#endif
}

// This puts upper and lower limits on the range of Exp
// Exp will be applied to each of the values
// Exp must not saturate and of the values
// Exp must not leave all results as zero
static void get_bounds_on_Exp(int* min_Exp, int* max_Exp, std::vector<double>& values,
                              int bound_width) 
{
  assert(values.size() > 0);

  int min_value = *std::min_element( std::begin(values), std::end(values));
  int max_value = *std::max_element( std::begin(values), std::end(values));
  
  int exp_of_min, exp_of_max;
  std::frexp(min_value, &exp_of_min);
  std::frexp(max_value, &exp_of_max);

  int max_exponent = std::max(exp_of_min, exp_of_max);

  *min_Exp = -max_exponent - 1;
  *max_Exp = *min_Exp + bound_width;
}

enum quant_error {
  Accu = 0,
  Multiplier = 1,
  Other = 2
};

//v *= 2**exp_adjust; //and round like the vpu does
int16_t quantise_accu(int32_t v, int exp_adjust) throw (quant_error){

  if(exp_adjust < -31)
    return 0;
  
  if(exp_adjust < 0){
    int shr = -exp_adjust;
    int64_t t = v;
    t = t + (1 << ((int16_t)(shr-1)));   //Round
    v = (int32_t) (t >> shr); 
  } else if(exp_adjust < 16) {
    // exp_adjust >= 0
    // shifting left 

    v = (unsigned)v << exp_adjust;
  } else {
    // std::cout << std::string("quantise_accu: exp_adjust_too_high") << " exp_adjust: "  << exp_adjust << " v:" << v<< std::endl;
    throw Accu;
  }

  if(clrsb(v) < 16){
    // std::cout << std::string("quantise_accu :Not enough space for accu") << std::endl;
    throw Accu;
  }

  return (int16_t)v;
}

template<class activationT>
int16_t quantise_multiplier(activationT v, int exp_adjust) throw (quant_error){

  int32_t v_q = std::round(std::ldexp(static_cast<double>(v), exp_adjust));

  if(clrsb(v_q) < 16){
    // std::cout << std::string("Not enough space for multiplier") <<std::endl;
    throw Multiplier;
  }

  return (int16_t)v_q;
}

int32_t quantise_bias(int32_t p, int p_exp_adjust, int exp_adjust) throw (quant_error){

  if(exp_adjust > p_exp_adjust){
    // std::cout << std::string("exp_adjust is too high") <<std::endl;
    throw Other;
  }

  int r = p_exp_adjust - exp_adjust;

  if (r > 31)
    return 0;
  
  if (r < 16){
    int32_t t = (((int64_t)p + (1<<15))&0xffff0000) >> r;

    if(clrsb(t) < 2){
      // std::cout << std::string("bias too big") <<std::endl;
      throw Other;
    }
    return t;
  } else {
    return  ((int64_t)p + (1<<(r-1)))>> r;
  }
}

int32_t compute_product(int16_t a, int16_t m) throw (quant_error){
  int32_t p = (int16_t)a * (int16_t)m;
  return (int32_t)p;
}

int32_t compute_sum(int32_t p, int32_t b) throw (quant_error){
  int64_t r = (int64_t)p + (int64_t)b;
  if(clrsbll(r) < 32){
    // std::cout << std::string("Not enough space for result") <<std::endl;
    throw Other;
  }
    
  return (int32_t)r;
}

template<class activationT>
int get_max_exponent(std::vector<activationT> & arr){

  auto min_arr = *std::min_element( std::begin(arr), std::end(arr));
  auto max_arr = *std::max_element( std::begin(arr), std::end(arr));
  
  int exp_of_min, exp_of_max;
  std::frexp(min_arr, &exp_of_min);
  std::frexp(max_arr, &exp_of_max);

  return std::max(exp_of_min, exp_of_max);

}
// Select B, A, M such that 
// ((accu * 2**A) * (mul * 2**M) + bias*2**B) gives the most precision
template<class activationT>
std::tuple<int, int, int> solve_for_constraint(
    std::vector<activationT> & multiplier,
    std::vector<activationT> & bias, 
    std::vector<int32_t> & accu_min,
    std::vector<int32_t> & accu_max)
{

  int ch_count = accu_min.size();

  assert(multiplier.size() == ch_count);
  assert(bias.size() == ch_count);
  assert(accu_max.size() == ch_count);

  //These could)and should) be refined by inspecting accu_min, accu_max
  int max_A = std::min(15 - get_max_exponent(accu_min), 15 - get_max_exponent(accu_max));
  int max_multiplier_exponent = get_max_exponent(multiplier);
  int max_M = 15 - max_multiplier_exponent;

  // printf("max_multiplier_exponent:%d\n", max_multiplier_exponent);
  //find the largest exponent such that (bias * 2**B) fits in a 32 bit register
  int max_bias_exponent = get_max_exponent(bias);

  //convert bias to a vector of int32_t such that there is no headroom
  int32_t q_bias[ch_count];
  int q_bias_exp_adjust = 31 - max_bias_exponent;
  for (auto ch = 0; ch < ch_count; ++ch){
    int64_t q = (int64_t)std::round(std::ldexp(bias[ch], q_bias_exp_adjust));
    q_bias[ch]  = std::max(std::min(q, (int64_t)INT32_MAX), (int64_t)INT32_MIN);
  }

  int accu_hr = 0;
  int mul_hr = 0;

  //TODO need to think about these, i.e. if A is positive then precision should be given to M
  int A = max_A+1;
  int M = max_M+1;
  
  while (1){
    int B = A + M; 

    try {
      for(auto ch = 0; ch < ch_count; ++ch){

        int16_t m = quantise_multiplier(multiplier[ch], M);
        int32_t b = quantise_bias(q_bias[ch], q_bias_exp_adjust, B);

        //Check the max fits
        int16_t a_max = quantise_accu(accu_max[ch], A);
        compute_sum(compute_product(a_max, m), b);

        //Check the min fits
        int16_t a_min = quantise_accu(accu_min[ch], A);
        compute_sum(compute_product(a_min, m), b);
        
      }
      
      return std::make_tuple(B, A, M);
    } catch (quant_error q){

      //try again but with better numbers
      switch(q){
        case Multiplier: --M; mul_hr++; break;
        case Accu: --A; accu_hr++; break;
        case Other:
          //make a decision based on which one would lose the least data
          //this will do for now
          if(mul_hr > accu_hr){
            --A; accu_hr++;
          } else {
            --M; mul_hr++;
          }
          break;
      }
    }
  }

  printf("fail\n");
  assert(0);
}

template<class T>
void pad(std::vector<T> &vec, int pad_boundary, T pad_val){
  vec.resize(vec.size() + (pad_boundary - vec.size() % pad_boundary) % pad_boundary, pad_val);
}

template<class T, std::size_t S>
static void fill_array(T (&arr)[S], T v){
  std::fill_n(arr, sizeof arr / sizeof (T), v);
}

void xor_popcount_to_vlmaccr1(
    std::vector<int32_t> & accu_min,
    std::vector<int32_t> & accu_max,
    std::vector<int32_t> & accu_overlaps, 
    int32_t accu_clamp_min,
    int32_t accu_clamp_max)
{


}

void calc_post_accumulation_clamps(
    std::vector<int32_t> & accu_min,
    std::vector<int32_t> & accu_max,
    std::vector<int32_t> & accu_overlaps, 
    int32_t accu_clamp_min,
    int32_t accu_clamp_max,
    int accu_shr)
{
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
}

QuantisationParams OutputTransformFnInt8::quantise_activation(
    std::vector<double> & output_transform_multiplier,
    std::vector<double> & output_transform_bias, 
    std::vector<int32_t> & accu_min,
    std::vector<int32_t> & accu_max )
{

  assert (output_transform_multiplier.size() == output_transform_bias.size());

  int B, A, M;
  std::tie(B, A, M) = solve_for_constraint(output_transform_multiplier, output_transform_bias, accu_min, accu_max);   

  int min_16_bit_B, max_16_bit_B;

  get_bounds_on_Exp(&min_16_bit_B, &max_16_bit_B, output_transform_bias, 16);

  int16_t bias_multipler = (int16_t)(1 << std::max(0, B - max_16_bit_B));
  int adjusted_B = std::min(B, max_16_bit_B);

  QuantisationParams q;

  std::fill_n(q.otv.bias_multipler, sizeof q.otv.bias_multipler / sizeof bias_multipler, bias_multipler);

  // The -8 is here to leave the result in a 16 bit form so that the quantisation to 8 bit 
  // can deal with the asymertic rounding.
  int16_t final_shr = B - 8; 
  assert(final_shr >= 0);
  fill_array(q.otv.final_shr, final_shr);

  for (auto f : output_transform_multiplier){
    int32_t pa_mul = (int32_t)round(ldexp(f, M));
    assert(clrsb(pa_mul) >= 16); // make sure there is no overflow
    q.multipliers.push_back((int16_t)pa_mul);
  }

  for (auto f : output_transform_bias){
    int32_t pa_bias = (int32_t)round(ldexp(f, adjusted_B));
    pa_bias = std::min(INT16_MAX, pa_bias); //TODO think about this
    q.biases.push_back((int16_t)pa_bias);
  }

  //TODO think about what should own this
  //pad q.biases and  q.multipliers to a multiple of VPU_INT16_EPV
  int16_t pad_val = 0; //this is arbitrary
  pad(q.biases, VPU_INT16_EPV, pad_val);
  pad(q.multipliers, (int)VPU_INT16_EPV, pad_val);

  int16_t accu_shr = -A;
  if (accu_shr > 0){
    //use a vlsat
    fill_array(q.otv.accu_shr, accu_shr);
    q.otv.accu_shl = 0;
  } else {
    //use a vashr
    fill_array(q.otv.accu_shr, (int16_t)0);
    q.otv.accu_shl = accu_shr;
  }

  //TODO put this else where
  // int16_t no_clamp = 0;
  // fill_array(q.otv.clamp_near, no_clamp);
  // fill_array(q.otv.clamp_far_0, no_clamp);
  // fill_array(q.otv.clamp_far_1, no_clamp);
  
  return q;
}

int8_t * OT_int8::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group)
{

  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  int16_t* cur_post_activation_bias = params->biases + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_accu_modifier = params->accu_modifier + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_post_activation_mul = params->multipliers + output_channel_group * VPU_INT16_EPV;

  VSETC(vpu, MODE_S16);//check this

  VLDR(vpu, &A->vD);
  VLDD(vpu, &A->vR);

  // vpu_sim_print(vpu);

  vpu_vector_t temp_mem;
  memset(&temp_mem, 0, sizeof(temp_mem));

  //Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->accu_shr);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->otv->accu_shl);

  // printf("a\n");
  // vpu_sim_print(vpu);

  //Subtract the channel overlap
  VLADD(vpu, cur_accu_modifier);
  
  //Save the 16 bit accumulator, A, to scratch
  VSTR(vpu, &temp_mem);

  //Clear the ring buffer
  VCLRDR(vpu);

  //Multiply the channel-wise bias by the bias multiplier to make it 32 bit per channel
  VLDC(vpu, cur_post_activation_bias);
  VLMACC(vpu, params->otv->bias_multipler);

  // printf("b\n");
  // vpu_sim_print(vpu);

  //Multiply A by the post_activation_mul and accumulate it to the bias
  VLDC(vpu, &temp_mem);
  VLMACC(vpu, cur_post_activation_mul);
  // printf("c\n");
  // vpu_sim_print(vpu);

  //Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->final_shr);

  // printf("d\n");
  // vpu_sim_print(vpu);
  VDEPTH8_FIXED(vpu);
  
  //we need to know how many we are processing
  int output_count = std::min(params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV, (int)VPU_INT16_EPV);
  
  int mask = (1<<output_count)-1;

  VSTRPV(vpu, Y, mask);
  Y += output_count;

  return Y;
}

// OT_int8::Params::Params(
//   const int output_ch_count,
//   const int elements_per_channel,
//   const int8_t kernel_weights[],
//   const int32_t biases[],
//   const float effective_output_multiplier[],
//   const int8_t input_zero_point,
//   const int8_t output_zero_point): 
//     output_slice_channel_count(output_ch_count)
// {
//   /*
//       output_transform_values_t * otv;
//       int16_t * biases;//[output_slice_channel_count];
//       int16_t * multipliers;//[output_slice_channel_count];
//       int16_t * accu_modifier;//[output_slice_channel_count];
//   */

//   std::vector<double> output_transform_multiplier(output_ch_count);
//   std::vector<double> output_transform_bias(output_ch_count);
//   std::vector<int32_t> accu_min(output_ch_count);
//   std::vector<int32_t> accu_max(output_ch_count);


//   for (int output_ch = 0; output_ch < output_ch_count; output_ch++){
//     for(int element = 0; element < elements_per_channel; element++){
//       int32_t v = (int32_t)kernel_weights[element + output_ch * elements_per_channel];
//       accu_min[output_ch] += v * (v > 0 ? INT8_MIN:INT8_MAX);
//       accu_max[output_ch] += v * (v > 0 ? INT8_MAX:INT8_MIN);
//     }
//     output_transform_multiplier[output_ch] = effective_output_multiplier[output_ch];

//     output_transform_bias[output_ch] = biases[output_ch] + (double)input_zero_point * elements_per_channel - output_zero_point;
//   }

//   QuantisationParams qp = quantise_activation( output_transform_multiplier, output_transform_bias, accu_min, accu_max );

//   output_transform_values_t o;
  
//   // biases
// }

OTBinary_int8::Params::Params(int32_t output_slice_channel_count, OutputTransformValuesClamping * otv, 
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

  int16_t* cur_post_activation_bias = params->biases + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_accu_modifier = params->accu_modifier + output_channel_group * VPU_INT16_EPV;
  int16_t* cur_post_activation_mul = params->multipliers + output_channel_group * VPU_INT16_EPV;

  VSETC(vpu, MODE_S16);//check this

  VLDR(vpu, &A->vD);
  VLDD(vpu, &A->vR);

  // vpu_sim_print(vpu);

  vpu_vector_t temp_mem;
  memset(&temp_mem, 0, sizeof(temp_mem));

  //Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->accu_shr);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->otv->accu_shl);

  // printf("a\n");
  // vpu_sim_print(vpu);

  //Subtract the channel overlap
  VLADD(vpu, cur_accu_modifier);

  VLSUB(vpu, params->otv->clamp_near);
  VLSUB(vpu, params->otv->clamp_near);
  VLSUB(vpu, params->otv->clamp_far_0);
  VLSUB(vpu, params->otv->clamp_far_1);
  VLSUB(vpu, params->otv->clamp_far_1);
  VLSUB(vpu, params->otv->clamp_far_0);

  //Save the 16 bit accumulator, A, to scratch
  VSTR(vpu, &temp_mem);

  //Clear the ring buffer
  VCLRDR(vpu);

  //Multiply the channel-wise bias by the bias multiplier to make it 32 bit per channel
  VLDC(vpu, cur_post_activation_bias);
  VLMACC(vpu, params->otv->bias_multipler);

  // printf("b\n");
  // vpu_sim_print(vpu);

  //Multiply A by the post_activation_mul and accumulate it to the bias
  VLDC(vpu, &temp_mem);
  VLMACC(vpu, cur_post_activation_mul);
  // printf("c\n");
  // vpu_sim_print(vpu);

  //Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->final_shr);

  // printf("d\n");
  // vpu_sim_print(vpu);
  VDEPTH8_FIXED(vpu);
  
  //we need to know how many we are processing
  int output_count = std::min(params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV, (int)VPU_INT16_EPV);
  
  int mask = (1<<output_count)-1;

  VSTRPV(vpu, Y, mask);
  Y += output_count;

  return Y;
}

OTBinary_bin::OTBinary_bin(int16_t * thresholds): thresholds(thresholds){

}

int8_t * OTBinary_bin::output_transform_fn(int8_t * Y, vpu_ring_buffer_t * A, int32_t output_channel_group)
{
  //this is declared on the stack so that the asm can put the memory
  // in the constant pool
  int16_t zero_mem[32] = {0};

  int16_t * cur_thresholds = thresholds + output_channel_group * VPU_INT16_EPV;
  
  xs3_vpu vpu_mem;
  xs3_vpu * vpu = &vpu_mem;

  VSETC(vpu, MODE_S16); //check this - i dont think it's needed
  
  VLDR(vpu, &A->vD);
  VLDD(vpu, &A->vR);
  
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



/******************************
 * DirectWriteOutputTransform
 *****************************/

constexpr int DirectWriteOutputTransform::ChannelsPerOutputGroup;


////////// DirectWriteOutputTransform::Params //////////////
DirectWriteOutputTransform::Params::Params(const int image_channels) 
    : output_img_channels(image_channels)
{
}

DirectWriteOutputTransform::Params::Params(const nn::ImageGeometry& output_image) 
    : output_img_channels(output_image.depth)
{
}

DirectWriteOutputTransform::Params::Params(std::istream& stream)
{
  stream.read(reinterpret_cast<char*>(&this->output_img_channels), sizeof(int32_t));
}

void DirectWriteOutputTransform::Params::Serialize(std::ostream& stream) const
{
  stream.write(reinterpret_cast<const char*>(&this->output_img_channels), sizeof(this->output_img_channels));
}



////////// DirectWriteOutputTransform //////////////
DirectWriteOutputTransform::DirectWriteOutputTransform(const Params* params)
    : params(params)
{
}


int8_t * DirectWriteOutputTransform::output_transform_fn(int8_t * Y,
                                                         vpu_ring_buffer_t * acc,
                                                         int32_t output_channel_group)
{

  const int32_t first_channel = DirectWriteOutputTransform::ChannelsPerOutputGroup * output_channel_group;
  const int32_t count = std::min<int32_t>(DirectWriteOutputTransform::ChannelsPerOutputGroup, 
                                    this->params->output_img_channels - first_channel);

#ifdef NN_USE_REF
  std::memcpy(Y, &acc->vR[0], count);
#else
  volatile asm("mkmsk %0, %0" : "=r"(count));
  volatile asm("vldr %0[0]" : "r"(%acc->vR[0]));
  volatile asm("vstrpv %0[0], %1" : "r"(Y, count));
#endif // NN_USE_REF

return &Y[count];
}

/******************************
 * ShiftInt8OutputTransform
 *****************************/

constexpr int ShiftInt8OutputTransform::ChannelsPerOutputGroup;

////////// ShiftInt8OutputTransform::Params //////////////
ShiftInt8OutputTransform::Params::Params(const int image_channels, const int16_t* shifts) 
    : output_img_channels(image_channels), shifts(shifts)
{
}

ShiftInt8OutputTransform::Params::Params(const nn::ImageGeometry& output_image, const int16_t* shifts) 
    : output_img_channels(output_image.depth), shifts(shifts)
{
}

ShiftInt8OutputTransform::Params::Params(std::istream& stream, const int16_t* shifts)
    : shifts(shifts)
{
  stream.read(reinterpret_cast<char*>(&this->output_img_channels), sizeof(int32_t));
}

void ShiftInt8OutputTransform::Params::Serialize(std::ostream& stream) const
{
  stream.write(reinterpret_cast<const char*>(&this->output_img_channels), sizeof(this->output_img_channels));
}



////////// ShiftInt8OutputTransform //////////////
ShiftInt8OutputTransform::ShiftInt8OutputTransform(const Params* params)
    : params(params)
{
}


C_API void shift_int8_output_transform_ref(
    int8_t* output,
    const vpu_ring_buffer_t* acc,
    const int16_t* right_shifts,
    const int channel_count)
{
  uint32_t write_mask = uint32_t((1LL << channel_count) - 1);

  auto vpu = nn::VPU();
  vpu_vector_t vec_tmp;
  uint32_t tmp;

  vpu.vldr( vpu_vect_0x80 );
  vpu.vstrpv( output, write_mask );

  vpu.vldd(acc->vD);
  vpu.vldr(acc->vR);
  vpu.vsetc(MODE_S16);
  vpu.vlsat(right_shifts);
  vpu.vstr( &vec_tmp );
  vpu.vladd( vpu_vect_0x007F );
  vpu.vdepth1();
  vpu.vstrpv( &tmp, 0x0000000F );
  write_mask = write_mask & ~tmp;
  vpu.vlashr( &vec_tmp, -8 );
  vpu.vdepth8();
  vpu.vstrpv( output, write_mask );

}

int8_t * ShiftInt8OutputTransform::output_transform_fn(int8_t * Y,
                                                       vpu_ring_buffer_t * acc,
                                                       int32_t output_channel_group)
{
  const int32_t first_channel = ShiftInt8OutputTransform::ChannelsPerOutputGroup * output_channel_group;
  const int32_t count = std::min<int32_t>(ShiftInt8OutputTransform::ChannelsPerOutputGroup, 
                                    this->params->output_img_channels - first_channel);

  const int16_t* shifts = &this->params->shifts[first_channel];

#ifdef NN_USE_REF
  shift_int8_output_transform_ref(Y, acc, shifts, count);
#else
  shift_int8_output_transform_asm(Y, acc, shifts, count);
#endif // NN_USE_REF

return &Y[count];
}