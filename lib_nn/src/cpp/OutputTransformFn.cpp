#include "OutputTransformFn.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <tuple>

extern "C"
{
#include "../src/asm/asm_constants.h"
#include "vpu_sim.h"
#include "xs3_vpu.h"
}

using namespace nn;

  static int32_t sat(int64_t a, int bits){
    int64_t max_val = (1LL << bits)-1;
    int64_t min_val = -(1LL << bits);

    if(a > max_val)
      return (int32_t)max_val;

    if(a < min_val)
      return (int32_t)min_val;

    return a;
  }

  static int32_t shr(int32_t val, int shr_amount, int bits = 16 ){
    if (shr_amount > 0){
      return sat(((int64_t)val + (1LL << (shr_amount - 1))) >> shr_amount, bits);
    } else {
      return sat((int64_t)val << (-shr_amount), bits);
    }
  }

  static int32_t add(int32_t a, int32_t b, int bits = 16 ){
      return sat((int64_t)a + (int64_t)b, bits);
  }

  static int32_t mul(int32_t a, int32_t b, int bits = 16 ){
      int64_t prod = (int64_t)a * (int64_t)b;
      prod = prod + (1LL << (14-1));
      return sat(prod>>14, bits);
  }


static int64_t saturate_non_sym(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  const int64_t min_val = -max_val - 1;

  return (input > max_val) ? max_val : (input < min_val) ? min_val : input;
}

// This is an implementation of VDEPTH8 where the rounding is asymetric
// The acutal asm implements the following but in a more convoluted way
// in order to work around the rounds issue.
static void VDEPTH8_FIXED(xs3_vpu *vpu) {
  vpu_vector_t vec_tmp;
  memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
  memset(&(vpu->vR), 0, sizeof(vpu_vector_t));

  for (int i = 0; i < VPU_INT16_EPV; i++) {
    int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
    vpu->vR.s8[i] = saturate_non_sym(elm >> 8, 8);
  }
}

static int clrsbll(long long x) {
#if __has_builtin(__builtin_clrsbll)
  return __builtin_clrsbll(x);
#else
  for (unsigned i = 0; i < 64; i++) {
    int y = (x << i) >> i;
    if (y != x) return (i - 1);
  }
  return 64;
#endif
}

static int64_t shl(int32_t v, int amount_to_shl) {
  if (amount_to_shl >= 0) {
    // work around  the undefined behaviour
    uint64_t mask = (~0LLU) >> amount_to_shl;
    return ((uint64_t)v & mask) << amount_to_shl;
  } else {
    int amount_to_shr = -amount_to_shl;
    return ((int64_t)v + (1LL<<(amount_to_shr-1))) >> amount_to_shr;
  }
}

/**
 * Return the number of bits required to hold v (round up to the next highest
 * integer).
 */
static int count_bits(int64_t v) {
#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif
  return ((sizeof(int64_t) * CHAR_BIT) - clrsbll(v));
}

// Return true if v can be represented in bit_count bits
static bool check_val_fits(int64_t v, int bit_count) {
  return count_bits(v) <= bit_count;
}

template <class T>
void recitfy_min_max(T &v_min, T &v_max) {
  T actual_max = std::max(v_min, v_max);
  v_min = std::min(v_min, v_max);
  v_max = actual_max;
}
  template <class activationT>
  static int get_max_exponent(activationT f) {
    int e;
    std::frexp(f, &e);
    return e;
  }

  template <class activationT>
  static int get_max_exponent(std::vector<activationT> &arr) {
    int m = INT32_MIN;
    for (auto f : arr) m = std::max(m, get_max_exponent(f));
    return m;
  }

// Select A, M such that
// ((accu * 2**A) * (mul * 2**M) + bias*2**B) gives the most precision
// accu_min and accu_max should be pairwise correct, i.e. min is the min, max is
// the max for each channel.

std::tuple<int, int> solve_for_constraints(OutputTransformFn::MulsAndBias &activationParams, int vlmul_shr,
                                           bool verbose = false) {
  int accu_bits_max = 0;
  int max_multiplier_exponent = INT32_MIN;

  // If all the accumulators or multipliers are zero then there is no defined
  // range.
  bool accu_range_defined = false;
  bool multiplier_range_defined = false;

  for (auto activationParam : activationParams) {
    if (activationParam.accu_max_val) {
      int accu_max_bits =
          get_max_exponent(activationParam.accu_max_val);
      accu_bits_max = std::max(accu_bits_max, accu_max_bits);
      accu_range_defined |= true;
    }
    if (activationParam.accu_min_val) {
      int accu_min_bits =
          get_max_exponent(activationParam.accu_min_val);
      accu_bits_max = std::max(accu_bits_max, accu_min_bits);
      accu_range_defined |= true;
    }
    if (activationParam.multiplier) {
      int multiplier_bits =
          get_max_exponent(activationParam.multiplier);
      max_multiplier_exponent =
          std::max(max_multiplier_exponent, multiplier_bits);
      multiplier_range_defined |= true;
    }
  }

  // If either of the ranges are undefined(i.e. all zero) then the result is the
  // same: the product contributes nothing
  bool product_range_defined = multiplier_range_defined && accu_range_defined;

  if (!product_range_defined) {
    if (verbose) printf("undefined product range\n");

    // Then we only care about the biases -> we know they will always fit in an
    // 8 bit number so a bias exp of 0 will do fine.
    int A = 0;
    int M = vlmul_shr;
    int B = A + M - vlmul_shr;
    assert(B == 0);
    return std::make_tuple(A, M);
  }

  int max_A = 15 - accu_bits_max;
  int max_M = 15 - max_multiplier_exponent;

  int A = max_A;
  int M = max_M;

  int mul_sig_bits = 0, accu_sig_bits = 0;

  for (auto activationParam : activationParams) {
    int64_t mul_16 = std::round(ldexp(activationParam.multiplier, M));
    mul_sig_bits = std::max(mul_sig_bits, count_bits(mul_16));

    int64_t accu_max_16 = shl(activationParam.accu_max_val, A);
    int64_t accu_min_16 = shl(activationParam.accu_min_val, A);

    accu_sig_bits = std::max(accu_sig_bits, std::max(count_bits(accu_max_16),
                                                     count_bits(accu_min_16)));
  }

  if (verbose) {
    printf("accu_sig_bits: %d\nmul_sig_bits: %d\nA: %d\n M: %d\n",
           accu_sig_bits, mul_sig_bits, A, M);
  }

  bool trying = true;

  int64_t max_group_prod, min_group_prod, max_group_sum, min_group_sum;
  while (trying) {
    trying = false;

    max_group_prod = INT32_MIN;
    min_group_prod = INT32_MAX;
    max_group_sum = INT32_MIN;
    min_group_sum = INT32_MAX;

    for (auto activationParam : activationParams) {
      // check
      // accu*2**A fit in 16 bits
      // mul*2**B fit in 16 bits
      // bias*(2**(A+B-VLMUL_SHR)) fit in 16 bits
      // (must be representable by 16bit *
      // (1<<x)) (accu*2**A)*(mul*2**B) fit in 32 bits (accu*2**A)*(mul*2**B) +
      // bias*(2**(A+B)) fit in 32 bits

      int64_t accu_max_16 = shl(activationParam.accu_max_val, A);
      int64_t accu_min_16 = shl(activationParam.accu_min_val, A);
      if (!check_val_fits(accu_max_16, 16) ||
          !check_val_fits(accu_min_16, 16)) {
        A--;
        accu_sig_bits--;
        trying = true;

        if (verbose) {
          printf("Accu too big\n   accu_sig_bits: %d\n   A: %d\n",
                 accu_sig_bits, A);
        }
        break;
      }

      int64_t mul_16 = std::round(ldexp(activationParam.multiplier, M));
      if (!check_val_fits(mul_16, 16)) {
        M--;
        mul_sig_bits--;
        trying = true;

        if (verbose) {
          printf("mul too big\n   mul_sig_bits: %d\n   M: %d\n", mul_sig_bits,
                 M);
        }
        break;
      }

      int64_t bias_16 =
          std::round(ldexp(activationParam.bias, A + M - vlmul_shr));

      int64_t prod_max = shl(accu_max_16 * mul_16, -vlmul_shr);
      int64_t prod_min = shl(accu_min_16 * mul_16, -vlmul_shr);

      max_group_prod = std::max(max_group_prod, prod_max);
      min_group_prod = std::min(min_group_prod, prod_min);

      recitfy_min_max(prod_min, prod_max);

      int64_t sum_max = prod_max + bias_16;
      int64_t sum_min = prod_min + bias_16;

      max_group_sum = std::max(max_group_sum, sum_max);
      min_group_sum = std::min(min_group_sum, sum_min);

      // at least one of these must be true
      // one of them can saturate
      if (!check_val_fits(prod_max, 16) || !check_val_fits(prod_min, 16) ||
          !check_val_fits(sum_max, 16) || !check_val_fits(sum_min, 16) ||
          !check_val_fits(bias_16, 16)) {
        if (verbose) printf("overflow in prod or sum \n");
        if (A >= 0 || accu_sig_bits > mul_sig_bits) {
          A--;
          accu_sig_bits--;
          if (verbose) {
            printf("   accu_sig_bits: %d\n   A: %d\n", accu_sig_bits, A);
          }
        } else {
          M--;
          mul_sig_bits--;
          if (verbose) {
            printf("   mul_sig_bits: %d\n   M: %d\n", mul_sig_bits, M);
          }
        }

        trying = true;
        break;
      }
    }

    // want to have the same number of significant bits of accu and mul
    // must use as all 16 bits of the bias
  }
  if (verbose) {
    printf(
        "max_group_prod: %lld\nmin_group_prod: %lld\n"
        "max_group_sum: %lldmin_group_sum: %lld\n"
        "mul_sig_bits: %d\naccu_sig_bits: %d\n",
        max_group_prod, min_group_prod, max_group_sum, min_group_sum,
        mul_sig_bits, accu_sig_bits);
  }
  return std::make_tuple(A, M);
}

int32_t round_away_from_zero(float x) {
  if (x > 0)
    return std::ceil(x);
  else
    return std::floor(x);
}

int64_t round_up(float x) { return std::ceil(x); }

int64_t round_down(float x) { return std::floor(x); }

void nn::OutputTransformFn::ActivationParams::backprop_output_clamps_to_accu_limits(
  int64_t output_high,
  int64_t output_low,
  bool verbose) {
  
  // adjust accu_min and max to account for the saturation on the output
  if (multiplier == 0.0) {
    multiplier = 0.0;
    bias = 0.0;
    accu_min_val = 0;
    accu_max_val = 0;
    return;
  }
  recitfy_min_max(output_low, output_high);

  double hi =
      ((double)output_high - bias) / multiplier;
  double lo =
      ((double)output_low - bias) / multiplier;

  int64_t accu_meaningful_max = round_up(hi);
  int64_t accu_meaningful_min = round_down(lo);

  recitfy_min_max(accu_meaningful_min, accu_meaningful_max);

  if (verbose) {
    printf("accu_meaningful_min: %lld accu_meaningful_max: %lld\n",
            accu_meaningful_min, accu_meaningful_max);
    printf(
        "accu_min_val: %d accu_max_val: %d\n",
        accu_min_val, accu_max_val);
  }

  if (accu_meaningful_max < (int64_t)accu_max_val) {
    // ------accu_max------accu_meaningful_max++++++...

    // The input range of the accumulator is restricted by the output clamp.
    if (accu_meaningful_max < (int64_t)accu_min_val) {
      // ------accu_max------accu_min------accu_meaningful_max------...
      // There is no overlap between what the accumulator can be and what
      // could contribute to the output

      int32_t singular_output_value = std::max(
          std::min((int32_t)output_high, (int32_t)original_bias),
          (int32_t)output_low);

      multiplier = 0.0;
      bias = singular_output_value;

      accu_max_val = 0;
      accu_min_val = 0;

    } else {
      // ------accu_max------accu_meaningful_max++++++accu_min------...
      accu_max_val = accu_meaningful_max;
      if (accu_meaningful_min > (int64_t)accu_min_val) {
        // ------accu_max------accu_meaningful_max++++++accu_meaningful_min------accu_min------...
        accu_min_val = accu_meaningful_min;
      } else {
        // ------accu_max------accu_meaningful_max++++++accu_min------accu_meaningful_min------...
        // accu_min_val = accu_min_val;
      }
    }
  } else {
    // ------accu_meaningful_max------accu_max++++++...
    if (accu_meaningful_min > (int64_t)accu_max_val) {
      // ------accu_meaningful_max------accu_meaningful_min------accu_max------...
      // There is no overlap between what the accumulator can be and what
      // could contribute to the output
      int32_t singular_output_value = std::max(
          std::min((int32_t)output_high, (int32_t)original_bias),
          (int32_t)output_low);

      multiplier = 0.0;
      bias = singular_output_value;
      accu_max_val = 0;
      accu_min_val = 0;

    } else {
      // ------accu_meaningful_max------accu_max++++++accu_meaningful_min------...
      // accu_max_val = accu_max_val;

      if (accu_meaningful_min > (int64_t)accu_min_val) {
        // ------accu_meaningful_max------accu_max++++++accu_meaningful_min------accu_min------...
        accu_min_val = accu_meaningful_min;
      } else {
        // ------accu_meaningful_max------accu_max++++++accu_min------accu_meaningful_min------...
        // accu_min_val = accu_min_val;
      }
    }
  }
}

template <class T>
int16_t float_to_int16(T f, int e) {
  int32_t v = (int32_t)round(ldexp(f, e));

  assert(v <= INT16_MAX);
  assert(v >= INT16_MIN);
  v = std::min((int32_t)INT16_MAX, v);
  v = std::max((int32_t)INT16_MIN, v);
  return (int16_t)v;
}

QuantisationPerGroupStrategy::QuantisationPerGroupStrategy(
    OutputTransformFn::MulsAndBias &activationParams, 
    bool verbose): QantisationStrategy(activationParams){

  if (activationParams.size() == 0) {
    initial_shr = 0;
    final_shr = 0;
    return;
  }

  int A, M;

  int vlmul_shr = 14;

  std::tie(A, M) = solve_for_constraints(activationParams, verbose, vlmul_shr);

  int B = A + M - vlmul_shr;

  initial_shr = -A;
  final_shr = B - 8;

  if (verbose) {
    printf("final_shr: %d\n", final_shr);
  }

  // Quantise the multiplier and bias
  for (int ch = 0; ch < activationParams.size(); ++ch) {
    int16_t m = float_to_int16(activationParams[ch].multiplier, M);
    multipliers.push_back(m);
    int16_t b = float_to_int16(activationParams[ch].bias, B);
    biases.push_back(b);

    if (verbose)
      printf("multiplier: %d(%f) original: %f\n", m, std::ldexp(m, -M),
             activationParams[ch].original_multiplier);
    if (verbose)
      printf("bias: %d(%f) original: %f %f\n", b, std::ldexp(b, -B),
             activationParams[ch].original_bias, activationParams[ch].bias);
  }
}

//this one quantises with a per channel shift
QuantisationPerChannelStrategy::QuantisationPerChannelStrategy(
    OutputTransformFn::MulsAndBias &activationParams, bool verbose): QantisationStrategy(activationParams)
{
  if (activationParams.size() == 0)
    return;

  std::vector<int> M;
  std::vector<int> A;

  for (int ch=0;ch<activationParams.size();ch++){
    std::vector<OutputTransformFn::ActivationParams> slice;
    slice.push_back(activationParams[ch]);
    int a, m;
    std::tie(a, m) = solve_for_constraints(activationParams, verbose, 15);
    M.push_back(m);
    A.push_back(a);
  }
  
  //now combine the A

}

/**
 * Calculate the maximum average error between the reference and quantised implementations of the output 
 * transform over each channel. The average is defined over the range of non-saturating accumulators, 
 * i.e. accumulators that do not reach a saturating output in the int8 space. 
 * The result is the maximum average for all of the channels. 
 */
double QuantisationPerGroupStrategy::get_quant_error(
  bool use_high_precision)
{
  
  if (use_high_precision){

    double max_avg_abs_error = 0.0;

    for (int idx=0; idx < activation_params.size(); ++idx){
      
      int64_t abs_error_sum = 0;

      for(int accu = activation_params[idx].accu_min_val; accu <= activation_params[idx].accu_max_val; ++accu){

        int32_t t = shr(accu, initial_shr); //vlsat
        t = mul(t, multipliers[idx]);       //vlmul
        t = add(t, biases[idx]);            //vladd
        t = shr(t, final_shr);              //vlashr
        t = sat(shr(t, 8), 8);              //vdepth8
        
        double v = (double)accu * activation_params[idx].multiplier + activation_params[idx].bias;

        int expected = (int)std::round(v);

        expected = std::min(expected, INT8_MAX);
        expected = std::max(expected, INT8_MIN);

        abs_error_sum += std::abs(expected - t);
      }

      int64_t interesting_accumulators =  activation_params[idx].accu_max_val - activation_params[idx].accu_min_val + 1;
      
      if (interesting_accumulators > 0){
        double avg_abs_error = (double)abs_error_sum / (double)interesting_accumulators;
        max_avg_abs_error = std::max(max_avg_abs_error, avg_abs_error);
      }
    }
    return max_avg_abs_error;
  } else {
    //final_shr | number of decimal places | error
    //-8          0                          0.5   = 1*2*0
    //-7          1                          0.25  = 1*2*-1
    //-6          2                          0.125 = 1*2*-2
    //-5          3                          2*-3
    //-4          4                          2*-4
    //-3          5                          2*-5
    //-2          6                          2*-6
    //-1          7                          2*-7
    //0           8                          2*-8
    return std::ldexp(1, final_shr + 8);
  }
}


















extern "C" int8_t *output_transform_fn_impl_asm(const OTPerGroup::Params *params,
                                                int8_t *Y, VPURingBuffer *A,
                                                int16_t *multipliers_and_biases,
                                                int output_count);

#ifndef NN_USE_REF
int8_t *output_transform_fn_impl_asm_stub(const OTPerGroup::Params *params,
                                          int8_t *Y, VPURingBuffer *A,
                                          int32_t output_channel_group,
                                          int16_t *multipliers_and_biases) {
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);
  multipliers_and_biases += output_channel_group * VPU_INT16_EPV * 2;
  return output_transform_fn_impl_asm(params, Y, A, multipliers_and_biases,
                                      output_count);
}
#endif

int8_t *output_transform_fn_impl(const OTPerGroup::Params *params, int8_t *Y,
                                 VPURingBuffer *A, int32_t output_channel_group,
                                 int16_t *multipliers_and_biases) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  // we need to know how many we are processing
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);
  int16_t *cur_post_activation_mul =
      multipliers_and_biases + output_channel_group * VPU_INT16_EPV * 2;

  int16_t *cur_post_activation_bias = cur_post_activation_mul + output_count;

  VSETC(vpu, MODE_S16);

  VLDR(vpu, &A->vR);
  VLDD(vpu, &A->vD);

  vpu_vector_t temp_mem;

  if (params->initial_shift > 0) {
    for (int i = 0; i < VPU_INT16_EPV; ++i)
      temp_mem.s16[i] = params->initial_shift;

    VLSAT(vpu, &temp_mem);
  } else {
    for (int i = 0; i < VPU_INT16_EPV; ++i) temp_mem.s16[i] = 0;
    VLSAT(vpu, &temp_mem);

    VSTR(vpu, &temp_mem);
    VLASHR(vpu, &temp_mem, params->initial_shift);
  }

  VLMUL(vpu, cur_post_activation_mul);

  VLADD(vpu, cur_post_activation_bias);

  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->final_shr);

  VDEPTH8_FIXED(vpu);

  int mask = (1 << output_count) - 1;
  VSTRPV(vpu, Y, mask);
  Y += output_count;

  return Y;
}

int8_t *OTPerGroup::output_transform_fn(int8_t *Y, VPURingBuffer *A,
                                     int32_t output_channel_group) {
#ifdef NN_USE_REF
  return output_transform_fn_impl(this->params, Y, A, output_channel_group,
                                  multipliers_and_biases);
#else
  return output_transform_fn_impl_asm_stub(
      this->params, Y, A, output_channel_group, multipliers_and_biases);
#endif  // NN_USE_REF
}

int8_t *output_transform_fn_int_clamped_impl(
    const OTPerGroupClamped::Params *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *offsets_multipliers_and_biases) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  // we need to know how many we are processing
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);

  // Thge 3 is due to the serialisation of 3 arrays in chunks of VPU_INT16_EPV
  // elements
  int16_t *cur_post_activation_offset =
      offsets_multipliers_and_biases + output_channel_group * VPU_INT16_EPV * 3;

  int16_t *cur_post_activation_mul = cur_post_activation_offset + output_count;

  int16_t *cur_post_activation_bias = cur_post_activation_mul + output_count;

  VSETC(vpu, MODE_S16);
  VLDR(vpu, &A->vR);
  ;
  VLADD(vpu, cur_post_activation_offset);

  // Remove the kernel overlap
  VPOS(vpu);

  vpu_vector_t temp_mem;
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->initial_shift);

  VLMUL(vpu, cur_post_activation_mul);
  VLADD(vpu, cur_post_activation_bias);

  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->final_shr);

  VDEPTH8_FIXED(vpu);

  int mask = (1 << output_count) - 1;
  VSTRPV(vpu, Y, mask);
  Y += output_count;

  return Y;
}

extern "C" int8_t *output_transform_fn_int_clamped_impl_asm(
    const OTPerGroupClamped::Params *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *offsets_multipliers_and_biases);

int8_t *OTPerGroupClamped::output_transform_fn(int8_t *Y, VPURingBuffer *A,
                                             int32_t output_channel_group) {
#ifdef NN_USE_REF

  return output_transform_fn_int_clamped_impl(
      this->params, Y, A, output_channel_group, offsets_multipliers_and_biases);
#else

  return output_transform_fn_int_clamped_impl_asm(
      this->params, Y, A, output_channel_group, offsets_multipliers_and_biases);
#endif  // NN_USE_REF
}

int8_t *output_transform_fn_binary_impl(int8_t *Y, VPURingBuffer *A,
                                        int32_t output_channel_group,
                                        threshold_t *thresholds) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  threshold_t *cur_thresholds =
      thresholds + output_channel_group * VPU_INT16_EPV;

  // do we need this?
  VSETC(vpu, MODE_S16);

  // dont need D as we will assume that the sum is 16 bit - else we will use the
  // reference.
  VLDR(vpu, &A->vR);
  VLADD(vpu, cur_thresholds);
  VDEPTH1(vpu);

  // This can only process 16 channels at a time
  int output_bytes = VPU_INT16_EPV / CHAR_BIT;

  alignas(4) int32_t temp_mem;
  VSTRPV(vpu, &temp_mem, (1 << output_bytes) - 1);
  memcpy(Y, &temp_mem, output_bytes);

  Y += output_bytes;

  return Y;
}

extern "C" int8_t *output_transform_fn_binary_impl_asm(
    int8_t *Y, VPURingBuffer *A, int32_t output_channel_group,
    int16_t *thresholds);

int8_t *OTBinary::output_transform_fn(int8_t *Y, VPURingBuffer *A,
                                       int32_t output_channel_group) {
#ifdef NN_USE_REF
  return output_transform_fn_binary_impl(Y, A, output_channel_group,
                                         thresholds);
#else
  return output_transform_fn_binary_impl_asm(Y, A, output_channel_group,
                                             thresholds);
#endif  // NN_USE_REF
}
