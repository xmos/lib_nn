// Copyright 2021-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "OutputTransformFn.hpp"

#include <algorithm>
#include <cassert>
#include <cstdint>
#include <limits>
#include <tuple>

extern "C" {
#include "vpu_sim.h"
#include "xs3_vpu.h"
}

using namespace nn;

/** @brief Round a floating-point value up to the next integer. */
static int64_t round_up(float x) { return std::ceil(x); }

/** @brief Round a floating-point value down to the previous integer. */
static int64_t round_down(float x) { return std::floor(x); }

/** @brief Saturate a signed value to a non-symmetric integer range. */
static int64_t saturate_non_sym(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  const int64_t min_val = -max_val - 1;
  return (input > max_val) ? max_val : (input < min_val) ? min_val : input;
}

/** @brief Return the smaller of two signed 32-bit integers. */
static inline int32_t min_int32(const int32_t lhs, const int32_t rhs) {
  return lhs < rhs ? lhs : rhs;
}

/** @brief Implement VDEPTH8 with asymmetric rounding. */
static void VDEPTH8_FIXED(vpu_t *vpu) {
  vpu_vector_t vec_tmp;
  memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
  memset(&(vpu->vR), 0, sizeof(vpu_vector_t));

  for (int i = 0; i < VPU_INT16_EPV; i++) {
    int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
    vpu->vR.s8[i] = saturate_non_sym(elm >> 8, 8);
  }
}

/** @brief Count the leading redundant sign bits in a signed 64-bit value. */
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

/** @brief Shift a signed 64-bit value with rounding for right shifts. */
static int64_t shl(int64_t v, int amount_to_shl) {
  if (amount_to_shl >= 0) {
    // work around  the undefined behaviour
    uint64_t mask = (~0LLU) >> amount_to_shl;
    return ((uint64_t)v & mask) << amount_to_shl;
  } else {
    int amount_to_shr = -amount_to_shl;
    return ((int64_t)v + (1LL << (amount_to_shr - 1))) >> amount_to_shr;
  }
}

/** @brief Return the number of bits required to represent a signed value. */
static int count_bits(int64_t v) {
#ifndef CHAR_BIT
#define CHAR_BIT 8
#endif
  return ((sizeof(int64_t) * CHAR_BIT) - clrsbll(v));
}

/** @brief Return whether a signed value fits in the requested bit width. */
static bool check_val_fits(int64_t v, int bit_count) {
  return count_bits(v) <= bit_count;
}

template <class T>
void recitfy_min_max(T &v_min, T &v_max) {
  T actual_max = std::max(v_min, v_max);
  v_min = std::min(v_min, v_max);
  v_max = actual_max;
}

/** @brief Convert a floating-point value to a rounded scaled integer. */
template <class T>
static int64_t float_to_int(T f, int e) {
  return (int64_t)std::rint(ldexp(f, e));
}

/** @brief Convert a floating-point value to a saturated 16-bit integer. */
template <class T>
static int16_t float_to_int16(T f, int e) {
  int64_t v = float_to_int(f, e);
  v = std::min((int64_t)INT16_MAX, v);
  v = std::max((int64_t)INT16_MIN, v);
  return (int16_t)v;
}

/** @brief Convert a scaled bias while compensating for double rounding. */
template <class T>
static int16_t float_to_int16_with_bias(T f, int e) {
  return float_to_int16(f - (1.0 / (1<<(e))), e);
}

// Select A, M such that
// ((accu * 2**A) * (mul * 2**M) + bias*2**B) gives the most precision
// accu_min and accu_max should be pairwise correct, i.e. min is the min, max is
// the max for each channel.

std::tuple<int, int>
OutputTransformFnInt8_Group::Quantizer::solve_for_constraints(
    MulsAndBias &activationParams, int vlmul_shr, bool verbose) {
  (void)verbose;
  int accu_bits_max = 0;
  int max_multiplier_exponent = INT32_MIN;

  // If all the accumulators or multipliers are zero then there is no defined
  // range.
  bool accu_range_defined = false;
  bool multiplier_range_defined = false;

  // for each set of mults + biases
  for (auto activationParam : activationParams) {
    if (activationParam.accu_max_val) {
      // get exponent required to represent max val
      int accu_max_bits =
          OutputTransformFn::get_max_exponent(activationParam.accu_max_val);
      accu_bits_max = std::max(accu_bits_max, accu_max_bits);
      accu_range_defined |= true;
    }
    if (activationParam.accu_min_val) {
      int accu_min_bits =
          OutputTransformFn::get_max_exponent(activationParam.accu_min_val);
      accu_bits_max = std::max(accu_bits_max, accu_min_bits);
      accu_range_defined |= true;
    }
    if (activationParam.multiplier) {
      int multiplier_bits =
          OutputTransformFn::get_max_exponent(activationParam.multiplier);
      max_multiplier_exponent =
          std::max(max_multiplier_exponent, multiplier_bits);
      multiplier_range_defined |= true;
    }
  }

  // If either of the ranges are undefined(i.e. all zero) then the result is the
  // same: the product contributes nothing
  bool product_range_defined = multiplier_range_defined && accu_range_defined;

  if (!product_range_defined) {
    // Then we only care about the biases -> we know they will always fit in an
    // 8 bit number so a bias exp of 0 will do fine.
    int A = 0;
    int M = vlmul_shr;
    int B = A + M - vlmul_shr;
    assert(B == 0);
    return std::make_tuple(A, M);
  }

  // bits left after required exponents
  int max_A = 15 - accu_bits_max;
  int max_M = 15 - max_multiplier_exponent;

  int A = max_A;
  int M = max_M;

  int mul_sig_bits = 0, accu_sig_bits = 0;

  for (auto activationParam : activationParams) {
    // multiplier raised to exponent M
    int64_t mul_16 = float_to_int(activationParam.multiplier, M);
    // Greatest sig bits
    mul_sig_bits = std::max(mul_sig_bits, count_bits(mul_16));

    // shift acc limits by A
    int64_t accu_max_16 = shl(activationParam.accu_max_val, A);
    int64_t accu_min_16 = shl(activationParam.accu_min_val, A);

    // greatest sig bits
    accu_sig_bits = std::max(accu_sig_bits, std::max(count_bits(accu_max_16),
                                                     count_bits(accu_min_16)));
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
        break;
      }

      int64_t mul_16 = float_to_int(activationParam.multiplier, M);
      if (!check_val_fits(mul_16, 16)) {
        M--;
        mul_sig_bits--;
        trying = true;
        break;
      }

      int64_t bias_16 =
          float_to_int(activationParam.bias, A + M - vlmul_shr);

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
        if (A >= 0 || accu_sig_bits > mul_sig_bits) {
          A--;
          accu_sig_bits--;
        } else {
          M--;
          mul_sig_bits--;
        }

        trying = true;
        break;
      }
    }
  }
  return std::make_tuple(A, M);
}

// Select an A and M for each set of Activation Parameters such that for
// a fixed B (that gives most precision for range of parameters),
// B = A + M - vlmul_shr, and A, M give most precision for each parameter set
std::tuple<std::vector<int>, std::vector<int>>
OutputTransformFnInt8_Channelwise::Quantizer::solve_for_constraints(
    MulsAndBias &activationParams, int vlmul_shr, bool verbose) {
  (void)verbose;
  std::vector<int> As, Ms;
  int global_B = 0;

  // Select largest valid B
  for (auto activationParam : activationParams) {
    if (activationParam.bias) {
      int bias_bits = OutputTransformFn::get_max_exponent(activationParam.bias);
      int64_t bias_16 = float_to_int(activationParam.bias, 15 - bias_bits);
      if (check_val_fits(bias_16, 16)) {
        global_B = std::max(global_B, bias_bits);
      }
    }
  }
  global_B = 15 - global_B;
  if(!(global_B > 0)) global_B = 1;
  assert(global_B > 0);

  // Select A and M
  for (auto activationParam : activationParams) {
    int M, A;

    int64_t bias_16 = float_to_int(activationParam.bias, global_B);

    int accu_bits_max = 0;
    int max_multiplier_exponent = INT32_MIN;

    bool accu_range_defined = false;
    bool multiplier_range_defined = false;

    if (activationParam.accu_max_val) {
      // get exponent required to represent max val
      accu_bits_max =
          OutputTransformFn::get_max_exponent(activationParam.accu_max_val);
      accu_range_defined |= true;
    }
    if (activationParam.accu_min_val) {
      // negative as initially 32 bit
      int accu_min_bits =
          OutputTransformFn::get_max_exponent(activationParam.accu_min_val);
      accu_bits_max = std::max(accu_bits_max, accu_min_bits);
      accu_range_defined |= true;
    }
    if (activationParam.multiplier) {
      max_multiplier_exponent =
          OutputTransformFn::get_max_exponent(activationParam.multiplier);
      multiplier_range_defined |= true;
    }
    // If either of the ranges are undefined(i.e. all zero) then the result is
    // the same: the product contributes nothing
    bool product_range_defined = multiplier_range_defined && accu_range_defined;

    if (!product_range_defined) {
      // As B is constant for all values, set A as maximum valid value and later
      // A is adjusted to fit constraints
      A = 0;
      M = global_B - A + vlmul_shr;
      As.push_back(A);
      Ms.push_back(M);
    } else {
      int max_A = 15 - accu_bits_max;
      int max_M = 15 - max_multiplier_exponent;

      // Maximum valid A and M for activation param set
      A = max_A;
      M = max_M;

      int mul_sig_bits = 0, accu_sig_bits = 0;

      // multiplier raised to exponent M
      int64_t mul_16 = float_to_int(activationParam.multiplier, M);
      // Greatest sig bits
      mul_sig_bits = std::max(mul_sig_bits, count_bits(mul_16));

      // shift acc limits by A
      int64_t accu_max_16 = shl(activationParam.accu_max_val, A);
      int64_t accu_min_16 = shl(activationParam.accu_min_val, A);
      // greatest sig bits
      accu_sig_bits =
          std::max(accu_sig_bits,
                   std::max(count_bits(accu_max_16), count_bits(accu_min_16)));

      int64_t max_group_prod, min_group_prod, max_group_sum, min_group_sum;
      bool trying = true;
      while (trying) {
        trying = false;

        max_group_prod = INT32_MIN;
        min_group_prod = INT32_MAX;
        max_group_sum = INT32_MIN;
        min_group_sum = INT32_MAX;

        // check
        // accu*2**A fit in 16 bits
        // mul*2**B fit in 16 bits
        // (must be representable by 16bit *
        // (1<<x)) (accu*2**A)*(mul*2**B) fit in 32 bits (accu*2**A)*(mul*2**B)
        // +

        int64_t accu_max_16 = shl(activationParam.accu_max_val, A);
        int64_t accu_min_16 = shl(activationParam.accu_min_val, A);
        if (!check_val_fits(accu_max_16, 16) ||
            !check_val_fits(accu_min_16, 16)) {
          A--;
          accu_sig_bits--;
          trying = true;

        }

        int64_t mul_16 = float_to_int(activationParam.multiplier, M);
        if (!check_val_fits(mul_16, 16)) {
          M--;
          mul_sig_bits--;
          trying = true;

        }

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
        if (!check_val_fits(prod_max, 16) || !check_val_fits(prod_min, 16)) {
          if (A >= 0 || accu_sig_bits > mul_sig_bits) {
            A--;
            accu_sig_bits--;
          } else {
            M--;
            mul_sig_bits--;
          }
          trying = true;
        }
      }
      As.push_back(A);
      Ms.push_back(M);
    }
  }

  assert(As.size() == activationParams.size());
  assert(Ms.size() == activationParams.size());

  // Check A is zero or negative
  int Amax = 0;
  for (unsigned ch = 0; ch < activationParams.size(); ch++) {
    bool trying = true;
    while (trying) {
      trying = false;
      if (As[ch] > 0) {
        Amax = std::max(Amax, As[ch] + 1);
        As[ch]--;
        trying = true;
      }
    }
  }

  // Reduce B to fit smallest A and M
  for (unsigned ch = 0; ch < activationParams.size(); ch++) {
    bool trying = true;
    while (trying) {
      trying = false;
      if (global_B > (As[ch] + Ms[ch] - vlmul_shr)) {
        global_B--;
        trying = true;
      }
    }
  }

  for (unsigned ch = 0; ch < activationParams.size(); ch++) {
    int mul_sig_bits = 0, accu_sig_bits = 0;

    // multiplier raised to exponent M
    int64_t mul_16 = float_to_int(activationParams[ch].multiplier, Ms[ch]);
    // Greatest sig bits
    mul_sig_bits = std::max(mul_sig_bits, count_bits(mul_16));

    // shift acc limits by A
    int64_t accu_max_16 = shl(activationParams[ch].accu_max_val, As[ch]);
    int64_t accu_min_16 = shl(activationParams[ch].accu_min_val, As[ch]);
    // greatest sig bits
    accu_sig_bits = std::max(accu_sig_bits, std::max(count_bits(accu_max_16),
                                                     count_bits(accu_min_16)));

    bool trying = true;
    while (trying) {
      trying = false;

      // Reduce largest A and M to match B
      if (global_B < (As[ch] + Ms[ch] - vlmul_shr)) {
        trying = true;
        if (accu_sig_bits > mul_sig_bits) {
          As[ch]--;
          accu_sig_bits--;
        } else {
          Ms[ch]--;
          mul_sig_bits--;
        }
      }
    }
    assert(As[ch] <= 0);
  }

  return std::make_tuple(As, Ms);
}

void nn::OutputTransformFn::ActivationParams::
    backprop_output_clamps_to_accu_limits(bool verbose, bool debug) {
  (void)verbose;
  (void)debug;
  // adjust accu_min and max to account for the saturation on the output
  if (multiplier == 0.0) {
    multiplier = 0.0;
    bias = 0.0;
    accu_min_val = 0;
    accu_max_val = 0;
    output_max_val = 0;
    output_min_val = 0;
    return;
  }

  double hi = ((double)original_output_max_val - original_bias) / original_multiplier;
  double lo = ((double)original_output_min_val - original_bias) / original_multiplier;

  recitfy_min_max(lo, hi);

  //This is the min/max interesting accumultor range due to the output clamp.
  int64_t accu_out_clamp_max = round_up(hi);
  int64_t accu_out_clamp_min = round_down(lo);

  recitfy_min_max(accu_min_val, accu_max_val);

  int64_t union_max = std::min(accu_out_clamp_max, (int64_t)accu_max_val);
  int64_t union_min = std::max(accu_out_clamp_min, (int64_t)accu_min_val);

  if( union_max <= union_min){
      int32_t singular_output_value =
          std::max(std::min((int32_t)output_max_val, (int32_t)original_bias),
                   (int32_t)output_min_val);

      multiplier = 0.0;
      bias = singular_output_value;
      accu_max_val = 0;
      accu_min_val = 0;
      output_max_val = singular_output_value;
      output_min_val = singular_output_value;
  } else {
    multiplier = original_multiplier;
    bias = original_bias;
    accu_max_val = union_max;
    accu_min_val = union_min;
    output_max_val =std::max(std::min((int32_t)output_max_val, (int32_t)std::round(union_max * original_multiplier + original_bias)),
                   (int32_t)output_min_val);
    output_min_val =std::max(std::min((int32_t)output_max_val, (int32_t)std::round(union_min * original_multiplier + original_bias)),
                   (int32_t)output_min_val);
  }

}

OutputTransformFnInt8_Group::QuantisationParams
OutputTransformFnInt8_Group::Quantizer::quantise_activation(
    MulsAndBias &activationParams, nn_vlmul_shr_t vlmul_shr , bool verbose) {
  (void)verbose;
  if (activationParams.size() == 0) {
    QuantisationParams q;
    q.initial_shr = 0;
    q.final_shr = 0;
    return q;
  }

  // Ensure the order is correct
  for (auto &activationParam : activationParams)
    recitfy_min_max(activationParam.accu_min_val, activationParam.accu_max_val);

  int A, M;

  std::tie(A, M) = solve_for_constraints(activationParams, vlmul_shr, false);
  int B = A + M - vlmul_shr;

  QuantisationParams q;

  q.initial_shr = -A;
  q.final_shr = B - 8;

  // Quantise the multiplier and bias
  for (unsigned ch = 0; ch < activationParams.size(); ++ch) {
    int16_t m = float_to_int16(activationParams[ch].multiplier, M);
    q.multipliers.push_back(m);
    int16_t b = float_to_int16_with_bias(activationParams[ch].bias, B);
    q.biases.push_back(b);

  }
  return q;
}

OutputTransformFnInt8_Channelwise::QuantisationParams
OutputTransformFnInt8_Channelwise::Quantizer::quantise_activation(
    MulsAndBias &activationParams, nn_vlmul_shr_t vlmul_shr, bool verbose) {
  (void)verbose;
  if (activationParams.size() == 0) {
    QuantisationParams q;
    q.initial_shr = 0;
    q.final_shr = 0;
    return q;
  }
  std::vector<int> As, Ms;
  std::tie(As, Ms) = solve_for_constraints(activationParams, vlmul_shr, false);
  int B = As[0] + Ms[0] - vlmul_shr;
  // Ensure the order is correct
  for (auto &activationParam : activationParams)
    recitfy_min_max(activationParam.accu_min_val, activationParam.accu_max_val);

  QuantisationParams q;
  q.final_shr = B - 8;

  // Quantise the multiplier and bias
  for (unsigned ch = 0; ch < activationParams.size(); ++ch) {
    int A = As[ch];
    int M = Ms[ch];

    assert(B == A + M - vlmul_shr);
    q.initial_shifts.push_back(-A);

    int16_t m = float_to_int16(activationParams[ch].multiplier, M);
    q.multipliers.push_back(m);
    int16_t b = float_to_int16(activationParams[ch].bias, B);
    q.biases.push_back(b);

  }
  q.initial_shr = q.initial_shifts[0];

  return q;
}

//----------------------- INT8 -----------------------
#ifdef NN_USE_REF
int8_t *output_transform_fn_ref(
  const otfn_int8_params_t *params,  int8_t *Y, VPURingBuffer *A, 
  int32_t output_channel_group, int16_t *multipliers_and_biases) 
{
  vpu_t vpu_mem;
  vpu_t *vpu = &vpu_mem;
  vpu_vector_t temp_mem;

  // Determine how many output channels this group contains.
  const int output_slice_channel_count = params->output_slice_channel_count;
  const int group_channel_offset = output_channel_group * VPU_INT16_EPV;
  const int remaining_channels = output_slice_channel_count - group_channel_offset;
  const int output_count = std::min(remaining_channels, (int32_t)VPU_INT16_EPV);
  const int mask = (1 << output_count) - 1;
  const int16_t in_shift =  params->initial_shift;
  const int16_t sat_shift = in_shift > 0 ? in_shift : 0;

  int16_t *cur_post_activation_mul = multipliers_and_biases + output_channel_group * VPU_INT16_EPV * 2;
  int16_t *cur_post_activation_bias = cur_post_activation_mul + output_count;

  // Set VPU mode
  VSETC(vpu, MODE_S16);

  // Load accumulator into D and R Registers
  VLDR(vpu, &A->vR);
  VLDD(vpu, &A->vD);

  // Saturate the accumulator to 16 bits before multiplication.
  for (int i = 0; i < VPU_INT16_EPV; ++i) {
    temp_mem.s16[i] = sat_shift;
  }
  VLSAT(vpu, &temp_mem);

  if (in_shift <= 0) {
    VSTR(vpu, &temp_mem);
    VLASHR(vpu, &temp_mem, in_shift);
  }
  VLMUL(vpu, cur_post_activation_mul);
  VLADD(vpu, cur_post_activation_bias);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->final_shr);
  VDEPTH8_FIXED(vpu);
  VSTRPV(vpu, Y, mask);
  Y += output_count;
  return Y;
}
#else
extern "C" int8_t *output_transform_fn_impl_asm(
  const otfn_int8_params_t *params, int8_t *Y, VPURingBuffer *A,
  int16_t *multipliers_and_biases, int output_count
);

int8_t *output_transform_fn_asm(
  const otfn_int8_params_t *params, int8_t *Y, VPURingBuffer *A,
  int32_t output_channel_group, int16_t *multipliers_and_biases) 
{
  const int32_t output_slice_channel_count = params->output_slice_channel_count;
  const int32_t group_channel_offset = output_channel_group * VPU_INT16_EPV;
  const int32_t remaining_channels = output_slice_channel_count - group_channel_offset;
  const int32_t output_count = min_int32(remaining_channels, (const int32_t)VPU_INT16_EPV);
  multipliers_and_biases += output_channel_group * VPU_INT16_EPV * 2;
  return output_transform_fn_impl_asm(params, Y, A, multipliers_and_biases, output_count);
}
#endif

int8_t *nn::otfn_int8(
  const otfn_int8_params_t *params, int8_t *Y, VPURingBuffer *A,
  int32_t output_channel_group, int16_t *multipliers_and_biases) 
{
#ifdef NN_USE_REF
  return output_transform_fn_ref(params, Y, A, output_channel_group, multipliers_and_biases);
#else
  return output_transform_fn_asm(params, Y, A, output_channel_group, multipliers_and_biases);
#endif  // NN_USE_REF
}


//----------------------- INT8 CHANNELWISE -----------------------
extern "C" int8_t *output_transform_fn_int_channelwise_impl_asm(
    const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
    int16_t *multipliers_and_biases, int output_count);

#ifndef NN_USE_REF
int8_t *output_transform_fn_int_channelwise_impl_asm_stub(
    const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *multipliers_and_biases) {
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);
  multipliers_and_biases += output_channel_group * VPU_INT16_EPV * 3;
  return output_transform_fn_int_channelwise_impl_asm(
      params, Y, A, multipliers_and_biases, output_count);
}
#endif

int8_t *output_transform_fn_int_channelwise_impl(
    const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *multipliers_and_biases) 
{
  const int32_t output_slice_channel_count = params->output_slice_channel_count;
  const int32_t output_channel_offset = output_channel_group * VPU_INT16_EPV;
  const int output_count = std::min(output_slice_channel_count - output_channel_offset,(int32_t)VPU_INT16_EPV);
  const unsigned mask = (1 << output_count) - 1;

  vpu_t vpu_mem;
  vpu_t *vpu = &vpu_mem;
  vpu_vector_t temp_mem;
  int16_t *cur_initial_shift = multipliers_and_biases + output_channel_group * VPU_INT16_EPV * 3;
  int16_t *cur_post_activation_mul = cur_initial_shift + output_count;
  int16_t *cur_post_activation_bias = cur_post_activation_mul + output_count;

  VSETC(vpu, MODE_S16);
  VLDR(vpu, &A->vR);
  VLDD(vpu, &A->vD);

  // Set temp_mem to hold initial shifts up to output count
  for (int i = 0; i < VPU_INT16_EPV; i++){
    temp_mem.s16[i] = cur_initial_shift[i];
  }
  for (int i = output_count; i < VPU_INT16_EPV; i++) {
    temp_mem.s16[i] = 0;
  }
  VLSAT(vpu, &temp_mem);
  VLMUL(vpu, cur_post_activation_mul);
  VLADD(vpu, cur_post_activation_bias);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->final_shr);
  VDEPTH8_FIXED(vpu);
  VSTRPV(vpu, Y, mask);
  Y += output_count;
  return Y;
}

int8_t *nn::otfn_int8_channelwise(const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
                                                 int32_t output_channel_group, int16_t *multipliers_and_biases) {
#if defined(NN_USE_REF)
  return output_transform_fn_int_channelwise_impl(
      params, Y, A, output_channel_group, multipliers_and_biases);
#else
  return output_transform_fn_int_channelwise_impl_asm_stub(
      params, Y, A, output_channel_group, multipliers_and_biases);
#endif  // NN_USE_REF
}


//----------------------- INT8 MAXPOOL -----------------------
extern "C" int8_t *output_transform_maxpool_impl_asm(
    const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
    int16_t *multipliers_and_biases, int output_count);

#ifndef NN_USE_REF
int8_t *output_transform_fn_int_maxpool_impl_asm_stub(
    const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *multipliers_and_biases) {
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);
  return output_transform_maxpool_impl_asm(
      params, Y, A, multipliers_and_biases, output_count);
}
#endif

int8_t *output_transform_fn_int_maxpool_impl(
    const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *multipliers_and_biases) {

  // we need to know how many we are processing
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);

  for(int i = 0; i < output_count; i++) {
      ((int8_t *)Y)[i] = ((int8_t *)&A->vR)[i];
  }
  (void)multipliers_and_biases;
  return Y + output_count;
}

int8_t *nn::otfn_int8_maxpool(const otfn_int8_channelwise_params_t *params, int8_t *Y, VPURingBuffer *A,
                                                 int32_t output_channel_group, int16_t *multipliers_and_biases) {
#ifdef NN_USE_REF
  return output_transform_fn_int_maxpool_impl(
      params, Y, A, output_channel_group, multipliers_and_biases);
#else
  return output_transform_fn_int_maxpool_impl_asm_stub(
      params, Y, A, output_channel_group, multipliers_and_biases);
#endif  // NN_USE_REF
}


//----------------------- INT8 CLAMPED -----------------------
#if defined(NN_USE_REF)
int8_t *output_transform_fn_int_clamped_ref(
    const otfn_int8_clamped_params_t *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *offsets_multipliers_and_biases) {
  vpu_t vpu_mem;
  vpu_t *vpu = &vpu_mem;
  vpu_vector_t temp_mem;
  const int32_t out_slice_count = params->output_slice_channel_count;
  const int32_t out_ch_count = output_channel_group * VPU_INT16_EPV;
  const int output_count = std::min(out_slice_count - out_ch_count, (int32_t)VPU_INT16_EPV);
  const unsigned  mask = (1 << output_count) - 1;

  // Prepare offset, multiplier, and bias pointers for the current output channel group
  int16_t *cur_post_activation_offset = offsets_multipliers_and_biases + out_ch_count * 3;
  int16_t *cur_post_activation_mul = cur_post_activation_offset + output_count;
  int16_t *cur_post_activation_bias = cur_post_activation_mul + output_count;

  VSETC(vpu, MODE_S16);
  VLDR(vpu, &A->vR);
  VLADD(vpu, cur_post_activation_offset);
  VPOS(vpu);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->initial_shift);
  VLMUL(vpu, cur_post_activation_mul);
  VLADD(vpu, cur_post_activation_bias);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->final_shr);
  VDEPTH8_FIXED(vpu);
  VSTRPV(vpu, Y, mask);
  Y += output_count;
  return Y;
}
#elif defined(__XS3A__)
extern "C" int8_t *output_transform_fn_int_clamped_asm(
    const otfn_int8_clamped_params_t *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *offsets_multipliers_and_biases);

#elif defined(__VX4A__) || defined(__VX4B__)
extern "C" int8_t *output_transform_fn_int_clamped_asm_vpu(
  int8_t *Y, VPURingBuffer *A, int16_t *offset, int16_t *multiplier,
  int16_t *bias, int output_count, int initial_shift, int final_shr
);

int8_t *output_transform_fn_int_clamped_asm(
    const otfn_int8_clamped_params_t *params, int8_t *Y, VPURingBuffer *A,
    int32_t output_channel_group, int16_t *offsets_multipliers_and_biases) 
{
  const int32_t out_slice_count = params->output_slice_channel_count;
  const int32_t out_ch_count = output_channel_group * VPU_INT16_EPV;
  const int32_t out_count = min_int32(out_slice_count - out_ch_count, VPU_INT16_EPV);
  int16_t *cur_offset = offsets_multipliers_and_biases + out_ch_count * 3;
  int16_t *cur_mul = cur_offset + out_count;
  int16_t *cur_bias = cur_mul + out_count;
  const int ish = params->initial_shift;
  const int fshr = params->final_shr;
  return output_transform_fn_int_clamped_asm_vpu(Y, A, cur_offset, cur_mul, cur_bias, out_count, ish, fshr);
}
#else
#error "Not supported"
#endif

int8_t *nn::otfn_int8_clamped(const otfn_int8_clamped_params_t *params, int8_t *Y, VPURingBuffer *A,
                                             int32_t output_channel_group, int16_t *offsets_multipliers_and_biases) {
#if defined(NN_USE_REF)
  return output_transform_fn_int_clamped_ref(params, Y, A, output_channel_group, offsets_multipliers_and_biases);
#else
  return output_transform_fn_int_clamped_asm(
      params, Y, A, output_channel_group, offsets_multipliers_and_biases);
#endif  // NN_USE_REF
}


//----------------------- BINARY -----------------------
#ifdef NN_USE_REF
int8_t *output_transform_fn_binary_ref(
  int8_t *Y, 
  VPURingBuffer *A,
  int32_t output_channel_group,
  threshold_t *thresholds) 
{
  // allocate vpu and set up the pointer
  vpu_t vpu;
  threshold_t *cur_thresholds = (thresholds + output_channel_group * VPU_INT16_EPV);

  // set vpu mode 16
  VSETC(&vpu, MODE_S16);

  // no need VD as int16 assumed
  VLDR(&vpu, &A->vR);
  VLADD(&vpu, cur_thresholds);
  VDEPTH1(&vpu);

  // This can only process 16 channels at a time
  unsigned output_bytes = VPU_INT16_EPV / CHAR_BIT;
  int32_t temp_mem;
  VSTRPV(&vpu, &temp_mem, (1 << output_bytes) - 1);
  memcpy(Y, &temp_mem, output_bytes);
  Y += output_bytes;
  return Y;
}

#else
extern "C" int8_t *output_transform_fn_binary_asm(
  int8_t *Y, 
  VPURingBuffer *A, 
  int32_t output_channel_group,
  int16_t *thresholds
);
#endif

int8_t *nn::otfn_binary(void *p, int8_t *Y, VPURingBuffer *A, int32_t output_channel_group, int16_t *thresholds) {
#ifdef NN_USE_REF
  return output_transform_fn_binary_ref(Y, A, output_channel_group, thresholds);
#else
  return output_transform_fn_binary_asm(Y, A, output_channel_group, thresholds);
#endif  // NN_USE_REF
(void)p;
}
