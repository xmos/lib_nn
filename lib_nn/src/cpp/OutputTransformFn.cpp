#include "OutputTransformFn.hpp"

#include <algorithm>
#include <cassert>
#include <cmath>
#include <cstdint>
#include <tuple>

#include "../src/asm/asm_constants.h"
#include "vpu_sim.h"
#include "xs3_vpu.h"

using namespace nn;

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

static int clrsb(int x) {
#if __has_builtin(__builtin_clrsb)
  return __builtin_clrsb(x);
#else
  for (unsigned i = 0; i < 32; i++) {
    int y = (x << i) >> i;
    if (y != x) return (i - 1);
  }
  return 32;
#endif
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

template <class activationT>
int get_max_exponent(std::vector<activationT> &arr) {
  activationT min_arr = std::numeric_limits<activationT>::max();
  activationT max_arr = std::numeric_limits<activationT>::min();

  for (activationT a : arr) {
    min_arr = std::min(min_arr, a);
    max_arr = std::max(max_arr, a);
  }

  int exp_of_min, exp_of_max;
  std::frexp(min_arr, &exp_of_min);
  std::frexp(max_arr, &exp_of_max);

  return std::max(exp_of_min, exp_of_max);
}

static int64_t shl(int32_t v, int amount_to_shl) {
  if (amount_to_shl >= 0) {
    return ((uint64_t)v) << amount_to_shl;
  } else {
    return ((int64_t)v) >> (-amount_to_shl);
  }
}
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
// Select A, M such that
// ((accu * 2**A) * (mul * 2**M) + bias*2**B) gives the most precision
template <class activationT>
std::tuple<int, int> solve_for_constraint(std::vector<activationT> &multiplier,
                                          std::vector<activationT> &bias,
                                          std::vector<int32_t> &accu_min,
                                          std::vector<int32_t> &accu_max) {
  int ch_count = accu_min.size();

  // These could(and should) be refined by inspecting accu_min, accu_max
  int max_A = std::min(15 - get_max_exponent(accu_min),
                       15 - get_max_exponent(accu_max));
  int max_multiplier_exponent = get_max_exponent(multiplier);
  int max_M = 15 - max_multiplier_exponent;

  int A = max_A;
  int M = max_M;

  int mul_sig_bits = 0, accu_sig_bits = 0;
  for (auto ch = 0; ch < ch_count; ++ch) {
    int64_t mul_16 = std::round(ldexp(multiplier[ch], M));
    mul_sig_bits = std::max(mul_sig_bits, count_bits(mul_16));

    int64_t accu_max_16 = shl(accu_max[ch], A);
    int64_t accu_min_16 = shl(accu_min[ch], A);
    accu_sig_bits = std::max(accu_sig_bits, std::max(count_bits(accu_max_16),
                                                     count_bits(accu_min_16)));
  }

  if (A > 0) {
    accu_sig_bits -= A;
    A = 0;
  }

  bool trying = true;

  while (trying) {
    trying = false;
    for (auto ch = 0; ch < ch_count; ++ch) {
      // check
      // accu*2**A fit in 16 bits
      // mul*2**B fit in 16 bits
      // bias*(2**(A+B)) fit in 31 bits (must be representable by 16bit *
      // (1<<x)) (accu*2**A)*(mul*2**B) fit in 32 bits (accu*2**A)*(mul*2**B) +
      // bias*(2**(A+B)) fit in 32 bits

      int64_t accu_max_16 = shl(accu_max[ch], A);
      int64_t accu_min_16 = shl(accu_min[ch], A);
      if (!check_val_fits(accu_max_16, 16) ||
          !check_val_fits(accu_min_16, 16)) {
        A--;
        accu_sig_bits--;
        trying = true;
        break;
      }

      int64_t mul_16 = std::round(ldexp(multiplier[ch], M));
      if (!check_val_fits(mul_16, 16)) {
        M--;
        mul_sig_bits--;
        trying = true;
        break;
      }

      // TODO remove the lowest 16 bits of this
      int64_t bias_32 = std::round(ldexp(bias[ch], A + M));

      int64_t prod_max = accu_max_16 * mul_16;
      int64_t prod_min = accu_min_16 * mul_16;
      int64_t sum_max = prod_max + bias_32;
      int64_t sum_min = prod_min + bias_32;

      if (!check_val_fits(bias_32, 30) || !check_val_fits(prod_max, 32) ||
          !check_val_fits(prod_min, 32) || !check_val_fits(sum_max, 32) ||
          !check_val_fits(sum_min, 32)) {
        if (A > 0 || accu_sig_bits > mul_sig_bits) {
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

    // want to have the same number of significant bits of accu and mul
    // must use as all 16 bits of the bias
  }
  return std::make_tuple(A, M);
}

template <class T>
void pad(std::vector<T> &vec, int pad_boundary, T pad_val) {
  vec.resize(
      vec.size() + (pad_boundary - vec.size() % pad_boundary) % pad_boundary,
      pad_val);
}

template <class T, std::size_t S>
static void fill_array(T (&arr)[S], T v) {
  std::fill_n(arr, sizeof arr / sizeof(T), v);
}

int32_t round_away_from_zero(float x) {
  if (x > 0)
    return std::ceil(x);
  else
    return std::floor(x);
}

QuantisationParams OutputTransformFnInt8::quantise_activation(
    std::vector<double> &output_transform_multiplier,
    std::vector<double> &output_transform_bias, std::vector<int32_t> &accu_min,
    std::vector<int32_t> &accu_max) {
  assert(output_transform_multiplier.size() == output_transform_bias.size());
  assert(accu_min.size() == accu_max.size());
  assert(accu_min.size() == output_transform_bias.size());

  int ch_count = accu_min.size();

  // Ensure the order is correct
  for (int ch = 0; ch < ch_count; ch++) {
    int32_t a = accu_min[ch], b = accu_max[ch];
    if (a > b) {
      accu_min[ch] = b;
      accu_max[ch] = a;
    }
  }

  std::vector<int32_t> accu_min_adjusted(ch_count, 0);
  std::vector<int32_t> accu_max_adjusted(ch_count, 0);

  // adjust accu_min and max to account for the saturation on the output
  for (int ch = 0; ch < ch_count; ch++) {
    float accu_actual_max = ((int32_t)INT8_MAX - output_transform_bias[ch]) /
                            output_transform_multiplier[ch];
    float accu_actual_min = ((int32_t)INT8_MIN - output_transform_bias[ch]) /
                            output_transform_multiplier[ch];

    accu_max_adjusted[ch] =
        std::min(accu_max[ch], std::max(round_away_from_zero(accu_actual_max),
                                        round_away_from_zero(accu_actual_min)));
    accu_min_adjusted[ch] =
        std::max(accu_min[ch], std::min(round_away_from_zero(accu_actual_max),
                                        round_away_from_zero(accu_actual_min)));
  }

  int A, M;
  std::tie(A, M) =
      solve_for_constraint(output_transform_multiplier, output_transform_bias,
                           accu_min_adjusted, accu_max_adjusted);

  int bias_max_bits = 0;
  int bias_min_bits = 64;
  int mul_max_bits = 0;
  int mul_min_bits = 64;
  for (int ch = 0; ch < ch_count; ++ch) {
    int64_t bias_q = std::round(ldexp(output_transform_bias[ch], A + M));
    bias_max_bits = std::max(bias_max_bits, count_bits(bias_q));
    bias_min_bits = std::min(bias_min_bits, count_bits(bias_q));

    int64_t mul_q = std::round(ldexp(output_transform_multiplier[ch], M));
    mul_max_bits = std::max(mul_max_bits, count_bits(mul_q));
    mul_min_bits = std::min(mul_min_bits, count_bits(mul_q));
  }

  assert(bias_max_bits <=
         30);  // 30 bits is the most we can achieve with this method

  int B = A + M;

  int e = std::max(0, bias_max_bits - 16);
  int16_t bias_multipler = (int16_t)(1 << e);
  int adjusted_B = B - e;

  QuantisationParams q;

  std::fill_n(q.otv.bias_multipler,
              sizeof q.otv.bias_multipler / sizeof bias_multipler,
              bias_multipler);

  // The -8 is here to leave the result in a 16 bit form so that the
  // quantisation to 8 bit can deal with the asymertic rounding.
  int16_t final_shr = B - 8;
  assert(final_shr >= 0);

  fill_array(q.otv.final_shr, final_shr);

  for (auto f : output_transform_multiplier) {
    int32_t pa_mul = (int32_t)round(ldexp(f, M));
    assert(clrsb(pa_mul) >= 16);  // make sure there is no overflow
    q.multipliers.push_back((int16_t)pa_mul);
  }

  for (auto f : output_transform_bias) {
    int32_t pa_bias = (int32_t)round(ldexp(f, adjusted_B));
    pa_bias = std::min((int32_t)INT16_MAX, pa_bias);
    q.biases.push_back((int16_t)pa_bias);
  }

  // pad q.biases and  q.multipliers to a multiple of VPU_INT16_EPV
  int16_t pad_val = 0;  // this is arbitrary
  pad(q.biases, VPU_INT16_EPV, pad_val);
  pad(q.multipliers, (int)VPU_INT16_EPV, pad_val);

  int16_t accu_shr = -A;
  if (accu_shr > 0) {
    // use a vlsat
    fill_array(q.otv.accu_shr, accu_shr);
    q.otv.accu_shl = 0;
  } else {
    // use a vashr
    fill_array(q.otv.accu_shr, (int16_t)0);
    q.otv.accu_shl = accu_shr;
  }

  return q;
}
extern "C" int8_t *output_transform_fn_impl_asm(const OT_int8::Params *params,
                                                int8_t *Y, VPURingBuffer *A,
                                                int32_t output_channel_group);

int8_t *output_transform_fn_impl(const OT_int8::Params *params, int8_t *Y,
                                 VPURingBuffer *A,
                                 int32_t output_channel_group) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  int16_t *cur_post_activation_bias =
      params->biases + output_channel_group * VPU_INT16_EPV;
  int16_t *cur_post_activation_mul =
      params->multipliers + output_channel_group * VPU_INT16_EPV;

  VSETC(vpu, MODE_S16);

  VLDR(vpu, &A->vR);
  VLDD(vpu, &A->vD);

  vpu_vector_t temp_mem;
  memset(&temp_mem, 0, sizeof(temp_mem));

  // Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->accu_shr);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->otv->accu_shl);

  // Save the 16 bit accumulator, A, to scratch
  VSTR(vpu, &temp_mem);

  // Clear the ring buffer
  VCLRDR(vpu);

  // Multiply the channel-wise bias by the bias multiplier to make it 32 bit per
  // channel
  VLDC(vpu, cur_post_activation_bias);
  VLMACC(vpu, params->otv->bias_multipler);

  // Multiply A by the post_activation_mul and accumulate it to the bias
  VLDC(vpu, &temp_mem);
  VLMACC(vpu, cur_post_activation_mul);

  // Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->final_shr);

  VDEPTH8_FIXED(vpu);

  // we need to know how many we are processing
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);

  int mask = (1 << output_count) - 1;

  VSTRPV(vpu, Y, mask);
  Y += output_count;

  return Y;
}

int8_t *OT_int8::output_transform_fn(int8_t *Y, VPURingBuffer *A,
                                     int32_t output_channel_group) {
#ifdef NN_USE_REF
  return output_transform_fn_impl(this->params, Y, A, output_channel_group);
#else
  return output_transform_fn_impl_asm(this->params, Y, A, output_channel_group);
#endif  // NN_USE_REF
}
OTBinary_int8::Params::Params(int32_t output_slice_channel_count,
                              OutputTransformValuesClamping *otv,
                              int16_t *biases, int16_t *multipliers,
                              int16_t *accu_modifier)
    : output_slice_channel_count(output_slice_channel_count),
      otv(otv),
      biases(biases),
      multipliers(multipliers),
      accu_modifier(accu_modifier) {}

int8_t *output_transform_fn_impl(const OTBinary_int8::Params *params, int8_t *Y,
                                 VPURingBuffer *A,
                                 int32_t output_channel_group) {
  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  int16_t *cur_post_activation_bias =
      params->biases + output_channel_group * VPU_INT16_EPV;
  int16_t *cur_accu_modifier =
      params->accu_modifier + output_channel_group * VPU_INT16_EPV;
  int16_t *cur_post_activation_mul =
      params->multipliers + output_channel_group * VPU_INT16_EPV;

  VSETC(vpu, MODE_S16);

  VLDR(vpu, &A->vR);
  VLDD(vpu, &A->vD);

  vpu_vector_t temp_mem;

  // Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->accu_shr);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, params->otv->accu_shl);

  // Subtract the channel overlap
  VLADD(vpu, cur_accu_modifier);

  VLSUB(vpu, params->otv->clamp_near);
  VLSUB(vpu, params->otv->clamp_near);
  VLSUB(vpu, params->otv->clamp_far_0);
  VLSUB(vpu, params->otv->clamp_far_1);
  VLSUB(vpu, params->otv->clamp_far_1);
  VLSUB(vpu, params->otv->clamp_far_0);

  // Save the 16 bit accumulator, A, to scratch
  VSTR(vpu, &temp_mem);

  // Clear the ring buffer
  VCLRDR(vpu);

  // Multiply the channel-wise bias by the bias multiplier to make it 32 bit per
  // channel
  VLDC(vpu, cur_post_activation_bias);
  VLMACC(vpu, params->otv->bias_multipler);

  // Multiply A by the post_activation_mul and accumulate it to the bias
  VLDC(vpu, &temp_mem);
  VLMACC(vpu, cur_post_activation_mul);

  // Reduce the accumulator to 16 bits
  VLSAT(vpu, params->otv->final_shr);

  VDEPTH8_FIXED(vpu);

  // we need to know how many we are processing
  int output_count = std::min(
      params->output_slice_channel_count - output_channel_group * VPU_INT16_EPV,
      (int32_t)VPU_INT16_EPV);

  int mask = (1 << output_count) - 1;

  VSTRPV(vpu, Y, mask);
  Y += output_count;

  return Y;
}

int8_t *OTBinary_int8::output_transform_fn(int8_t *Y, VPURingBuffer *A,
                                           int32_t output_channel_group) {
  return output_transform_fn_impl(this->params, Y, A, output_channel_group);
}

OTBinary_bin::OTBinary_bin(int16_t *thresholds) : thresholds(thresholds) {}

int8_t *OTBinary_bin::output_transform_fn(int8_t *Y, VPURingBuffer *A,
                                          int32_t output_channel_group) {
  // this is declared on the stack so that the asm can put the memory
  // in the constant pool
  int16_t zero_mem[32] = {0};

  int16_t *cur_thresholds = thresholds + output_channel_group * VPU_INT16_EPV;

  xs3_vpu vpu_mem;
  xs3_vpu *vpu = &vpu_mem;

  VSETC(vpu, MODE_S16);

  VLDR(vpu, &A->vR);
  VLDD(vpu, &A->vD);

  VLSAT(vpu, &zero_mem);
  VLSAT(vpu, cur_thresholds);
  VDEPTH1(vpu);

  // Do a 16 bit store here
  int16_t *Y16 = (int16_t *)Y;
  Y16[0] = vpu->vR.s16[0];
  Y16 += 1;

  return (int8_t *)Y16;
}

/******************************
 * DirectWriteOutputTransform
 *****************************/

////////// DirectWriteOutputTransform::Params //////////////
DirectWriteOutputTransform::Params::Params(const int image_channels)
    : output_img_channels(image_channels) {}

DirectWriteOutputTransform::Params::Params(
    const nn::ImageGeometry &output_image)
    : output_img_channels(output_image.depth) {}

DirectWriteOutputTransform::Params::Params(std::istream &stream) {
  stream.read(reinterpret_cast<char *>(&this->output_img_channels),
              sizeof(int32_t));
}

void DirectWriteOutputTransform::Params::Serialize(std::ostream &stream) const {
  stream.write(reinterpret_cast<const char *>(&this->output_img_channels),
               sizeof(this->output_img_channels));
}

////////// DirectWriteOutputTransform //////////////
DirectWriteOutputTransform::DirectWriteOutputTransform(const Params *params)
    : params(params) {}

int8_t *DirectWriteOutputTransform::output_transform_fn(
    int8_t *Y, VPURingBuffer *acc, int32_t output_channel_group) {
  const int32_t first_channel =
      DirectWriteOutputTransform::ChannelsPerOutputGroup * output_channel_group;
  const int32_t count =
      std::min<int32_t>(DirectWriteOutputTransform::ChannelsPerOutputGroup,
                        this->params->output_img_channels - first_channel);

  // This might be slightly sped up with an xcore-specific implementation, but
  // it's likely unnecessary
  std::memcpy(Y, &acc->vR[0], count);

  return &Y[count];
}

/******************************
 * ShiftInt8OutputTransform
 *****************************/

////////// ShiftInt8OutputTransform::Params //////////////
ShiftInt8OutputTransform::Params::Params(const int image_channels,
                                         const int16_t shift)
    : output_img_channels(image_channels) {
  for (int k = 0; k < VPU_INT8_ACC_PERIOD; ++k) shifts[k] = shift;
}

ShiftInt8OutputTransform::Params::Params(const nn::ImageGeometry &output_image,
                                         const int16_t shift)
    : output_img_channels(output_image.depth) {
  for (int k = 0; k < VPU_INT8_ACC_PERIOD; ++k) shifts[k] = shift;
}

ShiftInt8OutputTransform::Params::Params(std::istream &stream) {
  int16_t shift;
#define READ(X) stream.read(reinterpret_cast<char *>(&X), sizeof(X))
  READ(this->output_img_channels);
  READ(shift);
#undef READ

  for (int k = 0; k < VPU_INT8_ACC_PERIOD; ++k) shifts[k] = shift;
}

void ShiftInt8OutputTransform::Params::Serialize(std::ostream &stream) const {
  auto shift = this->shifts[0];
#define WRITE(X) stream.write(reinterpret_cast<const char *>(&X), sizeof(X));
  WRITE(this->output_img_channels);
  WRITE(shift);
#undef WRITE
}

////////// ShiftInt8OutputTransform //////////////
ShiftInt8OutputTransform::ShiftInt8OutputTransform(const Params *params)
    : params(params) {}

C_API void shift_int8_output_transform_ref(int8_t *output,
                                           const VPURingBuffer *acc,
                                           const int16_t *right_shifts,
                                           const int channel_count) {
  uint32_t write_mask = uint32_t((1LL << channel_count) - 1);

  auto vpu = nn::VPU();
  vpu_vector_t vec_tmp;
  uint32_t tmp;

  vpu.vldr(vpu_vect_0x80);
  vpu.vstrpv(output, write_mask);

  vpu.vldd(acc->vD);
  vpu.vldr(acc->vR);
  vpu.vsetc(MODE_S16);
  vpu.vlsat(right_shifts);
  vpu.vstr(&vec_tmp);
  vpu.vladd(vpu_vect_0x007F);
  vpu.vdepth1();
  vpu.vstrpv(&tmp, 0x0000000F);
  write_mask = write_mask & ~tmp;
  vpu.vlashr(&vec_tmp, -8);
  vpu.vdepth8();
  vpu.vstrpv(output, write_mask);
}

int8_t *ShiftInt8OutputTransform::output_transform_fn(
    int8_t *Y, VPURingBuffer *acc, int32_t output_channel_group) {
  const int32_t first_channel =
      ShiftInt8OutputTransform::ChannelsPerOutputGroup * output_channel_group;
  const int32_t count =
      std::min<int32_t>(ShiftInt8OutputTransform::ChannelsPerOutputGroup,
                        this->params->output_img_channels - first_channel);

#if defined(NN_USE_REF) || !defined(__XS3A__)
  shift_int8_output_transform_ref(Y, acc, this->params->shifts, count);
#else
  shift_int8_output_transform_xcore(Y, acc, this->params->shifts, count);
#endif  // NN_USE_REF

  return &Y[count];
}
