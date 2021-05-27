// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <assert.h>
#include <limits.h>
#include <math.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "../nn_op_helper.h"
#include "nn_operator.h"

static inline int min(int a, int b) { return (a < b) ? a : b; }
static inline int max(int a, int b) { return (a > b) ? a : b; }

void bnn_populate_output_transform_values(output_transform_values_t* otv,
                                          const int16_t clamp_near,
                                          const int16_t clamp_far_0,
                                          const int16_t clamp_far_1,

                                          const int accu_shr,
                                          const int16_t bias_multiplier,
                                          const int16_t final_shr) {
  int16_t shr = max(0, accu_shr);
  int32_t shl = min(accu_shr, 0);

  // This is implemented with a vlsat, if it's less than zero then its going to
  // break
  assert(final_shr >= 0);

  otv->accu_shl = shl;  // for the vlashr
  for (unsigned i = 0; i < VPU_INT16_EPV; i++) {
    otv->clamp_near[i] = clamp_near;
    otv->clamp_far_0[i] = clamp_far_0;
    otv->clamp_far_1[i] = clamp_far_1;
    otv->accu_shr[i] = shr;  // for the vlsat
    otv->final_shr[i] = final_shr;
    otv->bias_multipler[i] = bias_multiplier;
  }
}

static int64_t vpu_saturate(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  const int64_t min_val = -max_val;

  return (input > max_val) ? max_val : (input < min_val) ? min_val : input;
}

static int64_t ashr(int64_t value, int shr) {
  if (shr > 0) {
    return (value + (1 << (shr - 1))) >> shr;
  } else {
    return (uint64_t)value << (-shr);
  }
}

int8_t bnn_post_activation_reference(
    const int32_t vpu_acc, const unsigned ch,
    const int16_t* post_activation_multiplier_q,
    const int16_t* post_activation_bias_q, const int accu_shr,
    const int16_t bias_multipler, const int final_shr) {
  int64_t scaled_accu = vpu_saturate(ashr(vpu_acc, accu_shr), 16);

  // TODO add clamps

  int64_t bias = vpu_saturate(
      (int64_t)post_activation_bias_q[ch] * (int64_t)bias_multipler, 32);
  int64_t product = vpu_saturate(
      (int64_t)post_activation_multiplier_q[ch] * (int64_t)scaled_accu + bias,
      32);
  int64_t product_shr = vpu_saturate(ashr(product, final_shr), 16);

  int64_t output = ashr(product_shr, 8);
  if (output < INT8_MIN) output = INT8_MIN;
  if (output > INT8_MAX) output = INT8_MAX;

  return (int8_t)output;
}

static int clrsb(int x) {
#if defined(__XS3A__)
  for (unsigned i = 0; i < 32; i++) {
    int y = (x << i) >> i;
    if (y != x) return (i - 1);
  }
  return 32;
#else
  return __builtin_clrsb(x);
#endif
}

static int clrsbll(long long x) {
#if defined(__XS3A__)
  for (unsigned i = 0; i < 64; i++) {
    int y = (x << i) >> i;
    if (y != x) return (i - 1);
  }
  return 64;
#else
  return __builtin_clrsbll(x);
#endif
}

// This puts upper and lower limits on the range of A
// A must reduce the vpu accumulator to 16 bit
// A must not remove all the imformation from the vpu accumulator
static void get_bounds_on_A(int* min_A, int* max_A, int32_t vpu_min_accu,
                            int32_t vpu_max_accu, int32_t vpu_clamp_min,
                            int32_t vpu_clamp_max) {
  int32_t max_out =
      max(max(max(vpu_min_accu, vpu_max_accu), vpu_clamp_min), vpu_clamp_max);
  int32_t min_out =
      min(min(min(vpu_min_accu, vpu_max_accu), vpu_clamp_min), vpu_clamp_max);
  int rsb = min(clrsb(max_out), clrsb(min_out));

  *max_A = rsb - 16;
  *min_A = *max_A - 16 + 1;
}

// This puts upper and lower limits on the range of Exp
// Exp will be applied to each of the values
// Exp must not saturate and of the values
// Exp must not leave all results as zero
static void get_bounds_on_Exp(int* min_Exp, int* max_Exp, float* values,
                              unsigned values_length, int bound_width) {
  assert(values_length > 0);
  int max_exponent = INT_MIN;
  for (unsigned i = 0; i < values_length; i++) {
    int e;
    frexp(values[i], &e);
    max_exponent = max(max_exponent, e);
  }

  *min_Exp = -max_exponent - 1;
  *max_Exp = *min_Exp + bound_width;
}

static void solve_constraint(int* B_res, int* A_res, int* M_res,

                             float* vpu_output_transform_multiplier,
                             float* vpu_output_transform_bias,
                             unsigned chans_out,

                             int32_t vpu_min_accu, int32_t vpu_max_accu,
                             int32_t vpu_clamp_min, int32_t vpu_clamp_max) {
  int min_A, max_A;
  int min_B, max_B;
  int min_M, max_M;

  get_bounds_on_A(&min_A, &max_A, vpu_min_accu, vpu_max_accu, vpu_clamp_min,
                  vpu_clamp_max);

  get_bounds_on_Exp(&min_M, &max_M, vpu_output_transform_multiplier, chans_out,
                    16);

  // This is 30 as we cannot make a 32 bit bias with a shr of 14
  get_bounds_on_Exp(&min_B, &max_B, vpu_output_transform_bias, chans_out,
                    16 + 14);

  // we also know that A + M = B;
  // Subtract one to ensure the addition is fine (one from A*M, B is already 30
  // bit at most)
  max_B = min(max_A + max_M - 1, max_B);

  // printf("min_B:%d max_B:%d\n", min_B, max_B);

  for (int A = max_A; A >= min_A; A--) {
    for (int M = max_M; M >= min_M; M--) {
      // We can squeeze a little more out of the arith by modelling
      // max_Product = max_A * max_M
      // this way we wouldnt need to subtract 2 from max_B

      int B = A + M;

      if ((B >= min_B) && (B <= max_B)) {
        *B_res = B;
        *A_res = A;
        *M_res = M;
        return;
      }
    }
  }
  assert(0);
}

void bnn_quantise_activation(
    int16_t* output_transform_multiplier_q, int16_t* output_transform_bias_q,

    float* output_transform_multiplier, float* output_transform_bias,

    unsigned chans_out,

    int32_t larq_clamp_min, int32_t larq_clamp_max,

    int16_t* quantised_accu_modifier, int16_t* clamp_near, int16_t* clamp_far_0,
    int16_t* clamp_far_1,

    int* accu_shr, int16_t* bias_multipler, int* final_shr,

    int32_t receptive_volume, int* chan_overlaps) {
  // This is the offset between the larq accumulator (xor_popcount) and the
  // xcore accumulator (macc//2). Note they have exactly the same units but a
  // constant offset.
  int vpu_offset = receptive_volume / 2;
  int vpu_multipler = -1;

  // XOR_POPCOUNT = VLMACCR1*vpu_multipler + vpu_offset
  // XOR_POPCOUNT = VLMACCR1*(-1) + receptive_volume / 2

  // output = clamp(clamp(V, (larq_clamp_min - (bv + ch_ov) * 2) / (mv * 2),
  // (larq_clamp_max - (bv + ch_ov) * 2) / (mv * 2)) * (m * mv * 2 ) + ((bv +
  // ch_ov) * 2 * m + b), INT8_MIN, INT8_MAX)

  // Low clamping value:  (larq_clamp_min - (bv + ch_ov) * 2) / (mv * 2)
  // High clamping value: (larq_clamp_max - (bv + ch_ov) * 2) / (mv * 2)
  // Multiplier: m * mv * 2
  // Bias: (bv + ch_ov) * 2 * m + b

  float* vpu_output_transform_multiplier =
      (float*)malloc(sizeof(float) * chans_out);
  float* vpu_output_transform_bias = (float*)malloc(sizeof(float) * chans_out);

  // Move to VPU scale
  for (unsigned ch = 0; ch < chans_out; ch++)
    vpu_output_transform_multiplier[ch] =
        output_transform_multiplier[ch] * vpu_multipler * 2;

  for (unsigned ch = 0; ch < chans_out; ch++) {
    vpu_output_transform_bias[ch] =
        output_transform_bias[ch] +
        2 * output_transform_multiplier[ch] * vpu_offset;
  }

  // This is the absolute value of the min and max clamp values in the VLMACCR1
  // space These values will need to be made 16 bit and converted to the output
  // space. clamp(V, (larq_clamp_min - bv * 2) / (mv * 2), (larq_clamp_max - bv
  // * 2) / (mv * 2))
  float vpu_clamp_min =
      (float)(larq_clamp_min - (vpu_offset)*2) / (vpu_multipler * 2);
  float vpu_clamp_max =
      (float)(larq_clamp_max - (vpu_offset)*2) / (vpu_multipler * 2);

  // TODO reorder min and max into min and max order

  int B, A, M;
  int vpu_min_accu = 0 * vpu_multipler + vpu_offset;
  int vpu_max_accu = receptive_volume * vpu_multipler + vpu_offset;

  solve_constraint(&B, &A, &M, vpu_output_transform_multiplier,
                   vpu_output_transform_bias, chans_out, vpu_min_accu,
                   vpu_max_accu, vpu_clamp_min, vpu_clamp_max);

  int min_16_bit_B, max_16_bit_B;

  get_bounds_on_Exp(&min_16_bit_B, &max_16_bit_B, vpu_output_transform_bias,
                    chans_out, 16);

  *bias_multipler = 1 << max(0, B - max_16_bit_B);
  int adjusted_B = min(B, max_16_bit_B);

  // The -8 is here to leave the result in a 16 bit form so that the
  // quantisation to 8 bit can deal with the asymertic rounding.
  *final_shr = B - 8;

  assert(*final_shr >= 0);

  // Now quantise the tensors
  for (unsigned ch = 0; ch < chans_out; ch++) {
    int32_t pa_mul =
        (int32_t)round(ldexp(vpu_output_transform_multiplier[ch], M));

    assert(clrsb(pa_mul) >= 16);  // make sure there is no overflow
    output_transform_multiplier_q[ch] = (int16_t)pa_mul;

    int32_t pa_bias =
        (int32_t)round(ldexp(vpu_output_transform_bias[ch], adjusted_B));

    // assert(clrsb(pa_bias) - 16 >= 0); // make sure there is no overflow
    pa_bias = min(INT16_MAX, pa_bias);  // TODO think about this

    output_transform_bias_q[ch] = (int16_t)pa_bias;
  }

  // todo check that post_activation_bias_q * adjusted_B is
  // ldexp(post_activation_bias, B)
  *accu_shr = -A;

  for (unsigned ch = 0; ch < chans_out; ch++) {
    if (chan_overlaps) {
      quantised_accu_modifier[ch] = ashr(chan_overlaps[ch], *accu_shr);
    } else {
      quantised_accu_modifier[ch] = 0;
    }
  }

  float min_shifted_accu = ldexp(vpu_clamp_min, -(*accu_shr));
  float max_shifted_accu = ldexp(vpu_clamp_max, -(*accu_shr));

  int32_t low_clamp_limit = -INT16_MAX * vpu_multipler;
  int32_t high_clamp_limit = INT16_MAX * vpu_multipler;

  int32_t t_low_clamp_offset =
      (int32_t)((float)low_clamp_limit - min_shifted_accu);  // round?
  int32_t t_high_clamp_offset =
      (int32_t)((float)high_clamp_limit - max_shifted_accu);

  int32_t t_clamp_near = t_low_clamp_offset,
          t_clamp_far_0 = t_high_clamp_offset;
  if (abs(t_clamp_near) >= abs(t_clamp_far_0)) {
    t_clamp_near = t_high_clamp_offset;
    t_clamp_far_0 = t_low_clamp_offset;
  }
  int32_t t_clamp_far_1 = t_clamp_far_0 / 2;
  t_clamp_far_0 -= t_clamp_far_1;

  *clamp_near = -t_clamp_near;
  *clamp_far_0 = -t_clamp_far_0;
  *clamp_far_1 = t_clamp_far_1;

  free(vpu_output_transform_multiplier);
  free(vpu_output_transform_bias);
}

void bnn_reorder_threshold_tensor(int32_t* thresh_boggled,
                                  const int32_t* thresholds_ref,
                                  const unsigned chans_out,
                                  const unsigned receptive_volume,
                                  int* chan_overlaps) {
  int16_t* thresholds = (int16_t*)thresh_boggled;

  for (unsigned i = 0; i < chans_out; i++) {
    unsigned bank = i / VPU_INT16_ACC_PERIOD;

    int32_t t = thresholds_ref[i] - (int32_t)(receptive_volume) / 2;

    if (chan_overlaps) t -= -chan_overlaps[i];

    unsigned idx = bank * 2 * VPU_INT16_ACC_PERIOD + i % VPU_INT16_ACC_PERIOD;
    thresholds[idx] = t;
    thresholds[idx + VPU_INT16_ACC_PERIOD] = (t >> 16);
  }
}

static unsigned xor_pop_32(bnn_b32_t a, bnn_b32_t b) {
  unsigned c = 0;
  bnn_b32_t v = a ^ b;
#if defined(__XS3A__)
  unsigned t = sizeof(bnn_b32_t);
  v = ~v;
  for (unsigned i = 0; i < t * 8; i++) {
    c += (v & 1);
    v >>= 1;
  }
#else
  c += __builtin_popcount(~v);
#endif
  return c;
}

void bnn_reorder_kernel_tensor(bnn_b32_t* K_p, const bnn_b32_t* K_ref_p,
                               const unsigned k_height, const unsigned k_width,
                               const unsigned chans_in,
                               const unsigned chans_out, int* chan_overlaps) {
  // This is the count of full vector words that can be applied to the data
  unsigned receptive_volume = chans_in * k_height * k_width;

  // The number of full XS3_VPU_VREG_WIDTH_BITS bit loads a single channel can
  // process
  unsigned complete_256_bit_groups = receptive_volume / XS3_VPU_VREG_WIDTH_BITS;

  // This is the number of words remaining after
  // complete_256_bit_groups*XS3_VPU_VREG_WIDTH_BITS bits have been processed
  unsigned remaining_input_words =
      (receptive_volume % XS3_VPU_VREG_WIDTH_BITS) / 32;

  const unsigned inputs_per_b32 = 32;
  assert(receptive_volume % inputs_per_b32 == 0);

  bnn_b32_t(*K_ref)[receptive_volume / inputs_per_b32] =
      (bnn_b32_t(*)[receptive_volume / inputs_per_b32]) K_ref_p;

  // the nuber of VPU_INT16_ACC_PERIOD groups there will be
  unsigned output_chan_groups_of_accu_period = chans_out / VPU_INT16_ACC_PERIOD;
  unsigned output_chans_reamining = chans_out % VPU_INT16_ACC_PERIOD;

  bnn_b32_t* p = (bnn_b32_t*)K_p;
  // This loops across groups of VPU_INT16_ACC_PERIOD output channels
  for (unsigned output_chan_group = 0;
       output_chan_group < output_chan_groups_of_accu_period;
       output_chan_group++) {
    // copy the groups of 256 input channels
    for (unsigned ic_group = 0; ic_group < complete_256_bit_groups;
         ic_group++) {
      // each group is of VPU_INT16_ACC_PERIOD channels
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD;
           sub_grp_idx++) {
        unsigned reversed_channel_order =
            VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;
        memcpy(p,
               &K_ref[output_chan_group * VPU_INT16_ACC_PERIOD +
                      reversed_channel_order][8 * ic_group],
               sizeof(bnn_b32_t) * 8);
        p += 8;
      }
    }

    if (remaining_input_words) {
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD;
           sub_grp_idx++) {
        unsigned reversed_channel_order =
            VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;
        memcpy(p,
               &K_ref[output_chan_group * VPU_INT16_ACC_PERIOD +
                      reversed_channel_order][8 * complete_256_bit_groups],
               sizeof(bnn_b32_t) * remaining_input_words);
        p += remaining_input_words;
      }
    }
  }

  // If there are remaining input channels deal with there here
  if (output_chans_reamining) {
    // copy the groups of 256 input channels
    for (unsigned ic_group = 0; ic_group < complete_256_bit_groups;
         ic_group++) {
      // each group is of VPU_INT16_ACC_PERIOD channels
      for (unsigned sub_grp_idx = 0; sub_grp_idx < output_chans_reamining;
           sub_grp_idx++) {
        unsigned reversed_channel_order =
            output_chans_reamining - 1 - sub_grp_idx;
        memcpy(p,
               &K_ref[output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD +
                      reversed_channel_order][8 * ic_group],
               sizeof(bnn_b32_t) * 8);
        p += 8;
      }
    }

    if (remaining_input_words) {
      for (unsigned sub_grp_idx = 0; sub_grp_idx < output_chans_reamining;
           sub_grp_idx++) {
        unsigned reversed_channel_order =
            output_chans_reamining - 1 - sub_grp_idx;
        memcpy(p,
               &K_ref[output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD +
                      reversed_channel_order][8 * complete_256_bit_groups],
               sizeof(bnn_b32_t) * remaining_input_words);
        p += remaining_input_words;
      }
    }
  }

  // This is for the case of no overlap in the kernels
  if (chan_overlaps == 0) return;

  memset(chan_overlaps, 0, sizeof(int) * chans_out);
  // Code only gets here if there is no overlap and hence no need to insert
  // padding.

  // The filler value could be anything it just needs to be a known value
  char filler = 0x55;
  memset(p, filler,
         sizeof(bnn_b32_t) *
             NN_BCONV2D_KERNEL_OVERRUN_WORDS);  // TODO minimise this

  // Reset the pointer for another pass to get the overlaps now that the memory
  // if laied out correctly
  p = (bnn_b32_t*)K_p;

  for (unsigned output_chan_group = 0;
       output_chan_group < output_chan_groups_of_accu_period;
       output_chan_group++) {
    p += (8 * VPU_INT16_ACC_PERIOD * complete_256_bit_groups);

    // printf("remaining_input_words %u\n", remaining_input_words);
    if (remaining_input_words) {
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD;
           sub_grp_idx++) {
        unsigned reversed_channel_order =
            VPU_INT16_ACC_PERIOD - 1 - sub_grp_idx;

        bnn_b32_t zeros = 0x00000000;
        int vlmaccr1_accu_overrun = 0;
        for (unsigned o = remaining_input_words; o < 8;
             o++) {  // 8 is 32 bit words per vpu load
          vlmaccr1_accu_overrun += (32 / 2) - (int)xor_pop_32(p[o], zeros);
        }
        chan_overlaps[output_chan_group * VPU_INT16_ACC_PERIOD +
                      reversed_channel_order] = vlmaccr1_accu_overrun;
        p += remaining_input_words;
      }

    } else {
      // This code is here for the case where the overlap is being used with
      // multiples 256 input channels.
      for (unsigned sub_grp_idx = 0; sub_grp_idx < VPU_INT16_ACC_PERIOD;
           sub_grp_idx++) {
        chan_overlaps[output_chan_group * VPU_INT16_ACC_PERIOD + sub_grp_idx] =
            0;
      }
    }
  }
  if (output_chans_reamining) {
    p += (8 * output_chans_reamining * complete_256_bit_groups);

    if (remaining_input_words) {
      for (unsigned sub_grp_idx = 0; sub_grp_idx < output_chans_reamining;
           sub_grp_idx++) {
        unsigned reversed_channel_order =
            output_chans_reamining - 1 - sub_grp_idx;

        bnn_b32_t zeros = 0x00000000;
        int vlmaccr1_accu_overrun = 0;
        // printf("ch %u\n",output_chan_groups_of_accu_period *
        // VPU_INT16_ACC_PERIOD + reversed_channel_order );
        for (unsigned o = remaining_input_words; o < 8;
             o++) {  // 8 is 32 bit words per vpu load
          // printf("%08x\n", p[o]);
          vlmaccr1_accu_overrun += (32 / 2) - (int)xor_pop_32(p[o], zeros);
        }
        // printf("\n");
        chan_overlaps[output_chan_groups_of_accu_period * VPU_INT16_ACC_PERIOD +
                      reversed_channel_order] = vlmaccr1_accu_overrun;
        p += remaining_input_words;
      }
    }
  }
}
