// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "nn_bin_types.h"
#include "xs3_vpu.h"

#define NN_BCONV2D_KERNEL_OVERRUN_WORDS 8

/**
 * Struct represents the parameters needed by each
 * `bconv2d_bin_DI_impl()` job.
 *
 * Values are set by `bconv2d_bin_DI_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  // These are in a specific order - do not change

  int32_t outer_x_h_step;
  int32_t output_channel_loop_counter;
  const int32_t* threshold_p;
  int32_t inner_x_v_step;
  int32_t k_v_step;
  int32_t inner_x_h_step;
  int32_t k_h_step;
  int32_t outer_x_v_step;

  int32_t y_v_step;
  int32_t k_height_loop_counter;
  int32_t k_width_loop_counter;
  int32_t x_height_loop_counter;
  int32_t x_width_loop_counter;
  int32_t input_channel_loop_counter;
  bnn_b32_t* Y;
  const bnn_b256_t* X;

  const bnn_b256_t* K;
  int32_t y_c_step;
} nn_bconv2d_bin_DI_impl_plan_t;

/**
 * Struct represents the parameters needed by each
 * `bconv2d_bin_DI_patch_asm()` job.
 *
 * Values are set by `bconv2d_bin_DI_patch_asm_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  // These are in a pacific order - do not change

  int32_t k_height_loop_counter;
  bnn_b32_t* data_scratch;
  int32_t k_width_loop_counter;
  int32_t inner_x_v_step;
  int32_t inner_x_h_step;
  int32_t data_scratch_adjust;
  int32_t output_channel_loop_counter;
  const int32_t* threshold_p;

  const bnn_b32_t* X;
  int32_t outer_x_h_step;
  int32_t outer_x_v_step;
  int32_t y_v_step;
  int32_t patch_loop_counter;
  int32_t x_width_loop_counter;
  const bnn_b32_t* K;
  int32_t x_height_loop_counter;

  int32_t input_channel_loop_counter;
  int32_t
      k_p_adjust;  // the amount to advance the kernel pointer after applying it
  bnn_b32_t* Y;
  int32_t outer_y_c_step;

} nn_bconv2d_bin_impl_plan_t;

/**
 * Struct represents the parameters needed by each
 * `bconv2d_int8_DIDO_impl()` job.
 *
 * Values are set by `bconv2d_int8_DIDO_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  // These are in a specific order - do not change
  const bnn_b256_t* X;
  int32_t outer_x_h_step;
  int32_t output_channel_loop_counter;
  const bnn_b256_t* K;
  int32_t inner_x_v_step;
  int32_t k_v_step;
  int32_t inner_x_h_step;
  int32_t k_h_step;

  int32_t outer_x_v_step;
  int32_t y_v_step;
  int32_t k_height_loop_counter;
  int32_t k_width_loop_counter;
  int32_t x_height_loop_counter;
  int32_t x_width_loop_counter;
  int16_t*
      cur_post_activation_mul;  // These are needed to hold variables that will
  int16_t* cur_post_activation_bias;  // be indexed with ldd

  const int16_t* vlsat;
  int32_t ashr;
  const int16_t* final_shr;
  const int16_t* bias_multiplier;
  const int16_t* post_activation_mul;
  const int16_t* post_activation_bias;
  int32_t input_channel_loop_counter;
  int8_t* Y;

  const int16_t* clamp_near;
  const int16_t* clamp_far_0;
  const int16_t* clamp_far_1;
  int32_t y_c_step;

} nn_bconv2d_int8_DIDO_impl_plan_t;

/**
 * Struct represents the parameters needed by each
 * `bconv2d_int8_DIDO_impl()` job.
 *
 * Values are set by `bconv2d_int8_DIDO_impl_prepare()`.
 *
 * @note This struct is intended to be opaque.
 */
typedef struct {
  // These are in a specific order - do not change

  int32_t inner_x_h_step;
  int32_t data_scratch_adjust;
  int32_t k_height_loop_counter;
  bnn_b32_t* data_scratch;
  int32_t k_width_loop_counter;
  int32_t inner_x_v_step;
  int32_t outer_x_v_step;
  int32_t y_v_step;

  int32_t output_channel_loop_counter;
  const bnn_b32_t* K;
  int16_t*
      cur_post_activation_mul;  // These are needed to hold variables that will
  int16_t* cur_post_activation_bias;  // be indexed with ldd
  const int16_t* vlsat;
  int32_t ashr;
  const int16_t* post_activation_mul;
  const int16_t* post_activation_bias;

  int32_t input_channel_loop_counter;
  int8_t* Y;
  const bnn_b32_t* X;
  int32_t outer_x_h_step;
  int32_t k_p_adjust;
  int32_t patch_branch;
  int32_t final_channels_mask;
  int32_t final_channels_bytes;
  int32_t patch_loop_counter;

  const int16_t* final_shr;
  int32_t k_p_rewind;
  int32_t x_width_loop_counter;
  int32_t x_height_loop_counter;
  const int16_t* bias_multiplier;
  int16_t* quantised_accu_modifier;
  const int16_t* clamp_near;
  const int16_t* clamp_far_0;

  const int16_t* clamp_far_1;

} nn_bconv2d_int8_impl_plan_t;

typedef struct {
  int16_t clamp_near[VPU_INT16_EPV];
  int16_t clamp_far_0[VPU_INT16_EPV];
  int16_t clamp_far_1[VPU_INT16_EPV];
  int16_t bias_multipler[VPU_INT16_EPV];
  int16_t final_shr[VPU_INT16_EPV];
  int16_t accu_shr[VPU_INT16_EPV];  // for the vlsat
  int32_t accu_shl;                 // for the vlashr

} output_transform_values_t;
