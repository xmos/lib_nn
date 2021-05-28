// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
// #include <stdint.h>
// #include <string.h>
// #include <assert.h>

#include "../nn_op_helper.h"
#include "nn_operator.h"
#include "vpu_sim.h"
#include "xs3_vpu.h"

void bconv2d_int8_DIDO_impl(nn_bconv2d_int8_DIDO_impl_plan_t* plan);

void bconv2d_int8_impl(nn_bconv2d_int8_impl_plan_t* plan);

static int64_t saturate_non_sym(const int64_t input, const unsigned bits) {
  const int64_t max_val = (((int64_t)1) << (bits - 1)) - 1;
  const int64_t min_val = -max_val - 1;

  return (input > max_val) ? max_val : (input < min_val) ? min_val : input;
}

// This is an implementation of VDEPTH8 where the rounding is asymetric
// The acutal asm implements the following but in a more convoluted way
// in order to work around the rounds issue.
static void VDEPTH8_FIXED(xs3_vpu* vpu) {
  vpu_vector_t vec_tmp;
  memcpy(&vec_tmp, &(vpu->vR), sizeof(vpu_vector_t));
  memset(&(vpu->vR), 0, sizeof(vpu_vector_t));

  for (int i = 0; i < VPU_INT16_EPV; i++) {
    int32_t elm = ((int32_t)vec_tmp.s16[i]) + (1 << 7);
    vpu->vR.s8[i] = saturate_non_sym(elm >> 8, 8);
  }
}

void bconv2d_int8_DIDO_impl_ref(nn_bconv2d_int8_DIDO_impl_plan_t* plan) {
  xs3_vpu vpu_data;
  xs3_vpu* vpu = &vpu_data;

  // Scratch mem
  vpu_vector_t temp_mem;

  VSETC(vpu, MODE_S16);

  void* X_p = (void*)plan->X;
  void* Y_p = (void*)plan->Y;

  for (int xh = plan->x_height_loop_counter; xh > 0; xh--) {
    for (int xv = plan->x_width_loop_counter; xv >= 0; xv--) {
      void* cur_post_activation_mul = (void*)plan->post_activation_mul;
      void* cur_post_activation_bias = (void*)plan->post_activation_bias;
      void* K_p = (void*)plan->K;
      for (int oc = plan->output_channel_loop_counter; oc >= 0; oc--) {
        void* X_cur_p = X_p;
        VCLRDR(vpu);

        for (int kh = plan->k_height_loop_counter; kh >= 0; kh--) {
          for (int kw = plan->k_width_loop_counter; kw >= 0; kw--) {
            for (int ic = plan->input_channel_loop_counter; ic >= 0; ic--) {
              VLDC(vpu, X_cur_p);
              X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

              for (unsigned l = 0; l < VPU_INT16_EPV; l++) {
                VLMACCR1(vpu, K_p);
                K_p += XS3_VPU_VREG_WIDTH_BYTES;
              }
            }
            X_cur_p += plan->inner_x_h_step;
            K_p += plan->k_h_step;
          }
          X_cur_p += plan->inner_x_v_step;
          K_p += plan->k_v_step;
        }

        // Reduce the accumulator to 16 bits

        VLSAT(vpu, plan->vlsat);
        VSTR(vpu, &temp_mem);
        VLASHR(vpu, &temp_mem, plan->ashr);

        // vpu_sim_mem_print(plan->clamp_near, vpu->mode);
        // vpu_sim_mem_print(plan->clamp_far_0, vpu->mode);
        // vpu_sim_mem_print(plan->clamp_far_1, vpu->mode);
        // vpu_sim_print(vpu);

        // Saturate to larq high and low
        VLSUB(vpu, plan->clamp_near);
        VLSUB(vpu, plan->clamp_near);

        // vpu_sim_print(vpu);

        VLSUB(vpu, plan->clamp_far_0);
        VLSUB(vpu, plan->clamp_far_1);
        VLSUB(vpu, plan->clamp_far_1);
        VLSUB(vpu, plan->clamp_far_0);

        // vpu_sim_print(vpu);

        // exit(1);
        // Save the 16 bit accumulator, A, to scratch
        VSTR(vpu, &temp_mem);

        // Clear the ring buffer
        VCLRDR(vpu);

        // Multiply the channel-wise bias by the bias multiplier to make it 32
        // bit per channel
        VLDC(vpu, cur_post_activation_bias);
        VLMACC(vpu, plan->bias_multiplier);

        // Multiply A by the post_activation_mul and accumulate it to the bias
        VLDC(vpu, &temp_mem);
        VLMACC(vpu, cur_post_activation_mul);

        // Reduce the accumulator to 16 bits
        VLSAT(vpu, plan->final_shr);

        VDEPTH8_FIXED(vpu);
        VSTRPV(vpu, Y_p, VPU_INT16_ACC_VR_MASK);
        Y_p += VPU_INT16_EPV;

        cur_post_activation_mul += XS3_VPU_VREG_WIDTH_BYTES;
        cur_post_activation_bias += XS3_VPU_VREG_WIDTH_BYTES;
      }
      X_p += plan->outer_x_h_step;
      Y_p += plan->y_c_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }
}

static void make_patch(xs3_vpu* vpu, nn_bconv2d_int8_impl_plan_t* plan,
                       void* X_p) {
  void* X_cur_p = X_p;
  void* D_p = plan->data_scratch;

  for (int kh = plan->k_height_loop_counter; kh >= 0; kh--) {
    for (int kw = plan->k_width_loop_counter; kw >= 0; kw--) {
      for (int ic = plan->input_channel_loop_counter; ic >= 0; ic--) {
        VLDD(vpu, X_cur_p);
        X_cur_p += XS3_VPU_VREG_WIDTH_BYTES;

        VSTD(vpu, D_p);
        D_p += XS3_VPU_VREG_WIDTH_BYTES;
      }
      X_cur_p += plan->inner_x_h_step;
      D_p += plan->data_scratch_adjust;
    }
    X_cur_p += plan->inner_x_v_step;
  }
  VCLRDR(vpu);
  VSTD(vpu, D_p);
}

void compute_patch(
    nn_bconv2d_int8_impl_plan_t* plan, void** K_p, int step, xs3_vpu* vpu,
    const int16_t* sat_mem, const int16_t* bias_shift, const int16_t* final_shr,
    const int16_t* clamp_near_mem, const int16_t* clamp_far_0_mem,
    const int16_t* clamp_far_1_mem, void* cur_post_activation_mul,
    void* cur_post_activation_bias, void* cur_quantised_accu_modifier) {
  VCLRDR(vpu);
  void* D_p = plan->data_scratch;
  for (unsigned p = plan->patch_loop_counter; p > 0; p--) {
    VLDC(vpu, D_p);
    D_p += XS3_VPU_VREG_WIDTH_BYTES;
    for (unsigned l = 0; l < VPU_INT16_EPV - 1; l++) {
      VLMACCR1(vpu, *K_p);
      *K_p += XS3_VPU_VREG_WIDTH_BYTES;
    }
    VLMACCR1(vpu, *K_p);
    *K_p += step;
  }

  VLDC(vpu, D_p);

  unsigned tail_loops = VPU_INT16_EPV - 1 + step / XS3_VPU_VREG_WIDTH_BYTES;
  for (unsigned l = 0; l < tail_loops; l++) {
    VLMACCR1(vpu, *K_p);
    *K_p += plan->k_p_adjust;
  }

  vpu_vector_t temp_mem;
  memset(&temp_mem, 0, sizeof(temp_mem));

  // Reduce the accumulator to 16 bits
  VLSAT(vpu, sat_mem);
  VSTR(vpu, &temp_mem);
  VLASHR(vpu, &temp_mem, plan->ashr);

  // Subtract the channel overlap
  VLADD(vpu, cur_quantised_accu_modifier);

  // Saturate to larq high and low

  VLSUB(vpu, clamp_near_mem);
  VLSUB(vpu, clamp_near_mem);
  VLSUB(vpu, clamp_far_0_mem);
  VLSUB(vpu, clamp_far_1_mem);
  VLSUB(vpu, clamp_far_1_mem);
  VLSUB(vpu, clamp_far_0_mem);

  // Save the 16 bit accumulator, A, to scratch
  VSTR(vpu, &temp_mem);

  // Clear the ring buffer
  VCLRDR(vpu);

  // Multiply the channel-wise bias by the bias multiplier to make it 32 bit per
  // channel
  VLDC(vpu, cur_post_activation_bias);
  VLMACC(vpu, bias_shift);

  // Multiply A by the post_activation_mul and accumulate it to the bias
  VLDC(vpu, &temp_mem);
  VLMACC(vpu, cur_post_activation_mul);

  // Reduce the accumulator to 16 bits
  VLSAT(vpu, final_shr);

  VDEPTH8_FIXED(vpu);
}

void bconv2d_int8_impl_ref(nn_bconv2d_int8_impl_plan_t* plan) {
  xs3_vpu vpu_data;
  memset(&vpu_data, 0, sizeof(vpu_data));
  xs3_vpu* vpu = &vpu_data;

  VSETC(vpu, MODE_S16);

  void* X_p = (void*)plan->X;
  void* Y_p = (void*)plan->Y;

  for (int xh = plan->x_height_loop_counter; xh > 0; xh--) {
    for (int xv = plan->x_width_loop_counter; xv >= 0; xv--) {
      make_patch(vpu, plan, X_p);

      void* cur_post_activation_mul = (void*)plan->post_activation_mul;
      void* cur_post_activation_bias = (void*)plan->post_activation_bias;
      void* cur_quantised_accu_modifier = (void*)plan->quantised_accu_modifier;
      void* K_p = (void*)plan->K;
      for (int oc = plan->output_channel_loop_counter; oc > 0; oc--) {
        compute_patch(plan, &K_p, XS3_VPU_VREG_WIDTH_BYTES, vpu, plan->vlsat,
                      plan->bias_multiplier, plan->final_shr, plan->clamp_near,
                      plan->clamp_far_0, plan->clamp_far_1,
                      cur_post_activation_mul, cur_post_activation_bias,
                      cur_quantised_accu_modifier);

        VSTRPV(vpu, Y_p, VPU_INT16_ACC_VR_MASK);
        Y_p += VPU_INT16_EPV;

        cur_post_activation_mul += XS3_VPU_VREG_WIDTH_BYTES;
        cur_post_activation_bias += XS3_VPU_VREG_WIDTH_BYTES;
        cur_quantised_accu_modifier += XS3_VPU_VREG_WIDTH_BYTES;
      }

      compute_patch(plan, &K_p, plan->k_p_rewind, vpu, plan->vlsat,
                    plan->bias_multiplier, plan->final_shr, plan->clamp_near,
                    plan->clamp_far_0, plan->clamp_far_1,
                    cur_post_activation_mul, cur_post_activation_bias,
                    cur_quantised_accu_modifier);

      VSTRPV(vpu, Y_p, plan->final_channels_mask);

      Y_p += plan->final_channels_bytes;  // to this we add the amount to skip
                                          // to the next pixel(i.e. skip
                                          // channels we are not writing to)
      X_p += plan->outer_x_h_step;
    }
    X_p += plan->outer_x_v_step;
    Y_p += plan->y_v_step;
  }
}

void compute_int8_patch_loop_params(int32_t* k_p_adjust,
                                    int32_t* patch_loop_counter,
                                    int32_t x_channels, int32_t k_height,
                                    int32_t k_width) {
  int32_t bytes_per_input_channel = x_channels / 8;

  int32_t total_bytes_copied_to_scratch =
      bytes_per_input_channel * k_height * k_width;

  *k_p_adjust = total_bytes_copied_to_scratch % XS3_VPU_VREG_WIDTH_BYTES;
  if (*k_p_adjust == 0) *k_p_adjust = XS3_VPU_VREG_WIDTH_BYTES;

  *patch_loop_counter =
      (total_bytes_copied_to_scratch - *k_p_adjust) / XS3_VPU_VREG_WIDTH_BYTES;
}

int32_t compute_int8_over_RW_bytes(int32_t x_channels, int32_t k_height,
                                   int32_t k_width, int32_t chans_out) {
  int32_t k_p_adjust, patch_loop_counter;

  compute_int8_patch_loop_params(&k_p_adjust, &patch_loop_counter, x_channels,
                                 k_height, k_width);

  int32_t tail_chans = chans_out % VPU_INT16_EPV;
  if (tail_chans == 0) tail_chans = VPU_INT16_EPV;

  int32_t over_bytes = (patch_loop_counter > 0) * (VPU_INT16_EPV - tail_chans) *
                           XS3_VPU_VREG_WIDTH_BYTES -
                       (k_p_adjust * tail_chans);

  // compute_patch() always ends in one extra vpu write
  if (over_bytes < XS3_VPU_VREG_WIDTH_BYTES)
    over_bytes = XS3_VPU_VREG_WIDTH_BYTES;

  return over_bytes;
}

static void bconv2d_int8_prepare(
    nn_bconv2d_int8_impl_plan_t* plan, int8_t* Y_p, const bnn_b32_t* X_p,
    const bnn_b32_t* K_p, bnn_b32_t* data_scratch,

    const int16_t* post_activation_multiplier_q,
    const int16_t* post_activation_bias_q,

    const int16_t* quantised_accu_modifier,

    const output_transform_values_t* otv,

    const nn_image_params_t* x, const nn_image_params_t* y,
    const nn_window_params_t* k, const unsigned y_loc_width,
    const unsigned y_loc_height, const unsigned y_sub_width,
    const unsigned y_sub_height, const unsigned x_loc_width,
    const unsigned x_loc_height, const unsigned y_loc_channel,
    const unsigned y_sub_channel) {
  const unsigned bits_per_b32 = 32;
  const unsigned chan_b32_in = (x->channels + bits_per_b32 - 1) / bits_per_b32;
  const unsigned chans_out = y->channels;

  int8_t(*Y)[y->width][chans_out] = (int8_t(*)[y->width][chans_out])Y_p;

  bnn_b32_t(*X)[x->width][chan_b32_in] =
      (bnn_b32_t(*)[x->width][chan_b32_in])X_p;

  plan->X = (bnn_b32_t*)X[x_loc_height][x_loc_width];
  plan->data_scratch = data_scratch;

  // Relocate the pointers to the start of the region we care about.
  plan->Y = (int8_t*)&(Y[y_loc_height][y_loc_width][y_loc_channel]);
  plan->K = &(K_p[y_loc_channel * k->shape.height * k->shape.width *
                  chan_b32_in]);  // dereference by y_loc_channel

  plan->post_activation_mul = &(post_activation_multiplier_q[y_loc_channel]);
  plan->post_activation_bias = &(post_activation_bias_q[y_loc_channel]);
  plan->quantised_accu_modifier =
      (int16_t*)&(quantised_accu_modifier[y_loc_channel]);

  plan->clamp_near = (const int16_t*)otv->clamp_near;
  plan->clamp_far_0 = (const int16_t*)otv->clamp_far_0;
  plan->clamp_far_1 = (const int16_t*)otv->clamp_far_1;
  plan->bias_multiplier = (const int16_t*)otv->bias_multipler;
  plan->final_shr = (const int16_t*)otv->final_shr;
  plan->vlsat = (const int16_t*)otv->accu_shr;
  plan->ashr = otv->accu_shl;

  int32_t bytes_per_input_channel = x->channels / 8;

  const int32_t out_chans_multiplier = 4;

  assert(x->channels > 0);
  assert(y->channels > 0);
  assert((x->channels % bits_per_b32) == 0);
  assert((y->channels % out_chans_multiplier) == 0);
  assert((y_sub_channel % out_chans_multiplier) == 0);
  assert((y_loc_channel % out_chans_multiplier) == 0);

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k->shape.width - 1;

  assert(k->dilation.horizontal >= 1);
  assert(k->dilation.vertical >= 1);

  int32_t h_dilation = k->dilation.horizontal;
  int32_t h_stride = k->stride.horizontal;
  int32_t v_stride = k->stride.vertical;

  plan->input_channel_loop_counter =
      ((x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS) -
      1;

  int32_t x_height_loops = y_sub_height;
  int32_t x_width_loops = y_sub_width;

  plan->x_height_loop_counter = x_height_loops;
  plan->x_width_loop_counter = x_width_loops - 1;

  int32_t channels_to_process_on_tail_output_loop =
      (y_sub_channel - 1) % VPU_INT16_EPV + 1;

  plan->output_channel_loop_counter =
      (y_sub_channel - channels_to_process_on_tail_output_loop) / VPU_INT16_EPV;

  plan->k_p_rewind =
      (channels_to_process_on_tail_output_loop - VPU_INT16_EPV + 1L) *
      XS3_VPU_VREG_WIDTH_BYTES;

  compute_int8_patch_loop_params(&(plan->k_p_adjust),
                                 &(plan->patch_loop_counter), x->channels,
                                 k->shape.height, k->shape.width);

  int32_t f_mod = (y->channels - y_sub_channel);

  plan->final_channels_bytes = channels_to_process_on_tail_output_loop + f_mod;

  plan->final_channels_mask =
      ((1 << channels_to_process_on_tail_output_loop) - 1);

  if (bytes_per_input_channel % XS3_VPU_VREG_WIDTH_BYTES)
    plan->data_scratch_adjust =
        (bytes_per_input_channel % XS3_VPU_VREG_WIDTH_BYTES) -
        XS3_VPU_VREG_WIDTH_BYTES;
  else
    plan->data_scratch_adjust = 0;

  plan->inner_x_h_step =
      bytes_per_input_channel * (h_dilation - 1) -
      (XS3_VPU_VREG_WIDTH_BYTES * (plan->input_channel_loop_counter + 1) -
       bytes_per_input_channel);

  // TODO multiply x->width by dilation
  plan->inner_x_v_step =
      (bytes_per_input_channel * ((x->width - k->shape.width)));

  // Outer Loop
  plan->outer_x_h_step = bytes_per_input_channel * h_stride;

  plan->outer_x_v_step =
      (bytes_per_input_channel * (int32_t)x->width * v_stride) -
      (plan->outer_x_h_step * x_width_loops);

  plan->y_v_step = chans_out * sizeof(int8_t) * (y->width - y_sub_width);
}

static void bconv2d_int8_DIDO_prepare(
    nn_bconv2d_int8_DIDO_impl_plan_t* plan, int8_t* Y_p, const bnn_b256_t* X_p,
    const bnn_b256_t* K_p,

    const int16_t* post_activation_multiplier_q,
    const int16_t* post_activation_bias_q,

    const output_transform_values_t* otv,

    const nn_image_params_t* x, const nn_image_params_t* y,
    const nn_window_params_t* k,

    const unsigned y_loc_width, const unsigned y_loc_height,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned x_loc_width, const unsigned x_loc_height,
    const unsigned y_loc_channel, const unsigned y_sub_channel) {
  const unsigned chan_b256_in =
      (x->channels + XS3_VPU_VREG_WIDTH_BITS - 1) / XS3_VPU_VREG_WIDTH_BITS;
  const unsigned chans_out = y->channels;

  int8_t(*Y)[y->width][chans_out] = (int8_t(*)[y->width][chans_out])Y_p;

  bnn_b256_t(*X)[x->width][chan_b256_in] =
      (bnn_b256_t(*)[x->width][chan_b256_in])X_p;

  plan->X = (bnn_b256_t*)X[x_loc_height][x_loc_width];

  // Relocate the pointers to the start of the region we care about.
  plan->Y = (int8_t*)&(Y[y_loc_height][y_loc_width][y_loc_channel]);
  plan->K = &(K_p[y_loc_channel * k->shape.height * k->shape.width *
                  chan_b256_in]);  // dereference by y_loc_channel

  plan->post_activation_mul = &(post_activation_multiplier_q[y_loc_channel]);
  plan->post_activation_bias = &(post_activation_bias_q[y_loc_channel]);

  plan->clamp_near = otv->clamp_near;
  plan->clamp_far_0 = otv->clamp_far_0;
  plan->clamp_far_1 = otv->clamp_far_1;
  plan->bias_multiplier = otv->bias_multipler;
  plan->final_shr = otv->final_shr;
  plan->vlsat = otv->accu_shr;
  plan->ashr = otv->accu_shl;

  int32_t bytes_per_input_channel = x->channels / 8;

  assert((x->channels % XS3_VPU_VREG_WIDTH_BITS) == 0);
  assert((y->channels % VPU_INT16_EPV) == 0);

  plan->k_height_loop_counter = k->shape.height - 1;
  plan->k_width_loop_counter = k->shape.width - 1;

  int32_t h_dilation = k->dilation.horizontal;

  int32_t h_stride = k->stride.horizontal;
  int32_t v_stride = k->stride.vertical;

  plan->input_channel_loop_counter =
      (x->channels / XS3_VPU_VREG_WIDTH_BITS) - 1;

  plan->output_channel_loop_counter = (y_sub_channel / VPU_INT16_EPV) - 1;
  plan->y_c_step = (y->channels - y_sub_channel);

  int32_t x_height_loops = y_sub_height;
  int32_t x_width_loops = y_sub_width;

  plan->x_height_loop_counter = x_height_loops;
  plan->x_width_loop_counter = x_width_loops - 1;

  // Inner Loop, minus one to account for the auto increment in the loop
  plan->inner_x_h_step = bytes_per_input_channel * (h_dilation - 1);

  // TODO multiply x->width by dilation
  plan->inner_x_v_step =
      (bytes_per_input_channel * ((x->width - k->shape.width))) -
      plan->inner_x_h_step;

  // Outer Loop
  plan->outer_x_h_step = bytes_per_input_channel * h_stride;

  plan->outer_x_v_step =
      (bytes_per_input_channel * (int32_t)x->width * v_stride) -
      (plan->outer_x_h_step * x_width_loops);

  plan->y_v_step = chans_out * sizeof(int8_t) * (y->width - y_sub_width);

  // TODO these are for implementing sub-kernels
  plan->k_v_step = 0;
  plan->k_h_step = 0;
}

void bconv2d_int8_DIDO(int8_t* Y_p, const bnn_b256_t* X_p,
                       const bnn_b256_t* K_p,

                       const int16_t* post_activation_multiplier_q,
                       const int16_t* post_activation_bias_q,

                       const output_transform_values_t* otv,

                       const nn_image_params_t* x,   // The full image of x
                       const nn_image_params_t* y,   // the full image of y
                       const nn_window_params_t* k,  // the full kernel k

                       const unsigned y_loc_width, const unsigned y_loc_height,
                       const unsigned y_sub_width, const unsigned y_sub_height,

                       const unsigned x_loc_width, const unsigned x_loc_height,
                       const unsigned y_loc_channel,
                       const unsigned y_sub_channel) {
  nn_bconv2d_int8_DIDO_impl_plan_t plan;

  bconv2d_int8_DIDO_prepare(&plan, Y_p, X_p, K_p, post_activation_multiplier_q,
                            post_activation_bias_q, otv, x, y, k, y_loc_width,
                            y_loc_height, y_sub_width, y_sub_height,
                            x_loc_width, x_loc_height, y_loc_channel,
                            y_sub_channel);

  bconv2d_int8_DIDO_impl(&plan);
}

void bconv2d_int8(int8_t* Y_p, const bnn_b32_t* X_p, const bnn_b32_t* K_p,

                  const int16_t* post_activation_multiplier_q,
                  const int16_t* post_activation_bias_q,
                  const int16_t* quantised_accu_modifier,

                  const output_transform_values_t* otv,

                  bnn_b32_t* data_scratch,

                  const nn_image_params_t* x,   // The full image of x
                  const nn_image_params_t* y,   // the full image of y
                  const nn_window_params_t* k,  // the full kernel k

                  const unsigned y_loc_width, const unsigned y_loc_height,
                  const unsigned y_sub_width, const unsigned y_sub_height,

                  const unsigned x_loc_width, const unsigned x_loc_height,
                  const unsigned y_loc_channel, const unsigned y_sub_channel) {
  nn_bconv2d_int8_impl_plan_t plan;

  bconv2d_int8_prepare(&plan, Y_p, X_p, K_p, data_scratch,
                       post_activation_multiplier_q, post_activation_bias_q,
                       quantised_accu_modifier, otv, x, y, k, y_loc_width,
                       y_loc_height, y_sub_width, y_sub_height, x_loc_width,
                       x_loc_height, y_loc_channel, y_sub_channel);

  bconv2d_int8_impl(&plan);
}

#ifdef NN_USE_REF

void bconv2d_int8_DIDO_impl(nn_bconv2d_int8_DIDO_impl_plan_t* plan) {
  bconv2d_int8_DIDO_impl_ref(plan);
}

void bconv2d_int8_impl(nn_bconv2d_int8_impl_plan_t* plan) {
  bconv2d_int8_impl_ref(plan);
}

#endif  // NN_USE_REF