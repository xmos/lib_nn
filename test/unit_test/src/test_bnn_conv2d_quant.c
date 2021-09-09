// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "helpers.h"
#include "tst_common.h"
#include "unity.h"

int clrsb(int x) {
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

static void measure_quantisation(
    const int16_t *post_activation_multiplier_q,
    const int16_t *post_activation_bias_q,

    const float *post_activation_multiplier, const float *post_activation_bias,

    const unsigned chans_out,

    const int32_t clamp_low, const int32_t clamp_high,

    const int accu_shr, const int16_t bias_multipler, const int final_shr,

    const int32_t receptive_volume,

    float *error_sum, float *abs_error_sum, unsigned *sum_count,
    int tollerant_to_error) {
  int min_bias_rsb = INT_MAX;
  int min_mul_rsb = INT_MAX;
  for (unsigned ch = 0; ch < chans_out; ch++) {
    int rsb = clrsb(post_activation_bias_q[ch]) - 16;
    min_bias_rsb = min(min_bias_rsb, rsb);
    rsb = clrsb(post_activation_multiplier_q[ch]) - 16;
    min_mul_rsb = min(min_mul_rsb, rsb);
  }

  assert((min_bias_rsb == 0) || (min_mul_rsb == 0));

  float max_abs_allowed_error = 1.0;
  if (!tollerant_to_error) max_abs_allowed_error = 3.0;

  for (unsigned ch = 0; ch < chans_out; ch++) {
    // Iterate over all possible VPU accumulator outputs
    for (int32_t vpu_acc = -receptive_volume / 2;
         vpu_acc < receptive_volume / 2; vpu_acc++) {
      // convert to larq accu space
      float larq_accu = -(float)(vpu_acc) + (float)receptive_volume / 2.0;
      float r = post_activation_multiplier[ch] * 2.0 * larq_accu +
                post_activation_bias[ch];
      int8_t ref_output =
          (int8_t)fmin(fmax(round(r), (double)INT8_MIN), (double)INT8_MAX);

      // asm implementation
      int8_t output = bnn_post_activation_reference(
          vpu_acc, ch, post_activation_multiplier_q, post_activation_bias_q,
          accu_shr, bias_multipler, final_shr);

      float error = (float)(ref_output - output);
      *error_sum += error;
      *abs_error_sum += fabs(error);
      *sum_count += 1;

      TEST_ASSERT_TRUE_MESSAGE(fabs(error) <= max_abs_allowed_error,
                               "Abs error too high");
    }
  }
}

void run_quantisation(void (*fun_ptr)(), int tollerant_to_error) {
  float error_sum = 0.0;
  float abs_error_sum = 0.0;
  unsigned sum_count = 0;

  int seed = 0;
  for (unsigned k_dim = 1; k_dim <= 7; k_dim += 2) {
    for (unsigned chans_in = 32; chans_in < 32 * 5; chans_in += 32) {
      for (unsigned chans_out = 4; chans_out < 32; chans_out += 4) {
        unsigned receptive_volume = k_dim * chans_in;

        int32_t larq_clamp_low = 0;
        int32_t larq_clamp_high = receptive_volume * 2;

        int16_t clamp_near;
        int16_t clamp_far_0;
        int16_t clamp_far_1;

        int16_t *post_activation_multiplier_q = (int16_t *)malloc(
            sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
        int16_t *post_activation_bias_q = (int16_t *)malloc(
            sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
        int16_t *quantised_accu_modifier = (int16_t *)malloc(
            sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

        float *post_activation_multiplier =
            (float *)malloc(sizeof(float) * chans_out);
        float *post_activation_bias =
            (float *)malloc(sizeof(float) * chans_out);
        int *chan_overlaps = (int *)malloc(sizeof(int) * (chans_out));
        memset(chan_overlaps, 0, sizeof(int) * (chans_out));

        (*fun_ptr)(post_activation_multiplier, post_activation_bias, chans_out,
                   receptive_volume, &seed);

        int16_t bias_multipler;
        int accu_shr, final_shr;

        bnn_quantise_activation(
            post_activation_multiplier_q, post_activation_bias_q,

            post_activation_multiplier, post_activation_bias,

            chans_out,

            larq_clamp_low, larq_clamp_high,

            quantised_accu_modifier, &clamp_near, &clamp_far_0, &clamp_far_1,

            &accu_shr, &bias_multipler, &final_shr, receptive_volume,
            chan_overlaps);

        measure_quantisation(
            post_activation_multiplier_q, post_activation_bias_q,

            post_activation_multiplier, post_activation_bias,

            chans_out,

            clamp_near, clamp_far_0, accu_shr, bias_multipler, final_shr,
            receptive_volume,

            &error_sum, &abs_error_sum, &sum_count, tollerant_to_error);

        free(post_activation_multiplier_q);
        free(post_activation_bias_q);
        free(quantised_accu_modifier);

        free(post_activation_multiplier);
        free(post_activation_bias);
        free(chan_overlaps);
      }
    }
  }

  TEST_ASSERT_TRUE_MESSAGE(fabs((float)sum_count / error_sum) > 512,
                           "Mean error too high");
  TEST_ASSERT_TRUE_MESSAGE(fabs((float)sum_count / abs_error_sum) > 256,
                           "Mean abs error too high");
}

void test_normal_quantisation() {
  run_quantisation(pick_post_activation_params, 0);
}

void test_extreme_bias_quantisation() {
  run_quantisation(pick_extreme_bias_post_activation_params, 0);
}

void test_extreme_mul_quantisation() {
  run_quantisation(pick_extreme_mul_post_activation_params, 1);
}

void test_bnn_conv2d_quant() {
  UNITY_SET_FILE();
  RUN_TEST(test_normal_quantisation);
  RUN_TEST(test_extreme_mul_quantisation);
  RUN_TEST(test_extreme_bias_quantisation);
}
