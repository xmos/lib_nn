// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <assert.h>
#include <stdlib.h>
#include <string.h>

#include "helpers.h"
#include "tst_common.h"
#include "unity.h"

#define X_REF_OVERREAD_WORDS (7)
#define DATA_SCRATCH_OVERREADWRITE_WORDS (8)

static const char undef_sentinal = 0x55;

// static const unsigned clamps_count = 16;
static const unsigned clamps_count = 1;

/*
X_ref and K_ref must be initialised before running this.
This function test whole images, i.e. it wont work on a sub image.
*/
static void run_int8_config(
    int8_t *Y_p, int8_t *Y_ref_p, bnn_b32_t *X_ref, bnn_b32_t *K_p,
    bnn_b32_t *K_ref_p,

    float *post_activation_multiplier, float *post_activation_bias,

    int16_t *post_activation_multiplier_q, int16_t *post_activation_bias_q,

    int16_t *quantised_accu_modifier,

    int *chan_overlaps,

    unsigned x_height, unsigned x_width, unsigned k_height, unsigned k_width,
    unsigned chans_in, unsigned chans_out, unsigned h_stride, unsigned v_stride,

    int32_t larq_clamp_min, int32_t larq_clamp_max,

    unsigned y_loc_channel, unsigned y_sub_channel,

    void (*test_fn)()) {
  // printf("*****y_loc_channel:%d y_sub_channel:%d y->channels:%d\n",
  // y_loc_channel, y_sub_channel, chans_out);
  assert(y_sub_channel <= chans_out);
  assert(y_loc_channel < chans_out);

  assert(Y_p != Y_ref_p);
  assert(K_p != K_ref_p);

  unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
  unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

  unsigned receptive_volume = k_width * k_height * chans_in;

  // memset(Y_ref_p, undef_sentinal, y_height * y_width * chans_out);
  // memset(Y_p, undef_sentinal, y_height * y_width * chans_out);

  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u x_height:%u
  // x_width:%u chans_in:%u chans_out:%u larq_clamp_min: %d larq_clamp_max:
  // %d\n",
  //   h_stride, v_stride, k_height, k_width, x_height, x_width, chans_in,
  //   chans_out, larq_clamp_min, larq_clamp_max);

  nn_image_params_t x;
  x.height = x_height;
  x.width = x_width;
  x.channels = chans_in;
  nn_image_params_t y;
  y.height = y_height;
  y.width = y_width;
  y.channels = chans_out;
  nn_window_params_t k;
  k.shape.height = k_height;
  k.shape.width = k_width;
  k.stride.horizontal = h_stride;
  k.stride.vertical = v_stride;
  k.dilation.horizontal = 1;
  k.dilation.vertical = 1;

  larq_ref_bconv2d_int8_out(&x, &y, &k, (int32_t *)X_ref, (int32_t *)K_ref_p,
                            (int8_t *)Y_ref_p, post_activation_multiplier,
                            post_activation_bias, larq_clamp_min,
                            larq_clamp_max);

  bnn_reorder_kernel_tensor(K_p, K_ref_p, k_height, k_width, chans_in,
                            chans_out, chan_overlaps);

  int16_t bias_multipler;
  int accu_shr, final_shr;

  int16_t clamp_near;
  int16_t clamp_far_0;
  int16_t clamp_far_1;

  bnn_quantise_activation(
      post_activation_multiplier_q, post_activation_bias_q,

      post_activation_multiplier, post_activation_bias,

      chans_out,

      larq_clamp_min, larq_clamp_max,

      quantised_accu_modifier, &clamp_near, &clamp_far_0, &clamp_far_1,

      &accu_shr, &bias_multipler, &final_shr, receptive_volume, chan_overlaps);

  test_fn((int8_t *)Y_p, (const bnn_b32_t *)X_ref, (const bnn_b32_t *)K_p,
          post_activation_multiplier_q, post_activation_bias_q,
          quantised_accu_modifier, clamp_near, clamp_far_0, clamp_far_1,
          accu_shr, bias_multipler, final_shr, &x, &y, &k, y_loc_channel,
          y_sub_channel);

  int8_t(*Y)[y.width][y.channels] = (int8_t(*)[y.width][y.channels])Y_p;

  int8_t(*Y_ref)[y.width][y.channels] = (int8_t(*)[y.width][y.channels])Y_ref_p;

  for (unsigned h = 0; h < y.height; h++) {
    for (unsigned w = 0; w < y.width; w++) {
      for (unsigned c = 0; c < y_loc_channel; c++) {
        TEST_ASSERT_EQUAL_INT8(undef_sentinal, Y[h][w][c]);
      }
      for (unsigned c = y_loc_channel; c < y_loc_channel + y_sub_channel; c++) {
        TEST_ASSERT_INT8_WITHIN(1, Y_ref[h][w][c], Y[h][w][c]);
      }
      for (unsigned c = y_loc_channel + y_sub_channel; c < y.channels; c++) {
        TEST_ASSERT_EQUAL_INT8(undef_sentinal, Y[h][w][c]);
      }
    }
  }

  // for (unsigned e=0;e<y_height * y_width * chans_out;++e)
  //   TEST_ASSERT_INT8_WITHIN(1, Y_ref_p[e], Y_p[e]);

  // FIXME - why wont this link? The above is a workaround
  // TEST_ASSERT_INT8_ARRAY_WITHIN(1, Y_ref_p, Y_p, y_height * y_width *
  // chans_out);
}

void impl_bconv2d_int8_pseudo_random(
    const unsigned min_k_height, const unsigned max_k_height,
    const unsigned min_k_width, const unsigned max_k_width,

    const unsigned min_chans_in, const unsigned max_chans_in,
    const unsigned min_chans_out, const unsigned max_chans_out,

    const unsigned chans_in_inc, const unsigned chans_out_inc,

    const unsigned min_v_stride, const unsigned max_v_stride,
    const unsigned min_h_stride, const unsigned max_h_stride,
    void (*valid_impl)()) {
  for (unsigned h_stride = min_h_stride; h_stride <= max_h_stride; ++h_stride) {
    for (unsigned v_stride = min_v_stride; v_stride <= max_v_stride;
         ++v_stride) {
      for (unsigned k_height = min_k_height; k_height <= max_k_height;
           ++k_height) {
        unsigned max_x_height = k_height;
        for (unsigned k_width = min_k_width; k_width <= max_k_width;
             ++k_width) {
          unsigned max_x_width = k_width;

          for (unsigned x_height = k_height; x_height <= max_x_height;
               ++x_height) {
            for (unsigned x_width = k_width; x_width <= max_x_width;
                 ++x_width) {
              unsigned y_height =
                  CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, v_stride);
              unsigned y_width =
                  CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, h_stride);

              for (unsigned chans_in = min_chans_in; chans_in <= max_chans_in;
                   chans_in += chans_in_inc) {
                for (unsigned chans_out = min_chans_out;
                     chans_out <= max_chans_out; chans_out += chans_out_inc) {
                  unsigned chan_words_in = chans_in / 32;

                  size_t K_ref_bytes =
                      sizeof(bnn_b32_t) *
                      (chans_out * k_height * k_width * chan_words_in);
                  bnn_b32_t *K_ref = (bnn_b32_t *)malloc(K_ref_bytes);

                  int32_t over_bytes = compute_int8_over_RW_bytes(
                      chans_in, k_height, k_width, chans_out);

                  bnn_b32_t *K = (bnn_b32_t *)malloc(K_ref_bytes + over_bytes);

                  size_t X_ref_bytes =
                      sizeof(bnn_b32_t) * (x_height * x_width * chan_words_in +
                                           X_REF_OVERREAD_WORDS);
                  bnn_b32_t *X_ref = (bnn_b32_t *)malloc(X_ref_bytes);
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

                  size_t Y_bytes =
                      sizeof(int8_t) * y_height * y_width * chans_out;
                  int8_t *Y = (int8_t *)malloc(Y_bytes);
                  int8_t *Y_ref = (int8_t *)malloc(Y_bytes);

                  assert(X_ref);
                  assert(Y);
                  assert(Y_ref);
                  assert(post_activation_multiplier_q);
                  assert(post_activation_bias_q);
                  assert(quantised_accu_modifier);
                  assert(K);
                  assert(K_ref);

                  assert(post_activation_multiplier);
                  assert(post_activation_bias);
                  assert(chan_overlaps);

                  // printf("h_stride:%u v_stride:%u k_height:%u k_width:%u
                  // x_height:%u x_width:%u chans_in:%u chans_out:%u\n",
                  //   h_stride, v_stride, k_height, k_width, x_height, x_width,
                  //   chans_in, chans_out);

                  int seed = 3;

                  for (unsigned b = 0; b < X_ref_bytes / sizeof(int); b++)
                    ((int *)X_ref)[b] = pseudo_rand(&seed);

                  for (unsigned b = 0; b < K_ref_bytes / sizeof(int); b++)
                    ((int *)K_ref)[b] = pseudo_rand(&seed);

                  unsigned channel_group_size = 16;
                  unsigned receptive_volume = k_width * k_height * chans_in;
                  pick_post_activation_params(post_activation_multiplier,
                                              post_activation_bias, chans_out,
                                              receptive_volume, &seed);

                  for (unsigned clamps_loop = 0; clamps_loop < clamps_count;
                       clamps_loop++) {
                    int32_t larq_clamp_min =
                        pseudo_rand(&seed) % (2 * receptive_volume);
                    int32_t larq_clamp_max =
                        larq_clamp_min +
                        pseudo_rand(&seed) % (2 * receptive_volume);

                    for (unsigned y_loc_channel = 0; y_loc_channel < chans_out;
                         y_loc_channel += channel_group_size) {
                      unsigned channel_groups =
                          (chans_out - y_loc_channel + channel_group_size - 1) /
                          channel_group_size;

                      for (unsigned ch_group_count = 0;
                           ch_group_count < channel_groups; ch_group_count++) {
                        unsigned y_sub_channel =
                            (ch_group_count + 1) * channel_group_size;
                        y_sub_channel =
                            min(y_sub_channel, chans_out - y_loc_channel);
                        assert(y_loc_channel + y_sub_channel <= chans_out);

                        memset(Y_ref, undef_sentinal, Y_bytes);
                        memset(Y, undef_sentinal, Y_bytes);

                        run_int8_config(
                            (int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref,
                            (bnn_b32_t *)K, (bnn_b32_t *)K_ref,
                            (float *)post_activation_multiplier,
                            (float *)post_activation_bias,
                            (int16_t *)post_activation_multiplier_q,
                            (int16_t *)post_activation_bias_q,
                            (int16_t *)quantised_accu_modifier,
                            (int *)chan_overlaps, x_height, x_width, k_height,
                            k_width, chans_in, chans_out, h_stride, v_stride,
                            larq_clamp_min, larq_clamp_max, y_loc_channel,
                            y_sub_channel, valid_impl);
                      }
                    }
                  }

                  for (int32_t delta_min = 0; delta_min <= 5; delta_min++) {
                    for (int32_t delta_max = -5; delta_max <= 5; delta_max++) {
                      int32_t larq_clamp_min = 0 + delta_min;
                      int32_t larq_clamp_max =
                          (int32_t)receptive_volume + delta_max;

                      for (unsigned y_loc_channel = 0;
                           y_loc_channel < chans_out;
                           y_loc_channel += channel_group_size) {
                        unsigned channel_groups = (chans_out - y_loc_channel +
                                                   channel_group_size - 1) /
                                                  channel_group_size;

                        for (unsigned ch_group_count = 0;
                             ch_group_count < channel_groups;
                             ch_group_count++) {
                          unsigned y_sub_channel =
                              (ch_group_count + 1) * channel_group_size;
                          y_sub_channel =
                              min(y_sub_channel, chans_out - y_loc_channel);
                          assert(y_loc_channel + y_sub_channel <= chans_out);
                          memset(Y_ref, undef_sentinal, Y_bytes);
                          memset(Y, undef_sentinal, Y_bytes);
                          run_int8_config(
                              (int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref,
                              (bnn_b32_t *)K, (bnn_b32_t *)K_ref,
                              (float *)post_activation_multiplier,
                              (float *)post_activation_bias,
                              (int16_t *)post_activation_multiplier_q,
                              (int16_t *)post_activation_bias_q,
                              (int16_t *)quantised_accu_modifier,
                              (int *)chan_overlaps, x_height, x_width, k_height,
                              k_width, chans_in, chans_out, h_stride, v_stride,
                              larq_clamp_min, larq_clamp_max, y_loc_channel,
                              y_sub_channel, valid_impl);
                        }
                      }
                    }
                  }

                  for (int32_t delta_min = 0; delta_min <= 5; delta_min++) {
                    for (int32_t delta_max = -5; delta_max <= 5; delta_max++) {
                      int32_t larq_clamp_min = 0 + delta_min;
                      int32_t larq_clamp_max =
                          2 * (int32_t)receptive_volume + delta_max;

                      for (unsigned y_loc_channel = 0;
                           y_loc_channel < chans_out;
                           y_loc_channel += channel_group_size) {
                        unsigned channel_groups = (chans_out - y_loc_channel +
                                                   channel_group_size - 1) /
                                                  channel_group_size;

                        for (unsigned ch_group_count = 0;
                             ch_group_count < channel_groups;
                             ch_group_count++) {
                          unsigned y_sub_channel =
                              (ch_group_count + 1) * channel_group_size;
                          y_sub_channel =
                              min(y_sub_channel, chans_out - y_loc_channel);
                          assert(y_loc_channel + y_sub_channel <= chans_out);
                          memset(Y_ref, undef_sentinal, Y_bytes);
                          memset(Y, undef_sentinal, Y_bytes);
                          run_int8_config(
                              (int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref,
                              (bnn_b32_t *)K, (bnn_b32_t *)K_ref,
                              (float *)post_activation_multiplier,
                              (float *)post_activation_bias,
                              (int16_t *)post_activation_multiplier_q,
                              (int16_t *)post_activation_bias_q,
                              (int16_t *)quantised_accu_modifier,
                              (int *)chan_overlaps, x_height, x_width, k_height,
                              k_width, chans_in, chans_out, h_stride, v_stride,
                              larq_clamp_min, larq_clamp_max, y_loc_channel,
                              y_sub_channel, valid_impl);
                        }
                      }
                    }
                  }

                  free(X_ref);
                  free(Y);
                  free(Y_ref);
                  free(post_activation_multiplier_q);
                  free(post_activation_bias_q);
                  free(quantised_accu_modifier);
                  free(K);
                  free(K_ref);

                  free(post_activation_multiplier);
                  free(post_activation_bias);
                  free(chan_overlaps);
                }
              }
            }
          }
        }
      }
    }
  }
}

void impl_bconv2d_int8_pseudo_random2(const unsigned max_x_height,
                                      const unsigned max_x_width,

                                      const unsigned chans_in,

                                      const unsigned min_chans_out,
                                      const unsigned max_chans_out,

                                      const unsigned chans_in_inc,
                                      const unsigned chans_out_inc,

                                      void (*valid_impl)()) {
  for (unsigned x_height = 1; x_height <= max_x_height; ++x_height) {
    for (unsigned x_width = 1; x_width <= max_x_width; ++x_width) {
      unsigned k_height = x_height;
      unsigned k_width = x_width;
      unsigned y_height = CONV2D_OUTPUT_LENGTH(x_height, k_height, 1, 1);
      unsigned y_width = CONV2D_OUTPUT_LENGTH(x_width, k_width, 1, 1);

      for (unsigned chans_out = min_chans_out; chans_out <= max_chans_out;
           chans_out += chans_out_inc) {
        unsigned chan_words_in = chans_in / 32;

        size_t K_ref_bytes = sizeof(bnn_b32_t) *
                             (chans_out * k_height * k_width * chan_words_in);
        bnn_b32_t *K_ref = (bnn_b32_t *)malloc(K_ref_bytes);
        int32_t over_bytes =
            compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);
        bnn_b32_t *K = (bnn_b32_t *)malloc(K_ref_bytes + over_bytes);

        size_t X_ref_bytes =
            sizeof(bnn_b32_t) *
            (x_height * x_width * chan_words_in + X_REF_OVERREAD_WORDS);
        bnn_b32_t *X_ref = (bnn_b32_t *)malloc(X_ref_bytes);
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

        size_t Y_bytes = sizeof(int8_t) * y_height * y_width * chans_out;
        int8_t *Y = (int8_t *)malloc(Y_bytes);
        int8_t *Y_ref = (int8_t *)malloc(Y_bytes);

        assert(X_ref);
        assert(Y);
        assert(Y_ref);
        assert(post_activation_multiplier_q);
        assert(post_activation_bias_q);
        assert(quantised_accu_modifier);
        assert(K);
        assert(K_ref);

        assert(post_activation_multiplier);
        assert(post_activation_bias);
        assert(chan_overlaps);

        // printf("k_height:%u k_width:%u x_height:%u x_width:%u chans_in:%u
        // chans_out:%u\n",
        //    k_height, k_width, x_height, x_width, chans_in, chans_out);

        int seed = 42;

        for (unsigned b = 0; b < X_ref_bytes / sizeof(int); b++)
          ((int *)X_ref)[b] = pseudo_rand(&seed);

        for (unsigned b = 0; b < K_ref_bytes / sizeof(int); b++)
          ((int *)K_ref)[b] = pseudo_rand(&seed);

        unsigned receptive_volume = k_width * k_height * chans_in;
        pick_post_activation_params(post_activation_multiplier,
                                    post_activation_bias, chans_out,
                                    receptive_volume, &seed);

        for (unsigned clamps_loop = 0; clamps_loop < clamps_count;
             clamps_loop++) {
          int32_t larq_clamp_min = pseudo_rand(&seed) % (2 * receptive_volume);
          int32_t larq_clamp_max =
              larq_clamp_min + pseudo_rand(&seed) % (2 * receptive_volume);

          unsigned channel_group_size = 16;
          for (unsigned y_loc_channel = 0; y_loc_channel < chans_out;
               y_loc_channel += channel_group_size) {
            unsigned channel_groups =
                (chans_out - y_loc_channel + channel_group_size - 1) /
                channel_group_size;

            for (unsigned ch_group_count = 0; ch_group_count < channel_groups;
                 ch_group_count++) {
              unsigned y_sub_channel =
                  (ch_group_count + 1) * channel_group_size;
              y_sub_channel = min(y_sub_channel, chans_out - y_loc_channel);

              memset(Y_ref, undef_sentinal, Y_bytes);
              memset(Y, undef_sentinal, Y_bytes);

              run_int8_config((int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref,
                              (bnn_b32_t *)K, (bnn_b32_t *)K_ref,
                              (float *)post_activation_multiplier,
                              (float *)post_activation_bias,
                              (int16_t *)post_activation_multiplier_q,
                              (int16_t *)post_activation_bias_q,
                              (int16_t *)quantised_accu_modifier,
                              (int *)chan_overlaps, x_height, x_width, k_height,
                              k_width, chans_in, chans_out, 1, 1,
                              larq_clamp_min, larq_clamp_max, y_loc_channel,
                              y_sub_channel, valid_impl);
            }
          }
        }
        free(X_ref);
        free(Y);
        free(Y_ref);
        free(post_activation_multiplier_q);
        free(post_activation_bias_q);
        free(quantised_accu_modifier);
        free(K);
        free(K_ref);

        free(post_activation_multiplier);
        free(post_activation_bias);
        free(chan_overlaps);
      }
    }
  }
}

static void run_int8_sub_image(
    int8_t *Y_p, const int8_t *Y_ref_p, const bnn_b32_t *X_p,
    const bnn_b32_t *K_p,

    int16_t *post_activation_multiplier_q, int16_t *post_activation_bias_q,

    int16_t *quantised_accu_modifier, int16_t clamp_near, int16_t clamp_far_0,
    int16_t clamp_far_1,

    const int accu_shr, const int16_t bias_multiplier, const int final_shr,

    const nn_image_params_t *x, const nn_image_params_t *y,
    const nn_window_params_t *k, unsigned y_loc_width, unsigned y_loc_height,
    unsigned y_sub_width, unsigned y_sub_height, unsigned y_loc_channel,
    unsigned y_sub_channel, void (*valid_impl)()) {
  valid_impl(
      Y_p, X_p, K_p, post_activation_multiplier_q, post_activation_bias_q,

      quantised_accu_modifier, clamp_near, clamp_far_0, clamp_far_1, accu_shr,
      bias_multiplier, final_shr, x, y, k, y_loc_width, y_loc_height,
      y_sub_width, y_sub_height, y_loc_channel, y_sub_channel);

  int8_t(*Y)[y->width][y->channels] = (int8_t(*)[y->width][y->channels])Y_p;

  int8_t(*Y_ref)[y->width][y->channels] =
      (int8_t(*)[y->width][y->channels])Y_ref_p;

  for (unsigned h = 0; h < y->height; h++) {
    for (unsigned w = 0; w < y->width; w++) {
      if ((h >= y_loc_height) && (h < (y_loc_height + y_sub_height)) &&
          (w >= y_loc_width) && (w < (y_loc_width + y_sub_width))) {
        // If the result should have been computed then check it against the
        // reference
        for (unsigned c = 0; c < y_loc_channel; c++) {
          TEST_ASSERT_EQUAL_INT8(undef_sentinal, Y[h][w][c]);
        }
        for (unsigned c = y_loc_channel; c < y_loc_channel + y_sub_channel;
             c++) {
          TEST_ASSERT_INT8_WITHIN(1, Y_ref[h][w][c], Y[h][w][c]);
        }
        for (unsigned c = y_loc_channel + y_sub_channel; c < y->channels; c++) {
          TEST_ASSERT_EQUAL_INT8(undef_sentinal, Y[h][w][c]);
        }
      } else {
        // Otherwise check thet is hasn't been written to
        for (unsigned c = 0; c < y->channels; c++) {
          TEST_ASSERT_EQUAL_INT8(undef_sentinal, Y[h][w][c]);
        }
      }
    }
  }
}

/*
This test check for a fixed x_height, x_width, k_height and k_width a sub-region
of the output is correctly computed. It check this for MIN_CHANS_IN and
MAX_CHANS_IN input channels and MIN_CHANS_OUT to MAX_CHANS_OUT output channels.
Stride are tested, dilations are untested currently.
*/
void impl_bconv2d_int8_sub_image(
    const unsigned full_x_height, const unsigned full_x_width,
    const unsigned full_k_height, const unsigned full_k_width,

    const unsigned min_chans_in, const unsigned max_chans_in,
    const unsigned min_chans_out, const unsigned max_chans_out,

    const unsigned chans_in_inc, const unsigned chans_out_inc,

    const unsigned min_v_stride, const unsigned max_v_stride,
    const unsigned min_h_stride, const unsigned max_h_stride,
    void (*valid_impl)()) {
#define X_V_DILATION 1
#define X_H_DILATION 1

  int seed = 42;

  for (unsigned chans_out = min_chans_out; chans_out <= max_chans_out;
       chans_out += chans_out_inc) {
    for (unsigned chans_in = min_chans_in; chans_in <= max_chans_in;
         chans_in += chans_in_inc) {
      unsigned chan_words_in = chans_in / 32;

      size_t K_ref_bytes = sizeof(bnn_b32_t) * (chans_out * full_k_height *
                                                full_k_width * chan_words_in);
      bnn_b32_t *K_ref = (bnn_b32_t *)malloc(K_ref_bytes);

      size_t X_ref_bytes =
          sizeof(bnn_b32_t) *
          (full_x_height * full_x_width * chan_words_in + X_REF_OVERREAD_WORDS);
      bnn_b32_t *X_ref = (bnn_b32_t *)malloc(X_ref_bytes);

      int16_t *post_activation_multiplier_q = (int16_t *)malloc(
          sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
      int16_t *post_activation_bias_q = (int16_t *)malloc(
          sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

      int16_t *quantised_accu_modifier = (int16_t *)malloc(
          sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
      int32_t over_bytes = compute_int8_over_RW_bytes(chans_in, full_k_height,
                                                      full_k_width, chans_out);

      bnn_b32_t *K = (bnn_b32_t *)malloc(
          sizeof(bnn_b32_t) *
              (chans_out * full_k_height * full_k_width * chan_words_in) +
          over_bytes);

      float *post_activation_multiplier =
          (float *)malloc(sizeof(float) * chans_out);
      float *post_activation_bias = (float *)malloc(sizeof(float) * chans_out);
      int *chan_overlaps = (int *)malloc(sizeof(int) * (chans_out));

      for (unsigned h_stride = min_h_stride; h_stride < max_h_stride;
           h_stride++) {
        for (unsigned v_stride = min_v_stride; v_stride < max_v_stride;
             v_stride++) {
          nn_image_params_t x;
          x.height = full_x_height;
          x.width = full_x_width;
          x.channels = chans_in;
          nn_image_params_t y;
          y.height = CONV2D_OUTPUT_LENGTH(full_x_height, full_k_height,
                                          X_V_DILATION, v_stride);
          y.width = CONV2D_OUTPUT_LENGTH(full_x_width, full_k_width,
                                         X_H_DILATION, h_stride);
          y.channels = chans_out;
          nn_window_params_t k;
          k.shape.height = full_k_height;
          k.shape.width = full_k_width;
          k.stride.horizontal = h_stride;
          k.stride.vertical = v_stride;
          k.dilation.horizontal = X_H_DILATION;
          k.dilation.vertical = X_V_DILATION;

          int8_t *Y_ref = (int8_t *)malloc(sizeof(int8_t) * y.height * y.width *
                                           y.channels);
          int8_t *Y = (int8_t *)malloc(sizeof(int8_t) * y.height * y.width *
                                       y.channels);

          if (y.height == 0 || y.width == 0) continue;

          for (unsigned itt = 0; itt < 1; itt++) {
            for (unsigned b = 0; b < X_ref_bytes / sizeof(int); b++)
              ((int *)X_ref)[b] = pseudo_rand(&seed);

            for (unsigned b = 0; b < K_ref_bytes / sizeof(int); b++)
              ((int *)K_ref)[b] = pseudo_rand(&seed);

            unsigned receptive_volume =
                k.shape.width * k.shape.height * x.channels;

            pick_post_activation_params(post_activation_multiplier,
                                        post_activation_bias, chans_out,
                                        receptive_volume, &seed);

            int32_t larq_clamp_min =
                pseudo_rand(&seed) % (2 * receptive_volume);
            int32_t larq_clamp_max =
                larq_clamp_min + pseudo_rand(&seed) % (2 * receptive_volume);

            // Calculate the entire reference image
            larq_ref_bconv2d_int8_out(
                &x, &y, &k, (const int32_t *)X_ref, (const int32_t *)K_ref,
                (int8_t *)Y_ref, post_activation_multiplier,
                post_activation_bias, larq_clamp_min, larq_clamp_max);

            bnn_reorder_kernel_tensor((bnn_b32_t *)K, (const bnn_b32_t *)K_ref,
                                      k.shape.height, k.shape.width, x.channels,
                                      y.channels, chan_overlaps);

            int accu_shr, final_shr;
            int16_t bias_multiplier;

            int16_t clamp_near;
            int16_t clamp_far_0;
            int16_t clamp_far_1;

            bnn_quantise_activation(
                post_activation_multiplier_q, post_activation_bias_q,

                post_activation_multiplier, post_activation_bias,

                chans_out,

                larq_clamp_min, larq_clamp_max,

                quantised_accu_modifier, &clamp_near, &clamp_far_0,
                &clamp_far_1,

                &accu_shr, &bias_multiplier, &final_shr, receptive_volume,
                chan_overlaps);

            for (unsigned y_loc_width = 0; y_loc_width < y.width;
                 ++y_loc_width) {
              for (unsigned y_loc_height = 0; y_loc_height < y.height;
                   ++y_loc_height) {
                for (unsigned y_sub_width = 1;
                     y_sub_width < y.width - y_loc_width; ++y_sub_width) {
                  for (unsigned y_sub_height = 1;
                       y_sub_height < y.height - y_loc_height; ++y_sub_height) {
                    size_t addressable_Y_bytes =
                        y.height * y.width * y.channels;
                    memset(Y, undef_sentinal, addressable_Y_bytes);

                    for (unsigned clamps_loop = 0; clamps_loop < clamps_count;
                         clamps_loop++) {
                      unsigned channel_group_size = 16;
                      for (unsigned y_loc_channel = 0;
                           y_loc_channel < chans_out;
                           y_loc_channel += channel_group_size) {
                        unsigned channel_groups = (chans_out - y_loc_channel +
                                                   channel_group_size - 1) /
                                                  channel_group_size;

                        for (unsigned ch_group_count = 0;
                             ch_group_count < channel_groups;
                             ch_group_count++) {
                          unsigned y_sub_channel =
                              (ch_group_count + 1) * channel_group_size;
                          y_sub_channel =
                              min(y_sub_channel, chans_out - y_loc_channel);

                          // TODO
                          memset(Y, undef_sentinal, addressable_Y_bytes);

                          run_int8_sub_image(
                              (int8_t *)Y, (const int8_t *)Y_ref,
                              (const bnn_b32_t *)X_ref, (const bnn_b32_t *)K,

                              post_activation_multiplier_q,
                              post_activation_bias_q, quantised_accu_modifier,
                              clamp_near, clamp_far_0, clamp_far_1,

                              (const int)accu_shr,
                              (const int16_t)bias_multiplier,
                              (const int)final_shr,

                              &x, &y, &k, y_loc_width, y_loc_height,
                              y_sub_width, y_sub_height, y_loc_channel,
                              y_sub_channel, valid_impl);
                        }
                      }
                    }
                  }
                }
              }
            }
          }
          free(Y_ref);
          free(Y);
        }
      }
      free(K_ref);
      free(K);
      free(X_ref);
      free(post_activation_multiplier);
      free(post_activation_bias);
      free(chan_overlaps);
      free(post_activation_multiplier_q);
      free(post_activation_bias_q);
      free(quantised_accu_modifier);
    }
  }
}

void impl_bconv2d_int8_directed(void (*valid_impl)()) {
#define h_stride 2
#define v_stride 2
#define k_height 3
#define k_width 3
#define x_height 9
#define x_width 9
#define y_height 4
#define y_width 4
#define chans_in 32
#define chans_out 64

  const unsigned receptive_volume = k_height * k_width * chans_in;
#define chan_words_in (chans_in / 32)

#define K_ref_size (chans_out * k_height * k_width * chan_words_in)
  const bnn_b32_t K_ref[K_ref_size] = {
      1539622939,  1523638586,  2144401211,  -394156741,  1721840443,
      125909467,   -1534772101, -1548935457, 589637337,   -1827854441,
      -1579770841, 1876088943,  -1856096107, -321758107,  1859275883,
      -770818626,  1952103723,  1826239786,  1210442292,  1210409600,
      270878344,   -1872379228, 3228352,     -2034250787, -1531620727,
      -2034675239, -554307077,  1806468858,  1403356922,  -668694790,
      1533068922,  1524942458,  1246021242,  1522585194,  1254147706,
      1852492922,  -418999809,  -191581837,  -61562591,   -1865047519,
      -1207623168, 3083905,     1012993537,  1682590860,  -3224225,
      1540546816,  533883136,   -1663677660, 180414496,   465750050,
      534563122,   683813892,   180374534,   401623530,   -1483800593,
      -448889619,  -448857969,  1199887789,  1200543917,  -1550921331,
      1384310732,  327607276,   -2088311891, 1447072214,  -2137016850,
      -2011267991, 792726552,   -18869866,   -1801473561, 543058977,
      2069909009,  -2074218,    -36679310,   -198962797,  -1609264140,
      1343628420,  -1859311743, -2127747452, -1030314199, -1835096648,
      -1768249351, -71110538,   1808980094,  -84764290,   -339684994,
      -271396354,  -3083906,    -339422257,  -1981467687, 409035105,
      -1878252656, -1876020348, -1859312748, -1874712496, 281935956,
      -255449068,  2035959894,  835340356,   2035998326,  -313126665,
      -199523606,  1618035408,  -316780417,  -451194773,  -1401330725,
      -329641910,  -467930837,  -1539316821, -1522002737, -700441459,
      1520558920,  -451013491,  -163570483,  -622727912,  -451012355,
      -164621171,  -631134776,  1177096986,  1177125690,  606700475,
      -752241990,  1395238842,  2058200890,  1932394431,  1537583098,
      1537582075,  475008000,   5711361,     741936769,   102239488,
      -2078653371, -2079717851, 531623380,   -1799173531, -2136487227,
      -1609592627, -529420127,  -634265430,  -1607216955, -524438805,
      -630857301,  -127561519,  -1868189213, -625098837,  639501517,
      1698240173,  1681856191,  2000492447,  -1540548705, 1631426439,
      -12746855,   -1221740647, -1280349296, -2821830,    2042943802,
      2030852756,  1473181994,  2144272170,  1036975401,  924758971,
      526957355,   2001484681,  -760958017,  -147998785,  -165104641,
      -1163020361, -416434241,  -1238517827, -1108275388, -1182986683,
      -2114606524, 2130710811,  -149908993,  -410076673,  134744577,
      218436497,   -8121445,    -1862357407, 4140545,     1302662657,
      801298779,   -170735254,  -1859311723, 785565809,   -187545234,
      -1859278955, 785303627,   2031532152,  -1859278915, 656290953,
      798645257,   769526829,   260742186,   1338694762,  1591385198,
      -1291544851, 428250350,   -2121361953, 1043152795,  -5900389,
      -328337957,  -1499776663, -1498726487, -1348813399, 730516076,
      -2104916409, 42631531,    -44789646,   -45035402,   -45039277,
      2030874811,  2030348338,  2072291355,  1361919635,  2133605891,
      2135693339,  -401017847,  1859812649,  1067079099,  -392649599,
      1054371113,  1071148313,  -1028807477, -1731678935, 514666394,
      -1065049497, -1466458013, -1596328661, -666525114,  -2002803613,
      -1533346261, -924475164,  -930175961,  -1265041877, 1577453005,
      1578527048,  1586389996,  -1898436336, -2032654284, -2033670092,
      -1446470601, -1316447178, -505110986,  2958976,     271394465,
      942417562,   -341922177,  -342180033,  -620053057,  -2975362,
      -2966146,    -140722690,  -787667016,  -1844632432, -16740037,
      -1878186072, -1842534480, -1559457413, -1874908543, -2136136910,
      -1509997961, -670222871,  1378828185,  1244610267,  -808797826,
      -270886530,  -270870148,  1058281365,  -48983787,   -111886043,
      319071911,   1535545399,  1468194155,  285517732,   1535410614,
      1532019002,  -1859377232, 1461490588,  1578938683,  1858767226,
      -959805316,  -673538967,  1997836598,  -183201410,  -116092813,
      406679169,   -1741871487, -1707266431, 830439301,   893271941,
      -176209279,  272514177,   320486401,   -1306551263, 349596481,
      1071278405,  -1697219132, -1341748106, 1631591492,  1684674630,
      956730470,   1766231118,  1796660298,  419925710,   1095667340,
      -1822753332, 415263744,   478424854,   -1525804033, 448801792,
      -1616565453, -1489169409, 448965696,   -1112206473, -146926593,
      750929984,   -321763260,  -120428476,  1020418049,  751978582,
      -53322649,   751983619,   768756807,   -1395458841, -1609956347,
      -2130139773, -539183685,  -1593177052, -2142763904, -1713587039,
      760925284,   744148044,   -1940206559, -712992577,  288206469,
      261527671,   1438820396,  271429252,   1338284115,  1448454617,
      271394692,   -1496008325, -2134727710, 8553446,     1667367894,
      -61587054,   -129744558,  1969488854,  -47894253,   2103706131,
      1970084759,  -1849382362, -1778076153, 1195679237,  -783960409,
      -785537403,  1363973765,  -183714641,  1965332111,  1359855277,
      1612595396,  2034909418,  957236966,   -27471384,   1591363946,
      1574888366,  -696665831,  1587063162,  1589307770,  -1069124890,
      -1605987585, -406415137,  -1072295707, -535424779,  -720954673,
      -1067888425, -1056517401, -975908105,  656033865,   1719812304,
      -457158425,  1191856221,  1086341204,  1539625686,  -161480196,
      1480608460,  1533397718,  -946276870,  -569542722,  825044612,
      -19885702,   -560192088,  288304772,   -137310121,  -694509497,
      -1320210783, 272763972,   272760836,   272662548,   -62605421,
      -198363757,  2118497687,  -291586565,  -1348821509, -409297415,
      -568979131,  -704235187,  -586270243,  -559015739,  -1699792187,
      -1733313707, -56124211,   -621957563,  -1729873211, -129990366,
      657137032,   -1011893799, 791353865,   -944809508,  -657071821,
      128850389,   -120201165,  -1135317469, -2137545879, 1478494720,
      2137328662,  -896421533,  -1998378455, 224464409,   -825183909,
      -1932263829, -844293029,  -797597151,  -2143536127, 1739200989,
      70200961,    128604569,   125835741,   129642153,   130709977,
      -1614653090, -2033704624, -2046287532, -2041026224, -238503883,
      -1861437419, -1773610921, 2034695338,  1766235275,  2076499198,
      1388349504,  1388505090,  1531112192,  1522974570,  1522903722,
      1383050160,  -762006037,  1517633401,  373238673,   2002786712,
      -1876086316, -2066304925, 1734352281,  -1876090732, -1936285085,
      2145541913,  1387070652,  -1465997781, 53254612,    -455823069,
      925523340,   -762011369,  -1395491799, 931290588,   -862607309,
      -1483715511, 1408650693,  -157834882,  -1876056187, 1183486291,
      -57484949,   -1859377535, 1863489915,  -49242774,   288106129,
      2066259067,  -1745405552, -674045541,  -2074221659, -942378535,
      -674176526,  -1747522574, -825103494,  -674176014,  -976950282,
      1875516697,  1465915800,  -796458028,  -1534149589, 1801985433,
      1533346772,  -1600865245, 801651977,   2078623132,  -251342594,
      1896137943,  1830078419,  403580041,   -1774312055, -1633800311,
      -2034198404, 786102654,   -1883076482, -1500857045, -1497711191,
      -1499775621, 50988740,    -2146859324, -2045860266, -608025898,
      1212564052,  1213096534,  2086034500,  -61731836,   -61469631,
      1342848286,  269237300,   -1828014298, -678652930,  -476933122,
      -1013796353, 608844809,   2829965,     321851284,   794430465,
      1059729304,  1005526034,  459935232,   963419683,   963493391,
      -696208945,  -2145410007, 1633753104,  1474320793,  -2070548029,
      -1397266903, -1357095911, -4173488,    -1801984017, -828719254,
      1393208958,  1326091390,  -682609666,  -750865410,  -1018283137,
      794888853,   -676514849,  -1797803145, -2110937471, 264427899,
      -52764177,   -1842534779, 1876089211,  -170754778,  304981649,
      1876056423,  -246223827,  873078345,   -2026989092, -254499981,
      1202592220,  -791305933,  -1127207389, -254435021,  -1133441495,
      730549468};

  int32_t over_bytes =
      compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);
  bnn_b32_t *K =
      (bnn_b32_t *)malloc(K_ref_size * sizeof(bnn_b32_t) + over_bytes);

#define X_ref_size (x_height * x_width * chan_words_in)
  const bnn_b32_t X_ref[X_ref_size + X_REF_OVERREAD_WORDS] = {
      -1837422508, 276392448,   1516857880,
      317303634,   -1698321546, -2099171372,
      -2098484012, 181661554,   0,
      -221120386,  450455632,   1522658170,
      1541524350,  -271461518,  1868645146,
      1878250718,  148104050,   0,
      1876122748,  -822346382,  1341780858,
      2143374202,  1876023679,  2146519358,
      1801573972,  -757992590,  0,
      2143379290,  1525799294,  -355338446,
      800152376,   -271327901,  800213434,
      1395902190,  446030778,   0,
      802905656,   -271459458,  1608635262,
      -1968326,    -337537026,  -1410223114,
      -656300166,  -1966493834, 0,
      -1535386002, 2144993075,  1060969887,
      -1524031753, -1184543018, -1182986506,
      -1318228266, -512141617,  0,
      -327328569,  -1342472321, 560946902,
      825167606,   963595910,   560946886,
      -395878657,  -482615603,  0,
      1698651789,  2101305229,  1967095695,
      -784335185,  -314614099,  -448799505,
      1832804009,  -415244595,  0,
      0,           0,           0,
      0,           0,           0,
      0,           0,           0};
#undef X_ref_size
  const float post_activation_multiplier_original[chans_out] = {
      0.012351940385997295, 0.01575915887951851,  0.008943307213485241,
      0.010224299505352974, 0.01991458423435688,  0.008564063347876072,
      0.013291306793689728, 0.014130263589322567, 0.014040169306099415,
      0.008283263072371483, 0.012238231487572193, 0.00907240342348814,
      0.017485272139310837, 0.011195400729775429, 0.008301754482090473,
      0.01243166159838438,  0.011653591878712177, 0.013148283585906029,
      0.013521087355911732, 0.016607023775577545, 0.014706429094076157,
      0.013818399049341679, 0.010542008094489574, 0.01690087839961052,
      0.012508699670433998, 0.014033292420208454, 0.020144738256931305,
      0.016180073842406273, 0.013261908665299416, 0.017590731382369995,
      0.016404960304498672, 0.016385167837142944, 0.009051515720784664,
      0.012140316888689995, 0.017523031681776047, 0.009600243531167507,
      0.014154208824038506, 0.01888112537562847,  0.011789664626121521,
      0.0131512600928545,   0.012670175172388554, 0.01303008757531643,
      0.010212906636297703, 0.010999965481460094, 0.02007177844643593,
      0.02180858887732029,  0.013997333124279976, 0.011472060345113277,
      0.010532037355005741, 0.026670731604099274, 0.008986983448266983,
      0.016481904312968254, 0.02010033093392849,  0.021539287641644478,
      0.01189776323735714,  0.012550276704132557, 0.020099526271224022,
      0.010953588411211967, 0.018284793943166733, 0.0078839510679245,
      0.01716633513569832,  0.015060598962008953, 0.01391292829066515,
      0.017445214092731476};
  const float post_activation_bias_original[chans_out] = {
      -0.05970403924584389,  -0.03189457207918167,  -0.10995983332395554,
      -0.1819787621498108,   -0.08313516527414322,  -0.14007125794887543,
      -0.06474795192480087,  -0.05543561652302742,  -0.04212765023112297,
      -0.11703118681907654,  -0.08005634695291519,  -0.09276312589645386,
      -0.040493838489055634, -0.08497308194637299,  -0.10957623273134232,
      -0.06009465456008911,  -0.08109637349843979,  -0.06643685698509216,
      -0.14931024610996246,  -0.060489218682050705, -0.05786073952913284,
      -0.0758678987622261,   -0.08722170442342758,  -0.0533980168402195,
      -0.05867760628461838,  -0.0976688489317894,   -0.06088452786207199,
      -0.08050064742565155,  -0.047402285039424896, -0.07298173755407333,
      -0.042324043810367584, -0.0965159684419632,   -0.0762302428483963,
      -0.08213816583156586,  -0.037101179361343384, -0.08617115020751953,
      -0.06318537890911102,  -0.02164478600025177,  -0.08504659682512283,
      -0.06211598962545395,  -0.04860047996044159,  -0.05786619707942009,
      -0.0720161572098732,   -0.0612344816327095,   -0.04064817726612091,
      -0.037796907126903534, -0.059520620852708817, -0.05487140640616417,
      -0.06249295920133591,  -0.0308750718832016,   -0.06349707394838333,
      -0.06473486870527267,  -0.02874387428164482,  -0.029432030394673347,
      -0.062471356242895126, -0.07812318205833435,  -0.05101069062948227,
      -0.08313548564910889,  -0.036270491778850555, -0.08491639792919159,
      -0.03377199172973633,  -0.09227605909109116,  -0.038112759590148926,
      -0.06483688950538635};

  float output_scale = 0.0235294122248888;
  float output_zero_point = 0;
  float backtransform_add = receptive_volume;

  float post_activation_multiplier[chans_out];
  float post_activation_bias[chans_out];
  for (int j = 0; j < chans_out; j++) {
    const float post_mul = post_activation_multiplier_original[j];
    const float post_bias = post_activation_bias_original[j];
    post_activation_multiplier[j] = -1 * post_mul / output_scale;
    post_activation_bias[j] =
        (post_bias + backtransform_add * post_mul) / output_scale +
        output_zero_point;
  }

  int *chan_overlaps = (int *)malloc(sizeof(int) * (chans_out));
  int16_t *post_activation_multiplier_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
  int16_t *post_activation_bias_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int16_t *quantised_accu_modifier =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int8_t *Y = (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);
  int8_t *Y_ref =
      (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);

  assert(X_ref);
  assert(Y);
  assert(Y_ref);
  assert(post_activation_multiplier_q);
  assert(post_activation_bias_q);
  assert(quantised_accu_modifier);
  assert(K);
  assert(K_ref);

  assert(post_activation_multiplier);
  assert(post_activation_bias);
  assert(chan_overlaps);

  int32_t larq_clamp_min = 0;
  int32_t larq_clamp_max = receptive_volume;

  run_int8_config((int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref,
                  (bnn_b32_t *)K, (bnn_b32_t *)K_ref,
                  (float *)post_activation_multiplier,
                  (float *)post_activation_bias, post_activation_multiplier_q,
                  post_activation_bias_q,

                  quantised_accu_modifier,

                  (int *)chan_overlaps, x_height, x_width, k_height, k_width,
                  chans_in, chans_out, h_stride, v_stride, larq_clamp_min,
                  larq_clamp_max, 0, chans_out, valid_impl);

  free(Y);
  free(Y_ref);
  free(post_activation_multiplier_q);
  free(post_activation_bias_q);
  free(quantised_accu_modifier);
  free(K);
  free(chan_overlaps);
}

#undef h_stride
#undef v_stride
#undef k_height
#undef k_width
#undef x_height
#undef x_width
#undef y_height
#undef y_width
#undef chans_in
#undef chans_out
#undef chan_words_in
#undef K_ref_size

void impl_bconv2d_int8_directed2(void (*valid_impl)()) {
#define h_stride 1
#define v_stride 1
#define k_height 1
#define k_width 1
#define x_height 1
#define x_width 1
#define y_height 1
#define y_width 1
#define chans_in 256
#define chans_out 16

  const unsigned receptive_volume = k_height * k_width * chans_in;
#define chan_words_in (chans_in / 32)

#define K_ref_size (chans_out * k_height * k_width * chan_words_in)
  const bnn_b32_t K_ref[K_ref_size] = {
      -1667941237, 1448713456,  -1552317250, -2046627932, 683863560,
      -238015391,  1889621163,  -1146744408, 1120641604,  -1250437226,
      -277467601,  -978365725,  -1522250878, 308013372,   -331097853,
      -2125863291, -1919775941, 1029375758,  -1542469837, -1545955333,
      508442521,   1603061910,  -928550448,  -1650005601, 97032138,
      -928052188,  -1795119754, -754385992,  74226363,    1797003283,
      -598162959,  -1267387633, 368581440,   -1849924868, -1127313993,
      -243522407,  1359073845,  625109473,   2100806078,  965854538,
      -842927740,  -508734750,  1580722819,  -774109499,  1786423356,
      1664346098,  596410574,   1534559771,  208231309,   -737287323,
      -395619248,  1037513094,  -75270160,   821412392,   1960025060,
      404645925,   855517445,   -524824100,  -330281898,  292123595,
      2053550679,  -1039968514, 248779310,   -1945423403, 475366814,
      -191782438,  97737775,    1295415508,  -1428159434, -1978514180,
      1725228072,  487294617,   1423278517,  -2029817860, -596437754,
      1754644150,  1532534010,  2098954194,  -1090639935, -828374736,
      1251759581,  898017649,   -400654573,  1979261503,  -666527720,
      726909885,   1838809637,  -121340345,  1220601163,  -1159949700,
      1905274114,  2001411187,  -690776493,  907651649,   150515155,
      -918102542,  -208075213,  -1322382570, -1503061702, 793077911,
      -782702963,  1747957102,  -873231871,  -152213171,  863055996,
      1298732047,  -1753739469, -1062086736, -321165637,  -860659334,
      -887888935,  1692263905,  -1768497726, -605856536,  514206894,
      1022412182,  -962773383,  -365333555,  -1182180909, 1485982561,
      -860245087,  1058781324,  407607861,   -266828163,  -833312433,
      344992463,   -751096207,  -1287424582};

  int32_t over_bytes =
      compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);
  bnn_b32_t *K =
      (bnn_b32_t *)malloc(K_ref_size * sizeof(bnn_b32_t) + over_bytes);

#define X_ref_size (x_height * x_width * chan_words_in)
  const bnn_b32_t X_ref[X_ref_size + X_REF_OVERREAD_WORDS] = {
      334618952,   990892152,   -1092432719, 1188540535,
      -1814923493, -1584797214, 408949552,   -699399485};
#undef X_ref_size

  const float post_activation_multiplier_original[chans_out] = {
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583};
  const float post_activation_bias_original[chans_out] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  float output_scale = 0.019607843831181526;
  float output_zero_point = -26;
  float backtransform_add = receptive_volume;

  float post_activation_multiplier[chans_out];
  float post_activation_bias[chans_out];
  for (int j = 0; j < chans_out; j++) {
    const float post_mul = post_activation_multiplier_original[j];
    const float post_bias = post_activation_bias_original[j];
    post_activation_multiplier[j] = -1 * post_mul / output_scale;
    post_activation_bias[j] =
        (post_bias + backtransform_add * post_mul) / output_scale +
        output_zero_point;
  }

  // printf("post_activation_bias: [");
  // for (int j = 0; j < chans_out; j++) {
  //   printf("%f, ", post_activation_bias[j]);
  // }
  // printf("]\n");
  // printf("post_activation_multiplier: [");
  // for (int j = 0; j < chans_out; j++) {
  //   printf("%f, ", post_activation_multiplier[j]);
  // }
  // printf("]\n");

  int *chan_overlaps = (int *)malloc(sizeof(int) * (chans_out));
  int16_t *post_activation_multiplier_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
  int16_t *post_activation_bias_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int16_t *quantised_accu_modifier =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int8_t *Y = (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);
  int8_t *Y_ref =
      (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);

  assert(X_ref);
  assert(Y);
  assert(Y_ref);
  assert(post_activation_multiplier_q);
  assert(post_activation_bias_q);
  assert(quantised_accu_modifier);
  assert(K);
  assert(K_ref);

  assert(post_activation_multiplier);
  assert(post_activation_bias);
  assert(chan_overlaps);

  int32_t larq_clamp_min = 0;
  int32_t larq_clamp_max = receptive_volume;

  run_int8_config((int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref,
                  (bnn_b32_t *)K, (bnn_b32_t *)K_ref,
                  (float *)post_activation_multiplier,
                  (float *)post_activation_bias, post_activation_multiplier_q,
                  post_activation_bias_q,

                  quantised_accu_modifier,

                  (int *)chan_overlaps, x_height, x_width, k_height, k_width,
                  chans_in, chans_out, h_stride, v_stride, larq_clamp_min,
                  larq_clamp_max, 0, chans_out, valid_impl);

  free(Y);
  free(Y_ref);
  free(post_activation_multiplier_q);
  free(post_activation_bias_q);
  free(quantised_accu_modifier);
  free(K);
  free(chan_overlaps);
}

#undef h_stride
#undef v_stride
#undef k_height
#undef k_width
#undef x_height
#undef x_width
#undef y_height
#undef y_width
#undef chans_in
#undef chans_out
#undef chan_words_in
#undef K_ref_size

void impl_bconv2d_int8_directed3(void (*valid_impl)()) {
#define h_stride 2
#define v_stride 1
#define k_height 6
#define k_width 4
#define x_height 12
#define x_width 8
#define chans_in 256
#define chans_out 16
#define y_height 7
#define y_width 3

  const unsigned receptive_volume = k_height * k_width * chans_in;
#define chan_words_in (chans_in / 32)

#define K_ref_size (chans_out * k_height * k_width * chan_words_in)

  const bnn_b32_t K_ref[K_ref_size] = {
      -1667941237, 1448713456,  -1552317250, -2046627932, 683863560,
      -238015391,  1889621163,  -1146744408, -510686028,  -1117492961,
      -1359528875, 1575638974,  -722571407,  -1132221422, -1794628694,
      -351730166,  2073851381,  -299825408,  -339197266,  86240880,
      1065204944,  -884812668,  1411160468,  -881556368,  -268921696,
      -1516645369, 1797568645,  917093437,   52623962,    930837777,
      374553873,   -1540581651, 885788055,   821476919,   -317846905,
      -1493396616, 1825118029,  -384905893,  1797368966,  1986906752,
      -501859916,  1449020872,  -1542883958, 1153968776,  156102132,
      1660464794,  -411684486,  -1017545459, 456771578,   -1604237591,
      -2110008794, -1263008829, -658925958,  -1043093987, 792415369,
      1232129818,  1050873902,  -1382741130, -974796704,  18050398,
      -1509459442, -1842985195, 1060688046,  2023663406,  1059660126,
      -151897591,  724181254,   411009610,   -2022854745, -1848363909,
      -1873328038, -1840368590, 358295689,   1915862718,  639105619,
      -2105444226, -883520861,  465945796,   759834270,   -419334040,
      -590126700,  137890549,   -721838176,  1032986018,  -1542711232,
      -102718783,  846493625,   -274066254,  1691187192,  1370948786,
      902352054,   1927602036,  566937337,   -210278301,  469894430,
      587513464,   582317797,   884040362,   -95852467,   617280905,
      -2087303928, -1880489655, 1281494289,  1711827513,  1901089844,
      -1272489895, 1844998611,  1638795977,  1417040717,  -1791913728,
      891291125,   -1726947273, -1177398108, 795568422,   1650474124,
      223671067,   -610707905,  259844201,   1525448706,  183457105,
      95437854,    -892550277,  804086976,   1029553192,  4329603,
      -743987387,  -1231163561, 1155699928,  -173577849,  1429335139,
      -430586326,  -322775347,  -1192876494, -2081842400, -711895613,
      1685507085,  -6460901,    1877650004,  -145157623,  2141632549,
      569330137,   502636811,   -1795800820, 481217137,   2127914031,
      1146310783,  -2027663775, -1707834214, -2091038253, -1387681643,
      -1750381361, 1044305327,  -797569616,  -914887025,  955953949,
      1905861531,  2140072119,  -1420465164, -1585426452, -876392241,
      822927436,   377439080,   -593159933,  -1431151484, -928117242,
      1245120517,  -2053658674, 608655455,   -1213929022, -1949002028,
      387622597,   1931737297,  -169700243,  -306218413,  49588901,
      -1358163642, -1325646177, -579376197,  -1896922477, -169573911,
      817332771,   16626886,    1476502317,  -2125605532, -1394967760,
      -2140940661, -1674083635, -324141352,  1182178370,  -52737195,
      662563825,   413760921,   1120641604,  -1250437226, -277467601,
      -978365725,  -1522250878, 308013372,   -331097853,  -2125863291,
      -1725117690, -1445525973, 1154367277,  601322983,   -1424108384,
      -217565623,  -1272143351, 1328064298,  259806782,   -1864131726,
      -1147615362, -156159418,  366215932,   421081801,   -688843398,
      1991629255,  -395828508,  365945591,   172830778,   1313109683,
      1607665156,  -121881757,  -1936327556, 2046212711,  -252030892,
      -2083031693, -2069977742, 1254663107,  -1522576285, -1198133754,
      1941662633,  -1450082101, 1012051825,  -44246503,   -1681476908,
      -1559671783, -494973537,  872440938,   1252093354,  1486587199,
      -694062764,  -65822295,   956649669,   112844649,   -1107121794,
      -182683395,  -486759621,  423257375,   214053375,   -1818516361,
      -1342585436, -1971519822, -1680569684, -1594261625, -1718840234,
      1139600479,  1067053432,  -927037260,  -1202370708, 64089229,
      -729507829,  -591751865,  -1820402212, 530214911,   -1448178731,
      -1134968599, -1834534102, 1624188908,  78290799,    -158280725,
      -1058373056, 1418447327,  1793911321,  -323674568,  -1083867082,
      -2130091880, -72951185,   619439057,   156145908,   -1888752974,
      -133135106,  -1346128821, -1887110818, -1106995698, 1374023680,
      -602267334,  -1328548693, 1047343182,  1129249344,  -338771866,
      60155234,    1530359181,  -1474981956, -1244038156, 557995479,
      -2098494726, 1506448140,  1135252417,  1347688253,  -1075347713,
      2093303952,  -1294572246, -1688281906, -1245365224, 381812964,
      -987019166,  572352865,   -948347794,  -1082135946, 1942644462,
      1297285610,  -552887825,  707977116,   56495731,    1212726581,
      746522704,   -830522772,  438895309,   690006296,   -482248230,
      -621580341,  1648756184,  601795596,   1283685904,  -1670508397,
      -1646038574, -1461316697, -341621714,  25120277,    -1144223900,
      1431570440,  1919526210,  316350954,   -1522726584, 2140590224,
      -642616892,  1277434433,  293337608,   -1785192669, -1327053570,
      -931543035,  1852724331,  -2064972234, 1443330176,  754939349,
      -1171480285, 557417725,   1856001029,  587588690,   -386267152,
      -1183869396, -664106585,  192598167,   -730427595,  1482463361,
      785588090,   386853927,   573378412,   -1610318369, 1087719694,
      1281232895,  398140326,   894841511,   1495105169,  1845792976,
      -2022330152, -1327152198, 1855858312,  -911631985,  1442394372,
      1766104157,  -1101962024, 30398597,    -974122142,  193666332,
      1984271243,  1582959064,  999081941,   2034505306,  -1075953166,
      811207728,   1086040063,  -1646115710, 1663279773,  -1919775941,
      1029375758,  -1542469837, -1545955333, 508442521,   1603061910,
      -928550448,  -1650005601, 1899852126,  -729246378,  2097472628,
      -1837392796, 1991189455,  1246733952,  -1880120237, -942651097,
      1272972211,  1815184983,  -1486878038, -1797031021, -661992692,
      753546717,   -1070909327, -922179567,  1891811062,  1996127373,
      -642061893,  1274918072,  516287671,   -7642009,    2094869494,
      -224221082,  467427099,   640756424,   -1821808651, -1476963450,
      -1565748334, 209142646,   495167143,   -1656492792, 208092062,
      1185781668,  -1616572742, 188832272,   1089780751,  1139930821,
      -10701255,   295491137,   -1636542696, 1158039203,  -1306157477,
      1638533402,  973465426,   207042203,   -1119626660, 1376691912,
      519265050,   473451469,   -313158308,  1285032804,  1437701882,
      -1910236517, 962234166,   -1938647398, 901946688,   -1469390201,
      738569693,   54805347,    -626348666,  2037225579,  1219427475,
      1202312037,  823874438,   769883314,   -1175364530, -1929888814,
      1271304238,  -240564843,  721906679,   -1682376342, -1670784742,
      1825393685,  1028608323,  1957479462,  364827202,   1261168878,
      1855736655,  1260997092,  -404618655,  -210619555,  -121239519,
      1255453061,  -658931044,  843966312,   -750013997,  -1454410032,
      913898360,   -586278647,  -902686826,  1467946219,  -433677267,
      2143407397,  -1375422519, -887887726,  207890498,   1828589249,
      -540533083,  -1239180549, 982371634,   1645416885,  1968168664,
      569957782,   -1528469289, 726432254,   -1682237637, 1336262576,
      -1464451601, -1178784556, 1988283690,  887587490,   -511788697,
      -1180574589, 1634858431,  304649925,   -1307527407, -270229599,
      -247824168,  -1996470449, 1230632038,  1305774516,  -1726380577,
      586295632,   1164703539,  370953858,   -193499839,  560549623,
      -1677146978, 458513726,   -1322815084, -871266679,  772741557,
      1731234157,  1727977598,  -923630308,  -383917358,  -695599419,
      -1186500697, 2084094398,  -1580322573, -1457076303, -439188681,
      -549930670,  372467801,   351201582,   -1929984931, 410862191,
      -1695762342, -485694014,  -1171656167, 1850942494,  -617071938,
      813598247,   2095020307,  531736533,   -1401576683, -352159218,
      -29307060,   1569253484,  -193953926,  -951109721,  -1420342907,
      1309800976,  822714836,   1235315866,  -1514865662, 575739347,
      1875527250,  -2133870533, -30401898,   -1571732410, -542190978,
      1673575850,  795860701,   1013818754,  -1814758930, 1711724355,
      -192961575,  -17028148,   87394740,    990644554,   -929538922,
      -128449753,  97032138,    -928052188,  -1795119754, -754385992,
      74226363,    1797003283,  -598162959,  -1267387633, -178069967,
      453580556,   1019346714,  -1870844102, -1042754276, -2133283482,
      1318907030,  -1262390465, -712495502,  1793984743,  940799802,
      -1445819417, 1403743572,  -39146788,   1304435112,  1672010402,
      -44804667,   -37394629,   -1133150657, 163730565,   -598584981,
      607821671,   1174280707,  1870414728,  48352308,    -1419598460,
      661411806,   1190655451,  2029308542,  299110212,   556831028,
      494419878,   143371939,   -1601028993, -1827682943, -670723776,
      721624800,   610903893,   266735321,   165299804,   -807040312,
      -1755165606, 997071353,   1709197121,  -2009492122, -1558970842,
      756690845,   1243330212,  1062357276,  -99567439,   -2090637415,
      -1591141357, -1947586052, 1107806435,  -69552022,   1485156293,
      -708673998,  1142362738,  697698448,   -1980328157, -738455663,
      4940971,     1709277485,  198519962,   503808160,   -676694481,
      -640406014,  171271155,   109912585,   -1549118656, 804722312,
      -564656023,  532978088,   -458518597,  -468527038,  -865747510,
      -396860308,  -662587754,  323666568,   929456329,   -719917634,
      -157862534,  914452441,   -1332395001, 1804707437,  1246683773,
      884512164,   1359581081,  -1642181681, -52344612,   1124246108,
      842731577,   -302062451,  -1018994327, -849005080,  1768701334,
      1399530376,  -1234216715, 1463770535,  778518992,   -545612677,
      -168888185,  2117525579,  -593241641,  978627603,   -1000626087,
      -1230873189, 1147178204,  -1919515876, -1035159159, -1708385215,
      1461253211,  430985793,   1768895399,  -781685585,  933169771,
      384447244,   1109655113,  184009518,   96636863,    1343831110,
      1800113532,  -2112394535, 611677538,   -2099875360, -1838794872,
      621536314,   187206110,   429526967,   546690918,   -2044339169,
      -1658679408, -1366062319, -1697427630, 1461731607,  -336037476,
      378553367,   -406019105,  1221064505,  -614807484,  -1941752817,
      1900499704,  -1370392092, -743881137,  2093385389,  2102810357,
      1549013738,  236804642,   -47837786,   -1647099020, 129620537,
      50153037,    1222983259,  1974839169,  -1923330292, 1644398946,
      1763451090,  -205849221,  947025374,   -619401808,  964315601,
      1125792313,  -401272577,  2031546762,  -1664725738, -563932605,
      -1091322838, -1275372549, 1215517701,  -702026396,  608223050,
      991493186,   1136556955,  -930030060,  -903102298,  -879512929,
      -226006849,  643611475,   -170598945,  -1376175513, -1273741931,
      352778640,   -1143349133, -2114934506, 368581440,   -1849924868,
      -1127313993, -243522407,  1359073845,  625109473,   2100806078,
      965854538,   -398710766,  -1922049838, -934835212,  1558546376,
      -1307030999, 1948088732,  -1247912718, 884515528,   1713928283,
      41640322,    -525017315,  1005343649,  214821236,   1650573662,
      1755487379,  1063098071,  853050880,   1496614893,  1958407620,
      -22887353,   1216010322,  2038043267,  -747299558,  -429667037,
      1416361750,  -20044480,   -1644850589, -1855706259, -260286862,
      -650361847,  -525938855,  -249778461,  -203619691,  532373689,
      2008953904,  -1811238593, 728982684,   1456739281,  1049553156,
      -1398736977, 726545991,   892697664,   -192182380,  2001448839,
      -1504229469, -1712477212, 50778625,    1317430335,  1055336887,
      1314690771,  704892935,   1617737201,  1884491193,  -642875302,
      -241562353,  1888384865,  -1583249035, 1170897559,  247302843,
      -1007740878, -1155409240, -348153714,  307125256,   856218022,
      -1804888027, -475628856,  169837292,   378332441,   1187207135,
      -751179729,  -483186394,  -462366449,  -717268275,  9951226,
      1170220665,  1813018353,  806158053,   1871023560,  1172794522,
      -2074022230, -165187699,  -1538157007, 1031415846,  1994899992,
      220006450,   1816702014,  2011463476,  -1117696994, 695089829,
      324570025,   -1665757126, -1181521642, -1575752542, -362889424,
      417935540,   1628283252,  13502526,    989379039,   1038413576,
      -904156688,  414271781,   1038974054,  306289850,   -1478368966,
      -461355859,  -1821570828, -1627246088, 1101954022,  2055583437,
      1396406156,  -1528190156, -1611678130, 1164274544,  -538083259,
      11141739,    37916033,    744076110,   -687026340,  -356915314,
      -886540402,  304803835,   -454165086,  2122237437,  1819326880,
      42587558,    -88049391,   1360049493,  801382297,   1251815268,
      -985555825,  1617156062,  -1420994494, 488594573,   -1807444728,
      530194834,   -1727554555, 741875364,   -1637536364, 63307854,
      202397404,   733960381,   -498131624,  -1313890894, -1627441347,
      -1435127798, -1988888410, -635450389,  1184688402,  1150857573,
      -1237321431, 2099371901,  -1053035364, 411330134,   61002284,
      1747535049,  -1665477613, 136906983,   368000154,   1191426193,
      -834291273,  -1505680544, -327852650,  -1911425331, 1598321990,
      -1423893253, 129220327,   -1531092039, 1326385627,  123056006,
      48988175,    -1876658289, 327805778,   1384516111,  313566776,
      -752291502,  215465436,   -1885926915, 883140858,   1795978298,
      -1240636921, 1474238891,  -815283439,  -463433301,  -1775549967,
      -842927740,  -508734750,  1580722819,  -774109499,  1786423356,
      1664346098,  596410574,   1534559771,  -1070122453, 932196447,
      -467778201,  -1362032638, -849511616,  1271822585,  2130304493,
      584085215,   -577279741,  -1369962780, -1012438802, -351123300,
      -1709051969, -216243733,  1984271163,  430200449,   1487488155,
      791842312,   -1859215477, -226666861,  1325748411,  2060162772,
      -1765353137, -413235022,  -1697103837, 175615170,   -263216146,
      -1573706313, 963725943,   1375284017,  1866358079,  -517857661,
      790182927,   -196978969,  30253897,    304714578,   -513932405,
      -669365248,  -1856882695, 739148309,   -583047682,  -799579938,
      -1554310137, -1106622555, 1541493376,  -1848925632, 198902051,
      1216665612,  1681703070,  -1981903806, 2007360278,  -322427046,
      -1202825958, -1445378264, -877518038,  -2135496149, 1996394576,
      690429346,   -929544393,  1191455387,  931198160,   2142332064,
      -1517942620, 482879309,   428361724,   1629070138,  388986083,
      -1940016529, 839572053,   -1885775661, -513658817,  -106718605,
      906530874,   -809041846,  236977636,   -800390340,  1182186885,
      -1261939211, 1354367730,  -712864560,  -732809062,  701001658,
      -832772893,  -1402993216, -361598839,  -1413024702, 133547178,
      818599434,   536143874,   -1969411046, 2054541725,  -452406747,
      430851951,   -1730756103, 1184830235,  -926988916,  1841867422,
      -1045783821, 758394283,   1637413551,  -1933654488, -954109708,
      -711468679,  -2100020789, 975591165,   -1909410889, 644420872,
      478489099,   169448860,   -1678304320, -180935191,  865352199,
      -733469164,  -1620892964, -1185067212, -594697868,  -1828565346,
      1169788132,  -773693279,  1506104244,  1145143211,  -93485158,
      -155534826,  -546832928,  127051227,   -1039703307, -942306830,
      -485673483,  577939587,   -118577596,  -1658162510, 1621520866,
      1562540543,  -1420038700, -856158159,  -848220765,  912022927,
      -695219491,  -1373152344, 1042437299,  1927923549,  -1456604470,
      -317535498,  -1861110636, -1622882371, 1051025782,  1382867836,
      -1071136940, 1843531507,  -1266698363, 3490709,     119394024,
      -1522463594, -585986075,  1563025400,  721203877,   -1006783969,
      -445287420,  1564661397,  745726301,   1962785996,  1893179218,
      1040757854,  -1028511361, -1317851320, -758094192,  1863754791,
      -530411977,  -1389168646, -1752928218, 1698518389,  -889680298,
      -983940539,  1961966310,  539647097,   331509865,   -257017972,
      2499012,     -1420487898, -1540770475, 2133864740,  1784719593,
      -1352933437, 1775037893,  208231309,   -737287323,  -395619248,
      1037513094,  -75270160,   821412392,   1960025060,  404645925,
      -235000740,  -678084181,  656527556,   206356371,   -1154609068,
      1518633787,  -66180454,   580628359,   -1594309686, -1223233157,
      -726512263,  990932089,   -1695573915, -887285580,  -1151608918,
      -1636434814, 1746032417,  -1712216521, -1829016896, -1092175733,
      -2119646256, 1849728993,  121809903,   -1643131293, -161744410,
      -1412583328, 1026871610,  -270591042,  1474411233,  1272222275,
      1881474891,  1845235561,  1784831919,  917432917,   1860114018,
      -1699794605, 612201877,   1567981402,  -1259251606, 1079982798,
      -1658209612, -1853224813, -1028479835, 1881106346,  1701233716,
      -294904716,  6160350,     867775115,   1881855496,  -1466961730,
      -629648659,  -1291504091, 1214988845,  1812464115,  -2001369950,
      -1810808044, -366885309,  -729724100,  -2014732027, 1629508191,
      -2138236103, 409226111,   856300212,   1352568737,  342097201,
      -1595218921, -420395222,  108470989,   679165076,   1593991888,
      799945815,   -559459109,  398959824,   936508838,   1789810615,
      -1117715194, -1314073830, 108442019,   -1203080131, -1968279604,
      -1447438555, -1972913822, -1970165339, 200436120,   -2130584816,
      133588996,   -659365515,  -1754700884, 1930118398,  683919544,
      -1193651133, 2106748728,  -2099282178, -435304308,  74180249,
      -834076677,  1182965538,  491245881,   1467266066,  1642694599,
      -245974880,  -1855633772, -982303003,  -354259470,  -574142925,
      -1094357744, 1917018266,  -1423913756, -2005434193, 187241422,
      1477439505,  1402395602,  772765768,   -527784564,  353217909,
      -1371451951, -297755436,  1204598619,  -1487726742, -1927825368,
      724823179,   2113614258,  -1286374507, 746531587,   883734714,
      1234106983,  1909882770,  1027847848,  846307660,   -631838354,
      1409520408,  2035381072,  848862541,   107309559,   2020935148,
      76422094,    -321073608,  1093673878,  -504881112,  632521932,
      1870360756,  -393197579,  233749326,   1862516197,  491401631,
      -1169868827, 1453863883,  -623108361,  108908243,   327512624,
      -485076312,  1945639199,  1366261538,  -587081779,  -592210228,
      1142999017,  -1613924264, 1424922832,  1635978494,  2061505397,
      1639758441,  -1080974912, 1931194458,  920613181,   2025707248,
      -1328071247, 1563537650,  547061628,   -1141938615, 1793428359,
      -860520251,  339494078,   -164766017,  54067583,    -1147236062,
      1443444718,  -678683274,  -1405301399, 2053473960,  -983187465,
      -1382441116, -1833347961, -447684848,  201297234,   855517445,
      -524824100,  -330281898,  292123595,   2053550679,  -1039968514,
      248779310,   -1945423403, 497783023,   -342710125,  -2112515545,
      1702009181,  -444029779,  -168835709,  -2012037728, 1323834811,
      2096788474,  -1933480890, -1516594395, -2002652866, -118079537,
      -1895174086, 1743871126,  1184034495,  1193546518,  -703150375,
      79169547,    -870655592,  1984063025,  -318638384,  -451740196,
      -1138204537, -989335443,  -1322244241, 1442823707,  -751651545,
      541078403,   352694195,   -1137966883, 1033791148,  484487014,
      886755453,   -2120591015, -2135496436, -305760903,  -1372028553,
      -346449791,  245437849,   -299199530,  2072440512,  1510071126,
      -1219630667, 1579735986,  -1546430255, -1183327191, -611527883,
      -907521727,  1771456512,  -1191898940, -682402855,  -1343025004,
      -1565031548, -2055272812, -871744403,  1430722241,  -441023268,
      -1060079326, -280636963,  -1484460826, -448773025,  619475127,
      -552432109,  885347607,   532608743,   -467449970,  153583300,
      -742856193,  -11757619,   -1372369265, 394366050,   -350988626,
      -543688093,  -579604206,  1632328249,  582024643,   1363118399,
      928539381,   -782018653,  227799644,   665337958,   -538269341,
      -1817427346, -1026642093, -158157524,  1340616551,  -1879985999,
      1651659422,  1880773368,  1119121039,  -346216348,  1774354653,
      943830154,   1847990948,  -862196993,  591287044,   -1148310950,
      1564859697,  283829882,   489102419,   -1226279965, 597507124,
      1481402529,  -1634097762, 441799478,   -1572389446, 792739207,
      -1605394713, 705592698,   228428786,   1094609549,  155962167,
      1453722077,  -761137042,  -431465467,  567787519,   1620196546,
      714679398,   344075410,   773557367,   1612758545,  1198355873,
      1654986014,  -2042858165, 305401732,   -327808482,  -1993685720,
      -2117161888, 246729904,   -1918737399, -1816410836, -1777651831,
      -900449444,  -1227986355, -1113257969, 823613054,   2088045790,
      2140897196,  -1726431121, -1812608658, -1530955459, -135696481,
      -646831380,  -1360303303, 1018031907,  561595028,   -1884758207,
      1117768483,  418129810,   1667089113,  -62334275,   -1581810215,
      478675949,   1829600380,  532165946,   504305565,   -1530303622,
      636958537,   -1906536732, 706065824,   -1367681469, -1455326738,
      1967237404,  1925436575,  -1753370677, 475767262,   1597197483,
      -1565356168, -1627259283, 2025779357,  -510317149,  323637897,
      1054399335,  870783126,   185544343,   -235668525,  -1192226886,
      1512104748,  77337162,    368313427,   1960556984,  133437588,
      1677721252,  475366814,   -191782438,  97737775,    1295415508,
      -1428159434, -1978514180, 1725228072,  487294617,   -1892868793,
      -1560100425, 537275657,   -616857226,  144029567,   -929998438,
      -1462898607, 14891458,    -966301517,  -874380283,  975732242,
      -1608179734, 881727115,   270758736,   -1460271375, 75818814,
      1163081222,  -991235142,  1113445731,  -946536654,  1508303649,
      1722811060,  1312297957,  1286396196,  549976225,   -312034157,
      357437724,   71561060,    1821733180,  261462042,   -1642447133,
      1459003800,  1141794322,  359592740,   653941513,   -113914712,
      -756592228,  1923980037,  630032662,   1028204479,  -263601490,
      1205766357,  -211582951,  1108188545,  2043786738,  883904180,
      -600318926,  -1992725843, 1805734051,  -1779649729, -1066395326,
      1211440242,  -517503204,  -288906516,  944987152,   2116428875,
      -623911020,  -468752170,  -1788963952, -127823563,  1620325460,
      229373024,   1786374364,  1962404802,  1261675102,  -8495285,
      -1544242367, 1748297160,  1463208352,  447052974,   643177945,
      -65122611,   -1105386897, -1946421159, 293042922,   -1741157654,
      -1193472858, 1992440681,  963974055,   1474866627,  674131741,
      997304712,   503970512,   -424691755,  153568484,   1061304454,
      898932552,   1284624682,  -28418009,   -1325745845, 1497939114,
      854092084,   2076832517,  -598071153,  1963506546,  -1075479647,
      -1650159438, 1988569689,  -2133591572, 1377323351,  267616519,
      -2054036717, 1980049615,  333586892,   1787814221,  231251801,
      1285322682,  -1129452180, -381092944,  1974005095,  -796268288,
      1114317574,  1351729440,  1398852541,  -2064390219, -1630040811,
      -569097572,  -235121399,  2107572566,  -1701560129, -1154365548,
      -1072163647, 379995194,   -2059175905, 293360107,   -963967022,
      -1491029961, -534947357,  -1026208498, 143906853,   1161885959,
      172403278,   750236162,   -966417531,  -632151596,  1939482549,
      2039161006,  1029477723,  1353215461,  1548159377,  832140960,
      252309642,   -730343752,  -539518807,  2060201523,  -1941919205,
      -984621419,  -1366696672, 1742908971,  -1147730605, 1997538784,
      1475467703,  -1137941533, 1547342004,  793215292,   -1317069625,
      -1198603265, -1253776971, -433812830,  1334981190,  -1169541960,
      920369481,   2008676122,  1442268186,  -1959036287, 1195186271,
      1543678515,  -1617184774, 1037181138,  -1315240047, 2122216330,
      1661443525,  -806936866,  1707059717,  -1282170810, 1142396964,
      1115169252,  1902705085,  -1585195981, 832428267,   1077926539,
      612830193,   1558211149,  -1041880019, 1423278517,  -2029817860,
      -596437754,  1754644150,  1532534010,  2098954194,  -1090639935,
      -828374736,  -2123245881, -1966797784, -1910632496, 2077398796,
      -138494434,  -446453826,  1297858432,  2133197662,  -886594648,
      -846526617,  -1387320208, -269737067,  -352735084,  -1526287894,
      -1055497562, -1336681760, 938527231,   -1924487518, -1080523010,
      1760622139,  1891979485,  -40059104,   -1015339616, 265270872,
      -1313857684, -73419614,   -1744348480, 726537794,   -1328957856,
      -177845751,  1076364404,  -1591718467, -1611664247, -1761671681,
      -572098827,  1673173340,  353211340,   -1115607933, 229092673,
      -65350519,   -1402825634, -372947611,  -1150013289, 117512970,
      74365924,    -121478476,  365927711,   1681032291,  -1437005894,
      819329982,   -450156635,  -29574692,   -320370308,  695443898,
      -165742644,  2049279299,  -1391514755, 576316790,   -456154535,
      1136254057,  212414844,   -866623135,  1469905226,  1026445823,
      -782841612,  -1172089275, -223196452,  1364713222,  -1920398869,
      961569481,   821662321,   174006754,   -555675576,  1540744782,
      -1634376392, -2054929461, 822643930,   331038019,   326942452,
      1260215901,  -1154849020, 1313701351,  1934434073,  750054738,
      163383603,   -1498125946, 773004023,   -805769700,  1190923617,
      -1475082287, 2026366716,  -1110741641, 2097965418,  2119378358,
      -1555942344, 236410598,   -429422441,  -230617732,  251485756,
      -1827213165, 92756813,    -1148017367, -52538236,   -408009556,
      -201744116,  836298633,   2051162872,  -441963042,  1869488127,
      -1234089586, -275786938,  -741117520,  -1625002251, 227563117,
      -1496513450, -494406605,  2001798511,  -1065698252, 1655344841,
      -1994704216, 378844319,   582458158,   231389109,   1017492107,
      156714697,   812667593,   1630605564,  -1978815656, -780568391,
      -1999950583, -1861096498, -1661585294, 1092055333,  72341132,
      974426984,   1936579869,  1004670569,  1106269643,  463176885,
      -1822943149, -1057163032, 2088363450,  452982708,   -1202076503,
      1263518341,  -1307867472, -1036327807, 1351974411,  946174506,
      -355717907,  318795714,   1034462786,  -1196246310, 1130399907,
      764362950,   1387838204,  -886656262,  1249347029,  1081743613,
      1334497491,  1858478109,  -1277688872, 193791760,   -435909979,
      -1093796313, 501896684,   -1121485767, -1543279992, 1016682898,
      1112181933,  1866756226,  -1492088331, 286912323,   -1917534904,
      -1454933913, 1790322195,  -893995565,  -116015169,  -1059549095,
      -2051831510, -1633591658, -81389380,   1569040462,  -1446446449,
      1251759581,  898017649,   -400654573,  1979261503,  -666527720,
      726909885,   1838809637,  -121340345,  -1884221428, -1967959326,
      798932860,   680308978,   56746366,    -2022688937, -803123713,
      1039550356,  -1688529488, 300505906,   -1069346685, 1247355921,
      -2004529676, 1279025508,  1834849500,  -1733100286, 1412223955,
      -1203463738, 1683288093,  -347233271,  449131025,   -586179603,
      -493765428,  957391955,   -1851851951, -1964942449, -1654994431,
      759274684,   -1324574365, -967279412,  -207976148,  -1375638718,
      1078018197,  1005706867,  -319136405,  -788259849,  -1445680865,
      -1674007570, -1590844653, 320600983,   902369354,   504114576,
      -739787636,  -589709139,  210144795,   -547537037,  -1696154476,
      1042401917,  -2145241656, 1977846923,  -1747554111, 418305851,
      -1917179098, 1483995188,  17430012,    933628754,   41632558,
      -901040322,  -1878745740, 1910385800,  1269412741,  1307469193,
      -932266078,  -1685625972, -1850884443, 668503032,   945028261,
      148245667,   -1483772759, 915490915,   1794208007,  -1201765911,
      -1056230755, -1465087440, -929195126,  306992412,   -2076124685,
      2060734901,  -686771896,  -1041741994, -214120095,  516604203,
      -1744900733, 1290487011,  1423921093,  -1383884825, -1703876881,
      761171450,   1186518128,  -730197945,  -4106466,    -821298293,
      -2107692460, 1615610835,  -1677443813, -652278859,  1224308397,
      1743004861,  1446066757,  348576521,   -897934243,  712911798,
      519966100,   820245630,   -964979944,  1272060489,  -407328944,
      727004123,   266519151,   566043267,   -2113118516, -1248717834,
      773873000,   -820750120,  -91001534,   -1825439116, 557435018,
      115784873,   1493951348,  1440221770,  -900887216,  -297108258,
      -1963959743, -1133885727, -1499453474, -400537878,  -1225846720,
      -219853961,  -741807960,  1402994773,  -1205869899, 1186233323,
      -237268505,  -672403521,  2006679715,  1513732167,  -70625360,
      881770270,   736051381,   -1387199313, 734580358,   -2009573534,
      655496847,   -1922380157, 1069940351,  1553978676,  292089250,
      -924302210,  -347236174,  -419673197,  2139411176,  -283307601,
      324558237,   1302913802,  -1055265843, 33021796,    1739460956,
      -2055383924, -374140667,  -1094605507, 117278872,   -332855418,
      -983892319,  793170060,   433092779,   -717403074,  1047441230,
      2021925213,  1946239342,  -885552716,  1433178375,  -795592984,
      510211084,   -5835994,    -1906366063, 410814361,   -78105580,
      -1792601061, -1553448098, -1882971582, -1528952694, -82409527,
      -1251482468, 734609349,   1220601163,  -1159949700, 1905274114,
      2001411187,  -690776493,  907651649,   150515155,   -918102542,
      1982220600,  2080179931,  1726285503,  -1740108718, 423906275,
      -1636944334, 1223430487,  302615699,   -509989774,  -811614459,
      1249431622,  1138768886,  -1568514772, 2012027777,  935308629,
      -614240793,  1026027312,  1078009369,  1442755545,  854787935,
      -561923889,  -707625689,  2046066177,  1353216774,  -1543662248,
      82556400,    -328487520,  1170545997,  -311453425,  -1974251506,
      1565262524,  624113127,   652983938,   36470831,    8249591,
      -1947736275, 594439045,   -64264165,   2043214125,  -2114623488,
      -373557041,  -1432860598, -507816355,  -970391509,  -1276129617,
      1818842484,  -1203715648, 1512716525,  -291500244,  405461184,
      384087122,   1684890061,  1360300780,  1519403356,  1439549275,
      1606073867,  305499208,   499208988,   1392214549,  -44429507,
      -727470110,  -979944365,  1347602287,  -212329108,  108138349,
      1877031459,  -279528944,  1802911080,  1943810904,  1429340608,
      1503552430,  -1748544512, -605970666,  -2089666262, -1862268687,
      -191367804,  -81713303,   22881827,    1321112822,  -1526623337,
      1894394892,  -2080979943, 1065886894,  197337084,   274178924,
      1228175179,  1100505720,  980780776,   714827942,   1598506207,
      608574109,   -2098868516, 1702008652,  2046918766,  -810758294,
      240411154,   -422444342,  -532712597,  -1931006875, 565412321,
      -80144140,   1210593751,  1383345196,  1370618544,  -1596394721,
      -1489958252, 1181853981,  1637777562,  2096932053,  638976488,
      -1238366965, -1479739107, 1572376333,  700324465,   -1869572795,
      -623381161,  -351555313,  758451780,   1217512064,  1976435193,
      1911506998,  1075469171,  1726439297,  1770021659,  -701884110,
      1052289426,  587376022,   1398861433,  -2146081301, -497780568,
      -1234208298, -1067800729, 597996849,   1811405609,  482394479,
      -2018918020, 1972254321,  910471623,   1476039049,  -204812449,
      -206447772,  1904301539,  14697494,    -1663561588, -1261633865,
      -1850825269, -24756671,   -554192507,  -1091276215, -2124443532,
      353488715,   1719757775,  1797923440,  -1204301820, 1855983547,
      2136593349,  1078542146,  -98284310,   2144433740,  -584487400,
      66760176,    703352297,   -1793826986, 1177132266,  -347413430,
      -1103964462, 399394531,   2025513279,  1960667447,  -390756871,
      -814575780,  -763005389,  -146370008,  1046904445,  -1356358108,
      1423826126,  1299532598,  -543808808,  196272479,   606471677,
      -720860532,  1209439336,  1820796860,  -332691105,  -208075213,
      -1322382570, -1503061702, 793077911,   -782702963,  1747957102,
      -873231871,  -152213171,  408126435,   2069741818,  -819616937,
      -438420224,  -1647396378, 1851210420,  -449156129,  -794965522,
      981379967,   -1337973517, 1053232945,  -407327866,  526423756,
      -1305294978, -73661010,   1005899393,  -1876160167, -303961062,
      532964577,   -757207848,  -1212838188, -654535806,  -372818498,
      1984515422,  2034878609,  -267535530,  -978549893,  1475181311,
      347627261,   -72084069,   -572336443,  206465130,   2132646571,
      -871429938,  1421825793,  -342787475,  2140594480,  -1491550671,
      -1848477545, -189293954,  -1669775850, 2058018283,  -2101004770,
      2023285635,  -90265004,   919784859,   520459786,   1901032840,
      -1758073818, 1847321274,  127556372,   2039070198,  1230010770,
      1135085963,  -1532162132, 997912076,   1022896027,  -1674855182,
      -243141775,  1171092015,  115960856,   649147299,   543219608,
      -279801145,  1698407499,  2040226559,  1974269417,  1186409578,
      -362204894,  2032062773,  1869609481,  -2119122965, 106614410,
      -1722686864, 2056097840,  -2047547794, 247009746,   -273135030,
      1632072302,  -512227907,  2014809362,  -1338997015, 1704237701,
      -1130914782, 1568447769,  20819538,    -2084849674, -286456829,
      885935361,   1922482420,  1124711659,  1722624721,  836948525,
      -1372872417, -1648703681, -154922737,  208746465,   -1834614021,
      -402576904,  -261358233,  -1445028197, -1154055421, -2017079293,
      1423300919,  -704226943,  -902008302,  1549100146,  1399647405,
      294785542,   312683094,   79486949,    -295180346,  -1997972241,
      1484749515,  1052507582,  20871674,    1093241711,  586721079,
      -1860026727, -96557163,   2098062292,  341794938,   136291901,
      1491838826,  -1252094163, 595064547,   -730876961,  -796764015,
      502184004,   160914341,   1041953500,  -1699419873, -1899967568,
      826468875,   1608787541,  -476848605,  573858659,   791246022,
      1006362381,  1781507646,  578383537,   -134460945,  -1709517728,
      -190546801,  -2054141382, 1927659238,  -50441114,   1315168030,
      -523637028,  544770467,   -1596512502, 1351456558,  1357077411,
      349995855,   -1800088366, 1973501941,  -376936124,  1143380198,
      -673759351,  -1547951414, 1041861941,  1751175171,  321206359,
      -858591610,  801695572,   934052623,   -167860821,  99234755,
      -1753642149, -488420401,  -1547907319, 1560438789,  1384607814,
      363778017,   426170566,   382974725,   1130588087,  -1075958602,
      237923160,   -704961197,  -1559438859, -984188425,  -254960832,
      1560346368,  863055996,   1298732047,  -1753739469, -1062086736,
      -321165637,  -860659334,  -887888935,  1692263905,  -1944464760,
      1309755808,  1664516593,  1116564154,  1391859275,  -2085654925,
      -1048064016, 241971813,   1334066131,  1043606424,  -1020484783,
      436716941,   667270407,   -1735133373, 1910478014,  -534874049,
      -1176883348, 1221729449,  -1803125587, 1565596902,  2102706805,
      567535857,   -2118009653, -521077701,  975194756,   -140038233,
      325367165,   550543258,   778817548,   773154402,   212460306,
      -1415118402, -1927542429, 1696131712,  1767827275,  -111582362,
      292718506,   -1547812498, -1499764839, 1555017577,  294587761,
      385124001,   1390247754,  -728393112,  2064207420,  1237247264,
      350014652,   1484981689,  -823970169,  1413371625,  -1553493112,
      -1075433497, -2059191944, 1434321548,  -1547713280, 835483623,
      879086219,   -1305159619, -131124205,  161926571,   -969014220,
      -1272576760, -1474948974, 1388256500,  -2022196621, 1014703855,
      -1486106911, 579036172,   1980903206,  -771846208,  1968324310,
      -1903841344, 1819978728,  -1079732692, 565710635,   -647887126,
      -1361385004, -1824265658, -1230280653, -1115610834, -1141001166,
      505231575,   1384372107,  1923576421,  928306731,   317749029,
      -1611399907, 1053662195,  -2144160669, -2005977557, -1108683986,
      1484776164,  156940497,   -1701426324, 1635304832,  605316766,
      576788462,   566748384,   102027797,   1876682414,  -768195962,
      -2137375113, -1598119353, 204189355,   -1529415220, -1528787413,
      -1179432889, -835070614,  509981129,   1021981352,  -708083391,
      -1264399416, -2051467711, -759324725,  -1584628166, -24532003,
      -343112578,  519956017,   -407962511,  -664819627,  286789927,
      -922347109,  1765009474,  1953035408,  -776024430,  -2119568352,
      -439618828,  1938943158,  -1728705127, -929866247,  -518861484,
      -1766497415, -1983366570, 15612672,    1163589109,  1952701044,
      1658741567,  -1479866303, 41472957,    646789411,   -1727639122,
      -304926139,  -941408151,  660554359,   794584252,   -669425158,
      551420196,   997172957,   -1724836647, 140144366,   1829071622,
      -1323142765, 2064785419,  -1113964881, -1253726357, 982610377,
      474235599,   1035545452,  1446478868,  1953142225,  1472839376,
      1071640547,  -791607264,  1693323827,  -1348430236, -351079814,
      -888078942,  -570338910,  26421464,    1747802094,  470174916,
      -333288522,  -910347368,  1966661790,  664535857,   -52348423,
      2014202022,  1670017111,  1933623887,  1670504565,  1566101633,
      -1550899859, -891336364,  -1732817757, -1768497726, -605856536,
      514206894,   1022412182,  -962773383,  -365333555,  -1182180909,
      1485982561,  1037642785,  -692434694,  -1908513534, -1107537150,
      -48240990,   1576143178,  -550282149,  1885400513,  -1781224979,
      556889263,   -999037700,  1025112882,  733566880,   133501465,
      1750821119,  -1169136863, 1475653090,  1838838648,  -688301490,
      -1357899677, 313996970,   797355358,   477773154,   1851791787,
      -569921228,  -848473704,  -1341194563, -107237646,  -1474275571,
      -189603167,  313084916,   -1868605837, -1135185737, 421542277,
      1856139659,  -816081246,  -1726539231, -449440716,  -1353137300,
      1268747474,  644944336,   161465421,   -1666177703, -1630549643,
      1939333029,  -1787416570, -1062614800, 910260900,   -499860775,
      1623032717,  1911436956,  -2072174815, 1855478156,  -1131525755,
      -2130475481, -592874822,  -665540680,  158667282,   -1184111002,
      -824535484,  -1876934495, 226329889,   164523821,   1581783269,
      780244812,   967235198,   778954940,   332957990,   -86103694,
      2051898092,  -960611974,  -566940880,  688163436,   1554325475,
      1115548999,  1884106076,  -618792348,  -1354756867, 596208148,
      98211832,    1784290943,  -1135955105, -741185679,  1137413633,
      -565593848,  1279136534,  1000178874,  1626778570,  -1056820425,
      -803116597,  -1028195403, 1323146405,  -2061032718, 922177098,
      790123941,   479807931,   -1850068662, -1918724173, 28949193,
      880271067,   -1475430027, 68464587,    846777310,   -1103911939,
      -1176553272, -1897633650, 504479106,   885350541,   -1835890572,
      -1625210860, -522624251,  -1839369957, -117247781,  1534380190,
      1263141443,  -586617443,  1543409058,  -2050432573, 1336700428,
      -835625630,  -190543031,  938415011,   1946737072,  457456147,
      -4862069,    -871472741,  1828966757,  1663150978,  1964032432,
      -746130880,  1287013257,  -1838223582, -701177150,  -1335398538,
      2025552398,  279138743,   588471153,   -1178083278, -1532215911,
      -1724653479, -1587181783, -915169206,  745928116,   1778532537,
      469970664,   -1770912137, 1612482200,  -207498997,  67389101,
      -1037330964, 938059208,   1293133500,  -1177164585, -943265783,
      661243467,   1001995457,  -1351091391, -545233734,  -1385973899,
      931367276,   -1761190563, 1798299643,  1049204425,  -67055593,
      573266570,   -1798237505, 157498286,   -2041954524, 924922362,
      -499608709,  -363004406,  -228729770,  -1316639394, 56982143,
      -1309390995, 1199726894,  -608411996,  -893132147,  -35711036,
      1504171657,  -661263515,  131655233,   -1248537103, -1919290051,
      -860245087,  1058781324,  407607861,   -266828163,  -833312433,
      344992463,   -751096207,  -1287424582, 469734191,   1577573428,
      1870723988,  1539763223,  751535061,   2036617100,  -1917479290,
      1220652326,  -2099056532, 863157690,   -794470832,  -457137913,
      -321353893,  -1367233006, 710237349,   -474253784,  -443529084,
      1351058625,  140428613,   2041021335,  2092387287,  -1497876331,
      1554258343,  130747573,   -1756890787, 872749054,   -963185821,
      -473180091,  -227266462,  -1575290309, -1184647,    950224821,
      449739082,   1058166499,  -952336759,  -1394441963, 170602170,
      -837306609,  1849874908,  1594044043,  2141554190,  1460440660,
      -748632537,  2081939335,  1290547267,  1362411208,  1363874511,
      1886154666,  -1538512071, 1978765289,  -2020388878, -1720077271,
      842070675,   1744009107,  1608047620,  1395373448,  -506828583,
      1706398011,  -1756278672, 2045412789,  811548132,   -783702929,
      1080688565,  2006823296,  -1444770451, -2130536553, 897991003,
      763562545,   -921058483,  -1183863211, -404565314,  -1780009425,
      -2004288795, -175374831,  1863100459,  -154244310,  2127836310,
      -1254226101, -1655039803, -1507789435, -1749872435, 2049720527,
      507574298,   -325786810,  -358295928,  -1588076812, -819129509,
      -1519459974, 587266187,   -1318691077, -750258029,  -370142195,
      149753817,   -886999293,  1643979240,  -1741268265, 2124101286,
      -2136495413, 1325252273,  -298880000,  753719767,   1879874742,
      -30328236,   418166881,   -2070311026, 87370692,    42876483,
      -225962622,  -1088834830, -591809361,  -65919099,   506363034,
      -303332344,  1509434654,  359055038,   1301549790,  701189053,
      1448606181,  723737401,   55982052,    680072085,   -571442925,
      1441083909,  721227188,   -968409712,  -309100808,  154083236,
      -1715534032, 657952665,   -1292418649, 350924381,   512803384,
      -1841868628, -840817530,  -1793786731, 1208436966,  267545080,
      -2077226706, 663876387,   1477049341,  1316531287,  -258179482,
      -895699856,  119468253,   1966474472,  -1846013934, 1450369165,
      -631005065,  -1419940947, -1419668328, 1485877843,  124041648,
      1320643872,  -569404618,  -1279182745, 660610247,   -674357403,
      -543141129,  -1572960354, -621752000,  1619139132,  51245632,
      602175581,   961455539,   1130974632,  1797899913,  1902857706,
      895935268,   245791109,   -1216989624, 185812709,   1680352703,
      -871779027,  -1026565849, -1991113060, 1722435770,  2074986788,
      1311499076,  1857940675,  575039088,   -674063114,  210713256,
      -1079890869, -609733618};

  int32_t over_bytes =
      compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);
  bnn_b32_t *K =
      (bnn_b32_t *)malloc(K_ref_size * sizeof(bnn_b32_t) + over_bytes);

#define X_ref_size (x_height * x_width * chan_words_in)
  const bnn_b32_t X_ref[X_ref_size + X_REF_OVERREAD_WORDS] = {
      334618952,   990892152,   -1092432719, 1188540535,  -1814923493,
      -1584797214, 408949552,   -699399485,  729873736,   -1664688196,
      2020303517,  1480057315,  -1652919757, 1348543039,  -9257380,
      -625415367,  1504281911,  -1943268141, 762630346,   1801682668,
      -453827149,  -1077900352, -1338399678, 2026636893,  1566309361,
      83186255,    -1761760276, -902980902,  109971881,   -1597388508,
      949583073,   936073207,   -1149722608, 1995813516,  -1106926095,
      -1700630222, 307624189,   -984983573,  -1770058237, -123772056,
      629549217,   1680182775,  -1318189275, 1912006624,  -735535552,
      -1972655310, 570854093,   -519210994,  -768325412,  -1892963369,
      209845024,   -888631061,  261423710,   1470658005,  1500718866,
      1567962070,  -621089056,  -901344866,  -1247257503, 418865405,
      905399499,   1832782945,  1468192205,  232764912,   798153438,
      -1607452505, 383400894,   299513956,   -358998783,  6218975,
      848953286,   -865517656,  -721492261,  -1609940579, -627331462,
      414573625,   1523386455,  -1535693987, 1725220760,  -2019478572,
      702361642,   1445881449,  -1348603564, 417312089,   -2134784428,
      1367334885,  -41720305,   466520571,   -1169712475, -705350854,
      -959479061,  -1834530915, 535398491,   -1817446401, -915533814,
      -459311787,  1884048871,  1662044830,  2088557221,  -502505992,
      -604393441,  1355235602,  451593141,   -422441769,  -1665793030,
      1715902700,  -1141398516, 1811165566,  -656462281,  -2053222356,
      -2077905342, -1467759755, 591974600,   590004126,   2085205663,
      -1978540661, 35015452,    -745754009,  1733958894,  539903628,
      -1256659915, 598594948,   473072260,   1374346415,  -1193582215,
      -359977027,  440082987,   1897731123,  -1914082250, 1500723546,
      -1819102966, 7258570,     -1044946551, 1693140649,  1206503204,
      -1426859594, 100090249,   -2043329067, 2030156969,  -192611352,
      -1432232008, 2096255083,  238828067,   1211978160,  -413885129,
      -683216726,  1086603235,  1009187808,  1875708538,  -157314879,
      -1448969836, -1505514391, 1817605606,  -684937330,  -1375074680,
      329785765,   -1731044947, 130975995,   -222087819,  437991046,
      1864118428,  1768440710,  -748848591,  -797756480,  -741574502,
      2056346595,  1329304878,  -1753751497, -1128735815, 159043197,
      2036548120,  445129530,   1408500530,  85135260,    -1674879919,
      -1518961516, -1095537294, 172373817,   1992999288,  -190451320,
      356133226,   -192729208,  -726950211,  -1164438744, -2081041538,
      115928535,   -768735196,  640095102,   -792568569,  1490893552,
      325753205,   -475702741,  1503442907,  1377580734,  -1564359272,
      -962346289,  108423974,   -2135653630, -1771460954, 1868374403,
      1852829859,  1488371717,  1707085559,  -578126640,  95696484,
      -936342501,  251030768,   1682758375,  -969823889,  -921723273,
      1876723735,  -223265192,  -628164027,  -1665313946, -956471335,
      -431068610,  -1686696786, -721288314,  54360124,    198874299,
      677952055,   -1125684071, -692879097,  1512092842,  32389719,
      267958429,   -1917798911, -1742347136, 1542219498,  -1989444232,
      -649291593,  -669038567,  -1907231190, -981145254,  -1534116787,
      -1710792494, -46820031,   1749129660,  285697641,   550381289,
      800321840,   -2048266501, 834573811,   -869177549,  1839701214,
      -1093517492, 1164159543,  521174769,   -788112871,  904347720,
      -1277494227, -390650363,  1430735305,  -1656983550, -1217620559,
      870043301,   1631480001,  2039054338,  -558735759,  -1654461956,
      -679330778,  893134654,   1154550863,  238633435,   176316914,
      -1850309593, -2039416623, 761322514,   502883938,   1551268968,
      357694508,   -866840931,  179663904,   -1508587816, -1468030419,
      99302147,    658753277,   -17352193,   -382074998,  -1127285511,
      -344300017,  -1029045260, -1008186234, -1170172282, 1194869681,
      -232986836,  103189083,   -482599036,  -972221107,  1440983785,
      35070582,    -1167323232, -1607699148, 1138121690,  1749819123,
      -474606223,  -1514491125, -293070607,  -1139860310, -1101775152,
      235591952,   404394701,   -1716605831, -863649197,  1925218426,
      1385168179,  -1712463439, -625665641,  242287398,   -1428595177,
      -1224344993, 183208614,   667127401,   -899201863,  1413490145,
      1271939635,  -1454260871, 1710308327,  1015836254,  1535289595,
      -1014300504, -160795843,  1753377965,  1603588180,  1656367083,
      831743063,   724192728,   86604350,    1349631021,  327097040,
      -1350581092, 425621532,   -1108049828, 1635002256,  1450435201,
      1667513178,  -1130890274, -2143348819, -903217164,  1633967793,
      -1700997501, 1618509033,  -2062608691, 1300074999,  -1568200659,
      1898850166,  -16393428,   1637665408,  -715288951,  -1583358056,
      -1190114003, 931491804,   -1201460518, -1691835876, -770233908,
      1473930813,  -1880488608, -680889542,  283136356,   310267102,
      -479921026,  -1924066032, 703660525,   -512981488,  1613413367,
      -2071365416, -311744811,  853645461,   758193213,   628907571,
      1475998831,  -2133835758, 321510813,   1149134034,  1284238388,
      1187476200,  1243032063,  25434447,    143469701,   -1480771113,
      1738944607,  -766137016,  145802670,   1883225105,  283273272,
      473324308,   152212852,   586643368,   644749549,   1768704670,
      1637557762,  -1153159032, -1073753713, -173023836,  -46957095,
      1034491832,  -2062528541, -1322569588, 1480687551,  -6668776,
      -1019742691, 1860978161,  -1912246424, 1034915409,  1006432861,
      -810644825,  757064299,   2084451311,  -655326028,  -588778997,
      1726858283,  848272199,   -307219651,  1035032319,  330479509,
      -784999158,  831474902,   483058293,   -295490447,  -58961909,
      93922996,    295256946,   -2003489136, 1662208452,  -1671813823,
      -118598998,  -407546809,  249004240,   1473373904,  -1598362074,
      -1257259912, 1511854890,  -1041712859, -146314573,  928132296,
      1501292629,  -2044763087, -253233377,  835616194,   -103916411,
      1758351217,  930497334,   -540310951,  -993170707,  -1622855609,
      -1332047319, 1542974113,  -1189909578, -198804513,  -430274968,
      438216919,   2007361177,  1166467128,  -1787798031, -1643357049,
      1331557600,  -1847846079, 1303821377,  -2127251278, 1034515709,
      1964022829,  -2121029334, -902406526,  -1286097053, -697353822,
      1318523485,  542269881,   -283439786,  -1891852076, -1935027647,
      596070718,   -1505191151, -770506689,  1550954878,  13162528,
      1236829643,  -1557372582, 1743547680,  269775235,   -1603511743,
      -1080311659, 1626623870,  -815939227,  -177390342,  -879434078,
      -70734193,   545827914,   654604588,   -397928622,  174522980,
      -1557831623, 1555382192,  -1360542489, 1723918711,  662728290,
      1827473532,  -1909160716, 177146754,   817990501,   1265171009,
      -1426396594, -1037823233, 1160202007,  -281806878,  15842961,
      171204785,   519727382,   -2106458416, 1963175894,  1140532127,
      689718896,   796382763,   -466901686,  1453733380,  1459609799,
      1873728287,  1239855417,  -1768885321, -1277283127, -353147259,
      1081497585,  1840692025,  1896251377,  681627869,   1770889047,
      -1799530879, 1924150436,  -609749747,  -96154205,   -810158379,
      1178803523,  -1915061984, -86324829,   146863951,   -780702378,
      -1294152150, 1224388426,  -1114661052, -2052968099, 340954725,
      670835696,   1338107283,  -1836123491, 1781624955,  -376258364,
      803360497,   -142647958,  -215040139,  373006339,   116094277,
      -2023650905, -1058111295, 1968998569,  1531365378,  -1881423408,
      665395157,   -1827709322, 1412298172,  -1578067222, 1580590296,
      -929383563,  1376947741,  -823243675,  -335822224,  -1589576927,
      -1276558627, -1711269981, 786224958,   -828910213,  100078456,
      629999129,   1111032071,  -1726956374, 639666286,   -1146946884,
      282590783,   -1357630191, -2099258596, -1711733191, 757999190,
      264796904,   -1214326732, -1252628052, -1528863805, -783046880,
      -58546208,   -1487065820, 1772989504,  -635009410,  746459774,
      -1315363464, -710133715,  -410363520,  650255922,   1151773197,
      1980577753,  455043436,   -1726843054, 404446410,   1800690667,
      1646252011,  578555443,   1158896878,  640663680,   129268829,
      -119299345,  -1678738061, 80283867,    167043197,   -2039278119,
      1789902282,  -1739856860, 790369492,   1348522610,  317660016,
      -931824778,  -1861958601, 1053833874,  1146767164,  -1631359782,
      -1361678907, -574546690,  -7437465,    -1880137207, -802214419,
      -674143360,  918315741,   -1980822480, -1625543931, -229252535,
      1789287302,  1126756399,  -363677381,  1473165027,  -1890280964,
      -895340275,  2056101458,  1482101507,  -1096266011, 1590338875,
      514478535,   -779436299,  318498593,   -140299459,  -936765221,
      -1095814047, 1646739916,  -887720152,  -217347709,  804272029,
      952408228,   -95204425,   -1373085191, -1718160816, 1971220053,
      1483591357,  1530738623,  -1037218125, 1434742834,  -1869563139,
      1378822553,  -493168865,  8001934,     -688655295,  -1920301420,
      31139480,    -1964969136, 489572710,   2136001869,  -1392295066,
      -1454206737, 2078732678,  -1579699382, 1391561372,  255974639,
      -1328470719, -188679552,  629474101,   -1450135748, -1925249738,
      -1745058683, 289616364,   -1992314577, 965925922,   -1685448615,
      431609054,   -532761640,  -279323476,  988076288,   2012797068,
      1708833387,  -484057763,  -447130978,  -1829829766, 1356826079,
      -2012254352, -1332872636, -110708680,  -2044397172, 177446570,
      497365355,   1887612484,  -1763362051, -1857377604, 1701482574,
      -2073940592, -824806860,  -1669689579, -347661061,  760998261,
      678631226,   -894367667,  1869683141,  -149327580,  -1877903400,
      25775682,    -794832886,  -1530949618, 1213695763,  1278732557,
      -448486481,  1675283292,  2051453324,  1643938067,  835660384,
      -1480022453, -640205751,  -548734590,  603263370,   1204512091,
      -1996735436, -1062008945, -1099027210, 1131761539,  -898020918,
      1414487205,  -1564001110, -2056497007, -1863367517, 351010316,
      -2130580341, -376974436,  1838584116,  -202414395,  -1550829782,
      -1828876688, 509873382,   -1666438342, 1449756595,  -609370887,
      1083910198,  -1696164703, 286213706,   1659072204,  1119360733,
      1141534212,  275031281,   -1949719450, -492769420,  -560903220,
      1399658371,  -371181132,  -784168752,  -1918444998, 228543543,
      -247171466,  -1439203975, 1534012206};
#undef X_ref_size

  const float post_activation_multiplier_original[chans_out] = {
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583};
  const float post_activation_bias_original[chans_out] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  float output_scale = 0.003921568859368563;
  float output_zero_point = -128;
  float backtransform_add = receptive_volume;

  float post_activation_multiplier[chans_out];
  float post_activation_bias[chans_out];
  for (int j = 0; j < chans_out; j++) {
    const float post_mul = post_activation_multiplier_original[j];
    const float post_bias = post_activation_bias_original[j];
    post_activation_multiplier[j] = -1 * post_mul / output_scale;
    post_activation_bias[j] =
        (post_bias + backtransform_add * post_mul) / output_scale +
        output_zero_point;
  }
  int *chan_overlaps = (int *)malloc(sizeof(int) * (chans_out));
  int16_t *post_activation_multiplier_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
  int16_t *post_activation_bias_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int16_t *quantised_accu_modifier =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int8_t *Y = (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);
  int8_t *Y_ref =
      (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);

  assert(X_ref);
  assert(Y);
  assert(Y_ref);
  assert(post_activation_multiplier_q);
  assert(post_activation_bias_q);
  assert(quantised_accu_modifier);
  assert(K);
  assert(K_ref);

  assert(post_activation_multiplier);
  assert(post_activation_bias);
  assert(chan_overlaps);

  int32_t larq_clamp_min = 0;
  int32_t larq_clamp_max = receptive_volume;

  run_int8_config((int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref,
                  (bnn_b32_t *)K, (bnn_b32_t *)K_ref,
                  (float *)post_activation_multiplier,
                  (float *)post_activation_bias, post_activation_multiplier_q,
                  post_activation_bias_q, quantised_accu_modifier,

                  (int *)chan_overlaps, x_height, x_width, k_height, k_width,
                  chans_in, chans_out, h_stride, v_stride, larq_clamp_min,
                  larq_clamp_max, 0, chans_out, valid_impl);

  free(Y);
  free(Y_ref);
  free(post_activation_multiplier_q);
  free(post_activation_bias_q);
  free(quantised_accu_modifier);
  free(K);
  free(chan_overlaps);
}

#undef h_stride
#undef v_stride
#undef k_height
#undef k_width
#undef x_height
#undef x_width
#undef y_height
#undef y_width
#undef chans_in
#undef chans_out
#undef chan_words_in
#undef K_ref_size

void impl_bconv2d_int8_directed4(void (*valid_impl)()) {
#define h_stride 2
#define v_stride 1
#define k_height 6
#define k_width 4
#define x_height 12
#define x_width 11
#define chans_in 512
#define chans_out 16

  const unsigned receptive_volume = k_height * k_width * chans_in;
#define chan_words_in (chans_in / 32)

#define y_height 7
#define y_width 4

#define K_ref_size (chans_out * k_height * k_width * chan_words_in)
  const bnn_b32_t K_ref[K_ref_size] = {
      -1667941237, 1448713456,  -1552317250, -2046627932, 683863560,
      -238015391,  1889621163,  -1146744408, -510686028,  -1117492961,
      -1359528875, 1575638974,  -722571407,  -1132221422, -1794628694,
      -351730166,  2073851381,  -299825408,  -339197266,  86240880,
      1065204944,  -884812668,  1411160468,  -881556368,  -268921696,
      -1516645369, 1797568645,  917093437,   52623962,    930837777,
      374553873,   -1540581651, 885788055,   821476919,   -317846905,
      -1493396616, 1825118029,  -384905893,  1797368966,  1986906752,
      -501859916,  1449020872,  -1542883958, 1153968776,  156102132,
      1660464794,  -411684486,  -1017545459, 456771578,   -1604237591,
      -2110008794, -1263008829, -658925958,  -1043093987, 792415369,
      1232129818,  1050873902,  -1382741130, -974796704,  18050398,
      -1509459442, -1842985195, 1060688046,  2023663406,  1059660126,
      -151897591,  724181254,   411009610,   -2022854745, -1848363909,
      -1873328038, -1840368590, 358295689,   1915862718,  639105619,
      -2105444226, -883520861,  465945796,   759834270,   -419334040,
      -590126700,  137890549,   -721838176,  1032986018,  -1542711232,
      -102718783,  846493625,   -274066254,  1691187192,  1370948786,
      902352054,   1927602036,  566937337,   -210278301,  469894430,
      587513464,   582317797,   884040362,   -95852467,   617280905,
      -2087303928, -1880489655, 1281494289,  1711827513,  1901089844,
      -1272489895, 1844998611,  1638795977,  1417040717,  -1791913728,
      891291125,   -1726947273, -1177398108, 795568422,   1650474124,
      223671067,   -610707905,  259844201,   1525448706,  183457105,
      95437854,    -892550277,  804086976,   1029553192,  4329603,
      -743987387,  -1231163561, 1155699928,  -173577849,  1429335139,
      -430586326,  -322775347,  -1192876494, -2081842400, -711895613,
      1685507085,  -6460901,    1877650004,  -145157623,  2141632549,
      569330137,   502636811,   -1795800820, 481217137,   2127914031,
      1146310783,  -2027663775, -1707834214, -2091038253, -1387681643,
      -1750381361, 1044305327,  -797569616,  -914887025,  955953949,
      1905861531,  2140072119,  -1420465164, -1585426452, -876392241,
      822927436,   377439080,   -593159933,  -1431151484, -928117242,
      1245120517,  -2053658674, 608655455,   -1213929022, -1949002028,
      387622597,   1931737297,  -169700243,  -306218413,  49588901,
      -1358163642, -1325646177, -579376197,  -1896922477, -169573911,
      817332771,   16626886,    1476502317,  -2125605532, -1394967760,
      -2140940661, -1674083635, -324141352,  1182178370,  -52737195,
      662563825,   413760921,   1591786726,  776966745,   482876981,
      -2109812241, -1063999825, 674091303,   109935080,   1459880009,
      1998638033,  1130061639,  -14576709,   252868079,   670839397,
      -484306462,  -883623697,  -1327856253, -256671581,  -1855927984,
      -1008444318, 154450187,   -1386782663, -522264823,  -2029170546,
      1513965769,  -157882411,  -1013700355, -1327701243, 1406572563,
      -944044957,  -2097932544, -1612950910, 1461038717,  316520129,
      -1780577599, 591255253,   933607692,   1502296159,  -761541844,
      500822164,   1816140042,  -1720464521, -489403739,  1555064787,
      -475642121,  -29728065,   -2086859638, 1836823425,  2130542840,
      1998327023,  679059354,   1649375569,  727779564,   503714236,
      585066382,   1368321202,  -494819920,  1420183569,  -1514101318,
      2039944711,  -1786011339, 615820487,   -2083776586, -2091048456,
      -1938177325, 1470944511,  -873645512,  -423729381,  1180739949,
      1300901646,  140434334,   -872547405,  1500426075,  451570533,
      1764802631,  1289737382,  1533809021,  -1658735276, 1402863630,
      683063664,   -972922969,  1497222113,  499641031,   -248573185,
      824088639,   -1150125760, 2004661329,  -1829783032, -2066964399,
      -1485250523, -1488806023, -1548302686, -1126825239, 2127178909,
      -1463920813, -1030381224, -982645831,  1637231689,  -1534873359,
      1612936975,  108920040,   -760138128,  1547795526,  -1803184456,
      -899171590,  399904675,   -250482358,  -1893810089, 1259500150,
      998796048,   1450256876,  -1991548850, 1645388835,  1081437690,
      2127911814,  294154779,   -260819812,  -964340587,  1643739595,
      2019834342,  -1430556640, -528639968,  1592816485,  1486757994,
      -1748533725, -1053376374, -872054212,  114380079,   -995324573,
      -707061708,  758495185,   1677700138,  -933654668,  1980715030,
      365154401,   930435253,   907570905,   -1576803149, 1402175784,
      -1721621728, -1002962641, -783954320,  -558491858,  -1783413660,
      -660684539,  714202277,   118003371,   -1939260637, 1494726559,
      2147471512,  -2003291128, 1045393215,  -1402244308, -1128076090,
      2142725242,  45903496,    859370229,   -582210589,  -284521158,
      1562012070,  -1947268217, 977409567,   -284383738,  -380306061,
      422310954,   -2034679227, -1884437178, -1100651865, 2073827760,
      126416181,   -1157411361, 1624770778,  -1401026250, 305381170,
      2038094346,  1805023105,  2126887236,  1875392306,  -494187797,
      975888557,   -866853416,  -402730606,  679212101,   -25828128,
      1132698443,  -1197569942, -45670508,   1488524941,  914173688,
      552583979,   -1128165022, 1272056870,  183089233,   1120641604,
      -1250437226, -277467601,  -978365725,  -1522250878, 308013372,
      -331097853,  -2125863291, -1725117690, -1445525973, 1154367277,
      601322983,   -1424108384, -217565623,  -1272143351, 1328064298,
      259806782,   -1864131726, -1147615362, -156159418,  366215932,
      421081801,   -688843398,  1991629255,  -395828508,  365945591,
      172830778,   1313109683,  1607665156,  -121881757,  -1936327556,
      2046212711,  -252030892,  -2083031693, -2069977742, 1254663107,
      -1522576285, -1198133754, 1941662633,  -1450082101, 1012051825,
      -44246503,   -1681476908, -1559671783, -494973537,  872440938,
      1252093354,  1486587199,  -694062764,  -65822295,   956649669,
      112844649,   -1107121794, -182683395,  -486759621,  423257375,
      214053375,   -1818516361, -1342585436, -1971519822, -1680569684,
      -1594261625, -1718840234, 1139600479,  1067053432,  -927037260,
      -1202370708, 64089229,    -729507829,  -591751865,  -1820402212,
      530214911,   -1448178731, -1134968599, -1834534102, 1624188908,
      78290799,    -158280725,  -1058373056, 1418447327,  1793911321,
      -323674568,  -1083867082, -2130091880, -72951185,   619439057,
      156145908,   -1888752974, -133135106,  -1346128821, -1887110818,
      -1106995698, 1374023680,  -602267334,  -1328548693, 1047343182,
      1129249344,  -338771866,  60155234,    1530359181,  -1474981956,
      -1244038156, 557995479,   -2098494726, 1506448140,  1135252417,
      1347688253,  -1075347713, 2093303952,  -1294572246, -1688281906,
      -1245365224, 381812964,   -987019166,  572352865,   -948347794,
      -1082135946, 1942644462,  1297285610,  -552887825,  707977116,
      56495731,    1212726581,  746522704,   -830522772,  438895309,
      690006296,   -482248230,  -621580341,  1648756184,  601795596,
      1283685904,  -1670508397, -1646038574, -1461316697, -341621714,
      25120277,    -1144223900, 1431570440,  1919526210,  316350954,
      -1522726584, 2140590224,  -642616892,  1277434433,  293337608,
      -1785192669, -1327053570, -931543035,  1852724331,  -2064972234,
      1443330176,  754939349,   -1171480285, 557417725,   1856001029,
      587588690,   -386267152,  -1183869396, -664106585,  192598167,
      -730427595,  1482463361,  785588090,   386853927,   573378412,
      -1610318369, 1087719694,  1281232895,  398140326,   894841511,
      1495105169,  1845792976,  -2022330152, -1327152198, 1855858312,
      -911631985,  1442394372,  1766104157,  -1101962024, 30398597,
      -974122142,  193666332,   1984271243,  1582959064,  999081941,
      2034505306,  -1075953166, 811207728,   1086040063,  -1646115710,
      1663279773,  -379236039,  93630459,    -288557108,  -668082017,
      773490811,   1868973757,  -2051147269, 110405744,   -482738489,
      884719611,   -1968178758, -460072533,  1945116602,  -1800611546,
      -280379323,  -862355917,  -1435180831, 802267530,   1249641675,
      -284619817,  -409895841,  309286418,   -1317086761, 939855898,
      478395003,   762094802,   1676675919,  -1652208087, 572691400,
      -825917769,  2133632408,  -850371217,  -442794519,  632216985,
      -1339450673, 244487882,   -330039526,  2006111393,  -1395602947,
      -530813512,  -1901720924, -1015679548, 1176074529,  1732529531,
      97718278,    563122108,   657564042,   -37309998,   253071811,
      296111152,   2078138048,  -264526088,  -1750498982, -1125452110,
      -1722492532, -1639310536, 51067744,    -1308773440, -119902967,
      -346309745,  2132537179,  482162235,   588814228,   -1000761119,
      551929091,   -526368335,  1580161654,  -1893260659, 672805404,
      -69202412,   838868935,   296618219,   500423408,   354368158,
      1497263981,  861408025,   -796855528,  -114996343,  -771883438,
      -430140874,  1360321862,  -1228206651, -1317217943, 2132855379,
      1831476369,  726126926,   -1545089243, 1083641668,  -1387888082,
      1152816642,  1078783737,  2123485616,  381103339,   736320298,
      -1974402614, -146586640,  1236079477,  -332737905,  1273165818,
      -858072393,  1470944670,  2137784136,  925601662,   -1152874158,
      -954086864,  -2146600750, -1839947741, -1165296086, -1722769274,
      -986099231,  -2114771439, -76981231,   -601846225,  -932629096,
      -1926075359, -2099398124, 1697310242,  2017270341,  1067074858,
      1174786731,  1394843221,  1599685971,  -92721151,   -898580521,
      1834529434,  50108647,    -1053583838, -1776071398, -2057035860,
      113783831,   -349462971,  -444549623,  1033154991,  -947292785,
      -517090904,  -188787670,  1167842739,  1734737095,  447574537,
      -19310867,   -894238545,  1901210655,  -963117529,  183367737,
      -472455448,  1229138489,  -1069552413, -164922047,  -79709752,
      652563311,   -1477634552, 289126063,   1138276452,  -55749592,
      -1704409277, -755971259,  1646130262,  -2124549134, -93980288,
      -481901935,  396858243,   843311920,   -1839936931, 648724872,
      1606209444,  -1815690649, -575212653,  -1369721441, -341932690,
      -157841726,  408197818,   311138956,   -983857787,  1903895589,
      1682235116,  424291796,   -1587478823, 1042433204,  -601910344,
      398748520,   -1732714631, 1182406967,  654062167,   -862046027,
      -1358038126, -1384870279, 2103688158,  -128887597,  1827247901,
      -1847797676, -994292559,  -1420430118, -1919775941, 1029375758,
      -1542469837, -1545955333, 508442521,   1603061910,  -928550448,
      -1650005601, 1899852126,  -729246378,  2097472628,  -1837392796,
      1991189455,  1246733952,  -1880120237, -942651097,  1272972211,
      1815184983,  -1486878038, -1797031021, -661992692,  753546717,
      -1070909327, -922179567,  1891811062,  1996127373,  -642061893,
      1274918072,  516287671,   -7642009,    2094869494,  -224221082,
      467427099,   640756424,   -1821808651, -1476963450, -1565748334,
      209142646,   495167143,   -1656492792, 208092062,   1185781668,
      -1616572742, 188832272,   1089780751,  1139930821,  -10701255,
      295491137,   -1636542696, 1158039203,  -1306157477, 1638533402,
      973465426,   207042203,   -1119626660, 1376691912,  519265050,
      473451469,   -313158308,  1285032804,  1437701882,  -1910236517,
      962234166,   -1938647398, 901946688,   -1469390201, 738569693,
      54805347,    -626348666,  2037225579,  1219427475,  1202312037,
      823874438,   769883314,   -1175364530, -1929888814, 1271304238,
      -240564843,  721906679,   -1682376342, -1670784742, 1825393685,
      1028608323,  1957479462,  364827202,   1261168878,  1855736655,
      1260997092,  -404618655,  -210619555,  -121239519,  1255453061,
      -658931044,  843966312,   -750013997,  -1454410032, 913898360,
      -586278647,  -902686826,  1467946219,  -433677267,  2143407397,
      -1375422519, -887887726,  207890498,   1828589249,  -540533083,
      -1239180549, 982371634,   1645416885,  1968168664,  569957782,
      -1528469289, 726432254,   -1682237637, 1336262576,  -1464451601,
      -1178784556, 1988283690,  887587490,   -511788697,  -1180574589,
      1634858431,  304649925,   -1307527407, -270229599,  -247824168,
      -1996470449, 1230632038,  1305774516,  -1726380577, 586295632,
      1164703539,  370953858,   -193499839,  560549623,   -1677146978,
      458513726,   -1322815084, -871266679,  772741557,   1731234157,
      1727977598,  -923630308,  -383917358,  -695599419,  -1186500697,
      2084094398,  -1580322573, -1457076303, -439188681,  -549930670,
      372467801,   351201582,   -1929984931, 410862191,   -1695762342,
      -485694014,  -1171656167, 1850942494,  -617071938,  813598247,
      2095020307,  531736533,   -1401576683, -352159218,  -29307060,
      1569253484,  -193953926,  -951109721,  -1420342907, 1309800976,
      822714836,   1235315866,  -1514865662, 575739347,   1875527250,
      -2133870533, -30401898,   -1571732410, -542190978,  1673575850,
      795860701,   1013818754,  -1814758930, 1711724355,  -192961575,
      -17028148,   87394740,    990644554,   -929538922,  -128449753,
      -1493618375, 3560,        2083956230,  933755416,   272262334,
      1511501931,  -1407407608, -888396081,  -1781021180, 146223140,
      1755902847,  1682278572,  1806854618,  1024990115,  -413417848,
      922415679,   1472670887,  -697017733,  1873011662,  -419969839,
      -1618564535, 1004487107,  380726232,   -245415881,  141806491,
      -2022431358, 1639377876,  -917928821,  1790793179,  -1991193468,
      1742707283,  342197632,   101129704,   1233278622,  1568839994,
      2130927037,  -1363588058, 1779845255,  124174071,   -764494922,
      -623623820,  948274038,   -1557407416, -590120451,  1426728461,
      1431963052,  976062592,   -800863431,  681919393,   -893471529,
      1488535074,  -833270833,  -533966456,  774819324,   -283643108,
      -450915052,  1655738374,  1088961743,  856341851,   1045571695,
      1726014249,  -443533318,  272297438,   958993172,   -399682432,
      -1959266493, 1489177352,  374108246,   146700168,   908237669,
      1325920606,  -318041845,  -38380603,   970857743,   -1312949036,
      -976431697,  -61424006,   -1156236591, 1875715022,  -1288905696,
      197015659,   -1051543450, -347819070,  1891791836,  26963473,
      -1370979447, 653388439,   2120699462,  1393900952,  761685408,
      288512255,   -1949182962, -917848586,  1777316853,  1884167186,
      596484601,   -1142281592, 1959353574,  -674006950,  -865234761,
      -1925839960, 104441209,   1412898392,  1653375354,  -1474108070,
      -819458155,  2070208765,  -1461669661, 1730701907,  -99327829,
      597607927,   505647299,   -1040881742, 1800009165,  -1846735844,
      -548137062,  -1487117092, 952041176,   -497687630,  1043700360,
      -647217314,  1410995400,  2048137237,  -310140454,  1389409600,
      -915561564,  399234231,   -1353865599, -109008305,  -1602942186,
      80956963,    743558656,   47226164,    -1851635170, 1649010946,
      -2069005557, 839042422,   138681382,   -702420299,  -78302680,
      651085016,   -268003971,  802836124,   -1792922217, 1689887032,
      -692278401,  -2137110061, -161917963,  -2115888394, -1005841036,
      2095647633,  -61217152,   -421291093,  -1740692180, 651967652,
      1460018051,  93713351,    1095611659,  784099378,   -956995734,
      -295036973,  1668589439,  589632330,   2132178176,  -1551125384,
      625610115,   701080811,   -1203823792, 1268356615,  1006732255,
      880784084,   31991232,    -312691301,  -572173338,  972734870,
      -872691005,  970232553,   2133300059,  -1104813897, 1144115456,
      1150459107,  -954230770,  -258836021,  -1136364407, 1514659567,
      847272711,   1607848129,  2048363716,  769560124,   912681553,
      -1867490442, -1060642139, 97032138,    -928052188,  -1795119754,
      -754385992,  74226363,    1797003283,  -598162959,  -1267387633,
      -178069967,  453580556,   1019346714,  -1870844102, -1042754276,
      -2133283482, 1318907030,  -1262390465, -712495502,  1793984743,
      940799802,   -1445819417, 1403743572,  -39146788,   1304435112,
      1672010402,  -44804667,   -37394629,   -1133150657, 163730565,
      -598584981,  607821671,   1174280707,  1870414728,  48352308,
      -1419598460, 661411806,   1190655451,  2029308542,  299110212,
      556831028,   494419878,   143371939,   -1601028993, -1827682943,
      -670723776,  721624800,   610903893,   266735321,   165299804,
      -807040312,  -1755165606, 997071353,   1709197121,  -2009492122,
      -1558970842, 756690845,   1243330212,  1062357276,  -99567439,
      -2090637415, -1591141357, -1947586052, 1107806435,  -69552022,
      1485156293,  -708673998,  1142362738,  697698448,   -1980328157,
      -738455663,  4940971,     1709277485,  198519962,   503808160,
      -676694481,  -640406014,  171271155,   109912585,   -1549118656,
      804722312,   -564656023,  532978088,   -458518597,  -468527038,
      -865747510,  -396860308,  -662587754,  323666568,   929456329,
      -719917634,  -157862534,  914452441,   -1332395001, 1804707437,
      1246683773,  884512164,   1359581081,  -1642181681, -52344612,
      1124246108,  842731577,   -302062451,  -1018994327, -849005080,
      1768701334,  1399530376,  -1234216715, 1463770535,  778518992,
      -545612677,  -168888185,  2117525579,  -593241641,  978627603,
      -1000626087, -1230873189, 1147178204,  -1919515876, -1035159159,
      -1708385215, 1461253211,  430985793,   1768895399,  -781685585,
      933169771,   384447244,   1109655113,  184009518,   96636863,
      1343831110,  1800113532,  -2112394535, 611677538,   -2099875360,
      -1838794872, 621536314,   187206110,   429526967,   546690918,
      -2044339169, -1658679408, -1366062319, -1697427630, 1461731607,
      -336037476,  378553367,   -406019105,  1221064505,  -614807484,
      -1941752817, 1900499704,  -1370392092, -743881137,  2093385389,
      2102810357,  1549013738,  236804642,   -47837786,   -1647099020,
      129620537,   50153037,    1222983259,  1974839169,  -1923330292,
      1644398946,  1763451090,  -205849221,  947025374,   -619401808,
      964315601,   1125792313,  -401272577,  2031546762,  -1664725738,
      -563932605,  -1091322838, -1275372549, 1215517701,  -702026396,
      608223050,   991493186,   1136556955,  -930030060,  -903102298,
      -879512929,  -226006849,  643611475,   -170598945,  -1376175513,
      -1273741931, 352778640,   -1143349133, -2114934506, -2045996516,
      984525319,   923312742,   -515167227,  -610111425,  963603727,
      -1408360158, -603212288,  -1491148002, 1398980197,  -1386510282,
      1879017062,  1810283780,  -732012874,  -415552323,  549405710,
      -1043097950, 367281603,   2045259503,  198404979,   442150558,
      -1640190307, -1207612144, 47833824,    -36711779,   129447734,
      1367372313,  1306826770,  -963688292,  -2059262619, -1346773530,
      -1538014849, -1065846917, -1324671694, -1442469987, 1930087080,
      295166796,   619513997,   871813692,   1878535148,  -80286361,
      -1051268214, -1377974877, 2073878438,  454696678,   258910835,
      -9887288,    -1348778760, 781628286,   -885558858,  -1228924899,
      1372268066,  -1991483555, 385994068,   658666286,   -69549597,
      336551734,   -749730487,  -110831421,  1094042388,  226404618,
      217651324,   -1788202472, 464333260,   -1896028119, -1789231149,
      231195582,   238321655,   565024094,   1359594568,  2010074153,
      301838175,   -746021778,  1912191350,  345125973,   -411777079,
      -1808465311, 363793792,   -422263290,  -660773978,  1934998341,
      -1424942177, 462108248,   -358792840,  -729845985,  1377595737,
      392218977,   -1527172757, 1533452167,  -19194875,   1830078097,
      -492840689,  -431913453,  599555269,   2026353096,  2087430900,
      -1897581768, 1580128912,  -154489412,  1642941438,  -1136112029,
      2035239203,  388060667,   -1088967839, -1627508103, -1190648701,
      7603060,     2108245107,  -2093654228, 2050582780,  -150378688,
      -1858128215, 1005666070,  -1060597970, -412069206,  -2114069005,
      1224865910,  2076606187,  2047221430,  1783692092,  -1438200674,
      130535548,   -2135013090, 1363031910,  -646974682,  -1389022629,
      1591364276,  -2066619095, -105206026,  -1400008559, -816019033,
      -1896458513, 373587637,   1831006352,  -675217849,  -517363707,
      434595678,   -839722765,  -363783252,  -283683791,  -195452597,
      1525306732,  -553407553,  -1395592420, -1692968512, -1825765903,
      2087521907,  635890059,   -615134117,  379652968,   486674708,
      -608582399,  -1029009669, 376995756,   -22178448,   -511678613,
      1317516697,  -36334675,   -750500904,  -1945831748, 1650142071,
      -43820068,   -290012984,  -1822552264, -851177011,  -2027107167,
      -849484146,  62400477,    -2115636576, 997173331,   -1358662604,
      -331673664,  -1705651816, -698644574,  1748618975,  2127326446,
      1370892577,  -1907433194, 61001015,    -656063512,  -1079535694,
      1458106684,  -1445799061, 1667800485,  1230084981,  -1401843237,
      1264469062,  -190224456,  1815131102,  -62328139,   -134882422,
      -1099885644, 368581440,   -1849924868, -1127313993, -243522407,
      1359073845,  625109473,   2100806078,  965854538,   -398710766,
      -1922049838, -934835212,  1558546376,  -1307030999, 1948088732,
      -1247912718, 884515528,   1713928283,  41640322,    -525017315,
      1005343649,  214821236,   1650573662,  1755487379,  1063098071,
      853050880,   1496614893,  1958407620,  -22887353,   1216010322,
      2038043267,  -747299558,  -429667037,  1416361750,  -20044480,
      -1644850589, -1855706259, -260286862,  -650361847,  -525938855,
      -249778461,  -203619691,  532373689,   2008953904,  -1811238593,
      728982684,   1456739281,  1049553156,  -1398736977, 726545991,
      892697664,   -192182380,  2001448839,  -1504229469, -1712477212,
      50778625,    1317430335,  1055336887,  1314690771,  704892935,
      1617737201,  1884491193,  -642875302,  -241562353,  1888384865,
      -1583249035, 1170897559,  247302843,   -1007740878, -1155409240,
      -348153714,  307125256,   856218022,   -1804888027, -475628856,
      169837292,   378332441,   1187207135,  -751179729,  -483186394,
      -462366449,  -717268275,  9951226,     1170220665,  1813018353,
      806158053,   1871023560,  1172794522,  -2074022230, -165187699,
      -1538157007, 1031415846,  1994899992,  220006450,   1816702014,
      2011463476,  -1117696994, 695089829,   324570025,   -1665757126,
      -1181521642, -1575752542, -362889424,  417935540,   1628283252,
      13502526,    989379039,   1038413576,  -904156688,  414271781,
      1038974054,  306289850,   -1478368966, -461355859,  -1821570828,
      -1627246088, 1101954022,  2055583437,  1396406156,  -1528190156,
      -1611678130, 1164274544,  -538083259,  11141739,    37916033,
      744076110,   -687026340,  -356915314,  -886540402,  304803835,
      -454165086,  2122237437,  1819326880,  42587558,    -88049391,
      1360049493,  801382297,   1251815268,  -985555825,  1617156062,
      -1420994494, 488594573,   -1807444728, 530194834,   -1727554555,
      741875364,   -1637536364, 63307854,    202397404,   733960381,
      -498131624,  -1313890894, -1627441347, -1435127798, -1988888410,
      -635450389,  1184688402,  1150857573,  -1237321431, 2099371901,
      -1053035364, 411330134,   61002284,    1747535049,  -1665477613,
      136906983,   368000154,   1191426193,  -834291273,  -1505680544,
      -327852650,  -1911425331, 1598321990,  -1423893253, 129220327,
      -1531092039, 1326385627,  123056006,   48988175,    -1876658289,
      327805778,   1384516111,  313566776,   -752291502,  215465436,
      -1885926915, 883140858,   1795978298,  -1240636921, 1474238891,
      -815283439,  -463433301,  -1775549967, -1856471705, 1374484776,
      421654742,   -1316090221, 21605673,    -653040370,  324567656,
      -1103956386, 1138033231,  -393693590,  -109336212,  -1304369136,
      1685038604,  -1574135439, 141611955,   1986657566,  -1471380331,
      1888575578,  -849117723,  -149530170,  1633323846,  -588547932,
      -1932457483, 997292065,   -1886436527, 1423546940,  248959926,
      323036523,   1007454740,  -962413928,  -1648463207, 56230170,
      -1842264625, 994234810,   -1270182027, -969475433,  -1883596826,
      -806498273,  -203486432,  618912927,   -264791856,  1877610597,
      368861114,   -914770814,  767107474,   730496501,   -516524296,
      -1641738741, -56483346,   596568312,   -924118228,  -1400967728,
      72828792,    1159789132,  -1688935749, 10626627,    -283710914,
      -1630896538, -1830304155, 148981017,   -1172445569, -1109155002,
      -856469631,  1095908973,  11901505,    1916960867,  751105127,
      -163599581,  -1217051923, 368734476,   -1362721821, -1408353180,
      1268812055,  -656785872,  156448630,   -1606722969, -16491197,
      -563370779,  1049894702,  -800116564,  259194064,   -2009389721,
      1110471970,  1660822704,  -2004342764, 626999020,   -80528587,
      -190073343,  1291980468,  351913774,   2018832237,  -1858063793,
      1965518828,  -2040651696, -1717964123, 1576817745,  -141576852,
      -1380315166, 1219442070,  2135892246,  -1394411308, -161445822,
      -522652581,  -1124481644, 1222874507,  2062634970,  36767759,
      387299719,   -2043334671, 421033864,   433122178,   476694108,
      462814752,   1666687813,  -1712791964, 1667017947,  1841218656,
      -493302230,  -1144955085, -1073907036, -1968746844, 1111261248,
      768556929,   -319298538,  -648987486,  1748844759,  -1421866490,
      472316978,   1969138179,  2099264330,  -628884658,  -1304398411,
      708324089,   1069870095,  1596551219,  -1250916845, -1418804103,
      -1957758875, -1284809718, 177766931,   -105312875,  -2130626584,
      964074973,   303637028,   -1998632239, 1019988245,  -1880840589,
      -1279042071, 1433653944,  -1275522678, -727715913,  97955875,
      1597566589,  -111782402,  562985254,   2008807645,  248134312,
      -177157420,  70989180,    517583452,   458676408,   1212048758,
      -468679913,  -820950563,  313334655,   1309699632,  -1806427910,
      871630848,   1053426565,  1669041133,  -1958499747, 1487744604,
      -487652394,  2146354522,  1156650700,  682290681,   553536750,
      160543579,   -2041462342, -1190946150, 1969097616,  -303357222,
      1091056064,  1123594976,  -1142502732, -1717246279, -1670327999,
      320189874,   715279547,   -1742199092, 1751554283,  -1125064555,
      -842927740,  -508734750,  1580722819,  -774109499,  1786423356,
      1664346098,  596410574,   1534559771,  -1070122453, 932196447,
      -467778201,  -1362032638, -849511616,  1271822585,  2130304493,
      584085215,   -577279741,  -1369962780, -1012438802, -351123300,
      -1709051969, -216243733,  1984271163,  430200449,   1487488155,
      791842312,   -1859215477, -226666861,  1325748411,  2060162772,
      -1765353137, -413235022,  -1697103837, 175615170,   -263216146,
      -1573706313, 963725943,   1375284017,  1866358079,  -517857661,
      790182927,   -196978969,  30253897,    304714578,   -513932405,
      -669365248,  -1856882695, 739148309,   -583047682,  -799579938,
      -1554310137, -1106622555, 1541493376,  -1848925632, 198902051,
      1216665612,  1681703070,  -1981903806, 2007360278,  -322427046,
      -1202825958, -1445378264, -877518038,  -2135496149, 1996394576,
      690429346,   -929544393,  1191455387,  931198160,   2142332064,
      -1517942620, 482879309,   428361724,   1629070138,  388986083,
      -1940016529, 839572053,   -1885775661, -513658817,  -106718605,
      906530874,   -809041846,  236977636,   -800390340,  1182186885,
      -1261939211, 1354367730,  -712864560,  -732809062,  701001658,
      -832772893,  -1402993216, -361598839,  -1413024702, 133547178,
      818599434,   536143874,   -1969411046, 2054541725,  -452406747,
      430851951,   -1730756103, 1184830235,  -926988916,  1841867422,
      -1045783821, 758394283,   1637413551,  -1933654488, -954109708,
      -711468679,  -2100020789, 975591165,   -1909410889, 644420872,
      478489099,   169448860,   -1678304320, -180935191,  865352199,
      -733469164,  -1620892964, -1185067212, -594697868,  -1828565346,
      1169788132,  -773693279,  1506104244,  1145143211,  -93485158,
      -155534826,  -546832928,  127051227,   -1039703307, -942306830,
      -485673483,  577939587,   -118577596,  -1658162510, 1621520866,
      1562540543,  -1420038700, -856158159,  -848220765,  912022927,
      -695219491,  -1373152344, 1042437299,  1927923549,  -1456604470,
      -317535498,  -1861110636, -1622882371, 1051025782,  1382867836,
      -1071136940, 1843531507,  -1266698363, 3490709,     119394024,
      -1522463594, -585986075,  1563025400,  721203877,   -1006783969,
      -445287420,  1564661397,  745726301,   1962785996,  1893179218,
      1040757854,  -1028511361, -1317851320, -758094192,  1863754791,
      -530411977,  -1389168646, -1752928218, 1698518389,  -889680298,
      -983940539,  1961966310,  539647097,   331509865,   -257017972,
      2499012,     -1420487898, -1540770475, 2133864740,  1784719593,
      -1352933437, 1775037893,  1770744617,  -603635795,  2115760772,
      112150834,   1348600477,  214973255,   1266232164,  448305404,
      -1040845852, 2091914452,  -880290535,  -2134481152, -1951496030,
      -80476513,   1159104742,  -496581879,  -1265992994, -532342163,
      357883465,   -159383432,  -2048388799, -1372444025, 1530767849,
      1649579766,  -1685035326, -2099255471, 1908814934,  339118273,
      1601243298,  -1564650343, -597843387,  1649671678,  826269369,
      2108188466,  -1382610324, -771523020,  -165095392,  2067535803,
      -73238934,   -541930225,  -1171524543, -1307199207, -844527973,
      1823042369,  -504667591,  501139678,   -1352554268, -210526460,
      -915331709,  953900599,   280284566,   -1900978372, 1566230150,
      -27721751,   472621247,   1369242787,  -1264385078, -2026728530,
      1246323542,  -1028405860, -598428140,  2099138081,  775765494,
      -1002417373, 1235151084,  -1943349078, -545352600,  1434777577,
      -151268167,  -1259005796, -1655785084, -1029254895, 929118805,
      673795768,   1804233204,  719266538,   -342620682,  1393831657,
      1555622472,  -1001755465, 924654791,   179156974,   -1131638912,
      1599088365,  1983218818,  -1811694139, 1688920047,  -1398355981,
      857396931,   1671528377,  -7979001,    -179122266,  -998981983,
      397242838,   -1368906136, 1753162380,  -85549339,   1792070645,
      250762375,   1623685222,  -901325446,  -1957076679, -228742082,
      -500405403,  891099810,   474799572,   1833540504,  -1206986788,
      1883136265,  631598915,   -637904909,  883874795,   658839033,
      722836865,   -1723319814, 2137051166,  56531408,    -652514729,
      -1815917920, 221626554,   -98459893,   258602252,   -1158519194,
      896158660,   -252320565,  1126274247,  -630614831,  -592885061,
      1647659885,  -1904595717, -2064492953, -1464855352, 571199426,
      861544309,   -440079007,  398867826,   -275975023,  -2091267632,
      -794269162,  1900227314,  -1046776678, 1954070864,  247859849,
      222385597,   722362769,   1454410746,  444174718,   1248168879,
      767311282,   1130671009,  1432416940,  -1627925243, -2105776217,
      -2022856901, 952399506,   1735292744,  1606287396,  1160760281,
      151983264,   1782325335,  2103391858,  165721434,   599705805,
      -1415124541, 1567939879,  869945193,   126707430,   1211399279,
      -1654392895, -763833207,  -1482252509, -259790338,  -649676936,
      -1801157477, 1005976952,  -292324210,  -51288037,   1140903846,
      -1890631434, -433758643,  -1941456747, -867409447,  922045455,
      711529357,   1157933699,  -180917394,  2577866,     -750041253,
      1389477984,  35668560,    2009910147,  1842561299,  208231309,
      -737287323,  -395619248,  1037513094,  -75270160,   821412392,
      1960025060,  404645925,   -235000740,  -678084181,  656527556,
      206356371,   -1154609068, 1518633787,  -66180454,   580628359,
      -1594309686, -1223233157, -726512263,  990932089,   -1695573915,
      -887285580,  -1151608918, -1636434814, 1746032417,  -1712216521,
      -1829016896, -1092175733, -2119646256, 1849728993,  121809903,
      -1643131293, -161744410,  -1412583328, 1026871610,  -270591042,
      1474411233,  1272222275,  1881474891,  1845235561,  1784831919,
      917432917,   1860114018,  -1699794605, 612201877,   1567981402,
      -1259251606, 1079982798,  -1658209612, -1853224813, -1028479835,
      1881106346,  1701233716,  -294904716,  6160350,     867775115,
      1881855496,  -1466961730, -629648659,  -1291504091, 1214988845,
      1812464115,  -2001369950, -1810808044, -366885309,  -729724100,
      -2014732027, 1629508191,  -2138236103, 409226111,   856300212,
      1352568737,  342097201,   -1595218921, -420395222,  108470989,
      679165076,   1593991888,  799945815,   -559459109,  398959824,
      936508838,   1789810615,  -1117715194, -1314073830, 108442019,
      -1203080131, -1968279604, -1447438555, -1972913822, -1970165339,
      200436120,   -2130584816, 133588996,   -659365515,  -1754700884,
      1930118398,  683919544,   -1193651133, 2106748728,  -2099282178,
      -435304308,  74180249,    -834076677,  1182965538,  491245881,
      1467266066,  1642694599,  -245974880,  -1855633772, -982303003,
      -354259470,  -574142925,  -1094357744, 1917018266,  -1423913756,
      -2005434193, 187241422,   1477439505,  1402395602,  772765768,
      -527784564,  353217909,   -1371451951, -297755436,  1204598619,
      -1487726742, -1927825368, 724823179,   2113614258,  -1286374507,
      746531587,   883734714,   1234106983,  1909882770,  1027847848,
      846307660,   -631838354,  1409520408,  2035381072,  848862541,
      107309559,   2020935148,  76422094,    -321073608,  1093673878,
      -504881112,  632521932,   1870360756,  -393197579,  233749326,
      1862516197,  491401631,   -1169868827, 1453863883,  -623108361,
      108908243,   327512624,   -485076312,  1945639199,  1366261538,
      -587081779,  -592210228,  1142999017,  -1613924264, 1424922832,
      1635978494,  2061505397,  1639758441,  -1080974912, 1931194458,
      920613181,   2025707248,  -1328071247, 1563537650,  547061628,
      -1141938615, 1793428359,  -860520251,  339494078,   -164766017,
      54067583,    -1147236062, 1443444718,  -678683274,  -1405301399,
      2053473960,  -983187465,  -1382441116, -1833347961, -447684848,
      201297234,   689360584,   1806566842,  -407235361,  1568082104,
      -1514645635, -1609688073, 640443283,   996418771,   -518530333,
      1417002757,  431715641,   1213619304,  869572289,   351243986,
      1147939036,  -1179710055, 1338951105,  314895397,   434040047,
      -1540583970, 1022751961,  -2093598442, -1340141086, 693711629,
      689564061,   16737325,    754382214,   -1528330194, -1267064378,
      929146094,   1213321811,  138952008,   -325753278,  1903656756,
      -530490329,  2081914721,  1209651466,  -346132824,  -178480629,
      309882161,   1289046639,  1759505187,  949748635,   -1992934886,
      -2004241684, -1539528074, -113075589,  1938496261,  -1257552951,
      -531616229,  452011170,   -693542759,  958681976,   906620990,
      -1870059764, -1589673684, -1822966241, -861601589,  2019672674,
      -93498087,   240921391,   -1016235789, -1593424511, -1215812633,
      -403316075,  1771478212,  482463841,   364109422,   -1223426433,
      1425420792,  1712427985,  1744542828,  1557496943,  1872107537,
      -1561773572, -1071551825, -1902141166, 171595208,   -1482520470,
      -167735438,  910909956,   -332847626,  -828502555,  -1794757262,
      -1698184427, -93339172,   -2105332505, -932539232,  324617766,
      -1003487063, 655732676,   -55796411,   -246922876,  1874943248,
      331180898,   1301016527,  923742551,   1907163305,  -156375030,
      974400682,   -259413246,  -1620361420, -622113570,  -945201486,
      -969968283,  2005213067,  -2085674613, 785479356,   -1269021425,
      140957091,   -453769162,  1236852710,  -891063092,  943225190,
      1445565578,  -1949038197, -1430336052, 1586671123,  -1293298248,
      -616858607,  1294733855,  -843984400,  860535882,   1317653939,
      529883611,   -1335760991, -657360607,  -771797456,  228379488,
      39501507,    1532640268,  -658848853,  299697621,   -204523762,
      -881707859,  1761532057,  1198483305,  405375877,   83380045,
      -1322450166, -127270425,  -2059223542, -1501133782, -1205688203,
      -719539835,  -1066385762, 1500406535,  -749510934,  994427356,
      1741995479,  -1789415063, 674847628,   55077347,    -247255087,
      -1698215612, -796804197,  1141417525,  -442036879,  231849631,
      770956892,   486635380,   -235747734,  1905665927,  1592771097,
      2121257481,  1508966179,  -1144057545, -1901776790, 2079025730,
      885712129,   1820684299,  240554139,   2094383139,  1960117670,
      -1646691032, 1117992687,  1750103106,  2049501448,  514032360,
      -2046266676, -1388630416, 1696952564,  1476129484,  255941331,
      1879928808,  444862970,   -1823345663, -1643926672, 1193175995,
      1453142222,  351410143,   -520235345,  855517445,   -524824100,
      -330281898,  292123595,   2053550679,  -1039968514, 248779310,
      -1945423403, 497783023,   -342710125,  -2112515545, 1702009181,
      -444029779,  -168835709,  -2012037728, 1323834811,  2096788474,
      -1933480890, -1516594395, -2002652866, -118079537,  -1895174086,
      1743871126,  1184034495,  1193546518,  -703150375,  79169547,
      -870655592,  1984063025,  -318638384,  -451740196,  -1138204537,
      -989335443,  -1322244241, 1442823707,  -751651545,  541078403,
      352694195,   -1137966883, 1033791148,  484487014,   886755453,
      -2120591015, -2135496436, -305760903,  -1372028553, -346449791,
      245437849,   -299199530,  2072440512,  1510071126,  -1219630667,
      1579735986,  -1546430255, -1183327191, -611527883,  -907521727,
      1771456512,  -1191898940, -682402855,  -1343025004, -1565031548,
      -2055272812, -871744403,  1430722241,  -441023268,  -1060079326,
      -280636963,  -1484460826, -448773025,  619475127,   -552432109,
      885347607,   532608743,   -467449970,  153583300,   -742856193,
      -11757619,   -1372369265, 394366050,   -350988626,  -543688093,
      -579604206,  1632328249,  582024643,   1363118399,  928539381,
      -782018653,  227799644,   665337958,   -538269341,  -1817427346,
      -1026642093, -158157524,  1340616551,  -1879985999, 1651659422,
      1880773368,  1119121039,  -346216348,  1774354653,  943830154,
      1847990948,  -862196993,  591287044,   -1148310950, 1564859697,
      283829882,   489102419,   -1226279965, 597507124,   1481402529,
      -1634097762, 441799478,   -1572389446, 792739207,   -1605394713,
      705592698,   228428786,   1094609549,  155962167,   1453722077,
      -761137042,  -431465467,  567787519,   1620196546,  714679398,
      344075410,   773557367,   1612758545,  1198355873,  1654986014,
      -2042858165, 305401732,   -327808482,  -1993685720, -2117161888,
      246729904,   -1918737399, -1816410836, -1777651831, -900449444,
      -1227986355, -1113257969, 823613054,   2088045790,  2140897196,
      -1726431121, -1812608658, -1530955459, -135696481,  -646831380,
      -1360303303, 1018031907,  561595028,   -1884758207, 1117768483,
      418129810,   1667089113,  -62334275,   -1581810215, 478675949,
      1829600380,  532165946,   504305565,   -1530303622, 636958537,
      -1906536732, 706065824,   -1367681469, -1455326738, 1967237404,
      1925436575,  -1753370677, 475767262,   1597197483,  -1565356168,
      -1627259283, 2025779357,  -510317149,  323637897,   1054399335,
      870783126,   185544343,   -235668525,  -1192226886, 1512104748,
      77337162,    368313427,   1960556984,  133437588,   1677721252,
      1134196042,  -804849205,  -730048185,  -1858970837, -1643748325,
      -526686668,  540963881,   -1738759265, -1951675215, -1878893345,
      2018847505,  2048845787,  -1349177409, 432313682,   -1048475566,
      1400767003,  -2063216678, -1613562447, 1371948251,  948248586,
      343026731,   -1403067725, -1798528187, -1377395904, -1361511564,
      1146886420,  -1083200371, -1168039471, 436646792,   799678524,
      -2071005100, -1963345364, -1001590757, -1134671888, -1682769419,
      -1875365599, -1674291254, 1627082471,  2029678529,  1279499376,
      211077697,   -1834219674, -1441492951, 1155596831,  190112920,
      645879539,   -411921381,  962270615,   350729029,   2003707765,
      -671082916,  167012511,   1650966162,  10172580,    -1666997015,
      318178492,   1525195243,  -415079156,  -272474362,  479266743,
      598310337,   -1381316894, -539374739,  -1262457336, 1478975784,
      860985121,   -1980066093, 243290013,   1549390638,  389566320,
      -1266022600, 1637746173,  -1256440283, 1619853876,  1156099580,
      -700275389,  493424298,   -781157289,  988236654,   1661447801,
      1526800693,  1968410789,  802149271,   -2079686332, -1003300457,
      760463905,   997259330,   6124785,     1080727384,  1916412128,
      818772733,   1253241297,  -92296439,   -1227133830, 269021963,
      2064012505,  -955877472,  936503607,   -161529146,  768913751,
      -661026878,  -1796189257, 1353450969,  639975193,   -1387146501,
      840600268,   -2068008882, -1694712785, 1575393658,  -2016208417,
      2071113757,  1191708340,  -1723710243, 1877105077,  -325879070,
      -1387371740, 865591425,   -886341447,  763609161,   -1838038664,
      872425555,   1121591433,  -248359228,  336482625,   1716320368,
      -2135331898, -791325646,  -553858347,  -1438069357, -350929258,
      1829048007,  -1836920269, 797668991,   -423868290,  1363008881,
      274200895,   -1838889632, -2084468958, -2054272836, 483903684,
      -198054186,  182807741,   1252860212,  1005498109,  -2062257868,
      872389974,   -1772192761, -370409201,  -1445050919, -2132267364,
      -119321166,  1033901534,  -248020229,  538150330,   1572594338,
      879290965,   1463904039,  -1599149139, 234237813,   1439699940,
      782341069,   -1255382744, 650575811,   643869299,   -1070467460,
      -1168907083, 1428315558,  -1049337357, 372561194,   1584849456,
      -85996963,   1409730168,  -1202116320, 875583697,   1289673909,
      -405479090,  -1076795673, -902491415,  785270257,   824381018,
      -107328267,  -1560595470, 347402576,   -1354462093, 1853463757,
      -130606889,  386752507,   1233216365,  1108028067,  775847713,
      1030260926,  352652688,   475366814,   -191782438,  97737775,
      1295415508,  -1428159434, -1978514180, 1725228072,  487294617,
      -1892868793, -1560100425, 537275657,   -616857226,  144029567,
      -929998438,  -1462898607, 14891458,    -966301517,  -874380283,
      975732242,   -1608179734, 881727115,   270758736,   -1460271375,
      75818814,    1163081222,  -991235142,  1113445731,  -946536654,
      1508303649,  1722811060,  1312297957,  1286396196,  549976225,
      -312034157,  357437724,   71561060,    1821733180,  261462042,
      -1642447133, 1459003800,  1141794322,  359592740,   653941513,
      -113914712,  -756592228,  1923980037,  630032662,   1028204479,
      -263601490,  1205766357,  -211582951,  1108188545,  2043786738,
      883904180,   -600318926,  -1992725843, 1805734051,  -1779649729,
      -1066395326, 1211440242,  -517503204,  -288906516,  944987152,
      2116428875,  -623911020,  -468752170,  -1788963952, -127823563,
      1620325460,  229373024,   1786374364,  1962404802,  1261675102,
      -8495285,    -1544242367, 1748297160,  1463208352,  447052974,
      643177945,   -65122611,   -1105386897, -1946421159, 293042922,
      -1741157654, -1193472858, 1992440681,  963974055,   1474866627,
      674131741,   997304712,   503970512,   -424691755,  153568484,
      1061304454,  898932552,   1284624682,  -28418009,   -1325745845,
      1497939114,  854092084,   2076832517,  -598071153,  1963506546,
      -1075479647, -1650159438, 1988569689,  -2133591572, 1377323351,
      267616519,   -2054036717, 1980049615,  333586892,   1787814221,
      231251801,   1285322682,  -1129452180, -381092944,  1974005095,
      -796268288,  1114317574,  1351729440,  1398852541,  -2064390219,
      -1630040811, -569097572,  -235121399,  2107572566,  -1701560129,
      -1154365548, -1072163647, 379995194,   -2059175905, 293360107,
      -963967022,  -1491029961, -534947357,  -1026208498, 143906853,
      1161885959,  172403278,   750236162,   -966417531,  -632151596,
      1939482549,  2039161006,  1029477723,  1353215461,  1548159377,
      832140960,   252309642,   -730343752,  -539518807,  2060201523,
      -1941919205, -984621419,  -1366696672, 1742908971,  -1147730605,
      1997538784,  1475467703,  -1137941533, 1547342004,  793215292,
      -1317069625, -1198603265, -1253776971, -433812830,  1334981190,
      -1169541960, 920369481,   2008676122,  1442268186,  -1959036287,
      1195186271,  1543678515,  -1617184774, 1037181138,  -1315240047,
      2122216330,  1661443525,  -806936866,  1707059717,  -1282170810,
      1142396964,  1115169252,  1902705085,  -1585195981, 832428267,
      1077926539,  612830193,   1558211149,  -1041880019, 1105356082,
      -474435525,  944929243,   -1922266435, 627907302,   1944494793,
      -543062559,  -2015895048, -1981305361, 1574365704,  -357087680,
      1751841721,  -1161286305, -559173098,  1185530195,  706227774,
      -1705979098, 320857472,   82212773,    -1260497952, -1592504693,
      -303738407,  623356906,   -1807156922, -2097465403, -145099683,
      1137856996,  1845496044,  603477650,   -1690163620, -2030306755,
      -781479403,  -938290759,  -101223557,  -1754746219, -1952901074,
      -1744832973, -806701795,  -1244731978, 1731322472,  -1017456025,
      -1687591040, -222408881,  133196798,   1299893225,  -1180505891,
      997258496,   360632113,   -2097600812, -3322497,    -2109670378,
      -1973084631, -239745317,  499301754,   272921711,   2024828878,
      -1192093532, 1348391599,  338744033,   1625776608,  -1828280625,
      -2025040330, 737573227,   732717413,   555606130,   1755090461,
      -1099196903, -66617319,   527481702,   925545855,   1542882746,
      -1630306339, -2115002782, 1385462773,  719000577,   -923541211,
      -1984221637, -630806771,  150905105,   1115247854,  -1207231921,
      988992361,   -68941167,   -614076966,  2119078653,  1077687398,
      -1190898256, 534786767,   -1060865273, 1525691223,  542151555,
      -2008034556, 1594800520,  -818682275,  -459185996,  874375253,
      -1424719204, 2030660122,  1056531106,  -627259332,  -944963480,
      1413195895,  -2046424366, 674324114,   1458388329,  47410722,
      -893775191,  -164513563,  259449477,   339732104,   2080625493,
      1484245832,  675746561,   -703128529,  1897689313,  -1767011292,
      999132264,   272366532,   -760114127,  -2118908560, 1651963799,
      935219994,   21820698,    -893791485,  -410624939,  308770661,
      -1108282152, -866984573,  -377882438,  1319318827,  -1455240892,
      1001422226,  -1880957224, -720305948,  -1840053690, 641887097,
      -1304844797, 1971200401,  1905280838,  -1129484095, 1971476942,
      1411407937,  -830655639,  -332288360,  -1563637522, -980182028,
      340496103,   298637180,   1943821238,  754056906,   -2011710623,
      -943102532,  -979696238,  1834559785,  -870987100,  -1742604974,
      -1801979410, 2025109239,  -876119127,  982400647,   -1516015509,
      -915733486,  -1748009576, -1789940593, -445083371,  1806863252,
      1527414232,  290854769,   2002988459,  -355690373,  902471902,
      176551294,   353817225,   229517745,   429014912,   -1330190479,
      27398791,    418302325,   380401185,   -1294376406, 1982218595,
      -1400187347, -868981838,  613223364,   -462474253,  -59795251,
      -332054051,  -841614356,  -673222728,  -1792197303, -163506555,
      1959925195,  1423278517,  -2029817860, -596437754,  1754644150,
      1532534010,  2098954194,  -1090639935, -828374736,  -2123245881,
      -1966797784, -1910632496, 2077398796,  -138494434,  -446453826,
      1297858432,  2133197662,  -886594648,  -846526617,  -1387320208,
      -269737067,  -352735084,  -1526287894, -1055497562, -1336681760,
      938527231,   -1924487518, -1080523010, 1760622139,  1891979485,
      -40059104,   -1015339616, 265270872,   -1313857684, -73419614,
      -1744348480, 726537794,   -1328957856, -177845751,  1076364404,
      -1591718467, -1611664247, -1761671681, -572098827,  1673173340,
      353211340,   -1115607933, 229092673,   -65350519,   -1402825634,
      -372947611,  -1150013289, 117512970,   74365924,    -121478476,
      365927711,   1681032291,  -1437005894, 819329982,   -450156635,
      -29574692,   -320370308,  695443898,   -165742644,  2049279299,
      -1391514755, 576316790,   -456154535,  1136254057,  212414844,
      -866623135,  1469905226,  1026445823,  -782841612,  -1172089275,
      -223196452,  1364713222,  -1920398869, 961569481,   821662321,
      174006754,   -555675576,  1540744782,  -1634376392, -2054929461,
      822643930,   331038019,   326942452,   1260215901,  -1154849020,
      1313701351,  1934434073,  750054738,   163383603,   -1498125946,
      773004023,   -805769700,  1190923617,  -1475082287, 2026366716,
      -1110741641, 2097965418,  2119378358,  -1555942344, 236410598,
      -429422441,  -230617732,  251485756,   -1827213165, 92756813,
      -1148017367, -52538236,   -408009556,  -201744116,  836298633,
      2051162872,  -441963042,  1869488127,  -1234089586, -275786938,
      -741117520,  -1625002251, 227563117,   -1496513450, -494406605,
      2001798511,  -1065698252, 1655344841,  -1994704216, 378844319,
      582458158,   231389109,   1017492107,  156714697,   812667593,
      1630605564,  -1978815656, -780568391,  -1999950583, -1861096498,
      -1661585294, 1092055333,  72341132,    974426984,   1936579869,
      1004670569,  1106269643,  463176885,   -1822943149, -1057163032,
      2088363450,  452982708,   -1202076503, 1263518341,  -1307867472,
      -1036327807, 1351974411,  946174506,   -355717907,  318795714,
      1034462786,  -1196246310, 1130399907,  764362950,   1387838204,
      -886656262,  1249347029,  1081743613,  1334497491,  1858478109,
      -1277688872, 193791760,   -435909979,  -1093796313, 501896684,
      -1121485767, -1543279992, 1016682898,  1112181933,  1866756226,
      -1492088331, 286912323,   -1917534904, -1454933913, 1790322195,
      -893995565,  -116015169,  -1059549095, -2051831510, -1633591658,
      -81389380,   1569040462,  -1446446449, -1518593083, -586578298,
      904024340,   1976387054,  875415536,   1060232821,  -1169237165,
      -1798510318, -416455805,  -1944048306, -873134528,  -457955726,
      1018848670,  2086922686,  -1719628182, -1466074083, 253736832,
      2031629389,  1959111217,  461357203,   2027952034,  683408210,
      1634484654,  -322424203,  1306089294,  -60061473,   -1952787524,
      1640847444,  1573693790,  1940659786,  -761438389,  1778274476,
      -2014564436, -245459561,  156562926,   978808266,   684342570,
      1616607287,  916926262,   -1494538594, -2142008002, -927064489,
      -1694067154, -1982114363, 37148524,    -229009015,  202655302,
      -1256686804, 828670495,   -1779492800, -1251538384, -613452455,
      564576960,   -202078914,  -521979395,  -315613328,  405732229,
      1087399504,  -394087391,  1686850300,  86085100,    1225047041,
      1876715458,  -1628902373, -2003113334, -35055130,   1952016780,
      108355956,   1929115387,  94104850,    -980290781,  -581314750,
      2143342377,  -1218638279, 882542875,   1274122038,  -22919489,
      556415357,   266426700,   1162654123,  673601964,   2145214759,
      -1402126830, 1354368596,  -1995919575, 1295711278,  2046522537,
      706133464,   -1864535751, 1522219797,  754167095,   -1639087869,
      -1440634625, -1527510508, 1520020837,  -1118616700, -141852808,
      -912831991,  4690857,     -734706580,  702503352,   517538155,
      1024184946,  670213637,   1028005670,  -1669250827, -1887840906,
      2112124767,  -18247155,   -391777436,  1541086167,  -2115532890,
      946503637,   -1878345281, 1065617893,  681763914,   1637315925,
      1323218220,  1022224325,  -2055145773, -191245686,  1418507244,
      -1677943370, -1316108608, 1097568293,  -1174372236, 318050788,
      1597921909,  -1254742806, -1430339132, -106559910,  -873478428,
      285399295,   -1092044561, 1715230489,  -773453971,  2139820053,
      540708251,   97080526,    -303430383,  1184000097,  771144971,
      685405800,   -220911111,  -198653709,  -1461822354, -186873156,
      1979374287,  1155365365,  1488315229,  1668883836,  2004536387,
      1669801871,  868513078,   -418676183,  -1127576424, 1176055612,
      -1795729625, 1344473546,  -1359993016, 1392264012,  -389668574,
      1269651038,  1154963583,  816873893,   -743321105,  -1732353383,
      -705597219,  -1863926376, 1578540222,  1773148224,  -1630060330,
      -540996325,  -1173263072, 1468585618,  -632453712,  1079031466,
      -86126200,   -1227070233, -750395476,  1739058870,  -113390694,
      -1319641890, -1114084618, -358022383,  -825421161,  163612769,
      -2140899004, 846580733,   -1751065558, -941946987,  1855548741,
      1251759581,  898017649,   -400654573,  1979261503,  -666527720,
      726909885,   1838809637,  -121340345,  -1884221428, -1967959326,
      798932860,   680308978,   56746366,    -2022688937, -803123713,
      1039550356,  -1688529488, 300505906,   -1069346685, 1247355921,
      -2004529676, 1279025508,  1834849500,  -1733100286, 1412223955,
      -1203463738, 1683288093,  -347233271,  449131025,   -586179603,
      -493765428,  957391955,   -1851851951, -1964942449, -1654994431,
      759274684,   -1324574365, -967279412,  -207976148,  -1375638718,
      1078018197,  1005706867,  -319136405,  -788259849,  -1445680865,
      -1674007570, -1590844653, 320600983,   902369354,   504114576,
      -739787636,  -589709139,  210144795,   -547537037,  -1696154476,
      1042401917,  -2145241656, 1977846923,  -1747554111, 418305851,
      -1917179098, 1483995188,  17430012,    933628754,   41632558,
      -901040322,  -1878745740, 1910385800,  1269412741,  1307469193,
      -932266078,  -1685625972, -1850884443, 668503032,   945028261,
      148245667,   -1483772759, 915490915,   1794208007,  -1201765911,
      -1056230755, -1465087440, -929195126,  306992412,   -2076124685,
      2060734901,  -686771896,  -1041741994, -214120095,  516604203,
      -1744900733, 1290487011,  1423921093,  -1383884825, -1703876881,
      761171450,   1186518128,  -730197945,  -4106466,    -821298293,
      -2107692460, 1615610835,  -1677443813, -652278859,  1224308397,
      1743004861,  1446066757,  348576521,   -897934243,  712911798,
      519966100,   820245630,   -964979944,  1272060489,  -407328944,
      727004123,   266519151,   566043267,   -2113118516, -1248717834,
      773873000,   -820750120,  -91001534,   -1825439116, 557435018,
      115784873,   1493951348,  1440221770,  -900887216,  -297108258,
      -1963959743, -1133885727, -1499453474, -400537878,  -1225846720,
      -219853961,  -741807960,  1402994773,  -1205869899, 1186233323,
      -237268505,  -672403521,  2006679715,  1513732167,  -70625360,
      881770270,   736051381,   -1387199313, 734580358,   -2009573534,
      655496847,   -1922380157, 1069940351,  1553978676,  292089250,
      -924302210,  -347236174,  -419673197,  2139411176,  -283307601,
      324558237,   1302913802,  -1055265843, 33021796,    1739460956,
      -2055383924, -374140667,  -1094605507, 117278872,   -332855418,
      -983892319,  793170060,   433092779,   -717403074,  1047441230,
      2021925213,  1946239342,  -885552716,  1433178375,  -795592984,
      510211084,   -5835994,    -1906366063, 410814361,   -78105580,
      -1792601061, -1553448098, -1882971582, -1528952694, -82409527,
      -1251482468, 734609349,   -309834618,  1183164712,  1081252172,
      -1683858899, 316326110,   -806082488,  -1338584394, -387222739,
      -510155655,  1566577340,  -1000077139, 2038464515,  2126312657,
      -245284645,  -329612668,  832426406,   735280768,   2135169801,
      -1810774938, -1450760198, 929556103,   321547885,   782807043,
      1858544810,  -2074365261, 1794689751,  671651420,   -1230129250,
      -507161382,  -1695791676, -1534246123, 1995118604,  1510237527,
      1357067598,  63650323,    -97645883,   -888264378,  349327954,
      2050556449,  -667419246,  -1856738929, 1531641742,  1934233563,
      -2112274760, -1501839128, 2105614335,  1043410064,  445272680,
      1409759427,  1352078344,  -916971417,  864309027,   1057947967,
      -749033894,  -630737731,  465477317,   1435590861,  589437354,
      2014847382,  -1615825354, -1058184903, 263492924,   -2019848793,
      2096958223,  -1982353535, 353510084,   806734095,   -1591555140,
      1887613515,  -972402581,  1509271673,  1386945508,  -1218791259,
      -174613869,  384277906,   -1214970726, -1955131506, -652853606,
      -2088204589, -895223404,  1003912965,  -2077498996, 1163348181,
      -745329252,  -136818852,  1550429477,  153711354,   -1845795495,
      -2013937379, 1488123267,  -1645575093, -2058256761, -480842553,
      359625812,   459717342,   2007953345,  -1102587049, -2062861789,
      209508861,   -1970411068, -1330745897, -1871378040, -35555601,
      1122264135,  -1437699998, -177104392,  -731042395,  2096907540,
      -283094649,  -939516566,  -55257803,   1364800787,  1293633082,
      1153768212,  -1358246209, 1374594804,  -519495781,  -2076128605,
      -37954011,   1371546185,  -581859215,  -231816850,  1485157964,
      1815317955,  -902326565,  279616532,   -1781505820, -2115741214,
      2055532536,  -1538202909, -1511321286, 143076441,   210483509,
      -275503784,  -71177135,   1289651171,  -1532250332, -171367426,
      -143812360,  -877632809,  1658414210,  -1129830761, 1904163165,
      820317325,   2072887047,  -781566734,  273570617,   46574607,
      -336857731,  -680593764,  1561853607,  1011663020,  -620268635,
      -1456917230, -1545429039, -2013124203, 2072998856,  435507184,
      794998732,   1435534855,  -421936391,  -96773922,   -588150011,
      -1288267693, 1451085845,  -1910210616, 368172665,   -54433939,
      -1227831714, 212187441,   1979091392,  -121385556,  -241610706,
      -1544880161, -147860183,  -1069897242, 209486605,   1169457394,
      1240478377,  1915756344,  -788973327,  -1066972997, -771199673,
      -1397463728, -383389081,  233788661,   -1512974275, 1369247995,
      1631629609,  -88037023,   -2077001505, -2039268825, 1220601163,
      -1159949700, 1905274114,  2001411187,  -690776493,  907651649,
      150515155,   -918102542,  1982220600,  2080179931,  1726285503,
      -1740108718, 423906275,   -1636944334, 1223430487,  302615699,
      -509989774,  -811614459,  1249431622,  1138768886,  -1568514772,
      2012027777,  935308629,   -614240793,  1026027312,  1078009369,
      1442755545,  854787935,   -561923889,  -707625689,  2046066177,
      1353216774,  -1543662248, 82556400,    -328487520,  1170545997,
      -311453425,  -1974251506, 1565262524,  624113127,   652983938,
      36470831,    8249591,     -1947736275, 594439045,   -64264165,
      2043214125,  -2114623488, -373557041,  -1432860598, -507816355,
      -970391509,  -1276129617, 1818842484,  -1203715648, 1512716525,
      -291500244,  405461184,   384087122,   1684890061,  1360300780,
      1519403356,  1439549275,  1606073867,  305499208,   499208988,
      1392214549,  -44429507,   -727470110,  -979944365,  1347602287,
      -212329108,  108138349,   1877031459,  -279528944,  1802911080,
      1943810904,  1429340608,  1503552430,  -1748544512, -605970666,
      -2089666262, -1862268687, -191367804,  -81713303,   22881827,
      1321112822,  -1526623337, 1894394892,  -2080979943, 1065886894,
      197337084,   274178924,   1228175179,  1100505720,  980780776,
      714827942,   1598506207,  608574109,   -2098868516, 1702008652,
      2046918766,  -810758294,  240411154,   -422444342,  -532712597,
      -1931006875, 565412321,   -80144140,   1210593751,  1383345196,
      1370618544,  -1596394721, -1489958252, 1181853981,  1637777562,
      2096932053,  638976488,   -1238366965, -1479739107, 1572376333,
      700324465,   -1869572795, -623381161,  -351555313,  758451780,
      1217512064,  1976435193,  1911506998,  1075469171,  1726439297,
      1770021659,  -701884110,  1052289426,  587376022,   1398861433,
      -2146081301, -497780568,  -1234208298, -1067800729, 597996849,
      1811405609,  482394479,   -2018918020, 1972254321,  910471623,
      1476039049,  -204812449,  -206447772,  1904301539,  14697494,
      -1663561588, -1261633865, -1850825269, -24756671,   -554192507,
      -1091276215, -2124443532, 353488715,   1719757775,  1797923440,
      -1204301820, 1855983547,  2136593349,  1078542146,  -98284310,
      2144433740,  -584487400,  66760176,    703352297,   -1793826986,
      1177132266,  -347413430,  -1103964462, 399394531,   2025513279,
      1960667447,  -390756871,  -814575780,  -763005389,  -146370008,
      1046904445,  -1356358108, 1423826126,  1299532598,  -543808808,
      196272479,   606471677,   -720860532,  1209439336,  1820796860,
      -332691105,  -1656783485, -1376253569, -418614434,  -1882207371,
      121710697,   -1365264227, -817613537,  568149360,   -1579619904,
      -1598805554, -1410466119, 653855282,   -1164321046, 956584348,
      1151294244,  954088499,   -1517634686, 1922369437,  739625220,
      248925415,   2070340609,  -542799298,  -1357589928, -1286241141,
      -604877059,  1984969532,  -2123321066, 630659668,   -832154755,
      1479843319,  1794391558,  675891153,   -1955092195, 802218208,
      2060128081,  -12876850,   -396057405,  -900206475,  -179130439,
      -2010928933, -529430850,  1996354070,  2024056511,  1080618217,
      1189162977,  293417222,   1025945782,  -635374496,  723704135,
      756866883,   -444825570,  369126536,   1429228898,  -12145491,
      -926021303,  -1685179670, -404778204,  -706973733,  693958181,
      -1363325337, 1962687572,  -1716657804, -779120296,  -1314807729,
      2090100650,  346928068,   2023601625,  -1402207343, 1874078629,
      544639793,   -11243314,   1752552374,  -501097671,  -1516551275,
      742950030,   -26313612,   1364747587,  -1842939934, 440240827,
      1372779299,  -1981970193, 2100215551,  2123025833,  -2027395543,
      2091921028,  1311115754,  898367954,   -118208124,  2076486928,
      -1855936108, -1553205850, 575333373,   1864971720,  1100130495,
      -179852065,  -643463979,  -1394160584, 338374485,   720364643,
      -2008813496, -1823674173, 418345043,   149418936,   -1696340037,
      1921285507,  -1842247489, 1166151898,  -1920526682, -1809548335,
      38550617,    1518420853,  1034527841,  -1585558553, -406044050,
      -1158123030, 726534032,   385872469,   631680651,   -1790437307,
      -1376706696, -479998627,  -320261147,  -875525680,  560383851,
      -272926320,  342850473,   -1730892386, -1932328947, -767777109,
      -2055437901, -1141726296, -1763910553, -1167003236, 2041973350,
      474352857,   -657845976,  2094526275,  350530773,   1157554180,
      748969515,   -997381715,  382892949,   -840971766,  -2061570788,
      -974795185,  347351307,   -858997213,  1032164435,  -238593319,
      -1657952020, -569151326,  -953829364,  1508561283,  -1060539910,
      -1423832936, -1551953327, -1784503853, 446769383,   1838759049,
      796975780,   -1261252266, -825062111,  1846902738,  63797435,
      1289458326,  2097636314,  -433758647,  -505356381,  -1293971970,
      -1605293974, -1589729811, -779871440,  671810104,   -1470261781,
      -1496653277, -2035987107, 2091652464,  356261375,   1320239144,
      -1243357498, -659259481,  -223774739,  729451189,   114388886,
      1533358905,  -1743950101, -609308262,  2066504460,  1353133876,
      1572588794,  -936606363,  -1882868959, -208075213,  -1322382570,
      -1503061702, 793077911,   -782702963,  1747957102,  -873231871,
      -152213171,  408126435,   2069741818,  -819616937,  -438420224,
      -1647396378, 1851210420,  -449156129,  -794965522,  981379967,
      -1337973517, 1053232945,  -407327866,  526423756,   -1305294978,
      -73661010,   1005899393,  -1876160167, -303961062,  532964577,
      -757207848,  -1212838188, -654535806,  -372818498,  1984515422,
      2034878609,  -267535530,  -978549893,  1475181311,  347627261,
      -72084069,   -572336443,  206465130,   2132646571,  -871429938,
      1421825793,  -342787475,  2140594480,  -1491550671, -1848477545,
      -189293954,  -1669775850, 2058018283,  -2101004770, 2023285635,
      -90265004,   919784859,   520459786,   1901032840,  -1758073818,
      1847321274,  127556372,   2039070198,  1230010770,  1135085963,
      -1532162132, 997912076,   1022896027,  -1674855182, -243141775,
      1171092015,  115960856,   649147299,   543219608,   -279801145,
      1698407499,  2040226559,  1974269417,  1186409578,  -362204894,
      2032062773,  1869609481,  -2119122965, 106614410,   -1722686864,
      2056097840,  -2047547794, 247009746,   -273135030,  1632072302,
      -512227907,  2014809362,  -1338997015, 1704237701,  -1130914782,
      1568447769,  20819538,    -2084849674, -286456829,  885935361,
      1922482420,  1124711659,  1722624721,  836948525,   -1372872417,
      -1648703681, -154922737,  208746465,   -1834614021, -402576904,
      -261358233,  -1445028197, -1154055421, -2017079293, 1423300919,
      -704226943,  -902008302,  1549100146,  1399647405,  294785542,
      312683094,   79486949,    -295180346,  -1997972241, 1484749515,
      1052507582,  20871674,    1093241711,  586721079,   -1860026727,
      -96557163,   2098062292,  341794938,   136291901,   1491838826,
      -1252094163, 595064547,   -730876961,  -796764015,  502184004,
      160914341,   1041953500,  -1699419873, -1899967568, 826468875,
      1608787541,  -476848605,  573858659,   791246022,   1006362381,
      1781507646,  578383537,   -134460945,  -1709517728, -190546801,
      -2054141382, 1927659238,  -50441114,   1315168030,  -523637028,
      544770467,   -1596512502, 1351456558,  1357077411,  349995855,
      -1800088366, 1973501941,  -376936124,  1143380198,  -673759351,
      -1547951414, 1041861941,  1751175171,  321206359,   -858591610,
      801695572,   934052623,   -167860821,  99234755,    -1753642149,
      -488420401,  -1547907319, 1560438789,  1384607814,  363778017,
      426170566,   382974725,   1130588087,  -1075958602, 237923160,
      -704961197,  -1559438859, -984188425,  -254960832,  1560346368,
      552043967,   1375175819,  1678912766,  1988852105,  -1464905688,
      163592528,   2095942098,  -1240199367, -791413851,  -572608518,
      -916276204,  -916234197,  1054725393,  -5594893,    -785986657,
      1548133480,  -1679824522, -1921725838, -2088841153, -57365568,
      -35376533,   1039875180,  1964745866,  1640591527,  -1590092245,
      -458558826,  1210148161,  917759612,   -1065684787, 740991983,
      -815530962,  -1265985719, 632838644,   -516992325,  -1511463388,
      884044812,   -1918813285, -1541759211, 2018308006,  923190862,
      -473988134,  -1528834563, 1926654397,  600272328,   726519892,
      -127490456,  -1441168832, -539213550,  2045598195,  -139959511,
      -1888871203, 211134411,   71682070,    1961903472,  -832303111,
      -1576349585, 1474360988,  266787397,   730664946,   -1810040979,
      -266270695,  956850236,   -1309311990, -546739882,  -114502391,
      -864456742,  -274808464,  1532839678,  2109188575,  -998062417,
      -981869118,  -2146399187, 543902973,   -343430318,  251455429,
      -875225598,  212631070,   -1403626262, 1808482902,  388412848,
      133547487,   -1523425564, -2118077120, 768798315,   -1887386212,
      -519068358,  -40917301,   -1511549863, -1862246272, 408131625,
      -1382978150, -1394551348, 314501584,   798051540,   -1819433897,
      1953330316,  -1847593214, -2014192399, -1747267618, -785391040,
      -1284482413, 1430809624,  1879290999,  -1406549197, -1445644294,
      -399585757,  -205005535,  -178263951,  -757331143,  -2096776376,
      1732124323,  436591086,   550597238,   1466585553,  -885639682,
      -1149096627, -1709860215, 1720194408,  -313147476,  -1241977439,
      354150818,   615468582,   -679067982,  692703625,   1599258936,
      1428028236,  469809880,   31307564,    1832599135,  435365181,
      -190778118,  1886232496,  -476812330,  1026368786,  1825809177,
      -1906497276, -839312851,  627387378,   -1225546751, -334973997,
      1005660227,  1100028678,  2066512299,  1180986788,  -744967248,
      838880844,   1515057956,  2039958566,  1688531532,  1499885222,
      -28897980,   -1404055868, -1459460881, 640342455,   -795477647,
      -990859609,  684570510,   1967869422,  -1823508494, -137805923,
      -1147241992, 2082656069,  -132609996,  266279565,   -301054569,
      266356413,   220838776,   -391117762,  771233196,   -1990107832,
      -102963880,  -1104801990, -883507005,  -700463555,  309033783,
      922229967,   -627229698,  1800039092,  1338183391,  1613758775,
      1722863691,  1417627020,  -253703327,  782593561,   -2055376886,
      -870371225,  -1197741433, 566637413,   -1367621048, 1518449051,
      1137534978,  1646184338,  863055996,   1298732047,  -1753739469,
      -1062086736, -321165637,  -860659334,  -887888935,  1692263905,
      -1944464760, 1309755808,  1664516593,  1116564154,  1391859275,
      -2085654925, -1048064016, 241971813,   1334066131,  1043606424,
      -1020484783, 436716941,   667270407,   -1735133373, 1910478014,
      -534874049,  -1176883348, 1221729449,  -1803125587, 1565596902,
      2102706805,  567535857,   -2118009653, -521077701,  975194756,
      -140038233,  325367165,   550543258,   778817548,   773154402,
      212460306,   -1415118402, -1927542429, 1696131712,  1767827275,
      -111582362,  292718506,   -1547812498, -1499764839, 1555017577,
      294587761,   385124001,   1390247754,  -728393112,  2064207420,
      1237247264,  350014652,   1484981689,  -823970169,  1413371625,
      -1553493112, -1075433497, -2059191944, 1434321548,  -1547713280,
      835483623,   879086219,   -1305159619, -131124205,  161926571,
      -969014220,  -1272576760, -1474948974, 1388256500,  -2022196621,
      1014703855,  -1486106911, 579036172,   1980903206,  -771846208,
      1968324310,  -1903841344, 1819978728,  -1079732692, 565710635,
      -647887126,  -1361385004, -1824265658, -1230280653, -1115610834,
      -1141001166, 505231575,   1384372107,  1923576421,  928306731,
      317749029,   -1611399907, 1053662195,  -2144160669, -2005977557,
      -1108683986, 1484776164,  156940497,   -1701426324, 1635304832,
      605316766,   576788462,   566748384,   102027797,   1876682414,
      -768195962,  -2137375113, -1598119353, 204189355,   -1529415220,
      -1528787413, -1179432889, -835070614,  509981129,   1021981352,
      -708083391,  -1264399416, -2051467711, -759324725,  -1584628166,
      -24532003,   -343112578,  519956017,   -407962511,  -664819627,
      286789927,   -922347109,  1765009474,  1953035408,  -776024430,
      -2119568352, -439618828,  1938943158,  -1728705127, -929866247,
      -518861484,  -1766497415, -1983366570, 15612672,    1163589109,
      1952701044,  1658741567,  -1479866303, 41472957,    646789411,
      -1727639122, -304926139,  -941408151,  660554359,   794584252,
      -669425158,  551420196,   997172957,   -1724836647, 140144366,
      1829071622,  -1323142765, 2064785419,  -1113964881, -1253726357,
      982610377,   474235599,   1035545452,  1446478868,  1953142225,
      1472839376,  1071640547,  -791607264,  1693323827,  -1348430236,
      -351079814,  -888078942,  -570338910,  26421464,    1747802094,
      470174916,   -333288522,  -910347368,  1966661790,  664535857,
      -52348423,   2014202022,  1670017111,  1933623887,  1670504565,
      1566101633,  -1550899859, -891336364,  -1732817757, 1665849874,
      947159773,   -1500664468, 1921566526,  -110128817,  -713736141,
      1415784021,  1620956849,  -1957509751, -2137222986, -1314185605,
      -135529499,  -791792001,  -1860906912, 1764943725,  1294975210,
      2118835,     -311942634,  -1689353229, -1192887608, -1323875072,
      -585582591,  -141890804,  542654949,   179104573,   125891361,
      -813271519,  1467512032,  117532534,   -536059414,  -1052071053,
      1887269971,  -681295719,  -1420366307, -1629602378, -1163733788,
      -2019763275, -962910878,  1253841913,  -1779962655, -994398413,
      1159566490,  -1746749913, -1630736384, -497263143,  792383100,
      1026417770,  -1881802762, -744992073,  -184726724,  -208548455,
      2056600801,  1599421692,  1098517166,  -552593026,  -1631567069,
      1627665183,  1053904166,  788928126,   -1510004414, -1577755131,
      1110644678,  1304636158,  894759554,   -1815415051, -1571249377,
      -362410559,  -1547823472, -1915267185, 534522424,   -2066391103,
      -350064990,  25471734,    -1016967089, -1835297934, -1919565823,
      1822856494,  1325160341,  -88737147,   -2111061216, -1909009399,
      798175982,   696633468,   -623655322,  -544044517,  1778586519,
      -1576013969, -761456483,  -1820911723, 87115865,    1508262949,
      -1331691144, -933847144,  1121599222,  -1689122394, 2128443672,
      -233845005,  986024586,   -1520169983, -229920997,  1582855986,
      1886155230,  -1167140975, 1847660489,  -1286117620, -11296180,
      -1287425374, -216161083,  -1816534495, -1366497994, -1258259996,
      1440541558,  -433788506,  1796507254,  1257849365,  -1249764515,
      -978595100,  -389013492,  522045237,   491044307,   1345211861,
      -147159388,  -1890239748, -1801115055, 1499624236,  -1240277082,
      1342661121,  -1007000510, 1615160590,  -427263058,  -126124330,
      -1299634798, 1538123322,  2117634397,  1074911745,  -571160284,
      1434266612,  -1974685313, 1944917257,  498292042,   -630826138,
      1386543955,  -1501087106, 1525657412,  -590930165,  -2136954625,
      71583107,    -69135848,   -1918929021, 1321599043,  1727561497,
      1816391786,  1559407070,  -1835413720, 1001428095,  -1299316737,
      -687490712,  1201631061,  328272769,   225710655,   -1679230786,
      -1416446277, 346023355,   -1755278413, 1068384913,  272328400,
      -515931246,  353535490,   1952938191,  -1843247018, -1633585215,
      -559210341,  -1823160921, 709552643,   -949178873,  66604904,
      877927786,   2038757496,  152325014,   1242102250,  -2105024966,
      -948713671,  543938036,   762181021,   -1880088756, 1929110794,
      732850446,   606831014,   291229837,   467106388,   -1373384909,
      1626900923,  -1768497726, -605856536,  514206894,   1022412182,
      -962773383,  -365333555,  -1182180909, 1485982561,  1037642785,
      -692434694,  -1908513534, -1107537150, -48240990,   1576143178,
      -550282149,  1885400513,  -1781224979, 556889263,   -999037700,
      1025112882,  733566880,   133501465,   1750821119,  -1169136863,
      1475653090,  1838838648,  -688301490,  -1357899677, 313996970,
      797355358,   477773154,   1851791787,  -569921228,  -848473704,
      -1341194563, -107237646,  -1474275571, -189603167,  313084916,
      -1868605837, -1135185737, 421542277,   1856139659,  -816081246,
      -1726539231, -449440716,  -1353137300, 1268747474,  644944336,
      161465421,   -1666177703, -1630549643, 1939333029,  -1787416570,
      -1062614800, 910260900,   -499860775,  1623032717,  1911436956,
      -2072174815, 1855478156,  -1131525755, -2130475481, -592874822,
      -665540680,  158667282,   -1184111002, -824535484,  -1876934495,
      226329889,   164523821,   1581783269,  780244812,   967235198,
      778954940,   332957990,   -86103694,   2051898092,  -960611974,
      -566940880,  688163436,   1554325475,  1115548999,  1884106076,
      -618792348,  -1354756867, 596208148,   98211832,    1784290943,
      -1135955105, -741185679,  1137413633,  -565593848,  1279136534,
      1000178874,  1626778570,  -1056820425, -803116597,  -1028195403,
      1323146405,  -2061032718, 922177098,   790123941,   479807931,
      -1850068662, -1918724173, 28949193,    880271067,   -1475430027,
      68464587,    846777310,   -1103911939, -1176553272, -1897633650,
      504479106,   885350541,   -1835890572, -1625210860, -522624251,
      -1839369957, -117247781,  1534380190,  1263141443,  -586617443,
      1543409058,  -2050432573, 1336700428,  -835625630,  -190543031,
      938415011,   1946737072,  457456147,   -4862069,    -871472741,
      1828966757,  1663150978,  1964032432,  -746130880,  1287013257,
      -1838223582, -701177150,  -1335398538, 2025552398,  279138743,
      588471153,   -1178083278, -1532215911, -1724653479, -1587181783,
      -915169206,  745928116,   1778532537,  469970664,   -1770912137,
      1612482200,  -207498997,  67389101,    -1037330964, 938059208,
      1293133500,  -1177164585, -943265783,  661243467,   1001995457,
      -1351091391, -545233734,  -1385973899, 931367276,   -1761190563,
      1798299643,  1049204425,  -67055593,   573266570,   -1798237505,
      157498286,   -2041954524, 924922362,   -499608709,  -363004406,
      -228729770,  -1316639394, 56982143,    -1309390995, 1199726894,
      -608411996,  -893132147,  -35711036,   1504171657,  -661263515,
      131655233,   -1248537103, -1919290051, -1732433323, 159946341,
      1072452743,  1778417327,  1902798455,  -136227093,  1818039727,
      -2109016793, -741326340,  -952487319,  1198317711,  -1084607504,
      -507061475,  -697869052,  -144120857,  816042036,   52075087,
      74749234,    785506277,   635882289,   -2131614323, 2045363677,
      851530164,   1595304187,  1670206531,  -196634566,  -1068892868,
      -1883729839, -1647542978, -1925929352, 1389197133,  -721341520,
      2099467114,  144786353,   1495152886,  -1070848323, 1461203563,
      42649538,    339145085,   640181609,   1744574904,  1719665956,
      25730591,    352282040,   726411024,   -938747992,  -1264693731,
      1764752981,  1801744335,  -1076444453, -131287924,  1042638646,
      1171397043,  -434259856,  -278160656,  1272417544,  1903832385,
      1088496815,  -237656962,  2053448482,  -809825370,  867795662,
      2030450788,  1002199523,  1012446867,  353664619,   889858219,
      1550050278,  -1740158574, -451956945,  2022208635,  -1702712072,
      412178343,   1623055544,  893232937,   -1822487717, -1508510116,
      -1055359453, 719516498,   414035005,   161459909,   1945235806,
      505953390,   -1396601505, 737122071,   604629261,   -942645938,
      -1574289018, 1320202111,  -865754034,  -765341238,  1347567122,
      1739010597,  -2030047406, -539216646,  -513970609,  -1556256048,
      1709262988,  235246938,   623199544,   423321032,   -1923099399,
      -1754583762, -24115536,   -422545961,  -1355064201, -263627916,
      2138979021,  -621542109,  -1171275782, 866827868,   -1617131201,
      -2023273634, -399190916,  -1529045249, 2132055689,  -251084459,
      523666616,   -428322915,  908134238,   -75429697,   -1067335643,
      1951573681,  1070311066,  -98773922,   -1938111768, 1541910728,
      -689782070,  -1553788089, -1751305967, 177294935,   -1354814111,
      -1231032406, 213641117,   1432441104,  -442242924,  -913171746,
      628961459,   -1320507847, -1028369003, 1629380904,  -779895510,
      296722807,   -355425238,  -1696247348, 324203260,   -1253782679,
      -223885234,  -13231864,   -1026083221, 91816433,    71880056,
      157983202,   -921611481,  1242820832,  -107216611,  1699784839,
      -1886427028, 429067757,   46506437,    1384206113,  1911364724,
      312452186,   119749896,   639308347,   647201036,   272924118,
      916801068,   8637576,     897671496,   -895465789,  -1811535359,
      -1151550200, -115151848,  -1390868875, -30375920,   -1571729496,
      -62989611,   1458062276,  -2030457959, 1895416082,  1451297139,
      -1165055795, 429491198,   -1492062768, -252128958,  -522809316,
      -1708705660, -626291628,  -1928624651, 1075474761,  224294861,
      -860245087,  1058781324,  407607861,   -266828163,  -833312433,
      344992463,   -751096207,  -1287424582, 469734191,   1577573428,
      1870723988,  1539763223,  751535061,   2036617100,  -1917479290,
      1220652326,  -2099056532, 863157690,   -794470832,  -457137913,
      -321353893,  -1367233006, 710237349,   -474253784,  -443529084,
      1351058625,  140428613,   2041021335,  2092387287,  -1497876331,
      1554258343,  130747573,   -1756890787, 872749054,   -963185821,
      -473180091,  -227266462,  -1575290309, -1184647,    950224821,
      449739082,   1058166499,  -952336759,  -1394441963, 170602170,
      -837306609,  1849874908,  1594044043,  2141554190,  1460440660,
      -748632537,  2081939335,  1290547267,  1362411208,  1363874511,
      1886154666,  -1538512071, 1978765289,  -2020388878, -1720077271,
      842070675,   1744009107,  1608047620,  1395373448,  -506828583,
      1706398011,  -1756278672, 2045412789,  811548132,   -783702929,
      1080688565,  2006823296,  -1444770451, -2130536553, 897991003,
      763562545,   -921058483,  -1183863211, -404565314,  -1780009425,
      -2004288795, -175374831,  1863100459,  -154244310,  2127836310,
      -1254226101, -1655039803, -1507789435, -1749872435, 2049720527,
      507574298,   -325786810,  -358295928,  -1588076812, -819129509,
      -1519459974, 587266187,   -1318691077, -750258029,  -370142195,
      149753817,   -886999293,  1643979240,  -1741268265, 2124101286,
      -2136495413, 1325252273,  -298880000,  753719767,   1879874742,
      -30328236,   418166881,   -2070311026, 87370692,    42876483,
      -225962622,  -1088834830, -591809361,  -65919099,   506363034,
      -303332344,  1509434654,  359055038,   1301549790,  701189053,
      1448606181,  723737401,   55982052,    680072085,   -571442925,
      1441083909,  721227188,   -968409712,  -309100808,  154083236,
      -1715534032, 657952665,   -1292418649, 350924381,   512803384,
      -1841868628, -840817530,  -1793786731, 1208436966,  267545080,
      -2077226706, 663876387,   1477049341,  1316531287,  -258179482,
      -895699856,  119468253,   1966474472,  -1846013934, 1450369165,
      -631005065,  -1419940947, -1419668328, 1485877843,  124041648,
      1320643872,  -569404618,  -1279182745, 660610247,   -674357403,
      -543141129,  -1572960354, -621752000,  1619139132,  51245632,
      602175581,   961455539,   1130974632,  1797899913,  1902857706,
      895935268,   245791109,   -1216989624, 185812709,   1680352703,
      -871779027,  -1026565849, -1991113060, 1722435770,  2074986788,
      1311499076,  1857940675,  575039088,   -674063114,  210713256,
      -1079890869, -609733618,  -314349705,  1414615828,  -207280053,
      -1985024489, -1913776309, 2005541559,  -1961617747, 2128598223,
      -1288393475, -974923022,  252833688,   -1771363873, -1859907461,
      -1999310394, 1949989258,  -469378122,  921897881,   216377818,
      -1687168361, -756373266,  -1761776328, 2141915757,  -1775054148,
      -863857898,  -2051239041, -571011484,  225618514,   -956530643,
      801723563,   -1997053251, 2039248647,  1856956225,  -1347311861,
      831678680,   1568037909,  -28833080,   -115764750,  768607241,
      -724992436,  1701803015,  -1651869783, -1507086790, -501770930,
      -947684824,  557261373,   1939119146,  1597006935,  2047640799,
      -569753582,  -1113174374, 693634714,   276313200,   -345115488,
      -772811634,  121750351,   -466235474,  -1773857682, 1409668468,
      1467301157,  -2116581157, 2049477044,  -452936371,  -2081173164,
      757035619,   -510381238,  -1118560227, 336537626,   -1111937974,
      973614965,   -1827543746, -619747845,  1384767428,  -362016678,
      -803797815,  1645235541,  -118734788,  46751891,    981832961,
      109786147,   -27624928,   1678897899,  947994421,   -1980643577,
      -775989046,  258956789,   -2071966013, -937587907,  1561958365,
      -286357406,  1929080874,  -923024843,  97179599,    -1509302051,
      171882508,   1415370858,  -2046785480, -1839972205, -373980428,
      56495547,    1692946533,  -1433220404, 241966238,   -1553334942,
      1254280791,  -1981289656, -340649750,  -2071937376, 793881926,
      -887463009,  -1598338012, -222833262,  1950378361,  1098688286,
      1367419815,  1172309175,  681981175,   2121321941,  -1726718346,
      -1450123766, -1114293893, -42437408,   574632321,   1994447639,
      -972330683,  1478604011,  -1109738539, 1973149174,  1493299539,
      -1274277415, -399170631,  -746093222,  921942675,   1140765790,
      -832452674,  -747666697,  -1903898437, -930531316,  1899638350,
      1135873146,  -1740832106, -583699646,  -968483149,  -1083933009,
      -1772035975, -894132322,  -1750817375, -170561336,  1180169878,
      -393220015,  559151184,   -1970024453, -1082414923, 106077224,
      -2114977817, -1337304798, 1006965026,  -960400580,  596161369,
      -1604296483, 58459410,    741881563,   722536580,   -398034725,
      -1792777379, -200368288,  897886615,   1127089602,  472965613,
      547172238,   954459245,   842535501,   -494188197,  431145848,
      -1839836223, -823377846,  1154042620,  -1306191003, 1563546878,
      -668059978,  382215864,   1035063354,  -475886653,  -1406459485,
      183861421,   -767700778,  -793905979,  -938354169,  1997125067,
      168399355,   305377492,   -545947273,  -172028390};

  int32_t over_bytes =
      compute_int8_over_RW_bytes(chans_in, k_height, k_width, chans_out);
  bnn_b32_t *K =
      (bnn_b32_t *)malloc(K_ref_size * sizeof(bnn_b32_t) + over_bytes);

#define X_ref_size (x_height * x_width * chan_words_in)

  const bnn_b32_t X_ref[X_ref_size + X_REF_OVERREAD_WORDS] = {
      334618952,   990892152,   -1092432719, 1188540535,  -1814923493,
      -1584797214, 408949552,   -699399485,  729873736,   -1664688196,
      2020303517,  1480057315,  -1652919757, 1348543039,  -9257380,
      -625415367,  1504281911,  -1943268141, 762630346,   1801682668,
      -453827149,  -1077900352, -1338399678, 2026636893,  1566309361,
      83186255,    -1761760276, -902980902,  109971881,   -1597388508,
      949583073,   936073207,   -1149722608, 1995813516,  -1106926095,
      -1700630222, 307624189,   -984983573,  -1770058237, -123772056,
      629549217,   1680182775,  -1318189275, 1912006624,  -735535552,
      -1972655310, 570854093,   -519210994,  -768325412,  -1892963369,
      209845024,   -888631061,  261423710,   1470658005,  1500718866,
      1567962070,  -621089056,  -901344866,  -1247257503, 418865405,
      905399499,   1832782945,  1468192205,  232764912,   798153438,
      -1607452505, 383400894,   299513956,   -358998783,  6218975,
      848953286,   -865517656,  -721492261,  -1609940579, -627331462,
      414573625,   1523386455,  -1535693987, 1725220760,  -2019478572,
      702361642,   1445881449,  -1348603564, 417312089,   -2134784428,
      1367334885,  -41720305,   466520571,   -1169712475, -705350854,
      -959479061,  -1834530915, 535398491,   -1817446401, -915533814,
      -459311787,  1884048871,  1662044830,  2088557221,  -502505992,
      -604393441,  1355235602,  451593141,   -422441769,  -1665793030,
      1715902700,  -1141398516, 1811165566,  -656462281,  -2053222356,
      -2077905342, -1467759755, 591974600,   590004126,   2085205663,
      -1978540661, 35015452,    -745754009,  1733958894,  539903628,
      -1256659915, 598594948,   473072260,   1374346415,  -1193582215,
      -359977027,  440082987,   1897731123,  -1914082250, 1500723546,
      -1819102966, 7258570,     -1044946551, 1693140649,  1206503204,
      -1426859594, 100090249,   -2043329067, 2030156969,  -192611352,
      -1432232008, 2096255083,  238828067,   1211978160,  -413885129,
      -683216726,  1086603235,  1009187808,  1875708538,  -157314879,
      -1448969836, -1505514391, 1817605606,  -684937330,  -1375074680,
      329785765,   -1731044947, 130975995,   -222087819,  437991046,
      1864118428,  1768440710,  -748848591,  -797756480,  -741574502,
      2056346595,  1329304878,  -1753751497, -1128735815, 159043197,
      2036548120,  445129530,   1408500530,  85135260,    -1674879919,
      -1518961516, -1095537294, 172373817,   1992999288,  -190451320,
      356133226,   -192729208,  -726950211,  -1164438744, -2081041538,
      115928535,   -768735196,  640095102,   -792568569,  1490893552,
      325753205,   -475702741,  1503442907,  1377580734,  -1564359272,
      -962346289,  108423974,   -2135653630, -1771460954, 1868374403,
      1852829859,  1488371717,  1707085559,  -578126640,  95696484,
      -936342501,  251030768,   1682758375,  -969823889,  -921723273,
      1876723735,  -223265192,  -628164027,  -1665313946, -956471335,
      -431068610,  -1686696786, -721288314,  54360124,    198874299,
      677952055,   -1125684071, -692879097,  1512092842,  32389719,
      267958429,   -1917798911, -1742347136, 1542219498,  -1989444232,
      -649291593,  -669038567,  -1907231190, -981145254,  -1534116787,
      -1710792494, -46820031,   1749129660,  285697641,   550381289,
      800321840,   -2048266501, 834573811,   -869177549,  1839701214,
      -1093517492, 1164159543,  521174769,   -788112871,  904347720,
      -1277494227, -390650363,  1430735305,  -1656983550, -1217620559,
      870043301,   1631480001,  2039054338,  -558735759,  -1654461956,
      -679330778,  893134654,   1154550863,  238633435,   176316914,
      -1850309593, -2039416623, 761322514,   502883938,   1551268968,
      357694508,   -866840931,  179663904,   -1508587816, -1468030419,
      99302147,    658753277,   -17352193,   -382074998,  -1127285511,
      -344300017,  -1029045260, -1008186234, -1170172282, 1194869681,
      -232986836,  103189083,   -482599036,  -972221107,  1440983785,
      35070582,    -1167323232, -1607699148, 1138121690,  1749819123,
      -474606223,  -1514491125, -293070607,  -1139860310, -1101775152,
      235591952,   404394701,   -1716605831, -863649197,  1925218426,
      1385168179,  -1712463439, -625665641,  242287398,   -1428595177,
      -1224344993, 183208614,   667127401,   -899201863,  1413490145,
      1271939635,  -1454260871, 1710308327,  1015836254,  1535289595,
      -1014300504, -160795843,  1753377965,  1603588180,  1656367083,
      831743063,   724192728,   86604350,    1349631021,  327097040,
      -1350581092, 425621532,   -1108049828, 1635002256,  1450435201,
      1667513178,  -1130890274, -2143348819, -903217164,  1633967793,
      -1700997501, 1618509033,  -2062608691, 1300074999,  -1568200659,
      1898850166,  -16393428,   1637665408,  -715288951,  -1583358056,
      -1190114003, 931491804,   -1201460518, -1691835876, -770233908,
      1473930813,  -1880488608, -680889542,  283136356,   310267102,
      -479921026,  -1924066032, 703660525,   -512981488,  1613413367,
      -2071365416, -311744811,  853645461,   758193213,   628907571,
      1475998831,  -2133835758, 321510813,   1149134034,  1284238388,
      1187476200,  1243032063,  25434447,    143469701,   -1480771113,
      1738944607,  -766137016,  145802670,   1883225105,  283273272,
      473324308,   152212852,   586643368,   644749549,   1768704670,
      1637557762,  -1153159032, -1073753713, -173023836,  -46957095,
      1034491832,  -2062528541, -1322569588, 1480687551,  -6668776,
      -1019742691, 1860978161,  -1912246424, 1034915409,  1006432861,
      -810644825,  757064299,   2084451311,  -655326028,  -588778997,
      1726858283,  848272199,   -307219651,  1035032319,  330479509,
      -784999158,  831474902,   483058293,   -295490447,  -58961909,
      93922996,    295256946,   -2003489136, 1662208452,  -1671813823,
      -118598998,  -407546809,  249004240,   1473373904,  -1598362074,
      -1257259912, 1511854890,  -1041712859, -146314573,  928132296,
      1501292629,  -2044763087, -253233377,  835616194,   -103916411,
      1758351217,  930497334,   -540310951,  -993170707,  -1622855609,
      -1332047319, 1542974113,  -1189909578, -198804513,  -430274968,
      438216919,   2007361177,  1166467128,  -1787798031, -1643357049,
      1331557600,  -1847846079, 1303821377,  -2127251278, 1034515709,
      1964022829,  -2121029334, -902406526,  -1286097053, -697353822,
      1318523485,  542269881,   -283439786,  -1891852076, -1935027647,
      596070718,   -1505191151, -770506689,  1550954878,  13162528,
      1236829643,  -1557372582, 1743547680,  269775235,   -1603511743,
      -1080311659, 1626623870,  -815939227,  -177390342,  -879434078,
      -70734193,   545827914,   654604588,   -397928622,  174522980,
      -1557831623, 1555382192,  -1360542489, 1723918711,  662728290,
      1827473532,  -1909160716, 177146754,   817990501,   1265171009,
      -1426396594, -1037823233, 1160202007,  -281806878,  15842961,
      171204785,   519727382,   -2106458416, 1963175894,  1140532127,
      689718896,   796382763,   -466901686,  1453733380,  1459609799,
      1873728287,  1239855417,  -1768885321, -1277283127, -353147259,
      1081497585,  1840692025,  1896251377,  681627869,   1770889047,
      -1799530879, 1924150436,  -609749747,  -96154205,   -810158379,
      1178803523,  -1915061984, -86324829,   146863951,   -780702378,
      -1294152150, 1224388426,  -1114661052, -2052968099, 340954725,
      670835696,   1338107283,  -1836123491, 1781624955,  -376258364,
      803360497,   -142647958,  -215040139,  373006339,   116094277,
      -2023650905, -1058111295, 1968998569,  1531365378,  -1881423408,
      665395157,   -1827709322, 1412298172,  -1578067222, 1580590296,
      -929383563,  1376947741,  -823243675,  -335822224,  -1589576927,
      -1276558627, -1711269981, 786224958,   -828910213,  100078456,
      629999129,   1111032071,  -1726956374, 639666286,   -1146946884,
      282590783,   -1357630191, -2099258596, -1711733191, 757999190,
      264796904,   -1214326732, -1252628052, -1528863805, -783046880,
      -58546208,   -1487065820, 1772989504,  -635009410,  746459774,
      -1315363464, -710133715,  -410363520,  650255922,   1151773197,
      1980577753,  455043436,   -1726843054, 404446410,   1800690667,
      1646252011,  578555443,   1158896878,  640663680,   129268829,
      -119299345,  -1678738061, 80283867,    167043197,   -2039278119,
      1789902282,  -1739856860, 790369492,   1348522610,  317660016,
      -931824778,  -1861958601, 1053833874,  1146767164,  -1631359782,
      -1361678907, -574546690,  -7437465,    -1880137207, -802214419,
      -674143360,  918315741,   -1980822480, -1625543931, -229252535,
      1789287302,  1126756399,  -363677381,  1473165027,  -1890280964,
      -895340275,  2056101458,  1482101507,  -1096266011, 1590338875,
      514478535,   -779436299,  318498593,   -140299459,  -936765221,
      -1095814047, 1646739916,  -887720152,  -217347709,  804272029,
      952408228,   -95204425,   -1373085191, -1718160816, 1971220053,
      1483591357,  1530738623,  -1037218125, 1434742834,  -1869563139,
      1378822553,  -493168865,  8001934,     -688655295,  -1920301420,
      31139480,    -1964969136, 489572710,   2136001869,  -1392295066,
      -1454206737, 2078732678,  -1579699382, 1391561372,  255974639,
      -1328470719, -188679552,  629474101,   -1450135748, -1925249738,
      -1745058683, 289616364,   -1992314577, 965925922,   -1685448615,
      431609054,   -532761640,  -279323476,  988076288,   2012797068,
      1708833387,  -484057763,  -447130978,  -1829829766, 1356826079,
      -2012254352, -1332872636, -110708680,  -2044397172, 177446570,
      497365355,   1887612484,  -1763362051, -1857377604, 1701482574,
      -2073940592, -824806860,  -1669689579, -347661061,  760998261,
      678631226,   -894367667,  1869683141,  -149327580,  -1877903400,
      25775682,    -794832886,  -1530949618, 1213695763,  1278732557,
      -448486481,  1675283292,  2051453324,  1643938067,  835660384,
      -1480022453, -640205751,  -548734590,  603263370,   1204512091,
      -1996735436, -1062008945, -1099027210, 1131761539,  -898020918,
      1414487205,  -1564001110, -2056497007, -1863367517, 351010316,
      -2130580341, -376974436,  1838584116,  -202414395,  -1550829782,
      -1828876688, 509873382,   -1666438342, 1449756595,  -609370887,
      1083910198,  -1696164703, 286213706,   1659072204,  1119360733,
      1141534212,  275031281,   -1949719450, -492769420,  -560903220,
      1399658371,  -371181132,  -784168752,  -1918444998, 228543543,
      -247171466,  -1439203975, 1534012206,  -1316071219, -1411882307,
      -2036477855, -1046198347, -1107612590, 346283373,   1499964902,
      -76006247,   654147629,   -1283368582, -220691733,  -1413436933,
      1806972541,  1296335176,  -766141952,  -669739485,  -1879493461,
      1163695984,  -2029941278, -1529694340, -39566437,   -928056446,
      -554048082,  -1945106614, -1763502683, 1136424910,  -2047793523,
      -372986956,  246147357,   1358592655,  797588562,   -1515987462,
      -2016229165, -990547907,  1827917068,  -42092114,   1590801962,
      841332426,   -1606153724, 522568318,   1052582871,  1282871739,
      1070660138,  952836205,   525763257,   1938565983,  1832139598,
      2054819414,  261307202,   1233562642,  -555216196,  -1981729092,
      -987476068,  -1572030262, 1954978379,  -742779817,  1971950985,
      746411662,   1944038837,  102269832,   -1140373092, 1480044254,
      813658979,   147502446,   -105807600,  -1016539756, 680665855,
      -1825485210, 2131150390,  251344037,   -862539009,  1234329186,
      -1363900526, 988642555,   -1398182749, -1984630687, 1527429845,
      -1243811569, 567909282,   -831000425,  1271483794,  785103218,
      -827388913,  337511540,   2129890143,  298070135,   1930242470,
      -101777932,  2008595252,  1975685919,  -574507187,  841575443,
      1519441096,  1311155908,  -1533839981, 1006432248,  1848959396,
      -1760468670, -2059839898, -1773541088, -1399307327, 1509455310,
      -594435349,  -384450823,  1558880105,  -145792537,  209200150,
      840275001,   -1222676950, -1898758990, 568130613,   1053805329,
      1543501878,  -968525161,  -2115166036, 1082188154,  510175840,
      -1081526732, -155841972,  213385486,   1202665222,  715953660,
      -1550641400, 431477735,   649293921,   2049180347,  -951150001,
      -1329683395, -1839441974, 1199374729,  -458583983,  -490438011,
      -213630031,  -449917590,  -1589519658, -681971001,  -1015182898,
      -1025883071, 1059437351,  180635551,   -858559507,  -1735503827,
      -1903890727, -1237799823, -1893550131, 1010275803,  1128126974,
      -200296622,  -1752954349, -185982006,  -575958004,  848767667,
      1150149750,  1402649005,  229476114,   -934748229,  1632776848,
      -873715925,  237098434,   -363426116,  -933274608,  -2063535651,
      -2050614431, -24015219,   -564677201,  -1281491640, -1866947583,
      1945671301,  -2066133900, 522166558,   -160145305,  -1993846319,
      -1259175696, 267706645,   345372643,   1664739059,  -1396733549,
      306731311,   661016683,   -2033894311, -737051393,  264315224,
      1922793407,  -424337502,  1560670266,  -1857821438, 1508675372,
      1583142977,  -1680231826, 1800919393,  436788060,   1516383450,
      1378067068,  192199781,   -648704246,  1798596870,  1562411144,
      -123687308,  -85622304,   -2119211925, 1368512783,  -249498654,
      805730480,   1830060122,  1351258452,  201532855,   -1838250186,
      -1961438365, 618983927,   1203428541,  1887206397,  -1953010281,
      -1151903887, 374328274,   -1044586159, 1547969123,  314878211,
      -1301469511, 129763185,   1982561817,  -1505544846, -1499793041,
      139976419,   1253749470,  522400020,   2039329006,  844261956,
      547438980,   1098400887,  -1664112685, -1704299371, 896458881,
      -696627704,  -504128161,  1542772347,  1360719562,  1076359410,
      -860987391,  337837698,   -1686706772, -696428846,  -1730002773,
      -2039192034, 2127534839,  1138439590,  -1537046233, 677849503,
      2127034046,  2033804393,  -1009322526, -376705947,  439809784,
      -1451570306, 1437149293,  -92676564,   -625117748,  1623952955,
      -1421709564, 666202936,   2046771918,  229783621,   253148252,
      -831570657,  977525406,   -2057306124, -887088483,  -481032281,
      654148483,   1563178354,  233767377,   -2054249542, 422128640,
      1119228272,  695710610,   168526754,   714418326,   -1076412818,
      -1428036385, -861399926,  -410848859,  -1987695750, 1359015638,
      1685558721,  -632279054,  1082938717,  788650768,   1559192918,
      1967873106,  2066671622,  -1552589050, 2130210028,  -1489059299,
      493322165,   -293649897,  810368358,   -1425021954, 2054874665,
      558345727,   -1018392703, 1641708308,  -1851607392, 1164447297,
      2024415059,  315586835,   1648874243,  1487158692,  -536432952,
      -2110709974, 223042167,   -1675195048, -572339424,  -391166293,
      307417809,   -1960273389, -2018811586, 924671917,   1775355875,
      -1544267097, -1387597973, 1829534545,  870580435,   200290626,
      -1845376094, -1177028060, 765772086,   -776729434,  -1543327302,
      -89306214,   1162305826,  1968073083,  1418905577,  -634505161,
      2055014289,  2134296468,  1617761231,  2076488671,  1712631756,
      1984086641,  607829337,   874154351,   -810310565,  1446809432,
      -574263022,  2110458878,  1198640253,  1845206541,  1147505024,
      1865938993,  -2016660996, 1992479416,  -1708403815, -1317083057,
      -80025175,   -1509698054, 18394801,    499804196,   1400474653,
      1566265558,  -58497013,   596429151,   2040957784,  -1825063150,
      921444019,   -1983892719, -323415067,  -2140538585, 28878393,
      1785398192,  1595885702,  772662049,   223018243,   1364883722,
      -576448354,  -1438059802, -68907257,   1653390192,  -987106363,
      1431859140,  -290349718,  1400281059,  -118319776,  -1200479348,
      279577322,   207132974,   -833037015,  1392754893,  -1624365917,
      -1987211244, -1164145577, -717616536,  -2031568610, 1280747504,
      -1338589534, 1122875850,  -1325711715, 1109442098,  -159141924,
      851104404,   178099649,   -1945822207, -976786168,  -1748721320,
      -688476447,  -398094231,  -268071204,  170204657,   -1445193344,
      547019098,   -1000794293, -926356443,  1010960454,  -1761026265,
      -1975497728, 1190032372,  -1193163195, 344939898,   -568964045,
      -907209272,  1615876228,  -87376659,   141965825,   460360135,
      -896204458,  622145638,   1701859119,  1547350014,  2132668097,
      -76283705,   86451890,    1299900973,  876668762,   749228657,
      -678480109,  199003,      1884777952,  1115748864,  1995692412,
      2051733492,  1519227986,  538063799,   -375447290,  630612630,
      -524017317,  692443698,   976778717,   -1660434583, 958626531,
      719020338,   682079927,   -649509673,  -1169627901, 1007310092,
      -2126338989, -1287556706, 1256784522,  1506672005,  -1782063381,
      -1165371175, 1233387033,  2099652261,  -1364163619, 1960706956,
      2110926437,  311693694,   -951297701,  -750450561,  1402241065,
      -1719837417, 1357437076,  -866086717,  -2008249930, 1628368281,
      1883341959,  -1437225115, 1897736145,  -109481574,  1555155275,
      904659305,   145882565,   -1147099139, -1392461349, -153355410,
      1232157005,  1817210555,  1590298950,  -331311696,  1078493065,
      695687286,   -977815239,  603242818,   -1776737831, 1903515975,
      710389509,   -1039525684, 1537430837,  -1320728367, -269276637,
      510113187,   -1522771446, -629890120,  -1371758735, -1688291559,
      802286037,   -1832274914, -718173834,  262104679,   -1617195398,
      1227900013,  -1814868103, -1870396182, -1984797451, 1865047749,
      892652491,   -830838737,  911479810,   -284240454,  -106081988,
      -1086780558, -48049386,   917530691,   1520755475,  1437483872,
      -2139499613, 459315574,   -671137498,  -1140993229, -553570860,
      919155455,   -37728027,   564404258,   1704800483,  468317692,
      -1610887820, 1572070163,  70198270,    -1357673021, 835538506,
      -564342591,  1762093506,  -304176072,  37823074,    1715449106,
      -1534641855, -1330319480, -1017160974, 1336525151,  1544874171,
      -683446969,  190028510,   -591939479,  -2142316914, 1029467429,
      1463798920,  -635949826,  -1910744541, 216478511,   1952871363,
      -183434089,  2094275798,  1613665425,  -1087248900, 1878058032,
      1640520712,  1744826937,  1312171532,  2029300439,  -873506481,
      -611774328,  -551988043,  294905162,   1533445876,  1045479587,
      2047278523,  -387409789,  986621056,   491910798,   -1885765121,
      1495211968,  587921974,   1020782933,  -1050959996, -429327434,
      539156270,   1429259714,  -1141224969, -178508755,  1904454680,
      1755348453,  -785664846,  1002199013,  1602267222,  -721835517,
      1365562193,  1592428253,  1751818104,  -1211846016, 843194944,
      -366456639,  -981887651,  524958185,   -1304065764, -963285891,
      -1353570452, 744923528,   -1353901491, 1174983887,  1935505863,
      140224990,   1030068367,  -318795758,  616756553,   -341465262,
      -1859912480, 1388843302,  1963099021,  -90044492,   519604835,
      -1608714796, -272116737,  -1619217870, -1292438012, 443686701,
      391340538,   2070968489,  -43760533,   1119420439,  1112541969,
      -1274311317, 1219738959,  -1731215253, -2067492916, 823856190,
      1356605176,  1412639978,  -780574053,  872558594,   -267230306,
      -982757001,  1186053312,  312785370,   2080233235,  -1392242215,
      -1023037999, 1593631362,  1249633360,  1231629039,  225303661,
      -1771783897, 1958537913,  -200887310,  1702028339,  1833083097,
      -1859015572, -1030029513, -1716145836, 66644637,    -329287584,
      -191224028,  713903888,   1203485233,  1019405230,  -267560913,
      2012102761,  -601802096,  -1357907260, -1103841728, -872589797,
      193379455,   1128069956,  934233057,   1958601427,  -1588451716,
      238533793,   918778919,   1503420693,  -642453533,  1489315483,
      2000583026,  -353086361,  81149068,    1538331,     1682876443,
      -709811181,  -575150723,  -1049304195, 388015012,   -384896090,
      -1734812557, -1830288134, -1137297584, -585985214,  -2052792447,
      -1181508375, -241201492,  -867950328,  -736308707,  -1878787586,
      1756629699,  617580932,   -1229222250, 849183379,   -2070989616,
      274968042,   -709832424,  -375008678,  1804085339,  1509421161,
      -731278687,  -128581776,  2671517,     523271376,   1610964744,
      -1742154257, 871174915,   -1015815609, 766172450,   -643321143,
      -767046192,  1580411955,  1367948698,  1381638167,  269680687,
      -1576576256, -88104624,   -1606733411, -1827555229, -573372679,
      1731878517,  2067857498,  -1000179070, 262750572,   854559696,
      436721218,   -1219635701, -293024252,  -1799093409, -1297793321,
      2138748601,  878009193,   1776663954,  -466351679,  -687396771,
      -1693069580, 245515570,   -85462503,   86942763,    216782494,
      996689137,   -584407695,  -1380066077, -1111006245, -183412061,
      1493565027,  641879349,   1236611935,  695615383,   -1684921932,
      691061922,   1817524640,  -2021567393, 1082750083,  -136815062,
      631502546,   1387303199,  -641525164,  214743742,   1054569424,
      -1705181202, -642834113,  1505988400,  -119383005,  -1376165602,
      -1732354148, 349410369,   -1126933224, 1961359903,  -1818987166,
      873291126,   -1659246046, -1643884628, 1590225987,  2121881298,
      -1872310908, 1202806103,  1743682596,  727440209,   -205346577,
      -82397809,   -1395751751, 88308906,    -1868762966, 727465344,
      1634691271,  -680332969,  -640754459,  -787056835,  125350494,
      -848278650,  1179670975,  -1794383119, 1691715512,  2123857377,
      -1570628510, 2010753574,  -898210848,  -638873918,  163492552,
      863195181,   262444947,   -1981803938, -691458322,  1292775538,
      -1590158119, -1269562586, -2012599121, -1032640200, -1549865867,
      -1407439422, -1792607308, -2010152701, 443587276,   1952610005,
      -228873507,  73538545,    2050617478,  -1884397703, -619644971,
      -1080974265, 103055653,   -254436940,  -1438301447, 2065442606,
      -709904764,  503323457,   -1596937173, -33781559,   754267144,
      182556575,   -503778761,  1060770140,  -2100429435, -1712871381,
      -60870649,   436061125,   -1918235599, -941189760,  544679514,
      1116981949,  1847248004,  382000704,   770023720,   368009005,
      -1510444621, 1436631216,  -262786802,  610531861,   -1511225211,
      967465873,   -749441958,  -1578057408, -909824691,  -1368363831,
      -1979707778, 907092572,   88380918,    -417283659,  1442941626,
      76225551,    -482683582,  645739981,   -512078063,  1717152676,
      -1478376468, -1368301382, 570620427,   -727544949,  893921508,
      -1481468514, 915244402,   -600364763,  931773328,   510839902,
      -749546663,  -225688366,  1909939386,  -307630533,  866612807,
      -1076526450, -1854744511, 878155251,   -2077317595, 244486383,
      -488648409,  -1215242732, -1688605886, -1821353493, -664066649,
      1304088192,  -1425015366, -1053148040, -1579158571, 1804451308,
      -1764207697, 207324753,   -1140890928, 917029866,   1556711334,
      821841537,   1833817729,  1574699699,  1495739773,  -1423997862,
      385002926,   1679015490,  -1188045371, 345687832,   -342276979,
      97038230,    -1204821850, 438479984,   807137610,   -574762891,
      -708495404,  863522675,   -310198760,  -40544228,   794825938,
      739357306,   -638587149,  -2001241626, -750798151,  974285039,
      1381401808,  1215264189,  -293494322,  -551998249,  433796066,
      1961113305,  1642409717,  794522778,   -1972189310, -929697739,
      -438476286,  805927932,   1796873933,  729177461,   180887338,
      7253648,     -1508369787, 296909594,   1772853687,  1931559877,
      1526684043,  -684346573,  -1756367801, -2000822497, -1049623274,
      -183680554,  -717401992,  -1005387589, -304397965,  1984269551,
      1368572861,  -887417689,  101302974,   1187118753,  -1669563641,
      847497588,   982456230,   -139069496,  2080647156,  1796659917,
      -574878363,  757441309,   -989049344,  1237641626,  636858666,
      -6876607,    489722474,   443625324,   217258293,   11885820,
      1330846591,  1121396970,  -1104644178, 340306424,   1524641407,
      -46165795,   -2039612114, -81118781,   2069467439,  1273159059,
      -1674254407, -724631015,  1859026317,  -225931627,  100554332,
      -1306601686, 722675699,   -1540802288, -839759213,  -1854588667,
      -579629553,  -1006616418, 1484798819,  92666916,    840609042,
      919189202,   1774180292,  -592928583,  -568165969,  -1066133854,
      -1559764485, -1592800799, 674997753,   304013645,   1960931467,
      1818586579,  -857734996,  -1512496516, -1216094419, 821779002,
      -171724816,  819405821,   -23759974,   2031703692,  -1890838356,
      -706516469,  1329233068,  -1917284258, 1890219814,  552183632,
      -118798617,  -88119172,   950310939,   -1023477503, 1728101416,
      220342563,   567634791,   1935279277,  -703800118,  1705380207,
      1398895426,  -1561218903, 1632969705,  1121458292,  -297253719,
      -503830384,  1501900755,  939204812,   782063199,   -1212299621,
      -407333597,  419900909,   -195960478,  -824012230,  -1297114899,
      -1773557642, -1294709522, 935908121,   200756833,   -268668404,
      1860212243,  -430655245,  1807315294,  117901623,   -1725525807,
      -1440507930, -2045951437, -1843881296, -1222455135, -623038434,
      -844399234,  1644563298,  437620973,   -2145162894, -178183162,
      1499955835,  135027526,   1681200415,  213201359,   2043791593,
      520934558,   1743940445,  -1938030245, 857819009,   -1993627500,
      -1470119369, 1998395815,  -143734899,  871701209,   618861681,
      -608773969,  1423299075,  -1876721858, 478956812,   117228456,
      1273331023,  171457655,   -967160509,  1912997635,  -1867738472,
      -195458999,  877837737,   170373380,   -706781538,  569508365,
      -1213473175, 1217001138,  -401231428,  -19991150,   1477842903,
      83925619,    -249904218,  88989008,    2125598748,  1867592452,
      2021931042,  -140215152,  -820082332,  334422651,   -323790498,
      -166194516,  -892703651,  1929267862,  -602641688,  372617465,
      903435727,   2099021134,  -856093551,  -1400416087, -1940754523,
      502725199,   916877545,   -1192990770, 251944660,   -32487715,
      319493062,   -240214028,  1815791192,  -84788752,   -1173209617,
      498870299,   -642154045,  -1376663449, 2069125932,  -2078959741,
      -296644476,  -531080498,  265156868,   -394527777,  1126570771,
      27099428,    1466754380,  -767171680,  1626114156,  338673844,
      862680424,   -951874295,  -1804285111, 443787887,   -642995542,
      -1602263276, -1878218314, 388061817,   -2074913717, 153126267,
      -1051725971, 377202580,   -569148155,  1925599524,  -417806528,
      1370287530,  1426629226,  486447761,   -1494880882, -987746340,
      1830380844,  165066727,   -1214613957, -424599451,  -826029561,
      993474787,   -1918718843, 1179068291,  -621466817,  1331906058,
      839713600,   -887136035,  1511573539,  -811316574,  -147057458,
      1127827430,  1693800057,  5171522,     635115646,   68540118,
      1101336531,  2050727770,  -1967237201, -1357696563, -1022277360,
      1117253084,  -1768687237, 1462328601,  -1257664649, 1314151854,
      991069019,   298070501,   1117460211,  -902927179,  1904038406,
      1717659038,  1454171015,  -1944028584, -1094678071, 1432605240,
      -432134547,  1139684117,  1394232148,  665809317,   -712052411,
      -1695606364, 1216694177,  1407029636,  -470589698,  1564040479,
      1731612086,  32626166,    164497444,   1544117467,  1988313169,
      -54323649,   1010378261,  1782330866,  570028872,   -584503049,
      -752958765,  2045888260,  -108595088,  -1284842062, -568240609,
      2089450354,  -1869588404, 2027249516,  248237878,   -1322190949,
      913207754,   307174439,   66433244,    -350470495,  1655690934,
      799855188,   -1050925857, 1688709391,  -1824515381, -1048030486,
      1655516623,  155881159,   287371582,   733412585,   1813795486,
      -1586294863, -1741895863, 1039833363,  346435612,   1298017188,
      1241184189,  -835298978,  -811221506,  -1774240722, 2133859176,
      200033787,   -1083514044, -55883138,   725392997,   277130385,
      1433032516,  1464483380,  1922066111,  1011975787,  1245304074,
      1239135008,  -1478254974, 1915681428,  690270312,   -906661289,
      845995753,   -1299472117, -1387248523, -267374628,  -947028795,
      -1116026039, -870073970,  257399645,   481973248,   744326777,
      -1360230129, -1318473068, -1494281388, -862107430,  1742555881,
      -679280587,  -1429482331, -194471500,  -792535046,  1404826756,
      1901413586,  -383229446,  -607817927,  1411859838,  -2083845976,
      583932582,   -1424657789, -1274869682, -1816939300, -1805229528,
      -1168735148, -1304142245, 809807823,   1027627534,  -139640918,
      1622215985,  -1102499366, 500707281,   -1856082246, -1504546665,
      -1109824330, 157639490,   1826335860,  1860435417,  1896971050,
      693610694,   964196140,   1530363177,  -575421950,  1412027350,
      690787132,   1679100168,  268850627,   -1286218406, 1898432215,
      -8475797,    -2138195629};
#undef X_ref_size

  const float post_activation_multiplier_original[chans_out] = {
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583, 0.9995002150535583, 0.9995002150535583,
      0.9995002150535583};
  const float post_activation_bias_original[chans_out] = {
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
      0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0};

  float output_scale = 0.0117647061124444;
  float output_zero_point = 42;
  float backtransform_add = receptive_volume;

  float post_activation_multiplier[chans_out];
  float post_activation_bias[chans_out];
  for (int j = 0; j < chans_out; j++) {
    const float post_mul = post_activation_multiplier_original[j];
    const float post_bias = post_activation_bias_original[j];
    post_activation_multiplier[j] = -1 * post_mul / output_scale;
    post_activation_bias[j] =
        (post_bias + backtransform_add * post_mul) / output_scale +
        output_zero_point;
  }

  int *chan_overlaps = (int *)malloc(sizeof(int) * (chans_out));
  int16_t *post_activation_multiplier_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));
  int16_t *post_activation_bias_q =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int16_t *quantised_accu_modifier =
      (int16_t *)malloc(sizeof(int16_t) * (chans_out + (16 - chans_out % 16)));

  int8_t *Y = (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);
  int8_t *Y_ref =
      (int8_t *)malloc(sizeof(int8_t) * y_height * y_width * chans_out);

  assert(X_ref);
  assert(Y);
  assert(Y_ref);
  assert(post_activation_multiplier_q);
  assert(post_activation_bias_q);
  assert(quantised_accu_modifier);
  assert(K);
  assert(K_ref);

  assert(post_activation_multiplier);
  assert(post_activation_bias);
  assert(chan_overlaps);

  int32_t larq_clamp_min = 0;
  int32_t larq_clamp_max = receptive_volume;

  run_int8_config(
      (int8_t *)Y, (int8_t *)Y_ref, (bnn_b32_t *)X_ref, (bnn_b32_t *)K,
      (bnn_b32_t *)K_ref, (float *)post_activation_multiplier,
      (float *)post_activation_bias, (int16_t *)post_activation_multiplier_q,
      (int16_t *)post_activation_bias_q,

      (int16_t *)quantised_accu_modifier,

      (int *)chan_overlaps, x_height, x_width, k_height, k_width, chans_in,
      chans_out, h_stride, v_stride, larq_clamp_min, larq_clamp_max, 0,
      chans_out, valid_impl);

  free(Y);
  free(Y_ref);
  free(post_activation_multiplier_q);
  free(post_activation_bias_q);
  free(quantised_accu_modifier);
  free(K);
  free(chan_overlaps);
}

#undef h_stride
#undef v_stride
#undef k_height
#undef k_width
#undef x_height
#undef x_width
#undef y_height
#undef y_width
#undef chans_in
#undef chans_out
#undef chan_words_in
#undef K_ref_size

static void generic_kernel_subregion(
    int8_t *Y_p, const bnn_b32_t *X_p, const bnn_b32_t *K_p,

    int16_t *post_activation_multiplier_q, int16_t *post_activation_bias_q,

    const int16_t *quantised_accu_modifier, const int16_t clamp_near,
    const int16_t clamp_far_0, const int16_t clamp_far_1,

    const int accu_shr, const int16_t bias_multiplier, const int16_t final_shr,

    const nn_image_params_t *x, const nn_image_params_t *y,
    const nn_window_params_t *k, unsigned y_loc_width, unsigned y_loc_height,
    unsigned y_sub_width, unsigned y_sub_height, unsigned y_loc_channel,
    unsigned y_sub_channel) {
  bnn_b32_t *data_scratch = (bnn_b32_t *)malloc(
      sizeof(bnn_b32_t) * (k->shape.height * k->shape.width * x->channels / 32 +
                           DATA_SCRATCH_OVERREADWRITE_WORDS));

  output_transform_values_t otv;

  bnn_populate_output_transform_values(&otv, clamp_near, clamp_far_0,
                                       clamp_far_1, accu_shr, bias_multiplier,
                                       final_shr);

  bconv2d_int8_valid(Y_p, X_p, K_p, post_activation_multiplier_q,
                     post_activation_bias_q, quantised_accu_modifier, &otv,
                     data_scratch, x, y, k, y_loc_width, y_loc_height,
                     y_sub_width, y_sub_height, y_loc_channel, y_sub_channel);
  free(data_scratch);
}

static void DIDO_kernel_subregion(
    int8_t *Y_p, const bnn_b32_t *X_p, const bnn_b32_t *K_p,

    int16_t *post_activation_multiplier_q, int16_t *post_activation_bias_q,

    const int16_t *quantised_accu_modifier, const int16_t clamp_near,
    const int16_t clamp_far_0, const int16_t clamp_far_1,

    const int accu_shr, const int16_t bias_multiplier, const int16_t final_shr,

    const nn_image_params_t *x, const nn_image_params_t *y,
    const nn_window_params_t *k, unsigned y_loc_width, unsigned y_loc_height,
    unsigned y_sub_width, unsigned y_sub_height, unsigned y_loc_channel,
    unsigned y_sub_channel) {
  output_transform_values_t otv;

  bnn_populate_output_transform_values(&otv, clamp_near, clamp_far_0,
                                       clamp_far_1, accu_shr, bias_multiplier,
                                       final_shr);

  bconv2d_int8_DIDO_valid(Y_p, (const bnn_b256_t *)X_p, (const bnn_b256_t *)K_p,
                          post_activation_multiplier_q, post_activation_bias_q,
                          &otv, x, y, k, y_loc_width, y_loc_height, y_sub_width,
                          y_sub_height, y_loc_channel, y_sub_channel);
}

static void generic_kernel_full_image(
    int8_t *Y_p, const bnn_b32_t *X_p, const bnn_b32_t *K_p,

    int16_t *post_activation_multiplier_q, int16_t *post_activation_bias_q,

    const int16_t *quantised_accu_modifier, const int16_t clamp_near,
    const int16_t clamp_far_0, const int16_t clamp_far_1,

    const int accu_shr, const int16_t bias_multiplier, const int16_t final_shr,

    const nn_image_params_t *x, const nn_image_params_t *y,
    const nn_window_params_t *k, unsigned y_loc_channel,
    unsigned y_sub_channel) {
  bnn_b32_t *data_scratch = (bnn_b32_t *)malloc(
      sizeof(bnn_b32_t) * (k->shape.height * k->shape.width * x->channels / 32 +
                           DATA_SCRATCH_OVERREADWRITE_WORDS));

  output_transform_values_t otv;

  bnn_populate_output_transform_values(&otv, clamp_near, clamp_far_0,
                                       clamp_far_1, accu_shr, bias_multiplier,
                                       final_shr);

  bconv2d_int8(Y_p, X_p, K_p, post_activation_multiplier_q,
               post_activation_bias_q, quantised_accu_modifier, &otv,
               data_scratch, x, y, k, 0, 0, y->width, y->height, 0, 0,
               y_loc_channel, y_sub_channel);
  free(data_scratch);
}

static void DIDO_kernel_full_image(
    int8_t *Y_p, const bnn_b32_t *X_p, const bnn_b32_t *K_p,

    int16_t *post_activation_multiplier_q, int16_t *post_activation_bias_q,

    const int16_t *quantised_accu_modifier, const int16_t clamp_near,
    const int16_t clamp_far_0, const int16_t clamp_far_1,

    const int accu_shr, const int16_t bias_multiplier, const int16_t final_shr,

    const nn_image_params_t *x, const nn_image_params_t *y,
    const nn_window_params_t *k, unsigned y_loc_channel,
    unsigned y_sub_channel) {
  output_transform_values_t otv;

  bnn_populate_output_transform_values(&otv, clamp_near, clamp_far_0,
                                       clamp_far_1, accu_shr, bias_multiplier,
                                       final_shr);

  bconv2d_int8_DIDO(Y_p, (const bnn_b256_t *)X_p, (const bnn_b256_t *)K_p,
                    post_activation_multiplier_q, post_activation_bias_q, &otv,
                    x, y, k, 0, 0, y->width, y->height, 0, 0, y_loc_channel,
                    y_sub_channel);
}

void test_bconv2d_int8_sub_image() {
  impl_bconv2d_int8_sub_image(5, 5, 3, 3, 32 * 1, 32 * 9, 4 * 1, 4 * 5, 32, 4,
                              1, 3, 1, 3, (void *)&generic_kernel_subregion);
}

void test_bconv2d_int8_DIDO_sub_image() {
  impl_bconv2d_int8_sub_image(5, 5, 3, 3, 256 * 1, 256 * 2, 16 * 1, 16 * 3, 256,
                              32, 1, 3, 1, 3, (void *)&DIDO_kernel_subregion);
}

void test_bconv2d_int8_pseudo_random() {
  impl_bconv2d_int8_pseudo_random(1, 5, 1, 5, 32 * 1, 32 * 9, 4 * 1, 4 * 5, 32,
                                  4, 1, 3, 1, 3,
                                  (void *)&generic_kernel_full_image);
}

void test_bconv2d_int8_DIDO_pseudo_random() {
  impl_bconv2d_int8_pseudo_random(1, 4, 1, 4, 256 * 1, 256 * 2, 32 * 1, 32 * 3,
                                  256, 32, 1, 3, 1, 3,
                                  (void *)&DIDO_kernel_full_image);
}

void test_bconv2d_int8_pseudo_random2() {
  impl_bconv2d_int8_pseudo_random2(1, 32, 32, 4, 4, 32, 4,
                                   (void *)&generic_kernel_full_image);
}

void test_bconv2d_int8_DIDO_pseudo_random2() {
  impl_bconv2d_int8_pseudo_random2(1, 32, 256, 32, 32, 256, 32,
                                   (void *)&DIDO_kernel_full_image);
}

void test_bconv2d_int8_directed() {
  impl_bconv2d_int8_directed((void *)&generic_kernel_full_image);
}
void test_bconv2d_int8_directed2() {
  impl_bconv2d_int8_directed2((void *)&generic_kernel_full_image);
}

void test_bconv2d_int8_directed3() {
  impl_bconv2d_int8_directed3((void *)&generic_kernel_full_image);
}
void test_bconv2d_int8_directed4() {
  impl_bconv2d_int8_directed4((void *)&generic_kernel_full_image);
}

void test_bnn_conv2d_int8() {
  UNITY_SET_FILE();

  RUN_TEST(test_bconv2d_int8_DIDO_pseudo_random);
  RUN_TEST(test_bconv2d_int8_DIDO_pseudo_random2);
  RUN_TEST(test_bconv2d_int8_DIDO_sub_image);

  RUN_TEST(test_bconv2d_int8_pseudo_random);
  RUN_TEST(test_bconv2d_int8_pseudo_random2);
  RUN_TEST(test_bconv2d_int8_sub_image);

  RUN_TEST(test_bconv2d_int8_directed);
  RUN_TEST(test_bconv2d_int8_directed2);
  RUN_TEST(test_bconv2d_int8_directed3);
  RUN_TEST(test_bconv2d_int8_directed4);
}
