// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <assert.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include "helpers.h"
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(group_pad);
TEST_SETUP(group_pad) {}
TEST_TEAR_DOWN(group_pad) {}
TEST_GROUP_RUNNER(group_pad) {
  RUN_TEST_CASE(group_pad, test_pad_3_to_4);
  RUN_TEST_CASE(group_pad, test_pad_1_to_4);
}

void impl_pad_x_to_4_param_space(
    const unsigned x_chan_words,
    const unsigned N_loop)
{
  const unsigned y_chan_words = 4;
  int seed = 0;
  for (unsigned pad_val_idx = 0; pad_val_idx < 8; pad_val_idx++) {
    // pick a pad value
    uint32_t pad_value = (uint32_t)pseudo_rand(&seed);

    for (unsigned x_height = 1, i = 0; i < N_loop; i++, x_height += 3) {
      for (unsigned x_width = 1, j = 0; j < N_loop; j++, x_width += 3) {
          // pad_3_to_4 takes number of bytes after the prep
          // while pad_1_to_4 only operates on 4 byte chunks
          // I love how the APIs are designed here
          size_t N_groups = x_height * x_width;
          size_t values_per_group = x_chan_words == 1 ? 4 : 1;
          size_t N_values = N_groups * values_per_group;
          size_t X_bytes = N_values * x_chan_words;
          size_t Y_bytes = N_values * y_chan_words;

          int8_t* X = (int8_t* )malloc(X_bytes);
          int8_t* Y_ref = (int8_t* )malloc(Y_bytes);
          int8_t* Y = (int8_t* )malloc(Y_bytes);

          for (unsigned b = 0; b < X_bytes; b++) {
            X[b] = (int8_t)pseudo_rand(&seed);
          }

          memset(Y, 0x55, Y_bytes);
          memset(Y_ref, 0xaa, Y_bytes);

          if (x_chan_words == 3) {
            uint32_t n_3;
            pad_3_to_4_prepare(&n_3, x_height, x_width);

            for (uint32_t pixel = 0; pixel < n_3; pixel++) {
              memcpy(Y_ref + pixel * 4, X + pixel * 3, 3);
              Y_ref[pixel * 4 + 3] = (int8_t)(pad_value >> 24);
            }
            pad_3_to_4_run(Y, X, n_3, pad_value);
          }
          else if (x_chan_words == 1) {
            for (size_t input_index = 0; input_index < N_values;
                 input_index++) {
              Y_ref[input_index * 4] = X[input_index];
              Y_ref[input_index * 4 + 1] = (int8_t)(pad_value >> 8);
              Y_ref[input_index * 4 + 2] = (int8_t)(pad_value >> 16);
              Y_ref[input_index * 4 + 3] = (int8_t)(pad_value >> 24);
            }
            pad_1_to_4_run(Y, X, N_groups, pad_value);
          }
          else {
            TEST_ASSERT(0);
          }

          TEST_ASSERT_EQUAL_INT8_ARRAY(Y, Y_ref, Y_bytes);

          free(Y);
          free(Y_ref);
          free(X);
      }
    }
  }
}

TEST(group_pad, test_pad_3_to_4) {
#ifdef TEST_BUILD_NATIVE
  // This test on native will run reference against itself
  // reference code is run against xs3a and vx4b
  // so we know it must work
  TEST_IGNORE_MESSAGE("pad_3_to_4 is not tested natively");
#endif // TEST_BUILD_NATIVE
  impl_pad_x_to_4_param_space(3, 4);
}

TEST(group_pad, test_pad_1_to_4) {
#ifdef TEST_BUILD_NATIVE
  // This test on native will run reference against itself
  // reference code is run against xs3a and vx4b
  // so we know it must work
  TEST_IGNORE_MESSAGE("pad_1_to_4 is not tested natively");
#endif // TEST_BUILD_NATIVE
  impl_pad_x_to_4_param_space(1, 4);
}
