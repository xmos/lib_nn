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

TEST_GROUP(group_pad_3_to_4);
TEST_SETUP(group_pad_3_to_4) {}
TEST_TEAR_DOWN(group_pad_3_to_4) {}
TEST_GROUP_RUNNER(group_pad_3_to_4) {
  RUN_TEST_CASE(group_pad_3_to_4, test_pad_3_to_4_param_space_int8);
}

void impl_pad_3_to_4_param_space(
    const unsigned N_loop)
{
  const int x_chan_words = 3, y_chan_words = 4;
  int seed = 0;
  for (unsigned pad_val_idx = 0; pad_val_idx < 8; pad_val_idx++) {
    // pick a pad value
    uint32_t pad_value = (uint32_t)pseudo_rand(&seed);
    unsigned x_height = 1;

    for (unsigned i = 0; i < N_loop; i++, x_height += 3) {
      unsigned x_width = 1;
      for (unsigned j = 0; j < N_loop; j++, x_width += 3) {

          size_t X_bytes = x_height * x_width * x_chan_words;
          int8_t* X = (int8_t* )malloc(X_bytes);

          unsigned y_height = x_height;
          unsigned y_width = x_width;

          size_t Y_bytes = y_height * y_width * y_chan_words;

          int8_t* Y_ref = (int8_t* )malloc(Y_bytes);
          int8_t* Y = (int8_t* )malloc(Y_bytes);

          for (unsigned b = 0; b < X_bytes; b++)
            X[b] = (int8_t)pseudo_rand(&seed);
          memset(Y, 0x55, Y_bytes);
          memset(Y_ref, 0xaa, Y_bytes);

          uint32_t n_3;
          pad_3_to_4_prepare(&n_3, x_height, x_width);

          pad_3_to_4_ref(Y_ref, X, n_3, pad_value);
          pad_3_to_4_run(Y, X, n_3, pad_value);

          TEST_ASSERT_EQUAL_INT8_ARRAY(Y, Y_ref, Y_bytes);

          free(Y);
          free(Y_ref);
          free(X);
      }
    }
  }
}

TEST(group_pad_3_to_4, test_pad_3_to_4_param_space_int8) {
#ifdef TEST_BUILD_NATIVE
  // This test on native will run reference againts itself
  // reference code is run against xs3a and vx4b
  // so we know it must work
  TEST_IGNORE_MESSAGE("pad_3_to_4 is not tested natively");
#endif // TEST_BUILD_NATIVE
  impl_pad_3_to_4_param_space(4);
}

