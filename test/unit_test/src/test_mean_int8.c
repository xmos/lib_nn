// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "nn_operator.h"

#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(group_mean_int8);
TEST_SETUP(group_mean_int8) { srand(0xBADC0DE); }
TEST_TEAR_DOWN(group_mean_int8) {}
TEST_GROUP_RUNNER(group_mean_int8) {
    RUN_TEST_CASE(group_mean_int8, test_mean_int8_asm_path);
    RUN_TEST_CASE(group_mean_int8, test_mean_int8_short_mean);
    RUN_TEST_CASE(group_mean_int8, test_mean_int8_ref_path);
    RUN_TEST_CASE(group_mean_int8, test_mean_int8_saturation);
}

// Largest tensor sizes used by the test cases below
#define MAX_INPUT_SIZE   (1024)
#define MAX_OUTPUT_SIZE  (32)

/**
 * Reference implementation, replicating the C fallback of mean_int8()
 * (see lib_nn/src/c/mean.c).
 */
static void ref_mean_int8(const int8_t *input, int8_t *output,
                          const int start_dim_size, const int mean_dim_size,
                          const int end_dim_size, const float in_zero_point,
                          const float out_zero_point, const float scale_mul) {
    const int32_t start = -in_zero_point * mean_dim_size;
    for (int i = 0; i < start_dim_size; ++i) {
        const int i_mul = i * mean_dim_size * end_dim_size;
        for (int k = 0; k < end_dim_size; ++k) {
            int32_t accumulator = start;
            for (int j = 0; j < mean_dim_size; ++j) {
                accumulator += input[i_mul + j * end_dim_size + k];
            }
            float quantized_value = (float)accumulator * scale_mul + out_zero_point;
            if (quantized_value > 127.0f)
                quantized_value = 127.0f;
            else if (quantized_value < -128.0f)
                quantized_value = -128.0f;
            output[i * end_dim_size + k] = (int8_t)(roundf(quantized_value));
        }
    }
}

/**
 * Run mean_int8() on the given input and compare against the reference,
 * allowing a tolerance of 1 (float operation ordering differs between the
 * VPU-optimized and C paths). Also checks for output buffer overruns.
 */
static void run_and_check(const int8_t *input, const int start_dim_size,
                          const int mean_dim_size, const int end_dim_size,
                          const float in_zero_point, const float out_zero_point,
                          const float scale_mul) {
    const int out_size = start_dim_size * end_dim_size;
    int8_t WORD_ALIGNED output[MAX_OUTPUT_SIZE + 1];
    int8_t ref_output[MAX_OUTPUT_SIZE];

    ref_mean_int8(input, ref_output, start_dim_size, mean_dim_size,
                  end_dim_size, in_zero_point, out_zero_point, scale_mul);

    memset(output, 0xAA, out_size);
    output[out_size] = 0x55;    // overrun sentinel
    mean_int8(input, output, start_dim_size, mean_dim_size, end_dim_size,
              in_zero_point, out_zero_point, scale_mul);
    TEST_ASSERT_EQUAL_INT8_ARRAY(ref_output, output, out_size);
}

static void fill_random(int8_t *input, const int size) {
    for (int i = 0; i < size; ++i) {
        input[i] = pseudo_rand_int8();
    }
}

// end_dim_size = 1 and mean_dim_size % 4 == 0 selects the VPU-optimized path
// on XS3A. mean_dim_size = 48 > 32 exercises the 32-element accumulation
// loop; start_dim_size = 20 (not a multiple of 16) exercises the channel
// tail loop.
TEST(group_mean_int8, test_mean_int8_asm_path)
{
    const int start_dim_size = 20;
    const int mean_dim_size = 48;
    const int end_dim_size = 1;
    int8_t WORD_ALIGNED input[MAX_INPUT_SIZE];

    fill_random(input, start_dim_size * mean_dim_size * end_dim_size);

    run_and_check(input, start_dim_size, mean_dim_size, end_dim_size,
                  -7.0f, 13.0f, 0.45f);
}

// As above, but with mean_dim_size < 32 to exercise the partial-word filter
// tail of the VPU-optimized path.
TEST(group_mean_int8, test_mean_int8_short_mean)
{
    const int start_dim_size = 5;
    const int mean_dim_size = 8;
    const int end_dim_size = 1;
    int8_t WORD_ALIGNED input[MAX_INPUT_SIZE];

    fill_random(input, start_dim_size * mean_dim_size * end_dim_size);

    run_and_check(input, start_dim_size, mean_dim_size, end_dim_size,
                  3.0f, -9.0f, 1.7f);
}

// end_dim_size != 1 (and mean_dim_size % 4 != 0) always selects the C
// reference path, on all platforms.
TEST(group_mean_int8, test_mean_int8_ref_path)
{
    const int start_dim_size = 4;
    const int mean_dim_size = 7;
    const int end_dim_size = 3;
    int8_t WORD_ALIGNED input[MAX_INPUT_SIZE];

    fill_random(input, start_dim_size * mean_dim_size * end_dim_size);

    run_and_check(input, start_dim_size, mean_dim_size, end_dim_size,
                  -2.0f, 5.0f, 0.9f);
}

// Extreme inputs with a large scale_mul must saturate to the int8 range.
TEST(group_mean_int8, test_mean_int8_saturation)
{
    const int start_dim_size = 3;
    const int mean_dim_size = 32;
    const int end_dim_size = 1;
    int8_t WORD_ALIGNED input[MAX_INPUT_SIZE];

    for (int j = 0; j < mean_dim_size; ++j) {
        input[0 * mean_dim_size + j] = 127;
        input[1 * mean_dim_size + j] = -128;
        input[2 * mean_dim_size + j] = (j & 1) ? 127 : -128;
    }

    run_and_check(input, start_dim_size, mean_dim_size, end_dim_size,
                  0.0f, 0.0f, 100.0f);
}
