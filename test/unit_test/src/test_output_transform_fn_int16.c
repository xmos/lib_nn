// Copyright 2023-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdint.h>
#include <string.h>

#include "nn_layers.h"

#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

TEST_GROUP(group_output_transform_fn_int16);
TEST_SETUP(group_output_transform_fn_int16) {}
TEST_TEAR_DOWN(group_output_transform_fn_int16) {}
TEST_GROUP_RUNNER(group_output_transform_fn_int16) {
  RUN_TEST_CASE(group_output_transform_fn_int16, test_output_transform_fn_int16);
  RUN_TEST_CASE(group_output_transform_fn_int16, test_output_transform_fn_int16_sat);
  RUN_TEST_CASE(group_output_transform_fn_int16, test_output_transform_fn_int16_kernel_transform);
}

TEST(group_output_transform_fn_int16, test_output_transform_fn_int16) {
    const int16_t expected_output[16] = {
        0x1001, 0x2001, 0x3001, 0x4001, 0x5001, 0x6001, 0x7001, 0x7fff,
        -28673, -24577, -20481, -16385, -12289, -8193, -4097, -1
    };
    const int16_t vDvR_input[32] = {
        9,10,11,12,13,14,15,16,
        -17,-18,-19,-20,-21,-22,-23,-24,
        1,2,3,4,5,6,7,8,
        -8,-7,-6,-5,-4,-3,-2,-1
    };
    int32_t mul_add[32] = {
        0, 0, 0, 0, 0, 0, 0, 0,
        0x04000000, 0x04000000, 0x04000000, 0x04000000,
        0x04000000, 0x04000000, 0x04000000, 0x04000000,
        0, 0, 0, 0, 0, 0, 0, 0,
        0x04000000, 0x04000000, 0x04000000, 0x04000000,
        0x04000000, 0x04000000, 0x04000000, 0x04000000
    };
    int16_t vDvR[32] __attribute__((aligned(8)));
    int16_t WORD_ALIGNED output[40];

    for(int j = 1; j <= 16; j++) {
        otfn_int16_params_t otfn_params = {j};
        memcpy(vDvR, vDvR_input, sizeof(vDvR));
        for(int i = 0; i < 40; i++) {
            output[i] = i*i+13;
        }
        int16_t *output_end =
            output_transform_fn_int16(&otfn_params, output+4, vDvR, 0, mul_add);
        TEST_ASSERT_EQUAL_PTR(output + 4 + j, output_end);
        for(int i = 0; i < 40; i++) {
            if (i < 4 || i >= 4+j) {
                TEST_ASSERT_EQUAL(output[i], i*i+13);
            }
        }
        for(int i = 0; i < j; i++) {
            TEST_ASSERT_EQUAL(output[i+4], expected_output[i]);
        }
    }
}

TEST(group_output_transform_fn_int16, test_output_transform_fn_int16_sat) {
    otfn_int16_params_t otfn_params = {2};
    int16_t vDvR[32] __attribute__((aligned(8))) = {0};
    int32_t mul_add[32] = {0};
    int16_t WORD_ALIGNED output[16];

    vDvR[0] = -1;
    vDvR[16] = INT16_MAX;
    vDvR[1] = 0;
    vDvR[17] = INT16_MIN;
    mul_add[16] = 1;
    mul_add[24] = 0x40000000;
    mul_add[0] = -1;
    mul_add[8] = 0x40000000;

    output_transform_fn_int16(&otfn_params, output, vDvR, 0, mul_add);

    TEST_ASSERT_EQUAL_INT16(INT16_MAX, output[0]);
    TEST_ASSERT_EQUAL_INT16(INT16_MIN, output[1]);
}

TEST(group_output_transform_fn_int16, test_output_transform_fn_int16_kernel_transform) {
    const int16_t expected_output[16] = {
        -22, 33, 95, 164, 239, 322, 410, 506,
        -616, -717, -825, -938, -1058, -1184, -1315, -1453
    };
    int8_t kernel_weights_in[16*8] = {0};
    int8_t kernel_weights_out[16*8] = {0};
    int16_t vDvR[32] __attribute__((aligned(8))) = {0};
    int16_t WORD_ALIGNED vDvRoutput[16];
    float channel_multipliers_in[16];
    int channel_bias_terms_in[16];
    int32_t mul_add_out[32];

    otfn_int16_params_t otfn_params = {16};
    for(int i = 0; i < 16; i++) {
        int32_t accumulator = i >= 8 ? -103*i : 101*i;
        channel_multipliers_in[i] = (i+16)/32.0;
        channel_bias_terms_in[i] = 6*i-45;
        vDvR[i] = accumulator;
        vDvR[i+16] = accumulator < 0 ? -1 : 0;
    }
    output_transform_fn_int16_kernel_transform(
        kernel_weights_in,
        channel_multipliers_in, channel_bias_terms_in,
        kernel_weights_out, mul_add_out,
        8, 16);

    for(int i = 0; i < 16; i++) {
        int bias_index = (i & 1) ? i/2 : 16+i/2;
        int multiplier_index = bias_index + 8;
        TEST_ASSERT_EQUAL_INT32(channel_bias_terms_in[i],
                                mul_add_out[bias_index]);
        TEST_ASSERT_EQUAL_INT32((i+16) << 25,
                                mul_add_out[multiplier_index]);
    }

    output_transform_fn_int16(&otfn_params, vDvRoutput, vDvR, 0, mul_add_out);
    for(int i = 0; i < 16; i++) {
        TEST_ASSERT_EQUAL_INT16(expected_output[i], vDvRoutput[i]);
    }
}
