// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "dequantize_int16.h"
#include "dequantize_int16_transform.h"

#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

#define N 25

TEST_GROUP(group_dequantize_int16);
TEST_SETUP(group_dequantize_int16) {}
TEST_TEAR_DOWN(group_dequantize_int16) {}
TEST_GROUP_RUNNER(group_dequantize_int16) {
    RUN_TEST_CASE(group_dequantize_int16, test_dequantize_tensor_int16);
}

TEST(group_dequantize_int16, test_dequantize_tensor_int16) {
    int16_t input1[N];
    int8_t blob[DEQUANTIZE_INT16_TENSOR_BYTES()];
    float output[N+1];
    float ref_output[N];
    for(int i = 0; i < N; i++) {
        input1[i] = 20000 - 2513 * i;
    }
    input1[3] = 22767;
    input1[4] = 21726;
    input1[5] = -21998;
    float scaler1 = 0.00004051757812;
    for(int i = 0; i < N; i++) {
        float o = input1[i] * scaler1;
        ref_output[i] = o;
    }
    char err_msg[ERR_MSG_DESCRIPTOR_FAIL_BYTES()];
    int success = dequantize_int16_tensor_blob(blob,
                                               scaler1,
                                               err_msg);
    
    TEST_ASSERT_EQUAL(1, success);
   
    output[N] = 0x5555;
    dequantize_int16_tensor(output, input1, N, blob);
    TEST_ASSERT_EQUAL(output[N], 0x5555);

    for(int i = 0; i < N; i++) {
        TEST_ASSERT_EQUAL_FLOAT(ref_output[i], output[i]);
    }
}
