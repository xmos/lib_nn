// Copyright 2024-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include <stdio.h>
#include <stdint.h>
#include <stdlib.h>
#include <string.h>
#include "math.h"

#include "add_int16.h"
#include "add_int16_transform.h"

#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"

#define N 39

TEST_GROUP(group_add_int16);
TEST_SETUP(group_add_int16) {}
TEST_TEAR_DOWN(group_add_int16) {}
TEST_GROUP_RUNNER(group_add_int16) {
    RUN_TEST_CASE(group_add_int16, test_add_tensor_int16);
}

TEST(group_add_int16, test_add_tensor_int16)
{
#if defined(__VX4A__) || defined(__VX4B__)
    // KNOWN ISSUE: add_int16_tensor_asm has a pre-existing bug (see the
    // "asm is broken" TODO in add_int16.c) that traps with an unhandled
    // LOAD_STORE exception on VX4, halting the whole binary.
    TEST_IGNORE_MESSAGE("add_int16_tensor_asm traps with LOAD_STORE on VX4");
#endif
    int16_t input1[N];
    int16_t input2[N];
    int8_t blob[ADD_INT16_TENSOR_BYTES()];
    int16_t output[N+1];
    int16_t ref_output[N];
    for(int j=1; j < N; j++) {
    for(int i = 0; i < j; i++) {
        input1[i] = 20000 - 2513 * i;
        input2[i] = 417 * i + 82;
    }
    input2[3] = 30001;
    input2[4] = 31003;
    input2[5] = -32003;
    input1[3] = 22767;
    input1[4] = 21726;
    input1[5] = -21998;
    float scaler1 = 0.00004051757812;
    float scaler2 = 0.00006123190654;
    float scalero = 0.00006213;
    for(int i = 0; i < j; i++) {
        float oo = (input1[i] * scaler1 + input2[i] * scaler2) / scalero;
        float o = round(oo);
        if (o >  32767) o =  32767;
        if (o < -32768) o = -32768;
        ref_output[i] = o;
    }
    char err_msg[ERR_MSG_DESCRIPTOR_FAIL_BYTES()];
    int success = add_int16_tensor_blob(blob,
                                        scaler1,
                                        scaler2,
                                        scalero,
                                        err_msg);
    
    TEST_ASSERT_EQUAL(1, success);
   
    output[j] = 0x5555;
    add_int16_tensor(output, input1, input2, j, blob);
    TEST_ASSERT_EQUAL(output[j], 0x5555);

    int sqerr = 0;
    for(int i = 0; i < j; i++) {
        int err = ref_output[i] - output[i];
        sqerr += err*err;
        TEST_ASSERT_INT_WITHIN(1, ref_output[i], output[i]);
    }
    TEST_ASSERT_INT_WITHIN(8, sqerr, 0);
    }
}
