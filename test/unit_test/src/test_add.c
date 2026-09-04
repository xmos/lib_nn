// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "nn_operator.h"
#include "nn_op_helper.h"
#include "nn_arch.h"
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"
#include "xs3_vpu.h"


#ifdef CONFIG_SYMMETRIC_SATURATION_GLOBAL
  #define CONFIG_SYMMETRIC_SATURATION_add_elementwise CONFIG_SYMMETRIC_SATURATION_GLOBAL
#else
  #ifndef CONFIG_SYMMETRIC_SATURATION_add_elementwise
    #define CONFIG_SYMMETRIC_SATURATION_add_elementwise (0)
  #endif 
#endif

#if CONFIG_SYMMETRIC_SATURATION_add_elementwise
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 

TEST_GROUP(group_add_elementwise);
TEST_SETUP(group_add_elementwise) { srand(563456); }
TEST_TEAR_DOWN(group_add_elementwise) {}
TEST_GROUP_RUNNER(group_add_elementwise) {
    RUN_TEST_CASE(group_add_elementwise, test_add_elementwise_case0);
    RUN_TEST_CASE(group_add_elementwise, test_add_elementwise_case1);
    RUN_TEST_CASE(group_add_elementwise, test_add_elementwise_case2);
}

TEST_GROUP(group_add_int16);
TEST_SETUP(group_add_int16) {}
TEST_TEAR_DOWN(group_add_int16) {}
TEST_GROUP_RUNNER(group_add_int16) {
    RUN_TEST_CASE(group_add_int16, test_add_tensor_int16);
}

TEST(group_add_elementwise, test_add_elementwise_case0)
{
    const unsigned LENGTH = 16;
    int8_t WORD_ALIGNED Y[LENGTH];
    int8_t WORD_ALIGNED X1[LENGTH];
    int8_t WORD_ALIGNED X2[LENGTH];
    
    int8_t Y_expected[LENGTH];

    for(int i = 0; i < LENGTH; i++){
        X1[i] = X2[i] = i;
    }

    nn_add_params_t params;
    int16_t m1 = 0x0001; // multiplier of 1 
    int16_t m2 = 0x0001; // multiplier of 1
    int16_t bias = 0;    // bias of 0
    int16_t shift = 1;   // divide by 2

    // Broadcast values into vectors
    for (int i = 0; i < 16; i++) {
        params.m1[i] = m1;
        params.m2[i] = m2;
        params.shift[i] = shift;
        params.bias_hi[i] = bias >> 16;
        params.bias_lo[i] =  (bias & 0XFFFF);
    }

    // we expect sum and divide by 2 be the same            
    for(int i = 0; i < LENGTH; i++){
        Y_expected[i] = i;
    }

    add_elementwise(Y, X1, X2, &params, 0, LENGTH);
    TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LENGTH);
}

TEST(group_add_elementwise, test_add_elementwise_case1)
{
    const unsigned LENGTH = 128;
    int8_t WORD_ALIGNED Y[LENGTH];
    int8_t WORD_ALIGNED X1[LENGTH];
    int8_t WORD_ALIGNED X2[LENGTH];
    
    int8_t Y_expected[LENGTH];

    for(int i = 0; i < LENGTH; i++){
        X1[i] = X2[i] = i;
    }

    nn_add_params_t params;
        // {   {   -8, 0x0001 },
        //     {   -7, 0x0002 } },
        //     {  -0x00008000, 8} };
    int m1 = 0x0001;
    int m2 = 0x0002;
    int bias = -0x00008000;
    int shift = 8;
    // Broadcast values into vectors
    for (int i = 0; i < 16; i++) {
        params.m1[i] = (int16_t)m1;
        params.m2[i] = (int16_t)m2;
        params.shift[i] = (int16_t)shift;
        params.bias_hi[i] = bias >> 16;
        params.bias_lo[i] = (int16_t) (bias & 0XFFFF);
    }
            
    for(int i = 0; i < LENGTH; i++){
        Y_expected[i] = vlsat_single_s8(m1*X1[i] + m2*X2[i] + bias, shift, NEG_SAT_VAL, VPU_INT8_MAX);
    }
    unsigned start = 0;

    { // 0 <= i < 16
        unsigned count = 16;    // One full vector
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count], LENGTH - (start + count) );
        start += count;
    }

    { // 16 <= i < 20
        unsigned count = 4;     // Less than one vector
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count], LENGTH - (start + count) );
        start += count;
    }

    { // 20 <= i < 52
        unsigned count = 32;    // Two full vectors
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[start + count], LENGTH - (start + count) );
        start += count;
    }

    { // 52 <= i < 128
        unsigned count = 76;    // 4 vectors and change.
        memset(Y, 0xCC, sizeof(Y));
        add_elementwise(Y, X1, X2, &params, start, count);
        TEST_ASSERT_EACH_EQUAL_INT8(0xCC, &Y[0], start);
        TEST_ASSERT_EQUAL_INT8_ARRAY(&Y_expected[start], &Y[start], count);
    }

}

TEST(group_add_elementwise, test_add_elementwise_case2)
{
    const unsigned LEN = 100;
    const unsigned REPS = 200;
    int8_t WORD_ALIGNED Y[LEN];
    int8_t WORD_ALIGNED X0[LEN];
    int8_t WORD_ALIGNED X1[LEN];
    int8_t Y_expected[LEN];
    
    for(int v = 0; v < REPS; v++){

        unsigned elm_start = (pseudo_rand_uint32() % LEN) & 0xFFFFFFFC;
        unsigned elm_count = pseudo_rand_uint32() % (LEN - elm_start);
        // printf("  rep %u... (%u <= k < %u)\n", v, elm_start, elm_start+elm_count);
        int32_t min = 0;
        int32_t max = 0;

        int m[2];
        for(int i = 0; i < 2; i++){
            m[i] = 1 + (pseudo_rand_uint16() >> 1);

            min += (((int32_t)-128))*m[i];
            max += (((int32_t) 127))*m[i];
        }

        uint32_t diff = max - min;

        unsigned scale = ceil_log2(diff);

        int shr = scale - 8;

        int bias = 0;

        pseudo_rand_bytes((char*)X0, LEN);
        pseudo_rand_bytes((char*)X1, LEN);

        memset(Y_expected, 0xCC, sizeof(Y_expected));

        for(int i = elm_start; i < elm_start+elm_count; i++){
            int32_t acc = bias;

            int32_t x0 = ((int32_t) X0[i]);
            acc += x0 * m[0];

            int32_t x1 = ((int32_t) X1[i]);
            acc += x1 * m[1];

            Y_expected[i] = vlsat_single_s8(acc, shr, NEG_SAT_VAL, VPU_INT8_MAX);
        }

        memset(Y, 0xCC, sizeof(Y));

        nn_add_params_t params;
        // Broadcast values into vectors
        for (int i = 0; i < 16; i++) {
            params.m1[i] = (int16_t)m[0];
            params.m2[i] = (int16_t)m[1];
            params.shift[i] = (int16_t)shr;
            params.bias_hi[i] = bias >> 16;
            params.bias_lo[i] = (int16_t) (bias & 0XFFFF);
        }

        add_elementwise(Y, X0, X1, &params, elm_start, elm_count);

        if(v == -1){
            printf("    params.input[0].multiplier = %d   (0x%04X)\n", m[0], (unsigned) m[0]);
            printf("    params.input[1].multiplier = %d   (0x%04X)\n", m[1], (unsigned) m[1]);

            printf("    max = %d\n", (int)max);
            printf("    min = %d\n", (int)min);
            printf("    diff = %u     (0x%08X)\n", (unsigned)diff, (unsigned)diff);
            printf("    scale = %u\n", scale);

            printf("    params.output.bias = %d    (0x%08X)\n", bias, (unsigned)bias);
            printf("    params.output.shr = %d\n", shr);

            unsigned m = 13;
            printf("      X0[%u] = %d\n", m, X0[m]);
            printf("      X1[%u] = %d\n", m, X1[m]);
            printf("      Y_expected[%u] = %d\n", m, Y_expected[m]);
            printf("      Y[%u] = %d\n", m, Y[m]);
        }
        TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LEN);
    }
}

TEST(group_add_int16, test_add_tensor_int16)
{
    const unsigned N = 39;
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
