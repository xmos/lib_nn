// Copyright 2020-2026 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.

#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "nn_operator.h"
#include "nn_op_helper.h"
#include "tst_common.h"
#include "unity.h"
#include "unity_fixture.h"
#include "xs3_vpu.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)

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

char msg_buff[200];

TEST_GROUP(group_add_elementwise);
TEST_SETUP(group_add_elementwise) { srand(563456); }
TEST_TEAR_DOWN(group_add_elementwise) {}
TEST_GROUP_RUNNER(group_add_elementwise) {
    RUN_TEST_CASE(group_add_elementwise, test_add_elementwise_case0);
    RUN_TEST_CASE(group_add_elementwise, test_add_elementwise_case1);
    RUN_TEST_CASE(group_add_elementwise, test_add_elementwise_case2);
}

#define LENGTH     (16)
TEST(group_add_elementwise, test_add_elementwise_case0)
{
    PRINTF("%s...\n", __func__);

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
#undef LENGTH


#define LENGTH     (128)
TEST(group_add_elementwise, test_add_elementwise_case1)
{
    PRINTF("%s...\n", __func__);

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
            
    for(int i = 0; i < LENGTH; i++)
        Y_expected[i] = vlsat_single_s8(m1*X1[i] + m2*X2[i] + bias, shift, NEG_SAT_VAL, VPU_INT8_MAX);

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
#undef LENGTH


#define LEN     (100)
#define REPS    (200)
TEST(group_add_elementwise, test_add_elementwise_case2)
{
    PRINTF("%s...\n", __func__);

    int8_t WORD_ALIGNED Y[LEN];
    int8_t WORD_ALIGNED X0[LEN];
    int8_t WORD_ALIGNED X1[LEN];
    int8_t Y_expected[LEN];
    
    for(int v = 0; v < REPS; v++){

        unsigned elm_start = (pseudo_rand_uint32() % LEN) & 0xFFFFFFFC;
        unsigned elm_count = pseudo_rand_uint32() % (LEN - elm_start);

        PRINTF("  rep %u... (%u <= k < %u)\n", v, elm_start, elm_start+elm_count);

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
            PRINTF("    params.input[0].multiplier = %d   (0x%04X)\n", m[0], (unsigned) m[0]);
            PRINTF("    params.input[1].multiplier = %d   (0x%04X)\n", m[1], (unsigned) m[1]);

            PRINTF("    max = %ld\n", max);
            PRINTF("    min = %ld\n", min);
            PRINTF("    diff = %lu     (0x%08lX)\n", diff, diff);
            PRINTF("    scale = %u\n", scale);

            PRINTF("    params.output.bias = %ld    (0x%08lX)\n", bias, (uint32_t)bias);
            PRINTF("    params.output.shr = %d\n", shr);

            unsigned m = 13;
            PRINTF("      X0[%u] = %d\n", m, X0[m]);
            PRINTF("      X1[%u] = %d\n", m, X1[m]);
            PRINTF("      Y_expected[%u] = %d\n", m, Y_expected[m]);
            PRINTF("      Y[%u] = %d\n", m, Y[m]);
        }


        TEST_ASSERT_EQUAL_INT8_ARRAY(Y_expected, Y, LEN);

    }
}
#undef LEN
#undef REPS
