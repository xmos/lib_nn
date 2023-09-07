
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>


#include "nn_operator.h"
#include "nn_op_helper.h"
#include "tst_common.h"
#include "unity.h"
#include "xs3_vpu.h"

#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)
#define LENGTH     (256)

static int32_t clamp(int32_t v, int32_t lo, int32_t hi){
    if(v < lo)
        return lo;
    if (v>hi)
        return hi;
    return v;
}

// Keep this real simple.
static void test_mul_elementwise_case0()
{
    PRINTF("%s...\n", __func__);

    nn_mul_params_t params;

    double in1Scale = 1. / 128.;
    double in2Scale = 1. / 128.;
    double outputScale= 1. / 128.;
    int8_t in1ZeroPoint = 0;
    int8_t in2ZeroPoint = 0; 
    int8_t outputZeroPoint = 0;

    mul_boggle(&params, in1Scale, in2Scale, outputScale,
        in1ZeroPoint, in2ZeroPoint, outputZeroPoint);
        
    int8_t in1[LENGTH];
    int8_t in2[LENGTH];
    int8_t out[LENGTH];
    int8_t expected[LENGTH];

    for(int i = 0; i < LENGTH; i++){
        in1[i] = i-128;
        in2[i] = -128;
        expected[i] = clamp(-((int32_t)i-128), INT8_MIN, INT8_MAX);
    }

    mul_elementwise(in1, in2, LENGTH, &params, out);

    TEST_ASSERT_EQUAL_INT8_ARRAY(expected, out, LENGTH);

}
#undef LENGTH

#define LENGTH     (128)
static void test_mul_elementwise_case1()
{
    PRINTF("%s...\n", __func__);

    nn_mul_params_t params;

    const double in1Scale = 1. / 128.;
    const double in2Scale = 1. / 128.;
    const double outputScale= 1. / 128.;
    const int8_t in1ZeroPoint = 0;
    const int8_t in2ZeroPoint = 0; 
    const int8_t outputZeroPoint = 0;

    mul_boggle(&params, in1Scale, in2Scale, outputScale,
        in1ZeroPoint, in2ZeroPoint, outputZeroPoint);
        
    int8_t in1[LENGTH];
    int8_t in2[LENGTH];
    int8_t out[LENGTH];
    int8_t expected[LENGTH];

    int abs_error = 0;
    int sum_error = 0;
    int test_count = 0;
    for(int j=0;j<16;j++){

        for(int i = 0; i < LENGTH; i++){
            in1[i] = pseudo_rand_int8();
            in2[i] = pseudo_rand_int8();
            float r = ((float)in1[i] - in1ZeroPoint) * in1Scale * ((float)in2[i]-in2ZeroPoint) * in2Scale / outputScale + outputZeroPoint;
            expected[i] = clamp(round(r), INT8_MIN, INT8_MAX);
        }

        mul_elementwise(in1, in2, LENGTH, &params, out);
        
        for(int i = 0; i < LENGTH; i++){
            TEST_ASSERT_INT8_WITHIN(1, expected[i], out[i]);
            int error = (expected[i]- out[i]);
            sum_error += error;
            if(error<0) error=-error;
            abs_error += error;
            test_count++;
        }
    }
    printf("test_count %d sum_error %d abs_error %d \n", test_count, sum_error, abs_error);

}
#undef LENGTH

#define LENGTH     (128)
static void test_mul_elementwise_case2()
{
    PRINTF("%s...\n", __func__);

    nn_mul_params_t params;

    const double in1Scale = 1. / 128.;
    const double in2Scale = 1. / 128.;
    const double outputScale= 1. / 128.;
     int8_t in1ZeroPoint = 0;
     int8_t in2ZeroPoint = 0; 
     int8_t outputZeroPoint = 0;

    int8_t in1[LENGTH];
    int8_t in2[LENGTH];
    int8_t out[LENGTH];
    int8_t expected[LENGTH];

    int abs_error = 0;
    int sum_error = 0;
    int test_count = 0;
    for(int j=0;j<128;j++){
        in1ZeroPoint = pseudo_rand_int8();
        in2ZeroPoint = pseudo_rand_int8();
        outputZeroPoint = pseudo_rand_int8();

        mul_boggle(&params, in1Scale, in2Scale, outputScale,
            in1ZeroPoint, in2ZeroPoint, outputZeroPoint);
        
        for(int i = 0; i < LENGTH; i++){
            in1[i] = pseudo_rand_int8();
            in2[i] = pseudo_rand_int8();
            float r = ((float)in1[i] - in1ZeroPoint) * in1Scale * ((float)in2[i]-in2ZeroPoint) * in2Scale / outputScale + outputZeroPoint;
            expected[i] = clamp(round(r), INT8_MIN, INT8_MAX);
        }
        
        mul_elementwise(in1, in2, LENGTH, &params, out);
        
        for(int i = 0; i < LENGTH; i++){
            TEST_ASSERT_INT8_WITHIN(1, expected[i], out[i]);
            int error = (expected[i]- out[i]);
            sum_error += error;
            if(error<0) error=-error;
            abs_error += error;
            test_count++;
        }
    }
    printf("test_count %d sum_error %d abs_error %d \n", test_count, sum_error, abs_error);
}
#undef LENGTH

#define LENGTH     (1256)
static void test_mul_elementwise_case3()
{
    PRINTF("%s...\n", __func__);

    nn_mul_params_t params;

    double in1Scale = 1. / 128.;
    double in2Scale = 1. / 128.;
    double outputScale= 1. / 128.;
    int8_t in1ZeroPoint = 0;
    int8_t in2ZeroPoint = 0; 
    int8_t outputZeroPoint = 0;

    int8_t in1[LENGTH];
    int8_t in2[LENGTH];
    int8_t out[LENGTH];
    int8_t expected[LENGTH];

    int abs_error = 0;
    int sum_error = 0;
    int test_count = 0;

    for(int j=0;j<16;j++){
        in1ZeroPoint = pseudo_rand_int8();
        in2ZeroPoint = pseudo_rand_int8();
        outputZeroPoint = pseudo_rand_int8();
        in1Scale = 1. / (float)(uint8_t)(pseudo_rand_int8() | 0x0f);
        in2Scale = 1. / (float)(uint8_t)(pseudo_rand_int8() | 0x0f);
        outputScale = 256.*(1.0 + (float)pseudo_rand_int8() / 2560.) *(in1Scale*in2Scale);

        mul_boggle(&params, in1Scale, in2Scale, outputScale,
            in1ZeroPoint, in2ZeroPoint, outputZeroPoint);

        for(int i = 0; i < LENGTH; i++){
            in1[i] = pseudo_rand_int8();
            in2[i] = pseudo_rand_int8();
            float r = ((float)in1[i] - in1ZeroPoint) * in1Scale * ((float)in2[i]-in2ZeroPoint) * in2Scale / outputScale + outputZeroPoint;
            expected[i] = clamp(round(r), INT8_MIN, INT8_MAX);
        }
        mul_elementwise(in1, in2, LENGTH, &params, out);

        for(int i = 0; i < LENGTH; i++){

            int error = ((int32_t)expected[i]- (int32_t)out[i]);
            sum_error += error;
            if(error<0) error=-error;

            if(error > 1){
                printf("in1ZeroPoint: %d\n", in1ZeroPoint);
                printf("in2ZeroPoint: %d\n", in2ZeroPoint);
                printf("outputZeroPoint: %d\n", outputZeroPoint);
                printf("in1Scale: %f\n", in1Scale);
                printf("in2Scale: %f\n", in2Scale);
                printf("outputScale: %f\n", outputScale);
                printf("params.scalar: %d\n", params.scalar);
                printf("params.bias: %d\n", params.bias);
                printf("params.vlashr_shr: %d\n", params.vlashr_shr);
                printf("in1[%d]: %d\n", i, in1[i]);
                printf("in1[%d]: %d\n", i, in2[i]);
            }
            abs_error += error;
            test_count++;
            TEST_ASSERT_INT8_WITHIN(1, expected[i], out[i]);
        }
    }
    printf("test_count %d sum_error %d abs_error %d \n", test_count, sum_error, abs_error);

}
#undef LENGTH


#define LENGTH     (48)
static void test_mul_elementwise_case4()
{
    PRINTF("%s...\n", __func__);

    nn_mul_params_t params;

    double in1Scale = 1. / 128.;
    double in2Scale = 1. / 128.;
    double outputScale= 1. / 128.;
    int8_t in1ZeroPoint = 0;
    int8_t in2ZeroPoint = 0; 
    int8_t outputZeroPoint = 0;

    int8_t in1[LENGTH];
    int8_t in2[LENGTH];
    int8_t out[LENGTH];
    int8_t expected[LENGTH];

    int abs_error = 0;
    int sum_error = 0;
    int test_count = 0;

    for(int test_length=1;test_length<=LENGTH;test_length++){
        in1ZeroPoint = pseudo_rand_int8();
        in2ZeroPoint = pseudo_rand_int8();
        outputZeroPoint = pseudo_rand_int8();
        in1Scale = 1. / (float)(uint8_t)(pseudo_rand_int8() | 0x0f);
        in2Scale = 1. / (float)(uint8_t)(pseudo_rand_int8() | 0x0f);
        outputScale = 256.*(1.0 + (float)pseudo_rand_int8() / 2560.) *(in1Scale*in2Scale);

        mul_boggle(&params, in1Scale, in2Scale, outputScale,
            in1ZeroPoint, in2ZeroPoint, outputZeroPoint);

        for(int i = 0; i < test_length; i++){
            in1[i] = pseudo_rand_int8();
            in2[i] = pseudo_rand_int8();
            float r = ((float)in1[i] - in1ZeroPoint) * in1Scale * ((float)in2[i]-in2ZeroPoint) * in2Scale / outputScale + outputZeroPoint;
            expected[i] = clamp(round(r), INT8_MIN, INT8_MAX);
        }        
        for(int i = test_length; i < LENGTH; i++){
            expected[i] = pseudo_rand_int8();
            out[i] = expected[i];
        }

        mul_elementwise(in1, in2, test_length, &params, out);

        for(int i = 0; i < LENGTH; i++){

            int error = ((int32_t)expected[i]- (int32_t)out[i]);
            sum_error += error;
            if(error<0) error=-error;

            if(error > 1){
                printf("in1ZeroPoint: %d\n", in1ZeroPoint);
                printf("in2ZeroPoint: %d\n", in2ZeroPoint);
                printf("outputZeroPoint: %d\n", outputZeroPoint);
                printf("in1Scale: %f\n", in1Scale);
                printf("in2Scale: %f\n", in2Scale);
                printf("outputScale: %f\n", outputScale);
                printf("params.scalar: %d\n", params.scalar);
                printf("params.bias: %d\n", params.bias);
                printf("params.vlashr_shr: %d\n", params.vlashr_shr);
                printf("in1[%d]: %d\n", i, in1[i]);
                printf("in1[%d]: %d\n", i, in2[i]);
            }
            abs_error += error;
            test_count++;
            TEST_ASSERT_INT8_WITHIN(1, expected[i], out[i]);
        }
    }
    printf("test_count %d sum_error %d abs_error %d \n", test_count, sum_error, abs_error);

}
#undef LENGTH
void test_mul_elementwise()
{
    srand(563456);

    UNITY_SET_FILE();
    
    RUN_TEST(test_mul_elementwise_case0);
    RUN_TEST(test_mul_elementwise_case1);
    RUN_TEST(test_mul_elementwise_case2);
    RUN_TEST(test_mul_elementwise_case3);
    RUN_TEST(test_mul_elementwise_case4);
}