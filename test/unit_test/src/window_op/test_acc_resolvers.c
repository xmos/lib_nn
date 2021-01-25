
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>

#include "../src/c/util/window_op.h"
#include "../tst_common.h"

#include "nn_operator.h"
#include "xs3_vpu.h"

#include "unity.h"



void test_conv2d_acc32_asymmetric_resolver()
{
    srand(3523466);

    printf("%s...\n", __func__);

    nn_acc32_to_int8_params_t WORD_ALIGNED quant_params;

    nn_image_t WORD_ALIGNED Y[ VPU_INT8_ACC_PERIOD ] = { 0 };

    nn_acc32_vector_t WORD_ALIGNED acc = {{0},{0}};

    int8_t expected[ VPU_INT8_ACC_PERIOD ] = { 0 };

    // The resolvers need args for the job context and wop_params, even though it only should use one value from it
    window_op_job_context_t job_context;
    nn_window_op_params_t wop_params;
    memset(&job_context, 0, sizeof(job_context));
    memset(&wop_params, 0, sizeof(wop_params));

    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = 0;

            expected[i] = 0;
        }

        conv2d_acc32_asymmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = i;

            expected[i] = i;
        }

        conv2d_acc32_asymmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 2;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = 8*i;

            expected[i] = 2*i;
        }

        conv2d_acc32_asymmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 4;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = i;

            expected[i] = 4*i;
        }

        conv2d_acc32_asymmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 2;
            quant_params.offset[i] = i+1;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = i;

            expected[i] = i + 2*(i+1);
        }

        conv2d_acc32_asymmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 1;

            acc.high[i] = 0;
            acc.low[i] = 4*i;

            expected[i] = 2*i;
        }

        conv2d_acc32_asymmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            int32_t acc32 = (i % 2 == 0)? 128 : -128;

            acc.high[i] = acc32 >> 16;
            acc.low[i] = acc32 & 0xFF;

            expected[i] = (i % 2 == 0)? 127 : -128;
        }

        conv2d_acc32_asymmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }
}






void test_conv2d_acc32_symmetric_resolver()
{
    srand(3523466);

    printf("%s...\n", __func__);

    nn_acc32_to_int8_params_t WORD_ALIGNED quant_params;

    nn_image_t WORD_ALIGNED Y[ VPU_INT8_ACC_PERIOD ] = { 0 };

    nn_acc32_vector_t WORD_ALIGNED acc = {{0},{0}};

    int8_t expected[ VPU_INT8_ACC_PERIOD ] = { 0 };

    // The resolvers need args for the job context and wop_params, even though it only should use one value from it
    window_op_job_context_t job_context;
    nn_window_op_params_t wop_params;
    memset(&job_context, 0, sizeof(job_context));
    memset(&wop_params, 0, sizeof(wop_params));


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));

        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = 0;

            expected[i] = 0;
        }

        conv2d_acc32_symmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = i;

            expected[i] = i;
        }

        conv2d_acc32_symmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 2;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = 8*i;

            expected[i] = 2*i;
        }

        conv2d_acc32_symmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 4;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = i;

            expected[i] = 4*i;
        }

        conv2d_acc32_symmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 2;
            quant_params.offset[i] = i+1;
            quant_params.shift2[i] = 0;

            acc.high[i] = 0;
            acc.low[i] = i;

            expected[i] = i + 2*(i+1);
        }

        conv2d_acc32_symmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 1;

            acc.high[i] = 0;
            acc.low[i] = 4*i;

            expected[i] = 2*i;
        }

        conv2d_acc32_symmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


    if( 1 ){
        memset(Y, 0xCC, sizeof(Y));
        
        for(int i = 0; i < VPU_INT8_ACC_PERIOD; i++){
            quant_params.shift1[i] = 0;
            quant_params.scale[i] = 1;
            quant_params.offset_scale[i] = 0;
            quant_params.offset[i] = 0;
            quant_params.shift2[i] = 0;

            int32_t acc32 = (i % 2 == 0)? 128 : -128;

            acc.high[i] = acc32 >> 16;
            acc.low[i] = acc32 & 0xFF;

            expected[i] = (i % 2 == 0)? 127 : -127;
        }

        conv2d_acc32_symmetric_resolver( Y, &acc, &quant_params, &job_context, &wop_params);

        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, VPU_INT8_ACC_PERIOD);
    }


}

void test_acc32_resolvers()
{

    UNITY_SET_FILE();
    
    RUN_TEST(test_conv2d_acc32_asymmetric_resolver);
    RUN_TEST(test_conv2d_acc32_symmetric_resolver);
}