
#include <stdlib.h>
#include <stdio.h>
#include <stdint.h>
#include <string.h>
#include <assert.h>
#include <syscall.h>

#include "tst_common.h"

#include "nn_operator.h"
#include "nn_types.h"
#include "xs3_vpu.h"

// #include "dsp_xs3_vector.h"
#include "unity.h"


#define DO_PRINT_EXTRA ((DO_PRINT_EXTRA_GLOBAL) && 0)




static void check_Y(
    const nn_image_t y_exp, 
    const nn_image_t* Y,
    const nn_image_params_t* y_params,
    const unsigned row,
    const unsigned col,
    const unsigned chn,
    const unsigned line)
{
    char str_buff[200];

    unsigned y_offset = IMG_ADDRESS_VECT(y_params, row, col, chn);

    int8_t y = Y[y_offset];

    //Only sprintf-ing if the test will fail saves a ton of time.
    if(y != y_exp)
        sprintf(str_buff, "(row, col, chn) = (%u, %u, %u)  [test vector @ line %u]", row, col, chn, line);

    TEST_ASSERT_EQUAL_MESSAGE(y_exp, y, str_buff);
}





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case0()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = 0;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = 0;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case1()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int bias = -10; bias < 10; bias++){

        PRINTF("\tbias_mult = %d...\n", bias);

        for(int k = 0; k < CHANS_OUT; k++){
            BSS.bias[k] = bias * k;
            BSS.shift1[k] = 0;
            BSS.scale[k] = 1;
            BSS.shift2[k] = 0;
        }
        nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

        conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = bias * chn;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case2()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int shift1 = 0; shift1 < 4; shift1++){

        PRINTF("\tshift1 = %d...\n", shift1);

        for(int k = 0; k < CHANS_OUT; k++){
            BSS.bias[k] = 16 * k;
            BSS.shift1[k] = shift1;
            BSS.scale[k] = 1;
            BSS.shift2[k] = 0;
        }
        nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

        conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = (16 * chn) >> shift1;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case3()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int scale = -10; scale < 10; scale++){

        PRINTF("\tscale = %d...\n", scale);

        for(int k = 0; k < CHANS_OUT; k++){
            BSS.bias[k] = k;
            BSS.shift1[k] = 0;
            BSS.scale[k] = scale;
            BSS.shift2[k] = 0;
        }
        nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

        conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = scale * chn;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case4()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int shift2 = 0; shift2 < 4; shift2++){

        PRINTF("\tshift2 = %d...\n", shift2);

        for(int k = 0; k < CHANS_OUT; k++){
            BSS.bias[k] = 16 * k;
            BSS.shift1[k] = 0;
            BSS.scale[k] = 1;
            BSS.shift2[k] = shift2;
        }
        nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                            (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

        memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

        conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

        for(int row = 0; row < y_params.height; row++){
            for(int col = 0; col < y_params.width; col++){
                for(int chn = 0; chn < y_params.channels; chn++){
                    int8_t y_exp = (16 * chn) >> shift2;
                    check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
                }
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case5()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);
    memset(K, 0, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case6()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 0, x_params.height * x_params.width * x_params.channels);
    memset(K, 1, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case7()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);
    memset(K, 1, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = CHANS_IN + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 36 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case8()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);
    memset(K, 1, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = CHANS_IN + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 36 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case9()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    memset(X, 1, x_params.height * x_params.width * x_params.channels);
    memset(K, 1, y_params.channels * conv2d_window.shape.height * conv2d_window.shape.width * x_params.channels);

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = CHANS_IN + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 1 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case10()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int chn = 0; chn < x_params.channels; chn++)
                X[row][col][chn] = chn;
                
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = cout;

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int chn = 0; chn < y_params.channels; chn++){
                int8_t y_exp = chn*((CHANS_IN-1)*(CHANS_IN/2)) + chn;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, chn, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 1 )
#define X_HEIGHT        ( 3 )
#define X_WIDTH         ( 2 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 2 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case11()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { 1, 1 } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = 0;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = 1 * CHANS_IN;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }

}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 1 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 1 )
#define X_WIDTH         ( 4 )
#define Y_HEIGHT        ( 1 )
#define Y_WIDTH         ( 2 )
#define K_V_STRIDE      ( 1 )
#define K_H_STRIDE      ( 1 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case12()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = col+1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = 0;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = 6 * x_params.channels;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 2 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 4 )
#define X_WIDTH         ( 6 )
#define Y_HEIGHT        ( 2 )
#define Y_WIDTH         ( 2 )
#define K_V_STRIDE      ( 2 )
#define K_H_STRIDE      ( 3 )
#define ZERO_POINT      ( -128 )
void test_conv2d_deep_case13()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { 0, 0 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = K_W * row + col + 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = 0;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = 21 * x_params.channels;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 4 )
#define CHANS_OUT       ( 4 )
#define K_H             ( 3 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 3 )
#define X_WIDTH         ( 3 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 3 )
#define K_V_STRIDE      ( 1 )
#define K_H_STRIDE      ( 1 )
#define ZERO_POINT      ( 2 )
void test_conv2d_deep_case14()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { -1, -1 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = 1;
                
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k;
        BSS.shift1[k] = 0;
        BSS.scale[k] = 1;
        BSS.shift2[k] = 0;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

    nn_image_t  WORD_ALIGNED Y_exp[Y_HEIGHT][Y_WIDTH] = {
        { 14, 12, 14},
        { 12,  9, 12},
        { 14, 12, 14},
    };

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = CHANS_IN * Y_exp[row][col] + cout;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 64 )
#define CHANS_OUT       ( 32 )
#define K_H             ( 3 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 5 )
#define X_WIDTH         ( 5 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 3 )
#define K_V_STRIDE      ( 2 )
#define K_H_STRIDE      ( 2 )
#define ZERO_POINT      ( 8 )
void test_conv2d_deep_case15()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job;

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { -1, -1 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    conv2d_deep_init(&plan, &job, &x_params, &y_params, NULL, &conv2d_window, ZERO_POINT, 1);

    nn_image_t X_vals[X_HEIGHT][X_WIDTH] = {
        {  2,  4, 2, 8, 4 },
        {  4,  2, 4, 2, 8 },
        {  8,  4, 2, 2, 4 },
        {  2, 16, 2, 8, 2 },
        {  2,  4, 2, 2, 4 },
    };
    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = X_vals[row][col];
                
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k * (1<<7);
        BSS.shift1[k] = 1;
        BSS.scale[k] = 2;
        BSS.shift2[k] = 7;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job);

/*       __ __
       |8  8  8| 8  8  8  8 
       |8  2  4| 2  8  4  8
       |8__4__2| 4  2  8  8
        8  8  4  2  2  4  8
        8  2 16  2  8  2  8
        8  2  4  2  2  4  8
        8  8  8  8  8  8  8

*/

    nn_image_t  WORD_ALIGNED Y_exp[Y_HEIGHT][Y_WIDTH] = {
        { 26, 23, 31},
        { 30, 21, 25},
        { 32, 29, 28},
    };

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = Y_exp[row][col] + cout;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT





///////////////////////////////////////////////////////////////////////////////
///////////////////////////////////////////////////////////////////////////////
#define CHANS_IN        ( 64 )
#define CHANS_OUT       ( 32 )
#define K_H             ( 3 )
#define K_W             ( 3 )
#define X_HEIGHT        ( 5 )
#define X_WIDTH         ( 5 )
#define Y_HEIGHT        ( 3 )
#define Y_WIDTH         ( 3 )
#define K_V_STRIDE      ( 2 )
#define K_H_STRIDE      ( 2 )
#define ZERO_POINT      ( 8 )
void test_conv2d_deep_case16()
{

    nn_tensor_t WORD_ALIGNED K[CHANS_OUT][K_H][K_W][CHANS_IN];
    nn_image_t  WORD_ALIGNED X[X_HEIGHT][X_WIDTH][CHANS_IN];
    nn_image_t  WORD_ALIGNED Y[Y_HEIGHT][Y_WIDTH][CHANS_OUT];
    
    struct {
        int32_t bias[CHANS_OUT];
        int16_t shift1[CHANS_OUT];
        int16_t scale[CHANS_OUT];
        int16_t shift2[CHANS_OUT];
    } BSS;

    nn_bss_block_t bss[BSS_BLOCK_COUNT(CHANS_OUT)];

    PRINTF("%s...\n", __func__);

    nn_conv2d_deep_plan_t plan;
    nn_conv2d_deep_job_t job[5];

    nn_conv2d_window_params_t conv2d_window = { { K_H, K_W }, { -1, -1 }, { K_V_STRIDE, K_H_STRIDE } }; 

    nn_image_params_t x_params = { X_HEIGHT, X_WIDTH, CHANS_IN };
    nn_image_params_t y_params = { Y_HEIGHT, Y_WIDTH, CHANS_OUT };

    nn_conv2d_job_params_t job_params[] = {
        {  {  0,  0,  0}, {  1, 3, 32}  },
        {  {  1,  0,  0}, {  2, 1, 16}  },
        {  {  1,  1,  0}, {  2, 2, 16}  },
        {  {  1,  0, 16}, {  1, 3, 16}  },
        {  {  2,  0, 16}, {  1, 2, 16}  },
        //Leaves Y[2,2,16:32] uncalculated
    };

    const unsigned job_count = sizeof(job_params) / sizeof(nn_conv2d_job_params_t);

    conv2d_deep_init(&plan, job, &x_params, &y_params, job_params, &conv2d_window, ZERO_POINT, job_count);

    nn_image_t X_vals[X_HEIGHT][X_WIDTH] = {
        {  2,  4, 2, 8, 4 },
        {  4,  2, 4, 2, 8 },
        {  8,  4, 2, 2, 4 },
        {  2, 16, 2, 8, 2 },
        {  2,  4, 2, 2, 4 },
    };
    for(int row = 0; row < x_params.height; row++)
        for(int col = 0; col < x_params.width; col++)
            for(int cin = 0; cin < x_params.channels; cin++)
                X[row][col][cin] = X_vals[row][col];
                
    for(int cout = 0; cout < y_params.channels; cout++)
        for(int row = 0; row < conv2d_window.shape.height; row++)
            for(int col = 0; col < conv2d_window.shape.width; col++)
                for(int cin = 0; cin < x_params.channels; cin++)
                    K[cout][row][col][cin] = 1;

    for(int k = 0; k < CHANS_OUT; k++){
        BSS.bias[k] = k * (1<<7);
        BSS.shift1[k] = 1;
        BSS.scale[k] = 2;
        BSS.shift2[k] = 7;
    }
    nn_standard_BSS_layout((data16_t*) bss, (int32_t*) &BSS.bias, (int16_t*) &BSS.shift1, 
                        (int16_t*) &BSS.scale, (int16_t*) &BSS.shift2, NULL, y_params.channels);

    memset(Y, 0xCC, y_params.height * y_params.width * y_params.channels);

    for(int i = 0; i < job_count; i++)
        conv2d_deep((nn_image_t*) Y, (nn_image_t*) X, (nn_tensor_t*) K, bss, &plan, &job[i]);

/*       __ __
       |8  8  8| 8  8  8  8 
       |8  2  4| 2  8  4  8
       |8__4__2| 4  2  8  8
        8  8  4  2  2  4  8
        8  2 16  2  8  2  8
        8  2  4  2  2  4  8
        8  8  8  8  8  8  8

*/

    nn_image_t  WORD_ALIGNED Y_exp[Y_HEIGHT][Y_WIDTH] = {
        { 26, 23, 31},
        { 30, 21, 25},
        { 32, 29, 28},
    };

    for(int row = 0; row < y_params.height; row++){
        for(int col = 0; col < y_params.width; col++){
            for(int cout = 0; cout < y_params.channels; cout++){
                int8_t y_exp = Y_exp[row][col] + cout;
                if(row == 2 && col == 2 && cout >= 16)
                    y_exp = 0xCC;
                check_Y(y_exp, (nn_image_t*) Y, &y_params, row, col, cout, __LINE__);
            }
        }
    }
}
#undef CHANS_IN
#undef CHANS_OUT
#undef K_H
#undef K_W
#undef X_HEIGHT
#undef X_WIDTH
#undef Y_HEIGHT
#undef Y_WIDTH
#undef K_V_STRIDE
#undef K_H_STRIDE
#undef ZERO_POINT







void test_conv2d_deep()
{
    UNITY_SET_FILE();
    
    RUN_TEST(test_conv2d_deep_case0);
    RUN_TEST(test_conv2d_deep_case1);
    RUN_TEST(test_conv2d_deep_case2);
    RUN_TEST(test_conv2d_deep_case3);
    RUN_TEST(test_conv2d_deep_case4);
    RUN_TEST(test_conv2d_deep_case5);
    RUN_TEST(test_conv2d_deep_case6);
    RUN_TEST(test_conv2d_deep_case7);
    RUN_TEST(test_conv2d_deep_case8);
    RUN_TEST(test_conv2d_deep_case9);
    RUN_TEST(test_conv2d_deep_case10);
    RUN_TEST(test_conv2d_deep_case11);
    RUN_TEST(test_conv2d_deep_case12);
    RUN_TEST(test_conv2d_deep_case13);
    RUN_TEST(test_conv2d_deep_case14);
    RUN_TEST(test_conv2d_deep_case15);
    RUN_TEST(test_conv2d_deep_case16);
}