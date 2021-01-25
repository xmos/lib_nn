
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

static int32_t acc32_from_pair( data16_t acc_hi, data16_t acc_lo )
{
    return ((int32_t)(acc_hi) << 16) + ((uint16_t)acc_lo);
}

static data16_t acc32_high16( int32_t acc32 )
{
    return acc32 >> 16;
}
static data16_t acc32_low16( int32_t acc32 )
{
    return acc32 & 0xFFFF;
}


#define Y_CHANS     (VPU_INT8_ACC_PERIOD)
#define X_CHANS     (VPU_INT8_EPV)
#define K_h         (1)
#define K_w         (1)
static void test_conv2d_patch_accumulator_dense0()
{
    srand(456);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED X[K_h][K_w][X_CHANS];
    int8_t WORD_ALIGNED K[Y_CHANS][K_h][K_w][X_CHANS];
    
    nn_acc32_vector_t WORD_ALIGNED bias;
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias, .K = &K[0][0][0][0] };

    nn_window_op_params_t wop_params = { 
        .input = { K_h, K_w, X_CHANS }, 
        .output = { 1, 1, Y_CHANS }, 
        .window = {  
            .shape = { K_h, K_w }, 
            .start = { 0, 0 }, 
            .stride = { 1, 1 }, 
            .dilation = { 1, 1 } }
    };

    window_op_job_context_t job_context = {
        .window = { 0, 0 },
        .cur_out_chan = 0,
        .out_chans = Y_CHANS,
        .flags = 0
    };
    

    // Set up biases
    for(int cout = 0; cout < Y_CHANS; cout++){
        bias.high[cout] = 0;
        bias.low[cout] = cout;
    }

    // Set up X values
    for(int row = 0; row < K_h; row++){
        for(int col = 0; col < K_w; col++){
            for(int cin = 0; cin < X_CHANS; cin++){
                X[row][col][cin] = 0;
            }
        }
    }

    // Set up K and expectation.
    for(int cout = 0; cout < Y_CHANS; cout++){

        int32_t exp = acc32_from_pair( bias.high[cout], bias.low[cout] );

        for(int row = 0; row < K_h; row++){
            for(int col = 0; col < K_w; col++){
                for(int cin = 0; cin < X_CHANS; cin++){
                    K[cout][row][col][cin] = 0;

                    int32_t p = ((int32_t)K[cout][row][col][cin]) * X[row][col][cin];

                    exp += p;
                }
            }
        }

        expected.high[cout] = acc32_high16( exp );
        expected.low[cout] = acc32_low16( exp );
    }

    memset(&acc, 0xCC, sizeof(acc));

    conv2d_patch_accumulator_dense( &X[0][0][0], &acc_context, &acc, &job_context, &wop_params );


    TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);
    
}
#undef X_CHANS
#undef Y_CHANS
#undef K_h
#undef K_w





#define Y_CHANS     (VPU_INT8_ACC_PERIOD)
#define X_CHANS     (VPU_INT8_EPV)
#define K_h         (1)
#define K_w         (1)
static void test_conv2d_patch_accumulator_dense1()
{
    srand(456);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED X[K_h][K_w][X_CHANS];
    int8_t WORD_ALIGNED K[Y_CHANS][K_h][K_w][X_CHANS];
    
    nn_acc32_vector_t WORD_ALIGNED bias;
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias, .K = &K[0][0][0][0] };

    nn_window_op_params_t wop_params = { 
        .input = { K_h, K_w, X_CHANS }, 
        .output = { 1, 1, Y_CHANS }, 
        .window = {  
            .shape = { K_h, K_w }, 
            .start = { 0, 0 }, 
            .stride = { 1, 1 }, 
            .dilation = { 1, 1 } }
    };

    window_op_job_context_t job_context = {
        .window = { 0, 0 },
        .cur_out_chan = 0,
        .out_chans = Y_CHANS,
        .flags = 0
    };
    

    // Set up biases
    for(int cout = 0; cout < Y_CHANS; cout++){
        bias.high[cout] = 0;
        bias.low[cout] = cout;
    }

    // Set up X values
    for(int row = 0; row < K_h; row++){
        for(int col = 0; col < K_w; col++){
            for(int cin = 0; cin < X_CHANS; cin++){
                X[row][col][cin] = 2;
            }
        }
    }

    // Set up K and expectation.
    for(int cout = 0; cout < Y_CHANS; cout++){

        int32_t exp = acc32_from_pair( bias.high[cout], bias.low[cout] );

        for(int row = 0; row < K_h; row++){
            for(int col = 0; col < K_w; col++){
                for(int cin = 0; cin < X_CHANS; cin++){
                    K[cout][row][col][cin] = 3;

                    int32_t p = ((int32_t)K[cout][row][col][cin]) * X[row][col][cin];

                    exp += p;
                }
            }
        }

        TEST_ASSERT_EQUAL_INT32( 192 + cout,  exp);

        expected.high[cout] = acc32_high16( exp );
        expected.low[cout] = acc32_low16( exp );
    }

    memset(&acc, 0xCC, sizeof(acc));

    conv2d_patch_accumulator_dense( &X[0][0][0], &acc_context, &acc, &job_context, &wop_params );


    TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);
    
}
#undef X_CHANS
#undef Y_CHANS
#undef K_h
#undef K_w





#define Y_CHANS     (VPU_INT8_ACC_PERIOD)
#define X_CHANS     (4)
#define K_h         (1)
#define K_w         (1)
static void test_conv2d_patch_accumulator_dense2()
{
    srand(456);

    printf("%s...\n", __func__);

    struct { 
        int8_t data[K_h+1][K_w][X_CHANS];
        int8_t dummy[VPU_INT8_EPV];
    } X = { {{{0}}}, {0}};


    int8_t WORD_ALIGNED K[Y_CHANS][K_h][K_w][X_CHANS] = {{{{0}}}};
    
    nn_acc32_vector_t WORD_ALIGNED bias;
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias, .K = &K[0][0][0][0] };

    nn_window_op_params_t wop_params = { 
        .input = { K_h, K_w, X_CHANS }, 
        .output = { 1, 1, Y_CHANS }, 
        .window = {  
            .shape = { K_h, K_w }, 
            .start = { 0, 0 }, 
            .stride = { 1, 1 }, 
            .dilation = { 1, 1 } }
    };

    window_op_job_context_t job_context = {
        .window = { 0, 0 },
        .cur_out_chan = 0,
        .out_chans = Y_CHANS,
        .flags = 0
    };
    

    // Set up biases
    for(int cout = 0; cout < Y_CHANS; cout++){
        bias.high[cout] = 0;
        bias.low[cout] = cout;
    }

    // Set up X values
    for(int row = 0; row < K_h; row++){
        for(int col = 0; col < K_w; col++){
            for(int cin = 0; cin < X_CHANS; cin++){
                X.data[row][col][cin] = 2;
            }
        }
    }

    // Set up K and expectation.
    for(int cout = 0; cout < Y_CHANS; cout++){

        int32_t exp = acc32_from_pair( bias.high[cout], bias.low[cout] );

        for(int row = 0; row < K_h; row++){
            for(int col = 0; col < K_w; col++){
                for(int cin = 0; cin < X_CHANS; cin++){
                    K[cout][row][col][cin] = 3;

                    int32_t p = ((int32_t)K[cout][row][col][cin]) * X.data[row][col][cin];

                    exp += p;
                }
            }
        }

        expected.high[cout] = acc32_high16( exp );
        expected.low[cout] = acc32_low16( exp );
    }

    memset(&acc, 0xCC, sizeof(acc));

    conv2d_patch_accumulator_dense( &X.data[0][0][0], &acc_context, &acc, &job_context, &wop_params );


    TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
    TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);
    
}
#undef X_CHANS
#undef Y_CHANS
#undef K_h
#undef K_w




#define REPS        (10)
#define Y_CHANS     (VPU_INT8_ACC_PERIOD)
#define X_CHANS     (3 * VPU_INT8_EPV)
#define K_h         (1)
#define K_w         (1)
static void test_conv2d_patch_accumulator_dense3()
{
    srand(457686);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED X[K_h][K_w][X_CHANS];
    int8_t WORD_ALIGNED K[Y_CHANS][K_h][K_w][X_CHANS];
    
    nn_acc32_vector_t WORD_ALIGNED bias;
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias, .K = &K[0][0][0][0] };

    nn_window_op_params_t wop_params = { 
        .input = { K_h, K_w, X_CHANS }, 
        .output = { 1, 1, Y_CHANS }, 
        .window = {  
            .shape = { K_h, K_w }, 
            .start = { 0, 0 }, 
            .stride = { 1, 1 }, 
            .dilation = { 1, 1 } }
    };

    window_op_job_context_t job_context = {
        .window = { 0, 0 },
        .cur_out_chan = 0,
        .out_chans = Y_CHANS,
        .flags = 0
    };

    for(int rep = 0; rep < REPS; rep++){
        printf("\trep %d...\n", rep);

        // Set up biases
        for(int cout = 0; cout < Y_CHANS; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias.high[cout] = acc32_high16( b );
            bias.low[cout] = acc32_low16(  b );
        }

        // Set up X values
        for(int row = 0; row < K_h; row++){
            for(int col = 0; col < K_w; col++){
                for(int cin = 0; cin < X_CHANS; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < Y_CHANS; cout++){

            int32_t exp = acc32_from_pair( bias.high[cout], bias.low[cout] );

            for(int row = 0; row < K_h; row++){
                for(int col = 0; col < K_w; col++){
                    for(int cin = 0; cin < X_CHANS; cin++){
                        K[cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( &X[0][0][0], &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS
#undef Y_CHANS
#undef K_h
#undef K_w
#undef REPS




#define REPS            (10)
#define Y_CHANS         (VPU_INT8_ACC_PERIOD)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h             (1)
#define K_w             (1)
static void test_conv2d_patch_accumulator_dense4()
{
    srand(457686);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h * K_w * X_CHANS_MAX + VPU_INT8_EPV]; 
    int8_t WORD_ALIGNED K_buff[Y_CHANS * K_h * K_w * X_CHANS_MAX];
    
    nn_acc32_vector_t WORD_ALIGNED bias;
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias, .K = K_buff };


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 40 :  
            (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;
        const channel_count_t y_chans = Y_CHANS;


        printf("\trep %d... (x_chans = %lu)\n", rep, x_chans);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { K_h, K_w, x_chans }, 
            .output = { 1, 1, y_chans }, 
            .window = {  
                .shape = { K_h, K_w }, 
                .start = { 0, 0 }, 
                .stride = { 1, 1 }, 
                .dilation = { 1, 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { 0, 0 },
            .cur_out_chan = 0,
            .out_chans = VPU_INT8_ACC_PERIOD,
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.input.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.input.width][wop_params.input.channels]) patch_buff;

        nn_image_t (*K)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels]) K_buff;


        // Set up biases
        for(int cout = 0; cout < wop_params.output.channels; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias.high[cout] = acc32_high16( b );
            bias.low[cout] = acc32_low16(  b );
        }

        // Set up X values
        for(int row = 0; row < wop_params.input.height; row++){
            for(int col = 0; col < wop_params.input.width; col++){
                for(int cin = 0; cin < wop_params.input.channels; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < wop_params.output.channels; cout++){

            int32_t exp = acc32_from_pair( bias.high[cout], bias.low[cout] );

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                    for(int cin = 0; cin < wop_params.input.channels; cin++){
                        K[cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( patch_buff, &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS_MAX
#undef Y_CHANS
#undef K_h
#undef K_w
#undef REPS




#define REPS            (10)
#define Y_CHANS         (4)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h             (1)
#define K_w             (1)
static void test_conv2d_patch_accumulator_dense5()
{
    srand(756332);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h * K_w * X_CHANS_MAX + VPU_INT8_EPV]; 
    int8_t WORD_ALIGNED K_buff[Y_CHANS * K_h * K_w * X_CHANS_MAX];
    
    nn_acc32_vector_t WORD_ALIGNED bias;
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias, .K = K_buff };


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 40 :  
            (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;
        const channel_count_t y_chans = Y_CHANS;


        printf("\trep %d... (x_chans = %lu)\n", rep, x_chans);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { K_h, K_w, x_chans }, 
            .output = { 1, 1, y_chans }, 
            .window = {  
                .shape = { K_h, K_w }, 
                .start = { 0, 0 }, 
                .stride = { 1, 1 }, 
                .dilation = { 1, 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { 0, 0 },
            .cur_out_chan = 0,
            .out_chans = Y_CHANS,
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.input.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.input.width][wop_params.input.channels]) patch_buff;

        nn_image_t (*K)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels]) K_buff;


        // Set up biases
        for(int cout = 0; cout < wop_params.output.channels; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias.high[cout] = acc32_high16( b );
            bias.low[cout] = acc32_low16(  b );
        }

        // Set up X values
        for(int row = 0; row < wop_params.input.height; row++){
            for(int col = 0; col < wop_params.input.width; col++){
                for(int cin = 0; cin < wop_params.input.channels; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < wop_params.output.channels; cout++){

            int32_t exp = acc32_from_pair( bias.high[cout], bias.low[cout] );

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                    for(int cin = 0; cin < wop_params.input.channels; cin++){
                        K[cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( patch_buff, &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS_MAX
#undef Y_CHANS
#undef K_h
#undef K_w
#undef REPS




#define REPS            (10)
#define Y_CHANS_MAX     (VPU_INT8_ACC_PERIOD)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h             (1)
#define K_w             (1)
static void test_conv2d_patch_accumulator_dense6()
{
    srand(756332);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h * K_w * X_CHANS_MAX + VPU_INT8_EPV]; 
    int8_t WORD_ALIGNED K_buff[Y_CHANS_MAX * K_h * K_w * X_CHANS_MAX];
    
    nn_acc32_vector_t WORD_ALIGNED bias;
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias, .K = K_buff };


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 40 :  
            (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;
        // Y has to be even
        const channel_count_t y_chans = (rep == 0)? 16 :
            (((pseudo_rand_uint32() % Y_CHANS_MAX) >> 1) + 1) << 1;


        printf("\trep %d... (y_chans = %lu)\n", rep, y_chans);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { K_h, K_w, x_chans }, 
            .output = { 1, 1, y_chans }, 
            .window = {  
                .shape = { K_h, K_w }, 
                .start = { 0, 0 }, 
                .stride = { 1, 1 }, 
                .dilation = { 1, 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { 0, 0 },
            .cur_out_chan = 0,
            .out_chans = y_chans,
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.input.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.input.width][wop_params.input.channels]) patch_buff;

        nn_image_t (*K)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels]) K_buff;


        // Set up biases
        for(int cout = 0; cout < wop_params.output.channels; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias.high[cout] = acc32_high16( b );
            bias.low[cout] = acc32_low16(  b );
        }

        // Set up X values
        for(int row = 0; row < wop_params.input.height; row++){
            for(int col = 0; col < wop_params.input.width; col++){
                for(int cin = 0; cin < wop_params.input.channels; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < wop_params.output.channels; cout++){

            int32_t exp = acc32_from_pair( bias.high[cout], bias.low[cout] );

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                    for(int cin = 0; cin < wop_params.input.channels; cin++){
                        K[cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( patch_buff, &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS_MAX
#undef Y_CHANS_MAX
#undef K_h
#undef K_w
#undef REPS




#define REPS            (10)
#define Y_CHANS_MAX     (4 * VPU_INT8_ACC_PERIOD)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h             (1)
#define K_w             (1)
static void test_conv2d_patch_accumulator_dense7()
{
    srand(741736);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h * K_w * X_CHANS_MAX + VPU_INT8_EPV]; 
    int8_t WORD_ALIGNED K_buff[Y_CHANS_MAX * K_h * K_w * X_CHANS_MAX];
    
    nn_acc32_vector_t WORD_ALIGNED bias[(Y_CHANS_MAX + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2];
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias[0], .K = K_buff };


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 32 :  
            (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;
        // Y has to be even
        const channel_count_t y_chans = (rep == 0)? 16 :
            (((pseudo_rand_uint32() % Y_CHANS_MAX) >> 1) + 1) << 1;

        const unsigned cog_count = (y_chans + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2;

        const unsigned cog_start = pseudo_rand_uint32() % cog_count;
        const unsigned cout_start = cog_start << VPU_INT8_ACC_PERIOD_LOG2;


        printf("\trep %d...\n", rep);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { K_h, K_w, x_chans }, 
            .output = { 1, 1, y_chans }, 
            .window = {  
                .shape = { K_h, K_w }, 
                .start = { 0, 0 }, 
                .stride = { 1, 1 }, 
                .dilation = { 1, 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { 0, 0 },
            .cur_out_chan = cout_start,
            .out_chans = ((y_chans - cout_start) >= VPU_INT8_ACC_PERIOD)? VPU_INT8_ACC_PERIOD : (y_chans - cout_start),
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.input.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.input.width][wop_params.input.channels]) patch_buff;

        nn_image_t (*K)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels]) K_buff;


        // Set up biases
        for(int cout = 0; cout < job_context.out_chans; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias[cog_start].high[cout] = acc32_high16( b );
            bias[cog_start].low[cout] = acc32_low16(  b );
        }

        // Set up X values
        for(int row = 0; row < wop_params.input.height; row++){
            for(int col = 0; col < wop_params.input.width; col++){
                for(int cin = 0; cin < wop_params.input.channels; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < job_context.out_chans; cout++){

            int32_t exp = acc32_from_pair( bias[cog_start].high[cout], bias[cog_start].low[cout] );

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                    for(int cin = 0; cin < wop_params.input.channels; cin++){
                        K[cout_start + cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout_start + cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( patch_buff, &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS_MAX
#undef Y_CHANS_MAX
#undef K_h
#undef K_w
#undef REPS




#define REPS            (10)
#define Y_CHANS_MAX     (4 * VPU_INT8_ACC_PERIOD)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h_MAX         (3)
#define K_w_MAX         (2)
static void test_conv2d_patch_accumulator_dense8()
{
    srand(752566);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h_MAX * K_w_MAX * X_CHANS_MAX + VPU_INT8_EPV]; 
    int8_t WORD_ALIGNED K_buff[Y_CHANS_MAX * K_h_MAX * K_w_MAX * X_CHANS_MAX];
    
    nn_acc32_vector_t WORD_ALIGNED bias[(Y_CHANS_MAX + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2];
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias[0], .K = K_buff };


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 32 :  
            (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;
        // Y has to be even
        const channel_count_t y_chans = (rep == 0)? 16 :
            (((pseudo_rand_uint32() % Y_CHANS_MAX) >> 1) + 1) << 1;

        const unsigned k_h = K_h_MAX;
        const unsigned k_w = K_w_MAX;

        const unsigned cog_count = (y_chans + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2;

        const unsigned cog_start = pseudo_rand_uint32() % cog_count;
        const unsigned cout_start = cog_start << VPU_INT8_ACC_PERIOD_LOG2;


        printf("\trep %d...\n", rep);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { k_h, k_w, x_chans }, 
            .output = { 1, 1, y_chans }, 
            .window = {  
                .shape = { k_h, k_w }, 
                .start = { 0, 0 }, 
                .stride = { 1, 1 }, 
                .dilation = { 1, 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { 0, 0 },
            .cur_out_chan = cout_start,
            .out_chans = ((y_chans - cout_start) >= VPU_INT8_ACC_PERIOD)? VPU_INT8_ACC_PERIOD : (y_chans - cout_start),
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.input.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.input.width][wop_params.input.channels]) patch_buff;

        nn_image_t (*K)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels]) K_buff;


        // Set up biases
        for(int cout = 0; cout < job_context.out_chans; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias[cog_start].high[cout] = acc32_high16( b );
            bias[cog_start].low[cout] = acc32_low16(  b );
        }

        // Set up X values
        for(int row = 0; row < wop_params.input.height; row++){
            for(int col = 0; col < wop_params.input.width; col++){
                for(int cin = 0; cin < wop_params.input.channels; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < job_context.out_chans; cout++){

            int32_t exp = acc32_from_pair( bias[cog_start].high[cout], bias[cog_start].low[cout] );

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                    for(int cin = 0; cin < wop_params.input.channels; cin++){
                        K[cout_start + cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout_start + cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( patch_buff, &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS_MAX
#undef Y_CHANS_MAX
#undef K_h_MAX
#undef K_w_MAX
#undef REPS




#define REPS            (10)
#define Y_CHANS_MAX     (4 * VPU_INT8_ACC_PERIOD)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h_MAX         (5)
#define K_w_MAX         (5)
static void test_conv2d_patch_accumulator_dense9()
{
    srand(88845);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h_MAX * K_w_MAX * X_CHANS_MAX + VPU_INT8_EPV]; 
    int8_t WORD_ALIGNED K_buff[Y_CHANS_MAX * K_h_MAX * K_w_MAX * X_CHANS_MAX];
    
    nn_acc32_vector_t WORD_ALIGNED bias[(Y_CHANS_MAX + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2];
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias[0], .K = K_buff };


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 32 :  
            (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;
        // Y has to be even
        const channel_count_t y_chans = (rep == 0)? 16 :
            (((pseudo_rand_uint32() % Y_CHANS_MAX) >> 1) + 1) << 1;

        const unsigned k_h = (pseudo_rand_uint32() % K_h_MAX) + 1;
        const unsigned k_w = (pseudo_rand_uint32() % K_w_MAX) + 1;

        const unsigned cog_count = (y_chans + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2;

        const unsigned cog_start = pseudo_rand_uint32() % cog_count;
        const unsigned cout_start = cog_start << VPU_INT8_ACC_PERIOD_LOG2;


        printf("\trep %d...\n", rep);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { k_h, k_w, x_chans }, 
            .output = { 1, 1, y_chans }, 
            .window = {  
                .shape = { k_h, k_w }, 
                .start = { 0, 0 }, 
                .stride = { 1, 1 }, 
                .dilation = { 1, 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { 0, 0 },
            .cur_out_chan = cout_start,
            .out_chans = ((y_chans - cout_start) >= VPU_INT8_ACC_PERIOD)? VPU_INT8_ACC_PERIOD : (y_chans - cout_start),
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.input.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.input.width][wop_params.input.channels]) patch_buff;

        nn_image_t (*K)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels]) K_buff;


        // Set up biases
        for(int cout = 0; cout < job_context.out_chans; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias[cog_start].high[cout] = acc32_high16( b );
            bias[cog_start].low[cout] = acc32_low16(  b );
        }

        // Set up X values
        for(int row = 0; row < wop_params.input.height; row++){
            for(int col = 0; col < wop_params.input.width; col++){
                for(int cin = 0; cin < wop_params.input.channels; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < job_context.out_chans; cout++){

            int32_t exp = acc32_from_pair( bias[cog_start].high[cout], bias[cog_start].low[cout] );

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                    for(int cin = 0; cin < wop_params.input.channels; cin++){
                        K[cout_start + cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout_start + cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( patch_buff, &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS_MAX
#undef Y_CHANS_MAX
#undef K_h_MAX
#undef K_w_MAX
#undef REPS











#define REPS            (10)
#define Y_CHANS_MAX     (4 * VPU_INT8_ACC_PERIOD)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h_MAX         (5)
#define K_w_MAX         (5)
static void test_conv2d_patch_accumulator_dense10()
{
    srand(6746);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h_MAX * K_w_MAX * X_CHANS_MAX + VPU_INT8_EPV]; 
    int8_t WORD_ALIGNED K_buff[Y_CHANS_MAX * K_h_MAX * K_w_MAX * X_CHANS_MAX];
    
    nn_acc32_vector_t WORD_ALIGNED bias[(Y_CHANS_MAX + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2];
    nn_acc32_vector_t WORD_ALIGNED acc;

    nn_acc32_vector_t expected;

    nn_conv2d_accumulator_params_t acc_context = { .bias = &bias[0], .K = K_buff };


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 32 :  
            (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;
        // Y has to be even
        const channel_count_t y_chans = (rep == 0)? 16 :
            (((pseudo_rand_uint32() % Y_CHANS_MAX) >> 1) + 1) << 1;

        const unsigned k_h = (rep == 0)? K_h_MAX :
             (pseudo_rand_uint32() % K_h_MAX) + 1;
        const unsigned k_w = (rep == 0)? K_w_MAX :
             (pseudo_rand_uint32() % K_w_MAX) + 1;

        const unsigned cog_count = (y_chans + (VPU_INT8_ACC_PERIOD-1)) >> VPU_INT8_ACC_PERIOD_LOG2;

        const unsigned cog_start = pseudo_rand_uint32() % cog_count;
        const unsigned cout_start = cog_start << VPU_INT8_ACC_PERIOD_LOG2;


        printf("\trep %d...(Y_c = %lu; X_c = %lu; K_h = %u; K_w = %u; cog_start = %u)\n", 
            rep, y_chans, x_chans, k_h, k_w, cog_start);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1, x_chans }, 
            .output = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1, y_chans }, 
            .window = {  
                .shape = { k_h, k_w }, 
                .start = { (pseudo_rand_uint32() % 13) - 6, (pseudo_rand_uint32() % 13) - 6 }, 
                .stride = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1 }, 
                .dilation = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { pseudo_rand_uint32() % wop_params.output.height, pseudo_rand_uint32() % wop_params.output.width },
            .cur_out_chan = cout_start,
            .out_chans = ((y_chans - cout_start) >= VPU_INT8_ACC_PERIOD)? VPU_INT8_ACC_PERIOD : (y_chans - cout_start),
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.width][wop_params.input.channels]) patch_buff;

        nn_image_t (*K)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.height][wop_params.window.shape.width][wop_params.input.channels]) K_buff;


        // Set up biases
        for(int cout = 0; cout < job_context.out_chans; cout++){
            int32_t b = pseudo_rand_int32() >> 8;
            bias[cog_start].high[cout] = acc32_high16( b );
            bias[cog_start].low[cout] = acc32_low16(  b );
        }

        // Set up X values (we're using window height and width because X isn't actually the input image! It's the patch
        // that it was im2colled to!)
        for(int row = 0; row < wop_params.window.shape.height; row++){
            for(int col = 0; col < wop_params.window.shape.width; col++){
                for(int cin = 0; cin < wop_params.input.channels; cin++){
                    X[row][col][cin] = pseudo_rand_int8();
                }
            }
        }

        // Set up K and expectation.
        for(int cout = 0; cout < job_context.out_chans; cout++){

            int32_t exp = acc32_from_pair( bias[cog_start].high[cout], bias[cog_start].low[cout] );

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                    for(int cin = 0; cin < wop_params.input.channels; cin++){
                        K[cout_start + cout][row][col][cin] = pseudo_rand_int8();

                        int32_t p = ((int32_t)K[cout_start + cout][row][col][cin]) * X[row][col][cin];

                        exp += p;
                    }
                }
            }

            expected.high[cout] = acc32_high16( exp );
            expected.low[cout] = acc32_low16( exp );
        }

        memset(&acc, 0xCC, sizeof(acc));

        conv2d_patch_accumulator_dense( patch_buff, &acc_context, &acc, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.high, acc.high, job_context.out_chans);
        TEST_ASSERT_EQUAL_INT16_ARRAY(expected.low , acc.low , job_context.out_chans);

    }
    
}
#undef X_CHANS_MAX
#undef Y_CHANS_MAX
#undef K_h_MAX
#undef K_w_MAX
#undef REPS






void test_conv2d_patch_accumulator_dense()
{

    UNITY_SET_FILE();
    
    RUN_TEST(test_conv2d_patch_accumulator_dense0);
    RUN_TEST(test_conv2d_patch_accumulator_dense1);
    RUN_TEST(test_conv2d_patch_accumulator_dense2);
    RUN_TEST(test_conv2d_patch_accumulator_dense3);
    RUN_TEST(test_conv2d_patch_accumulator_dense4);
    RUN_TEST(test_conv2d_patch_accumulator_dense5);
    RUN_TEST(test_conv2d_patch_accumulator_dense6);
    RUN_TEST(test_conv2d_patch_accumulator_dense7);
    RUN_TEST(test_conv2d_patch_accumulator_dense8);
    RUN_TEST(test_conv2d_patch_accumulator_dense9);
    RUN_TEST(test_conv2d_patch_accumulator_dense10);
}