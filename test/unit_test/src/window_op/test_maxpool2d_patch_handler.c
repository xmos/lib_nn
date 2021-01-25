
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




#define REPS            (100)
#define X_CHANS_MAX     (4 * VPU_INT8_EPV)
#define K_h_MAX         (5)
#define K_w_MAX         (5)
static void test_maxpool2d_patch_handler1()
{
    srand(6746);

    printf("%s...\n", __func__);

    int8_t WORD_ALIGNED patch_buff[K_h_MAX * K_w_MAX * X_CHANS_MAX + VPU_INT8_EPV]; 

    int8_t WORD_ALIGNED Y[ VPU_INT8_EPV ];

    int8_t expected[ VPU_INT8_EPV ];


    for(int rep = 0; rep < REPS; rep++){

        // X has a multiple of 4 channels
        const channel_count_t x_chans = (rep == 0)? 32 : (((pseudo_rand_uint32() % X_CHANS_MAX) >> 2) + 1) << 2;

        const unsigned k_h = (rep == 0)? K_h_MAX :  (pseudo_rand_uint32() % K_h_MAX) + 1;
        const unsigned k_w = (rep == 0)? K_w_MAX :  (pseudo_rand_uint32() % K_w_MAX) + 1;

        const unsigned cog_count = (x_chans + (VPU_INT8_EPV-1)) >> VPU_INT8_EPV_LOG2;

        const unsigned cog_start = pseudo_rand_uint32() % cog_count;
        const unsigned cout_start = cog_start << VPU_INT8_EPV_LOG2;


        printf("\trep %d...(X_c = %lu; K_h = %u; K_w = %u; cog_start = %u)\n", 
            rep, x_chans, k_h, k_w, cog_start);

        memset(patch_buff, 0, sizeof(patch_buff));
        
        nn_window_op_params_t wop_params = { 
            .input = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1, x_chans }, 
            .output = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1, x_chans }, 
            .window = {  
                .shape = { k_h, k_w }, 
                .start = { (pseudo_rand_uint32() % 13) - 6, (pseudo_rand_uint32() % 13) - 6 }, 
                .stride = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1 }, 
                .dilation = { (pseudo_rand_uint32() % 12) + 1, (pseudo_rand_uint32() % 12) + 1 } }
        };

        window_op_job_context_t job_context = {
            .window = { pseudo_rand_uint32() % wop_params.output.height, pseudo_rand_uint32() % wop_params.output.width },
            .cur_out_chan = cout_start,
            .out_chans = ((x_chans - cout_start) >= VPU_INT8_EPV)? VPU_INT8_EPV : (x_chans - cout_start),
            .flags = 0
        };

        
        nn_image_t (*X)[wop_params.window.shape.width][wop_params.input.channels] 
            = (nn_image_t (*)[wop_params.window.shape.width][wop_params.input.channels]) patch_buff;


        // Set up X values (we're using window height and width because X isn't actually the input image! It's the patch
        // that it was im2colled to!)
        for(int cin = 0; cin < job_context.out_chans; cin++){
            expected[cin] = -128;

            for(int row = 0; row < wop_params.window.shape.height; row++){
                for(int col = 0; col < wop_params.window.shape.width; col++){
                        const int8_t x = pseudo_rand_int8();
                        X[row][col][cout_start + cin] = x;

                        if(x > expected[cin])
                            expected[cin] = x;
                }
            }
        }

        memset(&Y, 0xCC, sizeof(Y));

        maxpool2d_patch_handler(NULL, Y, patch_buff, &job_context, &wop_params );


        TEST_ASSERT_EQUAL_INT8_ARRAY(expected, Y, job_context.out_chans);
    }
    
}
#undef X_CHANS_MAX
#undef K_h_MAX
#undef K_w_MAX
#undef REPS






void test_maxpool2d_patch_handler()
{

    UNITY_SET_FILE();
    
    RUN_TEST(test_maxpool2d_patch_handler1);
}