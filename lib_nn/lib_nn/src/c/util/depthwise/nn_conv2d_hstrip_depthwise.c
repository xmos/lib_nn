

#include "nn_operator.h"
#include "../../../nn_op_helper.h"
#include "nn_op_structs.h"

#include "xs3_vpu.h"
#include "../../vpu_sim.h"

#include <stdlib.h>
#include <stdint.h>
#include <string.h>
#include <stdio.h>
#include <assert.h>

#if CONFIG_SYMMETRIC_SATURATION_conv2d_depthwise
  #define NEG_SAT_VAL   (-127)
#else
  #define NEG_SAT_VAL   (-128)
#endif 

static inline int8_t sat_s8_lcl(
    const int32_t acc32)
{
    if(acc32 >= VPU_INT8_MAX)
        return VPU_INT8_MAX;
    if(acc32 < VPU_INT8_MIN)
        return NEG_SAT_VAL;
    
    return (int8_t) acc32;
}

static inline int8_t vlsat_single_s8_lcl(
    int32_t acc, 
    int16_t shr)
{
    shr = (shr <= 0)? 0 : shr;
    int64_t acc64 = acc;
    if(shr > 0) acc64 += 1<<(shr-1);
    return sat_s8_lcl(acc64 >> shr);
}

static void vlmacc8(
    int32_t* acc,
    const int8_t* X,
    const int8_t* W)
{
    for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++){
        // printf("!@ %d\t%d\t%d\n", k, X[k], W[k]);
        acc[k] += ((int32_t)X[k]) * W[k];
    }
}

WEAK_FUNC
void nn_conv2d_hstrip_depthwise(
    int8_t* Y,
    const int8_t* X_in, 
    const int8_t* K_in,
    const nn_bso_block_t* BSO,
    const unsigned K_h,
    const unsigned K_w,
    const int32_t xk_col_stride,
    const int32_t x_row_stride,
    const int32_t window_hstride,
    const int32_t y_col_stride,
    const unsigned out_cols,
    const unsigned chans_to_write)
{

    for(int out_col = 0; out_col < out_cols; out_col++){

        const int8_t* X = X_in;
        const int8_t* K = K_in;

        int32_t accs[VPU_INT8_VLMACC_ELMS];

        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] = ((int32_t)BSO->bias_hi[k]) << VPU_INT8_ACC_VR_BITS;
        
        for(int k = 0; k < VPU_INT8_VLMACC_ELMS; k++)
            accs[k] |= BSO->bias_lo[k];

        // These rows are inside image (vertically)
        for(int i = K_h; i > 0; i--){

            for(int j = xk_col_stride * K_w; j > 0; j-= xk_col_stride){
                vlmacc8(accs, X, K);
                X = ADDR(X, xk_col_stride);
                K = ADDR(K, xk_col_stride);
            }

            X = ADDR(X, x_row_stride);
        }
        
        for(int k = 0; k < chans_to_write; k++){
            int16_t shift1  = BSO->shift1[k];
            int16_t scale   = BSO->scale[k];
            int16_t offset_scale = BSO->offset_scale[k];
            int16_t offset  = BSO->offset[k];
            int16_t shift2  = BSO->shift2[k];
            accs[k] = vlsat_single_s16(accs[k], shift1);
            accs[k] = accs[k] * scale;
            accs[k] += ((int32_t)offset_scale)*offset;
            accs[k] = vlsat_single_s8_lcl(accs[k], shift2);
            Y[k] = (int8_t) accs[k];
        }
        
        X_in = ADDR(X_in, window_hstride);
        Y = ADDR(Y, y_col_stride);

    }
}






