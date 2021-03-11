// Copyright 2020-2021 XMOS LIMITED.
// This Software is subject to the terms of the XMOS Public Licence: Version 1.
#include "nn_operator.h"

void bconv2d_bin_DI_valid(bnn_b32_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, const int32_t* thresholds_p,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_width, const unsigned y_loc_height,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned y_loc_channel, const unsigned y_sub_channel
){  
    unsigned x_loc_width = y_loc_width*k->stride.horizontal;
    unsigned x_loc_height = y_loc_height*k->stride.vertical;

    bconv2d_bin_DI(Y_p, X_p, K_p, thresholds_p,
        x,  y, k, 
        y_loc_width, y_loc_height,
        y_sub_width, y_sub_height,
        x_loc_width,  x_loc_height, y_loc_channel, y_sub_channel);
}

void bconv2d_bin_valid(bnn_b32_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, const int32_t* thresholds_p,
    bnn_b32_t * data_scratch, 

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_width, const unsigned y_loc_height,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned y_loc_channel, const unsigned y_sub_channel
){  
    unsigned x_loc_width = y_loc_width*k->stride.horizontal;
    unsigned x_loc_height = y_loc_height*k->stride.vertical;

    bconv2d_bin(Y_p, X_p, K_p, thresholds_p, data_scratch,
        x,  y, k, 
        y_loc_width, y_loc_height,
        y_sub_width, y_sub_height,
        x_loc_width,  x_loc_height, y_loc_channel, y_sub_channel);
}

void bconv2d_int8_DIDO_valid(int8_t* Y_p,
    const bnn_b256_t* X_p, const bnn_b256_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,

    const output_transform_values_t * otv,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_width, const unsigned y_loc_height,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned y_loc_channel, const unsigned y_sub_channel
){  
    unsigned x_loc_width = y_loc_width*k->stride.horizontal;
    unsigned x_loc_height = y_loc_height*k->stride.vertical;

    bconv2d_int8_DIDO(Y_p, X_p, K_p, 

        post_activation_multiplier_q,
        post_activation_bias_q, 
        
        otv,

        x,  y,  k, 
        y_loc_width, y_loc_height,
        y_sub_width, y_sub_height, 
        x_loc_width,  x_loc_height, y_loc_channel, y_sub_channel);
}

void bconv2d_int8_valid(int8_t* Y_p,
    const bnn_b32_t* X_p, const bnn_b32_t* K_p, 
    
    const int16_t* post_activation_multiplier_q, 
    const int16_t* post_activation_bias_q,

    const int16_t * quantised_accu_modifier,

    const output_transform_values_t * otv,

    bnn_b32_t * data_scratch,

    const nn_image_params_t* x,
    const nn_image_params_t* y,
    const nn_window_params_t* k, 

    const unsigned y_loc_width, const unsigned y_loc_height,
    const unsigned y_sub_width, const unsigned y_sub_height,
    const unsigned y_loc_channel, const unsigned y_sub_channel
){  
    unsigned x_loc_width = y_loc_width*k->stride.horizontal;
    unsigned x_loc_height = y_loc_height*k->stride.vertical;

    bconv2d_int8(Y_p, X_p, K_p, 

        post_activation_multiplier_q,
        post_activation_bias_q, 

        quantised_accu_modifier, 
        
        otv,
        
        data_scratch,
        x,  y,  k, 
        y_loc_width, y_loc_height,
        y_sub_width, y_sub_height, 
        x_loc_width,  x_loc_height, y_loc_channel, y_sub_channel);
}
