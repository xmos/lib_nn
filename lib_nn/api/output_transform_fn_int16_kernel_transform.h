#ifndef _output_transform_fn_int16_kernel_transform_h_
#define _output_transform_fn_int16_kernel_transform_h_

#include <stdint.h>

/**
 * Function that performs all the transformations needed for a 16-bit convolution
 *  1. It transforms the weight ordering to reverse the weights in groups of 16 in preparation of VLMACCR shift-and-rotate
 *  2. It convers the channel multpliers to integer multipliers
 *  3. It interleaves the channel multpliers and channel biases into a single blob
 * The number of kernels is assumed to be input_channels x output_channels elements
 * The number of bias terms and multipliers is assumed to be output_channels
 * The number of elements in the array mul_add_out should be 2xoutput_channels
 * The number of elements in the kernel_weights_out array should be input_channels x output_channels
 */
extern void output_transform_fn_int16_kernel_transform(
    int8_t *kernel_weights_in,
    float *channel_multipliers_in, int *channel_bias_terms_in,
    int8_t *kernel_weights_out, int32_t *mul_add_out,
    int input_channels, int output_channels);

#endif
