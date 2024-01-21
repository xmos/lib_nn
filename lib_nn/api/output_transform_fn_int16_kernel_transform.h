#ifndef _output_transform_fn_int16_kernel_transform_h_
#define _output_transform_fn_int16_kernel_transform_h_

#include <stdint.h>

extern void output_transform_fn_int16_kernel_transform(
    const int8_t *kernel_weights_in,
    const float *channel_multipliers_in, const int *channel_bias_terms_in,
    int8_t *kernel_weights_out, int32_t *mul_add_out,
    int input_channels, int output_channels);

#endif
