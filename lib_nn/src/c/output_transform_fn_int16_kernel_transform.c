#include <stdio.h>
#include <math.h>

#include "output_transform_fn_int16_kernel_transform.h"
#include "output_transform_fn_int16_mappings.h"

void output_transform_fn_int16_kernel_transform(
    const int8_t *kernel_weights_in, const float *channel_multipliers_in, const int *channel_bias_terms_in,
    int8_t *kernel_weights_out, int32_t *mul_add_out,
    int input_channels, int output_channels) {
    for(int ochannel = 0; ochannel < output_channels; ochannel++) {
        int ochannel_group = ochannel & ~0xf;
        int ochannel_member = ochannel & 0xf;
        int add_index = ochannel_group * 2;
        if (ochannel_member & 1) {
            add_index += 0;
        } else {
            add_index += 16;
        }
        add_index += (ochannel_member >> 1) & 7;
        int mul_index = add_index + 8;
        mul_add_out[add_index] = channel_bias_terms_in[ochannel];
        mul_add_out[mul_index] = round(0x40000000 * channel_multipliers_in[ochannel]);
    }
}

